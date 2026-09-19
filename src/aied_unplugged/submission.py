from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .graders.answer_sheets import encode
from .tracks import (
    ANSWER_VALUES,
    COMPETENCES,
    LABEL_KEY,
    MAX_SUBMISSION_BYTES,
    QUESTION_KEY,
    SCORE_VALUES,
    Track,
    get_track,
)

MAX_REPORTED_ISSUES = 100


@dataclass
class Issue:
    row: int | None
    column: str | None
    code: str
    message: str


@dataclass
class ValidationReport:
    valid: bool
    issues: list[Issue] = field(default_factory=list)
    row_count: int = 0

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        if self.valid:
            return f"ValidationReport(valid, {self.row_count} rows)"
        lines = "\n".join(f"  - {i.message}" for i in self.issues[:10])
        more = "" if len(self.issues) <= 10 else f"\n  … {len(self.issues) - 10} more"
        return f"ValidationReport(invalid, {len(self.issues)} issue(s)):\n{lines}{more}"

    def raise_for_issues(self) -> None:
        if not self.valid:
            raise ValueError(repr(self))


def _structured_issues(track: Track, frame: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    column = track.target_columns[0]
    for position, raw in enumerate(frame[column], start=1):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            issues.append(
                Issue(
                    position,
                    column,
                    "structured_parse",
                    f"Row {position}, column {column!r} is not valid JSON.",
                )
            )
            continue
        if not isinstance(parsed, list):
            issues.append(
                Issue(
                    position,
                    column,
                    "structured_parse",
                    f"Row {position}, column {column!r} must be a JSON array.",
                )
            )
            continue

        seen: set[int] = set()
        bound = len(parsed)
        for at, entry in enumerate(parsed, start=1):
            if not isinstance(entry, dict):
                issues.append(
                    Issue(
                        position,
                        column,
                        "structured_parse",
                        f"Row {position}, entry {at} is not an object.",
                    )
                )
                continue
            number, label = entry.get(QUESTION_KEY), entry.get(LABEL_KEY)
            if isinstance(number, bool) or not isinstance(number, int):
                issues.append(
                    Issue(
                        position,
                        column,
                        "structured_keys",
                        f"Row {position}, entry {at}: {QUESTION_KEY!r} must be a "
                        'whole number. Write 1, not "1".',
                    )
                )
            elif not 1 <= number <= bound:
                issues.append(
                    Issue(
                        position,
                        column,
                        "structured_keys",
                        f"Row {position}, entry {at}: question {number} is outside "
                        f"the range 1 to {bound}.",
                    )
                )
            elif number in seen:
                issues.append(
                    Issue(
                        position,
                        column,
                        "structured_keys",
                        f"Row {position}: question {number} appears more than once.",
                    )
                )
            else:
                seen.add(number)
            if not isinstance(label, str) or label not in ANSWER_VALUES:
                issues.append(
                    Issue(
                        position,
                        column,
                        "structured_value",
                        f"Row {position}, entry {at}: {label!r} is not a permitted "
                        f"answer. Use one of {', '.join(ANSWER_VALUES)}.",
                    )
                )
        for question in range(1, bound + 1):
            if question not in seen:
                issues.append(
                    Issue(
                        position,
                        column,
                        "structured_keys",
                        f"Row {position}: question {question} is missing.",
                    )
                )
    return issues


def _closed_issues(track: Track, frame: pd.DataFrame) -> list[Issue]:
    column = track.target_columns[0]
    allowed = set(track.allowed_values)
    issues = []
    for position, raw in enumerate(frame[column], start=1):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw) == "":
            continue
        if str(raw) not in allowed:
            issues.append(
                Issue(
                    position,
                    column,
                    "closed_value",
                    f"Row {position}, column {column!r}: {str(raw)!r} is not one of "
                    "the permitted values.",
                )
            )
    return issues


def _scale_issues(frame: pd.DataFrame) -> list[Issue]:
    issues = []
    for column in COMPETENCES:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        for position, value in enumerate(numeric, start=1):
            if pd.isna(value):
                issues.append(
                    Issue(
                        position,
                        column,
                        "dtype",
                        f"Row {position}, column {column!r} is not a whole number.",
                    )
                )
            elif value not in SCORE_VALUES:
                issues.append(
                    Issue(
                        position,
                        column,
                        "closed_value",
                        f"Row {position}, column {column!r}: {value:g} is not on the "
                        f"scoring scale ({', '.join(str(v) for v in SCORE_VALUES)}).",
                    )
                )
    return issues


def validate(
    predictions: pd.DataFrame | str | Path,
    track: str | Track,
    expected_ids: list[str] | None = None,
    max_bytes: int = MAX_SUBMISSION_BYTES,
) -> ValidationReport:
    resolved = get_track(track)
    issues: list[Issue] = []

    if isinstance(predictions, (str, Path)):
        size = Path(predictions).stat().st_size
        if size > max_bytes:
            issues.append(
                Issue(
                    None,
                    None,
                    "too_large",
                    f"The file is {size:,} bytes. The limit is {max_bytes:,}.",
                )
            )
        frame = pd.read_csv(predictions)
    else:
        frame = predictions

    for column in resolved.submission_columns:
        if column not in frame.columns:
            issues.append(
                Issue(
                    None,
                    column,
                    "missing_column",
                    f"The header does not declare a column named {column!r}.",
                )
            )
    if any(i.code == "missing_column" for i in issues):
        return ValidationReport(False, issues[:MAX_REPORTED_ISSUES], len(frame))

    ids = frame[resolved.id_column].astype(str).str.strip()
    for position, value in enumerate(ids, start=1):
        if value in {"", "nan", "None"}:
            issues.append(
                Issue(
                    position,
                    resolved.id_column,
                    "empty_id",
                    f"Row {position} has no identifier.",
                )
            )
    duplicates = ids[ids.duplicated()].unique()
    for value in duplicates:
        issues.append(
            Issue(
                None,
                resolved.id_column,
                "duplicate_id",
                f"Identifier {value!r} appears more than once.",
            )
        )

    for column in resolved.target_columns:
        empty = frame[column].isna() | (frame[column].astype(str).str.strip() == "")
        for position in [i + 1 for i, e in enumerate(empty) if e]:
            issues.append(
                Issue(position, column, "dtype", f"Row {position}, column {column!r} is empty.")
            )

    if resolved.id == "aes":
        issues.extend(_scale_issues(frame))
    elif resolved.id == "math-diagnostic":
        issues.extend(_closed_issues(resolved, frame))
    else:
        issues.extend(_structured_issues(resolved, frame))

    if expected_ids is not None:
        wanted, given = set(expected_ids), set(ids)
        if len(frame) != len(wanted):
            issues.append(
                Issue(
                    None,
                    None,
                    "row_count",
                    f"Expected exactly {len(wanted)} data rows, found {len(frame)}.",
                )
            )
        for value in sorted(wanted - given)[:20]:
            issues.append(
                Issue(
                    None,
                    resolved.id_column,
                    "missing_id",
                    f"The submission is missing {value}.",
                )
            )
        for value in sorted(given - wanted)[:20]:
            issues.append(
                Issue(None, resolved.id_column, "unknown_id", f"{value} is not being graded.")
            )

    return ValidationReport(not issues, issues[:MAX_REPORTED_ISSUES], len(frame))


def build(track: str | Track, rows) -> pd.DataFrame:
    resolved = get_track(track)
    frame = pd.DataFrame(list(rows)) if not isinstance(rows, pd.DataFrame) else rows.copy()
    missing = [c for c in resolved.submission_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing column(s): {', '.join(missing)}")
    if resolved.id == "answer-sheet":
        frame["answers"] = [
            value if isinstance(value, str) else encode(value) for value in frame["answers"]
        ]
    if resolved.id == "aes":
        for column in COMPETENCES:
            frame[column] = pd.to_numeric(frame[column]).round().astype(int)
    return frame[list(resolved.submission_columns)]


def write(track: str | Track, rows, path: str | Path) -> Path:
    frame = build(track, rows)
    path = Path(path)
    frame.to_csv(path, index=False)
    return path

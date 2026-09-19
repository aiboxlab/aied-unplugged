from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..tracks import Track

SHOWN = 5


class SubmissionError(ValueError):
    pass


@dataclass
class GraderResult:
    track: str
    metrics: dict[str, float]
    primary_metric: str
    per_example: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_examples: int = 0

    @property
    def primary(self) -> float:
        return self.metrics[self.primary_metric]

    def __repr__(self) -> str:
        scores = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"GraderResult({self.track}, n={self.n_examples}, {scores})"

    def to_dict(self) -> dict[str, float]:
        return dict(self.metrics)


def as_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def listed(items: Sequence) -> str:
    items = list(items)
    head = ", ".join(str(i) for i in items[:SHOWN])
    return head if len(items) <= SHOWN else f"{head} and {len(items) - SHOWN} more"


def require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SubmissionError(
            f"{what} is missing the column(s): {listed(missing)}. "
            f"Expected: {', '.join(columns)}."
        )


def align(
    predictions: pd.DataFrame, references: pd.DataFrame, track: Track
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_column = track.id_column
    require_columns(predictions, [id_column], "The submission")
    require_columns(references, [id_column], "The reference")

    predictions = predictions.copy()
    references = references.copy()
    predictions[id_column] = predictions[id_column].astype(str).str.strip()
    references[id_column] = references[id_column].astype(str).str.strip()

    blank = predictions[id_column].isin(["", "nan", "None"])
    if blank.any():
        rows = [i + 1 for i in range(len(predictions)) if blank.iloc[i]]
        raise SubmissionError(
            f"{int(blank.sum())} row(s) have no value in {id_column!r}: row {listed(rows)}."
        )

    duplicated = predictions[id_column][predictions[id_column].duplicated()].unique()
    if len(duplicated):
        raise SubmissionError(
            f"{len(duplicated)} identifier(s) appear more than once: {listed(duplicated)}. "
            "Each item must appear exactly once."
        )

    wanted = set(references[id_column])
    given = set(predictions[id_column])

    missing = sorted(wanted - given)
    if missing:
        raise SubmissionError(
            f"The submission is missing {len(missing)} of {len(wanted)} required "
            f"item(s): {listed(missing)}."
        )
    unknown = sorted(given - wanted)
    if unknown:
        raise SubmissionError(
            f"The submission names {len(unknown)} item(s) that are not being graded: "
            f"{listed(unknown)}."
        )

    references = references.drop_duplicates(subset=id_column).set_index(id_column)
    predictions = predictions.set_index(id_column).loc[references.index]
    return predictions, references


def require_no_missing(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        empty = frame[column].isna() | (
            frame[column].astype(str).str.strip().isin(["", "nan", "None"])
        )
        if empty.any():
            ids = list(frame.index[empty][:SHOWN])
            raise SubmissionError(
                f"Column {column!r} is empty for {int(empty.sum())} item(s): {listed(ids)}."
            )

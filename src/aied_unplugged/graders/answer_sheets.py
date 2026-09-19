from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..tracks import ANSWER_SHEET, ANSWER_VALUES, LABEL_KEY, QUESTION_KEY
from . import metrics
from .common import (
    GraderResult,
    SubmissionError,
    align,
    as_frame,
    require_columns,
    require_no_missing,
)

TRACK = ANSWER_SHEET
COLUMN = "answers"
MISSING = "<missing>"


def decode(value, sheet_id: str, what: str) -> dict[int, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SubmissionError(f"{what} answers for {sheet_id} are not valid JSON.") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise SubmissionError(
            f"{what} answers for {sheet_id} must be a JSON array of "
            f'{{"{QUESTION_KEY}": n, "{LABEL_KEY}": "..."}} entries.'
        )

    answers: dict[int, str] = {}
    for position, entry in enumerate(value, start=1):
        if not isinstance(entry, dict) or QUESTION_KEY not in entry or LABEL_KEY not in entry:
            raise SubmissionError(
                f"{what} answers for {sheet_id}: entry {position} must be an object "
                f'with "{QUESTION_KEY}" and "{LABEL_KEY}".'
            )
        number, label = entry[QUESTION_KEY], entry[LABEL_KEY]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or int(number) != number
        ):
            raise SubmissionError(
                f"{what} answers for {sheet_id}: entry {position} has a "
                f'"{QUESTION_KEY}" that is not a whole number. Write 1, not "1".'
            )
        number = int(number)
        if number < 1:
            raise SubmissionError(
                f"{what} answers for {sheet_id}: entry {position} numbers a question below 1."
            )
        if number in answers:
            raise SubmissionError(
                f"{what} answers for {sheet_id}: question {number} appears more than once."
            )
        if not isinstance(label, str) or label not in ANSWER_VALUES:
            raise SubmissionError(
                f"{what} answers for {sheet_id}: entry {position} carries a value that "
                f"is not one of the {len(ANSWER_VALUES)} permitted answers."
            )
        answers[number] = label
    return answers


def score(
    predictions: pd.DataFrame | str | Path, references: pd.DataFrame | str | Path
) -> GraderResult:
    predictions, references = as_frame(predictions), as_frame(references)
    require_columns(predictions, TRACK.submission_columns, "The submission")
    require_columns(references, [COLUMN], "The reference")
    predictions, references = align(predictions, references, TRACK)
    require_no_missing(predictions, [COLUMN])

    rows, truth_cells, guess_cells = [], [], []
    for sheet_id in references.index:
        expected = decode(references.loc[sheet_id, COLUMN], sheet_id, "The reference")
        given = decode(predictions.loc[sheet_id, COLUMN], sheet_id, "The submission")

        correct = sum(1 for q, label in expected.items() if given.get(q) == label)
        for question, label in expected.items():
            truth_cells.append(label)
            guess_cells.append(given.get(question, MISSING))

        rows.append(
            {
                TRACK.id_column: sheet_id,
                "questions": len(expected),
                "answered": len(given),
                "correct": correct,
                "cell_accuracy": correct / len(expected) if expected else 0.0,
                "exact": given == expected,
            }
        )

    per_example = pd.DataFrame(rows).set_index(TRACK.id_column)
    present = [label for label in ANSWER_VALUES if label in set(truth_cells)]
    total = len(truth_cells)
    values = {
        "cell_accuracy": (sum(per_example["correct"]) / total) if total else 0.0,
        "sheet_exact_match": (per_example["exact"].mean() if len(per_example) else 0.0),
        "macro_f1": metrics.macro_f1(truth_cells, guess_cells, present),
    }
    return GraderResult(
        track=TRACK.id,
        metrics={k: float(values[k]) for k in TRACK.metrics},
        primary_metric=TRACK.primary_metric,
        per_example=per_example,
        n_examples=len(per_example),
    )


def grade(predictions, ground_truth) -> dict[str, float]:
    return score(predictions, ground_truth).to_dict()


def encode(answers) -> str:
    if isinstance(answers, dict):
        answers = [{QUESTION_KEY: int(q), LABEL_KEY: answers[q]} for q in sorted(answers)]
    if hasattr(answers, "tolist"):
        answers = answers.tolist()
    return json.dumps(
        [{QUESTION_KEY: int(a[QUESTION_KEY]), LABEL_KEY: a[LABEL_KEY]} for a in answers]
    )

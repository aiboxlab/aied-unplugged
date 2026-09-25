from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..tracks import DIAGNOSTIC_LABELS, MATH
from . import metrics
from .common import (
    GraderResult,
    SubmissionError,
    align,
    as_frame,
    listed,
    require_columns,
    require_no_missing,
    require_submission_columns,
)

TRACK = MATH
COLUMN = "diagnostic"


def _labels(frame: pd.DataFrame, what: str) -> list[str]:
    values = frame[COLUMN].astype(str)
    unknown = ~values.isin(DIAGNOSTIC_LABELS)
    if unknown.any():
        raise SubmissionError(
            f"{what} column {COLUMN!r} has {int(unknown.sum())} value(s) outside the "
            f"published taxonomy: {listed(list(frame.index[unknown]))}. "
            "Labels are compared exactly as written, including case."
        )
    return list(values)


def score(
    predictions: pd.DataFrame | str | Path, references: pd.DataFrame | str | Path
) -> GraderResult:
    predictions, references = as_frame(predictions), as_frame(references)
    require_submission_columns(predictions, TRACK)
    require_columns(references, [COLUMN], "The reference")
    predictions, references = align(predictions, references, TRACK)
    require_no_missing(predictions, [COLUMN])

    truth = _labels(references, "The reference")
    guess = _labels(predictions, "The submission")
    present = [label for label in DIAGNOSTIC_LABELS if label in set(truth)]

    per_example = pd.DataFrame(
        {
            "true": truth,
            "pred": guess,
            "correct": [t == g for t, g in zip(truth, guess, strict=False)],
        },
        index=predictions.index,
    )
    values = {
        "macro_f1": metrics.macro_f1(truth, guess, present),
        "accuracy": metrics.accuracy(truth, guess),
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


def per_class(
    predictions: pd.DataFrame | str | Path, references: pd.DataFrame | str | Path
) -> pd.DataFrame:
    result = score(predictions, references)
    truth = list(result.per_example["true"])
    guess = list(result.per_example["pred"])
    present = [label for label in DIAGNOSTIC_LABELS if label in set(truth)]
    f1 = metrics.per_class_f1(truth, guess, present)
    return pd.DataFrame(
        {
            "label": present,
            "support": [truth.count(label) for label in present],
            "predicted": [guess.count(label) for label in present],
            "f1": [f1[label] for label in present],
        }
    ).sort_values("support", ascending=False, ignore_index=True)

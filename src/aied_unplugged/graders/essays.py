from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..tracks import AES, COMPETENCES, SCORE_VALUES
from . import metrics
from .common import (
    GraderResult,
    SubmissionError,
    align,
    as_frame,
    listed,
    require_columns,
    require_no_missing,
)

TRACK = AES
_INDEX = {value: i for i, value in enumerate(SCORE_VALUES)}


def _scores(frame: pd.DataFrame, column: str, what: str) -> list[int]:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    bad = numeric.isna()
    if bad.any():
        raise SubmissionError(
            f"{what} column {column!r} must be whole numbers; "
            f"{int(bad.sum())} are not: {listed(list(frame.index[bad]))}."
        )
    off_grid = ~numeric.isin(SCORE_VALUES)
    if off_grid.any():
        allowed = ", ".join(str(v) for v in SCORE_VALUES)
        raise SubmissionError(
            f"{what} column {column!r} has {int(off_grid.sum())} value(s) outside the "
            f"scoring scale ({allowed}): {listed(list(frame.index[off_grid]))}."
        )
    return [int(v) for v in numeric]


def score(
    predictions: pd.DataFrame | str | Path, references: pd.DataFrame | str | Path
) -> GraderResult:
    predictions, references = as_frame(predictions), as_frame(references)
    require_columns(predictions, TRACK.submission_columns, "The submission")
    require_columns(references, list(COMPETENCES), "The reference")
    predictions, references = align(predictions, references, TRACK)
    require_no_missing(predictions, list(COMPETENCES))

    per_example = pd.DataFrame(index=predictions.index)
    kappas, errors, pooled_true, pooled_pred = {}, {}, [], []

    for competence in COMPETENCES:
        truth = _scores(references, competence, "The reference")
        guess = _scores(predictions, competence, "The submission")
        kappas[competence] = metrics.quadratic_weighted_kappa(
            [_INDEX[v] for v in truth], [_INDEX[v] for v in guess], range(len(SCORE_VALUES))
        )
        errors[competence] = metrics.rmse(truth, guess)
        per_example[f"true_{competence}"] = truth
        per_example[f"pred_{competence}"] = guess
        per_example[f"error_{competence}"] = [g - t for t, g in zip(truth, guess, strict=False)]
        pooled_true.extend(truth)
        pooled_pred.extend(guess)

    values = {
        "qwk_mean": sum(kappas.values()) / len(kappas),
        "exact_match": metrics.accuracy(pooled_true, pooled_pred),
        "rmse": metrics.rmse(pooled_true, pooled_pred),
        **{f"qwk_{k}": v for k, v in kappas.items()},
        **{f"rmse_{k}": v for k, v in errors.items()},
    }
    per_example["exact_all"] = [
        all(
            per_example[f"true_{c}"].iloc[i] == per_example[f"pred_{c}"].iloc[i]
            for c in COMPETENCES
        )
        for i in range(len(per_example))
    ]
    return GraderResult(
        track=TRACK.id,
        metrics={k: float(values[k]) for k in TRACK.metrics},
        primary_metric=TRACK.primary_metric,
        per_example=per_example,
        n_examples=len(per_example),
    )


def grade(predictions, ground_truth) -> dict[str, float]:
    return score(predictions, ground_truth).to_dict()

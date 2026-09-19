from __future__ import annotations

from collections.abc import Callable

from ..tracks import Track, get_track
from . import answer_sheets, essays, math_diagnostic, metrics
from .common import GraderResult, SubmissionError

_MODULES = {
    "aes": essays,
    "math-diagnostic": math_diagnostic,
    "answer-sheet": answer_sheets,
}


def get_grader(track: str | Track) -> Callable[..., GraderResult]:
    return _MODULES[get_track(track).id].score


def grade(track: str | Track, predictions, ground_truth) -> dict[str, float]:
    return _MODULES[get_track(track).id].grade(predictions, ground_truth)


def evaluate(track: str | Track, predictions, references) -> GraderResult:
    return get_grader(track)(predictions, references)


__all__ = [
    "GraderResult",
    "SubmissionError",
    "answer_sheets",
    "essays",
    "evaluate",
    "get_grader",
    "grade",
    "math_diagnostic",
    "metrics",
]

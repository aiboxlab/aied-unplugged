from __future__ import annotations

from .data import (
    graded_ids,
    load_metadata,
    load_schema,
    load_track,
    open_image,
    sample_submission,
    verify,
)
from .graders import GraderResult, SubmissionError, evaluate, get_grader
from .submission import ValidationReport, build, validate, write
from .tracks import (
    ANSWER_VALUES,
    COMPETENCES,
    DIAGNOSTIC_LABELS,
    SCORE_VALUES,
    TRACKS,
    Track,
    get_track,
)

__version__ = "0.1.0"

__all__ = [
    "ANSWER_VALUES",
    "COMPETENCES",
    "DIAGNOSTIC_LABELS",
    "SCORE_VALUES",
    "TRACKS",
    "GraderResult",
    "SubmissionError",
    "Track",
    "ValidationReport",
    "__version__",
    "build",
    "evaluate",
    "get_grader",
    "get_track",
    "graded_ids",
    "load_metadata",
    "load_schema",
    "load_track",
    "open_image",
    "sample_submission",
    "validate",
    "verify",
    "write",
]

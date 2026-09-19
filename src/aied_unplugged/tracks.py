from __future__ import annotations

from dataclasses import dataclass, field

COMPETENCES: tuple[str, ...] = tuple(f"competence_{i}" for i in range(1, 6))
SCORE_VALUES: tuple[int, ...] = (0, 40, 80, 120, 160, 200)

DIAGNOSTIC_LABELS: tuple[str, ...] = (
    "Correct answer",
    "Multiple answers",
    "Correct direct answer",
    "Incorrect direct answer",
    "Interpretation error",
    "Wrong operation",
    "Operation setup error",
    "Partial answer",
    "Incomplete calculation",
    "Addition error",
    "Subtraction error",
    "Multiplication error",
    "Division error",
)

ANSWER_VALUES: tuple[str, ...] = ("A", "B", "C", "D", "Blank", "Multiple", "Other")

QUESTION_KEY = "question_number"
LABEL_KEY = "label"
MAX_SUBMISSION_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class Track:
    id: str
    config: str
    name: str
    id_column: str
    target_columns: tuple[str, ...]
    image_columns: tuple[str, ...]
    metrics: tuple[str, ...]
    primary_metric: str
    sample_file: str
    allowed_values: tuple[str, ...] = ()
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def submission_columns(self) -> tuple[str, ...]:
        return (self.id_column, *self.target_columns)


AES = Track(
    id="aes",
    config="essays",
    name="Automated Essay Scoring",
    id_column="essay_id",
    target_columns=COMPETENCES,
    image_columns=("image",),
    metrics=(
        "qwk_mean",
        "exact_match",
        "rmse",
        *(f"qwk_{c}" for c in COMPETENCES),
        *(f"rmse_{c}" for c in COMPETENCES),
    ),
    primary_metric="qwk_mean",
    sample_file="sample_submission_aes.csv",
    aliases=("essay", "essays"),
)

MATH = Track(
    id="math-diagnostic",
    config="math",
    name="Mathematical Exam Diagnostic",
    id_column="equation_id",
    target_columns=("diagnostic",),
    image_columns=("question_image", "equation_image"),
    metrics=("macro_f1", "accuracy"),
    primary_metric="macro_f1",
    sample_file="sample_submission_math_diagnostic.csv",
    allowed_values=DIAGNOSTIC_LABELS,
    aliases=("math", "math_diagnostic"),
)

ANSWER_SHEET = Track(
    id="answer-sheet",
    config="answer_sheets",
    name="Answer Sheet Detection",
    id_column="sheet_id",
    target_columns=("answers",),
    image_columns=("image",),
    metrics=("cell_accuracy", "sheet_exact_match", "macro_f1"),
    primary_metric="cell_accuracy",
    sample_file="sample_submission_answer_sheet.csv",
    allowed_values=ANSWER_VALUES,
    aliases=("sheets", "answer_sheets", "answer_sheet", "sheet"),
)

TRACKS: tuple[Track, ...] = (AES, MATH, ANSWER_SHEET)


def get_track(track: str | Track) -> Track:
    if isinstance(track, Track):
        return track
    key = track.strip().lower().replace(" ", "-")
    for candidate in TRACKS:
        names = {candidate.id, candidate.config, *candidate.aliases}
        if key in names or key.replace("-", "_") in names:
            return candidate
    known = ", ".join(t.id for t in TRACKS)
    raise KeyError(f"unknown track {track!r}; expected one of {known}")

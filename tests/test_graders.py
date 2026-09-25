from __future__ import annotations

import json

import pandas as pd
import pytest

from aied_unplugged import evaluate, get_track
from aied_unplugged.graders import (
    SubmissionError,
    answer_sheets,
    essays,
    math_diagnostic,
    metrics,
)


def essay_reference():
    return pd.DataFrame(
        {
            "essay_id": ["a", "b", "c", "d"],
            "competence_1": [0, 80, 160, 200],
            "competence_2": [40, 80, 120, 160],
            "competence_3": [80, 80, 80, 80],
            "competence_4": [0, 40, 80, 120],
            "competence_5": [200, 160, 120, 80],
        }
    )


def math_reference():
    return pd.DataFrame(
        {
            "equation_id": ["m1", "m2", "m3", "m4"],
            "diagnostic": [
                "Correct answer",
                "Correct answer",
                "Addition error",
                "Wrong operation",
            ],
        }
    )


def sheet_reference():
    return pd.DataFrame(
        {
            "sheet_id": ["s1", "s2"],
            "answers": [
                json.dumps(
                    [{"question_number": i, "label": v} for i, v in enumerate("ABCD", 1)]
                ),
                json.dumps(
                    [
                        {"question_number": i, "label": v}
                        for i, v in enumerate(["A", "Blank", "C"], 1)
                    ]
                ),
            ],
        }
    )


def test_perfect_submission_scores_one():
    reference = essay_reference()
    result = essays.score(reference.copy(), reference)
    assert result.metrics["qwk_mean"] == 1.0
    assert result.metrics["exact_match"] == 1.0
    assert result.metrics["rmse"] == 0.0
    assert result.primary == 1.0
    assert set(result.metrics) == set(get_track("aes").metrics)


def test_constant_prediction_does_not_produce_nan():
    reference = essay_reference()
    flat = reference.copy()
    for column in [f"competence_{i}" for i in range(1, 6)]:
        flat[column] = 80
    result = essays.score(flat, reference)
    assert result.metrics["qwk_competence_1"] == 0.0
    assert result.metrics["qwk_competence_3"] == 1.0
    assert result.metrics["qwk_mean"] == pytest.approx(0.2)
    assert not any(pd.isna(v) for v in result.metrics.values())


def test_essay_rejects_off_grid_scores():
    reference = essay_reference()
    wrong = reference.copy()
    wrong.loc[0, "competence_1"] = 100
    with pytest.raises(SubmissionError, match="scoring scale"):
        essays.score(wrong, reference)


def test_essay_rejects_non_integer():
    reference = essay_reference()
    wrong = reference.copy().astype({"competence_2": object})
    wrong.loc[1, "competence_2"] = "high"
    with pytest.raises(SubmissionError, match="whole numbers"):
        essays.score(wrong, reference)


def test_missing_rows_are_refused():
    reference = essay_reference()
    with pytest.raises(SubmissionError, match="missing 1 of 4"):
        essays.score(reference.iloc[:3], reference)


def test_unknown_rows_are_refused():
    reference = essay_reference()
    extra = pd.concat([reference, reference.tail(1).assign(essay_id="zz")], ignore_index=True)
    with pytest.raises(SubmissionError, match="not being graded"):
        essays.score(extra, reference)


def test_duplicate_ids_are_refused():
    reference = essay_reference()
    doubled = pd.concat([reference, reference.tail(1)], ignore_index=True)
    with pytest.raises(SubmissionError, match="more than once"):
        essays.score(doubled, reference)


def test_missing_column_is_refused():
    reference = essay_reference()
    with pytest.raises(SubmissionError, match="missing the column"):
        essays.score(reference.drop(columns=["competence_5"]), reference)


def test_empty_cell_is_refused():
    reference = essay_reference()
    blank = reference.copy().astype({"competence_1": object})
    blank.loc[2, "competence_1"] = None
    with pytest.raises(SubmissionError, match="empty"):
        essays.score(blank, reference)


def test_math_macro_f1_averages_over_reference_classes():
    reference = math_reference()
    prediction = reference.copy()
    prediction.loc[3, "diagnostic"] = "Addition error"
    result = math_diagnostic.score(prediction, reference)
    assert result.metrics["accuracy"] == 0.75
    # Correct answer 1.0, Addition error 2/3, Wrong operation 0.0
    assert result.metrics["macro_f1"] == pytest.approx((1.0 + 2 / 3 + 0.0) / 3)


def test_math_rejects_label_outside_taxonomy():
    reference = math_reference()
    wrong = reference.copy()
    wrong.loc[0, "diagnostic"] = "correct answer"
    with pytest.raises(SubmissionError, match="taxonomy"):
        math_diagnostic.score(wrong, reference)


def test_math_per_class_table():
    reference = math_reference()
    table = math_diagnostic.per_class(reference.copy(), reference)
    assert list(table.columns) == ["label", "support", "predicted", "f1"]
    assert table.support.sum() == 4


def test_sheet_metrics():
    reference = sheet_reference()
    prediction = reference.copy()
    answers = json.loads(prediction.loc[0, "answers"])
    answers[0]["label"] = "B"
    prediction.loc[0, "answers"] = json.dumps(answers)

    result = answer_sheets.score(prediction, reference)
    assert result.metrics["cell_accuracy"] == pytest.approx(6 / 7)
    assert result.metrics["sheet_exact_match"] == 0.5
    assert result.per_example.loc["s1", "correct"] == 3


@pytest.mark.parametrize(
    ("questions", "message"), [((1, 2), "missing question"), ((1, 2, 3, 4), "not on the sheet")]
)
def test_sheet_rejects_a_question_set_unlike_the_sheet(questions, message):
    reference = sheet_reference()
    prediction = reference.copy()
    prediction.loc[1, "answers"] = json.dumps(
        [{"question_number": q, "label": "A"} for q in questions]
    )
    with pytest.raises(SubmissionError, match=message):
        answer_sheets.score(prediction, reference)


def test_sheet_rejects_float_question_number():
    reference = sheet_reference()
    prediction = reference.copy()
    prediction.loc[1, "answers"] = (
        '[{"question_number": 1.0, "label": "A"}, {"question_number": 2, "label": "Blank"},'
        ' {"question_number": 3, "label": "C"}]'
    )
    with pytest.raises(SubmissionError, match="not 1.0"):
        answer_sheets.score(prediction, reference)


def test_math_rejects_padded_label():
    reference = math_reference()
    prediction = reference.copy()
    prediction.loc[0, "diagnostic"] = " Correct answer "
    with pytest.raises(SubmissionError, match="taxonomy"):
        math_diagnostic.score(prediction, reference)


@pytest.mark.parametrize(
    ("grader", "reference"),
    [
        (essays, essay_reference),
        (math_diagnostic, math_reference),
        (answer_sheets, sheet_reference),
    ],
)
def test_graders_reject_extra_columns_and_empty_files(grader, reference, tmp_path):
    frame = reference()
    with pytest.raises(SubmissionError, match="does not accept: confidence"):
        grader.score(frame.assign(confidence=0.5), frame)
    empty = tmp_path / "submission.csv"
    empty.write_text("")
    with pytest.raises(SubmissionError, match="empty"):
        grader.score(empty, frame)


def test_sheet_rejects_bad_json():
    reference = sheet_reference()
    prediction = reference.copy()
    prediction.loc[0, "answers"] = "not json"
    with pytest.raises(SubmissionError, match="valid JSON"):
        answer_sheets.score(prediction, reference)


def test_sheet_rejects_string_question_number():
    reference = sheet_reference()
    prediction = reference.copy()
    prediction.loc[0, "answers"] = json.dumps([{"question_number": "1", "label": "A"}] * 1)
    with pytest.raises(SubmissionError, match="whole number"):
        answer_sheets.score(prediction, reference)


def test_sheet_rejects_unknown_label():
    reference = sheet_reference()
    prediction = reference.copy()
    prediction.loc[0, "answers"] = json.dumps([{"question_number": 1, "label": "E"}])
    with pytest.raises(SubmissionError, match="permitted"):
        answer_sheets.score(prediction, reference)


def test_sheet_accepts_list_references_from_the_dataset():
    reference = sheet_reference()
    as_lists = reference.assign(answers=[json.loads(a) for a in reference.answers])
    result = answer_sheets.score(reference, as_lists)
    assert result.metrics["cell_accuracy"] == 1.0


def test_grade_returns_flat_floats_for_the_leaderboard():
    for track, reference in (
        ("aes", essay_reference()),
        ("math-diagnostic", math_reference()),
        ("answer-sheet", sheet_reference()),
    ):
        declared = set(get_track(track).metrics)
        result = evaluate(track, reference.copy(), reference).to_dict()
        assert set(result) == declared
        assert all(isinstance(v, float) for v in result.values())


def test_error_messages_do_not_quote_reference_labels():
    reference = math_reference()
    wrong = reference.copy()
    wrong.loc[0, "diagnostic"] = "Nonsense label"
    with pytest.raises(SubmissionError) as caught:
        math_diagnostic.score(wrong, reference)
    assert "Correct answer" not in str(caught.value)


def test_qwk_matches_a_worked_example():
    true = [0, 1, 2, 3]
    pred = [0, 1, 2, 2]
    # one off-diagonal cell at (3, 2): 1 - (1/9) / (32/36)
    assert metrics.quadratic_weighted_kappa(true, pred, range(4)) == pytest.approx(0.875)


def test_macro_f1_ignores_classes_absent_from_reference():
    true = ["a", "a", "b"]
    pred = ["a", "a", "b"]
    assert metrics.macro_f1(true, pred, ["a", "b"]) == 1.0
    assert metrics.macro_f1(true, pred, ["a", "b", "c"]) == pytest.approx(2 / 3)

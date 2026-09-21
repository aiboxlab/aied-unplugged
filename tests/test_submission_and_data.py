from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from aied_unplugged import build, data, get_track, validate, write
from aied_unplugged.data import find_release

RELEASE = Path(__file__).resolve().parents[2] / "competition-dataset"
needs_release = pytest.mark.skipif(not RELEASE.exists(), reason="no release tree")


def test_validate_accepts_a_good_essay_file():
    frame = pd.DataFrame(
        {"essay_id": ["a", "b"], **{f"competence_{i}": [80, 120] for i in range(1, 6)}}
    )
    report = validate(frame, "aes", expected_ids=["a", "b"])
    assert report.valid and bool(report)


def test_validate_reports_off_grid_scores():
    frame = pd.DataFrame({"essay_id": ["a"], **{f"competence_{i}": [90] for i in range(1, 6)}})
    report = validate(frame, "aes")
    assert not report.valid
    assert all(issue.code == "closed_value" for issue in report.issues)


def test_validate_reports_missing_and_unknown_ids():
    frame = pd.DataFrame({"equation_id": ["m1", "zz"], "diagnostic": ["Correct answer"] * 2})
    report = validate(frame, "math", expected_ids=["m1", "m2"])
    codes = {issue.code for issue in report.issues}
    assert {"missing_id", "unknown_id"} <= codes


def test_validate_reports_structured_problems():
    frame = pd.DataFrame(
        {
            "sheet_id": ["s1", "s2", "s3"],
            "answers": [
                json.dumps([{"question_number": "1", "label": "A"}]),
                json.dumps([{"question_number": 1, "label": "Z"}]),
                "{}",
            ],
        }
    )
    report = validate(frame, "answer-sheet")
    codes = {issue.code for issue in report.issues}
    assert {"structured_keys", "structured_value", "structured_parse"} <= codes


def test_validate_reports_duplicates_and_blanks():
    frame = pd.DataFrame(
        {"equation_id": ["m1", "m1", " "], "diagnostic": ["Correct answer"] * 3}
    )
    report = validate(frame, "math")
    codes = {issue.code for issue in report.issues}
    assert "duplicate_id" in codes and "empty_id" in codes


def test_build_encodes_answers_and_orders_columns(tmp_path):
    rows = [
        {"sheet_id": "s1", "answers": [{"question_number": 1, "label": "A"}]},
        {"sheet_id": "s2", "answers": {1: "B", 2: "C"}},
    ]
    frame = build("answer-sheet", rows)
    assert list(frame.columns) == ["sheet_id", "answers"]
    assert json.loads(frame.answers.iloc[1]) == [
        {"question_number": 1, "label": "B"},
        {"question_number": 2, "label": "C"},
    ]
    path = write("answer-sheet", rows, tmp_path / "submission.csv")
    assert validate(path, "answer-sheet").valid


def test_build_requires_every_column():
    with pytest.raises(ValueError, match="missing column"):
        build("aes", [{"essay_id": "a", "competence_1": 80}])


def test_get_track_accepts_the_spellings_people_use():
    assert get_track("math").id == "math-diagnostic"
    assert get_track("math_diagnostic").id == "math-diagnostic"
    assert get_track("essays").id == "aes"
    assert get_track("sheets").id == "answer-sheet"
    with pytest.raises(KeyError):
        get_track("nope")


def test_download_registers_the_kaggle_tree_as_the_default_root(tmp_path, monkeypatch):
    tree = tmp_path / "download" / "aied-unplugged-preview"
    (tree / "metadata").mkdir(parents=True)
    (tree / "schema").mkdir()
    kagglehub = types.SimpleNamespace(
        dataset_download=lambda dataset, force_download=False: str(tree.parent)
    )
    monkeypatch.setitem(sys.modules, "kagglehub", kagglehub)
    monkeypatch.setattr(data, "_downloaded", None)
    monkeypatch.delenv("AIED_UNPLUGGED_DATA", raising=False)
    monkeypatch.chdir(tmp_path)

    assert data.download() == tree.resolve()
    assert find_release() == tree.resolve()


@needs_release
def test_release_loads_and_samples_validate():
    from aied_unplugged import graded_ids, load_metadata, sample_submission

    assert find_release(RELEASE) == RELEASE.resolve()
    for name in ("essays", "math", "answer_sheets"):
        track = get_track(name)
        frame = load_metadata(track, RELEASE)
        assert len(frame) and set(frame.split) <= {"train", "validation", "test"}
        for column in track.image_columns:
            assert Path(frame[f"{column}_path"].iloc[0]).exists()

        sample = sample_submission(track, RELEASE)
        report = validate(sample, track, expected_ids=graded_ids(track, RELEASE))
        assert report.valid, repr(report)


@needs_release
def test_samples_score_against_the_answer_keys():
    from aied_unplugged import evaluate

    keys = RELEASE.parent / "private" / "ground_truth"
    if not keys.exists():
        pytest.skip("no answer keys")
    for track_name, key in (
        ("aes", "aes.csv"),
        ("math-diagnostic", "math_diagnostic.csv"),
        ("answer-sheet", "answer_sheet.csv"),
    ):
        track = get_track(track_name)
        sample = RELEASE / "samples" / track.sample_file
        result = evaluate(track, sample, keys / key)
        assert 0.0 <= result.primary <= 1.0
        assert set(result.metrics) == set(track.metrics)

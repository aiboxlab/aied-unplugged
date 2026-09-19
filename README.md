# aied-unplugged

Python SDK for the [AIED Preview Competition](https://tools-competition.org/winner/aied/):
the official graders, dataset loaders, exploration helpers, and scaffolding to run a
Hugging Face model on a track.

```sh
pip install aied-unplugged            # graders, loaders, submission tooling
pip install 'aied-unplugged[all]'     # + datasets, matplotlib, transformers
```

The SDK is optional. It reads the published dataset and writes the published
submission format.

---

## The tracks

| Track | `track` | Predict | Ranked by |
|---|---|---|---|
| Automated Essay Scoring | `aes`, `essays` | `competence_1` … `competence_5` | mean QWK |
| Mathematical Exam Diagnostic | `math-diagnostic`, `math` | `diagnostic` | macro F1 |
| Answer Sheet Detection | `answer-sheet`, `sheets` | `answers` | cell accuracy |

---

## 1. Grading

Same implementation the leaderboard runs, so a local score matches the submitted one.

```python
from aied_unplugged import evaluate

result = evaluate("math", "my_predictions.csv", "validation_reference.csv")
result.primary  # 0.41…  the ranking metric
result.metrics  # every metric the leaderboard records
result.per_example  # one row per item: true, pred, correct
```

References can be a CSV or a `validation` split loaded from the dataset, with the
same target columns either way.

```python
from aied_unplugged import load_track, evaluate

validation = load_track("math", "validation")
evaluate("math", predictions, validation)
```

### Metrics

| Track | Primary | Definition | Also reported |
|---|---|---|---|
| Essays | mean QWK | Quadratic weighted kappa per competence, averaged over the five. Uses the full six-point scale rather than the values in the split, so scores don't shift with the split. | `exact_match` (per competence cell), `rmse` pooled over the five, `qwk_competence_1`…`_5`, `rmse_competence_1`…`_5` |
| Math | macro F1 | Unweighted mean of per-class F1 over the classes present in the reference. Predicting an absent class still costs you a false negative on the correct class. | `accuracy` |
| Answer sheets | cell accuracy | Correct cells over total reference cells, so a 26-question sheet counts more than a 16-question one. An omitted question counts as wrong. | `sheet_exact_match` (1 only when every question on a sheet matches), cell-level `macro_f1` |

Two details worth knowing. A constant prediction zeroes the QWK expected-agreement
denominator; that case scores 1.0 if the prediction is right everywhere and 0.0
otherwise, instead of NaN. Macro F1 skips classes absent from the reference because
averaging over the full taxonomy would hand every submission a guaranteed zero for
classes the split never uses, and the preview sample omits two of the thirteen.

Per-competence numbers are worth reading on the essay track: a model can hold a
decent mean while collapsing on competence 5.

### Validation checks

A grader rejects the whole file rather than scoring the part it can read. It raises
`SubmissionError` naming the ids and columns at fault, and never quotes a target
value. Rejected files include:

- a missing column
- a blank or duplicated identifier
- an empty prediction
- an ungraded row, or a missing row
- a competence score off the 0/40/80/120/160/200 grid
- a diagnostic outside the thirteen labels
- an answer-sheet value outside the seven
- a `question_number` written as `"1"` instead of `1`
- a question repeated within a sheet

---

## 2. Loading

```python
from aied_unplugged import load_track, load_metadata, open_image

splits = load_track("essays")  # local release tree if present, else the Hub
splits.train, splits.validation, splits.test

train = load_track("math", "train")  # one split
image = open_image(train.iloc[0], "equation_image")
```

Local loading looks for `competition-dataset/` in the working directory, the path
in `AIED_UNPLUGGED_DATA`, or a `root=` you pass, and returns pandas DataFrames
carrying the Parquet metadata plus resolved absolute paths in `<column>_path`.

```python
load_track("math", source="hf")  # datasets.DatasetDict with Image() features
load_schema("math")  # the published taxonomy, fields and metrics
graded_ids("aes")  # the ids a submission must carry
sample_submission("aes")  # the published placeholder file
verify()  # re-check every SHA-256 in the release
```

---

## 3. Exploring

Needs `aied-unplugged[explore]`.

```python
from aied_unplugged import explore

explore.summary()  # rows per track and split
explore.label_distribution("math")  # counts and shares, taxonomy order
explore.plot_label_distribution("math")
explore.describe_taxonomy("math")  # labels with their Portuguese source terms

explore.show_essay("essay-1-3")  # the page, with its five competence scores
explore.show_math("math-604-01")  # question above, student working below
explore.show_sheet(0)  # a sheet with what was marked
explore.render_math("math-604-01")  # the same pair as one PIL image
```

Name an item by id, by position, or by passing its metadata row.

---

## 4. Running a model

Needs `aied-unplugged[models]`. Two strategies, enough to produce a valid
submission and a baseline score.

```python
from aied_unplugged import models, submission

# Predict the training majority for everything.
predictions = models.majority_baseline("math")

# Zero-shot with a local vision-language model.
predictions = models.predict("math", "Qwen/Qwen2.5-VL-3B-Instruct", limit=20)

submission.write("math", predictions, "submission.csv")
```

`predict` prompts the model once per item with the track's images and parses the
reply into the submission schema. Unparseable replies fall back to a safe default,
so every run produces a gradeable file.

Fine-tuning covers the math track only, as image classification over the student's
working:

```python
trainer = models.finetune(model="google/vit-base-patch16-224-in21k", epochs=3)
predictions = models.predict_with_classifier(trainer, split="test")
```

Not included: essay or answer-sheet fine-tuning, multi-GPU, LoRA, hyperparameter
search.

---

## 5. Submissions

```python
from aied_unplugged import build, validate, write, graded_ids

frame = build("answer-sheet", [{"sheet_id": "s1", "answers": {1: "A", 2: "Blank"}}])
report = validate(frame, "answer-sheet", expected_ids=graded_ids("answer-sheet"))
if not report.valid:
    print(report)
write("answer-sheet", frame, "submission.csv")
```

`validate` runs the checks the competition site runs before upload, including the
JSON encoding of the answer-sheet column, plus a competence-scale check the site
does not yet perform. `build` accepts answers as a list of records or as a
`{question_number: label}` mapping, and encodes either correctly.

---

## Development

```sh
uv run --with-editable . --with pytest pytest -q
uv run ruff check .
```

Tests needing the release tree skip when `competition-dataset/` is absent.

MIT licensed. The dataset itself is CC BY 4.0 and is published separately.

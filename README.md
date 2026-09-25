# AIED-Unplugged Dataset SDK

[![PyPI](https://img.shields.io/pypi/v/aied-unplugged)](https://pypi.org/project/aied-unplugged/)
[![Python](https://img.shields.io/pypi/pyversions/aied-unplugged)](https://pypi.org/project/aied-unplugged/)
[![License](https://img.shields.io/pypi/l/aied-unplugged)](https://github.com/aiboxlab/aied-unplugged/blob/main/LICENSE)

Python SDK for the [AIED Preview Competition](https://tools-competition.org/winner/aied/):
the official graders, dataset loaders, exploration helpers, and scaffolding to run a
Hugging Face model on a track.

```sh
pip install aied-unplugged            # graders, loaders, submission tooling
pip install 'aied-unplugged[all]'     # + kagglehub, datasets, matplotlib, transformers
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

```python
from aied_unplugged import evaluate

result = evaluate("math", "my_predictions.csv", "validation_reference.csv")
result.primary  # 0.41…  the ranking metric
result.metrics  # every metric the track reports
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
| Answer sheets | cell accuracy | Correct cells over total reference cells, so a 26-question sheet counts more than a 16-question one. | `sheet_exact_match` (1 only when every question on a sheet matches), cell-level `macro_f1` |

A constant prediction zeroes the QWK expected-agreement denominator. That case
scores 1.0 when the prediction is right everywhere and 0.0 otherwise, instead of
NaN. Macro F1 skips classes absent from the reference: averaging over the full
taxonomy hands you a guaranteed zero for classes the split never uses, and the
preview sample omits two of the thirteen.

### Validation checks

A grader rejects the whole file instead of scoring the rows it can read. It raises
`SubmissionError` naming the ids and columns at fault, without quoting a target
value. Rejected files include:

- an empty file, a missing column, or an extra column
- a blank or duplicated identifier
- an empty prediction
- an ungraded row, or a missing row
- a competence score off the 0/40/80/120/160/200 grid
- a diagnostic outside the thirteen labels
- an answer-sheet value outside the seven
- a `question_number` that is not an integer (`1.0` and `"1"` both fail)
- a question repeated, missing, or not on the sheet

---

## 2. Loading

### Getting the dataset

`download()` pulls the dataset from Kaggle. Needs `aied-unplugged[kaggle]`.

```python
from aied_unplugged import download, load_track

download()  # into the kagglehub cache, returns the path
load_track("math")  # the loaders below now find it
load_track("math", source="kaggle")  # same, in one call
load_track("math", source="hf")  # in-memory DatasetDict from the HF Hub
```

Later calls reuse the cache; pass `force=True` to re-fetch. `download()` also sets
the default root for the process, so `explore` and `verify` run without `root=`.

Both mirrors carry the same release:

- Kaggle: <https://www.kaggle.com/datasets/aibox-lab/aied-unplugged-preview>
- Hugging Face: <https://huggingface.co/datasets/aiboxlab/aied-unplugged-preview>

To unpack one by hand, put `metadata/` and `schema/` under a single directory, then
name that directory with `root=`, with `AIED_UNPLUGGED_DATA`, or as
`competition-dataset/` in your working directory. `root=` wins over
`AIED_UNPLUGGED_DATA`, which wins over `download()`, which wins over
`competition-dataset/`.

### Loading a track

```python
from aied_unplugged import load_track, load_metadata, open_image

splits = load_track("essays")  # local release tree if present, else the Hub
splits.train, splits.validation, splits.test

train = load_track("math", "train")  # one split
image = open_image(train.iloc[0], "equation_image")
```

The local loaders return pandas DataFrames with the Parquet metadata and absolute
image paths in `<column>_path`.

```python
load_schema("math")  # the published taxonomy, fields and metrics
graded_ids("aes")  # the ids a submission must carry
question_counts()  # questions per graded answer sheet
sample_submission("aes")  # the published placeholder file
verify()  # re-check every SHA-256 in the release
```

---

## 3. Exploring

Needs `aied-unplugged[explore]` and a local copy of the dataset, from `download()`
or from a mirror you unpacked yourself.

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

Needs `aied-unplugged[models]`. Two strategies, either one enough for a valid
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
reply into the submission schema. An unparseable reply falls back to a safe
default, so a run always ends with a gradeable file.

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
from aied_unplugged import build, graded_ids, question_counts, validate, write

frame = build("answer-sheet", [{"sheet_id": "s1", "answers": {1: "A", 2: "Blank"}}])
report = validate(
    frame,
    "answer-sheet",
    expected_ids=graded_ids("answer-sheet"),
    expected_questions=question_counts(),
)
if not report.valid:
    print(report)
write("answer-sheet", frame, "submission.csv")
```

`validate` checks the exact column set, the identifiers, the competence scale, the
diagnostic labels, and the JSON encoding of the answer-sheet column, including
integer question numbers. With `expected_questions`, it also checks the question
count on each sheet. `build` takes answers as a list of records or as a
`{question_number: label}` mapping and encodes both to the same JSON.

---

## Development

```sh
uv run --with-editable . --with pytest pytest -q
uv run ruff check .
```

Tests needing the release tree skip when `competition-dataset/` is absent.

MIT licensed. The dataset itself is CC BY 4.0 and is published separately.

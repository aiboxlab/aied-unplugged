from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .data import load_metadata, open_image
from .submission import build
from .tracks import (
    ANSWER_VALUES,
    COMPETENCES,
    DIAGNOSTIC_LABELS,
    SCORE_VALUES,
    Track,
    get_track,
)

PROMPTS = {
    "aes": (
        "This is a handwritten Brazilian ENEM essay. Score it on the five ENEM "
        "competences. Each score is one of 0, 40, 80, 120, 160, 200. "
        'Answer with JSON only: {"competence_1": 0, "competence_2": 0, '
        '"competence_3": 0, "competence_4": 0, "competence_5": 0}'
    ),
    "math-diagnostic": (
        "The first image is a maths question, the second is a student's handwritten "
        "working. Name what happened in the response, using exactly one of these "
        "labels:\n{labels}\nAnswer with the label and nothing else."
    ),
    "answer-sheet": (
        "This is a scanned multiple-choice answer sheet with {n} questions, possibly "
        "rotated or skewed. Report what the student marked for every question, using "
        "A, B, C, D, Blank (nothing marked), Multiple (more than one) or Other. "
        'Answer with JSON only: [{{"question_number": 1, "label": "A"}}, ...]'
    ),
}


def majority_baseline(
    track: str | Track, split: str = "test", root: str | Path | None = None
) -> pd.DataFrame:
    resolved = get_track(track)
    train = load_metadata(resolved, root, "train")
    target = load_metadata(resolved, root, split)

    if resolved.id == "aes":
        rows = {c: int(train[c].mode().iloc[0]) for c in COMPETENCES}
        return build(resolved, target[[resolved.id_column]].assign(**rows))

    if resolved.id == "math-diagnostic":
        label = train.diagnostic.mode().iloc[0]
        return build(resolved, target[[resolved.id_column]].assign(diagnostic=label))

    cells = [entry["label"] for answers in train.answers for entry in answers]
    label = pd.Series(cells).mode().iloc[0]
    answers = [
        [{"question_number": i, "label": label} for i in range(1, int(n) + 1)]
        for n in target.num_questions
    ]
    return build(resolved, target[[resolved.id_column]].assign(answers=answers))


def _snap(value: float) -> int:
    return min(SCORE_VALUES, key=lambda v: abs(v - value))


def _parse_essay(text: str) -> dict[str, int]:
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    try:
        payload = json.loads(re.search(r"\{.*\}", text, re.S).group())
        return {c: _snap(float(payload[c])) for c in COMPETENCES}
    except Exception:
        scores = (numbers + [80] * 5)[:5]
        return {c: _snap(s) for c, s in zip(COMPETENCES, scores, strict=False)}


def _parse_diagnostic(text: str) -> str:
    lowered = text.lower()
    hits = [label for label in DIAGNOSTIC_LABELS if label.lower() in lowered]
    return max(hits, key=len) if hits else "Correct answer"


def _parse_answers(text: str, n: int) -> list[dict]:
    fallback = [{"question_number": i, "label": "Blank"} for i in range(1, n + 1)]
    try:
        payload = json.loads(re.search(r"\[.*\]", text, re.S).group())
    except Exception:
        return fallback
    answers = {}
    for entry in payload:
        try:
            number = int(entry["question_number"])
            label = str(entry["label"])
        except Exception:
            continue
        if 1 <= number <= n and label in ANSWER_VALUES:
            answers[number] = label
    return [{"question_number": i, "label": answers.get(i, "Blank")} for i in range(1, n + 1)]


def predict(
    track: str | Track,
    model: str,
    split: str = "test",
    root: str | Path | None = None,
    limit: int | None = None,
    max_new_tokens: int = 256,
    **pipeline_kwargs,
) -> pd.DataFrame:
    from transformers import pipeline

    resolved = get_track(track)
    frame = load_metadata(resolved, root, split)
    if limit:
        frame = frame.head(limit)

    generator = pipeline("image-text-to-text", model=model, **pipeline_kwargs)
    rows = []
    for row in frame.itertuples():
        images = [open_image(row._asdict(), c, root) for c in resolved.image_columns]
        prompt = PROMPTS[resolved.id].format(
            labels="\n".join(f"- {label}" for label in DIAGNOSTIC_LABELS),
            n=getattr(row, "num_questions", 0),
        )
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        output = generator(
            text=[{"role": "user", "content": content}], max_new_tokens=max_new_tokens
        )
        text = _text_of(output)

        identifier = getattr(row, resolved.id_column)
        if resolved.id == "aes":
            rows.append({resolved.id_column: identifier, **_parse_essay(text)})
        elif resolved.id == "math-diagnostic":
            rows.append({resolved.id_column: identifier, "diagnostic": _parse_diagnostic(text)})
        else:
            rows.append(
                {
                    resolved.id_column: identifier,
                    "answers": _parse_answers(text, int(row.num_questions)),
                }
            )
    return build(resolved, rows)


def _text_of(output) -> str:
    while isinstance(output, list) and output:
        output = output[0]
    if isinstance(output, dict):
        generated = output.get("generated_text", output)
        if isinstance(generated, list) and generated:
            last = generated[-1]
            return last.get("content", "") if isinstance(last, dict) else str(last)
        return str(generated)
    return str(output)


def finetune(
    model: str = "google/vit-base-patch16-224-in21k",
    root: str | Path | None = None,
    output_dir: str | Path = "runs/math-vit",
    epochs: float = 3.0,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
):
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Trainer,
        TrainingArguments,
    )

    from .graders import metrics as metric_module

    track = get_track("math-diagnostic")
    train = load_metadata(track, root, "train")
    validation = load_metadata(track, root, "validation")
    labels = sorted(set(train.diagnostic))
    index = {label: i for i, label in enumerate(labels)}

    processor = AutoImageProcessor.from_pretrained(model)
    net = AutoModelForImageClassification.from_pretrained(
        model,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=index,
        ignore_mismatched_sizes=True,
    )

    class Images(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, position):
            row = self.frame.iloc[position]
            image = open_image(row, "equation_image", root).convert("RGB")
            item = processor(image, return_tensors="pt")
            return {
                "pixel_values": item["pixel_values"][0],
                "labels": index.get(row.diagnostic, 0),
            }

    def compute(evaluation):
        predicted = evaluation.predictions.argmax(-1)
        true = [labels[i] for i in evaluation.label_ids]
        guess = [labels[i] for i in predicted]
        present = [label for label in labels if label in set(true)]
        return {
            "macro_f1": metric_module.macro_f1(true, guess, present),
            "accuracy": metric_module.accuracy(true, guess),
        }

    trainer = Trainer(
        model=net,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=25,
            remove_unused_columns=False,
            report_to=[],
        ),
        train_dataset=Images(train),
        eval_dataset=Images(validation),
        compute_metrics=compute,
    )
    trainer.train()
    return trainer


def predict_with_classifier(
    trainer, split: str = "test", root: str | Path | None = None
) -> pd.DataFrame:
    import numpy as np

    track = get_track("math-diagnostic")
    frame = load_metadata(track, root, split)
    labels = [trainer.model.config.id2label[i] for i in range(trainer.model.config.num_labels)]

    dataset = trainer.eval_dataset.__class__(frame.assign(diagnostic=labels[0]))
    logits = trainer.predict(dataset).predictions
    guessed = [labels[i] for i in np.asarray(logits).argmax(-1)]
    return build(
        track, pd.DataFrame({track.id_column: frame[track.id_column], "diagnostic": guessed})
    )

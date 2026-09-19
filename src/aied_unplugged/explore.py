from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_metadata, open_image
from .tracks import ANSWER_VALUES, COMPETENCES, DIAGNOSTIC_LABELS, Track, get_track


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting needs matplotlib: pip install 'aied-unplugged[explore]'"
        ) from exc
    return plt


def summary(root: str | Path | None = None) -> pd.DataFrame:
    rows = []
    for name in ("essays", "math", "answer_sheets"):
        track = get_track(name)
        frame = load_metadata(track, root)
        counts = frame.split.value_counts()
        rows.append(
            {
                "track": track.id,
                "config": track.config,
                "rows": len(frame),
                "train": int(counts.get("train", 0)),
                "validation": int(counts.get("validation", 0)),
                "test": int(counts.get("test", 0)),
            }
        )
    return pd.DataFrame(rows)


def label_distribution(
    track: str | Track, split: str | None = None, root: str | Path | None = None
) -> pd.DataFrame:
    resolved = get_track(track)
    frame = load_metadata(resolved, root, split)
    frame = frame[frame.split != "test"] if split is None else frame

    if resolved.id == "math-diagnostic":
        counts = frame.diagnostic.value_counts()
        order = [label for label in DIAGNOSTIC_LABELS if label in counts.index]
        return pd.DataFrame(
            {"label": order, "count": [int(counts[label]) for label in order]}
        ).assign(share=lambda d: d["count"] / d["count"].sum())

    if resolved.id == "answer-sheet":
        cells = [entry["label"] for answers in frame.answers for entry in answers]
        counts = pd.Series(cells).value_counts()
        order = [value for value in ANSWER_VALUES if value in counts.index]
        return pd.DataFrame(
            {"label": order, "count": [int(counts[value]) for value in order]}
        ).assign(share=lambda d: d["count"] / d["count"].sum())

    long = frame.melt(
        id_vars=["essay_id"],
        value_vars=list(COMPETENCES),
        var_name="competence",
        value_name="score",
    ).dropna(subset=["score"])
    table = long.groupby(["competence", "score"]).size().rename("count").reset_index()
    table["score"] = table["score"].astype(int)
    return table.sort_values(["competence", "score"], ignore_index=True)


def plot_label_distribution(
    track: str | Track, split: str | None = None, root: str | Path | None = None, ax=None
):
    plt = _pyplot()
    resolved = get_track(track)
    table = label_distribution(resolved, split, root)

    if resolved.id == "aes":
        pivot = table.pivot(index="score", columns="competence", values="count").fillna(0)
        ax = pivot.plot(kind="bar", ax=ax, figsize=(9, 4.5), width=0.82)
        ax.set_xlabel("score")
        ax.set_ylabel("essays")
        ax.legend(title=None, fontsize=8)
    else:
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.5))
        ax.barh(table["label"][::-1], table["count"][::-1], color="#4573a7")
        ax.set_xlabel("cells" if resolved.id == "answer-sheet" else "items")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax.set_title(f"{resolved.name} — {split or 'train + validation'}")
    ax.figure.tight_layout()
    return ax


def show_essay(item, root: str | Path | None = None, ax=None):
    plt = _pyplot()
    row = _row("essays", item, root)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(open_image(row, "image", root))
    ax.axis("off")
    scores = [row[c] for c in COMPETENCES]
    total = "withheld" if any(pd.isna(s) for s in scores) else str(int(sum(scores)))
    shown = " / ".join("–" if pd.isna(s) else str(int(s)) for s in scores)
    ax.set_title(f"{row['essay_id']}  ·  {shown}  ·  total {total}", fontsize=10)
    ax.figure.tight_layout()
    return ax


def show_math(item, root: str | Path | None = None, axes=None):
    plt = _pyplot()
    row = _row("math", item, root)
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[1, 1])
    for ax, column, caption in zip(
        axes, ("question_image", "equation_image"), ("question", "working"), strict=False
    ):
        ax.imshow(open_image(row, column, root))
        ax.axis("off")
        ax.set_title(caption, fontsize=9, loc="left")
    label = row["diagnostic"] if isinstance(row["diagnostic"], str) else "withheld"
    axes[0].figure.suptitle(f"{row['equation_id']}  ·  {label}", fontsize=10)
    axes[0].figure.tight_layout()
    return axes


def show_sheet(item, root: str | Path | None = None, ax=None):
    plt = _pyplot()
    row = _row("answer_sheets", item, root)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(open_image(row, "image", root))
    ax.axis("off")
    answers = row["answers"]
    marked = (
        "withheld"
        if answers is None or len(answers) == 0
        else ", ".join(f"{a['question_number']}{a['label']}" for a in list(answers)[:12])
    )
    header = f"{row['sheet_id']}  ·  {row['exam_id']}  ·  {row['num_questions']} questions"
    ax.set_title(f"{header}\n{marked}", fontsize=9)
    ax.figure.tight_layout()
    return ax


def render_math(item, root: str | Path | None = None, width: int = 900):
    from PIL import Image

    row = _row("math", item, root)
    images = [
        open_image(row, c, root).convert("RGB") for c in ("question_image", "equation_image")
    ]
    scaled = []
    for image in images:
        height = max(1, round(image.height * width / image.width))
        scaled.append(image.resize((width, height), Image.LANCZOS))
    canvas = Image.new("RGB", (width, sum(i.height for i in scaled) + 12), "white")
    offset = 0
    for image in scaled:
        canvas.paste(image, (0, offset))
        offset += image.height + 12
    return canvas


def _row(track: str, item, root):
    resolved = get_track(track)
    if isinstance(item, pd.Series):
        return item
    frame = load_metadata(resolved, root)
    if isinstance(item, int):
        return frame.iloc[item]
    match = frame[frame[resolved.id_column] == item]
    if match.empty:
        raise KeyError(f"{item!r} is not in the {resolved.config} track")
    return match.iloc[0]


def describe_taxonomy(track: str | Track, root: str | Path | None = None) -> pd.DataFrame:
    from .data import load_schema

    resolved = get_track(track)
    schema = load_schema(resolved, root)
    if resolved.id == "math-diagnostic":
        return pd.DataFrame(schema["taxonomy"])
    if resolved.id == "answer-sheet":
        meanings = schema["answer_meanings"]
        return pd.DataFrame({"label": list(meanings), "detail": list(meanings.values())})
    return pd.DataFrame(schema["rubric"])

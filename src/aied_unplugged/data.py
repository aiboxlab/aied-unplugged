from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from .tracks import Track, get_track

SPLITS = ("train", "validation", "test")
ENV_ROOT = "AIED_UNPLUGGED_DATA"
HF_REPO = "aiboxlab/aied-unplugged-preview"


class Splits(dict):
    def __getattr__(self, name: str) -> pd.DataFrame:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def find_release(root: str | Path | None = None) -> Path:
    candidates = [root, os.environ.get(ENV_ROOT), "competition-dataset", "."]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "metadata").is_dir() and (path / "schema").is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "no release tree found. Pass root=..., set "
        f"{ENV_ROOT}, or run from a directory holding competition-dataset/."
    )


def load_metadata(
    track: str | Track, root: str | Path | None = None, split: str | None = None
) -> pd.DataFrame:
    resolved = get_track(track)
    base = find_release(root)
    frame = pd.read_parquet(base / "metadata" / f"{resolved.config}.parquet")
    for column in resolved.image_columns:
        frame[f"{column}_path"] = [str(base / p) for p in frame[column]]
    if split is not None:
        frame = frame[frame.split == split].reset_index(drop=True)
    return frame


def load_track(
    track: str | Track,
    split: str | None = None,
    root: str | Path | None = None,
    source: str = "auto",
    repo: str = HF_REPO,
):
    resolved = get_track(track)
    if source not in {"auto", "local", "hf"}:
        raise ValueError("source must be 'auto', 'local' or 'hf'")

    if source in {"auto", "local"}:
        try:
            base = find_release(root)
        except FileNotFoundError:
            if source == "local":
                raise
        else:
            if split is not None:
                return load_metadata(resolved, base, split)
            return Splits({s: load_metadata(resolved, base, s) for s in SPLITS})

    from datasets import load_dataset

    dataset = load_dataset(repo, resolved.config)
    return dataset[split] if split is not None else dataset


def load_schema(name: str | Track, root: str | Path | None = None) -> dict:
    base = find_release(root)
    key = "dataset" if name == "dataset" else get_track(name).config
    return json.loads((base / "schema" / f"{key}.json").read_text(encoding="utf-8"))


def open_image(row, column: str = "image", root: str | Path | None = None):
    from PIL import Image

    value = row[f"{column}_path"] if f"{column}_path" in row else row[column]
    path = Path(value)
    if not path.is_absolute():
        path = find_release(root) / path
    return Image.open(path)


def sample_submission(track: str | Track, root: str | Path | None = None) -> pd.DataFrame:
    resolved = get_track(track)
    return pd.read_csv(find_release(root) / "samples" / resolved.sample_file)


def graded_ids(track: str | Track, root: str | Path | None = None) -> list[str]:
    resolved = get_track(track)
    frame = load_metadata(resolved, root, "test")
    return list(frame[resolved.id_column])


def verify(track: str | Track | None = None, root: str | Path | None = None) -> pd.DataFrame:
    base = find_release(root)
    tracks = (
        [get_track(track)]
        if track is not None
        else [get_track(t) for t in ("essays", "math", "answer_sheets")]
    )
    rows = []
    for resolved in tracks:
        frame = load_metadata(resolved, base)
        for column in resolved.image_columns:
            digest_column = f"{column}_sha256" if column != "image" else "image_sha256"
            for path, expected in zip(
                frame[f"{column}_path"], frame[digest_column], strict=False
            ):
                file = Path(path)
                actual = (
                    hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
                )
                if actual != expected:
                    rows.append(
                        {
                            "track": resolved.id,
                            "column": column,
                            "path": path,
                            "problem": "missing" if actual is None else "checksum",
                        }
                    )
    return pd.DataFrame(rows, columns=["track", "column", "path", "problem"])

#!/usr/bin/env python

import os
from pathlib import Path

import pandas as pd

from ingest_flatten import stream_and_flatten
from aggregate_impute import aggregate_and_impute
from featurize import build_features


def resolve_data_path(path: str, *, anchor_file: str | None = None) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    if anchor_file:
        anchor = Path(anchor_file).resolve()
        project_root = None
        for parent in [anchor.parent, *anchor.parents]:
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
        if project_root is None:
            project_root = anchor.parent
    else:
        project_root = Path.cwd()

    search_roots = [
        project_root,
        project_root / "src",
        project_root / "src" / "data",
        project_root / "src" / "data" / "daily_dataset",
        project_root / "data",
        project_root / "data" / "daily_dataset",
    ]

    for root in search_roots:
        resolved = root / path
        if resolved.exists():
            return str(resolved)

    return path


def prepare_modelready_if_missing(
    modelready_path: str,
    split: str,
    batch_size: int,
    flat_dir: str,
    daily_path: str,
    max_records: int | None = None,
) -> None:
    modelready_path = resolve_data_path(modelready_path)
    flat_dir = resolve_data_path(flat_dir)
    daily_path = resolve_data_path(daily_path)

    if os.path.exists(modelready_path):
        return

    os.makedirs(flat_dir, exist_ok=True)
    stream_and_flatten(
        split=split,
        batch_size=batch_size,
        output_dir=flat_dir,
        max_records=max_records,
    )
    aggregate_and_impute(input_dir=flat_dir, output_path=daily_path)
    build_features(input_path=daily_path, output_path=modelready_path)


def load_modelready(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"])
    return df

"""
io_utils — load raw data, merge, save outputs.

Runde-1 module.  Provides the minimum interface required by
main_build_datasets.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as cfg


def ensure_output_dirs() -> None:
    """Create output directory tree if it does not exist."""
    for d in (cfg.OUTPUT_DIR, cfg.OUTPUT_DATASETS_DIR,
              cfg.OUTPUT_AUDIT_DIR, cfg.OUTPUT_METADATA_DIR,
              cfg.OUTPUT_ORANGE_EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[io] Output dirs ready under {cfg.OUTPUT_DIR}")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and items.csv from ``cfg.DATA_DIR``.

    Returns
    -------
    df_train_raw, df_items
    """
    if not cfg.TRAIN_CSV.exists():
        raise FileNotFoundError(f"[io] Train file not found: {cfg.TRAIN_CSV}")
    if not cfg.ITEMS_CSV.exists():
        raise FileNotFoundError(f"[io] Items file not found: {cfg.ITEMS_CSV}")

    df_train = pd.read_csv(cfg.TRAIN_CSV, sep="|")
    df_items = pd.read_csv(cfg.ITEMS_CSV, sep="|")
    print(
        f"[io] Loaded train: {len(df_train):,} rows, items: {len(df_items):,} rows")
    return df_train, df_items


def merge_train_items(
    df_train: pd.DataFrame,
    df_items: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join train onto items by ``pid``."""
    df = df_train.merge(df_items, on="pid", how="left")
    print(f"[io] Merged: {len(df):,} rows")
    return df


# ── Save helpers ─────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Save DataFrame as parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[save] {path.name}  ({len(df):,} rows)")


def save_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Save DataFrame as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save] {path.name}  ({len(df):,} rows)")


def save_text_report(text: str, path: Path | str) -> None:
    """Save plain-text report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[save] {path.name}")

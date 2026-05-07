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
    expected_train_cols = [
        "lineID", "day", "pid", "adFlag", "availability",
        "competitorPrice", "click", "basket", "order", "price", "revenue",
    ]
    expected_items_cols = [
        "pid", "manufacturer", "group", "content", "unit", "pharmForm",
        "genericProduct", "salesIndex", "category", "campaignIndex", "rrp",
    ]

    if not cfg.TRAIN_CSV.exists() or not cfg.ITEMS_CSV.exists():
        msg = (
            "\n[io] Raw data not found.\n"
            f"  Expected files:\n"
            f"    - {cfg.TRAIN_CSV}\n"
            f"    - {cfg.ITEMS_CSV}\n"
            "\n"
            "  If you do not have the full dataset, run the sample pipeline:\n"
            "      python scripts/run_pipeline.py --sample\n"
            "      python scripts/run_pipeline.py            (alias for --sample)\n"
        )
        raise FileNotFoundError(msg)

    df_train = pd.read_csv(cfg.TRAIN_CSV, sep="|")
    df_items = pd.read_csv(cfg.ITEMS_CSV, sep="|")

    missing_train = [c for c in expected_train_cols if c not in df_train.columns]
    missing_items = [c for c in expected_items_cols if c not in df_items.columns]
    if missing_train or missing_items:
        raise ValueError(
            "[io] Raw data schema mismatch.\n"
            f"  Train file: {cfg.TRAIN_CSV}\n"
            f"    missing columns: {missing_train}\n"
            f"  Items file: {cfg.ITEMS_CSV}\n"
            f"    missing columns: {missing_items}\n"
            "  Expected separator is '|'."
        )

    print(
        f"[io] Loaded train: {len(df_train):,} rows, items: {len(df_items):,} rows"
    )
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

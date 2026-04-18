"""
split — chronological train / test / validation split.

Runde-1 module.  Provides the minimum interface required by
main_build_datasets.py.
"""

from __future__ import annotations

import pandas as pd

import config as cfg


def run_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* chronologically by ``day``.

    Train       : days  26 – 70   (cfg.TRAIN_DAY_START .. TRAIN_DAY_END)
    Test        : days  71 – 81   (cfg.TEST_DAY_START  .. TEST_DAY_END)
    Validation  : days  82 – 92   (cfg.VAL_DAY_START   .. VAL_DAY_END)

    Rows outside these ranges are silently dropped (e.g. days 1–25).
    """
    print(f"\n{'='*60}")
    print("CHRONOLOGICAL SPLIT")
    print(f"{'='*60}")

    if "day" not in df.columns:
        raise ValueError("[split] Column 'day' missing")

    df_train = df.loc[
        (df["day"] >= cfg.TRAIN_DAY_START) & (df["day"] <= cfg.TRAIN_DAY_END)
    ].copy()

    df_test = df.loc[
        (df["day"] >= cfg.TEST_DAY_START) & (df["day"] <= cfg.TEST_DAY_END)
    ].copy()

    df_validation = df.loc[
        (df["day"] >= cfg.VAL_DAY_START) & (df["day"] <= cfg.VAL_DAY_END)
    ].copy()

    n_dropped = len(df) - len(df_train) - len(df_validation) - len(df_test)

    print(
        f"  Train : days {cfg.TRAIN_DAY_START}–{cfg.TRAIN_DAY_END}  →  {len(df_train):>9,} rows")
    print(
        f"  Test  : days {cfg.TEST_DAY_START}–{cfg.TEST_DAY_END}  →  {len(df_test):>9,} rows")
    print(
        f"  Validation : days {cfg.VAL_DAY_START}–{cfg.VAL_DAY_END}  →  {len(df_validation):>9,} rows")
    if n_dropped > 0:
        print(f"  Dropped (outside ranges): {n_dropped:,} rows")
    print()

    return df_train, df_validation, df_test

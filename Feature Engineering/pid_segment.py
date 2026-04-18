"""
pid_segment — Head / Mid / Tail segmentation based on event frequency.

Computed on the full Train set and mapped onto all splits.
Must be available BEFORE sampling (used as stratification variable).
Becomes a model feature only when promoted in the conditional stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Fit ──────────────────────────────────────────────────────────────────────

def fit_pid_segment(df_train: pd.DataFrame) -> dict[int, str]:
    """Compute pid → segment mapping from the training set.

    Segments (based on cumulative event share, deterministic ranking):
        Head : top 10 % of PIDs by event count
        Mid  : next 40 % (10 %–50 %)
        Tail : bottom 50 %

    Parameters
    ----------
    df_train : DataFrame with at least column ``pid``.

    Returns
    -------
    dict  mapping pid (int) → segment label (str).
    """
    if "pid" not in df_train.columns:
        raise ValueError("[pid_segment] Column 'pid' missing from df_train")
    if df_train.empty:
        raise ValueError("[pid_segment] df_train is empty")

    counts = (
        df_train.groupby("pid")
        .size()
        .reset_index(name="n_events")
        .sort_values(["n_events", "pid"], ascending=[False, True])  # deterministic
        .reset_index(drop=True)
    )

    n_pids = len(counts)
    n_head = max(1, int(np.ceil(n_pids * 0.10)))
    n_mid_end = max(n_head + 1, int(np.ceil(n_pids * 0.50)))

    counts["segment"] = "Tail"
    counts.loc[counts.index < n_head, "segment"] = "Head"
    counts.loc[
        (counts.index >= n_head) & (counts.index < n_mid_end), "segment"
    ] = "Mid"

    mapping: dict[int, str] = dict(zip(counts["pid"], counts["segment"]))

    # Summary
    for seg in ("Head", "Mid", "Tail"):
        seg_pids = [p for p, s in mapping.items() if s == seg]
        seg_events = counts.loc[counts["segment"] == seg, "n_events"].sum()
        print(
            f"[pid_segment] {seg:4s}: {len(seg_pids):,} PIDs, "
            f"{seg_events:,} events"
        )

    return mapping


# ── Apply ────────────────────────────────────────────────────────────────────

def apply_pid_segment(
    df: pd.DataFrame,
    pid_segment_map: dict[int, str],
) -> pd.DataFrame:
    """Map ``pid_segment`` onto *df*.  Unknown PIDs → ``'Tail'``."""
    df["pid_segment"] = df["pid"].map(pid_segment_map).fillna("Tail")
    return df


# ── Persist ──────────────────────────────────────────────────────────────────

def save_pid_segment_map(
    pid_segment_map: dict[int, str],
    path: Path | str,
) -> None:
    """Save mapping as CSV (pid, segment)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        list(pid_segment_map.items()), columns=["pid", "pid_segment"]
    )
    df.to_csv(path, index=False)
    print(f"[pid_segment] Saved mapping ({len(df):,} PIDs) → {path.name}")


def load_pid_segment_map(path: Path | str) -> dict[int, str]:
    """Load mapping from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[pid_segment] File not found: {path}")
    df = pd.read_csv(path)
    mapping = dict(zip(df["pid"].astype(int), df["pid_segment"].astype(str)))
    print(f"[pid_segment] Loaded mapping ({len(mapping):,} PIDs) from {path.name}")
    return mapping

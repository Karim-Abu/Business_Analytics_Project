"""
Conditional feature engineering — features requiring train-fit or time-aware encoding.

All functions here guard against future-leakage by design:
- Cumulative features use only days >= TRAIN_DAY_START up to day-1.
- OOF encodings use expanding-window forward folds (no random KFold).
- Test/Validation receive encodings fit on the entire Train set.
- The first OOF block uses a conservative cold-start fallback,
  never estimated from later Train blocks.

Module has no side effects beyond adding columns to DataFrames.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config as cfg


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    """Raise if any *cols* are missing from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Missing columns: {missing}")


def _week_block(day: pd.Series, start: int = cfg.TRAIN_DAY_START) -> pd.Series:
    """Week-block index relative to *start*: block 1 = days start..start+6."""
    return ((day - start) // 7) + 1


# ═══════════════════════════════════════════════════════════════════════════
# 1. Cumulative features (ab Tag 26 bis day-1)
# ═══════════════════════════════════════════════════════════════════════════

def compute_cumulative_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add per-pid cumulative event counts from ``TRAIN_DAY_START`` to ``day-1``.

    New columns
    -----------
    pid_total_events — number of rows for this pid in [TRAIN_DAY_START, day-1]
    click_time       — cumulative sum of ``click``
    basket_time      — cumulative sum of ``basket``
    order_time       — cumulative sum of ``order``
    num_pid_order    — alias of ``order_time`` (dropped for REG downstream)

    All splits are concatenated so that test/validation see full prior history,
    then split back by tag.  Only rows with ``day >= TRAIN_DAY_START``
    contribute to the cumulative sums.
    """
    needed = ["pid", "day", "click", "basket", "order"]
    for label, d in [("train", df_train), ("test", df_test), ("validation", df_val)]:
        _require_columns(d, needed, f"cumulative/{label}")

    # Concat with MultiIndex so we can split back safely
    df = pd.concat(
        {"train": df_train, "test": df_test, "validation": df_val},
        names=["_split", "_idx"],
    )
    df = df.sort_values(["pid", "day"])

    # Only rows from TRAIN_DAY_START onward contribute to history
    valid = (df["day"] >= cfg.TRAIN_DAY_START).astype(int)

    # pid_total_events: running count of valid rows per pid, excluding current
    df["pid_total_events"] = (
        valid.groupby(df["pid"]).cumsum() - valid
    ).clip(lower=0).astype(int)

    # Event-level cumulative sums: cumsum minus current row = history-only
    for src, dst in [("click", "click_time"),
                     ("basket", "basket_time"),
                     ("order", "order_time")]:
        masked = df[src].fillna(0).astype(int) * valid
        df[dst] = (
            masked.groupby(df["pid"]).cumsum() - masked
        ).clip(lower=0).astype(int)

    df["num_pid_order"] = df["order_time"]

    # Split back — preserves original indices
    out_tr = df.xs("train", level="_split")
    out_te = df.xs("test", level="_split")
    out_va = df.xs("validation", level="_split")

    print("[conditional/cumulative] 5 features added to all splits")
    return out_tr, out_va, out_te


# ═══════════════════════════════════════════════════════════════════════════
# 2. Train-global aggregations (fit on Train, apply to all)
# ═══════════════════════════════════════════════════════════════════════════

def fit_global_aggregations(
    df_train: pd.DataFrame,
) -> dict[str, dict]:
    """Fit simple mean(order) mappings on the full Train set.

    Returns a dict with one entry per target column:
        ``'group12_order'``: {group12_value: mean_order, …}
        ``'group34_order'``: {group34_value: mean_order, …}
        ``'week_order'``   : {day_7_value:   mean_order, …}

    Leakage note
    -------------
    These are full-train-fit target encodings.  On the training set itself
    this introduces self-leakage.  For final model selection, prefer the
    time-aware OOF variants (``pid_prob``, ``day_7_likelihood``, etc.).
    These mappings are intended for fast prototyping / auxiliary signal.
    """
    _require_columns(df_train, ["group12", "group34", "day_7", "order"],
                     "fit_global_aggregations")

    global_mean = float(df_train["order"].mean())

    mappings: dict[str, dict] = {}

    for col, name in [("group12", "group12_order"),
                      ("group34", "group34_order"),
                      ("day_7",   "week_order")]:
        means = df_train.groupby(col)["order"].mean()
        mappings[name] = {
            "group_col": col,
            "group_means": means.to_dict(),
            "global_mean": global_mean,
        }
        print(f"[conditional/agg] {name}: {len(means)} groups, "
              f"global_mean={global_mean:.4f}")

    return mappings


def apply_global_aggregations(
    df: pd.DataFrame,
    mappings: dict[str, dict],
) -> pd.DataFrame:
    """Map pre-fitted aggregations onto *df*.

    Unknown groups receive ``global_mean`` (conservative fallback).
    """
    for name, m in mappings.items():
        col = m["group_col"]
        if col not in df.columns:
            raise ValueError(
                f"[apply_global_aggregations] Column '{col}' missing"
            )
        df[name] = (
            df[col]
            .map(m["group_means"])
            .fillna(m["global_mean"])
        )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Time-aware forward OOF encoding
# ═══════════════════════════════════════════════════════════════════════════

def _time_aware_forward_oof(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    cold_start: float,
    history_mask: pd.Series | None = None,
) -> tuple[pd.Series, dict]:
    """Expanding-window forward OOF encoding on the **training** set.

    For each week-block *b* (relative to ``TRAIN_DAY_START``):
    - history  = all rows from blocks < *b*
    - encoding = ``mean(target_col)`` per ``group_col`` in history
    - unseen groups → global mean of history (or ``cold_start`` if first block)

    If ``history_mask`` is provided, only rows where the mask is True
    contribute to the history aggregation (used for REG-only features
    where only ``order == 1`` rows carry valid target values).

    The **first block** receives ``cold_start`` for every row —
    no future information, no within-block self-leakage.

    Returns
    -------
    encoded : pd.Series  aligned to *df*.index
    fold_info : dict      metadata about blocks and cold-start used
    """
    wb = _week_block(df["day"])
    blocks = sorted(wb.unique())

    encoded = pd.Series(np.nan, index=df.index, dtype=float)
    fold_details: list[dict] = []

    for i, block in enumerate(blocks):
        block_idx = wb == block

        if i == 0:
            # ── First block: cold-start (no admissible history) ──────────
            encoded.loc[block_idx] = cold_start
            fold_details.append({
                "block": int(block),
                "type": "cold_start",
                "value": cold_start,
                "n_rows": int(block_idx.sum()),
            })
        else:
            # ── Expanding window: all prior blocks ───────────────────────
            hist_idx = wb.isin(blocks[:i])
            if history_mask is not None:
                hist_idx = hist_idx & history_mask

            history = df.loc[hist_idx]

            if history.empty or history[target_col].isna().all():
                encoded.loc[block_idx] = cold_start
                fold_details.append({
                    "block": int(block),
                    "type": "cold_start_empty_history",
                    "value": cold_start,
                    "n_rows": int(block_idx.sum()),
                })
                continue

            group_means = history.groupby(group_col)[target_col].mean()
            global_mean = float(history[target_col].mean())

            block_groups = df.loc[block_idx, group_col]
            encoded.loc[block_idx] = (
                block_groups.map(group_means).fillna(global_mean).values
            )

            fold_details.append({
                "block": int(block),
                "type": "expanding",
                "n_history_rows": len(history),
                "n_groups": len(group_means),
                "global_mean": round(global_mean, 6),
                "n_rows": int(block_idx.sum()),
            })

    fold_info = {
        "group_col": group_col,
        "target_col": target_col,
        "cold_start": cold_start,
        "n_blocks": len(blocks),
        "blocks": fold_details,
    }
    return encoded, fold_info


def _fit_full_train_encoding(
    df_train: pd.DataFrame,
    group_col: str,
    target_col: str,
    global_fallback: float,
    history_mask: pd.Series | None = None,
) -> dict:
    """Fit a simple group-mean encoding on the **entire** Train set.

    Used to produce test/validation encodings (no OOF needed there).

    Returns
    -------
    dict with ``group_means`` (dict) and ``global_mean`` (float).
    """
    source = df_train if history_mask is None else df_train.loc[history_mask]

    if source.empty or source[target_col].isna().all():
        return {"group_means": {}, "global_mean": global_fallback}

    group_means = source.groupby(group_col)[target_col].mean().to_dict()
    global_mean = float(source[target_col].mean())
    return {"group_means": group_means, "global_mean": global_mean}


def _apply_encoding(
    df: pd.DataFrame,
    group_col: str,
    col_name: str,
    encoding: dict,
) -> pd.DataFrame:
    """Map a pre-fitted encoding onto *df*, adding column ``col_name``."""
    df[col_name] = (
        df[group_col]
        .map(encoding["group_means"])
        .fillna(encoding["global_mean"])
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. Orchestration
# ═══════════════════════════════════════════════════════════════════════════

# OOF feature specifications:
#   (feature_name, group_col, target_col, cold_start, history_mask_fn, stages)
#   history_mask_fn: None or callable(df) -> pd.Series[bool]
#   stages: which stages may use this feature ("CLS", "REG", "both")

_OOF_SPECS: list[tuple[str, str, str, float, Any, str]] = [
    ("pid_prob",                "pid",          "order",
     cfg.OOF_COLD_START_PROB, None,                      "both"),
    ("availability_likelihood", "availability", "order",
     cfg.OOF_COLD_START_PROB, None,                      "both"),
    ("day_7_likelihood",        "day_7",        "order",
     cfg.OOF_COLD_START_PROB, None,                      "CLS"),
    ("day_7_qty_mean_oof",      "day_7",        "quantity",
     cfg.OOF_COLD_START_QTY, lambda d: d["order"] == 1, "REG"),
]


def run_all_conditional_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Apply all conditional features.  Train-fitted artefacts applied to all splits.

    Returns
    -------
    df_train, df_val, df_test : with new conditional columns
    metadata : dict with fitted artefacts for reproducibility / export
    """
    print(f"\n{'='*60}")
    print("CONDITIONAL FEATURE ENGINEERING")
    print(f"{'='*60}")

    metadata: dict[str, Any] = {}

    # ── 1. Cumulative features ───────────────────────────────────────────
    df_train, df_val, df_test = compute_cumulative_features(
        df_train, df_val, df_test,
    )

    # ── 2. Train-global aggregations ─────────────────────────────────────
    agg_maps = fit_global_aggregations(df_train)
    metadata["global_aggregation_maps"] = agg_maps
    for d in (df_train, df_val, df_test):
        apply_global_aggregations(d, agg_maps)

    # ── 3. Time-aware OOF on training set ────────────────────────────────
    oof_fold_info: dict[str, dict] = {}
    full_train_encodings: dict[str, dict] = {}

    for feat_name, group_col, target_col, cold, mask_fn, _stages in _OOF_SPECS:
        _require_columns(df_train, [group_col, target_col],
                         f"oof/{feat_name}")

        hist_mask = mask_fn(df_train) if mask_fn is not None else None

        # OOF values for training rows
        train_encoded, fold_info = _time_aware_forward_oof(
            df_train, group_col, target_col, cold,
            history_mask=hist_mask,
        )
        df_train[feat_name] = train_encoded
        oof_fold_info[feat_name] = fold_info

        # Full-train encoding for test/validation
        enc = _fit_full_train_encoding(
            df_train, group_col, target_col, cold,
            history_mask=hist_mask,
        )
        full_train_encodings[feat_name] = enc

        for d in (df_test, df_val):
            _apply_encoding(d, group_col, feat_name, enc)

        n_nan_train = int(df_train[feat_name].isna().sum())
        print(f"[conditional/oof] {feat_name}: "
              f"{fold_info['n_blocks']} blocks, "
              f"train NaN={n_nan_train}")

    metadata["oof_fold_info"] = oof_fold_info
    metadata["full_train_encodings"] = full_train_encodings

    # ── Summary ──────────────────────────────────────────────────────────
    cond_cols = [
        "pid_total_events", "click_time", "basket_time", "order_time",
        "num_pid_order", "group12_order", "group34_order", "week_order",
        "pid_prob", "availability_likelihood", "day_7_likelihood",
        "day_7_qty_mean_oof",
    ]
    present = [c for c in cond_cols if c in df_train.columns]
    print(f"[conditional] Done — {len(present)} conditional columns on Train")
    print(f"[conditional] pid_segment already present: "
          f"{'pid_segment' in df_train.columns}\n")

    return df_train, df_val, df_test, metadata

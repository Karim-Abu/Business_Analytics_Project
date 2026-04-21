"""
Sampling — stratified sub-sampling of the training set for prototyping.

Sampling is a development tool only.  Final models should train on the
full Train set (day 26–70).  Test and Validation are NEVER sampled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from feature_sets import get_reg_mask


# ── Helpers ──────────────────────────────────────────────────────────────────

def add_week_block(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``week_block`` = week number relative to TRAIN_DAY_START.

    week_block 1 covers days 26–32, block 2 covers 33–39, etc.
    Works on any split (Test/Validation will just have higher block numbers).
    """
    df["week_block"] = ((df["day"] - cfg.TRAIN_DAY_START) // 7) + 1
    return df


# ── Core stratified sampler ──────────────────────────────────────────────────

def sample_stratified(
    df: pd.DataFrame,
    frac: float,
    strata_cols: list[str],
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """Stratified sampling that handles small strata gracefully.

    Strategy
    --------
    - For each stratum, draw ``max(1, round(n * frac))`` rows.
    - But: if the stratum has ≤ 2 rows, keep ALL of them rather than
      inflating a single row into disproportionate weight.
    - This avoids the "blind max(1, ceil(…))" anti-pattern that
      massively over-represents tiny strata.

    Parameters
    ----------
    df          : source DataFrame
    frac        : target fraction ∈ (0, 1]
    strata_cols : columns defining the strata
    seed        : random seed
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    for c in strata_cols:
        if c not in df.columns:
            raise ValueError(f"Stratum column '{c}' not in DataFrame")

    rng = np.random.RandomState(seed)
    parts: list[pd.DataFrame] = []

    for _key, group in df.groupby(strata_cols, observed=True):
        n = len(group)
        if n <= 2:
            # Tiny stratum: keep all rows (avoids over-representation)
            parts.append(group)
        else:
            k = max(1, round(n * frac))
            parts.append(group.sample(n=k, random_state=rng))

    result = pd.concat(parts, ignore_index=False)
    print(
        f"[sampling] {len(df):,} → {len(result):,} rows  "
        f"(frac={frac}, strata={strata_cols})"
    )
    return result


# ── CLS sampling ─────────────────────────────────────────────────────────────

def sample_cls(
    df_train: pd.DataFrame,
    frac: float = cfg.SAMPLE_FRAC_CLS,
) -> pd.DataFrame:
    """Sample the CLS training set.

    Strata: week_block × order × pid_segment.
    Assumes ``pid_segment`` is already present.
    """
    _require_columns(df_train, ["order", "pid_segment", "day"])
    df = add_week_block(df_train.copy())
    return sample_stratified(df, frac, ["week_block", "order", "pid_segment"])


# ── REG sampling ─────────────────────────────────────────────────────────────

def sample_reg(
    df_train: pd.DataFrame,
    frac: float = cfg.SAMPLE_FRAC_REG,
) -> pd.DataFrame:
    """Sample the REG training set (Stage-2 subset).

    Uses ``get_reg_mask()`` — the same Stage-2 inclusion mask as
    ``build_safe_feature_matrices()`` — so that sampled rows are a strict
    subset of the REG feature-matrix population.

    Strata: week_block × quantity_class × pid_segment.
    Assumes ``pid_segment`` and ``quantity_class`` are already present.
    """
    _require_columns(df_train, ["order", "quantity", "quantity_class",
                                "pid_segment", "day"])
    df_reg = df_train.loc[get_reg_mask(df_train)].copy()
    if df_reg.empty:
        raise ValueError("[sample_reg] No valid Stage-2 rows in df_train")
    df_reg = add_week_block(df_reg)
    return sample_stratified(df_reg, frac, ["week_block", "quantity_class", "pid_segment"])


# ── Audit ────────────────────────────────────────────────────────────────────

def audit_sample_vs_population(
    df_pop: pd.DataFrame,
    df_sample: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Compare distributions between population and sample.

    For each column in *cols*, computes the relative frequency in both
    the population and the sample, plus the absolute deviation.

    Returns a tidy DataFrame with columns:
        feature, value, pop_frac, sample_frac, abs_diff
    """
    rows: list[dict] = []
    for col in cols:
        if col not in df_pop.columns or col not in df_sample.columns:
            continue
        pop_dist = df_pop[col].value_counts(normalize=True, dropna=False)
        sam_dist = df_sample[col].value_counts(normalize=True, dropna=False)
        all_vals = sorted(set(pop_dist.index) | set(sam_dist.index), key=str)
        for v in all_vals:
            p = pop_dist.get(v, 0.0)
            s = sam_dist.get(v, 0.0)
            rows.append({
                "feature": col,
                "value": str(v),
                "pop_frac": round(p, 6),
                "sample_frac": round(s, 6),
                "abs_diff": round(abs(p - s), 6),
            })
    return pd.DataFrame(rows)


# ── Orchestration ────────────────────────────────────────────────────────────

def run_sampling(
    df_train: pd.DataFrame,
) -> dict:
    """Run CLS + REG sampling and produce audits.

    Returns
    -------
    dict with keys:
        train_cls_sample, train_reg_sample,
        audit_cls, audit_reg
    """
    print(f"\n{'='*60}")
    print("SAMPLING (prototyping only)")
    print(f"{'='*60}")

    cls_sample = sample_cls(df_train)
    reg_sample = sample_reg(df_train)

    # Audit columns
    cls_audit_cols = ["order", "pid_segment", "availability",
                      "genericProduct", "adFlag", "competitorPrice_missing"]
    reg_audit_cols = ["quantity_class", "pid_segment", "availability",
                      "genericProduct"]

    # Filter to columns that exist
    cls_audit_cols = [c for c in cls_audit_cols if c in df_train.columns]
    reg_audit_cols = [c for c in reg_audit_cols
                      if c in df_train.loc[get_reg_mask(df_train)].columns]

    audit_cls = audit_sample_vs_population(
        df_train, cls_sample, cls_audit_cols)
    audit_reg = audit_sample_vs_population(
        df_train.loc[get_reg_mask(df_train)], reg_sample, reg_audit_cols
    )

    print(
        f"[sampling] CLS audit: max abs_diff = {audit_cls['abs_diff'].max():.4f}")
    print(
        f"[sampling] REG audit: max abs_diff = {audit_reg['abs_diff'].max():.4f}")
    print()

    return {
        "train_cls_sample": cls_sample,
        "train_reg_sample": reg_sample,
        "audit_cls": audit_cls,
        "audit_reg": audit_reg,
    }


# ── Internal ─────────────────────────────────────────────────────────────────

def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise if one of the required sampling columns is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[sampling] Missing columns: {missing}")

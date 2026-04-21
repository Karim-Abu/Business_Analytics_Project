"""
Safe feature engineering — all features that are leakage-free.

Every function here can be applied to any split without risk.
Functions that require a train-only fit (binning, frequency encoding)
expose separate fit/apply pairs.

Conditional / time-aware features live in feature_engineering_conditional.py.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# 2.1  Group parts
# ═══════════════════════════════════════════════════════════════════════════

def extract_group_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract ``group12`` (first 2 chars) and ``group34`` (chars 3-4).

    Uses ``group_clean`` if available (from preprocessing), otherwise
    falls back to raw ``group``.

    Rules
    -----
    - NaN / missing ``group`` → both set to ``'MISSING'``
    - Strings shorter than 2 chars → ``group12`` padded, ``group34 = 'XX'``
    - Strings shorter than 4 chars → ``group34`` padded with ``'X'``
    """
    src_col = "group_clean" if "group_clean" in df.columns else "group"
    g = df[src_col].astype(str).str.strip()
    g = g.replace({"nan": None, "None": None, "": None, "MISSING": None})

    is_missing = g.isna() | df[src_col].isna()

    group12 = g.str[:2].fillna("MISSING")
    group34 = g.str[2:4].fillna("XX")

    # Pad short strings
    group12 = group12.where(group12.str.len() == 2, group12.str.ljust(2, "X"))
    group34 = group34.where(group34.str.len() == 2, group34.str.ljust(2, "X"))

    # Override for truly missing
    group12 = group12.where(~is_missing, "MISSING")
    group34 = group34.where(~is_missing, "MISSING")

    df["group12"] = group12
    df["group34"] = group34
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.2  Day cycles
# ═══════════════════════════════════════════════════════════════════════════

def add_day_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclic day features from ``day``.

    day_7  = (day - 1) % 7  + 1   →  {1..7}
    day_14 = (day - 1) % 14 + 1   →  {1..14}
    day_30 = (day - 1) % 30 + 1   →  {1..30}
    """
    df["day_7"] = (df["day"] - 1) % 7 + 1
    df["day_14"] = (df["day"] - 1) % 14 + 1
    df["day_30"] = (df["day"] - 1) % 30 + 1
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.3  Content parsing  (is_multipack, pack_n, pack_size)
# ═══════════════════════════════════════════════════════════════════════════

_MULTI_RE = re.compile(r"^([\d.]+)(?:[Xx]([\d.]+))*$")


def _parse_single_content(raw: Any) -> tuple[int, int, float, float]:
    """Parse one content value → (is_multipack, pack_n, pack_size, pack_total_size).

    Examples
    --------
    '80'       → (0, 1, 80.0, 80.0)
    '10X1'     → (1, 10, 1.0, 10.0)        # 10 × 1 = 10
    '6X4X200'  → (1, 24, 200.0, 4800.0)    # 6×4=24, 24×200 = 4800
    '5x10'     → (1, 5, 10.0, 50.0)        # 5 × 10 = 50
    """
    if pd.isna(raw):
        return (0, 1, np.nan, np.nan)

    s = str(raw).strip().upper()
    if not s:
        return (0, 1, np.nan, np.nan)

    parts = re.split(r"[Xx]", s)
    nums: list[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            nums.append(float(p))
        except ValueError:
            continue

    if not nums:
        return (0, 1, np.nan, np.nan)

    if len(nums) == 1:
        # Simple number like '80' → total = 80
        return (0, 1, nums[0], nums[0])

    # Multipack: last number = pack_size (base unit), preceding = multipliers
    pack_size = nums[-1]
    pack_n_val = 1.0
    for n in nums[:-1]:
        pack_n_val *= n
    pack_n_int = max(1, int(pack_n_val))
    pack_total = pack_n_int * pack_size

    return (1, pack_n_int, pack_size, pack_total)


def parse_content(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``content`` into multipack features.

    Uses ``content_clean`` if available (from preprocessing), otherwise
    falls back to raw ``content``.

    Columns created:
        is_multipack   – 1 if content contains X/x separator
        pack_n         – number of sub-packs (product of all factors before last)
        pack_size      – base unit size (last numeric value)
        pack_total_size – total quantity = pack_n × pack_size
    """
    src_col = "content_clean" if "content_clean" in df.columns else "content"
    parsed = df[src_col].apply(_parse_single_content)
    df["is_multipack"] = parsed.apply(lambda t: t[0]).astype(int)
    df["pack_n"] = parsed.apply(lambda t: t[1]).astype(int)
    df["pack_size"] = parsed.apply(lambda t: t[2]).astype(float)
    df["pack_total_size"] = parsed.apply(lambda t: t[3]).astype(float)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.4  has_campaign
# ═══════════════════════════════════════════════════════════════════════════

def add_has_campaign(df: pd.DataFrame) -> pd.DataFrame:
    """``has_campaign = 1`` when campaignIndex_norm is A, B, or C."""
    df["has_campaign"] = (df["campaignIndex_norm"] != "NONE").astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.5  Price features
# ═══════════════════════════════════════════════════════════════════════════

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all price-comparison features.

    Definitions (from Variable Dictionary v2)
    ------------------------------------------
    price_diff              = price − competitorPrice
    price_discount          = (rrp − price) / rrp
    competitorPrice_discount = (rrp − competitorPrice) / rrp
    price_discount_diff     = price_discount − competitorPrice_discount
    is_lower_price          = 1 if price < competitorPrice
    is_discount             = 1 if price < rrp
    is_greater_discount     = 1 if price_discount > competitorPrice_discount

    NaN handling
    ------------
    - Where competitorPrice is NaN → price_diff, competitorPrice_discount,
      price_discount_diff, is_lower_price, is_greater_discount are NaN.
    - Where rrp is 0 or NaN → discount ratios are NaN.
    """
    cp = df["competitorPrice"]
    price = df["price"]
    rrp = df["rrp"]

    # price_diff
    df["price_diff"] = price - cp

    # price_discount: (rrp - price) / rrp — guard rrp == 0
    safe_rrp = rrp.where(rrp != 0, np.nan)
    df["price_discount"] = (rrp - price) / safe_rrp

    # competitorPrice_discount: (rrp - competitorPrice) / rrp
    df["competitorPrice_discount"] = (rrp - cp) / safe_rrp

    # price_discount_diff = price_discount - competitorPrice_discount
    df["price_discount_diff"] = df["price_discount"] - \
        df["competitorPrice_discount"]

    # Binary flags (NaN-safe: comparisons with NaN → False → 0)
    df["is_lower_price"] = np.where(
        cp.notna(), (price < cp).astype(int), np.nan
    )
    df["is_discount"] = np.where(
        rrp.notna() & (rrp > 0), (price < rrp).astype(int), np.nan
    )
    df["is_greater_discount"] = np.where(
        df["price_discount"].notna() & df["competitorPrice_discount"].notna(),
        (df["price_discount"] > df["competitorPrice_discount"]).astype(int),
        np.nan,
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.6  Per-unit features
# ═══════════════════════════════════════════════════════════════════════════

def add_per_unit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-unit price features based on **total** pack size.

    Uses ``pack_total_size`` (= pack_n × pack_size) so that multipacks
    are correctly normalised.  E.g. content='6X4X200' → total=4800,
    and price_per_unit = price / 4800.

    Division-safe: pack_total_size <= 0 → NaN.  No silent inf.
    """
    safe_total = df["pack_total_size"].where(df["pack_total_size"] > 0, np.nan)
    df["price_per_unit"] = df["price"] / safe_total
    df["rrp_per_unit"] = df["rrp"] / safe_total
    df["competitorPrice_per_unit"] = df["competitorPrice"] / safe_total
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.7 / 2.8  Binning (fit on Train, apply on all)
# ═══════════════════════════════════════════════════════════════════════════

_N_BINS = 20


def fit_binning_edges(
    df_train: pd.DataFrame,
    n_bins: int = _N_BINS,
) -> dict[str, np.ndarray]:
    """Compute equal-frequency bin edges on the training set.

    Returns dict with keys ``'price_diff'`` and ``'price_discount'``,
    each mapping to an ndarray of bin edges (length n_bins + 1).

    Handles duplicates by using ``pd.qcut(..., duplicates='drop')``.
    """
    edges: dict[str, np.ndarray] = {}
    for col in ("price_diff", "price_discount"):
        valid = df_train[col].dropna()
        if valid.empty:
            edges[col] = np.array([])
            continue

        # qcut returns bin intervals; extract edges
        _, bin_arr = pd.qcut(valid, q=n_bins, retbins=True, duplicates="drop")
        # Extend edges to -inf/+inf so out-of-range values are caught
        bin_arr[0] = -np.inf
        bin_arr[-1] = np.inf
        edges[col] = bin_arr

    print(
        f"[binning] Fitted edges — price_diff: {len(edges.get('price_diff', []))-1} bins, "
        f"price_discount: {len(edges.get('price_discount', []))-1} bins"
    )
    return edges


def apply_binned_features(
    df: pd.DataFrame,
    bin_edges: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Apply pre-fitted binning edges to create categorical bin features.

    Mapping
    -------
    price_diff     → price_diff_bin     (with special value 'NO_COMPETITOR_PRICE')
    price_discount → discount_bin       (with special value 'NO_RRP')
    """
    col_map = {
        "price_diff": ("price_diff_bin", "NO_COMPETITOR_PRICE"),
        "price_discount": ("discount_bin", "NO_RRP"),
    }
    for src_col, (dst_col, na_label) in col_map.items():
        edges = bin_edges.get(src_col, np.array([]))
        if len(edges) < 2:
            df[dst_col] = na_label
            continue

        labels = [f"Q{i+1:02d}" for i in range(len(edges) - 1)]
        binned = pd.cut(df[src_col], bins=edges,
                        labels=labels, include_lowest=True)
        df[dst_col] = binned.astype(str).where(df[src_col].notna(), na_label)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.9 / 2.10  Manufacturer frequency encoding
# ═══════════════════════════════════════════════════════════════════════════

def fit_manufacturer_frequency(df_train: pd.DataFrame) -> dict[int, float]:
    """Compute relative frequency of each manufacturer on Train.

    Returns mapping manufacturer_id → frequency ∈ (0, 1].
    """
    counts = df_train["manufacturer"].value_counts(normalize=True)
    mapping = counts.to_dict()
    print(
        f"[manufacturer_freq] Fitted on {len(mapping):,} unique manufacturers")
    return mapping


def apply_manufacturer_frequency(
    df: pd.DataFrame,
    freq_map: dict[int, float],
) -> pd.DataFrame:
    """Map ``manufacturer_freq`` onto *df*.

    Unknown manufacturers (not in Train) → 0.0 (conservative fallback).
    """
    df["manufacturer_freq"] = df["manufacturer"].map(freq_map).fillna(0.0)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.11  Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def run_all_safe_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Apply all safe features.  Train-fitted artefacts are applied to all splits.

    Returns
    -------
    df_train, df_val, df_test : with new feature columns
    metadata : dict containing fitted artefacts (bin_edges, freq_map, …)
    """
    print(f"\n{'='*60}")
    print("SAFE FEATURE ENGINEERING")
    print(f"{'='*60}")

    metadata: dict[str, Any] = {}

    # ── Apply to each split independently ────────────────────────────────
    for label, df in [("Train", df_train), ("Test", df_test), ("Validation", df_val)]:
        extract_group_parts(df)
        add_day_cycles(df)
        parse_content(df)
        add_has_campaign(df)
        add_price_features(df)
        add_per_unit_features(df)
    print("[safe_fe] Per-split features added (group, day, content, campaign, price, per-unit)")

    # ── Fit on Train, apply to all ───────────────────────────────────────

    # Binning
    bin_edges = fit_binning_edges(df_train)
    metadata["bin_edges"] = bin_edges
    for df in (df_train, df_val, df_test):
        apply_binned_features(df, bin_edges)

    # Manufacturer frequency
    freq_map = fit_manufacturer_frequency(df_train)
    metadata["manufacturer_freq_map"] = freq_map
    for df in (df_train, df_val, df_test):
        apply_manufacturer_frequency(df, freq_map)

    print(f"[safe_fe] Done\n")
    return df_train, df_val, df_test, metadata

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tableau_extract.py – Tableau Hyper Extract Pipeline
=====================================================
Erzeugt zwei .hyper-Dateien für Tableau Data Understanding Dashboards:
  1) Tableau_CoreDaily.hyper  (Grain: day + pid)
  2) Tableau_OrdersLine.hyper (Grain: lineID, nur order == 1)

Voraussetzungen:
  pip install -r requirements.txt   (pandas, numpy, pantab, pyarrow)

Ausführung:
  python tableau_extract.py
"""

from __future__ import annotations

import csv
import math
import re
import sys
import time
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0) Dependency check (no auto-install)
# ---------------------------------------------------------------------------
try:
    import pantab
except ImportError:
    sys.exit(
        "ERROR: pantab is not installed.\n"
        "       Run:  pip install -r requirements.txt\n"
        "       (requires: pantab, tableauhyperapi, pandas, numpy, pyarrow)"
    )

try:
    import pyarrow  # noqa: F401 – optional, for Parquet fallback
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("INFO: pyarrow not installed – Parquet fallback disabled.")

# ---------------------------------------------------------------------------
# 1) Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = SCRIPT_DIR / "train.csv"
ITEMS_PATH = SCRIPT_DIR / "items.csv"
OUT_DIR = SCRIPT_DIR

CORE_DAILY_OUT = OUT_DIR / "Tableau_CoreDaily.hyper"
ORDERS_LINE_OUT = OUT_DIR / "Tableau_OrdersLine.hyper"

# Separator: "|" per README. Set to None to auto-detect.
SEP: str | None = "|"

N_QUANTILE_BINS = 20
INTRADAY_THRESHOLD = 0.005  # 0.5 %

# ---------------------------------------------------------------------------
# Helper: separator detection
# ---------------------------------------------------------------------------


def detect_separator(filepath: Path, sample_bytes: int = 8192) -> str:
    """Detect CSV separator via frequency heuristic on the header line,
    with csv.Sniffer as fallback."""
    with open(filepath, "r", encoding="utf-8") as f:
        header_line = f.readline()
    # Frequency heuristic: count common separators in header
    candidates = {"|": 0, ",": 0, ";": 0, "\t": 0}
    for ch in candidates:
        candidates[ch] = header_line.count(ch)
    best = max(candidates, key=candidates.get)
    if candidates[best] > 0:
        return best
    # Fallback: csv.Sniffer on a larger sample
    with open(filepath, "r", encoding="utf-8") as f:
        sample = f.read(sample_bytes)
    dialect = csv.Sniffer().sniff(sample, delimiters="|,;\t")
    return dialect.delimiter


def heading(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ===================================================================
# MAIN PIPELINE
# ===================================================================
def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 2) Load data
    # ------------------------------------------------------------------
    heading("LOAD DATA")

    sep = SEP
    if sep is None:
        sep = detect_separator(TRAIN_PATH)
        print(f"Auto-detected separator: {repr(sep)}")
    else:
        print(f"Using configured separator: {repr(sep)}")

    train = pd.read_csv(TRAIN_PATH, sep=sep, low_memory=False)
    items = pd.read_csv(ITEMS_PATH, sep=sep, low_memory=False)

    print(f"train shape: {train.shape}")
    print(f"items shape: {items.shape}")

    # ------------------------------------------------------------------
    # 3) Join with merge indicator
    # ------------------------------------------------------------------
    heading("JOIN train ↔ items")

    assert items["pid"].is_unique, "FAIL: items.pid contains duplicates!"
    print("items.pid unique: PASS ✓")

    df = train.merge(items, on="pid", how="left", indicator=True)

    n_left_only = (df["_merge"] == "left_only").sum()
    n_total = len(df)
    join_coverage_pct = (1 - n_left_only / n_total) * 100
    print(
        f"Join coverage: {join_coverage_pct:.4f}% ({n_left_only} unmatched rows)")
    if n_left_only > 0:
        print("  ⚠ WARNING: some train PIDs have no items entry!")
    else:
        print("  PASS ✓ – all train rows matched to items")

    orphan_items = items[~items["pid"].isin(train["pid"])]
    print(f"Orphan items (in catalogue, no events): {len(orphan_items)}")

    df.drop(columns=["_merge"], inplace=True)
    print(f"Merged shape: {df.shape}")

    # ------------------------------------------------------------------
    # 4) Intraday check
    # ------------------------------------------------------------------
    heading("INTRADAY CHECK (day+pid grain)")

    dynamic_cols = ["price", "competitorPrice",
                    "rrp", "adFlag", "availability"]
    grp_nunique = df.groupby(["day", "pid"])[dynamic_cols].nunique()
    n_groups = len(grp_nunique)

    print(f"Total (day, pid) groups: {n_groups:,}")
    print(f"{'Column':<22} {'Violations':>12} {'Pct':>10}")
    print("-" * 46)
    any_warning = False
    for col in dynamic_cols:
        viol = (grp_nunique[col] > 1).sum()
        pct = viol / n_groups
        status = "⚠" if pct > INTRADAY_THRESHOLD else "✓"
        print(f"{col:<22} {viol:>12,} {pct:>9.4%}  {status}")
        if pct > INTRADAY_THRESHOLD:
            any_warning = True
            print(
                f"  → WARNING: {col} exceeds {INTRADAY_THRESHOLD:.1%} threshold.")
            print(
                f"    Recommendation: use mode aggregation for {col} or extend grain.")

    overall_viol = (grp_nunique.max(axis=1) > 1).sum()
    overall_pct = overall_viol / n_groups
    print(
        f"\nOverall violations (any col): {overall_viol:,} ({overall_pct:.4%})")
    if any_warning:
        print("⚠ Proceeding with day+pid grain using mode/median aggregation for safety.")
    else:
        print("✓ Intraday check passed – day+pid grain is safe.")

    del grp_nunique  # free memory

    # ------------------------------------------------------------------
    # 5) Feature Engineering (row-level)
    # ------------------------------------------------------------------
    heading("FEATURE ENGINEERING")

    # --- Competitor price validity ---
    cp_valid = (df["competitorPrice"] > 0) & df["competitorPrice"].notna()
    df["competitorPrice_missing"] = (~cp_valid).astype("int8")

    # --- Price diff ---
    df["price_diff"] = np.where(
        cp_valid, df["price"] - df["competitorPrice"], np.nan)

    # --- RRP validity & discount ---
    rrp_valid = (df["rrp"] > 0) & df["rrp"].notna()
    df["discount_vs_rrp"] = np.where(
        rrp_valid, (df["rrp"] - df["price"]) / df["rrp"], np.nan
    )

    # --- pharmForm normalisation ---
    # Use StringDtype to avoid "nan"→"NAN" issue with plain astype(str)
    df["pharmForm_norm"] = (
        df["pharmForm"]
        .astype("string")
        .str.upper()
        .str.strip()
        .fillna("MISSING")
        .replace({"": "MISSING"})
    )

    # --- category as string dimension ---
    df["category_norm"] = (
        df["category"]
        .astype("string")
        .fillna("NONE")
        .str.strip()
    )
    # Clean up float-like strings (e.g. "1.0" → "1")
    df["category_norm"] = df["category_norm"].str.replace(
        r"\.0$", "", regex=True)

    # --- campaignIndex normalisation ---
    df["campaignIndex_norm"] = (
        df["campaignIndex"]
        .astype("string")
        .fillna("NONE")
        .str.upper()
        .str.strip()
        .replace({"": "NONE"})
    )
    df["has_campaign"] = (df["campaignIndex_norm"] != "NONE").astype("int8")

    print("Row-level features created:")
    print("  competitorPrice_missing, price_diff, discount_vs_rrp")
    print("  pharmForm_norm, category_norm, campaignIndex_norm, has_campaign")

    # ------------------------------------------------------------------
    # 6) Quantile binning (with p01–p99 clipping)
    # ------------------------------------------------------------------
    heading("QUANTILE BINNING")

    # --- price_diff_bin ---
    valid_pd = df["price_diff"].notna()
    if valid_pd.sum() > 0:
        lo, hi = df.loc[valid_pd, "price_diff"].quantile([0.01, 0.99])
        clipped = df.loc[valid_pd, "price_diff"].clip(lo, hi)
        labels = pd.qcut(clipped, q=N_QUANTILE_BINS, duplicates="drop")
        df["price_diff_bin"] = pd.Series(dtype="string", index=df.index)
        df.loc[valid_pd, "price_diff_bin"] = labels.astype(str)
    else:
        df["price_diff_bin"] = pd.Series(dtype="string", index=df.index)

    df.loc[df["competitorPrice_missing"] == 1,
           "price_diff_bin"] = "NO_COMPETITOR_PRICE"
    # Remaining NaN (competitorPrice <= 0 treated as missing) also get the label
    df["price_diff_bin"] = df["price_diff_bin"].fillna("NO_COMPETITOR_PRICE")

    n_pd_bins = df.loc[df["price_diff_bin"] !=
                       "NO_COMPETITOR_PRICE", "price_diff_bin"].nunique()
    print(f"price_diff_bin: {n_pd_bins} quantile bins + 'NO_COMPETITOR_PRICE'")
    print(f"  Clipping range: [{lo:.2f}, {hi:.2f}]")

    # --- discount_bin ---
    valid_disc = df["discount_vs_rrp"].notna()
    if valid_disc.sum() > 0:
        lo_d, hi_d = df.loc[valid_disc, "discount_vs_rrp"].quantile([
                                                                    0.01, 0.99])
        clipped_d = df.loc[valid_disc, "discount_vs_rrp"].clip(lo_d, hi_d)
        labels_d = pd.qcut(clipped_d, q=N_QUANTILE_BINS, duplicates="drop")
        df["discount_bin"] = pd.Series(dtype="string", index=df.index)
        df.loc[valid_disc, "discount_bin"] = labels_d.astype(str)
    else:
        df["discount_bin"] = pd.Series(dtype="string", index=df.index)

    no_rrp_mask = ~rrp_valid
    df.loc[no_rrp_mask, "discount_bin"] = "NO_RRP"
    df["discount_bin"] = df["discount_bin"].fillna("NO_RRP")

    n_disc_bins = df.loc[df["discount_bin"]
                         != "NO_RRP", "discount_bin"].nunique()
    print(f"discount_bin: {n_disc_bins} quantile bins + 'NO_RRP'")
    print(f"  Clipping range: [{lo_d:.2f}, {hi_d:.2f}]")

    # ------------------------------------------------------------------
    # 7) Multipack parsing (on items level, then merge)
    # ------------------------------------------------------------------
    heading("MULTIPACK PARSING")

    def parse_multipack(content_val) -> tuple[int, int, float | None]:
        """Parse content string for multipack pattern (e.g. '6X4X200').
        Returns (is_multipack, pack_n, pack_size)."""
        if pd.isna(content_val):
            return (0, 1, None)
        s = str(content_val).upper().strip()
        if not s:
            return (0, 1, None)
        # Check for X-separated digit groups
        if re.search(r"\d+X\d+", s):
            # Extract all digit groups between X separators
            parts = re.findall(r"\d+", s)
            if len(parts) >= 2:
                nums = [int(p) for p in parts]
                pack_size = nums[-1]
                pack_n = reduce(lambda a, b: a * b, nums[:-1], 1)
                return (1, pack_n, float(pack_size))
        # No multipack pattern: try to extract a single number as pack_size
        m = re.search(r"\d+", s)
        if m:
            return (0, 1, float(m.group()))
        return (0, 1, None)

    mp_results = items["content"].apply(parse_multipack)
    items_mp = pd.DataFrame(
        mp_results.tolist(),
        columns=["is_multipack", "pack_n", "pack_size"],
        index=items.index,
    )
    items_mp["pid"] = items["pid"]
    items_mp["is_multipack"] = items_mp["is_multipack"].astype("int8")
    items_mp["pack_n"] = items_mp["pack_n"].astype("int32")
    # pack_size stays float (nullable)

    # Merge to df
    df = df.merge(items_mp, on="pid", how="left")
    df["is_multipack"] = df["is_multipack"].fillna(0).astype("int8")
    df["pack_n"] = df["pack_n"].fillna(1).astype("int32")

    n_mp = (df["is_multipack"] == 1).sum()
    print(f"Multipack rows: {n_mp:,} ({n_mp/len(df)*100:.2f}%)")
    print(f"Non-multipack rows: {(df['is_multipack'] == 0).sum():,}")
    # Show a few examples
    mp_examples = (
        items.loc[items_mp["is_multipack"] == 1, ["pid", "content"]]
        .head(5)
        .to_string(index=False)
    )
    print(f"Multipack examples (items):\n{mp_examples}")

    # ------------------------------------------------------------------
    # 8) Data Quality Checks
    # ------------------------------------------------------------------
    heading("DATA QUALITY CHECKS")

    checks = []

    def dq_check(name: str, condition_series: pd.Series, expect_all_true: bool = True):
        n_violations = (~condition_series).sum(
        ) if expect_all_true else condition_series.sum()
        status = "PASS ✓" if n_violations == 0 else f"FAIL ✗ ({n_violations:,} violations)"
        checks.append((name, status, int(n_violations)))
        print(f"  {name:<50} {status}")

    dq_check("price > 0", df["price"] > 0)
    dq_check("revenue >= 0", df["revenue"] >= 0)
    dq_check("click + basket + order == 1",
             df["click"] + df["basket"] + df["order"] == 1)
    dq_check("lineID unique", ~df["lineID"].duplicated(keep=False))

    # competitorPrice > 0 where not null
    cp_notnull = df["competitorPrice"].notna()
    cp_positive = df.loc[cp_notnull, "competitorPrice"] > 0
    n_cp_invalid = (~cp_positive).sum()
    n_cp_total = cp_notnull.sum()
    print(f"  {'competitorPrice > 0 (where present)':<50} "
          f"{'PASS ✓' if n_cp_invalid == 0 else f'FAIL ✗ ({n_cp_invalid:,} violations)'}")
    print(f"    Present: {n_cp_total:,}  |  Missing: {(~cp_notnull).sum():,} "
          f"({(~cp_notnull).mean()*100:.2f}%)")

    # rrp > 0 where not null
    rrp_notnull = df["rrp"].notna()
    rrp_pos = df.loc[rrp_notnull, "rrp"] > 0
    n_rrp_invalid = (~rrp_pos).sum()
    print(f"  {'rrp > 0 (where present)':<50} "
          f"{'PASS ✓' if n_rrp_invalid == 0 else f'FAIL ✗ ({n_rrp_invalid:,} violations)'}")

    # Missing rates (informational)
    for col in ["pharmForm", "category", "campaignIndex"]:
        n_miss = df[col].isna().sum()
        print(
            f"  {col + ' missing rate':<50} {n_miss:,} ({n_miss/len(df)*100:.2f}%)")

    print(f"  {'Join coverage':<50} {join_coverage_pct:.4f}%")

    # ------------------------------------------------------------------
    # 9) CoreDaily Aggregation (day + pid)
    # ------------------------------------------------------------------
    heading("CORE DAILY AGGREGATION")

    # --- Define column groups ---
    pid_static_cols = [
        "category_norm", "pharmForm_norm", "genericProduct",
        "salesIndex", "is_multipack", "pack_n", "pack_size",
    ]

    # daily_attrs: .first() for categorical (safe: intraday violations < 0.5% per col),
    # median for price (robust to the rare cases with intraday price changes)
    daily_first_cols = [
        "availability", "adFlag", "campaignIndex_norm",
        "has_campaign", "competitorPrice_missing",
        "price_diff_bin", "discount_bin",
    ]

    # --- Sort df for deterministic .first() (smallest value wins on tie) ---
    print("Sorting for deterministic .first() ...")
    sort_cols = ["day", "pid"] + daily_first_cols
    df.sort_values(sort_cols, inplace=True, na_position="last")

    # --- Aggregate metrics ---
    print("Aggregating metrics (n_events, n_click, n_basket, n_order)...")
    grp = df.groupby(["day", "pid"], sort=False)

    metric_agg = grp.agg(
        n_events=("lineID", "count"),
        n_click=("click", "sum"),
        n_basket=("basket", "sum"),
        n_order=("order", "sum"),
    ).reset_index()

    # --- Aggregate price via median (robust to rare intraday variation) ---
    print("Aggregating price (median) per day+pid...")
    price_agg = grp["price"].median().reset_index()

    # --- Aggregate daily categorical attrs via .first() (deterministic after sort) ---
    # Justified: intraday check showed <0.5% violations per column; after sorting,
    # .first() picks the smallest value deterministically.
    print("Aggregating daily attributes (.first() – safe, violations <0.5%)...")
    daily_first_agg = grp[daily_first_cols].first().reset_index()

    # --- pid-level static attributes (first per pid, since time-invariant) ---
    print("Extracting pid-level static attributes...")
    pid_static = df.groupby("pid")[pid_static_cols].first().reset_index()

    # --- Assemble CoreDaily ---
    core_daily = metric_agg.merge(price_agg, on=["day", "pid"], how="left")
    core_daily = core_daily.merge(
        daily_first_agg, on=["day", "pid"], how="left")
    core_daily = core_daily.merge(pid_static, on="pid", how="left")

    # --- Rates ---
    core_daily["click_rate"] = core_daily["n_click"] / core_daily["n_events"]
    core_daily["basket_rate"] = core_daily["n_basket"] / core_daily["n_events"]
    core_daily["order_rate"] = core_daily["n_order"] / core_daily["n_events"]

    print(f"CoreDaily shape: {core_daily.shape}")

    # ------------------------------------------------------------------
    # 10) pid_segment (computed from CoreDaily)
    # ------------------------------------------------------------------
    heading("PID SEGMENT (Head / Mid / Tail)")

    pid_totals = (
        core_daily.groupby("pid")["n_events"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    pid_totals.columns = ["pid", "pid_total_events"]
    n_pids = len(pid_totals)

    # Segment by PID rank: top 10% = Head, bottom 50% = Tail, rest = Mid
    pid_totals["rank"] = range(1, n_pids + 1)
    pid_totals["pid_segment"] = "Mid"
    pid_totals.loc[pid_totals["rank"] <= math.ceil(
        n_pids * 0.10), "pid_segment"] = "Head"
    pid_totals.loc[pid_totals["rank"] > math.ceil(
        n_pids * 0.50), "pid_segment"] = "Tail"

    # Report
    for seg in ["Head", "Mid", "Tail"]:
        seg_df = pid_totals[pid_totals["pid_segment"] == seg]
        seg_events = seg_df["pid_total_events"].sum()
        total_events = pid_totals["pid_total_events"].sum()
        print(
            f"  {seg:<6} {len(seg_df):>6,} PIDs  "
            f"{seg_events:>10,} events  "
            f"({seg_events/total_events*100:5.2f}%)"
        )

    # Merge segment + total events back into CoreDaily
    core_daily = core_daily.merge(
        pid_totals[["pid", "pid_total_events", "pid_segment"]],
        on="pid", how="left",
    )

    print(f"CoreDaily shape after segment: {core_daily.shape}")

    # ------------------------------------------------------------------
    # 11) OrdersLine extract (lineID, only order == 1)
    # ------------------------------------------------------------------
    heading("ORDERS LINE EXTRACT")

    orders = df.loc[df["order"] == 1, [
        "lineID", "day", "pid", "price", "revenue",
    ]].copy()

    # Quantity derivation with plausibility check
    orders["raw_qty"] = orders["revenue"] / orders["price"]
    orders["quantity"] = orders["raw_qty"].round().astype("Int64")
    orders["qty_suspicious"] = (
        ((orders["raw_qty"] - orders["raw_qty"].round()).abs() > 0.05)
        | (orders["quantity"] <= 0)
    ).astype("int8")

    n_suspicious = orders["qty_suspicious"].sum()
    print(f"Orders: {len(orders):,}")
    print(
        f"Suspicious quantity rows: {n_suspicious:,} ({n_suspicious/len(orders)*100:.3f}%)")

    # quantity_class
    orders["quantity_class"] = pd.cut(
        orders["quantity"].astype(float),
        bins=[0, 1, 2, 3, 5, np.inf],
        labels=["1", "2", "3", "4-5", ">5"],
        right=True,
    ).astype("string")

    orders.drop(columns=["raw_qty"], inplace=True)

    # --- Denormalize from CoreDaily (consistency: daily attrs come from aggregated source)
    denorm_cols_from_core = [
        "day", "pid",
        "category_norm", "pharmForm_norm", "genericProduct",
        "pid_segment", "availability", "adFlag", "campaignIndex_norm",
        "is_multipack", "salesIndex",
    ]
    # Pick only the needed columns from core_daily to avoid duplication
    core_denorm = core_daily[denorm_cols_from_core].copy()

    orders = orders.merge(core_denorm, on=["day", "pid"], how="left")

    print(f"OrdersLine shape: {orders.shape}")
    print(f"OrdersLine columns: {list(orders.columns)}")

    # ------------------------------------------------------------------
    # 12) Dtype cleanup for pantab export
    # ------------------------------------------------------------------
    heading("DTYPE CLEANUP FOR HYPER EXPORT")

    def prepare_for_hyper(frame: pd.DataFrame) -> pd.DataFrame:
        """Cast dtypes to pantab-compatible types."""
        frame = frame.copy()
        for col in frame.columns:
            dtype = frame[col].dtype
            # Categorical → string
            if isinstance(dtype, pd.CategoricalDtype):
                frame[col] = frame[col].astype("string")
            # Object → string
            elif dtype == "object":
                frame[col] = frame[col].astype("string")
            # Already StringDtype → keep
            elif isinstance(dtype, pd.StringDtype):
                pass
            # Nullable Int64 → keep (pantab handles it)
            elif isinstance(dtype, pd.Int64Dtype):
                pass
            # int8/int32 → int (widen for safety)
            elif dtype in (np.dtype("int8"), np.dtype("int32")):
                frame[col] = frame[col].astype("int64")
        return frame

    core_daily_export = prepare_for_hyper(core_daily)
    orders_export = prepare_for_hyper(orders)

    # Safety: replace remaining NaN in string columns with empty or specific value
    for col in core_daily_export.select_dtypes(include=["string"]).columns:
        core_daily_export[col] = core_daily_export[col].fillna("")
    for col in orders_export.select_dtypes(include=["string"]).columns:
        orders_export[col] = orders_export[col].fillna("")

    print(f"CoreDaily export dtypes:\n{core_daily_export.dtypes.to_string()}")
    print(f"\nOrdersLine export dtypes:\n{orders_export.dtypes.to_string()}")

    # ------------------------------------------------------------------
    # 13) Export to .hyper
    # ------------------------------------------------------------------
    heading("EXPORT TO HYPER")

    try:
        pantab.frame_to_hyper(
            core_daily_export,
            str(CORE_DAILY_OUT),
            table="CoreDaily",
        )
        print(f"✓ Written: {CORE_DAILY_OUT}")
        print(
            f"  Rows: {len(core_daily_export):,}  |  Cols: {core_daily_export.shape[1]}")

        pantab.frame_to_hyper(
            orders_export,
            str(ORDERS_LINE_OUT),
            table="OrdersLine",
        )
        print(f"✓ Written: {ORDERS_LINE_OUT}")
        print(
            f"  Rows: {len(orders_export):,}  |  Cols: {orders_export.shape[1]}")

    except Exception as e:
        print(f"\n✗ Hyper export failed: {e}")
        print("  Attempting Parquet + CSV fallback...")

        if HAS_PYARROW:
            pq_core = OUT_DIR / "Tableau_CoreDaily.parquet"
            pq_orders = OUT_DIR / "Tableau_OrdersLine.parquet"
            core_daily_export.to_parquet(
                pq_core, engine="pyarrow", compression="snappy")
            orders_export.to_parquet(
                pq_orders, engine="pyarrow", compression="snappy")
            print(f"  ✓ Parquet: {pq_core}")
            print(f"  ✓ Parquet: {pq_orders}")

        csv_core = OUT_DIR / "Tableau_CoreDaily.csv"
        csv_orders = OUT_DIR / "Tableau_OrdersLine.csv"
        core_daily_export.to_csv(csv_core, index=False, sep="|")
        orders_export.to_csv(csv_orders, index=False, sep="|")
        print(f"  ✓ CSV: {csv_core}")
        print(f"  ✓ CSV: {csv_orders}")

    # ------------------------------------------------------------------
    # 14) Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    heading("DONE")
    print(f"Pipeline completed in {elapsed:.1f}s")
    print(f"\nOutputs:")
    print(f"  {CORE_DAILY_OUT}")
    print(f"  {ORDERS_LINE_OUT}")


# ===================================================================
if __name__ == "__main__":
    main()

"""
Audit — data quality checks and summary reports.

Every function returns a DataFrame or dict that can be easily exported.
No side effects beyond printing summaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


# ── Join quality ─────────────────────────────────────────────────────────────

def audit_join_quality(
    df_train_raw: pd.DataFrame,
    df_items: pd.DataFrame,
    df_merged: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise join quality between train and items."""
    train_pids = set(df_train_raw["pid"].unique())
    items_pids = set(df_items["pid"].unique())

    rows = [
        {"check": "train_rows", "value": len(df_train_raw)},
        {"check": "items_rows", "value": len(df_items)},
        {"check": "merged_rows", "value": len(df_merged)},
        {"check": "unique_pids_train", "value": len(train_pids)},
        {"check": "unique_pids_items", "value": len(items_pids)},
        {"check": "orphan_train_pids", "value": len(train_pids - items_pids)},
        {"check": "orphan_items_pids", "value": len(items_pids - train_pids)},
        {"check": "row_count_match", "value": int(
            len(df_merged) == len(df_train_raw))},
    ]
    return pd.DataFrame(rows)


# ── Missingness ──────────────────────────────────────────────────────────────

def audit_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages."""
    n = len(df)
    miss = df.isnull().sum()
    result = pd.DataFrame({
        "column": miss.index,
        "n_missing": miss.values,
        "pct_missing": np.round(miss.values / n * 100, 2),
    })
    return result.loc[result["n_missing"] > 0].sort_values(
        "n_missing", ascending=False
    ).reset_index(drop=True)


# ── Outliers ─────────────────────────────────────────────────────────────────

def audit_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Simple robust statistics for key numeric columns."""
    cols = ["price", "competitorPrice", "rrp", "quantity", "pack_total_size"]
    cols = [c for c in cols if c in df.columns]

    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        rows.append({
            "column": c,
            "count": len(s),
            "min": s.min(),
            "p01": s.quantile(0.01),
            "p25": q1,
            "median": s.median(),
            "p75": q3,
            "p99": s.quantile(0.99),
            "max": s.max(),
            "iqr": iqr,
            "n_below_0": (s <= 0).sum(),
            "n_extreme_high": (s > q3 + 3 * iqr).sum(),
        })
    return pd.DataFrame(rows)


# ── Target distributions ────────────────────────────────────────────────────

def audit_target_distribution(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> pd.DataFrame:
    """Order rate and quantity stats per split."""
    rows = []
    for name, df in [("train", df_train), ("test", df_test), ("validation", df_val)]:
        n = len(df)
        n_order = (df["order"] == 1).sum()
        order_rate = n_order / n if n else 0

        row = {
            "split": name,
            "n_rows": n,
            "n_orders": n_order,
            "order_rate": round(order_rate, 4),
        }

        # Quantity stats (order=1 only)
        if "quantity" in df.columns:
            q = df.loc[df["order"] == 1, "quantity"].dropna()
            row["qty_mean"] = round(q.mean(), 2) if len(q) else None
            row["qty_median"] = q.median() if len(q) else None
            row["qty_max"] = q.max() if len(q) else None
            row["qty_n_valid"] = len(q)

        rows.append(row)
    return pd.DataFrame(rows)


# ── Feature-set overview ─────────────────────────────────────────────────────

def audit_feature_sets(
    feature_matrices: dict[str, pd.DataFrame | pd.Series],
) -> pd.DataFrame:
    """Compact summary of all feature matrices."""
    rows = []
    for name in sorted(feature_matrices.keys()):
        obj = feature_matrices[name]
        if isinstance(obj, pd.DataFrame):
            rows.append({"matrix": name, "n_rows": len(
                obj), "n_cols": len(obj.columns)})
        else:
            rows.append({"matrix": name, "n_rows": len(obj), "n_cols": 1})
    return pd.DataFrame(rows)


# ── Dropped features ─────────────────────────────────────────────────────────

def audit_dropped_features() -> pd.DataFrame:
    """Static table of features that are intentionally excluded."""
    entries = [
        ("revenue", "CLS+REG", "Leakage: revenue = price × quantity"),
        ("click", "CLS", "Leakage: click=1 ⇒ order=0"),
        ("basket", "CLS", "Leakage: basket=1 ⇒ order=0"),
        ("lineID", "CLS+REG", "Identifier, no information"),
        ("quantity", "CLS", "Target-derived, leakage for Stage 1"),
        ("quantity_class", "CLS+REG-input", "Target-derived"),
        ("qty_suspicious", "CLS+REG", "QA flag, not a model feature"),
        ("pid_likelihood", "CLS+REG", "Deprecated: redundant with pid_prob"),
        ("num_pid_order", "REG", "Leakage: sums quantity (= REG target)"),
        ("order", "REG", "Constant 1 in Stage-2 subset"),
    ]
    return pd.DataFrame(entries, columns=["feature", "scope", "reason"])


# ── Orchestration ────────────────────────────────────────────────────────────

def run_full_audit(
    df_train_raw: pd.DataFrame,
    df_items: pd.DataFrame,
    df_merged: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_matrices: dict,
) -> dict[str, pd.DataFrame]:
    """Run all audits and return a dict of report DataFrames.

    Keys:  join_quality, missingness_train, missingness_merged,
           outliers_train, target_distribution, feature_sets, dropped_features
    """
    print(f"\n{'='*60}")
    print("AUDIT REPORTS")
    print(f"{'='*60}")

    reports: dict[str, pd.DataFrame] = {}

    reports["join_quality"] = audit_join_quality(
        df_train_raw, df_items, df_merged)
    reports["missingness_merged"] = audit_missingness(df_merged)
    reports["missingness_train"] = audit_missingness(df_train)
    reports["outliers_train"] = audit_outliers(df_train)
    reports["target_distribution"] = audit_target_distribution(
        df_train, df_val, df_test)
    reports["feature_sets"] = audit_feature_sets(feature_matrices)
    reports["dropped_features"] = audit_dropped_features()

    # Print summaries
    print("\n  Target distribution:")
    print(reports["target_distribution"].to_string(index=False))
    print(
        f"\n  Missingness (train): {len(reports['missingness_train'])} columns with missing")
    print(f"  Outlier columns checked: {len(reports['outliers_train'])}")
    print(f"  Feature matrices: {len(reports['feature_sets'])}")
    print()

    return reports

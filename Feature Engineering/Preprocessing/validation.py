"""
validation — defensive checks for feature lists and data integrity.

Runde-1 module.  Extended in Runde 6 with export-specific checks.
Extended in Runde 7 with preprocessing integrity checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assert_no_forbidden_features(
    feature_list: list[str],
    forbidden: list[str],
    context: str = "",
) -> None:
    """Raise if any feature in *feature_list* appears in *forbidden*.

    Parameters
    ----------
    feature_list : features to check
    forbidden    : features that must not appear
    context      : label for error message
    """
    overlap = sorted(set(feature_list) & set(forbidden))
    if overlap:
        raise ValueError(
            f"[{context}] Forbidden feature(s) found: {overlap}"
        )


def assert_no_duplicate_features(
    feature_list: list[str],
    context: str = "",
) -> None:
    """Raise if *feature_list* contains duplicate entries."""
    seen: set[str] = set()
    dupes: list[str] = []
    for f in feature_list:
        if f in seen:
            dupes.append(f)
        seen.add(f)
    if dupes:
        raise ValueError(
            f"[{context}] Duplicate feature(s): {dupes}"
        )


def assert_cross_split_columns(
    frames: dict[str, pd.DataFrame],
    context: str = "",
) -> None:
    """Raise if DataFrames in *frames* have different column lists/order."""
    # The first frame becomes the contract for all sibling splits.
    ref_name, ref_cols = None, None
    for name, df in frames.items():
        cols = list(df.columns)
        if ref_cols is None:
            ref_name, ref_cols = name, cols
        elif cols != ref_cols:
            raise ValueError(
                f"[{context}] Column mismatch between '{ref_name}' and "
                f"'{name}'.\n  {ref_name}: {ref_cols}\n  {name}: {cols}"
            )


def assert_reg_stage2_only(
    df: pd.DataFrame,
    context: str = "",
) -> None:
    """Raise if *df* contains rows that violate REG Stage-2 criteria.

    Stage-2 requires: order == 1, quantity not NaN,
    and qty_suspicious == 0 (if the column is present).
    """
    # Count each violation separately to make debugging failed exports easier.
    bad_order = (df["order"] != 1).sum() if "order" in df.columns else 0
    bad_qty = df["quantity"].isna().sum() if "quantity" in df.columns else 0
    bad_susp = 0
    if "qty_suspicious" in df.columns:
        bad_susp = (df["qty_suspicious"] != 0).sum()

    violations = bad_order + bad_qty + bad_susp
    if violations:
        raise ValueError(
            f"[{context}] REG Stage-2 violation: "
            f"order!=1: {bad_order}, quantity NaN: {bad_qty}, "
            f"qty_suspicious!=0: {bad_susp}"
        )


def assert_preprocessing_integrity(
    df: pd.DataFrame,
    context: str = "",
) -> None:
    """Check postconditions of run_all_preprocessing().

    Validates
    ---------
    - competitorPrice: no values <= 0  (NaN is allowed — imputation still open)
    - competitorPrice_missing: binary {0, 1}
    - Normalised columns exist and contain no empty strings
    - quantity only set where order == 1
    - qty_suspicious is binary {0, 1} and consistent with quantity/order
    """
    errors: list[str] = []

    # competitorPrice: no invalid positives
    if "competitorPrice" in df.columns:
        bad_cp = (df["competitorPrice"] <= 0).sum()
        if bad_cp:
            errors.append(f"competitorPrice has {bad_cp} values <= 0")

    # competitorPrice_missing: binary
    if "competitorPrice_missing" in df.columns:
        vals = set(df["competitorPrice_missing"].unique())
        if not vals <= {0, 1}:
            errors.append(
                f"competitorPrice_missing not binary: {vals}")

    # Normalised / cleaned columns: must exist and have no empty strings
    norm_cols = [
        "campaignIndex_norm", "category_norm", "pharmForm_norm",
        "unit_norm", "group_clean", "content_clean",
    ]
    for col in norm_cols:
        if col not in df.columns:
            errors.append(f"Expected column '{col}' missing")
        else:
            n_empty = (df[col].astype(str).str.strip() == "").sum()
            if n_empty:
                errors.append(f"'{col}' has {n_empty} empty strings")

    # quantity only on order == 1
    if "quantity" in df.columns and "order" in df.columns:
        order0_with_qty = (
            (df["order"] == 0) & df["quantity"].notna()
        ).sum()
        if order0_with_qty:
            errors.append(
                f"quantity set on {order0_with_qty} rows with order==0")

    # qty_suspicious: binary and consistent
    if "qty_suspicious" in df.columns:
        vals = set(df["qty_suspicious"].unique())
        if not vals <= {0, 1}:
            errors.append(f"qty_suspicious not binary: {vals}")
        # order==0 must never be suspicious
        if "order" in df.columns:
            sus_on_0 = (
                (df["order"] == 0) & (df["qty_suspicious"] == 1)
            ).sum()
            if sus_on_0:
                errors.append(
                    f"qty_suspicious==1 on {sus_on_0} rows with order==0")

    if errors:
        msg = "; ".join(errors)
        raise ValueError(f"[{context}] Preprocessing integrity: {msg}")
    print(f"[validation] Preprocessing integrity OK ({context})")

"""
Feature sets — list management, validation, matrix assembly and export.

Reads feature-set definitions from config.py and provides helpers
to build clean (X, y) pairs for each model stage × feature tier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

import config as cfg
from validation import assert_no_forbidden_features, assert_no_duplicate_features


# ── Feature-list registry ────────────────────────────────────────────────────

_REGISTRY: dict[str, list[str]] = {
    "CLS_BASE_SAFE": cfg.CLS_BASE_SAFE,
    "CLS_EXPANDED_SAFE": cfg.CLS_EXPANDED_SAFE,
    "REG_BASE_SAFE": cfg.REG_BASE_SAFE,
    "REG_EXPANDED_SAFE": cfg.REG_EXPANDED_SAFE,
    "CLS_CONDITIONAL": cfg.CLS_CONDITIONAL,
    "REG_CONDITIONAL": cfg.REG_CONDITIONAL,
    "CLS_FINAL": cfg.CLS_FINAL,
    "REG_FINAL": cfg.REG_FINAL,
    "CLS_FINAL_NO_GROUPS": cfg.CLS_FINAL_NO_GROUPS,
    "REG_FINAL_NO_GROUP34": cfg.REG_FINAL_NO_GROUP34,
}

# Fail-fast: Final sets drive both matrix assembly and Orange export. Any
# duplicate here would silently create unstable training/export schemas.
for _fname in ("CLS_FINAL", "REG_FINAL",
               "CLS_FINAL_NO_GROUPS", "REG_FINAL_NO_GROUP34"):
    assert_no_duplicate_features(_REGISTRY[_fname], f"feature_sets.{_fname}")


def get_feature_list(set_name: str) -> list[str]:
    """Return the feature list for *set_name*.

    Supported names: CLS_BASE_SAFE, CLS_EXPANDED_SAFE, REG_BASE_SAFE,
    REG_EXPANDED_SAFE, CLS_CONDITIONAL, REG_CONDITIONAL.
    """
    if set_name not in _REGISTRY:
        raise ValueError(
            f"Unknown feature set '{set_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return list(_REGISTRY[set_name])  # defensive copy


# ── Validation ───────────────────────────────────────────────────────────────

def validate_feature_list(
    df: pd.DataFrame,
    feature_list: list[str],
    set_name: str = "feature_set",
) -> None:
    """Raise if any feature in *feature_list* is missing from *df*."""
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{set_name}] {len(missing)} feature(s) missing from DataFrame: "
            f"{missing}"
        )


# ── Matrix assembly ──────────────────────────────────────────────────────────

def assemble_X_y(
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
    set_name: str = "",
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) from *df* with validation.

    Steps
    -----
    1. Validate all features are present.
    2. Validate target column is present.
    3. Check forbidden features (CLS or REG depending on target).
    4. Return defensive copies.
    """
    validate_feature_list(df, feature_list, set_name)

    if target_col not in df.columns:
        raise ValueError(
            f"[{set_name}] Target column '{target_col}' not in DataFrame"
        )

    # Forbidden-feature guard
    if target_col == "order":
        assert_no_forbidden_features(feature_list, cfg.VERBOTEN_CLS, set_name)
    elif target_col == "quantity":
        assert_no_forbidden_features(feature_list, cfg.VERBOTEN_REG, set_name)

    X = df[feature_list].copy()
    y = df[target_col].copy()
    return X, y


# ── Shared REG mask ───────────────────────────────────────────────────────────

def get_reg_mask(df: pd.DataFrame) -> pd.Series:
    """Stage-2 (REG) inclusion mask: order==1, valid quantity, not suspicious.

    Single source of truth for REG row selection — used by feature-matrix
    assembly AND REG sampling.
    """
    # Keep the Stage-2 definition centralized so sampling, matrices and Orange
    # exports all refer to the exact same regression population.
    mask = (df["order"] == 1) & (df["quantity"].notna())
    if "qty_suspicious" in df.columns:
        mask = mask & (df["qty_suspicious"] == 0)
    return mask


# ── Build all safe matrices ──────────────────────────────────────────────────

def build_safe_feature_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Build all SAFE feature matrices for CLS and REG.

    For REG matrices, only rows with ``order == 1`` are used.

    Returns
    -------
    dict  with keys like ``X_train_cls_base``, ``y_train_cls``, etc.
    """
    print(f"\n{'='*60}")
    print("BUILDING SAFE FEATURE MATRICES")
    print(f"{'='*60}")

    result: dict[str, pd.DataFrame | pd.Series] = {}

    splits = {
        "train": df_train,
        "test": df_test,
        "validation": df_val,
    }

    # ── CLS matrices ─────────────────────────────────────────────────────
    for split_name, df in splits.items():
        for tier, set_name in [("base", "CLS_BASE_SAFE"), ("expanded", "CLS_EXPANDED_SAFE")]:
            features = get_feature_list(set_name)
            label = f"{split_name}_cls_{tier}"
            X, y = assemble_X_y(df, features, "order", set_name=label)
            result[f"X_{label}"] = X
            if f"y_{split_name}_cls" not in result:
                result[f"y_{split_name}_cls"] = y

    # ── REG matrices (order==1 AND valid quantity target) ──────────────
    #   Stage 2 must not contain NaN targets or suspicious derivations,
    #   otherwise regression / filter / wrapper methods will break.
    for split_name, df in splits.items():
        reg_mask = get_reg_mask(df)
        df_reg = df.loc[reg_mask].copy()

        n_excluded = (df["order"] == 1).sum() - len(df_reg)
        if n_excluded > 0:
            print(f"  [{split_name}] REG: excluded {n_excluded} order=1 rows "
                  f"(NaN quantity or suspicious)")
        if df_reg.empty:
            print(f"  [WARN] {split_name} has 0 valid REG rows")

        for tier, set_name in [("base", "REG_BASE_SAFE"), ("expanded", "REG_EXPANDED_SAFE")]:
            features = get_feature_list(set_name)
            label = f"{split_name}_reg_{tier}"
            X, y = assemble_X_y(df_reg, features, "quantity", set_name=label)
            result[f"X_{label}"] = X
            if f"y_{split_name}_reg" not in result:
                result[f"y_{split_name}_reg"] = y

    print(f"  Built {len(result)} matrices")
    _print_summary(result)
    return result


# ── Summary ──────────────────────────────────────────────────────────────────

def summarize_feature_sets(
    feature_matrices: dict[str, pd.DataFrame | pd.Series],
) -> pd.DataFrame:
    """Create a summary table of all matrices.

    Columns: matrix_name, n_rows, n_cols, target_name.
    """
    rows = []
    for name, obj in sorted(feature_matrices.items()):
        if isinstance(obj, pd.DataFrame):
            rows.append({
                "matrix": name,
                "n_rows": len(obj),
                "n_cols": len(obj.columns),
                "target": "",
            })
        elif isinstance(obj, pd.Series):
            rows.append({
                "matrix": name,
                "n_rows": len(obj),
                "n_cols": 1,
                "target": obj.name if obj.name else name,
            })
    return pd.DataFrame(rows)


def _print_summary(result: dict) -> None:
    """Quick console summary."""
    x_keys = sorted(k for k in result if k.startswith("X_"))
    y_keys = sorted(k for k in result if k.startswith("y_"))

    print("\n  Feature matrices:")
    for k in x_keys:
        obj = result[k]
        print(f"    {k:35s}  {obj.shape[0]:>7,} rows × {obj.shape[1]:>3} cols")

    print("  Targets:")
    for k in y_keys:
        obj = result[k]
        print(f"    {k:35s}  {len(obj):>7,} rows")
    print()


# ── Build conditional matrices ───────────────────────────────────────────────

def build_conditional_feature_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Build CONDITIONAL feature matrices for CLS and REG.

    Assumes that all conditional columns already exist on the DataFrames
    (produced by ``run_all_conditional_features``).

    Returns
    -------
    dict  with keys like ``X_train_cls_conditional``, ``y_train_cls``, etc.
    """
    print(f"\n{'='*60}")
    print("BUILDING CONDITIONAL FEATURE MATRICES")
    print(f"{'='*60}")

    result: dict[str, pd.DataFrame | pd.Series] = {}

    splits = {
        "train": df_train,
        "test": df_test,
        "validation": df_val,
    }

    # ── CLS conditional ──────────────────────────────────────────────────
    for split_name, df in splits.items():
        features = get_feature_list("CLS_CONDITIONAL")
        label = f"{split_name}_cls_conditional"
        X, y = assemble_X_y(df, features, "order", set_name=label)
        result[f"X_{label}"] = X
        if f"y_{split_name}_cls" not in result:
            result[f"y_{split_name}_cls"] = y

    # ── REG conditional ──────────────────────────────────────────────────
    for split_name, df in splits.items():
        df_reg = df.loc[get_reg_mask(df)].copy()

        if df_reg.empty:
            print(f"  [WARN] {split_name} has 0 valid REG rows")

        features = get_feature_list("REG_CONDITIONAL")
        label = f"{split_name}_reg_conditional"
        X, y = assemble_X_y(df_reg, features, "quantity", set_name=label)
        result[f"X_{label}"] = X
        if f"y_{split_name}_reg" not in result:
            result[f"y_{split_name}_reg"] = y

    print(f"  Built {len(result)} conditional matrices")
    _print_summary(result)
    return result


# ── Export ────────────────────────────────────────────────────────────────────

def export_matrices(
    feature_matrices: dict[str, pd.DataFrame | pd.Series],
    output_dir: Path | str = cfg.OUTPUT_DATASETS_DIR,
) -> None:
    """Save all matrices as parquet files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in feature_matrices.items():
        path = output_dir / f"{name}.parquet"
        if isinstance(obj, pd.Series):
            obj = obj.to_frame(name=obj.name if obj.name else name)
        obj.to_parquet(path, index=False)

    print(f"[export] {len(feature_matrices)} files → {output_dir}")

"""
orange_export — build and export model-ready CSV files for Orange.

Produces exactly 8 CSV files (features + target, target as last column)
plus an export_manifest.csv under ``cfg.OUTPUT_ORANGE_EXPORTS_DIR``.

Usage:  called from ``main_build_datasets.py`` after feature engineering,
        matrix assembly and sampling are complete.
"""

from __future__ import annotations

import datetime as dt
import pandas as pd

import config as cfg
from feature_sets import get_feature_list, get_reg_mask
from validation import (
    assert_no_forbidden_features,
    assert_no_duplicate_features,
    assert_cross_split_columns,
    assert_reg_stage2_only,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_orange_df(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Select *features* + *target_col*, target as last column.

    Categorical columns listed in ``cfg.CATEGORICAL_AS_STRING`` are
    explicitly cast to ``str`` so Orange does not infer them as numeric.
    """
    # Orange expects a flat table where the target is the final column.
    cols = list(features) + [target_col]
    out = df[cols].copy()
    for c in cfg.CATEGORICAL_AS_STRING:
        if c in out.columns:
            out[c] = out[c].astype(str)
    # Prefix numeric-looking categoricals so Orange treats them as discrete.
    for c, prefix in cfg.ORANGE_DISCRETE_PREFIX.items():
        if c in out.columns:
            out[c] = prefix + out[c].astype(str)
    return out


def _validate_export(
    frames: dict[str, pd.DataFrame],
    stage: str,
    features: list[str],
    target_col: str,
) -> None:
    """Pre-export sanity checks for one stage (CLS or REG)."""
    # Final sets are fixed contract surfaces. Fail early if the config drifts.
    assert_no_duplicate_features(features, f"orange_export_{stage}")

    # Prevent leakage features from reaching Orange exports.
    forbidden = cfg.VERBOTEN_CLS if stage == "CLS" else cfg.VERBOTEN_REG
    assert_no_forbidden_features(features, forbidden, f"orange_export_{stage}")

    expected_cols = list(features) + [target_col]

    for name, df in frames.items():
        actual = list(df.columns)
        if actual != expected_cols:
            raise ValueError(
                f"[orange_export] Column mismatch in {name}.\n"
                f"  Expected ({len(expected_cols)}): {expected_cols}\n"
                f"  Got      ({len(actual)}): {actual}"
            )
        if df[target_col].isna().any():
            n_na = df[target_col].isna().sum()
            raise ValueError(
                f"[orange_export] {name} has {n_na} NaN values in target "
                f"'{target_col}'"
            )

    # Train/test/validation/sample must stay column-identical so Orange models can
    # be trained and applied without any manual column reconciliation.
    assert_cross_split_columns(frames, f"orange_export_{stage}_splits")


# ── Main export entry point ─────────────────────────────────────────────────

def export_orange_csvs(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    sampling_result: dict,
    build_mode: str,
) -> None:
    """Build, validate and write 8 Orange-ready CSVs + manifest.

    Files
    -----
    cls_train_full.csv, cls_train_sample.csv, cls_test.csv, cls_val.csv
    reg_train_full.csv, reg_train_sample.csv, reg_test.csv, reg_val.csv
    export_manifest.csv
    """
    # ── Guard: build_mode must be valid ──────────────────────────────────
    if build_mode not in cfg.BUILD_MODES:
        raise ValueError(
            f"[orange_export] Invalid build_mode '{build_mode}'. "
            f"Allowed: {cfg.BUILD_MODES}"
        )

    out_dir = cfg.OUTPUT_ORANGE_EXPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("ORANGE CSV EXPORT")
    print(f"{'='*60}")

    cls_features = get_feature_list("CLS_FINAL")
    reg_features = get_feature_list("REG_FINAL")

    # ── CLS DataFrames ───────────────────────────────────────────────────
    cls_train_sample_df = sampling_result["train_cls_sample"]

    cls_frames = {
        "cls_train_full":   _build_orange_df(df_train, cls_features, "order"),
        "cls_train_sample": _build_orange_df(cls_train_sample_df, cls_features, "order"),
        "cls_test":         _build_orange_df(df_test, cls_features, "order"),
        "cls_val":          _build_orange_df(df_val, cls_features, "order"),
    }

    _validate_export(cls_frames, "CLS", cls_features, "order")

    # ── REG DataFrames (Stage-2 mask applied) ────────────────────────────
    reg_train_sample_df = sampling_result["train_reg_sample"]

    df_train_reg = df_train.loc[get_reg_mask(df_train)].copy()
    df_val_reg = df_val.loc[get_reg_mask(df_val)].copy()
    df_test_reg = df_test.loc[get_reg_mask(df_test)].copy()

    # Validate before projection to feature columns so Stage-2 violations are
    # diagnosed against the full source frame, not a reduced export slice.
    for tag, src in [("reg_train_full", df_train_reg),
                     ("reg_test", df_test_reg),
                     ("reg_val", df_val_reg)]:
        assert_reg_stage2_only(src, f"orange_export_{tag}")

    reg_frames = {
        "reg_train_full":   _build_orange_df(df_train_reg, reg_features, "quantity"),
        "reg_train_sample": _build_orange_df(reg_train_sample_df, reg_features, "quantity"),
        "reg_test":         _build_orange_df(df_test_reg, reg_features, "quantity"),
        "reg_val":          _build_orange_df(df_val_reg, reg_features, "quantity"),
    }

    _validate_export(reg_frames, "REG", reg_features, "quantity")

    # ── Write CSVs ───────────────────────────────────────────────────────
    all_frames = {**cls_frames, **reg_frames}
    for name, df in all_frames.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(
            f"  [orange] {path.name:30s}  {len(df):>9,} rows × {len(df.columns):>3} cols")

    # ── Conservative variant exports (train_full / val / test only) ──────
    cls_ng_features = get_feature_list("CLS_FINAL_NO_GROUPS")
    reg_ng_features = get_feature_list("REG_FINAL_NO_GROUP34")

    variant_frames: dict[str, tuple[pd.DataFrame, str, str]] = {}
    #                    name -> (df, feature_set_name, stage)

    for vname, src, feats, tgt, fset in [
        ("cls_train_full_no_groups", df_train,
         cls_ng_features, "order",    "CLS_FINAL_NO_GROUPS"),
        ("cls_val_no_groups",        df_val,
         cls_ng_features, "order",    "CLS_FINAL_NO_GROUPS"),
        ("cls_test_no_groups",       df_test,
         cls_ng_features, "order",    "CLS_FINAL_NO_GROUPS"),
        ("reg_train_full_no_group34", df_train_reg,
         reg_ng_features, "quantity", "REG_FINAL_NO_GROUP34"),
        ("reg_val_no_group34",        df_val_reg,
         reg_ng_features, "quantity", "REG_FINAL_NO_GROUP34"),
        ("reg_test_no_group34",       df_test_reg,
         reg_ng_features, "quantity", "REG_FINAL_NO_GROUP34"),
    ]:
        built = _build_orange_df(src, feats, tgt)
        path = out_dir / f"{vname}.csv"
        built.to_csv(path, index=False)
        print(
            f"  [orange] {path.name:30s}  {len(built):>9,} rows × {len(built.columns):>3} cols")
        variant_frames[vname] = (
            built, fset, "CLS" if vname.startswith("cls") else "REG")

    # ── Manifest ─────────────────────────────────────────────────────────
    # The manifest is the machine-readable inventory of all Orange exports.
    manifest_rows = []
    for name, df in all_frames.items():
        stage = "CLS" if name.startswith("cls") else "REG"
        split = name.replace("cls_", "").replace("reg_", "")
        is_sampled = "sample" in name
        target = "order" if stage == "CLS" else "quantity"
        fset = "CLS_FINAL" if stage == "CLS" else "REG_FINAL"
        sampling_frac = None
        if is_sampled:
            sampling_frac = (
                cfg.SAMPLE_FRAC_CLS if stage == "CLS" else cfg.SAMPLE_FRAC_REG
            )

        manifest_rows.append({
            "file_name":        f"{name}.csv",
            "stage":            stage,
            "split":            split,
            "n_rows":           len(df),
            "n_features":       len(df.columns) - 1,
            "target_name":      target,
            "feature_set_name": fset,
            "sampling_used":    is_sampled,
            "sampling_frac":    sampling_frac,
            "build_mode":       build_mode,
            "source_pipeline":  "main_build_datasets.py",
            "reg_mask_applied": stage == "REG",
            "created_at":       dt.datetime.now().isoformat(timespec="seconds"),
        })

    # Append variant entries to manifest
    now_ts = dt.datetime.now().isoformat(timespec="seconds")
    for vname, (vdf, vfset, vstage) in variant_frames.items():
        # Derive split from name: strip stage prefix and variant suffix
        raw = vname.replace("cls_", "").replace("reg_", "")
        split = raw.replace("_no_groups", "").replace("_no_group34", "")
        target = "order" if vstage == "CLS" else "quantity"
        manifest_rows.append({
            "file_name":        f"{vname}.csv",
            "stage":            vstage,
            "split":            split,
            "n_rows":           len(vdf),
            "n_features":       len(vdf.columns) - 1,
            "target_name":      target,
            "feature_set_name": vfset,
            "sampling_used":    False,
            "sampling_frac":    None,
            "build_mode":       build_mode,
            "source_pipeline":  "main_build_datasets.py",
            "reg_mask_applied": vstage == "REG",
            "created_at":       now_ts,
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "export_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"  [orange] {'export_manifest.csv':30s}  {len(manifest)} entries")

    print(f"\n  Orange exports → {out_dir}")

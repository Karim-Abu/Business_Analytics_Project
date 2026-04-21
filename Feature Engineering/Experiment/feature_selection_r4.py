"""
feature_selection_r4 — Round 4: Pruned sets, Embedded selection, Family ablation.

Builds on filter results from Round 3 (feature_selection.py).
No SHAP, no Permutation Importance, no Test-set evaluation.

Usage:
    cd "Feature Engineering"
    python feature_selection_r4.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

import config as cfg
from io_utils import save_csv, save_text_report

OUT = cfg.OUTPUT_FEATURE_SELECTION_DIR
# subsample for linear models (feature selection, not final)
LINEAR_SUBSAMPLE = 200_000

# ═════════════════════════════════════════════════════════════════════════════
#  1.  Pruned candidate-set definitions
# ═════════════════════════════════════════════════════════════════════════════

# --- Redundancy resolution (from filter reports) ---
#
# CLS r=1.000  is_lower_price ↔ is_greater_discount  → keep is_greater_discount (MI=0.001)
# CLS r=1.000  order_time ↔ num_pid_order            → keep order_time (MI=0.020)
# BOTH r≥0.97  rrp ↔ competitorPrice                 → keep competitorPrice (higher MI)
# BOTH r≥0.97  per-unit trio                          → keep price_per_unit only
# REG  r=-0.97 availability_likelihood ↔ day_7_qty_mean_oof → both MI=0 in REG, drop both

_DROP_REDUNDANCY = {
    "is_lower_price":           "redundant r=1.0 with is_greater_discount",
    "rrp":                      "redundant r≥0.97 with competitorPrice",
    "rrp_per_unit":             "redundant r≥0.97 with price_per_unit",
    "competitorPrice_per_unit": "redundant r≥0.98 with price_per_unit",
    "num_pid_order":            "redundant r=1.0 with order_time",
}

_DROP_NZV = {
    "competitorPrice_missing":  "NZV (near-zero variance)",
    "pack_n":                   "NZV + MI=0",
    "is_discount":              "NZV",
}

# MI=0 across relevant stage — no univariate signal
_DROP_CLS_WEAK = {
    "genericProduct":           "MI_CLS=0",
    "manufacturer_freq":        "MI_CLS=0",
    "is_multipack":             "MI_CLS=0",
    "pack_size":                "MI_CLS=0",
    "pack_total_size":          "MI_CLS=0",
    "price_diff":               "MI_CLS=0",
    "price_discount":           "MI_CLS=0",
    "competitorPrice_discount": "MI_CLS=0",
    "price_discount_diff":      "MI_CLS=0",
    "campaignIndex_norm":       "MI_CLS≈0  (0.000034)",
}
_DROP_CLS_COND_WEAK = {
    "week_order":               "MI_CLS=0.0007 — weak, user-flagged",
}

_DROP_REG_WEAK = {
    "adFlag":                   "MI_REG=0",
    "has_campaign":             "MI_REG=0",
    "group12":                  "MI_REG=0",
    "pack_size":                "MI_REG=0",
    "pack_total_size":          "MI_REG=0",
    "competitorPrice_discount": "MI_REG=0",
    "price_discount_diff":      "MI_REG=0",
    "is_greater_discount":      "MI_REG=0",
    "price_per_unit":           "MI_REG=0",
}
# availability is NZV in REG (order==1 rows)
_DROP_REG_NZV = {
    "availability":             "NZV in REG subset",
}
_DROP_REG_COND_WEAK = {
    "week_order":               "MI_REG=0",
    "availability_likelihood":  "MI_REG=0 + redundant r=-0.97",
    "day_7_qty_mean_oof":       "MI_REG=0 + redundant r=-0.97",
}

# ── Operational availability ──
_OPERATIONAL_CONDITIONAL = {
    "pid_total_events", "click_time", "basket_time", "order_time",
    "num_pid_order", "group12_order", "group34_order", "week_order",
    "pid_prob", "availability_likelihood", "day_7_likelihood",
    "day_7_qty_mean_oof", "pid_segment",
}


def _all_safe() -> list[str]:
    """Return the full safe-feature superset used as the pruning baseline."""
    return list(cfg.CLS_EXPANDED_SAFE)  # 35 features, superset


def _build_pruned_sets() -> dict:
    """Return dict with CLS_SAFE_PRUNED, CLS_FULL_PRUNED,
    REG_SAFE_PRUNED, REG_FULL_PRUNED as lists."""

    safe_all = _all_safe()
    cls_cond_all = list(cfg.CLS_CONDITIONAL)
    reg_cond_all = list(cfg.REG_CONDITIONAL)

    # --- CLS SAFE ---
    cls_safe_drop = {**_DROP_REDUNDANCY, **_DROP_NZV, **_DROP_CLS_WEAK}
    cls_safe = [f for f in safe_all if f not in cls_safe_drop]

    # --- CLS FULL ---
    cls_cond_drop = {**_DROP_REDUNDANCY, **_DROP_CLS_COND_WEAK}
    cls_cond_keep = [f for f in cls_cond_all if f not in cls_cond_drop]
    cls_full = cls_safe + cls_cond_keep

    # --- REG SAFE ---
    reg_safe_drop = {**_DROP_REDUNDANCY, **_DROP_NZV,
                     **_DROP_REG_WEAK, **_DROP_REG_NZV}
    reg_safe = [f for f in safe_all if f not in reg_safe_drop]

    # --- REG FULL ---
    reg_cond_drop = {**_DROP_REDUNDANCY, **_DROP_REG_COND_WEAK}
    reg_cond_keep = [f for f in reg_cond_all if f not in reg_cond_drop]
    reg_full = reg_safe + reg_cond_keep

    return {
        "CLS_SAFE_PRUNED": cls_safe,
        "CLS_FULL_PRUNED": cls_full,
        "REG_SAFE_PRUNED": reg_safe,
        "REG_FULL_PRUNED": reg_full,
    }


def build_candidate_reports() -> tuple[dict, pd.DataFrame]:
    """Build and export feature_set_candidates.json + .csv."""

    sets = _build_pruned_sets()

    safe_all = set(_all_safe())
    cls_cond_all = set(cfg.CLS_CONDITIONAL)
    reg_cond_all = set(cfg.REG_CONDITIONAL)
    all_features = sorted(safe_all | cls_cond_all | reg_cond_all)

    rows: list[dict] = []
    for feat in all_features:
        for stage in ("CLS", "REG"):
            spruned = f"{stage}_SAFE_PRUNED"
            fpruned = f"{stage}_FULL_PRUNED"
            in_safe = feat in sets[spruned]
            in_full = feat in sets[fpruned]

            # Determine reasons
            drop_reasons: list[str] = []
            keep_reasons: list[str] = []
            redundancy = False

            # Check all drop dicts
            if feat in _DROP_REDUNDANCY:
                drop_reasons.append(_DROP_REDUNDANCY[feat])
                redundancy = True
            if feat in _DROP_NZV:
                drop_reasons.append(_DROP_NZV[feat])
            if stage == "CLS":
                if feat in _DROP_CLS_WEAK:
                    drop_reasons.append(_DROP_CLS_WEAK[feat])
                if feat in _DROP_CLS_COND_WEAK:
                    drop_reasons.append(_DROP_CLS_COND_WEAK[feat])
            else:
                if feat in _DROP_REG_WEAK:
                    drop_reasons.append(_DROP_REG_WEAK[feat])
                if feat in _DROP_REG_NZV:
                    drop_reasons.append(_DROP_REG_NZV[feat])
                if feat in _DROP_REG_COND_WEAK:
                    drop_reasons.append(_DROP_REG_COND_WEAK[feat])

            if in_full and not drop_reasons:
                keep_reasons.append("filter signal")
            if in_full and feat in _OPERATIONAL_CONDITIONAL:
                keep_reasons.append("conditional — requires history/encoder")
            elif in_full and feat in safe_all:
                keep_reasons.append("operational — always available")

            operational = "conditional" if feat in _OPERATIONAL_CONDITIONAL else "available"

            rows.append({
                "feature_name": feat,
                "stage": stage,
                "in_safe_pruned": in_safe,
                "in_full_pruned": in_full,
                "keep_reason": "; ".join(keep_reasons) if keep_reasons else "",
                "drop_reason": "; ".join(drop_reasons) if drop_reasons else "",
                "redundancy_flag": redundancy,
                "operational_flag": operational,
            })

    df = pd.DataFrame(rows)

    # JSON export
    json_obj = {
        "description": "Pruned candidate feature sets — Round 4",
        "sets": {k: v for k, v in sets.items()},
        "counts": {k: len(v) for k, v in sets.items()},
    }
    json_path = OUT / "feature_set_candidates.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_obj, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[r4] Saved {json_path.name}")

    save_csv(df, OUT / "feature_set_candidates.csv")
    return sets, df


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers — data loading
# ═════════════════════════════════════════════════════════════════════════════

def _load(name: str) -> pd.DataFrame:
    """Load one parquet matrix used by the Round-4 selection routines."""
    p = cfg.OUTPUT_DATASETS_DIR / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def _encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode object/category columns in-place (copy first)."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            le = LabelEncoder()
            mask = df[col].notna()
            df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
            df[col] = df[col].astype(float)
    return df


def _prep_Xy(
    features: list[str],
    x_train_name: str, y_train_name: str,
    x_eval_name: str, y_eval_name: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load, subset, encode, fillna for train and evaluation split."""

    Xt_raw = _load(x_train_name)
    Xv_raw = _load(x_eval_name)
    yt = _load(y_train_name)[target_col]
    yv = _load(y_eval_name)[target_col]

    # For CLS full / REG full we need to concat base+conditional columns
    # The matrices are separate parquets, so combine them
    avail_t = set(Xt_raw.columns)
    missing = [f for f in features if f not in avail_t]
    if missing:
        # Try loading conditional matrix
        cond_name = x_train_name.replace("_base", "_conditional").replace(
            "_expanded", "_conditional")
        if cond_name != x_train_name:
            eval_cond_name = x_eval_name.replace("_base", "_conditional").replace(
                "_expanded", "_conditional")
            Xc_t = _load(cond_name)
            Xc_v = _load(eval_cond_name)
            Xt_raw = pd.concat(
                [Xt_raw, Xc_t[[c for c in missing if c in Xc_t.columns]]], axis=1)
            Xv_raw = pd.concat(
                [Xv_raw, Xc_v[[c for c in missing if c in Xc_v.columns]]], axis=1)

    feats_avail = [f for f in features if f in Xt_raw.columns]
    Xt = _encode_cats(Xt_raw[feats_avail]).fillna(-999)
    Xv = _encode_cats(Xv_raw[feats_avail]).fillna(-999)
    return Xt, yt, Xv, yv


# ═════════════════════════════════════════════════════════════════════════════
#  2.  Embedded selection
# ═════════════════════════════════════════════════════════════════════════════

def _hgb_feature_importances(model, n_features: int) -> np.ndarray:
    """Compute gain-based feature importances from HistGradientBoosting."""
    total = np.zeros(n_features)
    for stage_preds in model._predictors:
        for predictor in stage_preds:
            nodes = predictor.nodes
            mask = nodes["is_leaf"] == 0
            for node in nodes[mask]:
                total[node["feature_idx"]] += node["gain"]
    s = total.sum()
    return total / s if s > 0 else total


def _cls_metrics(y_true, y_pred, y_prob) -> dict:
    """Return the core classification metrics used in Round 4."""
    return {
        "f1": round(f1_score(y_true, y_pred), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def _reg_metrics(y_true, y_pred) -> dict:
    """Return the core regression metrics used in Round 4."""
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "median_ae": round(median_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
    }


def _subsample(X, y, n=LINEAR_SUBSAMPLE, seed=cfg.SEED):
    """Stratified subsample for linear models."""
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def run_embedded_cls(
    features: list[str],
    variant: str,
    Xt: pd.DataFrame, yt: pd.Series,
    Xv: pd.DataFrame, yv: pd.Series,
) -> list[dict]:
    """Run L1 LogReg, ElasticNet LogReg, HistGBClassifier on CLS."""
    results: list[dict] = []

    print(f"    [emb-cls] {variant} — {len(features)} features, "
          f"train={len(Xt):,}, eval={len(Xv):,}", flush=True)

    # Subsample for linear models
    Xt_s, yt_s = _subsample(Xt, yt)
    print(f"      Linear subsample: {len(Xt_s):,} rows", flush=True)

    # --- L1 Logistic Regression ---
    print("      L1 LogReg \u2026", flush=True)
    lr_l1 = LogisticRegression(
        penalty="l1", C=1.0, solver="liblinear", max_iter=1000,
        random_state=cfg.SEED,
    )
    lr_l1.fit(Xt_s, yt_s)
    coefs = lr_l1.coef_[0]
    preds = lr_l1.predict(Xv)
    proba = lr_l1.predict_proba(Xv)[:, 1]
    m = _cls_metrics(yv, preds, proba)
    for i, f in enumerate(Xt.columns):
        results.append({
            "variant": variant, "model": "L1_LogReg",
            "feature_name": f, "coefficient": round(float(coefs[i]), 6),
            "importance": round(abs(float(coefs[i])), 6),
            "selected": abs(coefs[i]) > 1e-8,
            **m,
        })

    # --- ElasticNet Logistic Regression ---
    print("      ElasticNet LogReg …", flush=True)
    lr_en = LogisticRegression(
        penalty="elasticnet", C=1.0, solver="saga", l1_ratio=0.5,
        max_iter=500, random_state=cfg.SEED, n_jobs=-1,
    )
    lr_en.fit(Xt_s, yt_s)
    coefs_en = lr_en.coef_[0]
    preds_en = lr_en.predict(Xv)
    proba_en = lr_en.predict_proba(Xv)[:, 1]
    m_en = _cls_metrics(yv, preds_en, proba_en)
    for i, f in enumerate(Xt.columns):
        results.append({
            "variant": variant, "model": "ElasticNet_LogReg",
            "feature_name": f, "coefficient": round(float(coefs_en[i]), 6),
            "importance": round(abs(float(coefs_en[i])), 6),
            "selected": abs(coefs_en[i]) > 1e-8,
            **m_en,
        })

    # --- HistGradientBoostingClassifier ---
    print("      HistGBC …", flush=True)
    hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=50, random_state=cfg.SEED,
    )
    hgb.fit(Xt, yt)
    preds_h = hgb.predict(Xv)
    proba_h = hgb.predict_proba(Xv)[:, 1]
    m_h = _cls_metrics(yv, preds_h, proba_h)
    imps = _hgb_feature_importances(hgb, Xt.shape[1])
    for i, f in enumerate(Xt.columns):
        results.append({
            "variant": variant, "model": "HistGBC",
            "feature_name": f, "coefficient": np.nan,
            "importance": round(float(imps[i]), 6),
            "selected": float(imps[i]) > 0.001,
            **m_h,
        })

    return results


def run_embedded_reg(
    features: list[str],
    variant: str,
    Xt: pd.DataFrame, yt: pd.Series,
    Xv: pd.DataFrame, yv: pd.Series,
) -> list[dict]:
    """Run Lasso, ElasticNet, HistGBRegressor on REG."""
    results: list[dict] = []

    print(f"    [emb-reg] {variant} — {len(features)} features, "
          f"train={len(Xt):,}, eval={len(Xv):,}", flush=True)

    # Subsample for linear models
    Xt_s, yt_s = _subsample(Xt, yt)
    print(f"      Linear subsample: {len(Xt_s):,} rows", flush=True)

    # --- Lasso ---
    print("      Lasso \u2026", flush=True)
    lasso = Lasso(alpha=0.01, max_iter=5000, random_state=cfg.SEED)
    lasso.fit(Xt_s, yt_s)
    preds_l = lasso.predict(Xv)
    m_l = _reg_metrics(yv, preds_l)
    for i, f in enumerate(Xt.columns):
        results.append({
            "variant": variant, "model": "Lasso",
            "feature_name": f, "coefficient": round(float(lasso.coef_[i]), 6),
            "importance": round(abs(float(lasso.coef_[i])), 6),
            "selected": abs(lasso.coef_[i]) > 1e-8,
            **m_l,
        })

    # --- ElasticNet ---
    print("      ElasticNet …", flush=True)
    enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000,
                      random_state=cfg.SEED)
    enet.fit(Xt_s, yt_s)
    preds_e = enet.predict(Xv)
    m_e = _reg_metrics(yv, preds_e)
    for i, f in enumerate(Xt.columns):
        results.append({
            "variant": variant, "model": "ElasticNet",
            "feature_name": f, "coefficient": round(float(enet.coef_[i]), 6),
            "importance": round(abs(float(enet.coef_[i])), 6),
            "selected": abs(enet.coef_[i]) > 1e-8,
            **m_e,
        })

    # --- HistGradientBoostingRegressor ---
    print("      HistGBR …", flush=True)
    hgb = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=50, random_state=cfg.SEED,
    )
    hgb.fit(Xt, yt)
    preds_h = hgb.predict(Xv)
    m_h = _reg_metrics(yv, preds_h)
    imps = _hgb_feature_importances(hgb, Xt.shape[1])
    for i, f in enumerate(Xt.columns):
        results.append({
            "variant": variant, "model": "HistGBR",
            "feature_name": f, "coefficient": np.nan,
            "importance": round(float(imps[i]), 6),
            "selected": float(imps[i]) > 0.001,
            **m_h,
        })

    return results


def run_embedded_selection(sets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run embedded selection for all 4 pruned sets."""
    cls_results: list[dict] = []
    reg_results: list[dict] = []

    for variant in ("SAFE_PRUNED", "FULL_PRUNED"):
        key = f"CLS_{variant}"
        feats = sets[key]
        Xt, yt, Xv, yv = _prep_Xy(
            feats,
            "X_train_cls_expanded", "y_train_cls",
            "X_test_cls_expanded", "y_test_cls",
            "order",
        )
        cls_results.extend(run_embedded_cls(feats, variant, Xt, yt, Xv, yv))

    for variant in ("SAFE_PRUNED", "FULL_PRUNED"):
        key = f"REG_{variant}"
        feats = sets[key]
        Xt, yt, Xv, yv = _prep_Xy(
            feats,
            "X_train_reg_expanded", "y_train_reg",
            "X_test_reg_expanded", "y_test_reg",
            "quantity",
        )
        reg_results.extend(run_embedded_reg(feats, variant, Xt, yt, Xv, yv))

    cls_df = pd.DataFrame(cls_results)
    reg_df = pd.DataFrame(reg_results)
    save_csv(cls_df, OUT / "embedded_cls_results.csv")
    save_csv(reg_df, OUT / "embedded_reg_results.csv")
    return cls_df, reg_df


def _embedded_summary(cls_df: pd.DataFrame, reg_df: pd.DataFrame,
                      sets: dict) -> str:
    """Generate embedded_selection_summary.txt."""
    L: list[str] = [
        "=" * 72,
        "  EMBEDDED SELECTION — SUMMARY  (Round 4)",
        "  Methoden: L1 LogReg, ElasticNet LogReg, HistGBC  (CLS)",
        "  Methoden: Lasso, ElasticNet, HistGBR             (REG)",
        f"  Linear-Modelle: Subsample {LINEAR_SUBSAMPLE:,} Zeilen aus Train",
        "  Tree-Modelle: voller Train",
        "  Train → Test, keine Validation, kein Random-CV",
        "=" * 72,
    ]

    for stage, df, metric_main in [
        ("CLS", cls_df, "pr_auc"),
        ("REG", reg_df, "mae"),
    ]:
        L.append(f"\n{'─' * 72}")
        L.append(f"  {stage}")
        L.append(f"{'─' * 72}")

        for variant in ("SAFE_PRUNED", "FULL_PRUNED"):
            sub = df[df["variant"] == variant]
            if sub.empty:
                continue
            n_feats = sets[f"{stage}_{variant}"]
            L.append(f"\n  ▸ {variant} ({len(n_feats)} features)")

            for model in sub["model"].unique():
                ms = sub[sub["model"] == model]
                row0 = ms.iloc[0]

                if stage == "CLS":
                    L.append(f"    {model:25s}  "
                             f"F1={row0['f1']:.4f}  "
                             f"PR-AUC={row0['pr_auc']:.4f}  "
                             f"ROC-AUC={row0['roc_auc']:.4f}")
                else:
                    L.append(f"    {model:25s}  "
                             f"MAE={row0['mae']:.4f}  "
                             f"MedAE={row0['median_ae']:.4f}  "
                             f"RMSE={row0['rmse']:.4f}")

                selected = ms[ms["selected"]]
                dropped = ms[~ms["selected"]]
                top5 = ms.nlargest(5, "importance")

                L.append(f"      Selected: {len(selected)}/{len(ms)} features")
                L.append(f"      Top-5 Importance:")
                for _, r in top5.iterrows():
                    coef_str = (f"coef={r['coefficient']:.4f}"
                                if pd.notna(r["coefficient"]) else "")
                    L.append(f"        {r['feature_name']:<30s}  "
                             f"imp={r['importance']:.6f}  {coef_str}")
                if len(dropped):
                    L.append(f"      Dropped (coef≈0): "
                             f"{', '.join(dropped['feature_name'].tolist()[:10])}"
                             + ("…" if len(dropped) > 10 else ""))

    L.append(f"\n{'═' * 72}")
    L.append("  Nächster Schritt: Family Ablation → Wrapper-Vorbereitung")
    L.append(f"{'═' * 72}")
    return "\n".join(L)


# ═════════════════════════════════════════════════════════════════════════════
#  3.  Family ablation
# ═════════════════════════════════════════════════════════════════════════════

# Map families for ablation — merge conditional sub-families
ABLATION_FAMILIES: dict[str, list[str]] = {
    "price_absolute":      cfg.FEATURE_FAMILIES["price_absolute"],
    "price_relative":      cfg.FEATURE_FAMILIES["price_relative"],
    "per_unit":            cfg.FEATURE_FAMILIES["per_unit"],
    "time":                cfg.FEATURE_FAMILIES["time"],
    "campaign_ad":         cfg.FEATURE_FAMILIES["campaign_ad"],
    "product_master":      cfg.FEATURE_FAMILIES["product_master"],
    "pack_structure":      cfg.FEATURE_FAMILIES["pack_structure"],
    "conditional_history": (
        cfg.FEATURE_FAMILIES["conditional_cumulative"]
        + cfg.FEATURE_FAMILIES["conditional_aggregation"]
    ),
    "conditional_oof_segment": (
        cfg.FEATURE_FAMILIES["conditional_oof"]
        + cfg.FEATURE_FAMILIES["conditional_segment"]
    ),
}


def _train_eval_cls(feats, Xt_full, yt, Xv_full, yv):
    """Quick HistGBC train+eval, return metrics dict."""
    if not feats:
        return {"f1": 0.0, "pr_auc": 0.0, "roc_auc": 0.0}
    Xt = Xt_full[feats]
    Xv = Xv_full[feats]
    m = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=50, random_state=cfg.SEED)
    m.fit(Xt, yt)
    pred = m.predict(Xv)
    prob = m.predict_proba(Xv)[:, 1]
    return _cls_metrics(yv, pred, prob)


def _train_eval_reg(feats, Xt_full, yt, Xv_full, yv):
    """Quick HistGBR train+eval, return metrics dict."""
    if not feats:
        return {"mae": 999.0, "median_ae": 999.0, "rmse": 999.0}
    Xt = Xt_full[feats]
    Xv = Xv_full[feats]
    m = HistGradientBoostingRegressor(
        max_iter=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=50, random_state=cfg.SEED)
    m.fit(Xt, yt)
    pred = m.predict(Xv)
    return _reg_metrics(yv, pred)


def run_family_ablation(sets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-family-out + add-one-family ablation."""
    cls_rows: list[dict] = []
    reg_rows: list[dict] = []

    full_cls = sets["CLS_FULL_PRUNED"]
    full_reg = sets["REG_FULL_PRUNED"]

    # Load and prepare all data once
    print("  [ablation] Loading data …", flush=True)
    Xt_cls, yt_cls, Xv_cls, yv_cls = _prep_Xy(
        full_cls, "X_train_cls_expanded", "y_train_cls",
        "X_test_cls_expanded", "y_test_cls", "order")
    Xt_reg, yt_reg, Xv_reg, yv_reg = _prep_Xy(
        full_reg, "X_train_reg_expanded", "y_train_reg",
        "X_test_reg_expanded", "y_test_reg", "quantity")

    # --- CLS ablation ---
    print("  [ablation] CLS baseline …", flush=True)
    base_cls = _train_eval_cls(full_cls, Xt_cls, yt_cls, Xv_cls, yv_cls)
    cls_rows.append({"experiment": "ALL_FULL_PRUNED",
                     "family": "—", "n_features": len(full_cls), **base_cls})

    for fam, members in ABLATION_FAMILIES.items():
        pruned_feats = [f for f in full_cls if f not in members]
        if len(pruned_feats) == len(full_cls):
            continue  # family not present
        print(f"    Leave-out {fam} ({len(full_cls)-len(pruned_feats)} feats) …",
              flush=True)
        m = _train_eval_cls(pruned_feats, Xt_cls, yt_cls, Xv_cls, yv_cls)
        cls_rows.append({
            "experiment": f"leave_out_{fam}",
            "family": fam,
            "n_features": len(pruned_feats),
            "f1_delta": round(m["f1"] - base_cls["f1"], 4),
            "pr_auc_delta": round(m["pr_auc"] - base_cls["pr_auc"], 4),
            **m,
        })

    # Add-one: start from time (core), add families
    core_cls = [f for f in full_cls
                if f in cfg.FEATURE_FAMILIES.get("time", [])]
    if core_cls:
        m0 = _train_eval_cls(core_cls, Xt_cls, yt_cls, Xv_cls, yv_cls)
        cls_rows.append({"experiment": "core_time_only",
                         "family": "time", "n_features": len(core_cls), **m0})
        for fam, members in ABLATION_FAMILIES.items():
            if fam == "time":
                continue
            added = core_cls + [f for f in full_cls if f in members]
            if len(added) == len(core_cls):
                continue
            print(f"    Add {fam} to core …", flush=True)
            m = _train_eval_cls(added, Xt_cls, yt_cls, Xv_cls, yv_cls)
            cls_rows.append({
                "experiment": f"core_plus_{fam}",
                "family": fam,
                "n_features": len(added),
                "f1_delta": round(m["f1"] - m0["f1"], 4),
                "pr_auc_delta": round(m["pr_auc"] - m0["pr_auc"], 4),
                **m,
            })

    # --- REG ablation ---
    print("  [ablation] REG baseline …", flush=True)
    base_reg = _train_eval_reg(full_reg, Xt_reg, yt_reg, Xv_reg, yv_reg)
    reg_rows.append({"experiment": "ALL_FULL_PRUNED",
                     "family": "—", "n_features": len(full_reg), **base_reg})

    for fam, members in ABLATION_FAMILIES.items():
        pruned_feats = [f for f in full_reg if f not in members]
        if len(pruned_feats) == len(full_reg):
            continue
        print(f"    Leave-out {fam} ({len(full_reg)-len(pruned_feats)} feats) …",
              flush=True)
        m = _train_eval_reg(pruned_feats, Xt_reg, yt_reg, Xv_reg, yv_reg)
        reg_rows.append({
            "experiment": f"leave_out_{fam}",
            "family": fam,
            "n_features": len(pruned_feats),
            "mae_delta": round(m["mae"] - base_reg["mae"], 4),
            "median_ae_delta": round(m["median_ae"] - base_reg["median_ae"], 4),
            **m,
        })

    core_reg = [f for f in full_reg
                if f in cfg.FEATURE_FAMILIES.get("time", [])]
    if core_reg:
        m0 = _train_eval_reg(core_reg, Xt_reg, yt_reg, Xv_reg, yv_reg)
        reg_rows.append({"experiment": "core_time_only",
                         "family": "time", "n_features": len(core_reg), **m0})
        for fam, members in ABLATION_FAMILIES.items():
            if fam == "time":
                continue
            added = core_reg + [f for f in full_reg if f in members]
            if len(added) == len(core_reg):
                continue
            print(f"    Add {fam} to core …", flush=True)
            m = _train_eval_reg(added, Xt_reg, yt_reg, Xv_reg, yv_reg)
            reg_rows.append({
                "experiment": f"core_plus_{fam}",
                "family": fam,
                "n_features": len(added),
                "mae_delta": round(m["mae"] - m0["mae"], 4),
                "median_ae_delta": round(m["median_ae"] - m0["median_ae"], 4),
                **m,
            })

    cls_abl = pd.DataFrame(cls_rows)
    reg_abl = pd.DataFrame(reg_rows)
    save_csv(cls_abl, OUT / "family_ablation_cls.csv")
    save_csv(reg_abl, OUT / "family_ablation_reg.csv")
    return cls_abl, reg_abl


def _ablation_summary(cls_abl: pd.DataFrame, reg_abl: pd.DataFrame) -> str:
    """Render the text summary for the family-ablation experiments."""
    L: list[str] = [
        "=" * 72,
        "  FAMILY ABLATION — SUMMARY  (Round 4)",
        "  Modell: HistGradientBoosting (max_iter=200, depth=5)",
        "  Evaluiert auf Test (Tage 71–81), keine Validation",
        "=" * 72,
    ]

    for stage, df, main_metric, delta_col in [
        ("CLS", cls_abl, "pr_auc", "pr_auc_delta"),
        ("REG", reg_abl, "mae", "mae_delta"),
    ]:
        L.append(f"\n{'─' * 72}")
        L.append(f"  {stage}")
        L.append(f"{'─' * 72}")

        base_row = df[df["experiment"].str.startswith("ALL_")]
        if len(base_row):
            br = base_row.iloc[0]
            if stage == "CLS":
                L.append(f"  Baseline (alle Features): "
                         f"F1={br['f1']:.4f}  PR-AUC={br['pr_auc']:.4f}  "
                         f"ROC-AUC={br['roc_auc']:.4f}  "
                         f"({int(br['n_features'])} features)")
            else:
                L.append(f"  Baseline (alle Features): "
                         f"MAE={br['mae']:.4f}  MedAE={br['median_ae']:.4f}  "
                         f"RMSE={br['rmse']:.4f}  "
                         f"({int(br['n_features'])} features)")

        # Leave-one-out
        loo = df[df["experiment"].str.startswith("leave_out_")]
        if len(loo):
            L.append(f"\n  Leave-One-Family-Out:")
            for _, r in loo.iterrows():
                delta = r.get(delta_col, 0.0) if pd.notna(
                    r.get(delta_col)) else 0.0
                direction = "↑" if (
                    (stage == "CLS" and delta < 0) or
                    (stage == "REG" and delta > 0)
                ) else ("↓" if delta != 0 else "=")
                effect = "hurt" if (
                    (stage == "CLS" and delta < -0.002) or
                    (stage == "REG" and delta > 0.005)
                ) else "negligible" if abs(delta) < 0.001 else "marginal"
                L.append(f"    ohne {r['family']:<25s}  "
                         f"{main_metric}={r[main_metric]:.4f}  "
                         f"Δ={delta:+.4f} {direction}  [{effect}]")

        # Add-one
        ao = df[df["experiment"].str.startswith("core_plus_")]
        core = df[df["experiment"] == "core_time_only"]
        if len(core):
            cr = core.iloc[0]
            L.append(f"\n  Add-One-Family (Kern = time):")
            if stage == "CLS":
                L.append(
                    f"    Kern:  F1={cr['f1']:.4f}  PR-AUC={cr['pr_auc']:.4f}")
            else:
                L.append(
                    f"    Kern:  MAE={cr['mae']:.4f}  MedAE={cr['median_ae']:.4f}")
        if len(ao):
            for _, r in ao.iterrows():
                delta = r.get(delta_col, 0.0) if pd.notna(
                    r.get(delta_col)) else 0.0
                L.append(f"    + {r['family']:<25s}  "
                         f"{main_metric}={r[main_metric]:.4f}  "
                         f"Δ={delta:+.4f}")

    L.append(f"\n{'═' * 72}")
    L.append("  Nächster Schritt: Wrapper auf reduziertem Set / SHAP")
    L.append(f"{'═' * 72}")
    return "\n".join(L)


# ═════════════════════════════════════════════════════════════════════════════
#  4.  Main
# ═════════════════════════════════════════════════════════════════════════════

def run_round4() -> None:
    """Execute the full Round-4 workflow and persist all generated outputs."""
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ROUND 4 — Pruned Sets / Embedded / Family Ablation")
    print("=" * 60)

    # Step 1: Pruned candidate sets
    print("\n[Step 1] Building pruned candidate sets …", flush=True)
    sets, cand_df = build_candidate_reports()
    for k, v in sets.items():
        print(f"  {k}: {len(v)} features")

    # Step 2: Embedded selection
    print("\n[Step 2] Running embedded selection …", flush=True)
    cls_emb, reg_emb = run_embedded_selection(sets)
    emb_txt = _embedded_summary(cls_emb, reg_emb, sets)
    save_text_report(emb_txt, OUT / "embedded_selection_summary.txt")
    print(emb_txt)

    # Step 3: Family ablation
    print("\n[Step 3] Running family ablation …", flush=True)
    cls_abl, reg_abl = run_family_ablation(sets)
    abl_txt = _ablation_summary(cls_abl, reg_abl)
    save_text_report(abl_txt, OUT / "family_ablation_summary.txt")
    print(abl_txt)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"ROUND 4 COMPLETE — {elapsed:.1f}s")
    print(f"Outputs in: {OUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_round4()

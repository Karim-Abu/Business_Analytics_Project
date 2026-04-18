"""
feature_selection — Filter-based feature diagnostics for CLS and REG.

Produces per-feature reports for each feature tier (base / expanded /
conditional), redundancy checks, a family-level summary, a SAFE-vs-
CONDITIONAL comparison, and a structured text summary.

Runs on already-exported train matrices from ``outputs/datasets/``.

Usage:
    cd "Feature Engineering"
    python feature_selection.py
"""

from __future__ import annotations

import argparse
import textwrap
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import config as cfg
from io_utils import save_csv, save_text_report


# ── Tuning constants ─────────────────────────────────────────────────────────

NZV_UNIQUE_RATIO = 0.01
NZV_FREQ_RATIO = 19.0
CORR_THRESHOLD = 0.95
MI_N_NEIGHBORS = 5
MI_SUBSAMPLE = 50_000


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _family_of(feat: str) -> str:
    for fam, members in cfg.FEATURE_FAMILIES.items():
        if feat in members:
            return fam
    return "unknown"


def _feat_type(s: pd.Series) -> str:
    if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    if s.dropna().nunique() <= 2:
        return "binary"
    return "numeric"


def _sub_idx(n: int, seed: int) -> np.ndarray | None:
    if n <= MI_SUBSAMPLE:
        return None
    return np.random.RandomState(seed).choice(n, MI_SUBSAMPLE, replace=False)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 20:
        return np.nan
    rho, _ = stats.spearmanr(x[m], y[m])
    return float(rho)


def _safe_chi2(x: pd.Series, y: pd.Series, n_bins: int = 10) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 20:
        return np.nan
    xv, yv = x[m], y[m]
    if pd.api.types.is_numeric_dtype(xv):
        xb = pd.qcut(xv, q=n_bins, duplicates="drop")
    else:
        xb = xv.astype(str)
    if xb.nunique() < 2:
        return np.nan
    ct = pd.crosstab(xb, yv)
    stat, _, _, _ = stats.chi2_contingency(ct)
    return float(stat)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-feature filter report
# ═══════════════════════════════════════════════════════════════════════════════

def compute_filter_report(
    X: pd.DataFrame,
    y: pd.Series,
    stage: Literal["CLS", "REG"],
) -> pd.DataFrame:
    n_rows = len(X)

    # encode categoricals for MI
    X_enc = X.copy()
    discrete_mask = np.zeros(X_enc.shape[1], dtype=bool)
    for i, col in enumerate(X_enc.columns):
        if X_enc[col].dtype == object or isinstance(
                X_enc[col].dtype, pd.CategoricalDtype):
            X_enc[col] = X_enc[col].astype("category").cat.codes
            discrete_mask[i] = True
    X_filled = X_enc.fillna(-999)

    # MI on subsample
    sub = _sub_idx(n_rows, cfg.SEED)
    Xm = X_filled.iloc[sub] if sub is not None else X_filled
    ym = y.iloc[sub] if sub is not None else y

    print(f"      MI on {len(Xm):,} rows …", flush=True)
    if stage == "CLS":
        mi_vals = mutual_info_classif(
            Xm, ym, discrete_features=discrete_mask,
            n_neighbors=MI_N_NEIGHBORS, random_state=cfg.SEED)
    else:
        mi_vals = mutual_info_regression(
            Xm, ym, discrete_features=discrete_mask,
            n_neighbors=MI_N_NEIGHBORS, random_state=cfg.SEED)

    # per-feature loop
    records: list[dict] = []
    for i, col in enumerate(X.columns):
        s = X[col]
        n_miss = int(s.isna().sum())
        miss_rate = n_miss / n_rows

        vc = s.dropna().value_counts()
        n_unique = len(vc)
        is_constant = n_unique <= 1

        unique_ratio = n_unique / n_rows if n_rows else 0
        freq_ratio = (vc.iloc[0] / vc.iloc[1]) if len(vc) >= 2 else np.inf
        nzv = (unique_ratio < NZV_UNIQUE_RATIO) and (
            freq_ratio > NZV_FREQ_RATIO or n_unique <= 1)

        mi = float(mi_vals[i])
        ftype = _feat_type(s)
        notes: list[str] = []

        rec: dict = {
            "feature_name": col,
            "dtype": str(s.dtype),
            "feature_type": ftype,
            "family": _family_of(col),
            "missing_rate": round(miss_rate, 6),
            "n_unique": n_unique,
            "is_constant": is_constant,
            "near_zero_variance": nzv,
        }

        if stage == "CLS":
            rec["mutual_information"] = round(mi, 6)
            sub_i = _sub_idx(n_rows, cfg.SEED + i)
            if sub_i is not None:
                chi2_val = _safe_chi2(s.iloc[sub_i], y.iloc[sub_i])
            else:
                chi2_val = _safe_chi2(s, y)
            rec["chi2_available"] = True
            rec["chi2_score"] = round(
                chi2_val, 4) if pd.notna(chi2_val) else np.nan
        else:
            rec["mutual_information_reg"] = round(mi, 6)
            sub_i = _sub_idx(n_rows, cfg.SEED + i)
            if sub_i is not None:
                sp = _safe_spearman(s.iloc[sub_i], y.iloc[sub_i])
            else:
                sp = _safe_spearman(s, y)
            rec["spearman_corr"] = round(sp, 6) if pd.notna(sp) else np.nan

        if is_constant:
            notes.append("CONSTANT")
        if nzv and not is_constant:
            notes.append("NEAR_ZERO_VAR")
        if miss_rate > 0.50:
            notes.append("HIGH_MISSING")
        rec["notes"] = "; ".join(notes)
        records.append(rec)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
#  Redundancy
# ═══════════════════════════════════════════════════════════════════════════════

def find_redundant_pairs(X: pd.DataFrame) -> pd.DataFrame:
    num = X.select_dtypes(include="number").columns.tolist()
    if len(num) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r"])
    corr = X[num].corr()
    pairs: list[dict] = []
    for i, a in enumerate(num):
        for j in range(i + 1, len(num)):
            b = num[j]
            r = corr.iloc[i, j]
            if abs(r) >= CORR_THRESHOLD:
                pairs.append({"feature_a": a, "feature_b": b,
                              "pearson_r": round(float(r), 6)})
    df = pd.DataFrame(pairs)
    if len(df):
        df = df.sort_values("pearson_r", key=abs, ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Family report
# ═══════════════════════════════════════════════════════════════════════════════

def build_family_report(
    cls_all: pd.DataFrame,
    reg_all: pd.DataFrame,
    cls_rd: pd.DataFrame,
    reg_rd: pd.DataFrame,
) -> pd.DataFrame:
    mi_c = "mutual_information"
    mi_r = "mutual_information_reg"

    red_feats_cls = (set(cls_rd["feature_a"]) | set(cls_rd["feature_b"])
                     if len(cls_rd) else set())
    red_feats_reg = (set(reg_rd["feature_a"]) | set(reg_rd["feature_b"])
                     if len(reg_rd) else set())

    rows: list[dict] = []
    for fam, members in cfg.FEATURE_FAMILIES.items():
        cs = cls_all[cls_all["family"] == fam]
        rs = reg_all[reg_all["family"] == fam]
        n = max(len(cs), len(rs))
        if n == 0:
            continue

        avg_miss = float(np.nanmean([
            cs["missing_rate"].mean() if len(cs) else np.nan,
            rs["missing_rate"].mean() if len(rs) else np.nan]))
        avg_mi_c = cs[mi_c].mean() if (
            len(cs) and mi_c in cs.columns) else np.nan
        avg_mi_r = rs[mi_r].mean() if (
            len(rs) and mi_r in rs.columns) else np.nan

        reds = sorted(set(m for m in members
                          if m in red_feats_cls or m in red_feats_reg))

        signals: list[str] = []
        if pd.notna(avg_mi_c) and avg_mi_c > 0.002:
            signals.append("CLS-relevant")
        if pd.notna(avg_mi_r) and avg_mi_r > 0.002:
            signals.append("REG-relevant")
        if reds:
            signals.append("has_redundancy")
        nzv_n = int(cs["near_zero_variance"].sum()) if len(cs) else 0
        if nzv_n:
            signals.append(f"{nzv_n} NZV")

        rows.append({
            "family_name": fam,
            "n_features": n,
            "avg_missing_rate": round(avg_miss, 4),
            "avg_mi_cls": round(avg_mi_c, 6) if pd.notna(avg_mi_c) else np.nan,
            "avg_mi_reg": round(avg_mi_r, 6) if pd.notna(avg_mi_r) else np.nan,
            "redundant_features": ", ".join(reds),
            "assessment": "; ".join(signals) if signals else "neutral",
            "features": ", ".join(members),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFE vs CONDITIONAL comparison
# ═══════════════════════════════════════════════════════════════════════════════

_SAFE_FAMILIES = {"price_absolute", "price_relative", "per_unit",
                  "time", "campaign_ad", "product_master", "pack_structure"}
_COND_FAMILIES = {"conditional_cumulative", "conditional_aggregation",
                  "conditional_oof", "conditional_segment"}


def _build_safe_vs_cond(cls_all: pd.DataFrame, reg_all: pd.DataFrame) -> str:
    lines: list[str] = [
        "",
        "=" * 72,
        "  SAFE vs CONDITIONAL — Vergleich",
        "=" * 72,
    ]
    for stage, df in [("CLS", cls_all), ("REG", reg_all)]:
        mi = "mutual_information" if stage == "CLS" else "mutual_information_reg"
        if mi not in df.columns or not len(df):
            continue
        safe = df[df["family"].isin(_SAFE_FAMILIES)]
        cond = df[df["family"].isin(_COND_FAMILIES)]

        lines.append(
            f"\n── {stage} ──────────────────────────────────────────")
        if len(safe):
            best_s = safe.loc[safe[mi].idxmax()]
            lines.append(f"  SAFE ({len(safe)} features)  "
                         f"avg MI={safe[mi].mean():.6f}  "
                         f"max MI={best_s[mi]:.6f} ({best_s['feature_name']})")
        if len(cond):
            best_c = cond.loc[cond[mi].idxmax()]
            lines.append(f"  COND ({len(cond)} features)  "
                         f"avg MI={cond[mi].mean():.6f}  "
                         f"max MI={best_c[mi]:.6f} ({best_c['feature_name']})")
            strong = cond[cond[mi] > 0.002]
            weak = cond[cond[mi] <= 0.002]
            if len(strong):
                lines.append(f"    Signal (MI>0.002):  "
                             f"{', '.join(strong['feature_name'])}")
            if len(weak):
                lines.append(f"    Schwach (MI≤0.002): "
                             f"{', '.join(weak['feature_name'])}")
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Text summary
# ═══════════════════════════════════════════════════════════════════════════════

def _build_summary(
    reports: dict[str, pd.DataFrame],
    redundant: dict[str, pd.DataFrame],
    family_df: pd.DataFrame,
    safe_vs_cond: str,
) -> str:
    L: list[str] = [
        "=" * 72,
        "  FEATURE SELECTION — FILTER REPORT SUMMARY",
        f"  MI subsample: {MI_SUBSAMPLE:,}  |  Corr threshold: {CORR_THRESHOLD}",
        f"  Sampling fracs in config: CLS={cfg.SAMPLE_FRAC_CLS}, "
        f"REG={cfg.SAMPLE_FRAC_REG}",
        "  HINWEIS: Feature Selection läuft auf vollen Train-Matrizen,",
        "           nicht auf gesampelten Daten.",
        "=" * 72,
    ]

    for stage in ("CLS", "REG"):
        mi = "mutual_information" if stage == "CLS" else "mutual_information_reg"
        label = "Klassifikation (order)" if stage == "CLS" else "Regression (quantity)"
        L.append(f"\n{'─' * 72}")
        L.append(f"  {stage} — {label}")
        L.append(f"{'─' * 72}")

        for tier in ("base", "expanded", "conditional"):
            key = f"{stage.lower()}_filter_{tier}"
            if key not in reports:
                continue
            st = reports[key]
            rk = f"{stage.lower()}_{tier}"
            rd = redundant.get(rk, pd.DataFrame())

            L.append(f"\n  ▸ {tier.upper()} ({len(st)} Features)")

            const = st[st["is_constant"]]
            nzv = st[st["near_zero_variance"] & ~st["is_constant"]]
            hi_miss = st[st["missing_rate"] > 0.50]

            L.append(f"    Konstant:         {len(const)}  "
                     f"({', '.join(const['feature_name']) if len(const) else '—'})")
            L.append(f"    Near-Zero-Var.:   {len(nzv)}  "
                     f"({', '.join(nzv['feature_name']) if len(nzv) else '—'})")
            L.append(f"    High Missing:     {len(hi_miss)}  "
                     f"({', '.join(hi_miss['feature_name']) if len(hi_miss) else '—'})")
            L.append(f"    Redundant Pairs:  {len(rd)}")
            for _, r in rd.head(5).iterrows():
                L.append(f"      {r['feature_a']}  ↔  {r['feature_b']}  "
                         f"(r={r['pearson_r']:.4f})")

            top5 = st.nlargest(5, mi)
            L.append("    Top-5 MI:")
            for _, r in top5.iterrows():
                L.append(f"      {r['feature_name']:<35s}  MI={r[mi]:.6f}")
            bot3 = st.nsmallest(3, mi)
            L.append("    Bottom-3 MI:")
            for _, r in bot3.iterrows():
                L.append(f"      {r['feature_name']:<35s}  MI={r[mi]:.6f}")

    # Family
    L.append(f"\n{'─' * 72}")
    L.append("  FEATURE-FAMILIEN")
    L.append(f"{'─' * 72}")
    for _, fr in family_df.iterrows():
        L.append(f"\n  {fr['family_name']}")
        L.append(f"    Features:    {fr['n_features']}")
        if pd.notna(fr["avg_mi_cls"]):
            L.append(f"    avg MI CLS:  {fr['avg_mi_cls']:.6f}")
        if pd.notna(fr["avg_mi_reg"]):
            L.append(f"    avg MI REG:  {fr['avg_mi_reg']:.6f}")
        if fr["redundant_features"]:
            L.append(f"    Redundanz:   {fr['redundant_features']}")
        L.append(f"    Bewertung:   {fr['assessment']}")

    # SAFE vs CONDITIONAL
    L.append(safe_vs_cond)

    # Recommendations
    L.append(f"\n{'═' * 72}")
    L.append("  EMPFEHLUNGEN (Filter-basiert, keine finale Entscheidung)")
    L.append(f"{'═' * 72}")
    L.append("")
    L.append("  → Basiert ausschliesslich auf Filter-Methoden.")
    L.append("  → Wrapper/Embedded (RFECV, SHAP, Permutation) folgen in Runde 4.")
    L.append("  → Finale Drop-Entscheide erst nach Wrapper-Runde.")
    L.append("")

    for stage in ("CLS", "REG"):
        mi = "mutual_information" if stage == "CLS" else "mutual_information_reg"
        all_st = pd.concat([
            v for k, v in reports.items() if k.startswith(stage.lower())
        ], ignore_index=True).drop_duplicates(subset="feature_name")
        if not len(all_st):
            continue

        L.append(f"  {stage}:")
        top10 = all_st.nlargest(10, mi)
        L.append("    Wahrscheinlich stark (Top-10 MI):")
        for _, r in top10.iterrows():
            L.append(f"      {r['feature_name']:<35s}  MI={r[mi]:.6f}")

        drops = all_st[all_st["is_constant"] | all_st["near_zero_variance"]]
        if len(drops):
            L.append("    Wahrscheinlich Drop-Kandidaten:")
            for _, r in drops.iterrows():
                L.append(f"      {r['feature_name']:<35s}  ({r['notes']})")
        else:
            L.append("    Keine offensichtlichen Drop-Kandidaten.")

        weak = all_st[all_st[mi] < 0.0005]
        if len(weak):
            L.append("    Schwaches Signal (MI<0.0005):")
            for _, r in weak.iterrows():
                L.append(f"      {r['feature_name']:<35s}  MI={r[mi]:.6f}")

        # Features for next round
        keep = all_st[~all_st["is_constant"] & ~all_st["near_zero_variance"]]
        L.append(
            f"    Für Wrapper-Runde empfohlen ({len(keep)}/{len(all_st)}):")
        L.append(f"      {', '.join(keep['feature_name'])}")
        L.append("")

    L.append("=" * 72)
    L.append("  Nächster Schritt: Wrapper-/Embedded-Runde (RFECV, SHAP)")
    L.append("=" * 72)
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════════
#  Load helper
# ═══════════════════════════════════════════════════════════════════════════════

def _load(name: str) -> pd.DataFrame:
    p = cfg.OUTPUT_DATASETS_DIR / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return pd.read_parquet(p)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_feature_selection() -> None:
    t0 = time.time()
    out = cfg.OUTPUT_FEATURE_SELECTION_DIR
    out.mkdir(parents=True, exist_ok=True)

    y_cls = _load("y_train_cls")["order"]
    y_reg = _load("y_train_reg")["quantity"]

    jobs: list[tuple[str, str, pd.Series, str]] = [
        ("cls_filter_base",        "X_train_cls_base",        y_cls, "CLS"),
        ("cls_filter_expanded",    "X_train_cls_expanded",    y_cls, "CLS"),
        ("cls_filter_conditional", "X_train_cls_conditional", y_cls, "CLS"),
        ("reg_filter_base",        "X_train_reg_base",        y_reg, "REG"),
        ("reg_filter_expanded",    "X_train_reg_expanded",    y_reg, "REG"),
        ("reg_filter_conditional", "X_train_reg_conditional", y_reg, "REG"),
    ]

    reports: dict[str, pd.DataFrame] = {}
    redundant: dict[str, pd.DataFrame] = {}

    for key, mat, target, stage in jobs:
        print(f"\n[fs] ── {key} ──", flush=True)
        try:
            X = _load(mat)
        except FileNotFoundError:
            print(f"      SKIP ({mat} not found)", flush=True)
            continue

        print(f"      {X.shape[0]:,} × {X.shape[1]}", flush=True)
        st = compute_filter_report(X, target, stage)
        reports[key] = st

        tier = key.rsplit("_", 1)[-1]
        rk = f"{stage.lower()}_{tier}"
        rd = find_redundant_pairs(X)
        redundant[rk] = rd

        csv_name = f"{stage.lower()}_filter_report_{tier}.csv"
        save_csv(st, out / csv_name)
        if len(rd):
            save_csv(rd, out / f"{stage.lower()}_{tier}_redundant_pairs.csv")

    if not reports:
        print("[fs] No matrices found.", flush=True)
        return

    # Merge for family / comparison views
    cls_all = pd.concat([v for k, v in reports.items()
                         if k.startswith("cls")],
                        ignore_index=True).drop_duplicates("feature_name")
    reg_all = pd.concat([v for k, v in reports.items()
                         if k.startswith("reg")],
                        ignore_index=True).drop_duplicates("feature_name")
    cls_rd = pd.concat([v for k, v in redundant.items()
                        if k.startswith("cls")],
                       ignore_index=True).drop_duplicates()
    reg_rd = pd.concat([v for k, v in redundant.items()
                        if k.startswith("reg")],
                       ignore_index=True).drop_duplicates()

    fam = build_family_report(cls_all, reg_all, cls_rd, reg_rd)
    save_csv(fam, out / "feature_family_report.csv")

    svc = _build_safe_vs_cond(cls_all, reg_all)

    summary = _build_summary(reports, redundant, fam, svc)
    save_text_report(summary, out / "feature_selection_summary.txt")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"FEATURE SELECTION COMPLETE — {elapsed:.1f}s", flush=True)
    print(f"Reports in: {out}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(summary)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter-based feature selection (all tiers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Analyses all available train matrices (base/expanded/conditional).
            Run main_build_datasets.py --mode safe_plus_conditional first.
        """),
    )
    parser.parse_args()
    run_feature_selection()


if __name__ == "__main__":
    main()

"""
preprocessing — data cleaning and derived columns.

Runde-1 module.  Provides the minimum interface required by
main_build_datasets.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def run_all_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps in order.

    Steps
    -----
    1. competitorPrice <= 0 → NaN, add competitorPrice_missing flag
    2. Normalise campaignIndex → campaignIndex_norm
    3. Normalise category → category_norm
    4. Normalise pharmForm → pharmForm_norm
    5. Normalise unit → unit_norm  (trim, upper, MISSING)
    6. Clean group → group_clean  (trim, upper, MISSING)
    7. Clean content → content_clean  (trim, upper, separator normalisation, MISSING)
    8. Derive quantity = revenue / price  (order==1 rows only)
    9. Derive quantity_class
    10. Flag suspicious quantity rows  (qty_suspicious)
    11. salesIndex as numeric

    Missing-Strategie nach Feldtyp
    ------------------------------
    Kategoriale Felder (campaignIndex, category, pharmForm, unit, group):
        NaN / leer → expliziter Sentinel-String ("MISSING", "NONE").
        Damit kann das Modell Missing als eigene Kategorie lernen.

    Numerische Felder (competitorPrice, salesIndex):
        NaN bleibt NaN (competitorPrice) bzw. wird mit 0 gefüllt (salesIndex).
        competitorPrice-Imputation bleibt bewusst offen (→ spätere Runde).

    Textfelder (group, content):
        Werden hier nur minimal normalisiert (trim, upper, MISSING).
        Inhaltliche Zerlegung / Parsing erfolgt erst in feature_engineering_safe.py.

    Returns the DataFrame with new / cleaned columns (in-place).
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING")
    print(f"{'='*60}")

    # ── 1. competitorPrice cleanup ───────────────────────────────────────
    invalid_cp = df["competitorPrice"] <= 0
    df.loc[invalid_cp, "competitorPrice"] = np.nan
    df["competitorPrice_missing"] = df["competitorPrice"].isna().astype(int)
    print(f"[prep] competitorPrice: {invalid_cp.sum():,} values set to NaN")

    # ── 2. campaignIndex normalisation ───────────────────────────────────
    if "campaignIndex" in df.columns:
        ci = df["campaignIndex"].astype(str).str.strip().str.upper()
        ci = ci.replace({"NAN": "NONE", "": "NONE",
                        "0": "NONE", "0.0": "NONE"})
        df["campaignIndex_norm"] = ci.where(ci.isin(["A", "B", "C"]), "NONE")
    else:
        df["campaignIndex_norm"] = "NONE"
    print(
        f"[prep] campaignIndex_norm: {df['campaignIndex_norm'].value_counts().to_dict()}")

    # ── 3. category normalisation ────────────────────────────────────────
    if "category" in df.columns:
        df["category_norm"] = (
            df["category"].astype(str).str.strip().str.upper()
            .replace({"NAN": "MISSING", "": "MISSING"})
        )
    else:
        df["category_norm"] = "MISSING"

    # ── 4. pharmForm normalisation ───────────────────────────────────────
    if "pharmForm" in df.columns:
        df["pharmForm_norm"] = (
            df["pharmForm"].astype(str).str.strip().str.upper()
            .replace({"NAN": "MISSING", "": "MISSING"})
        )
    else:
        df["pharmForm_norm"] = "MISSING"

    # ── 5. unit normalisation ────────────────────────────────────────────
    #   Nur: trim, uppercase, MISSING.  Keine inhaltliche Interpretation.
    if "unit" in df.columns:
        df["unit_norm"] = (
            df["unit"].astype(str).str.strip().str.upper()
            .replace({"NAN": "MISSING", "NONE": "MISSING", "": "MISSING"})
        )
    else:
        df["unit_norm"] = "MISSING"
    print(f"[prep] unit_norm: {df['unit_norm'].nunique()} unique values")

    # ── 6. group cleaning ────────────────────────────────────────────────
    #   Nur: trim, uppercase, MISSING, offensichtliche Whitespace-Bereinigung.
    #   KEINE semantische Umkodierung oder Zeichenentfernung —
    #   die Zeichen bleiben erhalten, weil extract_group_parts() sie braucht.
    if "group" in df.columns:
        df["group_clean"] = (
            df["group"].astype(str).str.strip().str.upper()
            .str.replace(r"\s+", " ", regex=True)  # nur Mehrfach-Leerzeichen
            .replace({"NAN": "MISSING", "NONE": "MISSING", "": "MISSING"})
        )
    else:
        df["group_clean"] = "MISSING"
    print(f"[prep] group_clean: {df['group_clean'].nunique()} unique values")

    # ── 7. content cleaning ──────────────────────────────────────────────
    #   Nur minimal: trim, uppercase, Trennzeichen vereinheitlichen, MISSING.
    #   Noch NICHT: interpretieren, multiplizieren, zerlegen, numerisch behandeln.
    #   Die eigentliche Logik bleibt in parse_content() (feature_engineering_safe.py).
    if "content" in df.columns:
        df["content_clean"] = (
            df["content"].astype(str).str.strip().str.upper()
            .str.replace("x", "X", regex=False)  # Trennzeichen einheitlich
            .replace({"NAN": "MISSING", "NONE": "MISSING", "": "MISSING"})
        )
    else:
        df["content_clean"] = "MISSING"
    print(
        f"[prep] content_clean: {df['content_clean'].nunique()} unique values")

    # ── 8. quantity derivation ───────────────────────────────────────────
    mask_order = df["order"] == 1
    df["quantity"] = np.nan
    safe_price = df["price"].where(df["price"] > 0, np.nan)
    df.loc[mask_order, "quantity"] = (
        df.loc[mask_order, "revenue"] / safe_price[mask_order]
    )
    print(f"[prep] quantity derived for {mask_order.sum():,} order=1 rows")

    # ── 9. quantity_class ────────────────────────────────────────────────
    df["quantity_class"] = pd.cut(
        df["quantity"],
        bins=[0, 1, 2, 5, 10, np.inf],
        labels=["1", "2", "3-5", "6-10", "10+"],
        right=True,
        include_lowest=True,
    ).astype(str).replace({"nan": "no_order"})

    # ── 10. qty_suspicious flag ──────────────────────────────────────────
    #
    # Logik:  Eine Zeile gilt als "suspicious" wenn order==1 UND:
    #   a) quantity < 0               → negativer Wert (Datenartefakt)
    #   b) quantity != round(quantity) → nicht ganzzahlig (Preisfehler?)
    #   c) quantity ist NaN            → order==1 aber kein gültiger Preis
    #      (z.B. price==0 → safe_price=NaN → quantity=NaN)
    #
    # Konsequenz für REG (Stage 2):
    #   Der REG-Filter verlangt qty_suspicious==0.
    #   Fehlende quantity bei order==1 führt also zu AUSSCHLUSS aus REG,
    #   nicht zu einer Imputation.  Das ist Absicht.
    #
    # order==0 Zeilen sind per Definition nicht suspicious
    #   (sie haben kein quantity und gehören nicht zu Stage 2).
    #
    q = df["quantity"]
    df["qty_suspicious"] = (
        (q < 0) | (q != q.round(0)) | (q.isna() & mask_order)
    ).astype(int)
    df.loc[~mask_order, "qty_suspicious"] = 0
    n_sus = df["qty_suspicious"].sum()
    print(f"[prep] qty_suspicious: {n_sus:,} rows flagged")

    # ── 11. salesIndex as numeric ────────────────────────────────────────
    if "salesIndex" in df.columns:
        df["salesIndex"] = pd.to_numeric(
            df["salesIndex"], errors="coerce").fillna(0)

    print(f"[prep] Done — {len(df):,} rows, {len(df.columns)} columns\n")
    return df

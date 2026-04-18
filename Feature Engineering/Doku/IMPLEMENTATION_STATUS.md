## Projektkontext

Projekt: Dynamic Pricing / Versandapotheke  
Ziel: saubere Data-Preparation- und Feature-Engineering-Pipeline für ein Two-Stage-ML-Setup.

### Stage 1 — Klassifikation

- Zielvariable: `order` (0/1)
- Frage: Wird ein Produkt an einem bestimmten Tag gekauft?

### Stage 2 — Regression

- Zielvariable: `quantity`
- Nur auf Zeilen mit `order = 1`
- `quantity` wird aus `revenue / price` abgeleitet
- `revenue` darf **nie** als Input-Feature verwendet werden

---

## Methodische Leitplanken

Diese Regeln sind fix und dürfen nicht stillschweigend geändert werden.

### Split-Logik

- Kein Random Split
- Chronologischer Split:
  - Train: Tage 26–70
  - Test: Tage 71–81
  - Validation: Tage 82–92

### Strukturbruch

- Es gibt einen dokumentierten Strukturbruch ab ca. Tag 26
- Deshalb:
  - Training startet bewusst erst ab Tag 26
  - Tage 1–25 sind **nicht** Teil des Default-Trainings
  - Tage 1–25 sind **nicht** Default-History für cumulative Features

### History-Regel

- Default für cumulative Features:
  - nur Historie **ab Tag 26**
  - also `cumulative_from_day_26_until(day-1)`

### OOF-Regel

- Kein zufälliges KFold / Random OOF
- Für target-encoded / likelihood Features ist nur erlaubt:
  - **time-aware forward OOF**
  - also forward-chaining / expanding-window / blocked time folds
- Keine Zukunftsinformation innerhalb des Train-Encodings

### Sampling-Regel

- Sampling nur für Prototyping / Entwicklungsbeschleunigung
- Finale Modellierung nicht automatisch auf gesampelten Daten
- Test und Validation bleiben ungesampelt
- CLS und REG werden getrennt gesampelt

---

## Projektstruktur

```text
dynamic_pricing/
│
├── config.py
├── main_build_datasets.py
├── main_run_audit.py
│
├── data/
│   ├── train.csv
│   └── items.csv
│
├── outputs/
│   ├── datasets/
│   ├── audit/
│   └── metadata/
│
└── src/
    ├── __init__.py
    ├── io_utils.py
    ├── preprocessing.py
    ├── split.py
    ├── pid_segment.py
    ├── feature_engineering_safe.py
    ├── feature_engineering_conditional.py     # noch offen / im Aufbau
    ├── feature_sets.py
    ├── sampling.py
    ├── audit.py
    └── validation.py

    Bereits final entschiedene Fachregeln
Verbotene Features / Leakage

Diese Features dürfen nicht als Modellfeatures verwendet werden:

revenue
lineID
click
basket
quantity für Stage 1
quantity_class für Stage 1
num_pid_order für Stage 2
pid_likelihood ist deprecated / gedroppt
order ist nur Zielvariable in Stage 1, nicht Feature
quantity ist Zielvariable in Stage 2, nicht Feature
Preis-/Missing-Regeln
competitorPrice <= 0 -> NaN
danach competitorPrice_missing = 1
rrp <= 0 darf nicht für Discount-Berechnungen verwendet werden
Preisvergleiche immer NaN-sicher

### Missing-Strategie nach Feldtyp

| Feldtyp         | Strategie                                              | Beispiel                         |
|-----------------|-------------------------------------------------------|----------------------------------|
| Kategorial      | NaN/leer → expliziter Sentinel (`MISSING` / `NONE`)  | category_norm, pharmForm_norm    |
| Numerisch       | NaN bleibt NaN oder Fallback 0                        | competitorPrice (NaN), salesIndex (0) |
| Text (Rohwert)  | Minimale Normalisierung → `MISSING`, Parsing separat  | group_clean, content_clean       |

**Offen:** competitorPrice-Imputation (bewusst offen gelassen für spätere Runde).

campaignIndex
campaignIndex_norm normalisiert
has_campaign = 1, wenn campaignIndex_norm != "NONE", sonst 0
Missingness:
item-level ca. 93.9%
modelling/merged-level ca. 83.0%

unit
unit_norm: trim, uppercase, MISSING (keine inhaltliche Interpretation)
Hilfsspalte für spätere per-unit-Features und Kategorisierung

qty_suspicious
Eine order==1 Zeile ist suspicious wenn:
  - quantity < 0 (negativer Wert → Datenartefakt)
  - quantity != round(quantity) (nicht ganzzahlig → Preisfehler?)
  - quantity ist NaN (order==1 aber price==0 oder fehlend → safe_price=NaN → quantity=NaN)
order==0 Zeilen sind per Definition nicht suspicious
Konsequenz für REG (Stage 2):
  - REG-Filter verlangt qty_suspicious==0
  - Fehlende quantity bei order==1 → AUSSCHLUSS aus REG, keine Imputation
  - Das ist Absicht: lieber weniger Trainingsdaten als verzerrte Targets

group / content

group_clean: minimale Normalisierung (trim, uppercase, Whitespace, MISSING)
  - keine semantische Umkodierung
  - Zeichen bleiben erhalten für group12/group34-Extraktion
content_clean: minimale Normalisierung (trim, uppercase, Trennzeichen→X, MISSING)
  - noch keine Interpretation, Multiplikation oder Zerlegung
  - parse_content() in feature_engineering_safe.py nutzt content_clean falls vorhanden
group12 und group34 werden explizit aus group_clean (Fallback: group) extrahiert
content wird geparst zu:
is_multipack
pack_n
pack_size
pack_total_size
Per-Unit-Regel
Per-Unit-Features basieren auf pack_total_size, nicht auf pack_size
Beispiel:
6X4X200 -> pack_n=24, pack_size=200, pack_total_size=4800
day_7_likelihood / REG-Variante
CLS: day_7_likelihood = time-aware OOF auf mean(order) pro day_7
REG: separates Feature day_7_qty_mean_oof = time-aware OOF auf mean(quantity) pro day_7
pid_prob vs pid_likelihood
pid_prob ist erlaubt
pid_likelihood ist redundant und gedroppt
Final definierte Safe-Feature-Logik
Safe Features

Diese Features gelten als leakage-frei / direkt ableitbar und sind Teil der safe_only-Pipeline:

day
day_7
day_14
day_30
adFlag
availability
price
rrp
competitorPrice
competitorPrice_missing
genericProduct
salesIndex
category_norm
pharmForm_norm
campaignIndex_norm
has_campaign
manufacturer_freq
group12
group34
is_multipack
pack_n
pack_size
pack_total_size
price_diff
price_discount
competitorPrice_discount
price_discount_diff
is_lower_price
is_discount
is_greater_discount
rrp_per_unit
price_per_unit
competitorPrice_per_unit
price_diff_bin
discount_bin
REG-Subset-Regel

Stage 2 nutzt nur Zeilen mit:

order == 1
quantity.notna()
qty_suspicious == 0 falls vorhanden
Final definierte Conditional-Feature-Logik

Diese Features sind nicht Teil der ersten sicheren Pipeline.
Sie dürfen nur leakage-sicher gebaut werden.

Cumulative ab Tag 26
pid_total_events
click_time
basket_time
order_time
num_pid_order (nur CLS erlaubt, nicht REG)

Regel:

nur Historie ab Tag 26
nur bis day - 1
kein Rückgriff auf Tage 1–25 im Default
Train-global / mapped Aggregationen
group12_order
group34_order
week_order

Hinweis:

target-basierte Aggregationen nicht blind für CV/Selection leaken lassen
für finale Nutzung sauber dokumentieren
Time-aware OOF Features
pid_prob
availability_likelihood
day_7_likelihood
day_7_qty_mean_oof

Regel:

kein Random KFold
nur time-aware forward OOF
Val/Test mit Encoder fit auf gesamtem Train
pid_segment

pid_segment hat zwei Rollen:

Sampling-Metadaten
wird direkt nach dem Split auf vollem Train (26–70) gefittet
Mapping pid -> {Head, Mid, Tail}
auf Train / Val / Test gemappt
vor Sampling verfügbar
Optionales Modellfeature
darf später als Feature verwendet werden
technisch getrennt von der Sampling-Nutzung betrachten

Segment-Regel:

Top 10% = Head

10% bis 50% = Mid

Rest = Tail
Was bereits implementiert ist
Runde 1

Implementiert:

config.py
src/io_utils.py
src/validation.py
src/preprocessing.py
src/split.py

Funktional vorhanden:

Rohdaten laden
Join train + items
Grundvalidierungen
competitorPrice-Bereinigung
Missing-Flags
Normierungen
quantity-Ableitung
quantity_class
chronologischer Split

Hinweis:

In einigen Runde-1-Dateien existieren noch sys.path.insert-Hacks
funktional okay, aber später durch saubere Imports ersetzen:
import config as cfg
from src.validation import ...
Runde 2

Implementiert:

src/pid_segment.py
src/feature_engineering_safe.py
src/feature_sets.py

Funktional vorhanden:

pid_segment fit/apply/save/load
Safe Feature Engineering
Train-fit / apply auf Val/Test für:
Binning
manufacturer frequency
Feature-Matrizen für CLS und REG
Export / Zusammenfassung
Danach implementiert / funktionsfähig

Implementiert:

src/sampling.py
src/audit.py
main_build_datasets.py für safe_only

Der safe_only-Build kann aktuell:

Rohdaten laden
joinen
validieren
preprocessing durchführen
splitten
pid_segment fitten und mappen
Safe Features erzeugen
18 Matrizen bauen und exportieren
CLS- und REG-Sampling erzeugen
Audits erzeugen
Metadata exportieren
Bereits verifiziert
Per-Unit-Features basieren auf pack_total_size
REG-Matrizen filtern auf:
order == 1
quantity.notna()
qty_suspicious == 0
safe_only läuft end-to-end
Was noch offen ist
1. src/feature_engineering_conditional.py

Noch nicht final implementiert.

Muss noch enthalten:

cumulative Features ab Tag 26
time-aware forward OOF
CLS-/REG-saubere Trennung
keine Tage 1–25 als Default-History
day_7_qty_mean_oof nur für REG
num_pid_order nicht für REG
2. safe_plus_conditional
CLI akzeptiert Modus bereits / teilweise vorbereitet
aktuell noch NotImplementedError oder Stub
muss nach Conditional-Implementierung aktiviert werden
3. competitorPrice Imputation
aktuell nur Bereinigung + Missing-Flag
keine finale Imputation implementiert
für erste Pipeline bewusst offen gelassen
4. Runde-1-Imports aufräumen
sys.path.insert entfernen
auf saubere Projektimports umstellen
5. Danach erst:
Feature-Selection
Filter Methods
Wrapper / Embedded Methods
keine finale Modellierung vor sauberem Conditional-Build

### Runde 3 — Feature Selection (Filter)

Implementiert:

- `feature_selection.py` (komplett neu geschrieben)
- `config.py`: `FEATURE_FAMILIES` dict + `OUTPUT_FEATURE_SELECTION_DIR`

Funktional vorhanden:

- 6 separate Filter-Reports: base / expanded / conditional × CLS / REG
- CLS-Schema: feature_name, dtype, feature_type, family, missing_rate, n_unique, is_constant, near_zero_variance, mutual_information, chi2_available, chi2_score, notes
- REG-Schema: feature_name, dtype, feature_type, family, missing_rate, n_unique, is_constant, near_zero_variance, mutual_information_reg, spearman_corr, notes
- Near-Zero-Variance-Erkennung (unique_ratio < 1 % + freq_ratio > 19)
- Redundanz-Check: Pearson-Korrelationspaare mit |r| ≥ 0.95 (pro Tier)
- Feature-Family-Report (aggregiert über alle Tiers)
- SAFE vs CONDITIONAL Vergleich im Textreport
- Strukturierter Textreport mit Empfehlungen
- MI-Subsampling auf 50.000 Zeilen (KNN-basierte MI)
- Kategorische Kodierung für sklearn MI (discrete_features mask)
- Standalone CLI: `python feature_selection.py`

Export nach `outputs/feature_selection/` (14 Dateien):

- `cls_filter_report_base.csv`
- `cls_filter_report_expanded.csv`
- `cls_filter_report_conditional.csv`
- `reg_filter_report_base.csv`
- `reg_filter_report_expanded.csv`
- `reg_filter_report_conditional.csv`
- `cls_base_redundant_pairs.csv`
- `cls_expanded_redundant_pairs.csv`
- `cls_conditional_redundant_pairs.csv`
- `reg_base_redundant_pairs.csv`
- `reg_expanded_redundant_pairs.csv`
- `reg_conditional_redundant_pairs.csv`
- `feature_family_report.csv`
- `feature_selection_summary.txt`

Nicht enthalten (bewusst ausgeklammert):

- Wrapper Methods (RFECV, Sequential Selection)
- Embedded Methods (SHAP, Permutation Importance)
- Finale Modellierung

---

### Runde 4 — Strukturierte Feature-Auswahl (Pruned Sets / Embedded / Ablation)

Implementiert:

- `feature_selection_r4.py` (komplett neu, ~860 Zeilen)
- Baut auf den 14 Filter-Reports aus Runde 3 auf

Funktional vorhanden:

**Step 1 — Pruned Candidate Sets**
- 4 bereinigte Feature-Sets aus Filter-Ergebnissen:
  - CLS_SAFE_PRUNED: 18 Features (aus safe Features, MI > 0, keine Redundanz)
  - CLS_FULL_PRUNED: 28 Features (safe + conditional, MI > 0, keine Redundanz)
  - REG_SAFE_PRUNED: 18 Features
  - REG_FULL_PRUNED: 26 Features
- Redundanz-Bereinigung: is_lower_price, rrp, rrp_per_unit, competitorPrice_per_unit, num_pid_order gedroppt
- NZV-Drop: competitorPrice_missing, pack_n, is_discount
- Weak-Feature-Drop: Features mit MI ≈ 0.000 separat für CLS und REG
- JSON + CSV Export mit drop_reason, keep_reason, redundancy_flag, operational_flag

**Step 2 — Embedded Selection**
- CLS: L1 LogReg (liblinear), ElasticNet LogReg (saga), HistGradientBoostingClassifier
- REG: Lasso, ElasticNet, HistGradientBoostingRegressor
- Linear-Modelle: Subsample 200.000 Zeilen (SAGA auf 1.5M nicht praktikabel)
- Tree-Modelle: voller Train (HistGBC 300 iter, depth 6)
- `_hgb_feature_importances()`: custom gain-basierte Importance-Berechnung
  (sklearn 1.7.2 hat kein `feature_importances_` Attribut auf HistGradientBoosting)
- Evaluierung auf Val (Train→Val, kein Test, kein Random-CV)

**Step 3 — Family Ablation**
- 9 zusammengefasste Familien (conditional_cumulative + conditional_aggregation → conditional_history; conditional_oof + conditional_segment → conditional_oof_segment)
- Leave-One-Family-Out + Add-One-Family (Kern = time)
- HistGBC/HistGBR (200 iter, depth 5)
- Evaluierung auf Val

Zentrale Ergebnisse:

- CLS: conditional_oof_segment dominiert (PR-AUC +0.167 Add-One-Δ), conditional_history ebenfalls stark (+0.164)
- CLS SAFE_PRUNED (ohne conditional) hat nur PR-AUC ≈ 0.25 → kaum diskriminativ
- CLS FULL_PRUNED mit conditional Features: PR-AUC ≈ 0.42, ROC-AUC ≈ 0.72
- CLS HistGBC Top-Features: pid_prob, order_time, click_time, basket_time
- REG: time-Family am stärksten (ohne time → MAE sinkt um 0.022)
- REG HistGBR Top-Features: pid_prob, group12_order, group34_order, day, day_30

Export nach `outputs/feature_selection/` (8 neue Dateien):

- `feature_set_candidates.json`
- `feature_set_candidates.csv` (96 Zeilen)
- `embedded_cls_results.csv` (138 Zeilen)
- `embedded_reg_results.csv` (132 Zeilen)
- `embedded_selection_summary.txt`
- `family_ablation_cls.csv` (17 Zeilen)
- `family_ablation_reg.csv` (17 Zeilen)
- `family_ablation_summary.txt`

Technische Besonderheiten:

- sklearn 1.7.2 HistGradientBoosting hat kein `feature_importances_` → Berechnung aus `model._predictors[stage][tree].nodes["gain"]` mit Mask `nodes["is_leaf"] == 0` (uint8, nicht bool)
- SAGA Solver auf >200k Zeilen extrem langsam → Subsample für lineare Modelle
- ElasticNet ConvergenceWarning erwartet (500 iter, Feature-Selection-Kontext akzeptabel)

Nicht enthalten (bewusst ausgeklammert):

- SHAP / Permutation Importance (explizit für spätere Runde vorgesehen)
- RFECV / Sequential Wrapper auf >30 Features
- Test-Set-Evaluation
- Finale Modellierung
Sampling-Logik
CLS-Sampling

Population:

volles Train-Set (26–70)

Strata:

week_block
order
pid_segment
REG-Sampling

Population:

nur Stage-2-kompatible Zeilen
also dieselbe REG-Maske wie in build_safe_feature_matrices()

Strata:

week_block
quantity_class
pid_segment

Regel:

kleine Strata nicht künstlich aufblähen
Validation/Test ungesampelt
Nächster konkreter Arbeitsauftrag für Copilot
Jetzt als Nächstes bauen
src/feature_engineering_conditional.py
main_build_datasets.py für safe_plus_conditional
kleinen Konsistenzcheck:
sample_reg() muss dieselbe REG-Maske verwenden wie build_safe_feature_matrices()
Dabei strikt beachten
keine Architekturänderung
keine neuen Module
kein Random KFold
keine Verwendung von Tagen 1–25 als Default-History
keine Modellierung
keine zusätzlichen Features ausserhalb der freigegebenen Logik
Was Copilot NICHT tun soll
kein automatisches Refactoring der gesamten Codebasis
keine Modellierung hinzufügen
keine zufälligen CV-Folds für OOF-Encoding
keine stillschweigende Imputation ausser explizit gewünscht
keine Änderung des Splits
keine Nutzung verbotener Features
pid_likelihood nicht wieder einführen
revenue nie als Modellfeature verwenden
```

### Runde 5 — Orange CSV Export (Handoff)

**Ziel:** Den Python-/Notebook-Teil abschliessen und modellierfertige CSV-Dateien für Orange exportieren. Keine Modellierung in Python.

**Architektur-Entscheidung:**

- Python = Data Preparation, Feature Engineering, Sampling, Export
- Orange = Modellierung, Modellvergleich, Evaluation

**Finale Feature-Sets (aus Runde 4):**

- `CLS_FINAL` = CLS_FULL_PRUNED (28 Features): day, day_7, day_14, day_30, adFlag, availability, price, competitorPrice, salesIndex, category_norm, pharmForm_norm, has_campaign, group12, group34, is_greater_discount, price_per_unit, price_diff_bin, discount_bin, pid_total_events, click_time, basket_time, order_time, group12_order, group34_order, pid_prob, availability_likelihood, day_7_likelihood, pid_segment
- `REG_FINAL` = REG_FULL_PRUNED (26 Features): day, day_7, day_14, day_30, price, competitorPrice, genericProduct, salesIndex, category_norm, pharmForm_norm, campaignIndex_norm, manufacturer_freq, group34, is_multipack, price_diff, price_discount, price_diff_bin, discount_bin, pid_total_events, click_time, basket_time, order_time, group12_order, group34_order, pid_prob, pid_segment

**Neue/geänderte Dateien:**

- `config.py` — `CLS_FINAL`, `REG_FINAL`, `OUTPUT_ORANGE_EXPORTS_DIR`, `CATEGORICAL_AS_STRING` hinzugefügt
- `feature_sets.py` — `CLS_FINAL` und `REG_FINAL` in `_REGISTRY` aufgenommen
- `orange_export.py` — Neues Modul: baut, validiert und schreibt 8 CSVs + Manifest
- `io_utils.py` — `OUTPUT_ORANGE_EXPORTS_DIR` in `ensure_output_dirs()` ergänzt
- `main_build_datasets.py` — Orange-Export als Schritt 14 in `run_safe_plus_conditional()` integriert

**Export-Konventionen:**

- Zielspalte als letzte Spalte (CLS: `order`, REG: `quantity`)
- Kategorische Spalten (`pid_segment`, `category_norm`, `pharmForm_norm`, `campaignIndex_norm`) explizit als String exportiert
- Forbidden-Feature-Check vor Export
- Spaltenreihenfolge identisch über alle Splits einer Stage

**Exportierte Dateien** (`outputs/orange_exports/`):

| Datei                | Stage | Split        | Rows       | Features      |
| -------------------- | ----- | ------------ | ---------- | ------------- |
| cls_train_full.csv   | CLS   | train_full   | 1,521,260  | 28 + order    |
| cls_train_sample.csv | CLS   | train_sample | 456,377    | 28 + order    |
| cls_test.csv         | CLS   | test         | 349,447    | 28 + order    |
| cls_val.csv          | CLS   | validation   | 363,483    | 28 + order    |
| reg_train_full.csv   | REG   | train_full   | 332,546    | 26 + quantity |
| reg_train_sample.csv | REG   | train_sample | 99,770     | 26 + quantity |
| reg_test.csv         | REG   | test         | 82,403     | 26 + quantity |
| reg_val.csv          | REG   | validation   | 87,746     | 26 + quantity |
| export_manifest.csv  | —     | —            | 8 Einträge | Metadaten     |

**Manifest-Felder:** file_name, stage, split, n_rows, n_features, target_name, feature_set_name, sampling_used, sampling_frac, build_mode, source_pipeline, created_at

**Pipeline-Lauf:** `python main_build_datasets.py --mode safe_plus_conditional` — 141.5s, alle Validierungen bestanden.

### Runde 6 — Orange Export Delta-Hardening

**Ziel:** Bestehenden Orange-Exportpfad gezielt härten, ohne Feature-/Split-/Sampling-Logik zu ändern.

**Änderungen:**

- `validation.py` — 3 neue Check-Funktionen:
  - `assert_no_duplicate_features()` — verhindert doppelte Features in Final-Sets
  - `assert_cross_split_columns()` — stellt identische Spaltenreihenfolge über alle Splits einer Stage sicher
  - `assert_reg_stage2_only()` — harter Check dass REG-Frames nur Stage-2-Zeilen enthalten (order==1, quantity.notna(), qty_suspicious==0)

- `feature_sets.py` — Module-Level Duplikat-Guard für `CLS_FINAL` und `REG_FINAL` beim Import

- `orange_export.py` — 5 Härtungen:
  1. `build_mode` wird gegen `cfg.BUILD_MODES` validiert (ValueError bei ungültigem Wert)
  2. Duplikat-Check für Feature-Listen vor Export
  3. Cross-Split Spalten-Konsistenz-Check für alle 4 Splits pro Stage
  4. Explizite REG Stage-2 Validierung via `assert_reg_stage2_only()` auf Quell-DataFrames
  5. `reg_mask_applied` als neues Manifest-Feld (True für REG, False für CLS)

- Manifest `feature_set_name` korrigiert: `"CLS_FINAL"` / `"REG_FINAL"` statt `"CLS_FULL_PRUNED"` / `"REG_FULL_PRUNED"`

**Manifest-Felder (13):** file_name, stage, split, n_rows, n_features, target_name, feature_set_name, sampling_used, sampling_frac, build_mode, source_pipeline, reg_mask_applied, created_at

**Manuelle Orange-Checkliste:**

- [ ] `category_norm`, `campaignIndex_norm`, `pharmForm_norm`, `pid_segment` werden als diskrete Attribute erkannt (nicht numerisch)
- [ ] Zielspalte (`order` / `quantity`) ist letzte Spalte im CSV
- [ ] Train-Sample hat weniger Zeilen als Train-Full
- [ ] REG-Dateien enthalten nur Zeilen mit order=1

### Runde 7 — Orange Modeling Readiness (E–G)

**Ziel:** Orange-Import technisch absichern, konservative Gegenvarianten bereitstellen, formale Modellierungsfreigabe erteilen.

**Änderungen:**

- `config.py`:
  - `CLS_FINAL_NO_GROUPS` = `CLS_FINAL` ohne `group12`, `group34` (26 Features)
  - `REG_FINAL_NO_GROUP34` = `REG_FINAL` ohne `group34` (25 Features)
  - `group12` zu `CATEGORICAL_AS_STRING` hinzugefügt
  - `ORANGE_DISCRETE_PREFIX = {"category_norm": "C_"}` — export-only Präfix für numerisch aussehende Kategorien

- `feature_sets.py`:
  - `CLS_FINAL_NO_GROUPS` und `REG_FINAL_NO_GROUP34` in `_REGISTRY` aufgenommen
  - Duplikat-Guard auf alle vier Final-Sets erweitert

- `orange_export.py`:
  - `_build_orange_df()`: wendet `ORANGE_DISCRETE_PREFIX` nach String-Cast an
  - `export_orange_csvs()`: erzeugt 6 zusätzliche Varianten-CSVs (train_full/val/test × 2 Varianten)
  - Manifest enthält jetzt 14 Einträge (8 Base + 6 Varianten)

- `Doku/ORANGE_MODELING_CHECKLIST.md` — neue manuelle Prüfliste für Orange-Import

**Keine Änderungen an:** Preprocessing, Splits, Sampling, Feature Engineering, bestehende Feature-Listen.

**Exportierte Dateien** (`outputs/orange_exports/`, 14 + Manifest):

| Datei                         | Stage | Feature-Set          | Split        | Features |
| ----------------------------- | ----- | -------------------- | ------------ | -------- |
| cls_train_full.csv            | CLS   | CLS_FINAL            | train_full   | 28       |
| cls_train_sample.csv          | CLS   | CLS_FINAL            | train_sample | 28       |
| cls_val.csv                   | CLS   | CLS_FINAL            | val          | 28       |
| cls_test.csv                  | CLS   | CLS_FINAL            | test         | 28       |
| cls_train_full_no_groups.csv  | CLS   | CLS_FINAL_NO_GROUPS  | train_full   | 26       |
| cls_val_no_groups.csv         | CLS   | CLS_FINAL_NO_GROUPS  | val          | 26       |
| cls_test_no_groups.csv        | CLS   | CLS_FINAL_NO_GROUPS  | test         | 26       |
| reg_train_full.csv            | REG   | REG_FINAL            | train_full   | 26       |
| reg_train_sample.csv          | REG   | REG_FINAL            | train_sample | 26       |
| reg_val.csv                   | REG   | REG_FINAL            | val          | 26       |
| reg_test.csv                  | REG   | REG_FINAL            | test         | 26       |
| reg_train_full_no_group34.csv | REG   | REG_FINAL_NO_GROUP34 | train_full   | 25       |
| reg_val_no_group34.csv        | REG   | REG_FINAL_NO_GROUP34 | val          | 25       |
| reg_test_no_group34.csv       | REG   | REG_FINAL_NO_GROUP34 | test         | 25       |

**Empfohlene Orange-Startreihenfolge:**

1. Schneller Vergleich mit `train_sample` + `val` + `test` (CLS_FINAL / REG_FINAL)
2. Sensitivitätsvergleich: CLS_FINAL vs. CLS_FINAL_NO_GROUPS auf train_full/val/test
3. Analog: REG_FINAL vs. REG_FINAL_NO_GROUP34

**READY_FOR_ORANGE_MODELING = YES**

# Workspace-Navigation

Diese Datei enthält weiterhin die ausführlichen Data-Understanding-Notizen.

Für den aktuellen Arbeitsstand der Pipeline sind zuerst diese Dateien relevant:

- `Feature Engineering/README.md` für die Ordnerstruktur und die aktiven Pfade
- `Feature Engineering/Doku/PROJEKT_ZUSAMMENFASSUNG.md` für den aktuellen Projektstand
- `Feature Engineering/Doku/IMPLEMENTATION_STATUS.md` für die vollständige technische Historie
- `Feature Engineering/Doku/ORANGE_MODELING_CHECKLIST.md` für die letzten manuellen Orange-Checks

Der Rest dieser Datei dokumentiert die frühere CRISP-DM-Phase zur Datenbasis und bleibt als Hintergrunddokument erhalten.

# Data Understanding (CRISP-DM) – Dynamic Pricing Mail-Order Pharmacy

## Business Context

Eine anonymisierte Versandapotheke betreibt **Dynamic Pricing** in ihrem Online-Shop: Preise werden täglich automatisch auf Produktebene angepasst (keine Preis-Personalisierung pro Kunde). Die Apotheke zeichnet für jedes User-Event (Click, Basket, Order) den eigenen Preis (`price`), den Wettbewerberpreis (`competitorPrice`), den Umsatz (`revenue`) sowie zeitinvariante Produktattribute auf. Ziel des Data-Mining-Projekts ist ein Modell, das pro Zeile (Event/Session) vorhersagt, ob ein Kauf stattfindet und – falls ja – die Menge bzw. den Umsatz ableitet. Der Forecast soll auf den Folgemonat generalisieren.

Die Daten liegen als strukturierte ASCII-Textdateien vor, Felder durch `|` getrennt, `.` als Dezimaltrennzeichen. Es gibt **keine Customer-ID**: Jede Zeile ist eine einzelne User-Aktion zu einem Produkt an einem Tag – es ist nicht möglich, Zeilen einem bestimmten Kunden zuzuordnen oder Kauf-Sessions zu rekonstruieren.

---

## 1) Collect Initial Data

### 1.1 Dateien

| Datei       | Beschreibung                                          | Rolle                    |
| ----------- | ----------------------------------------------------- | ------------------------ |
| `items.csv` | Zeitinvariante Produktstammdaten (Dimensionstabelle)  | 1 Zeile pro Produkt      |
| `train.csv` | Zeitvariable Event-/Transaktionsdaten (Faktentabelle) | Viele Zeilen pro Produkt |

**Join-Key:** `pid` (many-to-one: viele train-Zeilen → ein items-Eintrag).

### 1.2 Laden

```python
import pandas as pd
import numpy as np

TRAIN_PATH = "train.csv"
ITEMS_PATH = "items.csv"

train = pd.read_csv(TRAIN_PATH, sep="|", low_memory=False)
items = pd.read_csv(ITEMS_PATH, sep="|", low_memory=False)

print("train shape:", train.shape)
print("items shape:", items.shape)
print(train.head())
print(items.head())
```

**Hinweise:**

- `low_memory=False` vermeidet mixed-dtype-Warnungen bei grossen Dateien.
- Encoding: ASCII; Dezimaltrennzeichen `.` → kein `decimal`-Parameter nötig.
- Dtype-Handling: `competitorPrice`, `category`, `campaignIndex`, `rrp` können `NaN` enthalten → pandas liest sie als `float64` bzw. `object`.

---

## 2) Describe Data (Surface Properties)

### 2.1 Files & Schema

**train.csv – Spalten**

| Spalte            | dtype   | Skalenniveau     | Begründung                                                                                     |
| ----------------- | ------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| `lineID`          | int64   | Nominal (Key)    | Eindeutiger Zeilenschlüssel, keine numerische Ordnung                                          |
| `day`             | int64   | Intervall        | Tag 1–92 im Beobachtungszeitraum; kein echter Nullpunkt, kein Kalenderdatum                    |
| `pid`             | int64   | Nominal (FK)     | Produkt-ID, Join-Key zu items.csv                                                              |
| `adFlag`          | int64   | Nominal (binär)  | 1 = Produkt war an diesem Tag in Marketingkampagne beworben, 0 = nicht                         |
| `availability`    | int64   | Ordinal-Kandidat | Werte 1–4; Bedeutung nicht dokumentiert → empirisch prüfen, ob Ordinal oder Nominal sinnvoller |
| `competitorPrice` | float64 | Ratio            | Wettbewerberpreis; darf `NaN` sein (kein Wettbewerber bekannt)                                 |
| `click`           | int64   | Nominal (binär)  | 1 = Click-Event, sonst 0                                                                       |
| `basket`          | int64   | Nominal (binär)  | 1 = Warenkorb-Event, sonst 0                                                                   |
| `order`           | int64   | Nominal (binär)  | 1 = Kauf-Event, sonst 0                                                                        |
| `price`           | float64 | Ratio            | Eigener Preis an diesem Tag                                                                    |
| `revenue`         | float64 | Ratio            | Umsatz; ⚠ **Leakage-Risiko** – darf nicht als Feature verwendet werden                         |

> **Constraint:** Pro Zeile ist exakt eine der drei Aktionsspalten (`click`, `basket`, `order`) = 1, die anderen = 0.

**items.csv – Spalten**

| Spalte           | dtype       | Skalenniveau    | Begründung                                                                                     |
| ---------------- | ----------- | --------------- | ---------------------------------------------------------------------------------------------- |
| `pid`            | int64       | Nominal (PK)    | Primärschlüssel, eindeutig pro Produkt                                                         |
| `manufacturer`   | int64/float | Nominal         | Anonymisierte Hersteller-ID; hohe Kardinalität                                                 |
| `group`          | object      | Nominal         | Produktgruppen-Code (z. B. `2FOI`, `10OJ03JS`); alphanumerisch                                 |
| `content`        | object      | Nominal         | Packungsangabe; gemischt numerisch + Multipacks (`10X1`, `6X4X200`)                            |
| `unit`           | object      | Nominal         | Einheit: ST, G, ML, KG, L, CM, M, P                                                            |
| `pharmForm`      | object      | Nominal         | Darreichungsform (TAB, CRE, KAP, GLO, TRO …); ⚠ mögliche Inkonsistenz (Gross-/Kleinschreibung) |
| `genericProduct` | int64       | Nominal (binär) | 0/1-Flag: Generikum ja/nein                                                                    |
| `salesIndex`     | int64       | Nominal         | 4 Ausprägungen (40, 44, 52, 53); Bedeutung unklar                                              |
| `category`       | float64     | Nominal         | Numerischer Kategorie-Code; teilweise `NaN`                                                    |
| `campaignIndex`  | object      | Nominal         | A / B / C; ~94 % Missing → Missingness ist strukturell (nur Kampagnenprodukte haben Label)     |
| `rrp`            | float64     | Ratio           | Recommended Retail Price (UVP); Basis für Rabattberechnung                                     |

**Beziehung train ↔ items:**

- `items.pid` muss eindeutig sein (Dimensionstabelle) – **wird per Code verifiziert**.
- `items` kann Produkte enthalten, die im Beobachtungszeitraum keine Events hatten (orphan items).
- `train` sollte keine PIDs enthalten, die nicht in `items` vorkommen (orphan train PIDs wären ein Datenqualitätsproblem).

### 2.2 Sizes

> Alle Zahlen werden vom Code-Block berechnet und hier als Referenz dokumentiert.

| Metrik                                          | Wert                                                 |
| ----------------------------------------------- | ---------------------------------------------------- |
| train Rows × Cols                               | 2,756,003 × 11                                       |
| items Rows × Cols                               | 22,035 × 11                                          |
| Merged Rows × Cols                              | 2,756,003 × 21                                       |
| Unique PIDs in train                            | 21,928                                               |
| Unique PIDs in items                            | 22,035 (✓ keine Duplikate)                           |
| Unique Days                                     | 92 (day 1–92)                                        |
| PIDs in items ohne Events (orphan items)        | 107 (Produkte im Katalog ohne Aktivität im Zeitraum) |
| PIDs in train ohne items-Eintrag (orphan train) | 0 (✓ alle train PIDs in items vorhanden)             |

**Events per PID (train):**

| Statistik     | Wert   |
| ------------- | ------ |
| count         | 21,928 |
| mean          | 125.68 |
| std           | 529.28 |
| min           | 1      |
| 50 % (Median) | 32     |
| 90 %          | 260    |
| 99 %          | 1,506  |
| max           | 53,785 |

**Konzentration (Long Tail):**

| Segment                  | Anteil an allen Events |
| ------------------------ | ---------------------- |
| Top 1 % der Produkte     | 25.63 %                |
| Top 10 % der Produkte    | 64.74 %                |
| Bottom 50 % der Produkte | 5.34 %                 |

**Aktionsverteilung (Action Distribution):**

| Aktion | Count     | Rate    |
| ------ | --------- | ------- |
| Click  | 1,582,827 | 57.43 % |
| Basket | 468,086   | 16.98 % |
| Order  | 705,090   | 25.58 % |

Verlauf über Day → Plot 7 im Code-Block.

**Missing Values per Spalte (merged):**

| Spalte            | Missing   | Missing % |
| ----------------- | --------- | --------- |
| `competitorPrice` | 100,687   | 3.65 %    |
| `pharmForm`       | 194,124   | 7.04 %    |
| `category`        | 87,394    | 3.17 %    |
| `campaignIndex`   | 2,287,968 | 83.02 %   |

Alle anderen Spalten: 0 % Missing.

### 2.3 Basic Integrity Rules

Folgende Prüfungen werden vom Code-Block ausgeführt und als PASS/FAIL gemeldet:

| Check                                      | Was wird geprüft (einfach)                              | Regel / Logik                            | Ergebnis                |
| ------------------------------------------ | ------------------------------------------------------- | ---------------------------------------- | ----------------------- |
| Genau eine Aktion pro Zeile                | Pro Zeile ist genau **eine** Aktion gesetzt             | `click + basket + order == 1`            | ✓ PASS                  |
| Preis gültig                               | Preis ist immer > 0                                     | `price > 0`                              | ✓ PASS                  |
| Wettbewerberpreis gültig (falls vorhanden) | Falls `competitorPrice` vorhanden ist, muss er > 0 sein | `competitorPrice > 0` wenn nicht missing | ✗ FAIL (976 violations) |
| Revenue nicht negativ                      | Umsatz ist nie negativ                                  | `revenue >= 0`                           | ✓ PASS                  |
| Revenue passend bei Kauf                   | Wenn bestellt wurde, muss Umsatz > 0 sein               | `revenue > 0` wenn `order == 1`          | ✓ PASS                  |
| Revenue passend bei Nicht-Kauf             | Wenn **nicht** bestellt wurde, muss Umsatz = 0 sein     | `revenue == 0` wenn `order == 0`         | ✓ PASS                  |
| lineID eindeutig                           | Jede Zeile hat eine eindeutige ID                       | keine Duplikate in `lineID`              | ✓ PASS                  |
| items.pid eindeutig                        | Jede PID in `items` kommt nur einmal vor                | keine Duplikate in `items.pid`           | ✓ PASS                  |
| Keine verwaisten train-PIDs                | Jede PID in `train` existiert in `items`                | `train.pid ⊆ items.pid`                  | ✓ PASS                  |
| Orphan items (Info)                        | Produkte in `items` ohne Events in `train`              | `items.pid \ train.pid`                  | ℹ 107                   |

> **Finding:** 976 Zeilen haben `competitorPrice ≤ 0` → Integrity-Verletzung, muss in Data Preparation behandelt werden (als Missing setzen oder untersuchen).

---

## 3) Explore Data (Relationships & Patterns)

### 3.1 Target Definition & Leakage Audit

**Revenue-Logik:**

- Hypothese: `revenue > 0` nur bei `order == 1`.
- **Ergebnis: 0 Violations** — Revenue > 0 kommt ausschliesslich bei `order == 1` vor. ✓
- Umgekehrt: 0 Zeilen mit `order == 1` und `revenue == 0`. ✓

**Quantity-Ableitung:**

- `q_raw = revenue / price` für alle Zeilen mit `order == 1`.
- **Ganzzahl-Regel:** `quantity = round(q_raw)` **nur wenn** `abs(q_raw − round(q_raw)) ≤ 0.01`.
- Zeilen, die diese Regel nicht erfüllen → `quantity = NaN` (dokumentiert als "non-integer revenue events").
- **Ergebnis:** 705,090 von 705,090 Orders (100.00 %) erfüllen die Ganzzahl-Regel. 0 Non-Integer-Fälle.
- Quantity-Statistik: mean=1.38, std=1.58, min=1, median=1, max=306. Häufigste Werte: 1 (562,284×), 2 (94,381×), 3 (24,038×).

**Target-Setup (datenbasierte Entscheidung):**

1. **Binäres Target:** `order` (0/1) → Klassifikation (Kaufwahrscheinlichkeit).
2. **Mengen-Target:** `quantity` (nur für `order == 1`) → Regression.
3. **Begründung:** Near-integer Rate = 100.00 % (>> 95 %) → Quantity ist zuverlässig ableitbar. Formel `revenue = price × quantity` ist deterministisch in diesen Daten.

**Leakage-Blacklist:**

| Feature              | Grund für Ausschluss                                                   |
| -------------------- | ---------------------------------------------------------------------- |
| `revenue`            | Kodiert das Target direkt (`revenue = price × quantity` bei Kauf)      |
| `lineID`             | Reine ID, kein informativer Inhalt                                     |
| `q_raw` / `quantity` | Nur zur Inferenzzeit nicht verfügbar; als Target ok, nicht als Feature |

### 3.2 Key Driver Hypotheses (mit Tests)

Alle Hypothesen werden im Code-Block empirisch geprüft:

**Preis vs. Wettbewerber:**

- `price_diff = price − competitorPrice`
- `price_ratio = price / competitorPrice`
- Hypothese: Je teurer relativ zum Wettbewerb, desto niedriger die Order-Rate.
- Test: Order-Rate nach gebinntem `price_diff`.

**Rabatt vs. UVP:**

- `discount_vs_rrp = (rrp − price) / rrp` (nach Join mit items)
- Hypothese: Höherer Rabatt → höhere Kaufwahrscheinlichkeit.

**Marketing (adFlag):**

- Order-Rate bei `adFlag = 1` vs. `adFlag = 0`.
- Hypothese: Beworbene Produkte haben höhere Kaufrate.

**Verfügbarkeit (availability):**

- Order-Rate pro Ausprägung (1–4).
- Hypothese: Schlechtere Verfügbarkeit senkt Orders (Codes nicht dokumentiert → empirisch prüfen).
- Interpretation: vorsichtig, da Codierung unbekannt.

**Segmentierung:**

- Order-Rate nach `genericProduct`, `pharmForm`, `category`, `salesIndex` (Top/Bottom).
- Generika-Hypothese: Generika reagieren preissensibler.

**Zeit:**

- Order-Rate über `day` (Trendlinie).
- `day % 7` als Wochentag-Proxy. **Achtung:** Kein echtes Kalenderdatum vorhanden; `day % 7` ist nur ein struktureller Proxy und darf nicht als „echte Wochentag-Saisonalität" interpretiert werden.

**Aktionsverteilung:**

- Click / Basket / Order Anteile insgesamt (Tabelle) und über `day` (Linienplot).

### 3.3 Plots (mindestens 6)

| #   | Plot                                                   | Typ                           |
| --- | ------------------------------------------------------ | ----------------------------- |
| 1   | Verteilung `price` und `competitorPrice`               | Histogramm (log-Skala)        |
| 2   | Verteilung `discount_vs_rrp`                           | Histogramm                    |
| 3   | Order-Rate vs. gebinnter `price_diff`                  | Balkendiagramm                |
| 4   | Order-Rate nach `availability` und `adFlag`            | Gruppiertes Balkendiagramm    |
| 5   | Long-Tail: Events per PID (sortiert, absteigend)       | Linie (log-y)                 |
| 6   | Quantity-Verteilung (`order == 1`)                     | Histogramm (ganzzahlige Bins) |
| 7   | Order-Rate über `day` + Action Distribution über `day` | Linienplot                    |

---

## 4) Verify Data Quality (Issues, Risks, Opportunities)

### 4.1 Missing Values

| Spalte            | Missing (merged) | Missing % | Empfohlene Behandlung                                 | Begründung                                               |
| ----------------- | ---------------- | --------- | ----------------------------------------------------- | -------------------------------------------------------- |
| `competitorPrice` | 100,687          | 3.65 %    | Imputation (Median pro PID/Gruppe) + **Missing-Flag** | Missingness ist informativ: kein bekannter Wettbewerber  |
| `pharmForm`       | 194,124          | 7.04 %    | Missing-Flag; Normalisierung `.upper().strip()`       | Inkonsistente Schreibweise (278 raw → 183 nach upper())  |
| `category`        | 87,394           | 3.17 %    | Missing als eigene Kategorie ("NONE")                 | Strukturelles Missing; nicht reparierbar                 |
| `campaignIndex`   | 2,287,968        | 83.02 %   | Missing als eigene Kategorie ("NONE")                 | Nur Kampagnenprodukte haben Label; Fehlen ist informativ |
| `rrp`             | 0                | 0.00 %    | Kein Handling nötig                                   | Vollständig vorhanden                                    |
| Alle anderen      | 0                | 0.00 %    | Kein Handling nötig                                   | –                                                        |

### 4.2 Noisy / High-Cardinality Features

> Hier werden nur **Risiken** identifiziert. Konkrete Encoding-Entscheidungen gehören in die **Data Preparation** Phase.

| Feature         | Unique Values | Risiko                                                                 | Implikation für Data Preparation               |
| --------------- | ------------- | ---------------------------------------------------------------------- | ---------------------------------------------- |
| `manufacturer`  | 1,065         | Hohe Kardinalität                                                      | Encoding nötig (Frequency, Target, Top-N)      |
| `group`         | 533           | Hohe Kardinalität, alphanumerisch                                      | Grouping oder Embedding                        |
| `content`       | 548           | Gemischt; 5.58 % (153,875 Zeilen) Multipack-Muster                     | Parsing: `is_multipack`, `pack_n`, `pack_size` |
| `pharmForm`     | 278 → 183     | **Schreibweise-Inkonsistenz bestätigt** (278 raw vs. 183 nach upper()) | Normalisierung `.upper().strip()` **zwingend** |
| `category`      | 409           | Hohe Kardinalität                                                      | Grouping oder Top-N Encoding                   |
| `campaignIndex` | 3 (A/B/C)     | 83.02 % missing, strukturell sparse                                    | Kategorie mit "NONE" oder droppen              |

### 4.3 Outliers & Invalid Values

| Prüfung                               | Ergebnis                   | Bemerkung                                      |
| ------------------------------------- | -------------------------- | ---------------------------------------------- |
| `price ≤ 0`                           | **0 Zeilen** ✓             | Kein Problem                                   |
| `competitorPrice ≤ 0` (wo vorhanden)  | **976 Zeilen** ⚠           | Integrity-Verletzung; in Preparation behandeln |
| `competitorPrice` Extremwerte (3×IQR) | **82,817 Zeilen**          | IQR=[5.48, 15.06], Bounds=[−23.26, 43.80]      |
| `rrp ≤ 0`                             | **0 Zeilen** ✓             | Kein Problem                                   |
| `rrp < price` (negativer Rabatt)      | **32,847 Zeilen** (1.19 %) | Erwartbar bei Preiserhöhungen; dokumentiert    |
| `revenue < 0`                         | **0 Zeilen** ✓             | Kein Problem                                   |

### 4.4 Dataset Bias / Representativeness

**Long-Tail-Struktur:**

| Band | # PIDs | Events    | Anteil Events | Order-Rate | Risiko                                       |
| ---- | ------ | --------- | ------------- | ---------- | -------------------------------------------- |
| Head | 2,192  | 1,784,351 | 64.74 %       | 28.09 %    | Modellmetrik dominiert von wenigen Produkten |
| Mid  | 8,772  | 824,569   | 29.92 %       | 21.37 %    | Unterrepräsentiert in Gesamtmetrik           |
| Tail | 10,964 | 147,083   | 5.34 %        | 18.81 %    | Fast unsichtbar in Gesamtevaluation          |

**Konsequenz:** Ungewichtete Metriken (Accuracy, AUC) spiegeln primär Head-Produkte wider.
**Empfehlung:** Stratifizierte Evaluation pro Band (Head / Mid / Tail) + stratified Sampling bei Training (z. B. < 1 % der Daten, aber proportional nach Band + Order/Non-Order Ratio).

### 4.5 Time Leakage / Split Design

**Primärer Grund für Time-Based Split: Temporal Generalization**

- Das Ziel ist die Vorhersage zukünftiger Käufe. Ein Time Split testet, ob das Modell auf ungesehene zukünftige Tage generalisiert – genau das, was im Deployment gefordert ist.
- Ein Random Split misst nur „In-Sample Mixing" und überschätzt die echte Prognosequalität.

**Sekundärer Grund: Hard Leakage bei Lag Features**

- Falls in der Data Preparation Lag Features (z. B. Rolling-Mean Preis, Trend) erzeugt werden, würde ein Random Split zukünftige Werte in die Vergangenheit leaken → harte Leakage.

**Empfohlener Split:**

| Set        | Tage  | Zeilen    | Anteil |
| ---------- | ----- | --------- | ------ |
| Train      | 1–70  | 2,043,073 | 74.1 % |
| Validation | 71–81 | 349,447   | 12.7 % |
| Test       | 82–92 | 363,483   | 13.2 % |

---

## 5) Outputs for Next Phase (Data Preparation)

### 5.1 To-Do Liste

| #   | Task                        | Details                                                                                                                                                                                                                                              |
| --- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Join**                    | Left-Join `train` + `items` auf `pid`; Assert: 0 orphan train PIDs; dokumentiere orphan items count                                                                                                                                                  |
| 2   | **Normalize `pharmForm`**   | `.str.upper().str.strip()`                                                                                                                                                                                                                           |
| 3   | **Quantity ableiten**       | `quantity = round(revenue / price)` für `order == 1`, nur wenn `abs(q_raw − round(q_raw)) ≤ 0.01`; Rest → NaN                                                                                                                                        |
| 4   | **Leakage-Spalten droppen** | `revenue`, `q_raw` entfernen                                                                                                                                                                                                                         |
| 5   | **Feature Engineering**     | `price_diff`, `price_ratio`, `discount_vs_rrp`, `comp_missing_flag`                                                                                                                                                                                  |
| 6   | **Content parsen**          | Regex `\d+X\d+` → `is_multipack`, `pack_n`, `pack_size`                                                                                                                                                                                              |
| 7   | **Encoding**                | `availability`: Ordinal vs. One-Hot testen; `salesIndex`: One-Hot (4 Werte); `campaignIndex`: NA → "NONE", One-Hot; `manufacturer`: Frequency-/Target-Encoding oder Top-N OHE; `group`: Top-N oder Embedding; `category`: One-Hot mit Null-Kategorie |
| 8   | **Wochentag-Proxy**         | `day_mod7 = day % 7` (mit Caveat: kein echtes Datum)                                                                                                                                                                                                 |
| 9   | **Time-Based Split**        | Train 1–70, Val 71–81, Test 82–92                                                                                                                                                                                                                    |
| 10  | **Baseline-Modell**         | Logistic Regression auf `price_diff`, `discount_vs_rrp`, `adFlag`, `availability`, `genericProduct` → Benchmark AUC                                                                                                                                  |

### 5.2 Erwartete Feature-Engineering-Kandidaten

| Feature               | Formel / Logik                                      | Quelle        |
| --------------------- | --------------------------------------------------- | ------------- |
| `price_diff`          | `price − competitorPrice`                           | train         |
| `price_ratio`         | `price / competitorPrice`                           | train         |
| `discount_vs_rrp`     | `(rrp − price) / rrp`                               | train + items |
| `comp_missing_flag`   | `1 wenn competitorPrice NaN, sonst 0`               | train         |
| `is_multipack`        | Regex auf `content` → `\d+X` vorhanden?             | items         |
| `pack_n`, `pack_size` | Parsing von `content` (z. B. `10X1` → n=10, size=1) | items         |
| `pharmForm_norm`      | `pharmForm.str.upper().str.strip()`                 | items         |
| `day_mod7`            | `day % 7` (struktureller Proxy, kein echtes Datum)  | train         |

---

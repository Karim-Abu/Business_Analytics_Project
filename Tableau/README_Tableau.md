# Tableau Extracts – Anleitung & Dokumentation

## Übersicht

Das Script `tableau_extract.py` erzeugt aus den Rohdaten (`train.csv` + `items.csv`) zwei optimierte Tableau-Hyper-Extracts für **Data Understanding Dashboards**:

| Datei                      | Grain                 | Zeilen  | Zweck                                                |
| -------------------------- | --------------------- | ------- | ---------------------------------------------------- |
| `Tableau_CoreDaily.hyper`  | `day` + `pid`         | 751'851 | Tägliche Produkt-KPIs, Pricing-Driver-Bins, Segmente |
| `Tableau_OrdersLine.hyper` | `lineID` (nur Orders) | 705'090 | Quantity-Verteilung, Umsatzanalyse auf Bestellebene  |

---

## Ausführung

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. Extract erzeugen (ca. 35s)
python tableau_extract.py
```

Ausgabe: zwei `.hyper`-Dateien im selben Verzeichnis.

---

## Datenmodell in Tableau einrichten

> **Wichtig:** Beide Tabellen müssen im **selben Data Source** als Logical Tables verbunden sein – **nicht** als separate Data Sources, sonst funktioniert kein Cross-Filtering.

### Schritt-für-Schritt

1. **Tableau Desktop** → `Connect` → `More…` → **Tableau Extract (.hyper)** → `Tableau_CoreDaily.hyper` öffnen
2. Im **Data Source**-Tab: Doppelklick auf die Logical Table `CoreDaily` → die Physical Layer öffnet sich
3. Zurück zur **Logical Layer** (Pfeil oben links)
4. **Drag & Drop**: `Tableau_OrdersLine.hyper` in die Logical Layer ziehen (neben die CoreDaily-Tabelle)
5. **Relationship** definieren:
   - `CoreDaily.day = OrdersLine.day`
   - **AND** `CoreDaily.pid = OrdersLine.pid`
6. Fertig → Zum Worksheet wechseln

### Warum dieses Setup?

- **Relationship** (nicht Join): Tableau nutzt automatisch die richtige Aggregationsebene je nach Kontext
- **Denormalisierte Dimensionen** in OrdersLine (`category_norm`, `pharmForm_norm`, `pid_segment`, etc.): Filter-Pillen auf OrdersLine-Sheets brauchen keinen teuren Join auf CoreDaily
- **Cross-Filtering**: Ein Filter auf `pid_segment = Head` in einem CoreDaily-Sheet filtert automatisch auch OrdersLine-Sheets

---

## Spalten-Referenz

### CoreDaily (Tableau_CoreDaily.hyper)

| Spalte                    | Typ    | Beschreibung                                                                                |
| ------------------------- | ------ | ------------------------------------------------------------------------------------------- |
| `day`                     | int    | Tag 1–92 im Beobachtungszeitraum                                                            |
| `pid`                     | int    | Produkt-ID                                                                                  |
| `n_events`                | int    | Anzahl Events (Click+Basket+Order) an diesem Tag für dieses Produkt                         |
| `n_click`                 | int    | Anzahl Clicks                                                                               |
| `n_basket`                | int    | Anzahl Basket-Events                                                                        |
| `n_order`                 | int    | Anzahl Orders                                                                               |
| `click_rate`              | float  | n_click / n_events                                                                          |
| `basket_rate`             | float  | n_basket / n_events                                                                         |
| `order_rate`              | float  | n_order / n_events                                                                          |
| `price`                   | float  | Tagespreis (Median, falls intraday mehrere Werte)                                           |
| `availability`            | int    | Verfügbarkeit (1–4)                                                                         |
| `adFlag`                  | int    | 1 = an diesem Tag beworben                                                                  |
| `competitorPrice_missing` | int    | 1 = kein valider Wettbewerberpreis                                                          |
| `price_diff_bin`          | string | Equal-Frequency-Bin (20 Quantile) von `price − competitorPrice`, oder `NO_COMPETITOR_PRICE` |
| `discount_bin`            | string | Equal-Frequency-Bin (20 Quantile) von `(rrp − price) / rrp`, oder `NO_RRP`                  |
| `campaignIndex_norm`      | string | A / B / C / NONE                                                                            |
| `has_campaign`            | int    | 1 wenn Kampagnen-Produkt                                                                    |
| `category_norm`           | string | Produktkategorie (Dimension)                                                                |
| `pharmForm_norm`          | string | Darreichungsform (normalisiert: UPPER + TRIM)                                               |
| `genericProduct`          | int    | 0/1: Generikum                                                                              |
| `salesIndex`              | int    | Verkaufsindex (40, 44, 52, 53)                                                              |
| `is_multipack`            | int    | 1 wenn Multipack-Produkt                                                                    |
| `pack_n`                  | int    | Anzahl Einzelpackungen im Multipack (sonst 1)                                               |
| `pack_size`               | float  | Packungsgrösse (letzte Zahl im Content-Feld)                                                |
| `pid_total_events`        | int    | Gesamtanzahl Events für dieses Produkt (alle Tage)                                          |
| `pid_segment`             | string | Head (Top 10% PIDs) / Mid / Tail (Bottom 50% PIDs)                                          |

### OrdersLine (Tableau_OrdersLine.hyper)

| Spalte               | Typ    | Beschreibung                                                        |
| -------------------- | ------ | ------------------------------------------------------------------- |
| `lineID`             | int    | Eindeutige Zeilen-ID                                                |
| `day`                | int    | Tag                                                                 |
| `pid`                | int    | Produkt-ID                                                          |
| `price`              | float  | Preis bei Kauf                                                      |
| `revenue`            | float  | Umsatz (= price × quantity)                                         |
| `quantity`           | int    | Abgeleitete Menge: `round(revenue / price)`                         |
| `qty_suspicious`     | int    | 1 wenn Quantity-Ableitung unplausibel (Abweichung > 5% vom Integer) |
| `quantity_class`     | string | Klasse: 1, 2, 3, 4-5, >5                                            |
| `category_norm`      | string | Produktkategorie (denormalisiert)                                   |
| `pharmForm_norm`     | string | Darreichungsform (denormalisiert)                                   |
| `genericProduct`     | int    | Generikum-Flag (denormalisiert)                                     |
| `pid_segment`        | string | Head / Mid / Tail (denormalisiert)                                  |
| `availability`       | int    | Verfügbarkeit an diesem Tag (aus CoreDaily)                         |
| `adFlag`             | int    | Ad-Flag an diesem Tag (aus CoreDaily)                               |
| `campaignIndex_norm` | string | Kampagne (aus CoreDaily)                                            |
| `is_multipack`       | int    | Multipack-Flag (denormalisiert)                                     |
| `salesIndex`         | int    | Verkaufsindex (denormalisiert)                                      |

---

## Dashboard-Ideen: Was Tableau sinnvoll visualisieren kann

Die beiden Extracts sind optimiert für **Data Understanding** im CRISP-DM-Sinn. Hier die konkreten Dashboard-Vorschläge nach Analysezweck:

### 1. Pareto / Long-Tail-Analyse

**Quelle:** CoreDaily → `pid_segment`, `pid_total_events`, `n_events`

| Visualisierung                              | Wie                                                                                                        |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Pareto-Kurve (kumulierter Event-Anteil)** | X = PIDs (sortiert nach Events, absteigend), Y = kumulierter %-Anteil. Zeigt: Top 10% PIDs = ~65% Events   |
| **Segment-Vergleich**                       | Bar Chart: `pid_segment` (Head/Mid/Tail) × `SUM(n_events)`, `AVG(order_rate)` → Head hat höhere Order-Rate |
| **Segment × Kategorie Heatmap**             | Rows = `category_norm`, Cols = `pid_segment`, Color = `AVG(order_rate)`                                    |

**Erkenntnis:** Long-Tail-Struktur ist extrem – Bottom 50% der Produkte machen nur 5.3% der Events aus. Modellperformance wird von Head-Produkten dominiert.

### 2. Pricing-Driver-Analyse (pre-binned)

**Quelle:** CoreDaily → `price_diff_bin`, `discount_bin`, `order_rate`

| Visualisierung                         | Wie                                                                                                                                                   |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Order-Rate nach Preisdifferenz-Bin** | Bar: X = `price_diff_bin` (ist bereits sortierbar als Interval-String), Y = `AVG(order_rate)` → Zeigt: je teurer vs. Wettbewerb, desto weniger Orders |
| **Order-Rate nach Rabatt-Bin**         | Bar: X = `discount_bin`, Y = `AVG(order_rate)` → Höherer Rabatt vs. UVP korreliert mit höherer Kaufrate                                               |
| **Scatter: Price vs. Order-Rate**      | X = `AVG(price)`, Y = `AVG(order_rate)`, Detail = `pid`, Color = `pid_segment`                                                                        |

**Warum Bins in Python?** Tableau kann kein Equal-Frequency Binning – nur Equal-Width. Quantil-Bins vermeiden, dass Extremwerte wenige Bins dominieren. Die Bins sind geclippt auf p01–p99. Die Sonderklassen `NO_COMPETITOR_PRICE` und `NO_RRP` sind als eigene Filterwerte sichtbar.

### 3. Missingness als Signal

**Quelle:** CoreDaily → `competitorPrice_missing`, `order_rate`, `n_events`

| Visualisierung                                 | Wie                                                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Order-Rate: mit vs. ohne Wettbewerberpreis** | Bar: `competitorPrice_missing` (0/1) × `AVG(order_rate)` → Ist das Fehlen des Wettbewerberpreises ein Kauftreiber?                         |
| **Missingness-Heatmap über Zeit**              | X = `day`, Y = `category_norm`, Color = `AVG(competitorPrice_missing)` → Zeigt ob Missingness kategorie-/zeitabhängig ist                  |
| **Campaign-Signal**                            | Bar: `campaignIndex_norm` (A/B/C/NONE) × `AVG(order_rate)` → 83% der Zeilen sind NONE; Kampagnenprodukte zeigen abweichendes Kaufverhalten |

### 4. adFlag Confounding Grid

**Quelle:** CoreDaily → `adFlag`, `availability`, `pid_segment`, `order_rate`

| Visualisierung                   | Wie                                                                                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **adFlag × availability Matrix** | Rows = `availability` (1–4), Cols = `adFlag` (0/1), Color/Label = `AVG(order_rate)` → Prüft ob adFlag-Effekt auch nach Availability-Kontrolle besteht |
| **adFlag × pid_segment**         | Gleiche Logik, aufgesplittet nach Head/Mid/Tail → Confounding-Check: Head-Produkte werden häufiger beworben UND gekauft                               |
| **adFlag × genericProduct**      | Rows = `genericProduct`, Cols = `adFlag`, Color = `AVG(order_rate)`                                                                                   |

**Erkenntnis:** `adFlag` korreliert mit Order-Rate, aber Confounding mit Verfügbarkeit und Produktpopularität ist wahrscheinlich.

### 5. Zeittrend & Moving Average

**Quelle:** CoreDaily → `day`, `order_rate`, `n_events`, `n_order`

| Visualisierung                    | Wie                                                                                                                                             |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Order-Rate Zeitreihe + MA**     | Line: X = `day`, Y = `AVG(order_rate)`. Füge Table Calculation hinzu: Moving Average (Window = 7 Tage)                                          |
| **Action Distribution über Zeit** | Stacked Area: X = `day`, Y = `SUM(n_click)`, `SUM(n_basket)`, `SUM(n_order)` → Zeigt ob Click/Basket/Order-Anteile über die 92 Tage stabil sind |
| **Wochentag-Proxy**               | Calculated Field: `day % 7` → Bar × `AVG(order_rate)` → Gibt es zyklische Muster? (Caveat: kein echtes Datum)                                   |

**Tableau-Tipp:** Für den Moving Average: Rechtsklick auf Order-Rate-Pill → Add Table Calculation → Moving Average → vorherige 3 + nachfolgende 3 Perioden.

### 6. Quantity-Verteilung (nur Orders)

**Quelle:** OrdersLine → `quantity`, `quantity_class`, `price`, `revenue`

| Visualisierung                 | Wie                                                                                                                                |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Quantity-Histogramm**        | Bar: X = `quantity_class` (1, 2, 3, 4-5, >5), Y = `COUNT(lineID)` → Zeigt: 80% aller Orders haben Qty=1                            |
| **Quantity × Kategorie**       | Heatmap: Rows = `category_norm`, Cols = `quantity_class`, Color = `COUNT(lineID)` → Welche Kategorien haben Multi-Quantity-Orders? |
| **Umsatz pro Quantity-Klasse** | Bar: X = `quantity_class`, Y = `SUM(revenue)` → Umsatzanteil der Mehrfachbestellungen                                              |
| **Multipack-Einfluss**         | Bar: X = `is_multipack` (0/1), Y = `AVG(quantity)` → Multipacks korrelieren mit höherer Bestellmenge?                              |

### 7. Produktcharakteristik-Explorer

**Quelle:** CoreDaily → alle Dimensionen

| Visualisierung                  | Wie                                                                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **pharmForm Top-20**            | Bar: `pharmForm_norm` (Top 20 nach n_events) × `AVG(order_rate)` → Darreichungsform als Kauftreiber?                  |
| **Generika vs. Markenprodukte** | Side-by-side Bar: `genericProduct` × `AVG(order_rate)`, `AVG(price)` → Generika: günstigere Preise + andere Kaufrate? |
| **salesIndex Vergleich**        | Bar: `salesIndex` (40/44/52/53) × `AVG(order_rate)` → Bedeutung des Index empirisch ableiten                          |

---

## Technische Entscheidungen

| Entscheidung                                  | Begründung                                                                                                           |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Grain `day + pid`** (CoreDaily)             | Intraday-Check zeigt <0.5% Violations pro Spalte → sicher                                                            |
| **`.first()` statt `mode()`** für daily attrs | Violations <0.5%, Sort vor `.first()` macht es deterministisch (kleinster Wert gewinnt). 100× schneller als `mode()` |
| **`median`** für price                        | Robust gegen rare intraday Preisvariation (0.15% Violations)                                                         |
| **Clipping p01–p99 vor `qcut`**               | Verhindert, dass extreme Ausreisser Bin-Kapazität verschwenden                                                       |
| **Multipack-Parsing generisch**               | `re.findall(r"\d+")` handhabt beliebige Tiefe: `6X4X200` → pack_n=24, pack_size=200                                  |
| **Denormalisierung in OrdersLine**            | Values kommen aus CoreDaily (nicht Rohzeile) → konsistent mit der Aggregationslogik                                  |
| **`qty_suspicious` Flag**                     | Plausibilitätscheck: 0 Zeilen verdächtig (100% near-integer) – trotzdem als Sicherheitsnetz                          |
| **`pharmForm_norm`** via `pd.StringDtype`     | Vermeidet `NaN → "nan" → "NAN"`-Bug, den `astype(str)` produziert                                                    |

---

## Data Quality Summary

| Check                               | Ergebnis                                                                |
| ----------------------------------- | ----------------------------------------------------------------------- |
| price > 0                           | PASS ✓                                                                  |
| revenue >= 0                        | PASS ✓                                                                  |
| click + basket + order == 1         | PASS ✓                                                                  |
| lineID unique                       | PASS ✓                                                                  |
| competitorPrice > 0 (where present) | FAIL – 976 Zeilen (als missing behandelt via `competitorPrice_missing`) |
| rrp > 0 (where present)             | PASS ✓                                                                  |
| Join coverage                       | 100%                                                                    |
| Suspicious quantities               | 0 (0%)                                                                  |

---

## Dateien im Projekt

| Datei                      | Beschreibung                                   |
| -------------------------- | ---------------------------------------------- |
| `tableau_extract.py`       | Pipeline-Script (erzeugt die .hyper-Dateien)   |
| `requirements.txt`         | Python-Abhängigkeiten                          |
| `Tableau_CoreDaily.hyper`  | Daily-Aggregat (751'851 Zeilen, 26 Spalten)    |
| `Tableau_OrdersLine.hyper` | Order-Level-Daten (705'090 Zeilen, 17 Spalten) |
| `README.md`                | Data Understanding Dokumentation (CRISP-DM)    |
| `README_Tableau.md`        | **Diese Datei** – Tableau-Anleitung            |
| `Data_understanding.ipynb` | Explorative Analyse (Notebook)                 |

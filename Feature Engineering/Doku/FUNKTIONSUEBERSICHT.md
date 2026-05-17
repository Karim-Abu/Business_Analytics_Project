# Funktionsübersicht und Pipeline-Verständnis

Dieses Dokument erklärt die Feature-Engineering- und Data-Preparation-Pipeline auf Basis des tatsächlich vorhandenen Python-Codes.
Funktionen mit führendem `_` sind interne Hilfsfunktionen.

## Wie dieses Dokument aufgebaut ist

- Es trennt explizit zwischen **technischem Code-Ablauf** und **methodischer Logik**.
- Wenn die Reihenfolge im Code von einer idealisierten CRISP-DM-/Modellierungslogik abweicht, wird der **tatsächliche Code-Ablauf** als primär dokumentiert.
- Die methodische Einordnung wird separat erklärt, damit man fachlich argumentieren kann, ohne den realen Implementierungsfluss zu verfälschen.

## Big Picture der Pipeline (echter Code-Ablauf)

Die Entry Points sind `run_safe_only()` und `run_safe_plus_conditional()` in `Preprocessing/main_build_datasets.py`.

1. `ensure_output_dirs()` erstellt alle Zielordner.
2. `load_raw_data()` lädt `train.csv` und `items.csv`.
3. `merge_train_items(df_train, df_items)` macht den Left Join über `pid`.
4. `run_all_preprocessing(df)` bereinigt und normalisiert Rohspalten, leitet `quantity` und `qty_suspicious` ab.
5. `assert_preprocessing_integrity(df, context)` prüft zentrale Nachbedingungen.
6. `run_split(df)` teilt chronologisch in Train/Test/Validation.
7. `fit_pid_segment(df_train)` lernt Head/Mid/Tail-Segmente auf Train.
8. `apply_pid_segment(df, pid_segment_map)` schreibt `pid_segment` in alle Splits.
9. `run_all_safe_features(df_train, df_val, df_test)` erzeugt leakage-freie Features; train-only Fits werden auf alle Splits angewendet.
10. `build_safe_feature_matrices(df_train, df_val, df_test)` baut CLS/REG-Matrizen für SAFE-Features.
11. Nur im Modus `safe_plus_conditional`: `run_all_conditional_features(df_train, df_val, df_test)` erzeugt Historien-/OOF-basierte Features.
12. Nur im Modus `safe_plus_conditional`: `build_conditional_feature_matrices(df_train, df_val, df_test)` baut zusätzliche Conditional-Matrizen.
13. `run_sampling(df_train)` erzeugt prototypische Samples für CLS und REG (nur Train).
14. `run_full_audit(...)` erstellt Join-, Missingness-, Outlier-, Target- und Matrix-Audits.
15. `_export_all(...)` speichert Matrizen, Audits und Metadaten.
16. Nur im Modus `safe_plus_conditional`: `export_orange_csvs(...)` erzeugt Orange-CSV-Dateien inklusive Manifest.

### Methodische Logik (separat zur Code-Reihenfolge)

- Fachlich kann man die Pipeline als Fluss erklären: Rohdaten → Qualitätskontrolle → Features → Modellmatrizen → Prototyping/Audit → Tooling-Export.
- Die **Feature Selection** (`Experiment/feature_selection.py`, `Experiment/feature_selection_r4.py`) ist im Code **downstream** und nicht Teil des Entry-Point-Laufs in `main_build_datasets.py`.
- Methodisch gehört Feature Selection zur Modellierungs- und Reduktionsphase, technisch ist sie ein separater Analyse-Schritt auf bereits exportierten Matrizen.

## Zentrale Projektlogik

### Warum Two-Stage?

- Stage 1 (CLS): Ziel `order` (Kauf ja/nein).
- Stage 2 (REG): Ziel `quantity`, aber nur für Zeilen mit `order == 1`.
- Die Trennung ist im Code verankert durch `get_reg_mask(df)` und `assert_reg_stage2_only(df, context)`.

### Warum `quantity = revenue / price`?

- In den Rohdaten wird `quantity` in `run_all_preprocessing(df)` aus `revenue / price` abgeleitet.
- `price <= 0` wird über `safe_price` zu `NaN` geschützt, dadurch entstehen keine Divisionen durch 0.
- Problematische Fälle werden über `qty_suspicious` markiert und später für REG ausgeschlossen.

### Warum sind `revenue`, `quantity`, `order`, `click`, `basket` je nach Stage problematisch?

- `revenue`: direkt zielnah (`revenue = price * quantity`), daher Leakage-Risiko.
- `quantity`/`quantity_class`: für CLS verboten, weil Stage-2-nahe Zielinformation.
- `order`: für REG verboten, weil in Stage-2 konstant bzw. direkt Selektionskriterium.
- `click`, `basket`: laut Projektlogik in `audit_dropped_features()` leakage-gefährdet für CLS.
- Diese Regeln sind explizit in `config.py` (`VERBOTEN_CLS`, `VERBOTEN_REG`) und werden durch `assert_no_forbidden_features(...)` erzwungen.

### Warum chronologischer Split?

- `run_split(df)` trennt strikt nach Tagen (Train 26-70, Test 71-81, Validation 82-92).
- So wird ein realistischeres Vorhersageszenario abgebildet als bei zufälligem Split.
- Zeitabhängige Features (insbesondere OOF) hängen methodisch an dieser zeitlichen Struktur.

### Warum Orange-Export separat?

- Orange benötigt ein eigenes tabellarisches Schema.
- `export_orange_csvs(...)` bildet final selektierte Feature-Sets ab, erzwingt Spaltenkonsistenz und schreibt ein Manifest.
- Die Orange-Schicht ist bewusst getrennt von den Parquet-Matrizen für Python-Analysen.

## Safe Features vs. Conditional Features

### Safe Features

- Ableitung aus aktueller Zeile oder Stammdaten.
- Train-only-Fits mit sauberem Apply-Muster:
  - `fit_binning_edges(df_train, n_bins)` → `apply_binned_features(df, bin_edges)`
  - `fit_manufacturer_frequency(df_train)` → `apply_manufacturer_frequency(df, freq_map)`
- Kein Target-History-Training pro Zeile.

### Conditional Features

- Nutzt Historie, Aggregationen oder targetnahe Signale.
- Methodisch oft stärker, aber leakage-gefährdeter.
- Schutzmechanismen im Code:
  - Kumulative Historienwerte über `cumsum - current` in `compute_cumulative_features(...)`.
  - Train-only-Fits für Mappings.
  - Forward OOF über `_time_aware_forward_oof(...)`.
  - Fallbacks über `cold_start`/`global_mean`.

## Forward OOF verständlich erklärt

OOF bedeutet: Jede Train-Zeile bekommt einen Encoding-Wert, der aus **anderen** Zeilen gelernt wurde, nicht aus sich selbst.

### Problem ohne OOF

- Normales Target-Encoding auf dem gesamten Train (z. B. Mittelwert `order` pro `pid`) enthält Self Leakage.
- Bei Zeitdaten wäre zusätzlich Future Leakage möglich, wenn spätere Tage in frühere Zeilen einfließen.

### Warum Random OOF hier nicht reicht

- Random-Folds mischen Zeit und erlauben indirekt Informationen aus späteren Tagen in frühere Fold-Berechnungen.

### Warum Forward OOF

- `_time_aware_forward_oof(df, group_col, target_col, cold_start, history_mask)` nutzt Wochenblöcke relativ zu `TRAIN_DAY_START`.
- Block 1 erhält `cold_start`, da es keine Vergangenheit gibt.
- Block 2 nutzt nur Block 1 als Historie.
- Block 3 nutzt nur Block 1+2 usw.

### Cold Start

- Für erste Zeitblöcke oder leere Historie wird ein konservativer Fallback verwendet.
- In `config.py`: `OOF_COLD_START_PROB = 0.5`, `OOF_COLD_START_QTY = 1.0`.

### Unterschied Train-OOF vs. Val/Test-Encoding

- Train: `_time_aware_forward_oof(...)` für leakagesichere Werte pro Train-Zeile.
- Val/Test: `_fit_full_train_encoding(...)` auf ganzem Train und `_apply_encoding(...)` auf neue Daten.

### Mini-Beispiel Forward OOF

Vereinfachte Wochenblöcke (`group_col = pid`, `target_col = order`):

| day | week_block | pid | order | Encoding-Idee                                   |
| --- | ---------: | --- | ----: | ----------------------------------------------- |
| 26  |          1 | A   |     1 | cold_start (0.5)                                |
| 28  |          1 | B   |     0 | cold_start (0.5)                                |
| 33  |          2 | A   |     1 | Mittelwert A aus Block 1 = 1.0                  |
| 35  |          2 | C   |     0 | unbekannt in Historie → global_mean aus Block 1 |

## Wichtige Verteidigungssätze

- „Wir splitten zeitlich statt zufällig, weil unser Einsatzfall eine Zukunftsprognose ist.“
- „SAFE und CONDITIONAL sind getrennt, weil sie unterschiedliche Leakage-Risiken haben.“
- „`revenue` bleibt aus Features draußen, weil es direkt `price * quantity` enthält.“
- „Sampling ist nur Prototyping auf Train; Test und Validation bleiben vollständig.“
- „Stage 2 REG läuft nur auf `order == 1` mit validem, nicht-suspicious `quantity`.“
- „Export- und Validierungschecks sind kein Extra, sondern Schema- und Leakage-Schutz.“

---

## `io_utils.py`

### Rolle im Gesamtprozess

Diese Datei kapselt Laden, Join und Speichern. Sie wird früh im Pipeline-Lauf aufgerufen und später in `_export_all(...)` für alle Artefakte genutzt. Ohne diese I/O-Schicht wären Pfade und Dateiformate über mehrere Module verteilt.

### Wichtige Inputs

- `cfg.TRAIN_CSV`, `cfg.ITEMS_CSV`
- DataFrames `df_train`, `df_items`
- Zielpfade (`Path | str`)

### Wichtige Outputs

- Geladene Roh-DataFrames
- Gemergtes DataFrame
- Persistierte Parquet/CSV/Text-Dateien

### Funktionen

#### `ensure_output_dirs()`

- Zweck: Legt alle Output-Ordner an.
- Input: Keine.
- Output: Verzeichnisstruktur vorhanden.
- Was passiert intern: Iteration über konfigurierte Pfade, `mkdir(parents=True, exist_ok=True)`.
- Warum ist das methodisch wichtig: Stabiler Lauf ohne Pfadfehler.
- Risiken / Stolperstellen: Falsche `config.py`-Pfade.
- Einfacher Verteidigungssatz: „Wir initialisieren die Output-Struktur zentral, bevor wir Artefakte schreiben.“

#### `load_raw_data()`

- Zweck: Lädt Rohdaten.
- Input: `cfg.TRAIN_CSV`, `cfg.ITEMS_CSV`.
- Output: `(df_train, df_items)`.
- Was passiert intern: Dateiexistenzprüfung, `pd.read_csv(..., sep='|')`.
- Warum ist das methodisch wichtig: Fail-fast bei fehlenden Dateien.
- Risiken / Stolperstellen: Falscher Separator oder ungültige Rohdateien.
- Einfacher Verteidigungssatz: „Dateien werden vor dem Lesen validiert, damit Fehler früh sichtbar sind.“

#### `merge_train_items(df_train, df_items)`

- Zweck: Left Join über `pid`.
- Input: Train- und Item-DataFrame.
- Output: `df` mit angereicherten Item-Informationen.
- Was passiert intern: `df_train.merge(df_items, on='pid', how='left')`.
- Warum ist das methodisch wichtig: Train-Zeilen bleiben vollständig erhalten.
- Risiken / Stolperstellen: Fehlende Item-PIDs erzeugen Missingness downstream.
- Einfacher Verteidigungssatz: „Wir verlieren keine Train-Zeilen, weil der Join linksseitig ist.“

#### `save_parquet(df, path)`

- Zweck: Speichert DataFrame als Parquet.
- Input: `df`, `path`.
- Output: Datei auf Disk.
- Was passiert intern: Zielordner erstellen, `to_parquet(index=False)`.
- Warum ist das methodisch wichtig: Reproduzierbare Modellmatrizen.
- Risiken / Stolperstellen: Schemaänderungen zwischen Läufen.
- Einfacher Verteidigungssatz: „Parquet ist unser reproduzierbares Austauschformat für Python-Matrizen.“

#### `save_csv(df, path)`

- Zweck: Speichert DataFrame als CSV.
- Input: `df`, `path`.
- Output: CSV-Datei.
- Was passiert intern: Ordner prüfen, `to_csv(index=False)`.
- Warum ist das methodisch wichtig: Audits und Exporte sind transparent lesbar.
- Risiken / Stolperstellen: CSV-Typinferenz in Fremdtools.
- Einfacher Verteidigungssatz: „CSV nutzen wir für Audit- und Tooling-Kompatibilität.“

#### `save_text_report(text, path)`

- Zweck: Speichert Textberichte.
- Input: String-Report, Pfad.
- Output: Textdatei.
- Was passiert intern: UTF-8 Write.
- Warum ist das methodisch wichtig: Lesbare Zusammenfassungen für Review/Präsentation.
- Risiken / Stolperstellen: Keine strukturelle Validierung des Report-Inhalts.
- Einfacher Verteidigungssatz: „Wir ergänzen Tabellen um lesbare Zusammenfassungen für die fachliche Diskussion.“

## `Preprocessing/preprocessing.py`

### Rolle im Gesamtprozess

`run_all_preprocessing(df)` ist der erste fachliche Transformationsschritt nach dem Join. Hier werden Rohspalten in stabile Modellierungsgrundlagen überführt. Spätere Stufen wie Safe/Conditional Features setzen diese Normalisierungen voraus.

### Wichtige Inputs

- Gemergtes DataFrame nach Join
- Spalten wie `price`, `revenue`, `campaignIndex`, `group`, `content`, `order`

### Wichtige Outputs

- Normalisierte Felder (`*_norm`, `group_clean`, `content_clean`)
- `quantity`, `quantity_class`, `qty_suspicious`
- Bereinigtes `competitorPrice`

### Funktionen

#### `run_all_preprocessing(df)`

- Zweck: End-to-end Bereinigung und Basiskonstruktion.
- Input: Roh-/Join-DataFrame.
- Output: Transformiertes DataFrame mit zusätzlichen Qualitäts- und Hilfsspalten.
- Was passiert intern:
  - `competitorPrice <= 0` wird zu `NaN`; `competitorPrice_missing` wird gesetzt.
  - Kategorische Felder werden vereinheitlicht (`campaignIndex_norm`, `category_norm`, `pharmForm_norm`, `unit_norm`).
  - `group_clean`, `content_clean` werden textuell normalisiert.
  - `quantity = revenue / price` nur für `order == 1`, mit `safe_price`-Guard.
  - `quantity_class` wird gebinnt.
  - `qty_suspicious` markiert negative, nicht-ganzzahlige oder fehlende Mengen bei `order == 1`.
  - `salesIndex` wird numerisch gecastet.
- Warum ist das methodisch wichtig:
  - Definiert die Stage-2-Basis (`quantity`, `qty_suspicious`).
  - Verhindert fragile Downstream-Logik durch inkonsistente Kategorien.
  - Erzwingt eine explizite Missingness-Strategie statt stiller Defaults.
- Risiken / Stolperstellen:
  - `quantity` hängt direkt an `revenue`/`price`-Qualität.
  - Stage-2 kann stark schrumpfen, wenn viele `qty_suspicious` entstehen.
  - `quantity` ist nicht immer ganzzahlig, dadurch Ausschlüsse möglich.
- Einfacher Verteidigungssatz: „Preprocessing ist bei uns nicht nur Putzen, sondern die formale Definition, welche Zeilen überhaupt Stage-2-fähig sind.“
- Mini-Beispiel:

| order | revenue | price | quantity | qty_suspicious |
| ----: | ------: | ----: | -------: | -------------: |
|     1 |      20 |    10 |      2.0 |              0 |
|     1 |      20 |     0 |      NaN |              1 |
|     0 |       0 |    10 |      NaN |              0 |

## `Preprocessing/validation.py`

### Rolle im Gesamtprozess

Diese Datei enthält Guardrails, die in Matrixbau und Export aufgerufen werden. Sie sorgt dafür, dass Feature-Listen, Splits und Stage-2-Daten regelkonform bleiben. Ohne diese Checks könnten Leakage oder Schemafehler unbemerkt durchlaufen.

### Wichtige Inputs

- Feature-Listen
- Verbotene Listen (`VERBOTEN_CLS`, `VERBOTEN_REG`)
- Split-DataFrames / Export-Frames

### Wichtige Outputs

- Keine Artefakte; bei Verstoß `ValueError`.

### Funktionen

#### `assert_no_forbidden_features(feature_list, forbidden, context)`

- Zweck: Blockiert leakage-gefährdete Features.
- Input: `feature_list`, `forbidden`, `context`.
- Output: None oder Fehler.
- Was passiert intern: Schnittmenge zwischen Featureliste und Verboten wird gebildet.
- Warum ist das methodisch wichtig: Zentrale Leakagesperre für CLS/REG.
- Risiken / Stolperstellen: Verbotene Liste muss aktuell gehalten werden.
- Einfacher Verteidigungssatz: „Selbst wenn jemand eine schlechte Featureliste konfiguriert, stoppt der Guard die Pipeline.“

#### `assert_no_duplicate_features(feature_list, context)`

- Zweck: Verhindert doppelte Featureeinträge.
- Input: `feature_list`, `context`.
- Output: None oder Fehler.
- Was passiert intern: Set-basierte Duplikaterkennung.
- Warum ist das methodisch wichtig: Stabiles Schema für Training/Export.
- Risiken / Stolperstellen: Gleichnamige Features aus Merge-Fehlern.
- Einfacher Verteidigungssatz: „Wir erzwingen eindeutige Featurelisten, damit das Modellschema deterministisch bleibt.“

#### `assert_cross_split_columns(frames, context)`

- Zweck: Prüft gleiche Spalten und Reihenfolge über mehrere Splits.
- Input: Dict von Frames.
- Output: None oder Fehler.
- Was passiert intern: Erstes Frame ist Referenz; alle anderen werden exakt verglichen.
- Warum ist das methodisch wichtig: Train/Test/Val/Sample müssen schema-identisch sein.
- Risiken / Stolperstellen: Reihenfolgeabweichung ist bereits kritisch für Tools.
- Einfacher Verteidigungssatz: „Wir validieren nicht nur Spaltenmengen, sondern auch Reihenfolge als Vertragsfläche.“

#### `assert_reg_stage2_only(df, context)`

- Zweck: Sichert Stage-2-Definition für REG.
- Input: `df`, `context`.
- Output: None oder Fehler.
- Was passiert intern: Zählt Verstöße für `order != 1`, `quantity` NaN, `qty_suspicious != 0`.
- Warum ist das methodisch wichtig: Verhindert Mischpopulation in REG-Bewertungen.
- Risiken / Stolperstellen: Wenn `qty_suspicious` fehlt, reduziert sich die Prüfstrenge.
- Einfacher Verteidigungssatz: „REG-Daten müssen dieselbe Stage-2-Definition erfüllen, sonst sind Metriken nicht interpretierbar.“

#### `assert_preprocessing_integrity(df, context)`

- Zweck: Postconditions nach Preprocessing.
- Input: `df`, `context`.
- Output: None oder Fehler.
- Was passiert intern: Prüft u. a. `competitorPrice`, Binärflags, Normspalten, `quantity`-Konsistenz.
- Warum ist das methodisch wichtig: Früher Qualitätsanker vor Split und Featurebau.
- Risiken / Stolperstellen: Neue Spalten im Preprocessing müssen im Check ggf. ergänzt werden.
- Einfacher Verteidigungssatz: „Wir lassen keine implizite Preprocessing-Drift zu, weil Integritätsprüfungen direkt danach laufen.“

## `Preprocessing/audit.py`

### Rolle im Gesamtprozess

Audit-Funktionen fassen Datenqualität und Ergebnisstruktur zusammen. Sie werden über `run_full_audit(...)` gesammelt nach Sampling/Matrixbau aufgerufen. Die Reports landen als CSV und sind Teil der Rechenschaft.

### Wichtige Inputs

- Rohdaten, gemergte Daten, Splits, Matrix-Dict

### Wichtige Outputs

- Reports wie `join_quality`, `missingness_*`, `outliers_train`, `target_distribution`, `feature_sets`, `dropped_features`

### Funktionen

#### `audit_join_quality(df_train_raw, df_items, df_merged)`

- Zweck: Join-Gesundheit quantifizieren.
- Input: Roh-, Item-, Merge-Frame.
- Output: Kennzahlentabelle.
- Was passiert intern: PID-Mengenvergleiche, Orphan-Zählung, Row-Count-Match.
- Warum ist das methodisch wichtig: Früher Nachweis, ob Join erwartungskonform war.
- Risiken / Stolperstellen: Viele Orphans erhöhen Missingness downstream.
- Einfacher Verteidigungssatz: „Wir messen Join-Qualität explizit statt sie anzunehmen.“

#### `audit_missingness(df)`

- Zweck: Fehlwerte pro Spalte.
- Input: Beliebiges DataFrame.
- Output: Nur Spalten mit Missingness.
- Was passiert intern: `isnull().sum()` + Prozentberechnung.
- Warum ist das methodisch wichtig: Priorisierung von Imputation/Feature-Entscheiden.
- Risiken / Stolperstellen: Missingness kann zwischen Splits differieren.
- Einfacher Verteidigungssatz: „Missingness wird pro Split transparent gemacht, nicht global versteckt.“

#### `audit_outliers(df)`

- Zweck: Robuste Outlier-Indikatoren.
- Input: Numerische Kernspalten.
- Output: Quantil-/IQR-basierte Tabelle.
- Was passiert intern: p01/p25/median/p75/p99, IQR, Extremzähler.
- Warum ist das methodisch wichtig: Erklärt Ausreißerlast vor Modellierung.
- Risiken / Stolperstellen: Nur einfache Heuristik, keine domänenspezifische Outlier-Logik.
- Einfacher Verteidigungssatz: „Wir nutzen robuste Statistik, um Ausreißer strukturiert zu dokumentieren.“

#### `audit_target_distribution(df_train, df_val, df_test)`

- Zweck: Zielverteilungen über Splits.
- Input: Train/Val/Test.
- Output: Tabelle mit `order_rate` und Quantity-Stats.
- Was passiert intern: Splitweise Aggregation, bei `quantity` nur `order==1`.
- Warum ist das methodisch wichtig: Basis für Distribution Shift Diskussion.
- Risiken / Stolperstellen: Nur deskriptiv, keine formalen Shift-Tests.
- Einfacher Verteidigungssatz: „Wir prüfen Zielverteilungen splitweise, um zeitliche Verschiebungen sichtbar zu machen.“

#### `audit_feature_sets(feature_matrices)`

- Zweck: Matrix-Überblick.
- Input: Dict aus DataFrame/Series.
- Output: Tabelle mit Zeilen/Spalten je Matrix.
- Was passiert intern: Typabhängige Verdichtung.
- Warum ist das methodisch wichtig: Vollständigkeitscheck nach Matrixbau.
- Risiken / Stolperstellen: Keine inhaltliche Feature-Prüfung.
- Einfacher Verteidigungssatz: „Wir protokollieren jedes Matrix-Artefakt mit Form, damit nichts still fehlt.“

#### `audit_dropped_features()`

- Zweck: Dokumentiert bewusst ausgeschlossene Features.
- Input: Keine.
- Output: Statische Begründungstabelle.
- Was passiert intern: Definierte Liste wird als DataFrame zurückgegeben.
- Warum ist das methodisch wichtig: Erklärt Verbote transparent.
- Risiken / Stolperstellen: Liste muss bei Regeländerungen mitgezogen werden.
- Einfacher Verteidigungssatz: „Unsere Drop-Entscheide sind explizit dokumentiert und nicht implizit.“

#### `run_full_audit(...)`

- Zweck: Führt alle Audit-Teile in einem Lauf aus.
- Input: Rohdaten, Splits, Matrizen.
- Output: Report-Dictionary.
- Was passiert intern: Orchestriert alle Auditfunktionen und druckt Kurzsummary.
- Warum ist das methodisch wichtig: Standardisierte Qualitätsabnahme pro Pipeline-Lauf.
- Risiken / Stolperstellen: Audit erkennt keine kausalen Fehler, nur Symptome.
- Einfacher Verteidigungssatz: „Audit ist unsere standardisierte Qualitätsabnahme vor Weitergabe.“

## `Preprocessing/pid_segment.py`

### Rolle im Gesamtprozess

`pid_segment` wird auf Train gelernt, auf alle Splits übertragen und dient später in Sampling und als Feature (conditional). Damit verbindet die Datei Datenrepräsentation und Sampling-Strategie.

### Wichtige Inputs

- `df_train` mit `pid`
- Mapping `pid -> segment`

### Wichtige Outputs

- `pid_segment_map`
- Spalte `pid_segment` in allen Splits
- Optional CSV-Mapping

### Funktionen

#### `fit_pid_segment(df_train)`

- Zweck: Segmentiert Produkte in Head/Mid/Tail nach Eventhäufigkeit.
- Input: Train-DataFrame.
- Output: Dict `pid -> segment`.
- Was passiert intern:
  - Gruppiert nach `pid` und zählt Events.
  - Sortiert deterministisch (`n_events` absteigend, dann `pid` aufsteigend).
  - Top 10% = Head, nächste 40% = Mid, Rest = Tail.
- Warum ist das methodisch wichtig:
  - Liefert strukturierte Populationsinformation für Stratified Sampling.
  - Train-only Fit vermeidet Test-Einfluss.
- Risiken / Stolperstellen:
  - Segmentgrenzen sind heuristisch.
  - Bei stark schiefen Verteilungen kann „Tail“ sehr heterogen werden.
- Einfacher Verteidigungssatz: „Wir lernen Produktsegmente nur auf Train und nutzen sie als kontrollierte Strukturvariable.“

#### `apply_pid_segment(df, pid_segment_map)`

- Zweck: Schreibt `pid_segment` in DataFrame.
- Input: `df`, Mapping.
- Output: DataFrame mit Segmentspalte.
- Was passiert intern: Map mit Fallback `Tail` für unbekannte PIDs.
- Warum ist das methodisch wichtig: Verhindert fehlende Segmente in neuen Splits.
- Risiken / Stolperstellen: Viele unbekannte PIDs verschieben Segmentverteilung.
- Einfacher Verteidigungssatz: „Unbekannte PIDs fallen konservativ in Tail statt NaN zu erzeugen.“

#### `save_pid_segment_map(pid_segment_map, path)`

- Zweck: Persistiert Segmentmapping.
- Input: Dict, Pfad.
- Output: CSV.
- Was passiert intern: Dict zu DataFrame und Write.
- Warum ist das methodisch wichtig: Reproduzierbarkeit.
- Risiken / Stolperstellen: Mapping-Datei und Modelllauf müssen zusammenpassen.
- Einfacher Verteidigungssatz: „Das Segmentmapping wird versionierbar abgespeichert.“

#### `load_pid_segment_map(path)`

- Zweck: Lädt Persistenz zurück.
- Input: Pfad.
- Output: Dict.
- Was passiert intern: CSV lesen, Typkonversion.
- Warum ist das methodisch wichtig: Konsistente Wiederverwendung.
- Risiken / Stolperstellen: Veraltetes Mapping.
- Einfacher Verteidigungssatz: „Wir können denselben Segmentierungsstand in späteren Läufen wiederverwenden.“

## `Preprocessing/feature_engineering_safe.py`

### Rolle im Gesamtprozess

Diese Datei erzeugt leakage-freie Features und train-only-fit Artefakte (Binning/Frequency). Sie läuft nach Split und vor Matrixbau. Die Ergebnisse bilden den Kern der SAFE-Matrizen und Grundlage für spätere Conditional-Erweiterung.

### Wichtige Inputs

- Splits `df_train`, `df_val`, `df_test`
- Normalisierte Spalten aus Preprocessing

### Wichtige Outputs

- Safe Feature-Spalten in allen Splits
- Metadaten: `bin_edges`, `manufacturer_freq_map`

### Funktionen

#### `extract_group_parts(df)`

- Zweck: Extrahiert `group12` und `group34`.
- Input: `group_clean` oder `group`.
- Output: Zwei Gruppenteile.
- Was passiert intern: String-Slicing, Padding, Missing-Fallback.
- Warum ist das methodisch wichtig: Strukturierte Gruppenmerkmale für Modell und Aggregationen.
- Risiken / Stolperstellen: Garbage-In bleibt als Garbage strukturiert.
- Einfacher Verteidigungssatz: „Wir erzwingen ein stabiles 2+2-Gruppenschema für nachgelagerte Features.“

#### `add_day_cycles(df)`

- Zweck: Zyklische Tagesfeatures (`day_7`, `day_14`, `day_30`).
- Input: `day`.
- Output: Drei Zyklusspalten.
- Was passiert intern: Modulo-Arithmetik.
- Warum ist das methodisch wichtig: Wocheneffekte/periodische Muster modellierbar.
- Risiken / Stolperstellen: Kein explizites Feiertagswissen.
- Einfacher Verteidigungssatz: „Wir geben dem Modell periodische Zeitstruktur ohne Historienleckage.“

#### `_parse_single_content(raw)` (interne Hilfsfunktion)

- Zweck: Zerlegt einen einzelnen `content`-Wert in Packungslogik.
- Input: `raw`.
- Output: `(is_multipack, pack_n, pack_size, pack_total_size)`.
- Was passiert intern:
  - Split über `X/x`, numerische Teile extrahieren.
  - Einzelwert => `pack_n=1`, `pack_size=value`.
  - Mehrere Faktoren => Produkt der vorderen Faktoren als `pack_n`, letzter Faktor als `pack_size`.
- Warum ist das methodisch wichtig: Einheitliche Mengenskalierung für per-unit Preisfeatures.
- Risiken / Stolperstellen:
  - Parser ist heuristisch.
  - `pack_n` wird über `int(...)` gekürzt.
  - Unklare Sondernotationen landen ggf. in NaN.
- Einfacher Verteidigungssatz: „Wir normalisieren uneinheitliche Content-Formate in eine konsistente Packungsmetrik.“
- Mini-Beispiel:

| content   | is_multipack | pack_n | pack_size | pack_total_size |
| --------- | -----------: | -----: | --------: | --------------: |
| `80`      |            0 |      1 |        80 |              80 |
| `10X1`    |            1 |     10 |         1 |              10 |
| `6X4X200` |            1 |     24 |       200 |            4800 |

#### `parse_content(df)`

- Zweck: Wendet Content-Parser auf alle Zeilen an.
- Input: `df` mit `content_clean` oder `content`.
- Output: Vier Packungsfeatures.
- Was passiert intern: `apply(_parse_single_content)` und Spaltenzuweisung.
- Warum ist das methodisch wichtig: Bringt unstrukturierte Textangabe in modellierbare Form.
- Risiken / Stolperstellen: Parsingqualität hängt stark vom Rohformat ab.
- Einfacher Verteidigungssatz: „Die Batch-Parsing-Stufe macht aus Text ein quantitatives Packungsgerüst.“
- Mini-Beispiel (vereinfachter Ablauf):
  - Zeile 1: `content_clean='5X10'` -> `pack_total_size=50`
  - Zeile 2: `content_clean='MISSING'` -> `pack_total_size=NaN`
  - Zeile 3: `content_clean='20'` -> `pack_total_size=20`

#### `add_has_campaign(df)`

- Zweck: Binärfeature für Kampagnenpräsenz.
- Input: `campaignIndex_norm`.
- Output: `has_campaign`.
- Was passiert intern: `campaignIndex_norm != 'NONE'`.
- Warum ist das methodisch wichtig: Verdichtet kategoriale Kampagneninformation.
- Risiken / Stolperstellen: Kampagnenintensität wird nicht abgebildet.
- Einfacher Verteidigungssatz: „Wir kodieren Kampagnenpräsenz als robustes Basis-Signal.“

#### `add_price_features(df)`

- Zweck: Preisrelationen und Vergleichsflags.
- Input: `price`, `rrp`, `competitorPrice`.
- Output: `price_diff`, `price_discount`, `competitorPrice_discount`, `price_discount_diff`, `is_lower_price`, `is_discount`, `is_greater_discount`.
- Was passiert intern:
  - Rechenregeln gemäß Variable Dictionary.
  - `rrp == 0` wird zu NaN-Guard (`safe_rrp`).
  - Vergleichsflags sind NaN-sensitiv.
- Warum ist das methodisch wichtig:
  - Modelliert Preisposition relativ zum Markt und zur UVP.
  - Trennt absolute von relativen Preissignalen.
- Risiken / Stolperstellen:
  - Fehlender `competitorPrice` propagiert NaNs in Vergleichsfeatures.
  - Flags können stark korrelieren (später Redundanzthema).
- Einfacher Verteidigungssatz: „Wir erfassen nicht nur den Preis selbst, sondern seine relative Marktposition.“
- Mini-Beispiel:

| price | competitorPrice | rrp | price_diff | price_discount | is_lower_price |
| ----: | --------------: | --: | ---------: | -------------: | -------------: |
|     9 |              10 |  12 |         -1 |           0.25 |              1 |
|     9 |             NaN |  12 |        NaN |           0.25 |            NaN |

#### `add_per_unit_features(df)`

- Zweck: Preise auf Einheit normieren.
- Input: Preisgrößen + `pack_total_size`.
- Output: `price_per_unit`, `rrp_per_unit`, `competitorPrice_per_unit`.
- Was passiert intern: Division gegen `safe_total` (`pack_total_size > 0`).
- Warum ist das methodisch wichtig: Vergleichbarkeit über verschiedene Packungsgrößen.
- Risiken / Stolperstellen: Schlechte Content-Parsing-Werte wirken direkt weiter.
- Einfacher Verteidigungssatz: „Per-unit Features machen unterschiedliche Packungsgrößen vergleichbar.“

#### `fit_binning_edges(df_train, n_bins)`

- Zweck: Lernt Quantilgrenzen für Preisbinning auf Train.
- Input: `df_train`, `n_bins`.
- Output: Dict mit Kanten für `price_diff` und `price_discount`.
- Was passiert intern: `pd.qcut(..., duplicates='drop')`, Kanten auf `[-inf, inf]` erweitert.
- Warum ist das methodisch wichtig: Nichtlineare Preiswirkungen als robuste Kategorien.
- Risiken / Stolperstellen: Bei vielen Duplikaten sinkt effektive Bin-Anzahl.
- Einfacher Verteidigungssatz: „Binning wird nur auf Train gelernt und dann fix auf alle Splits angewendet.“
- Mini-Beispiel:
  - Train-`price_diff`: `[-2, -1, 0, 1, 5]`, `n_bins=3` -> Kanten z. B. `[-inf, -1, 1, inf]`.
  - Val-`price_diff=0.4` -> Bin `Q02`.

#### `apply_binned_features(df, bin_edges)`

- Zweck: Wendet gelernte Bins auf Daten an.
- Input: DataFrame, `bin_edges`.
- Output: `price_diff_bin`, `discount_bin`.
- Was passiert intern: `pd.cut` mit Labels `Q01...`; NaN-spezifische Label.
- Warum ist das methodisch wichtig: Konsistente Kategorisierung über Splits.
- Risiken / Stolperstellen: Falsche oder fehlende Kanten führen zu pauschalen Labeln.
- Einfacher Verteidigungssatz: „Wir vermeiden Split-abhängige Bin-Drift durch train-only Kanten.“

#### `fit_manufacturer_frequency(df_train)`

- Zweck: Lernt Herstellerhäufigkeit.
- Input: `manufacturer` auf Train.
- Output: Mapping `manufacturer -> freq`.
- Was passiert intern: `value_counts(normalize=True)`.
- Warum ist das methodisch wichtig: Verdichtet High-Cardinality-Information.
- Risiken / Stolperstellen: Frequenz ist datensatzspezifisch.
- Einfacher Verteidigungssatz: „Herstellerfrequenz ist ein kompaktes Stabilitätssignal aus dem Train-Split.“

#### `apply_manufacturer_frequency(df, freq_map)`

- Zweck: Wendet Herstellerfrequenzen an.
- Input: DataFrame, `freq_map`.
- Output: `manufacturer_freq`.
- Was passiert intern: Mapping mit Fallback `0.0`.
- Warum ist das methodisch wichtig: Keine Missingness bei unbekannten Herstellern.
- Risiken / Stolperstellen: Viele unbekannte Hersteller führen zu vielen `0.0`.
- Einfacher Verteidigungssatz: „Unbekannte Hersteller werden konservativ behandelt statt ausgelassen.“

#### `run_all_safe_features(df_train, df_val, df_test)`

- Zweck: Orchestriert alle Safe-Feature-Bausteine.
- Input: Drei Splits.
- Output: Drei angereicherte Splits + Metadaten.
- Was passiert intern:
  - Pro Split: Gruppen, Zyklen, Content, Kampagne, Preis, Per-Unit.
  - Train-fit/All-apply: Binning und Manufacturer-Frequency.
- Warum ist das methodisch wichtig:
  - Strikte Trennung zwischen zeilenbasierten Features und train-fitted Artefakten.
  - Grundlage für SAFE-Matrixbau und späteren Vergleich mit CONDITIONAL.
- Risiken / Stolperstellen: Fehler in früheren Normspalten propagieren in viele Features.
- Einfacher Verteidigungssatz: „SAFE-Features sind unsere robuste baseline ohne targethistorische Information.“

## `Preprocessing/feature_engineering_conditional.py`

### Rolle im Gesamtprozess

Dieses Modul erzeugt historien- und targetnahe Features unter Leakage-Schutz. Es läuft nur im Modus `safe_plus_conditional` und erweitert die Safe-Basis um kumulative, aggregierte und OOF-kodierte Signale. Für das Pipeline-Verständnis ist es das methodisch kritischste Modul.

### Wichtige Inputs

- `df_train`, `df_val`, `df_test` nach Safe-Features
- Zeitspalte `day`, Gruppen wie `pid`, `availability`, `day_7`
- Targets `order` bzw. `quantity`
- OOF-Parameter aus `config.py`

### Wichtige Outputs

- Conditional-Spalten in allen Splits
- Metadaten: `global_aggregation_maps`, `oof_fold_info`, `full_train_encodings`

### Funktionen

#### `_require_columns(df, cols, context)` (interne Hilfsfunktion)

- Zweck: Pflichtspalten prüfen.
- Input: DataFrame, Spaltenliste, Kontext.
- Output: None oder Fehler.
- Was passiert intern: Missing-Liste berechnen.
- Warum ist das methodisch wichtig: Fail-fast für saubere Pipelineabhängigkeiten.
- Risiken / Stolperstellen: Neue Features brauchen ggf. neue Pflichtspaltenlisten.
- Einfacher Verteidigungssatz: „Wir prüfen Vorbedingungen explizit, bevor wir Historienfeatures berechnen.“

#### `_week_block(day, start)` (interne Hilfsfunktion)

- Zweck: Tage in relative Wochenblöcke mappen.
- Input: `day`, `start`.
- Output: Blockindex.
- Was passiert intern: `((day - start) // 7) + 1`.
- Warum ist das methodisch wichtig: Zeitstruktur für Forward OOF.
- Risiken / Stolperstellen: Abhängigkeit vom korrekt gesetzten `TRAIN_DAY_START`.
- Einfacher Verteidigungssatz: „Forward OOF braucht eine deterministische Zeitblock-Definition.“

#### `compute_cumulative_features(df_train, df_val, df_test)`

- Zweck: Historienbasierte kumulative PID-Features (`day-1` Logik).
- Input: Drei Splits mit `pid`, `day`, `click`, `basket`, `order`.
- Output: Neue Spalten `pid_total_events`, `click_time`, `basket_time`, `order_time`, `num_pid_order`.
- Was passiert intern:
  - Splits werden per MultiIndex konkateniert.
  - Sortierung nach `pid`, `day`.
  - Nur Zeilen `day >= TRAIN_DAY_START` tragen zur Historie bei.
  - `cumsum - current` verhindert Self Leakage.
  - Danach sauberer Split zurück in Train/Test/Val.
- Warum ist das methodisch wichtig:
  - Modell bekommt Verlaufssignal je PID.
  - Kein Zugriff auf aktuelle Zeile als Historie.
  - Zeitlicher Verlauf bleibt konsistent über Splitgrenzen.
- Risiken / Stolperstellen:
  - Sehr kalte PIDs haben wenig Historie.
  - Alias `num_pid_order` ist redundant zu `order_time`.
  - Unklar im Code: Warum Alias `num_pid_order` zusätzlich gehalten wird, obwohl identisch zu `order_time`.
- Einfacher Verteidigungssatz: „Kumulative Features nutzen nur Vergangenheitsinformation, weil wir überall `cumsum - current` einsetzen.“
- Mini-Beispiel:

| pid | day | order | order_time |
| --- | --: | ----: | ---------: |
| A   |  30 |     1 |          0 |
| A   |  31 |     0 |          1 |
| A   |  40 |     1 |          1 |

Interpretation: Am Tag 40 hat PID A genau eine frühere Bestellung im Zähler.

#### `fit_global_aggregations(df_train)`

- Zweck: Lernt einfache full-train Gruppenmittelwerte für `order`.
- Input: Train mit `group12`, `group34`, `day_7`, `order`.
- Output: Mapping-Dict (`group12_order`, `group34_order`, `week_order`).
- Was passiert intern: `groupby(...).mean()` plus globaler Fallback-Mittelwert.
- Warum ist das methodisch wichtig: Schnelles Aggregationssignal als Ergänzung.
- Risiken / Stolperstellen:
  - Full-train-fit erzeugt Self Leakage auf Train.
  - Daher methodisch schwächer als OOF für Train-Bewertung.
- Einfacher Verteidigungssatz: „Diese Aggregationen sind Hilfssignale; die leakage-kritische Trainkodierung erfolgt über Forward OOF.“
- Mini-Beispiel:
  - Train: `group12='AB'` hat `order`-Mittel 0.3.
  - Jede Zeile mit `group12='AB'` bekommt `group12_order=0.3`.

#### `apply_global_aggregations(df, mappings)`

- Zweck: Wendet globale Aggregationsmappings an.
- Input: DataFrame + Mapping.
- Output: Neue Aggregationsspalten.
- Was passiert intern: Mapping je Gruppenspalte, Unbekannte -> `global_mean`.
- Warum ist das methodisch wichtig: Keine NaN-Lücken bei unbekannten Gruppen.
- Risiken / Stolperstellen: Starke Verteilungsshifts können Fallback übernutzen.
- Einfacher Verteidigungssatz: „Unbekannte Gruppen bekommen einen stabilen globalen Fallback statt Missing-Werte.“

#### `_time_aware_forward_oof(df, group_col, target_col, cold_start, history_mask)` (interne Hilfsfunktion)

- Zweck: Leakagesichere Train-Encodings mit expandierendem Zeitfenster.
- Input: Train-Frame, Gruppenspalte, Zielspalte, Cold-Start-Wert, optional `history_mask`.
- Output: `(encoded_series, fold_info)`.
- Was passiert intern:
  - Bildet Wochenblöcke.
  - Erster Block: immer `cold_start`.
  - Spätere Blöcke: nur Historie aus früheren Blöcken.
  - Optional maskierte Historie (z. B. REG nur `order==1`).
  - Unbekannte Gruppen im Block -> historischer globaler Mittelwert.
- Warum ist das methodisch wichtig:
  - Verhindert Future Leakage und Self Leakage im Train-Encoding.
  - Erzeugt realistisches „nur Vergangenheit sichtbar“-Signal.
- Risiken / Stolperstellen:
  - Bei dünner Historie dominieren Fallbacks.
  - Unklar im Code: Bei sehr kurzer Historie können viele Blöcke faktisch nahe am Cold-Start bleiben.
- Einfacher Verteidigungssatz: „Forward OOF erzwingt, dass jede Train-Zeile nur aus der Vergangenheit gelernt wird.“
- Mini-Beispiel:

| block | Historie für Kodierung | Ergebnisidee                            |
| ----: | ---------------------- | --------------------------------------- |
|     1 | keine                  | alle = `cold_start`                     |
|     2 | Block 1                | gruppenspezifische Mittel aus Block 1   |
|     3 | Block 1+2              | gruppenspezifische Mittel aus Block 1+2 |

#### `_fit_full_train_encoding(df_train, group_col, target_col, global_fallback, history_mask)` (interne Hilfsfunktion)

- Zweck: Lernt finales Encoding auf gesamtem Train für Val/Test-Anwendung.
- Input: Train-Daten + Parameter.
- Output: Dict mit `group_means`, `global_mean`.
- Was passiert intern: Optional maskiert, dann group mean + global mean.
- Warum ist das methodisch wichtig: Konsistente Anwendung auf neue Daten.
- Risiken / Stolperstellen: Kein OOF-Mechanismus, daher nur für Nicht-Train anwenden.
- Einfacher Verteidigungssatz: „Train wird OOF-kodiert, Val/Test erhalten ein stabiles Full-Train-Encoding.“

#### `_apply_encoding(df, group_col, col_name, encoding)` (interne Hilfsfunktion)

- Zweck: Wendet vortrainiertes Encoding an.
- Input: Ziel-DataFrame, Gruppenspalte, Zielname, Encoding-Dict.
- Output: Neue Spalte `col_name`.
- Was passiert intern: Mapping mit `global_mean` Fallback.
- Warum ist das methodisch wichtig: Einheitliches Apply-Verhalten.
- Risiken / Stolperstellen: Falscher Gruppenschlüssel erzeugt Fallback-lastige Spalte.
- Einfacher Verteidigungssatz: „Apply-Logik ist zentral vereinheitlicht, damit Encodings konsistent bleiben.“

#### `run_all_conditional_features(df_train, df_val, df_test)`

- Zweck: Orchestriert alle Conditional-Features inklusive OOF.
- Input: Drei Splits nach Safe-Features.
- Output: Drei erweiterte Splits + Metadaten.
- Was passiert intern:
  - Schritt 1: `compute_cumulative_features(...)`
  - Schritt 2: `fit_global_aggregations(df_train)` + `apply_global_aggregations(...)`
  - Schritt 3: OOF-Loop über `_OOF_SPECS`:
    - Train: `_time_aware_forward_oof(...)`
    - Val/Test: `_fit_full_train_encoding(...)` + `_apply_encoding(...)`
- Warum ist das methodisch wichtig:
  - Bündelt alle targetnahen Features unter kontrollierter Leakage-Logik.
  - Liefert Metadaten zur Reproduzierbarkeit (`oof_fold_info` etc.).
- Risiken / Stolperstellen:
  - Höhere Komplexität als SAFE.
  - Falsch verstandene OOF-Features führen schnell zu fehlerhaften Argumentationen.
- Einfacher Verteidigungssatz: „Conditional-Features sind leistungsfähig, weil sie Historie nutzen, aber sie werden bei uns strikt leakage-kontrolliert gebaut.“

## `Preprocessing/main_build_datasets.py`

### Rolle im Gesamtprozess

Dies ist der zentrale Orchestrator der gesamten Datenpipeline. Er definiert die ausführbare Reihenfolge und trennt `safe_only` von `safe_plus_conditional`. Damit ist er die Referenz für den realen technischen Ablauf.

### Wichtige Inputs

- CLI-`--mode`
- Rohdatenpfade aus `config.py`

### Wichtige Outputs

- Exportierte Matrizen/Audits/Metadaten
- Optional Orange-Exporte

### Funktionen

#### `run_safe_only()`

- Zweck: Vollständiger Lauf nur mit SAFE-Features.
- Input: Keine direkten Parameter (nutzt Konfiguration und Module).
- Output: Datasets, Audits, Metadaten.
- Was passiert intern: Orchestriert Schritte 1-11 (ohne Conditional- und Orange-Teil).
- Warum ist das methodisch wichtig: Liefert baseline ohne targethistorische Features.
- Risiken / Stolperstellen: Keine Orange-Dateien in diesem Modus.
- Einfacher Verteidigungssatz: „`safe_only` ist unser robuste Vergleichsbaseline.“

#### `run_safe_plus_conditional()`

- Zweck: Vollständiger Lauf inkl. Conditional-Features und Orange-Export.
- Input: Keine direkten Parameter.
- Output: Safe+Conditional Matrizen, Audits, Metadaten, Orange-CSV.
- Was passiert intern:
  - Baut erst SAFE, dann CONDITIONAL.
  - Vereinigt Matrizen (`all_matrices`).
  - Exportiert danach Orange.
- Warum ist das methodisch wichtig: End-to-end Variante für finalen Feature-Stack.
- Risiken / Stolperstellen: Höhere Leakage-Sensitivität bei Fehlkonfiguration.
- Einfacher Verteidigungssatz: „`safe_plus_conditional` ist unser vollständiger Produktionskandidat mit zusätzlicher Historieninformation.“

#### `_export_all(matrices, sampling_result, reports, fe_metadata)` (interne Hilfsfunktion)

- Zweck: Persistiert alle Artefakte.
- Input: Matrix-Dict, Sampling-Ergebnisse, Reports, Metadaten.
- Output: Dateien in `outputs/`.
- Was passiert intern: Parquet/CSV/Text plus JSON-Metadaten für Binning/OOF/Aggregationen.
- Warum ist das methodisch wichtig: Vollständiger Audit- und Reproduzierbarkeitspfad.
- Risiken / Stolperstellen: Schemaänderungen erfordern kompatible Leseroutinen.
- Einfacher Verteidigungssatz: „Wir exportieren nicht nur Matrizen, sondern auch die Lernartefakte, damit die Herleitung nachvollziehbar bleibt.“

#### `main()`

- Zweck: CLI-Einstieg.
- Input: `--mode`.
- Output: Startet passenden Pipelinezweig.
- Was passiert intern: Argumentparsing und Verzweigung.
- Warum ist das methodisch wichtig: Reproduzierbare Pipelineausführung.
- Risiken / Stolperstellen: Falscher Modus führt zu unerwarteten Outputs.
- Einfacher Verteidigungssatz: „Die Pipeline ist über einen klaren Modus-Parameter deterministisch steuerbar.“

## `Sampling/split.py`

### Rolle im Gesamtprozess

Dieses Modul setzt die zeitliche Grundstruktur. Da viele spätere Features zeitabhängig sind, ist `run_split(df)` kein rein technischer Schritt, sondern methodischer Kern der Evaluationslogik.

### Wichtige Inputs

- DataFrame mit `day`
- Zeitgrenzen aus `config.py`

### Wichtige Outputs

- `df_train`, `df_validation`, `df_test`

### Funktionen

#### `run_split(df)`

- Zweck: Chronologischer Split in drei Zeitfenster.
- Input: DataFrame mit `day`.
- Output: Tuple `(df_train, df_validation, df_test)`.
- Was passiert intern: Filter nach Day-Intervallen; Zeilen außerhalb werden gedroppt.
- Warum ist das methodisch wichtig:
  - Erhält zeitliche Kausalität.
  - Schafft konsistenten Bezug für Forward OOF und historische Features.
- Risiken / Stolperstellen:
  - Frühere Tage (z. B. <26) gehen verloren.
  - Grenzwerte müssen exakt zur Problemstellung passen.
- Einfacher Verteidigungssatz: „Unser Split simuliert echten Zukunftseinsatz statt zufälliger Datenmischung.“

## `Sampling/sampling.py`

### Rolle im Gesamtprozess

Sampling dient ausschließlich Prototyping auf Train. Es reduziert Rechenlast für schnelle Iterationen und erzeugt Audit-Tabellen zur Verteilungsprüfung. Finale Modellbewertung darf sich nicht darauf stützen.

### Wichtige Inputs

- `df_train` (nach Feature-Engineering)
- Samplingraten aus `config.py`
- Für REG zusätzlich Stage-2-Maske

### Wichtige Outputs

- `train_cls_sample`, `train_reg_sample`
- `audit_cls`, `audit_reg`

### Funktionen

#### `add_week_block(df)`

- Zweck: Fügt relative Wochenblöcke hinzu.
- Input: `day`.
- Output: `week_block`.
- Was passiert intern: Gleiches Blockschema wie OOF-Basis.
- Warum ist das methodisch wichtig: Zeitstruktur im Sampling-Strata.
- Risiken / Stolperstellen: Abhängigkeit von korrektem `TRAIN_DAY_START`.
- Einfacher Verteidigungssatz: „Zeitblöcke halten Sampling zeitlich strukturtreu.“

#### `sample_stratified(df, frac, strata_cols, seed)`

- Zweck: Stratifiziertes Sampling mit Schutz kleiner Strata.
- Input: DataFrame, Fraktion, Strata-Spalten, Seed.
- Output: Sample-DataFrame.
- Was passiert intern:
  - Gruppiert nach Strata.
  - Für kleine Strata (`n <= 2`) werden alle Zeilen behalten.
  - Sonst wird `k = max(1, round(n * frac))` gezogen.
- Warum ist das methodisch wichtig:
  - Verhindert, dass seltene Strata komplett verschwinden.
  - Begrenzt Übergewichtung durch naive Samplingregeln.
- Risiken / Stolperstellen:
  - Sehr viele kleine Strata können effektive Samplingrate erhöhen.
  - Nicht für finale Evaluation geeignet.
- Einfacher Verteidigungssatz: „Unser Stratified Sampling schützt seltene Gruppen statt sie zufällig wegzuwerfen.“
- Mini-Beispiel:

| Stratum |   n | frac=0.3 | Ergebnis  |
| ------- | --: | -------: | --------- |
| A       |   1 |      0.3 | behalte 1 |
| B       |   2 |      0.3 | behalte 2 |
| C       |  10 |      0.3 | ziehe 3   |

#### `sample_cls(df_train, frac)`

- Zweck: CLS-Trainsample.
- Input: Train-Frame.
- Output: Sample für CLS.
- Was passiert intern: `add_week_block` + `sample_stratified(..., ['week_block','order','pid_segment'])`.
- Warum ist das methodisch wichtig: Erhält Klassen- und Segmentstruktur.
- Risiken / Stolperstellen: Bei extremer Imbalance kann dennoch Rauschen entstehen.
- Einfacher Verteidigungssatz: „CLS-Sampling respektiert Zeit, Zielklasse und Produktsegment gleichzeitig.“

#### `sample_reg(df_train, frac)`

- Zweck: REG-Trainsample nur aus Stage-2.
- Input: Train-Frame.
- Output: REG-Sample.
- Was passiert intern: `get_reg_mask(df)` filtert Stage-2, danach stratifiziert über `week_block`, `quantity_class`, `pid_segment`.
- Warum ist das methodisch wichtig: REG-Sample ist echte Teilmenge der Stage-2-Population.
- Risiken / Stolperstellen: Bei kleiner Stage-2 kann Sampling instabil werden.
- Einfacher Verteidigungssatz: „REG wird vor dem Sampling strikt auf Stage-2 gefiltert.“

#### `audit_sample_vs_population(df_pop, df_sample, cols)`

- Zweck: Verteilungsvergleich Sample vs. Population.
- Input: Population, Sample, Spaltenliste.
- Output: Tabelle mit `pop_frac`, `sample_frac`, `abs_diff`.
- Was passiert intern: Relative Häufigkeiten pro Wert.
- Warum ist das methodisch wichtig: Quantifiziert Sampling-Verzerrung.
- Risiken / Stolperstellen: Nur univariate Audits.
- Einfacher Verteidigungssatz: „Wir prüfen Sampling-Bias explizit mit Verteilungsdifferenzen.“

#### `run_sampling(df_train)`

- Zweck: Führt CLS/REG-Sampling und Audits zusammen aus.
- Input: Train-Frame.
- Output: Dict mit Samples und Auditframes.
- Was passiert intern: `sample_cls`, `sample_reg`, Auditberechnung, Max-Diff-Logging.
- Warum ist das methodisch wichtig: Definiert den prototypischen Sampling-Workflow zentral.
- Risiken / Stolperstellen: Gefahr, Sampling-Resultate als final zu überinterpretieren.
- Einfacher Verteidigungssatz: „Sampling ist bei uns explizit als Prototyping markiert und wird auditierbar gemacht.“
- Mini-Beispiel (vereinfachter Ablauf):
  - Train 1.000.000 Zeilen -> CLS-Sample ~300.000.
  - REG-Population 80.000 Zeilen -> REG-Sample ~24.000.
  - Audit meldet maximale Abweichung je Merkmal.

#### `_require_columns(df, cols)` (interne Hilfsfunktion)

- Zweck: Pflichtspaltenprüfung.
- Input: DataFrame, Spaltenliste.
- Output: None oder Fehler.
- Was passiert intern: Missing-Check.
- Warum ist das methodisch wichtig: Verhindert stilles Sampling auf falschem Schema.
- Risiken / Stolperstellen: Muss bei neuen Sampling-Strata gepflegt werden.
- Einfacher Verteidigungssatz: „Sampling startet nur mit vollständigen Pflichtspalten.“

## `Sampling/feature_sets.py`

### Rolle im Gesamtprozess

Dieses Modul ist die Vertragsfläche zwischen Feature-Engineering und Modellierung: Welche Features in welche Matrix gehen, welche Zeilen für REG gültig sind und welche Guardrails gelten. Es ist zentral für fachliche Verteidigung, weil hier Stage- und Leakage-Regeln operationalisiert werden.

### Wichtige Inputs

- Feature-Listen aus `config.py`
- Splits als DataFrames
- Verbotene Featurelisten

### Wichtige Outputs

- SAFE- und CONDITIONAL-Matrizen (`X_*`, `y_*`)
- Parquet-Exports
- Zusammenfassungen

### Funktionen

#### `get_feature_list(set_name)`

- Zweck: Liefert registrierte Featureliste.
- Input: Setname.
- Output: Kopie der Liste.
- Was passiert intern: Lookup in `_REGISTRY`.
- Warum ist das methodisch wichtig: Zentralisierte Set-Definition.
- Risiken / Stolperstellen: Falscher Setname.
- Einfacher Verteidigungssatz: „Feature-Sets sind zentral registriert, nicht ad hoc in Scripts verteilt.“

#### `validate_feature_list(df, feature_list, set_name)`

- Zweck: Prüft, ob alle Features existieren.
- Input: DataFrame und Featureliste.
- Output: None oder Fehler.
- Was passiert intern: Missing-Spaltenliste.
- Warum ist das methodisch wichtig: Verhindert stillen Featureverlust.
- Risiken / Stolperstellen: Neue Spalten müssen in vorgelagerten Schritten erzeugt sein.
- Einfacher Verteidigungssatz: „Wir erlauben keinen impliziten Featureausfall.“

#### `assemble_X_y(df, feature_list, target_col, set_name)`

- Zweck: Baut valides `(X, y)` Paar.
- Input: DataFrame, Features, Zielspalte.
- Output: `X`, `y`.
- Was passiert intern:
  - Featureexistenz prüfen.
  - Zielspalte prüfen.
  - Forbidden-Guard stageabhängig (`order` vs. `quantity`).
  - Defensive Kopien zurückgeben.
- Warum ist das methodisch wichtig: Einheitliche Erzeugung von Modellinputs.
- Risiken / Stolperstellen: Falscher `target_col` führt zu falschem Verbotssatz.
- Einfacher Verteidigungssatz: „Jede Matrix entsteht über denselben Guarded-Build-Prozess.“

#### `get_reg_mask(df)`

- Zweck: Definiert Stage-2-Zeilen für REG.
- Input: DataFrame mit `order`, `quantity` und optional `qty_suspicious`.
- Output: Bool-Maske.
- Was passiert intern: `(order == 1) & quantity.notna()` und falls vorhanden `qty_suspicious == 0`.
- Warum ist das methodisch wichtig:
  - Single Source of Truth für REG-Population.
  - Wird in Matrixbau, Sampling und Export wiederverwendet.
- Risiken / Stolperstellen:
  - Wenn `qty_suspicious` fehlt, wird nur auf `order`/`quantity` gefiltert.
  - Unklar im Code: Es gibt keine harte Pflicht, dass `qty_suspicious` immer vorhanden sein muss.
- Einfacher Verteidigungssatz: „REG-Zeilen werden überall über dieselbe zentrale Maske bestimmt.“
- Mini-Beispiel:

| order | quantity | qty_suspicious | reg_mask |
| ----: | -------: | -------------: | -------: |
|     1 |      2.0 |              0 |     True |
|     1 |      NaN |              1 |    False |
|     0 |      NaN |              0 |    False |

#### `build_safe_feature_matrices(df_train, df_val, df_test)`

- Zweck: Erzeugt alle SAFE-Matrizen für CLS und REG.
- Input: Drei Splits.
- Output: Dict mit `X_*`/`y_*` für Base und Expanded.
- Was passiert intern:
  - CLS: baut base + expanded je Split.
  - REG: zuerst `get_reg_mask(df)`, dann base + expanded.
  - Excluded-Logs für ungültige REG-Zeilen.
- Warum ist das methodisch wichtig:
  - Trennung zwischen Stage 1 und Stage 2 ist technisch erzwungen.
  - Bewertungsvergleiche zwischen base/expanded sind sauber reproduzierbar.
- Risiken / Stolperstellen: Leere REG-Splits sind möglich und werden nur gewarnt.
- Einfacher Verteidigungssatz: „SAFE-Matrizen bauen wir stagegetrennt und guardgesichert.“

#### `summarize_feature_sets(feature_matrices)`

- Zweck: Tabellarischer Matrixüberblick.
- Input: Matrix-Dict.
- Output: Summary-DataFrame.
- Was passiert intern: Zeilen-/Spaltenzählung je Objekt.
- Warum ist das methodisch wichtig: Schnellprüfung auf Vollständigkeit.
- Risiken / Stolperstellen: Kein inhaltlicher Qualitätstest.
- Einfacher Verteidigungssatz: „Wir dokumentieren jede erzeugte Matrix mit Shape.“

#### `_print_summary(result)` (interne Hilfsfunktion)

- Zweck: Konsolenkurzbericht.
- Input: Matrix-Dict.
- Output: Print.
- Was passiert intern: Formatiert `X_` und `y_` Einträge.
- Warum ist das methodisch wichtig: Lauftransparenz.
- Risiken / Stolperstellen: Nur für Menschenlesbarkeit.
- Einfacher Verteidigungssatz: „Jeder Lauf zeigt sofort, welche Matrizen erzeugt wurden.“

#### `build_conditional_feature_matrices(df_train, df_val, df_test)`

- Zweck: Erzeugt Conditional-Matrizen für CLS und REG.
- Input: Drei Splits mit bereits vorhandenen Conditional-Spalten.
- Output: Dict mit `X_*_conditional` und Targets.
- Was passiert intern:
  - CLS direkt auf allen Zeilen.
  - REG erneut über `get_reg_mask(df)` gefiltert.
- Warum ist das methodisch wichtig:
  - Conditional-Mehrwert wird als eigener Matrixsatz messbar.
  - Stage-2-Logik bleibt auch hier konsistent.
- Risiken / Stolperstellen: Fehlen Conditional-Spalten, bricht der Build früh.
- Einfacher Verteidigungssatz: „Conditional-Matrizen sind ein separater, validierter Ausbau auf derselben Stage-Logik.“

#### `export_matrices(feature_matrices, output_dir)`

- Zweck: Speichert Matrizen als Parquet.
- Input: Matrix-Dict, Zielordner.
- Output: Parquet-Dateien.
- Was passiert intern: Serien werden vor Export in DataFrames gewandelt.
- Warum ist das methodisch wichtig: Einheitliches Artefaktformat.
- Risiken / Stolperstellen: Downstream erwartet diese Dateinamen.
- Einfacher Verteidigungssatz: „Alle Modellinputs liegen standardisiert als Parquet mit festen Namen vor.“

## `Sampling/orange_export.py`

### Rolle im Gesamtprozess

Dieses Modul bildet den Übergang von Python-Matrizen zu Orange-kompatiblen CSVs. Es ist schema- und leakage-kritisch, weil hier finale Feature-Sets, Stage-2-Regeln und Spaltenverträge erzwungen werden.

### Wichtige Inputs

- `df_train`, `df_val`, `df_test`
- `sampling_result`
- `build_mode`
- Final-Sets aus `config.py`

### Wichtige Outputs

- Standard- und Varianten-CSV-Exporte
- `export_manifest.csv`

### Funktionen

#### `_build_orange_df(df, features, target_col)` (interne Hilfsfunktion)

- Zweck: Baut Orange-Tabelle mit Target als letzte Spalte.
- Input: DataFrame, Featureliste, Zielspalte.
- Output: Export-DataFrame.
- Was passiert intern:
  - Selektiert `features + [target_col]`.
  - Castet konfigurierte Kategorische zu String.
  - Prefix für numerisch aussehende diskrete Kategorien.
- Warum ist das methodisch wichtig: Orange soll diskrete Spalten korrekt behandeln.
- Risiken / Stolperstellen: Falsche Cast-/Prefixregeln können Tool-Interpretation ändern.
- Einfacher Verteidigungssatz: „Wir bereiten Typen explizit Orange-gerecht auf, statt Inferenz dem Tool zu überlassen.“

#### `_validate_export(frames, stage, features, target_col)` (interne Hilfsfunktion)

- Zweck: Validiert Exportkonsistenz je Stage.
- Input: Frame-Dict, Stage, Features, Zielspalte.
- Output: None oder Fehler.
- Was passiert intern:
  - Duplicate-Check.
  - Forbidden-Check.
  - Exakter Spaltenvergleich.
  - Target-NaN-Check.
  - `assert_cross_split_columns(...)`.
- Warum ist das methodisch wichtig: Verhindert Leakage und Schema-Mismatch vor Tool-Export.
- Risiken / Stolperstellen: Strenge Checks brechen bei jeder Drift sofort.
- Einfacher Verteidigungssatz: „Orange-Export hat denselben Guardrail-Standard wie unser Matrixbau.“

#### `export_orange_csvs(df_train, df_val, df_test, sampling_result, build_mode)`

- Zweck: Erzeugt alle Orange-CSV-Dateien plus Manifest.
- Input: Splits, Sampling-Resultat, Build-Modus.
- Output: Mehrere CSV-Dateien + `export_manifest.csv`.
- Was passiert intern:
  - `build_mode` wird validiert.
  - CLS-Frames (full/sample/test/val) werden gebaut und geprüft.
  - REG-Frames werden zuerst über `get_reg_mask(df)` gefiltert.
  - `assert_reg_stage2_only(...)` prüft REG-Quellframes.
  - Standardsets und konservative Varianten werden exportiert.
  - Manifest schreibt Metadaten zu allen Exporten.
- Warum ist das methodisch wichtig:
  - Tool-Handover ist formalisiert und auditierbar.
  - Stage-2-Regeln bleiben bis in den Export erhalten.
- Risiken / Stolperstellen:
  - Fehlendes Sampling-Ergebnis bricht Export.
  - Falsche Final-Set-Definitionen erzeugen harte Fehler.
- Einfacher Verteidigungssatz: „Beim Orange-Export erzwingen wir dieselben Stage- und Leakage-Regeln wie im Modellpfad.“
- Mini-Beispiel (vereinfachter Ablauf):
  1.  CLS: baue `cls_train_full`, `cls_train_sample`, `cls_test`, `cls_val`.
  2.  REG: filtere Splits auf Stage-2, prüfe `assert_reg_stage2_only`.
  3.  Schreibe CSVs + Manifest mit `n_rows`, `feature_set_name`, `sampling_used`.

## `Experiment/feature_selection.py`

### Rolle im Gesamtprozess

Dieses Modul ist ein nachgelagerter Analyse-Schritt auf bereits exportierten Train-Matrizen. Es trifft keine finalen Modellentscheidungen, sondern liefert Filterdiagnostik (Signal, Missingness, Redundanz, Familienebene).

### Wichtige Inputs

- Parquet-Matrizen aus `outputs/datasets`
- Targets `y_train_cls['order']`, `y_train_reg['quantity']`

### Wichtige Outputs

- Filterreports, Redundanz-CSV, Familienreport, Textsummary

### Funktionen

#### `_family_of(feat)` (interne Hilfsfunktion)

- Zweck: Ordnet Feature einer Familie zu.
- Input: Feature-Name.
- Output: Familienname oder `unknown`.
- Was passiert intern: Lookup in `cfg.FEATURE_FAMILIES`.
- Warum ist das methodisch wichtig: Familienberichte/Ablation-Bezug.
- Risiken / Stolperstellen: Nicht zugeordnete Features werden `unknown`.
- Einfacher Verteidigungssatz: „Familiensicht ist bei uns konfigurationsgetrieben, nicht manuell.“

#### `_feat_type(s)` (interne Hilfsfunktion)

- Zweck: Typheuristik (`categorical`, `binary`, `numeric`).
- Input: Series.
- Output: Typstring.
- Was passiert intern: Dtype/Nunique-Heuristik.
- Warum ist das methodisch wichtig: Typabhängige Metriken.
- Risiken / Stolperstellen: Heuristik kann Grenzfälle falsch einsortieren.
- Einfacher Verteidigungssatz: „Wir nutzen eine einfache, reproduzierbare Typheuristik für Filtermetriken.“

#### `_sub_idx(n, seed)` (interne Hilfsfunktion)

- Zweck: Subsample-Index für MI-Berechnung.
- Input: Zeilenzahl, Seed.
- Output: `None` oder Indexarray.
- Was passiert intern: Bei großen Daten Zufallsstichprobe bis `MI_SUBSAMPLE`.
- Warum ist das methodisch wichtig: Rechenzeit begrenzen.
- Risiken / Stolperstellen: Metrikrauschen durch Subsampling.
- Einfacher Verteidigungssatz: „MI wird bei sehr großen Matrizen kontrolliert subsampled, um Laufzeit beherrschbar zu halten.“

#### `_safe_spearman(x, y)` (interne Hilfsfunktion)

- Zweck: Robuste Spearman-Korrelation.
- Input: Zwei Series.
- Output: Korrelation oder NaN.
- Was passiert intern: Nur bei mindestens 20 validen Paaren.
- Warum ist das methodisch wichtig: Vermeidet Scheinwerte aus Mini-Stichproben.
- Risiken / Stolperstellen: Viele NaNs bei dünnen Features.
- Einfacher Verteidigungssatz: „Wir berechnen Spearman nur bei ausreichender Datenbasis.“

#### `_safe_chi2(x, y, n_bins)` (interne Hilfsfunktion)

- Zweck: Robuster Chi2-Score.
- Input: Feature, Ziel, optional Bins.
- Output: Score oder NaN.
- Was passiert intern: Numeric ggf. via `qcut` binned, dann Kontingenz/Chi2.
- Warum ist das methodisch wichtig: Einfache Diskriminationssicht für CLS.
- Risiken / Stolperstellen: Binning-Entscheid kann Score beeinflussen.
- Einfacher Verteidigungssatz: „Chi2 wird robust über gebinnte Kontingenzen geschätzt.“

#### `compute_filter_report(X, y, stage)`

- Zweck: Pro Feature Missingness-, Varianz- und Signalmetriken.
- Input: Matrix `X`, Ziel `y`, Stage.
- Output: Report-DataFrame.
- Was passiert intern:
  - Kategorische Kodierung für MI.
  - MI (ggf. subsampled).
  - CLS: Chi2; REG: Spearman.
  - Flags für `is_constant`, `near_zero_variance`, `HIGH_MISSING`.
- Warum ist das methodisch wichtig:
  - Schneller, systematischer Vorfilter vor Embedded/Wrapper.
  - Vergleichbar über Base/Expanded/Conditional.
- Risiken / Stolperstellen:
  - Univariate Sicht, keine Interaktionseffekte.
  - Subsampling kann Rangfolgen verändern.
- Einfacher Verteidigungssatz: „Filterreport ist unser strukturierter Vorcheck, nicht die finale Featureentscheidung.“

#### `find_redundant_pairs(X)`

- Zweck: Findet hochkorrelierte numerische Featurepaare.
- Input: Matrix `X`.
- Output: Paarliste mit Pearson `r`.
- Was passiert intern: Korrelation numerischer Spalten, Schwellwert `CORR_THRESHOLD`.
- Warum ist das methodisch wichtig: Redundanzsicht vor Pruning.
- Risiken / Stolperstellen: Erfasst nur numerisch-paarweise Redundanz.
- Einfacher Verteidigungssatz: „Wir markieren starke lineare Redundanz explizit, bevor wir reduzieren.“

#### `build_family_report(cls_all, reg_all, cls_rd, reg_rd)`

- Zweck: Verdichtet Ergebnisse auf Featurefamilien.
- Input: CLS/REG Reports und Redundanzlisten.
- Output: Familienreport.
- Was passiert intern: Aggregiert MI, Missingness, Redundanz je Familie.
- Warum ist das methodisch wichtig: Diskussion auf Ebene sinnvoller Featuregruppen.
- Risiken / Stolperstellen: Aggregation kann Einzelspitzen glätten.
- Einfacher Verteidigungssatz: „Familienreport hilft, Entscheidungen fachlich statt rein spaltenweise zu begründen.“

#### `_build_safe_vs_cond(cls_all, reg_all)` (interne Hilfsfunktion)

- Zweck: Vergleicht SAFE gegen CONDITIONAL nach Signal.
- Input: CLS/REG Gesamtreports.
- Output: Textblock.
- Was passiert intern: Mittel/Max-MI je Bereich, starke/schwache Conditional-Features.
- Warum ist das methodisch wichtig: Bewertet den Mehrwert von Conditional Features explizit.
- Risiken / Stolperstellen: Nur univariate Signalperspektive.
- Einfacher Verteidigungssatz: „Wir vergleichen SAFE und CONDITIONAL transparent auf derselben Metrikbasis.“

#### `_build_summary(reports, redundant, family_df, safe_vs_cond)` (interne Hilfsfunktion)

- Zweck: Baut lesbaren Gesamtbericht.
- Input: Alle Reportbausteine.
- Output: Text.
- Was passiert intern: Stage-/Tier-Strukturen, Top/Bottom-MI, Empfehlungen.
- Warum ist das methodisch wichtig: Review- und Kommunikationsartefakt.
- Risiken / Stolperstellen: Empfehlungen sind filterbasiert, nicht final.
- Einfacher Verteidigungssatz: „Der Textbericht fasst alle Filterindikatoren in einem argumentierbaren Dokument zusammen.“

#### `_load(name)` (interne Hilfsfunktion)

- Zweck: Lädt Matrix-Parquet.
- Input: Artefaktname.
- Output: DataFrame.
- Was passiert intern: Pfad auflösen, Existenzprüfung, Lesen.
- Warum ist das methodisch wichtig: Standardisierte Artefaktquelle.
- Risiken / Stolperstellen: Fehlende vorgelagerte Pipelineartefakte.
- Einfacher Verteidigungssatz: „Feature Selection arbeitet auf festen Exportartefakten, nicht auf ad-hoc Datenständen.“

#### `run_feature_selection()`

- Zweck: Führt komplette Filterrunde aus.
- Input: Keine direkten Parameter.
- Output: Reports + Summary-Dateien.
- Was passiert intern: Lädt Matrizen, berechnet Filterreports/Redundanz, baut Familien-/Summaryreports.
- Warum ist das methodisch wichtig: Erstes strukturiertes Screening vor Runde 4.
- Risiken / Stolperstellen: Kein Final-Selector.
- Einfacher Verteidigungssatz: „Runde 3 liefert Diagnostik, Runde 4 liefert vertiefte Selektionssicht.“

#### `main()`

- Zweck: CLI-Start der Filterselektion.
- Input: CLI.
- Output: Startet `run_feature_selection()`.
- Was passiert intern: Argumentparser, dann Run.
- Warum ist das methodisch wichtig: Reproduzierbar ausführbarer Analysepfad.
- Risiken / Stolperstellen: Erwartet vorhandene Matrixartefakte.
- Einfacher Verteidigungssatz: „Filteranalyse ist als reproduzierbarer CLI-Run gekapselt.“

## `Experiment/feature_selection_r4.py`

### Rolle im Gesamtprozess

Runde 4 baut auf den exportierten Matrizen und ergänzt den Filterblick um Embedded Selection und Family Ablation. Diese Datei ist wichtig für die Verteidigung, weil sie zeigt, wie aus Diagnostik konkrete Kandidatensets und Familienbeiträge abgeleitet werden.

### Wichtige Inputs

- Exportierte Train/Test-Matrizen
- Konfigurationslisten und Drop-Regeln

### Wichtige Outputs

- Kandidatensets (JSON/CSV)
- Embedded-Results (CLS/REG)
- Ablationsberichte (CLS/REG)

### Funktionen

#### `_all_safe()` (interne Hilfsfunktion)

- Zweck: Liefert Safe-Superset.
- Input: Keine.
- Output: Liste.
- Was passiert intern: `cfg.CLS_EXPANDED_SAFE` kopieren.
- Warum ist das methodisch wichtig: Definiert Pruning-Ausgangspunkt.
- Risiken / Stolperstellen: Änderungen an Config wirken direkt in Runde 4.
- Einfacher Verteidigungssatz: „Pruning startet konsistent vom selben Safe-Superset.“

#### `_build_pruned_sets()` (interne Hilfsfunktion)

- Zweck: Baut vier pruned Sets (CLS/REG, SAFE/FULL).
- Input: Drop-Dicts + Config-Listen.
- Output: Set-Dict.
- Was passiert intern: Kombiniert Redundanz-, NZV- und Weak-Signal-Drops.
- Warum ist das methodisch wichtig: Nachvollziehbarer Übergang von Diagnose zu Kandidaten.
- Risiken / Stolperstellen: Heuristische Drop-Regeln.
- Einfacher Verteidigungssatz: „Pruned Sets entstehen regelbasiert und begründet, nicht per Bauchgefühl.“

#### `build_candidate_reports()`

- Zweck: Exportiert Kandidatensets und Gründe.
- Input: Keine direkten Parameter.
- Output: `(sets, df)` + JSON/CSV.
- Was passiert intern: Bewertet jede Feature-Stage-Kombination auf keep/drop rationale.
- Warum ist das methodisch wichtig: Auditierbare Entscheidungsdokumentation.
- Risiken / Stolperstellen: Keep/Drop-Logik hängt an gepflegten Regeln.
- Einfacher Verteidigungssatz: „Jedes Kandidatenfeature hat eine dokumentierte Keep/Drop-Begründung.“

#### `_load(name)` (interne Hilfsfunktion)

- Zweck: Lädt Round-4-Parquet.
- Input: Name.
- Output: DataFrame.
- Was passiert intern: Existenzprüfung + Read.
- Warum ist das methodisch wichtig: Stabile Datenbasis.
- Risiken / Stolperstellen: Fehlende Upstream-Artefakte.
- Einfacher Verteidigungssatz: „Runde 4 arbeitet auf fest gespeicherten Matrizennamen.“

#### `_encode_cats(df)` (interne Hilfsfunktion)

- Zweck: Label-encodiert kategoriale Spalten.
- Input: DataFrame.
- Output: Numerisches DataFrame.
- Was passiert intern: Spaltenweise LabelEncoder auf nicht-missing Kategorien.
- Warum ist das methodisch wichtig: Kompatibilität für lineare/GB-Modelle in Runde 4.
- Risiken / Stolperstellen: Ordinale Interpretation bei Label-Encoding.
- Einfacher Verteidigungssatz: „Für Embedded-Modelle kodieren wir kategorische Features konsistent numerisch.“

#### `_prep_Xy(features, x_train_name, y_train_name, x_eval_name, y_eval_name, target_col)` (interne Hilfsfunktion)

- Zweck: Lädt und bereitet Train/Eval-Daten für ein Set vor.
- Input: Featureliste + Artefaktnamen.
- Output: `Xt, yt, Xv, yv`.
- Was passiert intern:
  - Lädt train/eval Matrices und Targets.
  - Fehlende Features werden ggf. aus Conditional-Matrix ergänzt.
  - Encode + `fillna(-999)`.
- Warum ist das methodisch wichtig: Einheitliche Datenvorbereitung für alle Embedded-Läufe.
- Risiken / Stolperstellen:
  - Wenn Features weder in expanded noch conditional liegen, werden sie still weggelassen.
  - Unklar im Code: Fehlende Features werden nicht als eigener Fehler erzwungen.
- Einfacher Verteidigungssatz: „Wir harmonisieren die Inputs für alle Embedded-Experimente über einen gemeinsamen Prep-Schritt.“

#### `_hgb_feature_importances(model, n_features)` (interne Hilfsfunktion)

- Zweck: Berechnet gain-basierte Importances für HistGB.
- Input: Modell, Anzahl Features.
- Output: Importance-Array.
- Was passiert intern: Summiert Node-Gains über `_predictors`, normiert auf Summe.
- Warum ist das methodisch wichtig: Vergleichbarer Importance-Score für Baummodell.
- Risiken / Stolperstellen: Nutzung interner Modellstruktur.
- Einfacher Verteidigungssatz: „Für HistGB nutzen wir gain-basierte, normalisierte Featurebeiträge.“

#### `_cls_metrics(y_true, y_pred, y_prob)` (interne Hilfsfunktion)

- Zweck: CLS-Metriken bündeln.
- Input: Ground truth, Prädiktion, Wahrscheinlichkeiten.
- Output: Dict mit `f1`, `pr_auc`, `roc_auc`.
- Was passiert intern: Direkte Sklearn-Metriken.
- Warum ist das methodisch wichtig: Einheitlicher Metrikvergleich über Modelle.
- Risiken / Stolperstellen: Schwellenwertabhängigkeit bei F1.
- Einfacher Verteidigungssatz: „Wir vergleichen CLS-Modelle mit denselben Kernmetriken.“

#### `_reg_metrics(y_true, y_pred)` (interne Hilfsfunktion)

- Zweck: REG-Metriken bündeln.
- Input: Ground truth, Prädiktion.
- Output: `mae`, `median_ae`, `rmse`.
- Was passiert intern: Direkte Sklearn-Metriken.
- Warum ist das methodisch wichtig: Einheitliche Fehlerperspektive.
- Risiken / Stolperstellen: Keine Baseline im selben Funktionsaufruf.
- Einfacher Verteidigungssatz: „Wir bewerten REG über robuste und quadratische Fehlermaße parallel.“

#### `_subsample(X, y, n, seed)` (interne Hilfsfunktion)

- Zweck: Reduziert große Trainingsmengen für lineare Modelle.
- Input: `X`, `y`, Größe, Seed.
- Output: Teilmenge oder Original.
- Was passiert intern: Zufällige Auswahl ohne Replacement.
- Warum ist das methodisch wichtig: Laufzeiten beherrschbar.
- Risiken / Stolperstellen:
  - Unklar im Code: Docstring sagt „stratified“, Implementierung ist reines Zufallssampling ohne Stratifikation.
- Einfacher Verteidigungssatz: „Lineare Embedded-Läufe werden aus Laufzeitgründen subsampled; wir kennzeichnen das offen als Approximation.“

#### `run_embedded_cls(features, variant, Xt, yt, Xv, yv)`

- Zweck: Embedded-Selection für CLS mit drei Modellfamilien.
- Input: Features, Variantename, Train/Eval-Daten.
- Output: Liste mit featureweisen Modellresultaten.
- Was passiert intern:
  - L1 Logistic Regression (subsampled train).
  - ElasticNet Logistic Regression (subsampled train).
  - HistGradientBoostingClassifier (voller train).
  - Pro Feature: Importance + selected-Flag + globale Metriken.
- Warum ist das methodisch wichtig:
  - Kombiniert lineare und nichtlineare Wichtigkeitssichten.
  - Ergänzt Filtermethoden um modellbasierte Relevanz.
- Risiken / Stolperstellen:
  - Auswahlschwellen (`>1e-8`, `>0.001`) sind heuristisch.
  - Keine Cross-Validation in dieser Stufe.
- Einfacher Verteidigungssatz: „Wir triangulieren Feature-Relevanz über mehrere Modelltypen statt nur eine Metrik zu verwenden.“
- Mini-Beispiel (vereinfachter Ablauf):
  1.  Trainiere L1-LogReg auf subsample.
  2.  Wenn `abs(coef(feature_X)) > 1e-8`, markiere `selected=True`.
  3.  Vergleiche mit HistGB-Importance desselben Features.

#### `run_embedded_reg(features, variant, Xt, yt, Xv, yv)`

- Zweck: Embedded-Selection für REG mit drei Modellfamilien.
- Input: Features, Variantename, Train/Eval-Daten.
- Output: Liste mit featureweisen Modellresultaten.
- Was passiert intern:
  - Lasso (subsampled train).
  - ElasticNet (subsampled train).
  - HistGradientBoostingRegressor (voller train).
- Warum ist das methodisch wichtig:
  - Modellübergreifende Relevanzsicht auf Mengenprognose.
- Risiken / Stolperstellen:
  - Subsampling/Thresholds beeinflussen Selektionsstabilität.
- Einfacher Verteidigungssatz: „REG-Featurestärke wird bei uns nicht nur korrelativ, sondern modellbasiert geprüft.“
- Mini-Beispiel:
  - Feature A: Lasso-Koeffizient 0 -> nicht selektiert.
  - Feature A: HistGB-Importance 0.004 -> selektiert.
  - Interpretation: Nichtlinear relevant, linear schwach.

#### `run_embedded_selection(sets)`

- Zweck: Führt Embedded-Läufe für alle pruned Sets aus.
- Input: Set-Dict.
- Output: `(cls_df, reg_df)` + CSV-Exports.
- Was passiert intern: Lädt je Variante Daten via `_prep_Xy`, ruft `run_embedded_*` auf.
- Warum ist das methodisch wichtig: Einheitlicher Batch über SAFE/FULL und CLS/REG.
- Risiken / Stolperstellen: Abhängigkeit von korrekten Artefaktnamen.
- Einfacher Verteidigungssatz: „Alle Kandidatensets laufen durch denselben Embedded-Benchmark-Prozess.“

#### `_embedded_summary(cls_df, reg_df, sets)` (interne Hilfsfunktion)

- Zweck: Textsummary Embedded-Runde.
- Input: Ergebnisframes + Sets.
- Output: Text.
- Was passiert intern: Modellmetriken, Top-Importances, Selected-Counts.
- Warum ist das methodisch wichtig: Vergleichbarer Reviewbericht.
- Risiken / Stolperstellen: Bericht hängt an ausgewählten Metrikschwellen.
- Einfacher Verteidigungssatz: „Die Embedded-Summary macht Modell- und Featureeffekte in einem Bericht vergleichbar.“

#### `_train_eval_cls(feats, Xt_full, yt, Xv_full, yv)` (interne Hilfsfunktion)

- Zweck: Schneller HistGBC-Benchmark für Ablation.
- Input: Featureliste + Daten.
- Output: CLS-Metrikdict.
- Was passiert intern: Trainiert HistGBC auf selektierten Features.
- Warum ist das methodisch wichtig: Standardisierte Ablationseinheit.
- Risiken / Stolperstellen: Ein Modelltyp für Ablation.
- Einfacher Verteidigungssatz: „Ablation nutzt ein konstantes Basismodell, damit Deltas vergleichbar bleiben.“

#### `_train_eval_reg(feats, Xt_full, yt, Xv_full, yv)` (interne Hilfsfunktion)

- Zweck: Schneller HistGBR-Benchmark für Ablation.
- Input: Featureliste + Daten.
- Output: REG-Metrikdict.
- Was passiert intern: Trainiert HistGBR auf selektierten Features.
- Warum ist das methodisch wichtig: Einheitliche REG-Ablationslogik.
- Risiken / Stolperstellen: Keine Modellvielfalt in dieser Teilanalyse.
- Einfacher Verteidigungssatz: „Familieneffekte werden mit einem festen REG-Basismodell isoliert gemessen.“

#### `run_family_ablation(sets)`

- Zweck: Leave-one-out und Add-one auf Featurefamilien.
- Input: Kandidatensets.
- Output: `(cls_abl, reg_abl)` + CSV.
- Was passiert intern:
  - Baseline mit Full-Pruned.
  - Leave-one-family-out: je Familie entfernen und Deltas messen.
  - Add-one: von Zeit-Kern aus Familien ergänzen.
- Warum ist das methodisch wichtig:
  - Liefert familienbezogene Verteidigungsargumente statt nur Einzelspalten.
- Risiken / Stolperstellen:
  - Ergebnisabhängig vom gewählten Kern (`time`).
  - Keine CV; testabhängige Varianz bleibt.
- Einfacher Verteidigungssatz: „Family Ablation zeigt, welche Featuregruppen wirklich Leistung tragen, nicht nur einzelne Spalten.“
- Mini-Beispiel:
  - Baseline PR-AUC = 0.30.
  - Ohne `conditional_oof_segment`: PR-AUC = 0.27 (`-0.03`).
  - Aussage: Diese Familie trägt substanziell zur CLS-Leistung bei.

#### `_ablation_summary(cls_abl, reg_abl)` (interne Hilfsfunktion)

- Zweck: Verdichteter Textbericht der Ablation.
- Input: Ablations-DataFrames.
- Output: Text.
- Was passiert intern: Baseline, Leave-out-, Add-one-Abschnitte je Stage.
- Warum ist das methodisch wichtig: Präsentationsfähige Familienwirkung.
- Risiken / Stolperstellen: Delta-Grenzen sind heuristisch bewertet.
- Einfacher Verteidigungssatz: „Der Ablationsbericht übersetzt Experimente in klare Familienaussagen.“

#### `run_round4()`

- Zweck: Orchestriert Kandidatensets, Embedded Selection und Ablation.
- Input: Keine direkten Parameter.
- Output: Vollständiges Runde-4-Artefaktpaket.
- Was passiert intern: Step 1 Kandidaten, Step 2 Embedded, Step 3 Ablation.
- Warum ist das methodisch wichtig: Schließt den Bogen von Filterdiagnostik zu modellgestützter Priorisierung.
- Risiken / Stolperstellen: Kein automatischer Rückfluss in produktive Final-Sets.
- Einfacher Verteidigungssatz: „Runde 4 liefert fundierte Entscheidungsgrundlagen, ersetzt aber nicht automatisch die fachliche Endentscheidung.“

---

## Kritische methodische Punkte

1. Leakage kann entstehen, wenn targetnahe Größen (`revenue`, `quantity`, `order`) unkontrolliert in Features gelangen.
2. Besonders sensibel sind Conditional-Features mit Targetbezug (`pid_prob`, `day_7_likelihood`, `day_7_qty_mean_oof`).
3. Full-Train-Aggregationen (`fit_global_aggregations`) sind für Val/Test-Anwendung vertretbar, aber für Train-Bewertung ohne OOF kritisch.
4. Forward OOF reduziert Future/Self Leakage, ist aber bei dünner Historie fallback-lastig.
5. Sampling (`run_sampling`) ist nur für Prototyping; finale Bewertung muss auf vollständigen Splits passieren.
6. REG muss separat betrachtet werden, weil Population und Zieldefinition anders sind (Stage-2-Subset).
7. Eine Baseline wie „immer `quantity = 1`“ ist für REG wichtig, um den echten Mehrwert komplexer Modelle zu belegen.
8. Checks zur Risikoreduktion im Code:
   - `assert_no_forbidden_features(...)`
   - `assert_reg_stage2_only(...)`
   - `assert_cross_split_columns(...)`
   - `assert_preprocessing_integrity(...)`
   - `get_reg_mask(df)` als zentrale Stage-2-Definition
   - OOF mit `_time_aware_forward_oof(...)`

## Glossar

- **CLS**: Klassifikationsstufe, Ziel `order`.
- **REG**: Regressionsstufe, Ziel `quantity` nur auf Stage-2-Zeilen.
- **Leakage**: Unerlaubter Informationsfluss, der Modellbewertung künstlich verbessert.
- **Target Leakage**: Feature enthält direkt/indirekt Zielinformation.
- **Future Leakage**: Zukunftsinformation fließt in Vergangenheitstraining ein.
- **Self Leakage**: Zeile nutzt ihre eigene Zielinformation bei Featurebildung.
- **Safe Feature**: Feature ohne targethistorisches Lernen, aus aktueller Zeile oder train-only fit mit sauberem Apply.
- **Conditional Feature**: Feature mit Historien-, Aggregations- oder targetnaher Information.
- **OOF**: Out-of-Fold Encoding; Zeile wird mit außerhalb der eigenen „Train-Untermenge“ gelernten Werten kodiert.
- **Forward OOF**: Zeitbewusste OOF-Variante mit expandierendem Fenster nur aus früheren Blöcken.
- **Cold Start**: Fallback-Wert, wenn für einen Block keine zulässige Historie vorliegt.
- **Train/Test/Validation Split**: Chronologische Aufteilung in Lern-, Test- und Validierungsfenster.
- **Stage 2**: REG-Teilmenge mit `order == 1` und validem `quantity`.
- **Feature Matrix**: Modellinput `X` plus Zielvektor `y`.
- **Orange Export**: CSV-Handover mit finalen Feature-Sets und konsistentem Schema für Orange.
- **Ablation**: Systematisches Entfernen/Hinzufügen von Featuregruppen zur Wirkungsmessung.
- **Embedded Selection**: Featurebewertung aus trainierten Modellen (z. B. Koeffizienten/Importances).
- **Filter Method**: Univariate Vorbewertung ohne Modellinteraktion (z. B. MI, Spearman, Chi2).

## Prüfpunkte für die fachliche Verteidigung

Relevante Prüfpunktfragen:

- Warum Two-Stage?
- Warum Time Split?
- Warum Safe vs. Conditional Features?
- Warum Forward OOF?
- Warum `revenue` verboten?
- Warum Sampling nur Prototyping?
- Warum REG nur Stage 2?
- Welche Features sind kritisch?
- Was passiert beim Orange Export?
- Was macht Feature Selection downstream?

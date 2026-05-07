# Setup und Ausfuehrungsanleitung

Dieses Dokument beschreibt die Voraussetzungen und Schritte, um die Skripte und Notebooks in diesem Projekt lokal auszufuehren.

Es gibt zwei typische Wege:

1. Nur die Modellierungs-Notebooks mit bereits vorbereiteten CSV-Dateien aus OneDrive ausfuehren.
2. Die Datensaetze aus den Rohdaten neu erzeugen und danach die Modellierungs-Notebooks starten.

---

## 1. Voraussetzungen

### Python

- Python 3.13 (getestet mit 3.13.6)
- Download: https://www.python.org/downloads/

### VS Code und Erweiterungen

- Visual Studio Code: https://code.visualstudio.com/
- Erweiterung `ms-python.python`
- Erweiterung `ms-toolsai.jupyter`

### Hardware

- Empfohlen: mindestens 16 GB RAM
- Fuer `full_scale_cls_modeling.ipynb` und `full_scale_reg_modeling_minimal.ipynb` je nach Hardware etwa 20 bis 60 Minuten Laufzeit

---

## 2. Projekt lokal einrichten

### Repository klonen

```bash
git clone https://github.com/Karim-Abu/Business_Analytics_Project.git
cd Business_Analytics_Project
```

Falls das Projekt nicht per `git clone`, sondern per ZIP oder OneDrive uebergeben wird, ist der Ordnername egal. Wichtig ist nur, dass die Ordnerstruktur unveraendert bleibt.

### Virtuelle Umgebung erstellen

```bash
python -m venv .venv
```

### Abhaengigkeiten installieren

Windows:

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### VS Code richtig oeffnen

Das Projekt muss als gesamter Workspace-Root geoeffnet werden, also der komplette Projektordner. Nicht nur `Modellierung/` oder `Feature Engineering/` einzeln oeffnen.

Grund: Die Notebooks verwenden `Path().resolve()` und erwarten den Projekt-Root als Arbeitsverzeichnis.

### Notebook-Kernel waehlen

Im Notebook oben rechts `Select Kernel` waehlen und die virtuelle Umgebung `.venv` auswaehlen.

---

## 3. Welche Daten werden benoetigt?

Die grossen Datendateien sind nicht im Git-Repository enthalten. Sie muessen separat ueber OneDrive bereitgestellt werden.

### Variante A: Nur Modellierungs-Notebooks ausfuehren

Dann werden die vorbereiteten Modellierungsdateien aus OneDrive benoetigt.

### Variante B: Datensaetze aus Rohdaten neu erzeugen

Dann werden zunaechst nur die Rohdaten benoetigt:

```text
Data/
|- train.csv
`- items.csv
```

`merged_clean.csv` ist fuer den aktuellen Build der Pipeline nicht erforderlich.

---

## 4. Datenablage fuer die Notebooks

### 4.1 CLS-Notebook-Daten

Diese Dateien muessen in `Modellierung/CLS/Daten/` liegen:

| Datei                          | Verwendet in                                                            |
| ------------------------------ | ----------------------------------------------------------------------- |
| `cls_train_full.csv`           | `full_scale_cls_modeling.ipynb`, `quick_challenger_lgbm_catboost.ipynb` |
| `cls_test.csv`                 | `full_scale_cls_modeling.ipynb`, `quick_challenger_lgbm_catboost.ipynb` |
| `cls_val.csv`                  | `full_scale_cls_modeling.ipynb`, `quick_challenger_lgbm_catboost.ipynb` |
| `cls_train_full_no_groups.csv` | `cls_no_groups_sensitivity.ipynb`                                       |
| `cls_test_no_groups.csv`       | `cls_no_groups_sensitivity.ipynb`                                       |

```text
Modellierung/
`- CLS/
   `- Daten/
      |- cls_train_full.csv
      |- cls_test.csv
      |- cls_val.csv
      |- cls_train_full_no_groups.csv
      `- cls_test_no_groups.csv
```

### 4.2 Threshold-Tuning-Datei

Diese Datei muss direkt in `Modellierung/` liegen:

| Datei                                     | Verwendet in                      |
| ----------------------------------------- | --------------------------------- |
| `cls_test_100k_predictions_gb_logreg.csv` | `threshold_tuning_cls_100k.ipynb` |

```text
Modellierung/
`- cls_test_100k_predictions_gb_logreg.csv
```

Hinweis: Diese Datei wird von der aktuellen Feature-Engineering-Pipeline nicht automatisch erzeugt und muss separat ueber OneDrive bereitgestellt werden.

### 4.3 REG-Notebook-Daten

Diese Dateien muessen in `Feature Engineering/outputs/orange_exports/` liegen:

| Datei                | Verwendet in                            |
| -------------------- | --------------------------------------- |
| `reg_train_full.csv` | `full_scale_reg_modeling_minimal.ipynb` |
| `reg_test.csv`       | `full_scale_reg_modeling_minimal.ipynb` |
| `reg_val.csv`        | `full_scale_reg_modeling_minimal.ipynb` |

```text
Feature Engineering/
`- outputs/
   `- orange_exports/
      |- reg_train_full.csv
      |- reg_test.csv
      `- reg_val.csv
```

---

## 5. Feature-Engineering-Pipeline neu aus Rohdaten bauen

Wenn die Datensaetze nicht aus OneDrive kopiert werden sollen, koennen sie aus `Data/train.csv` und `Data/items.csv` neu erzeugt werden.

### Pipeline starten

Aus dem Ordner `Feature Engineering/` ausfuehren:

```bash
cd "Feature Engineering"
python main_build_datasets.py --mode safe_only
python main_build_datasets.py --mode safe_plus_conditional
```

Die Pipeline erzeugt unter anderem:

- `Feature Engineering/outputs/datasets/`
- `Feature Engineering/outputs/audit/`
- `Feature Engineering/outputs/metadata/`
- `Feature Engineering/outputs/orange_exports/`

### Was danach fuer die Modellierungs-Notebooks zu tun ist

Nach `safe_plus_conditional` liegen die exportierten CSV-Dateien in `Feature Engineering/outputs/orange_exports/`.

Fuer das REG-Notebook reicht das direkt aus, weil `full_scale_reg_modeling_minimal.ipynb` diese Dateien dort erwartet.

Fuer die CLS-Notebooks muessen die erzeugten CSV-Dateien zusaetzlich nach `Modellierung/CLS/Daten/` kopiert werden:

- `cls_train_full.csv`
- `cls_test.csv`
- `cls_val.csv`
- `cls_train_full_no_groups.csv`
- `cls_test_no_groups.csv`

---

## 6. Empfohlene Ausfuehrungsreihenfolge

Die Notebooks sind weitgehend unabhaengig. Eine relevante Abhaengigkeit gibt es beim Challenger-Notebook:

1. `full_scale_cls_modeling.ipynb`
2. `quick_challenger_lgbm_catboost.ipynb`
3. `cls_no_groups_sensitivity.ipynb`
4. `threshold_tuning_cls_100k.ipynb`
5. `full_scale_reg_modeling_minimal.ipynb`

Warum zuerst `full_scale_cls_modeling.ipynb`?

`quick_challenger_lgbm_catboost.ipynb` liest die Datei `full_scale_cls_best_models.csv` als Referenz. Diese Datei wird vom Full-Scale-CLS-Notebook erzeugt.

---

## 7. Kurzcheck vor dem Start

Vor dem ersten Lauf sollte Folgendes stimmen:

1. Der komplette Projektordner ist in VS Code geoeffnet.
2. Die virtuelle Umgebung `.venv` ist als Notebook-Kernel ausgewaehlt.
3. Die benoetigten Daten liegen in den richtigen Ordnern.
4. Fuer `quick_challenger_lgbm_catboost.ipynb` existiert bereits `Modellierung/CLS/full_scale_cls_best_models.csv` oder das Full-Scale-Notebook wird zuerst ausgefuehrt.

---

## 8. Hauefige Fehlerbilder

### `FileNotFoundError` bei CSV-Dateien

Ursache: Eine Datei aus OneDrive liegt im falschen Ordner oder wurde noch nicht kopiert.

Pruefen:

- `Modellierung/CLS/Daten/` fuer CLS-Dateien
- `Modellierung/` fuer `cls_test_100k_predictions_gb_logreg.csv`
- `Feature Engineering/outputs/orange_exports/` fuer REG-Dateien

### Pfade zeigen auf den falschen Ordner

Ursache: Es wurde nur ein Unterordner in VS Code geoeffnet.

Loesung: Den gesamten Projektordner als Workspace oeffnen.

### `ModuleNotFoundError` in der Feature-Engineering-Pipeline

Ursache: Meist ein alter oder unvollstaendiger Projektstand.

Loesung: Aktuellen Stand aus GitHub verwenden und die Pipeline aus `Feature Engineering/` starten.

### Das Challenger-Notebook findet das Incumbent-Modell nicht

Ursache: `full_scale_cls_best_models.csv` wurde noch nicht erzeugt.

Loesung: Zuerst `full_scale_cls_modeling.ipynb` komplett ausfuehren.

---

## 9. Wichtige Pakete

Die vollstaendige Liste steht in `requirements.txt`. Die wichtigsten Pakete fuer die Modellierung sind:

| Paket          | Version | Zweck                      |
| -------------- | ------- | -------------------------- |
| `pandas`       | 3.0.1   | Datenverarbeitung          |
| `numpy`        | 2.4.2   | Numerik                    |
| `scikit-learn` | 1.8.0   | ML-Modelle                 |
| `lightgbm`     | 4.6.0   | Challenger-Modelle         |
| `catboost`     | 1.2.10  | Challenger-Modelle         |
| `matplotlib`   | 3.10.9  | Visualisierungen           |
| `plotly`       | 6.7.0   | Interaktive Plots          |
| `pyarrow`      | 23.0.1  | Parquet-Handling           |
| `ipykernel`    | 7.2.0   | Notebook-Kernel in VS Code |

# Dynamic Pricing Analytics Project

End-to-end data-preparation pipeline for a dynamic-pricing analytics study
(CRISP-DM data understanding -> feature engineering -> classification &
regression modelling).

The repository supports a sample pipeline run from a fresh clone on a small
sample dataset, without Orange, without Tableau, and without the original full
dataset.

## Quickstart

Requires Python 3.11+.

```powershell
# 1. Clone
git clone <repo-url>
cd "Analytics Project Code"

# 2. (Optional) virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# Windows/PyCharm: falls python nicht gefunden wird, py verwenden
py -m pip install -r requirements.txt

# 4. Run the sample pipeline (default: small synthetic data)
python scripts/run_pipeline.py

# Windows/PyCharm Alternative
py scripts/run_pipeline.py

# Optional: run sample pipeline plus quick model benchmark
python scripts/run_pipeline.py --benchmark

# 5. Smoke test
python scripts/smoke_test.py
```

The smoke test prints `SMOKE TEST PASSED` on success or `SMOKE TEST FAILED`
plus a reason on failure.

> **Wichtig:**
>
> - Orange ist **nicht** noetig, um den Code auszufuehren.
> - Tableau ist **nicht** noetig, um den Code auszufuehren.
> - Echte Full-Daten sind **optional** und nicht im Repo enthalten.
> - Der Sample-Run prueft die **technische Lauffaehigkeit**, nicht die
>   finale Modellqualitaet.

## Sample-Run

`python scripts/run_pipeline.py` (ohne Argumente) startet automatisch den
Sample-Lauf:

1. Liest die kleinen Sample-Rohdaten aus [data/sample/train_sample.csv](data/sample/train_sample.csv)
  und [data/sample/items_sample.csv](data/sample/items_sample.csv) (~300 Zeilen,
  8 PIDs, gepipte CSVs). Die Namen enthalten bewusst `_sample`, damit sie
  nicht mit den echten Full-Daten verwechselt werden.
2. Fuehrt das Preprocessing aus (Spalten bereinigen, `quantity` ableiten,
   suspekte Zeilen markieren).
3. Splittet chronologisch: Train (Tag 26-70), Test (71-81), Validation
   (82-92).
4. Berechnet Feature-Matrizen fuer Klassifikation (CLS) und Regression
   (REG) inklusive Sampling-Subsets.
5. Schreibt alle Artefakte nach [artifacts/sample_run/](artifacts/sample_run/)
   inklusive `RUN_MANIFEST.md` und `run_summary.json`.

Der Sample-Lauf dauert wenige Sekunden und erzeugt nur kleine Dateien
(Parquet-Matrizen mit wenigen hundert Zeilen, keine Modelle, kein Orange-Export).
Die Zahlen sind **nicht** als Modell- oder Geschaeftsergebnis interpretierbar.

## Full-Run mit echten Daten

Die echten Full-Daten (mehrere Millionen Zeilen) sind nicht Teil des
Repositories. Fuer Full-Runs muessen die echten Dateien manuell unter
`data/raw/` abgelegt werden. Erwartet werden exakt diese kleingeschriebenen
Dateinamen mit Trennzeichen `|`:

- `data/raw/train.csv`
- `data/raw/items.csv`

```powershell
python scripts/run_pipeline.py --full
python scripts/run_pipeline.py --full --mode safe_plus_conditional
python scripts/run_pipeline.py --full --benchmark
python scripts/run_pipeline.py --full --output-dir artifacts/my_run
python scripts/run_pipeline.py --full --no-orange-export
```

Windows/PyCharm-Alternative:

```powershell
py scripts/run_pipeline.py --full --benchmark
```

Fehlen die Rohdaten, bricht der Lauf **ohne Python-Traceback** ab und nennt
die erwarteten Pfade sowie den Sample-Befehl als Alternative.

CLI-Optionen:

| Flag                           | Effekt                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------ |
| `--sample` (default)           | Daten aus `data/sample/`, Output `artifacts/sample_run/`.                      |
| `--full`                       | Daten aus `data/raw/`, Output `artifacts/full_run/`.                           |
| `--mode safe_only` (default)   | Basisfeatures, keine Conditional/Orange-Exports.                               |
| `--mode safe_plus_conditional` | + Conditional Features + Orange CSVs.                                          |
| `--output-dir <pfad>`          | Eigenes Output-Verzeichnis.                                                    |
| `--no-orange-export`           | Orange-CSV-Export deaktivieren.                                                |
| `--benchmark`                  | Optionaler, diagnostischer Schnellbenchmark nach erfolgreichem Run.            |
| `--benchmark-max-train-rows N` | Maximale Trainingszeilen pro Task fuer den Benchmark; `0` = kein Limit.        |
| `--benchmark-max-eval-rows N`  | Maximale Validation/Test-Zeilen pro Task fuer den Benchmark; `0` = kein Limit. |

## Optionaler Modellbenchmark

Mit `--benchmark` startet nach der Datenaufbereitung automatisch ein schneller
sklearn-Benchmark auf den erzeugten Feature-Matrizen. Dieser Benchmark ist
optional und dient nur zur technischen Plausibilitaetspruefung des Pipeline-
Outputs. Er ersetzt nicht die finale Modellierung aus den separaten CLS-/REG-
Modellierungsauswertungen.

Verglichen werden alle verfuegbaren Matrix-Varianten (`conditional`,
`expanded`, `base`) und mehrere einfache Modelle. Die Auswahl erfolgt auf dem
Validation-Split, falls vorhanden, sonst auf dem Test-Split:

- Klassifikation (CLS): bestes Modell nach hoechstem F1.
- Regression (REG): bestes Modell nach niedrigstem MAE.

Beispiel:

```powershell
python scripts/run_pipeline.py --full --mode safe_plus_conditional --benchmark
```

Abweichende Quick-Benchmark-Winner sind wegen Train-/Eval-Caps, Feature-Set,
Threshold-Handling und Laufzeitbegrenzung moeglich. Die finale
Modellentscheidung bleibt in der Modellierungsdokumentation: CLS final
HistGradientBoosting mit Threshold 0.22; REG final Always-1 / DummyMedian, da
Median `quantity` = 1.

Die Ergebnisse werden nach `<output-dir>/benchmark/` geschrieben:

```
benchmark/
├── model_benchmark_results.csv       # alle Benchmark-Metriken
├── quick_benchmark_winners.csv       # diagnostische Gewinner pro Task
├── model_benchmark_summary.json      # maschinenlesbare Zusammenfassung
└── model_benchmark_summary.txt       # lesbare Kurzfassung
```

## Outputs

Nach `python scripts/run_pipeline.py` (Sample-Lauf, `safe_only`) liegt im
Output-Ordner:

```
artifacts/sample_run/
├── RUN_MANIFEST.md           # Menschenlesbare Zusammenfassung des Laufs
├── run_summary.json          # Maschinenlesbare Zusammenfassung
├── datasets/                 # Feature-Matrizen + Sampling-Subsets (Parquet)
│   ├── X_train_cls_base.parquet
│   ├── X_train_cls_expanded.parquet
│   ├── X_train_reg_base.parquet
│   ├── ...
│   ├── train_cls_sample.parquet
│   └── train_reg_sample.parquet
├── audit/                    # Audit-Reports (CSV/TXT)
│   ├── target_distribution.csv
│   ├── join_quality.csv
│   ├── missingness_merged.csv
│   ├── missingness_train.csv
│   ├── outliers_train.csv
│   ├── sampling_audit_cls.csv
│   ├── sampling_audit_reg.csv
│   ├── feature_sets.csv
│   ├── dropped_features.csv
│   └── feature_matrix_summary.txt
├── metadata/                 # Reproduzierbarkeits-Artefakte
│   ├── pid_segment_map.csv
│   └── binning_edges.json
├── orange_exports/           # nur in `safe_plus_conditional` befuellt
└── benchmark/                # nur mit `--benchmark` befuellt
```

Der wichtigste Einstieg fuer den Reviewer ist die `RUN_MANIFEST.md`: sie
listet Run-Typ, Build-Mode, Eingaben, Zeilenzahlen und alle erzeugten Dateien
mit Groessen.

## Repository-Struktur (Kurzueberblick)

```
scripts/
  run_pipeline.py        # Einstieg fuer Sample- und Full-Run
  smoke_test.py          # Selbsttest fuer einen frischen Clone
data/
  sample/                # kleine, getrackte Sample-Rohdaten (*_sample.csv)
  raw/                   # Platzhalter fuer echte Full-Daten (ignored)
Feature Engineering/     # eigentliche Pipeline (Module)
  config.py
  io_utils.py
  main_build_datasets.py
  feature_sets.py
  orange_export.py
  Preprocessing/
  Sampling/
Modellierung/            # Notebooks zu CLS/REG-Modellen + Analysen
Tableau/                 # optional, nicht fuer den Lauf noetig
```

## Troubleshooting

- `ModuleNotFoundError: pandas` -> `pip install -r requirements.txt` oder unter
  Windows/PyCharm `py -m pip install -r requirements.txt`.
- `Raw data not found.` bei `--full` -> echte Daten nach `data/raw/` legen
  und exakt als `train.csv` sowie `items.csv` benennen; oder ohne `--full`
  aufrufen (Sample-Lauf).
- `UnicodeEncodeError` auf alten Windows-Konsolen -> `set PYTHONUTF8=1`
  setzen oder die mitgelieferten Wrapper (`scripts/run_pipeline.py`,
  `scripts/smoke_test.py`) verwenden, die UTF-8 selbst aktivieren.
- Vorherige Outputs stoeren -> Output-Ordner loeschen oder `--output-dir`
  mit neuem Pfad verwenden.

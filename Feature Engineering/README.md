# Feature Engineering

Diese Pipeline wird aus dem Projektroot `Feature Engineering/` gestartet.

## Struktur

Projektroot:

- `config.py`
- `io_utils.py`
- `feature_sets.py`
- `orange_export.py`
- `main_build_datasets.py`
- `README.md`

Pakete:

- `Preprocessing/` mit `preprocessing.py`, `validation.py`, `feature_engineering_safe.py`, `feature_engineering_conditional.py`, `audit.py`
- `Sampling/` mit `split.py`, `pid_segment.py`, `sampling.py`

Weitere Ordner:

- `outputs/` für erzeugte Datasets, Audits, Metadaten und Orange-Exporte
- `Experiment/` für nachgelagerte Analyse- und Feature-Selection-Skripte
- `Doku/` für Projektdokumentation

## Start

Im Ordner `Feature Engineering/` ausführen:

```bash
python main_build_datasets.py --mode safe_only
python main_build_datasets.py --mode safe_plus_conditional
```

Die Root-Ausführung benötigt kein manuelles `PYTHONPATH` und keine `sys.path`-Anpassungen.

## Windows-Hinweis

Falls die Konsole UTF-8-Ausgabe nicht sauber darstellt, vor dem Lauf `PYTHONIOENCODING=utf-8` setzen.

PowerShell:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
```

cmd.exe:

```cmd
set PYTHONIOENCODING=utf-8
```

## Wichtige Outputs

- `outputs/datasets/` enthält Parquet-Matrizen und Samples
- `outputs/audit/` enthält Audit-Reports
- `outputs/metadata/` enthält Reproduzierbarkeitsartefakte
- `outputs/orange_exports/` enthält die Orange-CSV-Dateien inklusive `export_manifest.csv`

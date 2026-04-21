# Feature Engineering – Arbeitsübersicht

Diese Ordnerstruktur ist die aktuelle Arbeitsbasis für Data Preparation und Orange-Handoff.

## Was hier aktiv genutzt wird

Die produktive Pipeline besteht aktuell aus diesen Bereichen:

- `Preprocessing/` für Preprocessing, Validierung und den Hauptlauf
- `Sampling/` für Split, Sampling und aktive Feature-Set-Helfer
- `config.py` für zentrale Regeln, Pfade und finale Sets
- `feature_engineering_conditional.py` für Conditional Features
- `io_utils.py`, `audit.py`, `pid_segment.py`, `orange_export.py` für Hilfslogik und Export
- `outputs/` für alle erzeugten Artefakte
- `Doku/` für Status, Zusammenfassung und Orange-Checkliste

## Was nicht zum Kernpfad gehört

Diese Bereiche sind aktuell nicht Teil der laufenden Modellierungsfreigabe:

- `Experiment/` enthält Analyse- und Feature-Selection-Skripte
- `__pycache__/` ist reiner Python-Cache
- alte Lauf-Logs gehören nicht in den Hauptpfad

## Wichtige Unterordner in `outputs/`

- `outputs/orange_exports/` ist der wichtigste Zielordner für die Modellierung in Orange
- `outputs/audit/` enthält Kontroll- und Prüfdateien
- `outputs/metadata/` enthält Reproduzierbarkeitsartefakte
- `outputs/feature_selection/` enthält Analyseergebnisse, nicht die operative Pipeline
- `outputs/datasets/` enthält Python-seitige Matrizen und Samples

## Aktuell sinnvolle Startpunkte

Wenn Sie den Projektstand verstehen wollen:

1. `Doku/PROJEKT_ZUSAMMENFASSUNG.md`
2. `Doku/ORANGE_MODELING_CHECKLIST.md`
3. `outputs/orange_exports/export_manifest.csv`

Wenn Sie die Pipeline ausführen wollen:

1. In den Ordner `Feature Engineering/` wechseln
2. `python Preprocessing/main_build_datasets.py --mode safe_plus_conditional`

## Bewusst nicht anfassen

Für die laufende Orange-Modellierung sollten diese Bereiche nicht vereinfacht oder umgebaut werden:

- Preprocessing-Kern
- Conditional Features
- Stage-2-Maske
- Sampling-Logik
- Orange-Export-Logik
- finale Feature-Sets

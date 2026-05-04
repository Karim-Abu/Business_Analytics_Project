# Orange Modeling Checklist

Manuelle Prüfliste beim erstmaligen Import der Orange-CSVs.
Alle Dateien liegen unter `outputs/orange_exports/`.

Die eigentliche Schritt-für-Schritt-Anleitung für den Modellierungsablauf liegt in
`Doku/ORANGE_MODELING_SCHRITT_FUER_SCHRITT.md`.

---

## 1. Zielspalte (Target)

- [ ] **CLS-Dateien:** `order` wird als Zielvariable erkannt (letzte Spalte, binär 0/1)
- [ ] **REG-Dateien:** `quantity` wird als Zielvariable erkannt (letzte Spalte, numerisch ≥ 1)

## 2. Kategoriale Felder — Diskret-Erkennung

Folgende Spalten **müssen** in Orange als **diskret** (categorical) erkannt werden, nicht als numerisch:

| Spalte               | Erwartete Werte           | In Sets         |
| -------------------- | ------------------------- | --------------- |
| `category_norm`      | `C_1.0`, `C_2.0`, …       | CLS, REG        |
| `pharmForm_norm`     | `TAB`, `GEL`, `LOT`, …    | CLS, REG        |
| `campaignIndex_norm` | `A`, `B`, `C`, `NONE`     | REG             |
| `pid_segment`        | `Head`, `Mid`, `Tail`     | CLS, REG        |
| `group12`            | `21`, `12`, `10`, `2F`, … | CLS (nicht REG) |
| `group34`            | `OH`, `OZ`, `OI`, …       | CLS, REG        |

**Wenn Orange ein Feld numerisch inferiert:**
→ Nur die Exportrepräsentation in `orange_export.py` / `config.py` nachschärfen.
→ Kein Preprocessing, kein Feature Engineering, kein Sampling ändern.

## 3. Zeilenzahlen gegen Manifest prüfen

Öffne `export_manifest.csv` und vergleiche `n_rows` mit den tatsächlich geladenen Zeilen in Orange:

| Datei                | Stage | Split        | Erwartete Zeilen |
| -------------------- | ----- | ------------ | ---------------- |
| cls_train_full.csv   | CLS   | train_full   | 1,521,260        |
| cls_train_sample.csv | CLS   | train_sample | 456,377          |
| cls_val.csv          | CLS   | val          | 363,483          |
| cls_test.csv         | CLS   | test         | 349,447          |
| reg_train_full.csv   | REG   | train_full   | 332,546          |
| reg_train_sample.csv | REG   | train_sample | 99,770           |
| reg_val.csv          | REG   | val          | 87,746           |
| reg_test.csv         | REG   | test         | 82,403           |

Die konservativen Varianten haben dieselben Zeilenzahlen wie die entsprechenden Base-Splits.

## 4. Feature-Anzahl

| Feature-Set            | Erwartete Features (ohne Target) |
| ---------------------- | -------------------------------- |
| `CLS_FINAL`            | 28                               |
| `CLS_FINAL_NO_GROUPS`  | 26                               |
| `REG_FINAL`            | 26                               |
| `REG_FINAL_NO_GROUP34` | 25                               |

## 5. Empfohlene Startreihenfolge

Wichtig fuer dieses Projekt: Die Unterrichtslogik ist `Train -> Test -> Validation`.
Das heisst hier bewusst:

- `test` = Modellvergleich
- `val` = finale Endmessung

1. **Schneller erster Vergleich:** `cls_train_sample.csv` + `cls_test.csv`
2. **Analog REG:** `reg_train_sample.csv` + `reg_test.csv`
3. **Sensitivitätsvergleich:** `CLS_FINAL` vs. `CLS_FINAL_NO_GROUPS` auf `train_full/test`
4. **Analog REG:** `REG_FINAL` vs. `REG_FINAL_NO_GROUP34` auf `train_full/test`
5. **Finale Endmessung erst ganz am Schluss:** Gewinner einmal auf der passenden `val`-Datei messen, also Basis auf `cls_val.csv` / `reg_val.csv`, konservative Variante auf `cls_val_no_groups.csv` / `reg_val_no_group34.csv`

## 6. Abschluss

- [ ] Alle Zielspalten korrekt erkannt
- [ ] Alle kategorialen Felder als diskret erkannt
- [ ] Zeilenzahlen stimmen mit Manifest überein
- [ ] Kein unerwartetes Feature in den Spalten

**READY_FOR_ORANGE_MODELING = YES** wenn alle Punkte bestätigt.

# Funktionsübersicht

Kurze und direkte Beschreibung der wichtigsten Python-Module im Bereich Feature Engineering.
Funktionen mit führendem `_` sind interne Hilfsfunktionen.

## io_utils.py

- `ensure_output_dirs()`: Erstellt alle Ausgabeordner für Datasets, Audits, Metadaten und Orange-Exporte.
- `load_raw_data()`: Lädt die Rohdateien `train.csv` und `items.csv` aus dem konfigurierten Datenordner.
- `merge_train_items(df_train, df_items)`: Führt einen Left-Join von Train- und Item-Daten über `pid` aus.
- `save_parquet(df, path)`: Speichert ein DataFrame als Parquet-Datei und legt den Zielordner bei Bedarf an.
- `save_csv(df, path)`: Speichert ein DataFrame als CSV-Datei und legt den Zielordner bei Bedarf an.
- `save_text_report(text, path)`: Speichert einen Textbericht als Datei.

## Preprocessing/preprocessing.py

- `run_all_preprocessing(df)`: Bereinigt Rohdaten, normalisiert Kategorien und leitet Basis-Hilfsspalten wie `quantity` und `qty_suspicious` ab.

## Preprocessing/validation.py

- `assert_no_forbidden_features(feature_list, forbidden, context)`: Stoppt, wenn verbotene oder leakage-gefährdete Features in einer Liste auftauchen.
- `assert_no_duplicate_features(feature_list, context)`: Stoppt, wenn eine Featureliste doppelte Einträge enthält.
- `assert_cross_split_columns(frames, context)`: Prüft, ob mehrere Splits exakt dieselben Spalten in derselben Reihenfolge haben.
- `assert_reg_stage2_only(df, context)`: Prüft, ob ein REG-Datensatz nur gültige Stage-2-Zeilen enthält.
- `assert_preprocessing_integrity(df, context)`: Kontrolliert die wichtigsten Nachbedingungen des Preprocessing-Schritts.

## Preprocessing/audit.py

- `audit_join_quality(df_train_raw, df_items, df_merged)`: Bewertet, wie sauber der Join zwischen Train- und Item-Daten funktioniert hat.
- `audit_missingness(df)`: Liefert fehlende Werte pro Spalte inklusive Prozentanteil.
- `audit_outliers(df)`: Berechnet robuste Kennzahlen und einfache Extremwertsignale für numerische Kernspalten.
- `audit_target_distribution(df_train, df_val, df_test)`: Vergleicht Zielverteilungen und Mengenstatistiken über alle Splits.
- `audit_feature_sets(feature_matrices)`: Fasst Größe und Form aller erzeugten Feature-Matrizen zusammen.
- `audit_dropped_features()`: Dokumentiert bewusst ausgeschlossene Features samt Begründung.
- `run_full_audit(...)`: Führt alle Audit-Bausteine gesammelt aus und gibt die Reports als Dictionary zurück.

## Preprocessing/pid_segment.py

- `fit_pid_segment(df_train)`: Teilt PIDs anhand ihrer Häufigkeit in Head, Mid und Tail ein.
- `apply_pid_segment(df, pid_segment_map)`: Schreibt das ermittelte Segment für jede PID in einen DataFrame.
- `save_pid_segment_map(pid_segment_map, path)`: Speichert die PID-Segment-Zuordnung als CSV.
- `load_pid_segment_map(path)`: Lädt eine gespeicherte PID-Segment-Zuordnung wieder ein.

## Preprocessing/feature_engineering_safe.py

- `extract_group_parts(df)`: Zerlegt `group` in die Teilmerkmale `group12` und `group34`.
- `add_day_cycles(df)`: Ergänzt zyklische Tagesmerkmale für 7-, 14- und 30-Tage-Rhythmen.
- `_parse_single_content(raw)`: Parst einen einzelnen `content`-Wert in Packungsstruktur und Gesamtmenge.
- `parse_content(df)`: Wendet das Content-Parsing auf den gesamten DataFrame an und erzeugt Pack-Features.
- `add_has_campaign(df)`: Markiert, ob eine Kampagne vorhanden ist.
- `add_price_features(df)`: Leitet Preisabstände, Rabatte und Preisvergleichs-Flags ab.
- `add_per_unit_features(df)`: Berechnet Preise pro Einheit auf Basis der Gesamtpackungsgröße.
- `fit_binning_edges(df_train, n_bins)`: Lernt Quantil-Grenzen für Preis-Binning nur auf dem Train-Split.
- `apply_binned_features(df, bin_edges)`: Wandelt kontinuierliche Preismerkmale mit vorgegebenen Grenzen in Kategorien um.
- `fit_manufacturer_frequency(df_train)`: Bestimmt die relative Häufigkeit jedes Herstellers im Train-Split.
- `apply_manufacturer_frequency(df, freq_map)`: Schreibt die gelernte Herstellerhäufigkeit in einen DataFrame.
- `run_all_safe_features(df_train, df_val, df_test)`: Führt alle leakage-freien Feature-Engineering-Schritte aus und liefert zusätzlich Metadaten.

## Preprocessing/feature_engineering_conditional.py

- `_require_columns(df, cols, context)`: Stoppt, wenn für einen Verarbeitungsschritt Pflichtspalten fehlen.
- `_week_block(day, start)`: Wandelt Tageswerte in relative Wochenblöcke um.
- `compute_cumulative_features(df_train, df_val, df_test)`: Erzeugt zeitlich saubere kumulative Verlaufsmerkmale pro PID.
- `fit_global_aggregations(df_train)`: Lernt einfache gruppierte Mittelwerte auf dem gesamten Train-Split.
- `apply_global_aggregations(df, mappings)`: Wendet gelernte Gruppenmittelwerte auf einen DataFrame an.
- `_time_aware_forward_oof(df, group_col, target_col, cold_start, history_mask)`: Berechnet leakagesichere OOF-Encodings mit expandierenden Zeitfenstern.
- `_fit_full_train_encoding(df_train, group_col, target_col, global_fallback, history_mask)`: Lernt das finale Gruppen-Encoding auf dem kompletten Train-Split.
- `_apply_encoding(df, group_col, col_name, encoding)`: Mapped ein vortrainiertes Encoding auf neue Daten.
- `run_all_conditional_features(df_train, df_val, df_test)`: Führt alle bedingten, trainierten und zeitabhängigen Features gesammelt aus.

## Preprocessing/main_build_datasets.py

- `run_safe_only()`: Führt die komplette Pipeline nur mit sicheren Features aus.
- `run_safe_plus_conditional()`: Führt die komplette Pipeline inklusive conditional Features und Orange-Export aus.
- `_export_all(matrices, sampling_result, reports, fe_metadata)`: Speichert alle erzeugten Matrizen, Audits und Metadaten.
- `main()`: Liest den Modus aus der Kommandozeile und startet die passende Pipeline.

## Sampling/split.py

- `run_split(df)`: Zerlegt den Datensatz chronologisch in Train, Validation und Test.

## Sampling/sampling.py

- `add_week_block(df)`: Ergänzt einen Wochenblock relativ zum Trainingsstart.
- `sample_stratified(df, frac, strata_cols, seed)`: Zieht eine geschichtete Stichprobe und behandelt kleine Strata bewusst konservativ.
- `sample_cls(df_train, frac)`: Erstellt eine prototypische CLS-Train-Stichprobe.
- `sample_reg(df_train, frac)`: Erstellt eine prototypische REG-Train-Stichprobe nur aus gültigen Stage-2-Zeilen.
- `audit_sample_vs_population(df_pop, df_sample, cols)`: Vergleicht Verteilungen zwischen Grundgesamtheit und Stichprobe.
- `run_sampling(df_train)`: Führt CLS- und REG-Sampling inklusive Audit-Tabellen aus.
- `_require_columns(df, cols)`: Prüft, ob alle für das Sampling nötigen Spalten vorhanden sind.

## Sampling/feature_sets.py

- `get_feature_list(set_name)`: Gibt die Featureliste zu einem registrierten Set-Namen zurück.
- `validate_feature_list(df, feature_list, set_name)`: Prüft, ob alle Features einer Liste im DataFrame existieren.
- `assemble_X_y(df, feature_list, target_col, set_name)`: Baut validierte Feature- und Zielmatrizen aus einem DataFrame.
- `get_reg_mask(df)`: Definiert zentral, welche Zeilen für die Regressionsstufe gültig sind.
- `build_safe_feature_matrices(df_train, df_val, df_test)`: Erzeugt alle SAFE-Matrizen für Klassifikation und Regression.
- `summarize_feature_sets(feature_matrices)`: Erstellt eine tabellarische Übersicht aller Matrizen und Zielvektoren.
- `_print_summary(result)`: Druckt eine kompakte Konsolenzusammenfassung der erzeugten Matrizen.
- `build_conditional_feature_matrices(df_train, df_val, df_test)`: Erzeugt alle CONDITIONAL-Matrizen für Klassifikation und Regression.
- `export_matrices(feature_matrices, output_dir)`: Speichert alle Matrizen als Parquet-Dateien.

## Sampling/orange_export.py

- `_build_orange_df(df, features, target_col)`: Baut die Orange-kompatible Tabelle mit Zielspalte am Ende.
- `_validate_export(frames, stage, features, target_col)`: Prüft Exporte auf Spaltenkonsistenz, Leakage und fehlende Zielwerte.
- `export_orange_csvs(df_train, df_val, df_test, sampling_result, build_mode)`: Erzeugt alle Orange-CSV-Dateien inklusive Manifest.

## Experiment/feature_selection.py

- `_family_of(feat)`: Ordnet ein Feature seiner konfigurierten Familie zu.
- `_feat_type(s)`: Erkennt, ob ein Merkmal kategorial, binär oder numerisch ist.
- `_sub_idx(n, seed)`: Erzeugt bei großen Matrizen einen Zufallsindex für MI-Berechnungen auf Stichprobenbasis.
- `_safe_spearman(x, y)`: Berechnet robust die Spearman-Korrelation nur bei genügend gültigen Werten.
- `_safe_chi2(x, y, n_bins)`: Berechnet robust einen Chi-Quadrat-Score, bei Bedarf mit vorherigem Binning.
- `compute_filter_report(X, y, stage)`: Bewertet jedes Feature mit Missingness-, Varianz- und Signalmetriken.
- `find_redundant_pairs(X)`: Findet stark korrelierte numerische Feature-Paare.
- `build_family_report(cls_all, reg_all, cls_rd, reg_rd)`: Verdichtet Feature-Ergebnisse auf Familienebene.
- `_build_safe_vs_cond(cls_all, reg_all)`: Vergleicht SAFE- und CONDITIONAL-Features nach durchschnittlichem Signal.
- `_build_summary(reports, redundant, family_df, safe_vs_cond)`: Erstellt den ausführlichen Textbericht der Filterrunde.
- `_load(name)`: Lädt eine zuvor exportierte Feature-Matrix aus dem Output-Ordner.
- `run_feature_selection()`: Führt die komplette Filteranalyse für alle verfügbaren Matrizen aus.
- `main()`: Startet die Feature-Selection über die Kommandozeile.

## Experiment/feature_selection_r4.py

- `_all_safe()`: Liefert die vollständige sichere Featurebasis für die Pruning-Schritte.
- `_build_pruned_sets()`: Erzeugt die reduzierten Kandidatensets für CLS und REG.
- `build_candidate_reports()`: Schreibt Kandidatensets samt Begründungen als JSON und CSV.
- `_load(name)`: Lädt eine für Runde 4 benötigte Parquet-Matrix.
- `_encode_cats(df)`: Kodiert kategoriale Spalten numerisch für Modelltraining.
- `_prep_Xy(features, x_train_name, y_train_name, x_eval_name, y_eval_name, target_col)`: Lädt, kombiniert und bereitet Trainings- und Evaluationsdaten für ein Featureset vor.
- `_hgb_feature_importances(model, n_features)`: Schätzt Feature-Importances aus einem HistGradientBoosting-Modell.
- `_cls_metrics(y_true, y_pred, y_prob)`: Berechnet die zentralen Klassifikationsmetriken für Runde 4.
- `_reg_metrics(y_true, y_pred)`: Berechnet die zentralen Regressionsmetriken für Runde 4.
- `_subsample(X, y, n, seed)`: Reduziert sehr große Trainingsdaten für lineare Modelle auf eine Stichprobe.
- `run_embedded_cls(features, variant, Xt, yt, Xv, yv)`: Führt eingebettete Feature-Selektion für CLS mit mehreren Modellen aus.
- `run_embedded_reg(features, variant, Xt, yt, Xv, yv)`: Führt eingebettete Feature-Selektion für REG mit mehreren Modellen aus.
- `run_embedded_selection(sets)`: Startet die Embedded-Selektion für alle pruned Sets und speichert die Ergebnisse.
- `_embedded_summary(cls_df, reg_df, sets)`: Erstellt den Textbericht zur Embedded-Selektion.
- `_train_eval_cls(feats, Xt_full, yt, Xv_full, yv)`: Trainiert schnell ein CLS-Basismodell für Ablationstests.
- `_train_eval_reg(feats, Xt_full, yt, Xv_full, yv)`: Trainiert schnell ein REG-Basismodell für Ablationstests.
- `run_family_ablation(sets)`: Testet systematisch den Einfluss einzelner Feature-Familien.
- `_ablation_summary(cls_abl, reg_abl)`: Schreibt den zusammenfassenden Bericht zur Family-Ablation.
- `run_round4()`: Führt die komplette Runde 4 aus: Pruning, Embedded Selection und Family Ablation.

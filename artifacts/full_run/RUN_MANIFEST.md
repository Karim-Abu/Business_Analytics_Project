# Run Manifest

- Timestamp (UTC): `2026-05-17T20:47:42Z`
- Run type: **full**
- Build mode: `safe_only`
- Elapsed: `39.94 s`
- Orange export: skipped (only available in safe_plus_conditional)

## Inputs
- data_dir: `C:\Users\karim\OneDrive - FHNW\Documents\Analytics Project Code\data\raw`
- train_csv: `C:\Users\karim\OneDrive - FHNW\Documents\Analytics Project Code\data\raw\train.csv`
- items_csv: `C:\Users\karim\OneDrive - FHNW\Documents\Analytics Project Code\data\raw\items.csv`

## Row counts
- train_raw: 2,756,003
- items: 22,035
- merged: 2,756,003
- train: 1,521,260
- test: 349,447
- validation: 363,483

## Output directory
`C:\Users\karim\OneDrive - FHNW\Documents\Analytics Project Code\artifacts\full_run`

## What is in this folder?
- `datasets/` - Feature matrices and sampled training subsets (parquet).
- `audit/` - Audit reports (join quality, missingness, sampling, summary).
- `metadata/` - Reproducibility artefacts (pid_segment map, binning edges, encodings).
- `orange_exports/` - Orange-ready CSV exports (only in safe_plus_conditional).

## Generated files

| Path | Size (KB) |
|------|-----------|
| `audit/dropped_features.csv` | 0.5 |
| `audit/feature_matrix_summary.txt` | 1.0 |
| `audit/feature_sets.csv` | 0.5 |
| `audit/join_quality.csv` | 0.2 |
| `audit/missingness_merged.csv` | 0.2 |
| `audit/missingness_train.csv` | 0.5 |
| `audit/outliers_train.csv` | 0.4 |
| `audit/pharmform_group_target_summary.csv` | 0.7 |
| `audit/pharmform_mapping_coverage.csv` | 0.2 |
| `audit/pharmform_unmapped_top20.csv` | 0.7 |
| `audit/sampling_audit_cls.csv` | 0.6 |
| `audit/sampling_audit_reg.csv` | 0.6 |
| `audit/target_distribution.csv` | 0.2 |
| `benchmark/quick_benchmark_winners.csv` | 0.5 |
| `benchmark/model_benchmark_results.csv` | 1.9 |
| `benchmark/model_benchmark_summary.json` | 2.3 |
| `benchmark/model_benchmark_summary.txt` | 1.0 |
| `datasets/train_cls_sample.parquet` | 22009.9 |
| `datasets/train_reg_sample.parquet` | 5333.0 |
| `datasets/X_test_cls_base.parquet` | 9319.5 |
| `datasets/X_test_cls_expanded.parquet` | 9750.7 |
| `datasets/X_test_reg_base.parquet` | 2402.9 |
| `datasets/X_test_reg_expanded.parquet` | 2505.6 |
| `datasets/X_train_cls_base.parquet` | 40563.3 |
| `datasets/X_train_cls_expanded.parquet` | 42437.1 |
| `datasets/X_train_reg_base.parquet` | 9099.7 |
| `datasets/X_train_reg_expanded.parquet` | 9510.1 |
| `datasets/X_validation_cls_base.parquet` | 9744.5 |
| `datasets/X_validation_cls_expanded.parquet` | 10193.0 |
| `datasets/X_validation_reg_base.parquet` | 2566.7 |
| `datasets/X_validation_reg_expanded.parquet` | 2676.0 |
| `datasets/y_test_cls.parquet` | 49.7 |
| `datasets/y_test_reg.parquet` | 29.4 |
| `datasets/y_train_cls.parquet` | 205.8 |
| `datasets/y_train_reg.parquet` | 111.2 |
| `datasets/y_validation_cls.parquet` | 52.0 |
| `datasets/y_validation_reg.parquet` | 31.4 |
| `metadata/binning_edges.json` | 1.0 |
| `metadata/pid_segment_map.csv` | 233.1 |

## Notes
- Full run uses the real dataset from data/raw/.
- Quick benchmark results are diagnostic only and are available in benchmark/. They are not the final model selection.

## Suggested next steps
- Inspect feature matrices in `datasets/`
- Inspect audit reports in `audit/`
- For real data, run: python scripts/run_pipeline.py --full
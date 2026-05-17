# PharmForm Ablation Report

## Decision

- Decision: Kandidat für Challenger-Feature-Evaluation; potenziell geeignet zur Beibehaltung
- Primary decision model: HistGradientBoostingClassifier
- Models run: HistGradientBoostingClassifier
- Best variant by TEST F1 on primary model: CLS_PHARMFORM_V2_GROUP
- TEST F1 delta vs V0: 0.000253
- VAL F1 delta vs V0: 0.000108
- TEST PR-AUC delta vs V0: 0.002172
- VAL PR-AUC delta vs V0: 0.001593
- HistGB challenger variant: CLS_PHARMFORM_V2_GROUP

## Partial Outputs

- Partial metrics CSV: pharmform_ablation_metrics_partial.csv
- Partial metrics JSONL: pharmform_ablation_metrics_partial.jsonl
- Partial error log: partial_error_log.txt

## Best Variant By Model

```text
                         model best_variant_by_test_f1  delta_test_f1_vs_v0  delta_val_f1_vs_v0  delta_test_pr_auc_vs_v0  delta_val_pr_auc_vs_v0  delta_test_precision_vs_v0  delta_test_recall_vs_v0
HistGradientBoostingClassifier  CLS_PHARMFORM_V2_GROUP             0.000253            0.000108                 0.002172                0.001593                   -0.000069                  0.00136
```

## HistGB Candidate Deltas

Decision threshold: TEST F1 or TEST PR-AUC delta >= 0.002; corresponding VAL delta must be non-negative.

```text
                         variant  delta_test_f1  delta_val_f1  delta_test_pr_auc  delta_val_pr_auc  delta_test_precision  delta_test_recall
          CLS_PHARMFORM_V1_CLEAN      -0.000033      0.000208           0.001849          0.000693              0.000630          -0.002506
          CLS_PHARMFORM_V2_GROUP       0.000253      0.000108           0.002172          0.001593             -0.000069           0.001360
CLS_PHARMFORM_V3_CLEAN_AND_GROUP      -0.000033      0.000208           0.001849          0.000693              0.000630          -0.002506
```

## Mapping Coverage

```text
 events_total  events_mapped  events_missing  events_unmapped  coverage_mapped_by_events  pids_total  pids_mapped  pids_missing  pids_unmapped  coverage_mapped_by_pid
      2234190        1653873          159024           421293                   0.740256       21733        14342          2276           5115                0.659918
```

## Variant Metrics

Threshold strategy `test_f1_opt` optimizes on TEST and applies the same threshold to VAL.

### HistGradientBoostingClassifier

```text
                         model                          variant eval_split  threshold  n_features  precision   recall       f1  roc_auc   pr_auc
HistGradientBoostingClassifier        CLS_PHARMFORM_V0_BASELINE       TEST       0.23          27   0.365268 0.709484 0.482254 0.718793 0.416340
HistGradientBoostingClassifier        CLS_PHARMFORM_V0_BASELINE        VAL       0.23          27   0.373614 0.710268 0.489659 0.720630 0.426556
HistGradientBoostingClassifier           CLS_PHARMFORM_V1_CLEAN       TEST       0.23          28   0.365897 0.706978 0.482221 0.719190 0.418188
HistGradientBoostingClassifier           CLS_PHARMFORM_V1_CLEAN        VAL       0.23          28   0.373888 0.710156 0.489867 0.720588 0.427249
HistGradientBoostingClassifier           CLS_PHARMFORM_V2_GROUP       TEST       0.23          30   0.365198 0.710844 0.482507 0.719145 0.418511
HistGradientBoostingClassifier           CLS_PHARMFORM_V2_GROUP        VAL       0.23          30   0.373631 0.710659 0.489766 0.720820 0.428149
HistGradientBoostingClassifier CLS_PHARMFORM_V3_CLEAN_AND_GROUP       TEST       0.23          31   0.365897 0.706978 0.482221 0.719190 0.418188
HistGradientBoostingClassifier CLS_PHARMFORM_V3_CLEAN_AND_GROUP        VAL       0.23          31   0.373888 0.710156 0.489867 0.720588 0.427249
```

## Default Threshold Check

```text
                         model                          variant eval_split  threshold  n_features  precision   recall       f1  roc_auc   pr_auc
HistGradientBoostingClassifier        CLS_PHARMFORM_V0_BASELINE       TEST        0.5          27   0.538462 0.085949 0.148236 0.718793 0.416340
HistGradientBoostingClassifier        CLS_PHARMFORM_V0_BASELINE        VAL        0.5          27   0.546855 0.075111 0.132080 0.720630 0.426556
HistGradientBoostingClassifier           CLS_PHARMFORM_V1_CLEAN       TEST        0.5          28   0.542260 0.083216 0.144289 0.719190 0.418188
HistGradientBoostingClassifier           CLS_PHARMFORM_V1_CLEAN        VAL        0.5          28   0.552519 0.078301 0.137163 0.720588 0.427249
HistGradientBoostingClassifier           CLS_PHARMFORM_V2_GROUP       TEST        0.5          30   0.541019 0.085614 0.147835 0.719145 0.418511
HistGradientBoostingClassifier           CLS_PHARMFORM_V2_GROUP        VAL        0.5          30   0.552364 0.084591 0.146713 0.720820 0.428149
HistGradientBoostingClassifier CLS_PHARMFORM_V3_CLEAN_AND_GROUP       TEST        0.5          31   0.542260 0.083216 0.144289 0.719190 0.418188
HistGradientBoostingClassifier CLS_PHARMFORM_V3_CLEAN_AND_GROUP        VAL        0.5          31   0.552519 0.078301 0.137163 0.720588 0.427249
```

## Fixed Threshold 0.22 Check

```text
                         model                          variant eval_split  threshold  n_features  precision   recall       f1  roc_auc   pr_auc
HistGradientBoostingClassifier        CLS_PHARMFORM_V0_BASELINE       TEST       0.22          27   0.359380 0.731523 0.481977 0.718793 0.416340
HistGradientBoostingClassifier        CLS_PHARMFORM_V0_BASELINE        VAL       0.22          27   0.366935 0.734756 0.489443 0.720630 0.426556
HistGradientBoostingClassifier           CLS_PHARMFORM_V1_CLEAN       TEST       0.22          28   0.359596 0.730532 0.481955 0.719190 0.418188
HistGradientBoostingClassifier           CLS_PHARMFORM_V1_CLEAN        VAL       0.22          28   0.368024 0.732831 0.489982 0.720588 0.427249
HistGradientBoostingClassifier           CLS_PHARMFORM_V2_GROUP       TEST       0.22          30   0.359381 0.732382 0.482164 0.719145 0.418511
HistGradientBoostingClassifier           CLS_PHARMFORM_V2_GROUP        VAL       0.22          30   0.367118 0.734734 0.489602 0.720820 0.428149
HistGradientBoostingClassifier CLS_PHARMFORM_V3_CLEAN_AND_GROUP       TEST       0.22          31   0.359596 0.730532 0.481955 0.719190 0.418188
HistGradientBoostingClassifier CLS_PHARMFORM_V3_CLEAN_AND_GROUP        VAL       0.22          31   0.368024 0.732831 0.489982 0.720588 0.427249
```

## Notes

- LogisticRegression uses median numeric imputation, sparse one-hot categoricals, scaling, and class_weight=balanced.
- HistGradientBoostingClassifier uses median numeric imputation and ordinal categorical encoding; no dense one-hot matrix is created.
- HistGB parameters: max_iter=100, learning_rate=0.1, max_leaf_nodes=31, random_state=cfg.SEED.
- HistGB native categorical_features: enabled for ordinal-encoded categorical columns with cardinality <= 255; higher-cardinality categorical columns are ordinal-encoded and treated as numeric because sklearn HistGB cannot use them as native categorical features.
- HistGradientBoostingClassifier: Finalmodellnaeherer nichtlinearer Ablation-Check mit ordinal kodierten Kategorien und nativer categorical_features-Maske fuer Spalten mit maximal 255 Auspraegungen.
- No hyperparameter search was run.
- Existing train/test/validation split is used as exported by the pipeline.
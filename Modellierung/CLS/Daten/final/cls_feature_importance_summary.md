# CLS Feature Importance (Permutation, TEST-Sample) - Summary

_Generated: 2026-05-07T16:25:28Z_

## Methode

- Modell: HistGradientBoostingClassifier mit identischen Parametern aus `full_scale_cls_modeling.ipynb` (`{'learning_rate': 0.05, 'max_iter': 300, 'max_depth': 8, 'random_state': 42}`).
- Trainingsdaten: `cls_train_full.csv` (natuerliche Klassenverteilung).
- Evaluation: TEST-Sample stratifiziert nach `order`.
- Permutation Importance via `sklearn.inspection.permutation_importance`, n_repeats=3, random_state=42.
- Primaere Metrik: AUC-Delta. Sekundaer: F1-Delta@0.22, MCC-Delta@0.22.

## Reproduktion der TEST-Referenz

```
   metric  reproduced  reference     delta
      auc    0.719114   0.719114 -0.000000
  logloss    0.493439   0.493439 -0.000000
  f1@0.22    0.482480   0.482480  0.000000
prec@0.22    0.360131   0.360131  0.000000
 rec@0.22    0.730735   0.730735  0.000000
 mcc@0.22    0.274325   0.274325 -0.000000
```

- Reproduktionstoleranz: AUC +-0.005, F1@0.22 +-0.01. Toleranzen eingehalten.

## Run-Parameter

- Daten verwendet: TEST-Sample (nicht VAL, nicht TRAIN, nicht volles TEST).
- Sample-Groesse: 50000 (Ziel 50000).
- n_repeats: 3
- Permutation-Laufzeit: 70.1s (1.2 min)
- Modell-Fit-Laufzeit: 33.9s
- Threshold (fix, fuer F1/MCC-Delta): 0.22

## Top-20 Features (sortiert nach AUC-Delta)

```
 rank                 feature             feature_family leakage_audit_status  importance_mean_auc_delta  importance_std_auc_delta  f1_022_delta_mean  mcc_022_delta_mean
    1                pid_prob        product_history_pid                 safe                   0.123130                  0.002299           0.104480            0.149010
    2              order_time        product_history_pid                 safe                   0.076030                  0.002374           0.067797            0.085049
    3             basket_time        product_history_pid                 safe                   0.030154                  0.001345           0.027780            0.041435
    4              click_time        product_history_pid                 safe                   0.019850                  0.000460           0.016491            0.026996
    5        pid_total_events        product_history_pid                 safe                   0.002940                  0.000196           0.002324            0.003394
    6            availability               availability         not_in_audit                   0.002716                  0.000348           0.001313            0.002217
    7             pid_segment        product_history_pid                 safe                   0.000765                  0.000069           0.001174            0.001962
    8           group34_order              group_history           suspicious                   0.000579                  0.000230           0.000272            0.000453
    9          price_per_unit             price_discount         not_in_audit                   0.000460                  0.000099           0.000696            0.001178
   10                   price             price_discount         not_in_audit                   0.000456                  0.000034           0.000808            0.001396
   11         competitorPrice             price_discount         not_in_audit                   0.000433                  0.000058           0.000507            0.000870
   12            discount_bin             price_discount         not_in_audit                   0.000380                  0.000026           0.000116            0.000209
   13                   day_7                       time         not_in_audit                   0.000358                  0.000149           0.000709            0.001194
   14           category_norm product_category_pharmform         not_in_audit                   0.000354                  0.000009           0.000370            0.000636
   15                  day_14                       time         not_in_audit                   0.000224                  0.000187           0.001304            0.002220
   16          pharmForm_norm product_category_pharmform         not_in_audit                   0.000209                  0.000027           0.000990            0.001686
   17                 group34              group_history         not_in_audit                   0.000188                  0.000038           0.000693            0.001189
   18 availability_likelihood               availability                 safe                   0.000188                  0.000055           0.000144            0.000230
   19                  day_30                       time         not_in_audit                   0.000167                  0.000060          -0.000302           -0.000527
   20          price_diff_bin             price_discount         not_in_audit                   0.000114                  0.000034           0.000148            0.000249
```

## Familien-Aggregation

```
            feature_family  n_features  auc_delta_sum  auc_delta_mean  auc_delta_max  f1_022_delta_sum
       product_history_pid           6       0.252868        0.042145       0.123130          0.220047
              availability           2       0.002903        0.001452       0.002716          0.001457
            price_discount           6       0.001855        0.000309       0.000460          0.002377
             group_history           4       0.000934        0.000234       0.000579          0.001442
                      time           4       0.000675        0.000169       0.000358          0.002859
product_category_pharmform           2       0.000563        0.000281       0.000354          0.001360
           campaign_adflag           2       0.000033        0.000017       0.000029         -0.000011
                     other           2       0.000003        0.000002       0.000003         -0.000074
```

## Beobachtungen

Die staerksten AUC-Treiber (Top 5):
- `pid_prob` (Familie: product_history_pid, Audit: safe) -> AUC-Delta 0.1231 +- 0.0023, F1@0.22-Delta 0.1045.
- `order_time` (Familie: product_history_pid, Audit: safe) -> AUC-Delta 0.0760 +- 0.0024, F1@0.22-Delta 0.0678.
- `basket_time` (Familie: product_history_pid, Audit: safe) -> AUC-Delta 0.0302 +- 0.0013, F1@0.22-Delta 0.0278.
- `click_time` (Familie: product_history_pid, Audit: safe) -> AUC-Delta 0.0198 +- 0.0005, F1@0.22-Delta 0.0165.
- `pid_total_events` (Familie: product_history_pid, Audit: safe) -> AUC-Delta 0.0029 +- 0.0002, F1@0.22-Delta 0.0023.

## Audit-Caveat: suspicious Features in Top 20

- `group34_order`: Same caveat as group12_order

Diese Features werden ausschliesslich als historisches Gruppenpopularitaetssignal interpretiert. Sie liefern keine kausale Aussage ueber Kauftreiber.

## Wichtige Begrenzungen

- Permutation Importance misst Verschlechterung bei zerstoerter Feature-Information; sie zeigt prediktive Relevanz, keine Kausalitaet.
- Korrelierte Features koennen Importance untereinander aufteilen; Familien-Aggregation hilft, diese Effekte sichtbar zu machen.
- Keine VAL-Auswertung verwendet. Keine Modell- oder Threshold-Entscheidung geaendert.

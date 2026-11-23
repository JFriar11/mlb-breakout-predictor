# Week 3 modeling report

## Executive summary

M2 led on mean historical Average Precision (0.3682).
The best new model was M7 (0.3449 mean AP). M2, the Week 2 winner
on 2023 validation AP, averaged 0.3682 in backtesting.
The mean gap between M2 and M3 was
0.0005, compared with
M2's AP range of 0.2906 to 0.4840 across seasons. M1 led on mean
Precision@20 (45.0%) and M1 achieved
the lowest mean Brier score (0.1239). The tables show which model
families gained from Statcast and how those gains varied by season.

## Backtesting results

| validation_season | training_feature_seasons | training_outcome_max | training_observations | validation_observations | validation_breakouts | validation_prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| 2021 | 2017 2018 | 2019 | 406 | 192 | 35 | 0.1823 |
| 2022 | 2017 2018 2021 | 2022 | 598 | 202 | 25 | 0.1238 |
| 2023 | 2017 2018 2021 2022 | 2023 | 800 | 207 | 36 | 0.1739 |

Each prediction is made after the feature season ends. Training labels come from outcome
seasons completed by that date. The target remains a gain of at least 20 unrounded wRC+
points. Each experiment fits a fresh pipeline on identical training rows, using the
original class frequencies. HGB uses 150 iterations, 15 leaves, a 0.05 learning rate
and L2 of 1. XGBoost uses 200 depth-3 trees, a 0.05 learning rate and 0.8 row and
column subsampling. Early stopping was disabled. Exact parameters are in the run metadata.

| ID | Model | Features | Mean AP | SD AP | Mean AUC | Mean Brier | Mean P@20 | Mean R@20 | Pooled AP | 2021 AP | 2022 AP | 2023 AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 | Logistic Regression | traditional | 0.3682 | 0.1022 | 0.7279 | 0.1261 | 0.4 | 0.2485 | 0.3243 | 0.33 | 0.2906 | 0.484 |
| M3 | Logistic Regression | traditional_statcast | 0.3677 | 0.0916 | 0.7511 | 0.1282 | 0.4167 | 0.2545 | 0.3407 | 0.3827 | 0.2695 | 0.4508 |
| M5 | Random Forest | traditional_statcast | 0.3651 | 0.1025 | 0.7213 | 0.1253 | 0.4 | 0.2523 | 0.3226 | 0.3226 | 0.2906 | 0.4819 |
| M1 | Logistic Regression | baseline | 0.3627 | 0.0991 | 0.7005 | 0.1239 | 0.45 | 0.2806 | 0.3374 | 0.3588 | 0.2657 | 0.4637 |
| M4 | Random Forest | traditional | 0.3459 | 0.0878 | 0.7137 | 0.1262 | 0.4 | 0.2483 | 0.3131 | 0.2922 | 0.2983 | 0.4472 |
| M7 | HistGradientBoosting | traditional_statcast | 0.3449 | 0.0833 | 0.6973 | 0.1363 | 0.35 | 0.2164 | 0.3214 | 0.3327 | 0.2683 | 0.4336 |
| M9 | XGBoost | traditional_statcast | 0.3235 | 0.0844 | 0.6987 | 0.1366 | 0.3833 | 0.2311 | 0.2938 | 0.3198 | 0.2411 | 0.4097 |
| M6 | HistGradientBoosting | traditional | 0.3166 | 0.0892 | 0.6784 | 0.138 | 0.3167 | 0.2017 | 0.2804 | 0.2408 | 0.2941 | 0.4149 |
| M8 | XGBoost | traditional | 0.3122 | 0.069 | 0.6868 | 0.1348 | 0.3667 | 0.2338 | 0.2789 | 0.2661 | 0.279 | 0.3915 |

Mean and sample SD weight each fold equally. Pooled AP ranks all historical scores together,
so it can respond to shifts in score scale across seasons. Precision@20 and Recall@20 in
the comparison refer to separate top-20 lists each season. The full fold CSV also includes
ROC-AUC, Brier, precision, recall, F1, Precision@10 and confusion counts at 0.50.
Zero-positive fold/model combinations at 0.50 were M5 (2023).

<details>
<summary>Fold-level ranking and probability metrics</summary>

| ID | Season | AP | ROC-AUC | Brier | P@20 | R@20 |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | 2021 | 0.3588 | 0.6943 | 0.1409 | 0.4 | 0.2286 |
| M2 | 2021 | 0.33 | 0.6797 | 0.148 | 0.35 | 0.2 |
| M3 | 2021 | 0.3827 | 0.7394 | 0.141 | 0.5 | 0.2857 |
| M4 | 2021 | 0.2922 | 0.6379 | 0.1505 | 0.3 | 0.1714 |
| M5 | 2021 | 0.3226 | 0.6639 | 0.1455 | 0.3 | 0.1714 |
| M6 | 2021 | 0.2408 | 0.5916 | 0.1689 | 0.25 | 0.1429 |
| M7 | 2021 | 0.3327 | 0.6528 | 0.1628 | 0.3 | 0.1714 |
| M8 | 2021 | 0.2661 | 0.6127 | 0.1651 | 0.3 | 0.1714 |
| M9 | 2021 | 0.3198 | 0.6562 | 0.1598 | 0.35 | 0.2 |
| M1 | 2022 | 0.2657 | 0.6915 | 0.1058 | 0.35 | 0.28 |
| M2 | 2022 | 0.2906 | 0.7482 | 0.1094 | 0.3 | 0.24 |
| M3 | 2022 | 0.2695 | 0.711 | 0.1235 | 0.25 | 0.2 |
| M4 | 2022 | 0.2983 | 0.7668 | 0.1049 | 0.3 | 0.24 |
| M5 | 2022 | 0.2906 | 0.7281 | 0.1093 | 0.35 | 0.28 |
| M6 | 2022 | 0.2941 | 0.7297 | 0.1143 | 0.3 | 0.24 |
| M7 | 2022 | 0.2683 | 0.6949 | 0.1208 | 0.25 | 0.2 |
| M8 | 2022 | 0.279 | 0.7419 | 0.1108 | 0.35 | 0.28 |
| M9 | 2022 | 0.2411 | 0.6994 | 0.1237 | 0.2 | 0.16 |
| M1 | 2023 | 0.4637 | 0.7156 | 0.125 | 0.6 | 0.3333 |
| M2 | 2023 | 0.484 | 0.7558 | 0.1208 | 0.55 | 0.3056 |
| M3 | 2023 | 0.4508 | 0.8028 | 0.12 | 0.5 | 0.2778 |
| M4 | 2023 | 0.4472 | 0.7365 | 0.1233 | 0.6 | 0.3333 |
| M5 | 2023 | 0.4819 | 0.7718 | 0.1213 | 0.55 | 0.3056 |
| M6 | 2023 | 0.4149 | 0.7138 | 0.1308 | 0.4 | 0.2222 |
| M7 | 2023 | 0.4336 | 0.7443 | 0.1253 | 0.5 | 0.2778 |
| M8 | 2023 | 0.3915 | 0.7057 | 0.1285 | 0.45 | 0.25 |
| M9 | 2023 | 0.4097 | 0.7406 | 0.1263 | 0.6 | 0.3333 |

</details>

## Baseball interpretation

### Incremental Statcast value

| model_name | mean_delta_AP | AP_gain_folds | mean_delta_P20 | mean_delta_AUC | mean_delta_Brier |
| --- | --- | --- | --- | --- | --- |
| HistGradientBoosting | 0.0283 | 2 | 0.0333 | 0.019 | -0.0017 |
| Logistic Regression | -0.0005 | 1 | 0.0167 | 0.0231 | 0.0021 |
| Random Forest | 0.0192 | 2 | 0.0 | 0.0075 | -0.0009 |
| XGBoost | 0.0113 | 2 | 0.0167 | 0.012 | 0.0018 |

AP gains count strict improvements out of three folds. Every tested family lost AP in 2022.
Mean AP and mean ROC-AUC capture different parts of ranking performance. Correlated traditional and Statcast
variables can carry overlapping signal, including observed and expected wOBA.

### Feature importance

| feature | mean_importance | fold_std |
| --- | --- | --- |
| woba | -0.6574 | 0.1968 |
| wrc_plus | -0.5361 | 0.2119 |
| bb_rate | 0.3185 | 0.1309 |
| age | -0.3153 | 0.0359 |
| iso | 0.2474 | 0.2867 |

M2 coefficients represent approximate shifts in log odds per training-set standard
deviation, conditional on other predictors. Correlated measures can make signs
and magnitudes unstable. The tree results below measure average AP loss after
permuting one feature on historical validation rows. Variation across folds and
average repeat SD describe sensitivity to season and permutation, respectively.

| feature | mean_importance | fold_std | mean_permutation_std |
| --- | --- | --- | --- |
| wrc_plus | 0.0655 | 0.0452 | 0.0334 |
| woba | 0.05 | 0.0258 | 0.0325 |
| avg_exit_velocity | 0.0221 | 0.0035 | 0.0169 |
| babip | 0.0148 | 0.0189 | 0.0291 |
| bb_rate | 0.0139 | 0.0101 | 0.0122 |

Selected variables in M5, with mean permutation AP drop, include wrc_plus (+0.0655), age (+0.0106), babip (+0.0148), k_rate (+0.0096), bb_rate (+0.0139), chase_rate (+0.0033), contact_rate (-0.0004), avg_exit_velocity (+0.0221), barrel_rate (+0.0031), xwoba (+0.0044).
The starting wRC+ score is mathematically coupled to the improvement target.
Regression to the mean can contribute to this association. The same 20-point
gain also requires a higher ending wRC+ for a hitter with a higher starting wRC+.
Permutation importances describe model reliance and can be diluted among correlated features.

### Hitter age

| experiment_id | age_group | observations | breakouts | prevalence | average_precision | roc_auc | brier_score | precision_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | ≤24 | 96 | 22 | 0.2292 | 0.4406 | 0.6714 | 0.1639 | 0.5 |
| M1 | 25–29 | 311 | 49 | 0.1576 | 0.3714 | 0.7095 | 0.12 | 0.5 |
| M1 | ≥30 | 194 | 25 | 0.1289 | 0.2422 | 0.6764 | 0.1095 | 0.2 |
| M2 | ≤24 | 96 | 22 | 0.2292 | 0.4116 | 0.656 | 0.1692 | 0.4 |
| M2 | 25–29 | 311 | 49 | 0.1576 | 0.3599 | 0.731 | 0.121 | 0.4 |
| M2 | ≥30 | 194 | 25 | 0.1289 | 0.2458 | 0.6798 | 0.1115 | 0.1 |
| M7 | ≤24 | 96 | 22 | 0.2292 | 0.4112 | 0.664 | 0.186 | 0.5 |
| M7 | 25–29 | 311 | 49 | 0.1576 | 0.3704 | 0.7537 | 0.1229 | 0.5 |
| M7 | ≥30 | 194 | 25 | 0.1289 | 0.1967 | 0.5572 | 0.1316 | 0.1 |

M1 exceeded the within-group prevalence reference by +0.211 AP for ages ≤24, +0.214 AP for ages 25–29, +0.113 AP for ages ≥30. M2 exceeded the within-group prevalence reference by +0.182 AP for ages ≤24, +0.202 AP for ages 25–29, +0.117 AP for ages ≥30. M7 exceeded the within-group prevalence reference by +0.182 AP for ages ≤24, +0.213 AP for ages 25–29, +0.068 AP for ages ≥30. A constant score within each age band has AP equal to that band's
prevalence. The within-group AP gains show ranking signal beyond the age-group
base rate. M7 had a ROC-AUC of 0.56 in the oldest band and 0.75 at ages 25–29. Differences in the amount of gain are descriptive. Week 1's full-data
prevalences were 23.53%, 15.16%, and 12.04% from youngest to oldest.

## Error analysis

| player_name | season | wrc_plus | next_season_wrc_plus | predicted_probability | error_category |
| --- | --- | --- | --- | --- | --- |
| Jackie Bradley Jr. | 2021 | 36.3139 | 56.0329 | 0.863 | false_positive |
| Paul Goldschmidt | 2021 | 135.9161 | 175.4347 | 0.0244 | false_negative |
| Cody Bellinger | 2021 | 47.1504 | 83.1713 | 0.9011 | true_positive |

The [case studies](week3_case_studies.md) document at least three examples per category
and identify near misses and overlooked breakouts in the annual top-20 lists.
Feature-season context distinguishes hitters with favorable expected production
from those who started with weaker contact or more strikeouts. The available observations
support descriptions of starting profiles and realized wRC+ changes.

## Calibration and thresholds

| experiment_id | method | observations | prevalence | mean_probability | brier_score | intercept | slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | raw | 601 | 0.1597 | 0.1608 | 0.1236 | -0.2827 | 0.8153 |
| M1 | sigmoid | 601 | 0.1597 | 0.1547 | 0.1249 | 0.1002 | 1.0383 |
| M2 | raw | 601 | 0.1597 | 0.1754 | 0.1257 | -0.493 | 0.7297 |
| M2 | sigmoid | 601 | 0.1597 | 0.1496 | 0.1259 | -0.0911 | 0.8908 |
| M7 | raw | 601 | 0.1597 | 0.1626 | 0.1358 | -0.811 | 0.4152 |
| M7 | sigmoid | 601 | 0.1597 | 0.1482 | 0.1264 | 0.1115 | 1.011 |

Each sigmoid fits on inner out-of-fold log odds from earlier seasons. The learned
mapping is applied to scores from a fresh base model fit on the full outer training
window. Raw and sigmoid Brier scores use the same outer rows. Pooled calibration intercept
and slope regress outcomes on score log odds for diagnostics only. Ideal values are
0 and 1. Reliability counts are saved alongside the calibration comparison.

For M2, paired pooled Brier changed from 0.1257 to 0.1259 after sigmoid calibration. Brier improved in 2 of 3 paired
outer seasons for M2. The earliest sigmoid uses just one inner validation
season. Later folds pool additional inner seasons whose prevalence and base-model
training sizes can differ.

| threshold | predicted_breakouts | precision | recall | f1 | fp | fn |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 389 | 0.2082 | 0.8438 | 0.334 | 308 | 15 |
| 0.15 | 273 | 0.2527 | 0.7188 | 0.374 | 204 | 27 |
| 0.2 | 183 | 0.3115 | 0.5938 | 0.4086 | 126 | 39 |
| 0.25 | 135 | 0.3556 | 0.5 | 0.4156 | 87 | 48 |
| 0.3 | 95 | 0.3789 | 0.375 | 0.377 | 59 | 60 |
| 0.4 | 44 | 0.3864 | 0.1771 | 0.2429 | 27 | 79 |
| 0.5 | 22 | 0.4091 | 0.0938 | 0.1525 | 13 | 87 |

At 0.50, M2 flagged 22 of the
601 qualifying hitters, with
9.4% recall. At 0.20 it flagged
183.
Low prevalence and concentrated scores make 0.50 a selective screen. The Brier comparison and reliability curves diagnose probability calibration,
while the cutoff grid shows the precision and recall tradeoff. These pooled
cutoffs are retrospective. No operational threshold was selected. The historical
results support a fixed-capacity scouting shortlist, with threshold-based flags
requiring a cost or capacity target and further prospective calibration evidence.

## Limitations

Three folds and small samples limit precision, especially for subgroups and
calibration bins. The rare positive class magnifies changes from a few hitters.
The shortened 2020 season leaves two transitions empty under the unchanged
300-PA criterion. Requiring 300 PA in the outcome season retrospectively conditions
the population on future playing time. Repeat hitters link observations across
seasons. Regression to the mean and target coupling can inflate apparent starting
performance signal. Calibration transfer from smaller inner training windows to
outer refits may shift the probability scale. Repeated model comparisons on these folds can favor a
historical winner by chance.

## Final model recommendation

Use M2 provisionally for ranking, based on mean fold AP with exact ties
resolved by experiment ID. Its mean shortlist hit rate was 40.0%.
The fold spreads and competing models' results should guide interpretation of
small differences. The raw scores and sigmoid evidence can inform later work on
probability-based flags.

## Week 4 readiness

Before final testing, freeze M2 as primary and M1 as the comparison using
eligible historical data and raw probabilities. The
[Week 4 evaluation protocol](week4_evaluation_protocol.md) defines the single
2024→2025 evaluation and the secondary top-20 scouting analysis. The
[freeze manifest](../../models/week4/freeze_manifest.json) records the fitted
pipelines and protocol hashes. Week 3 used historical folds for model selection.

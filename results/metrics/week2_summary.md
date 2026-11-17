# Week 2 validation report

## Dataset and methodology

The existing Week 1 binary label marks a following-season improvement of at least 20 points in unrounded FanGraphs wRC+. Each row represents a hitter with at least 300 PA in both seasons. There are 1,206 qualifying player-seasons.

| split | feature_seasons | outcome_seasons | observations | positives | prevalence |
| --- | --- | --- | --- | --- | --- |
| train | 2017, 2018, 2021, 2022 | 2018, 2019, 2022, 2023 | 800 | 128 | 16.00% |
| validation | 2023 | 2024 | 207 | 36 | 17.39% |
| test | 2024 | 2025 | 199 | 24 | 12.06% |

- **baseline** includes `age`, `wrc_plus`
- **traditional** includes `age`, `pa`, `wrc_plus`, `woba`, `iso`, `babip`, `bb_rate`, `k_rate`, `chase_rate`, `contact_rate`, `swstr_rate`
- **traditional_statcast** includes `age`, `pa`, `wrc_plus`, `woba`, `iso`, `babip`, `bb_rate`, `k_rate`, `chase_rate`, `contact_rate`, `swstr_rate`, `avg_exit_velocity`, `hard_hit_rate`, `barrel_rate`, `avg_launch_angle`, `xba`, `xwoba`

The five pipelines fit on the training partition only. Each uses median imputation. Logistic regression additionally standardizes predictors and uses default regularization, a 2,000-iteration limit and seed 42. Random forests use 300 trees, depth at most 6, at least 5 observations per leaf and seed 42. The experiments use the original class frequencies and exclude identifiers and outcome fields via the Week 1 allowlist. Split membership is fixed by season and unique by player-season. The 2025 holdout contributed only split audit counts and row identifiers, with no model scoring or selection.

## Validation results

| Experiment | Model | Features | AP | ROC-AUC | Precision | Recall | F1 | Precision@20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | Logistic Regression | baseline | 0.4637 | 0.7156 | 0.75 | 0.0833 | 0.15 | 0.6 |
| M2 | Logistic Regression | traditional | 0.484 | 0.7558 | 0.8 | 0.1111 | 0.1951 | 0.55 |
| M3 | Logistic Regression | traditional_statcast | 0.4508 | 0.8028 | 0.4615 | 0.1667 | 0.2449 | 0.5 |
| M4 | Random Forest | traditional | 0.4472 | 0.7365 | 0.6667 | 0.0556 | 0.1026 | 0.6 |
| M5 | Random Forest | traditional_statcast | 0.4819 | 0.7718 | 0.0 | 0.0 | 0.0 | 0.55 |
| constant | Constant prevalence | none | 0.1739 | 0.5 | 0.0 | 0.0 | 0.0 | 0.2 |

Average Precision is computed with scikit-learn's `average_precision_score`. The constant score is the training prevalence (16.00%); its AP equals the validation prevalence (17.39%). Its top-k hit rates reflect the deterministic player-ID tie order. The metrics CSV includes Brier score, accuracy, confusion counts, top-k metrics and deltas versus M1.

Classification uses the prespecified 0.50 cutoff. Undefined precision when a model calls zero positives is recorded as zero (M5, constant). Recall has a positive denominator in this validation set. The top 20 are ranked by probability, then player ID and season. The leaderboard's correctness column uses the 0.50 class prediction, while Precision@20 counts hits if all 20 ranked hitters are treated as breakout candidates.

**M2** was selected on validation AP (0.4840), with exact ties resolved by experiment ID. The fitted training-only pipeline and 0.50 threshold are frozen for optional final evaluation.

## Incremental value of Statcast

Deltas are Statcast minus traditional on the same validation observations. Lower Brier scores are better.

| model_name | traditional | with_statcast | delta_ap | delta_roc_auc | delta_brier | delta_precision_at_20 | delta_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | M2 | M3 | -0.0332 | 0.0469 | -0.0008 | -0.05 | 0.0556 |
| Random Forest | M4 | M5 | 0.0348 | 0.0353 | -0.002 | -0.05 | -0.0556 |

- Logistic Regression (M2 → M3) AP decreased (change -0.0332). Precision@20 changed by -0.0500.
- Random Forest (M4 → M5) AP increased (change +0.0348). Precision@20 changed by -0.0500.

These comparisons come from one season with 36 breakouts. Small changes may be unstable across seasons, and a gain in AP can coexist with a decline at a specific cutoff or among the top 20.

## Baseball interpretation

Age and starting wRC+ jointly provide a useful reference. M1 had AP 0.4637 against the constant model's 0.1739. Adding the traditional bundle raised logistic regression AP (change +0.0203). The bundle combines plate discipline with playing time and other batting measures, so this experiment does not isolate the contribution of plate discipline. Statcast's incremental results are model-specific, as shown above.

M2 ranked 11 actual breakouts among its first 20 hitters (55%, compared with 17.39% prevalence). This is a historical shortlist within the qualifying cohort, and its hit rate may vary out of season.

## Limitations

Breakouts are a minority class, and there are only six observed feature seasons. Neither transition involving the shortened 2020 season qualifies under the 300-PA rule. Future playing time is required to define this retrospective cohort, limiting preseason applicability. Players can repeat across seasons. Regression to the mean and the use of starting wRC+ in the definition of improvement can contribute to the apparent signal from prior performance. A 20-point gain can still leave a hitter below league average (100 wRC+). The validation winner and calibration estimates may be unstable with a single validation season and sparse probability bins. These results describe predictive associations.

## Week 3 roadmap

Evaluate gradient boosting (including XGBoost), expanding-window backtesting, age-group performance, feature importance, false-positive and false-negative case studies, and calibration improvements.

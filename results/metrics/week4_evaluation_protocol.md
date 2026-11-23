# Frozen Week 4 evaluation protocol

## Models and training cutoff

M2 is the primary model because it had the highest mean historical fold Average
Precision (0.3682). M1 is the prespecified comparison because it had the highest
mean Precision@20 (0.45) and lowest mean Brier score (0.1239). Keep both roles
fixed through final reporting.

Both models use the existing Week 2 configuration. Each pipeline fits median
imputation and standardization before L2-regularized logistic regression.
M2 uses the traditional allowlist, and M1 uses age and wRC+. Use each
pipeline's raw positive-class probabilities. Training uses feature seasons
2017, 2018, 2021, 2022 and 2023, with outcomes complete by the end of 2024.
The freeze manifest records the exact predictor order, training fingerprint,
package versions and fitted pipeline hashes.

## One final evaluation

Verify the protocol and pipeline hashes in the freeze manifest before opening
the final labels. Use the frozen M2 and M1 pipelines to score the same qualifying
2024 feature-season rows once, in a single evaluation of the 2025 outcomes.
Save both models' predictions and metrics from that evaluation. Use the same
label and eligibility rules as in historical backtesting. Do not refit,
recalibrate, tune a threshold or change the primary model based on the holdout.
Reuse the saved outputs for subsequent tables and figures.

- **Primary endpoint.** M2 Average Precision, computed with scikit-learn's
  `average_precision_score`. Report M1 AP and the holdout breakout prevalence
  as prespecified references.
- **Secondary metrics.** Report ROC-AUC and Brier score for both models.
  Precision, recall, F1 and the confusion matrix at 0.50 are diagnostics.
- **Scouting application.** Rank both sets of raw scores and take the top 20
  separately for each model. Break probability ties by ascending player ID
  and then feature season. Report hits out of 20, Precision@20 and Recall@20.
  Use the top 20 as a shortlist for further scouting and analytics review.
  A score or shortlist position is insufficient for a definitive player prediction.

## Retrospective evaluation population

The cohort requires at least 300 PA in both 2024 and 2025. At the preseason
scouting decision, 2025 playing time is unknown. The reported metrics therefore
estimate performance conditional on reaching the following-season PA minimum.
The excluded players may differ in role, opportunity or performance, and their
absence can change prevalence and shortlist hit rates. A real preseason tool
would need a cohort defined from information available at scoring time and a
prespecified way to handle hitters with limited subsequent playing time.

No final scores or metrics were generated during the freeze.

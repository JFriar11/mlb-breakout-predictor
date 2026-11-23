"""Render the Week 3 report from the experiment outputs."""
from pandas import concat

from .evaluate_models import markdown_table


def write_report(path, folds, comparison, ages, importance, statcast, errors, calibration,
                 thresholds, selected, best_new, best_tree, blocker):
    winner = comparison.iloc[0]
    m2 = comparison.set_index("experiment_id").loc["M2"]
    shortlist = comparison.sort_values(["mean_precision_at_20", "experiment_id"],
                                       ascending=[False, True]).iloc[0]
    brier = comparison.sort_values(["mean_brier_score", "experiment_id"]).iloc[0]
    folds_table = folds.drop_duplicates("validation_season")[[
        "validation_season", "training_feature_seasons", "training_outcome_max",
        "training_observations", "validation_observations", "validation_breakouts",
        "validation_prevalence",
    ]]
    fold_scores = folds.pivot(index="experiment_id", columns="validation_season",
                              values="average_precision").reset_index()
    fold_scores.columns = fold_scores.columns.map(str)
    comparison_view = comparison[[
        "experiment_id", "model_name", "feature_set", "mean_average_precision",
        "std_average_precision", "mean_roc_auc", "mean_brier_score", "mean_precision_at_20",
        "mean_recall_at_20", "pooled_average_precision",
    ]].merge(fold_scores, on="experiment_id")
    comparison_view.columns = [
        "ID", "Model", "Features", "Mean AP", "SD AP", "Mean AUC", "Mean Brier",
        "Mean P@20", "Mean R@20", "Pooled AP", "2021 AP", "2022 AP", "2023 AP",
    ]
    fold_view = folds[["experiment_id", "validation_season", "average_precision", "roc_auc",
                       "brier_score", "precision_at_20", "recall_at_20"]]
    fold_view.columns = ["ID", "Season", "AP", "ROC-AUC", "Brier", "P@20", "R@20"]
    delta = statcast.groupby("model_name", as_index=False).agg(
        mean_delta_AP=("delta_average_precision", "mean"),
        AP_gain_folds=("delta_average_precision", lambda v: v.gt(0).sum()),
        mean_delta_P20=("delta_precision_at_20", "mean"),
        mean_delta_AUC=("delta_roc_auc", "mean"),
        mean_delta_Brier=("delta_brier_score", "mean"),
    )
    loss_seasons = statcast.groupby("validation_season").delta_average_precision.max()
    all_lost = ", ".join(map(str, loss_seasons.index[loss_seasons.lt(0)]))
    statcast_text = (f"Every tested family lost AP in {all_lost}." if all_lost else
                     "The season-by-season changes are in the Statcast comparison CSV.")
    coef = importance.loc[importance.experiment_id.eq("M2")].head()
    tree = importance.loc[importance.experiment_id.eq(best_tree)].head()
    tree_table = tree[["feature", "mean_importance", "fold_std", "mean_permutation_std"]]
    all_features = importance.loc[importance.experiment_id.eq(best_tree)]
    interest = ["wrc_plus", "age", "babip", "k_rate", "bb_rate", "chase_rate", "contact_rate",
                "avg_exit_velocity", "barrel_rate", "xwoba"]
    means = all_features.set_index("feature").mean_importance
    feature_text = ", ".join(
        f"{name} ({means[name]:+.4f})" if name in means else f"{name} (outside this feature set)"
        for name in interest
    )
    case_columns = ["player_name", "season", "wrc_plus", "next_season_wrc_plus",
                    "predicted_probability", "error_category"]
    cases = errors.loc[errors.experiment_id.eq(selected)]
    examples = [cases.loc[cases.error_category.eq(category)].sort_values(
        "predicted_probability", ascending=category == "false_negative", kind="stable"
    ).head(1) for category in ("false_positive", "false_negative", "true_positive")]
    examples_text = markdown_table(concat(examples)[case_columns])
    cutoffs = thresholds.loc[thresholds.experiment_id.eq(selected)].drop(columns="experiment_id")
    pooled_cal = calibration.loc[calibration.scope.eq("pooled_paired")].drop(
        columns=["scope", "calibration_observations"]
    )
    selected_cal = pooled_cal.loc[pooled_cal.experiment_id.eq(selected)].set_index("method")
    calibration_result = (
        f"For {selected}, paired pooled Brier changed from {selected_cal.loc['raw', 'brier_score']:.4f} "
        f"to {selected_cal.loc['sigmoid', 'brier_score']:.4f} after sigmoid calibration."
        if "sigmoid" in selected_cal.index else
        f"{selected} lacked sufficient chronological calibration data for a paired comparison."
    )
    zero = folds.loc[(folds.tp + folds.fp).eq(0), ["experiment_id", "validation_season"]]
    zero_text = ", ".join(f"{r.experiment_id} ({r.validation_season})" for r in zero.itertuples())
    season_ap = folds.loc[folds.experiment_id.eq(selected), "average_precision"]
    new_ap = comparison.set_index("experiment_id").loc[best_new, "mean_average_precision"]
    runner_up = comparison.iloc[1]
    calibration_seasons = calibration.loc[
        calibration.experiment_id.eq(selected) & calibration.scope.ne("pooled_paired")
    ].pivot(index="scope", columns="method", values="brier_score")
    paired_count = 0
    improved_count = 0
    if "sigmoid" in calibration_seasons:
        paired = calibration_seasons.dropna()
        paired_count = len(paired)
        improved_count = paired.sigmoid.lt(paired.raw).sum()
    age_text = " ".join(
        f"{experiment} exceeded the within-group prevalence reference by "
        + ", ".join(f"{r.average_precision - r.prevalence:+.3f} AP for ages {r.age_group}"
                    for r in group.itertuples()) + "."
        for experiment, group in ages.groupby("experiment_id")
    )
    new_ages = ages.loc[ages.experiment_id.eq(best_new)].set_index("age_group")
    age_auc = (
        f"{best_new} had a ROC-AUC of {new_ages.loc['≥30', 'roc_auc']:.2f} "
        f"in the oldest band and {new_ages.loc['25–29', 'roc_auc']:.2f} at ages 25–29."
    )
    status = (f"XGBoost unavailable. {blocker}\n\nInstall with "
              '`python -m pip install "xgboost>=2,<4"` and rerun.\n\n' if blocker else "")
    path.write_text(f"""# Week 3 modeling report

## Executive summary

{status}{selected} led on mean historical Average Precision ({winner.mean_average_precision:.4f}).
The best new model was {best_new} ({new_ap:.4f} mean AP). M2, the Week 2 winner
on 2023 validation AP, averaged {m2.mean_average_precision:.4f} in backtesting.
The mean gap between {selected} and {runner_up.experiment_id} was
{winner.mean_average_precision - runner_up.mean_average_precision:.4f}, compared with
{selected}'s AP range of {season_ap.min():.4f} to {season_ap.max():.4f} across seasons. {shortlist.experiment_id} led on mean
Precision@20 ({shortlist.mean_precision_at_20:.1%}) and {brier.experiment_id} achieved
the lowest mean Brier score ({brier.mean_brier_score:.4f}). The tables show which model
families gained from Statcast and how those gains varied by season.

## Backtesting results

{markdown_table(folds_table)}

Each prediction is made after the feature season ends. Training labels come from outcome
seasons completed by that date. The target remains a gain of at least 20 unrounded wRC+
points. Each experiment fits a fresh pipeline on identical training rows, using the
original class frequencies. HGB uses 150 iterations, 15 leaves, a 0.05 learning rate
and L2 of 1. XGBoost uses 200 depth-3 trees, a 0.05 learning rate and 0.8 row and
column subsampling. Early stopping was disabled. Exact parameters are in the run metadata.

{markdown_table(comparison_view)}

Mean and sample SD weight each fold equally. Pooled AP ranks all historical scores together,
so it can respond to shifts in score scale across seasons. Precision@20 and Recall@20 in
the comparison refer to separate top-20 lists each season. The full fold CSV also includes
ROC-AUC, Brier, precision, recall, F1, Precision@10 and confusion counts at 0.50.
Zero-positive fold/model combinations at 0.50 were {zero_text or 'absent'}.

<details>
<summary>Fold-level ranking and probability metrics</summary>

{markdown_table(fold_view)}

</details>

## Baseball interpretation

### Incremental Statcast value

{markdown_table(delta)}

AP gains count strict improvements out of three folds. {statcast_text}
Mean AP and mean ROC-AUC capture different parts of ranking performance. Correlated traditional and Statcast
variables can carry overlapping signal, including observed and expected wOBA.

### Feature importance

{markdown_table(coef[['feature', 'mean_importance', 'fold_std']])}

M2 coefficients represent approximate shifts in log odds per training-set standard
deviation, conditional on other predictors. Correlated measures can make signs
and magnitudes unstable. The tree results below measure average AP loss after
permuting one feature on historical validation rows. Variation across folds and
average repeat SD describe sensitivity to season and permutation, respectively.

{markdown_table(tree_table)}

Selected variables in {best_tree}, with mean permutation AP drop, include {feature_text}.
The starting wRC+ score is mathematically coupled to the improvement target.
Regression to the mean can contribute to this association. The same 20-point
gain also requires a higher ending wRC+ for a hitter with a higher starting wRC+.
Permutation importances describe model reliance and can be diluted among correlated features.

### Hitter age

{markdown_table(ages)}

{age_text} A constant score within each age band has AP equal to that band's
prevalence. The within-group AP gains show ranking signal beyond the age-group
base rate. {age_auc} Differences in the amount of gain are descriptive. Week 1's full-data
prevalences were 23.53%, 15.16%, and 12.04% from youngest to oldest.

## Error analysis

{examples_text}

The [case studies](week3_case_studies.md) document at least three examples per category
and identify near misses and overlooked breakouts in the annual top-20 lists.
Feature-season context distinguishes hitters with favorable expected production
from those who started with weaker contact or more strikeouts. The available observations
support descriptions of starting profiles and realized wRC+ changes.

## Calibration and thresholds

{markdown_table(pooled_cal)}

Each sigmoid fits on inner out-of-fold log odds from earlier seasons. The learned
mapping is applied to scores from a fresh base model fit on the full outer training
window. Raw and sigmoid Brier scores use the same outer rows. Pooled calibration intercept
and slope regress outcomes on score log odds for diagnostics only. Ideal values are
0 and 1. Reliability counts are saved alongside the calibration comparison.

{calibration_result} Brier improved in {improved_count} of {paired_count} paired
outer seasons for {selected}. The earliest sigmoid uses just one inner validation
season. Later folds pool additional inner seasons whose prevalence and base-model
training sizes can differ.

{markdown_table(cutoffs)}

At 0.50, {selected} flagged {int(cutoffs.iloc[-1].predicted_breakouts)} of the
{int(folds_table.validation_observations.sum())} qualifying hitters, with
{cutoffs.iloc[-1].recall:.1%} recall. At 0.20 it flagged
{int(cutoffs.loc[cutoffs.threshold.eq(.20), 'predicted_breakouts'].iloc[0])}.
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

Use {selected} provisionally for ranking, based on mean fold AP with exact ties
resolved by experiment ID. Its mean shortlist hit rate was {winner.mean_precision_at_20:.1%}.
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
""", encoding="utf-8")

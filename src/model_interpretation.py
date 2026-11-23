"""Fold-based coefficient and permutation summaries, plus pooled age metrics."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .advanced_models import EXPERIMENTS, features
from .evaluate_models import rank_predictions
from .train_models import SEED

AGE_GROUPS = ("≤24", "25–29", "≥30")


def age_performance(predictions, experiments):
    rows = []
    for experiment in experiments:
        frame = predictions.loc[predictions.experiment_id.eq(experiment)].copy()
        frame["age_group"] = pd.cut(frame.age, [-np.inf, 24, 29, np.inf], labels=AGE_GROUPS)
        for age_group, group in frame.groupby("age_group", observed=True):
            y, p = group.breakout, group.predicted_probability
            rows.append({
                "experiment_id": experiment, "age_group": age_group,
                "observations": len(group), "breakouts": int(y.sum()), "prevalence": y.mean(),
                "average_precision": average_precision_score(y, p) if y.sum() else np.nan,
                "roc_auc": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
                "brier_score": brier_score_loss(y, p),
                "precision_at_10": rank_predictions(group).head(10).breakout.mean()
                if len(group) >= 10 else np.nan,
            })
    return pd.DataFrame(rows)


def importance(folds, fits, tree_experiments, repeats=10):
    rows = []
    for fold in folds:
        logistic = fits["M2", fold.season]
        rows.extend({
            "experiment_id": "M2", "method": "standardized_coefficient",
            "validation_season": fold.season, "feature": name, "value": coefficient,
            "permutation_std": np.nan,
        } for name, coefficient in zip(logistic.feature_names_in_,
                                       logistic.named_steps["model"].coef_[0], strict=True))
        for experiment in tree_experiments:
            model = fits[experiment, fold.season]
            validation = features(fold.validation, experiment)
            result = permutation_importance(
                model, validation, fold.validation.breakout, scoring="average_precision",
                n_repeats=repeats, random_state=SEED,
            )
            rows.extend({
                "experiment_id": experiment, "method": "permutation_ap_drop",
                "validation_season": fold.season, "feature": name, "value": mean,
                "permutation_std": spread,
            } for name, mean, spread in zip(validation.columns, result.importances_mean,
                                           result.importances_std, strict=True))
    return pd.DataFrame(rows)


def summarize_importance(detail):
    summary = detail.groupby(["experiment_id", "method", "feature"], as_index=False).agg(
        mean_importance=("value", "mean"), fold_std=("value", "std"),
        mean_permutation_std=("permutation_std", "mean"), fold_count=("value", "size"),
    )
    summary["absolute_importance"] = summary.mean_importance.abs()
    summary["rank_value"] = np.where(summary.method.eq("standardized_coefficient"),
                                     summary.absolute_importance, summary.mean_importance)
    return summary.sort_values(["experiment_id", "rank_value", "feature"],
                               ascending=[True, False, True])


def statcast_differences(fold_metrics, pairs):
    rows = []
    measures = ["average_precision", "roc_auc", "brier_score", "precision_at_20"]
    for traditional, statcast in pairs:
        base = fold_metrics.loc[fold_metrics.experiment_id.eq(traditional)].set_index("validation_season")
        extra = fold_metrics.loc[fold_metrics.experiment_id.eq(statcast)].set_index("validation_season")
        if extra.empty:
            continue
        delta = extra[measures].subtract(base[measures]).add_prefix("delta_").reset_index()
        rows.append(delta.assign(model_name=EXPERIMENTS[traditional][0],
                                 traditional=traditional, with_statcast=statcast))
    return pd.concat(rows, ignore_index=True)

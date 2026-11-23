"""Nested chronological sigmoid calibration and retrospective threshold analysis."""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from .advanced_models import features, make_experiment
from .evaluate_models import evaluate_predictions

THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
BINS = (0, 0.10, 0.20, 0.35, 0.50, 1)


def logits(probabilities):
    return logit(np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)).reshape(-1, 1)


def fit_calibrator(history, experiment, evaluation_season):
    """Fit on inner OOF log odds using labels available before outer scoring."""
    if (history.empty or history.season.max() >= evaluation_season
            or history.next_season.max() > evaluation_season):
        raise ValueError("Calibration requires completed historical outcomes")
    parts = []
    for season in sorted(history.season.unique())[1:]:
        train = history.loc[history.season.lt(season) & history.next_season.le(season)]
        validation = history.loc[history.season.eq(season)]
        if train.breakout.nunique() < 2:
            continue
        model = make_experiment(experiment).fit(features(train, experiment), train.breakout)
        scores = model.predict_proba(features(validation, experiment))[:, 1]
        parts.append(pd.DataFrame({"p": scores, "y": validation.breakout.to_numpy()}))
    if not parts:
        return None, 0
    inner = pd.concat(parts)
    if inner.y.nunique() < 2:
        return None, len(inner)
    # Regularization stabilizes Platt fits on small historical calibration sets.
    return LogisticRegression().fit(logits(inner.p), inner.y), len(inner)


def reliability(frame):
    bins = pd.cut(frame.predicted_probability, BINS, include_lowest=True)
    table = frame.groupby(bins, observed=False).agg(
        count=("breakout", "size"), mean_probability=("predicted_probability", "mean"),
        observed_frequency=("breakout", "mean"),
    ).reset_index(names="bin")
    table["bin"] = table["bin"].astype(str)
    return table


def calibration_metrics(frame):
    scores, labels = frame.predicted_probability, frame.breakout
    x = logits(scores)
    intercept, slope = np.nan, np.nan
    if labels.nunique() == 2 and np.ptp(x) > 0:
        # A separating or quasi-separating cutoff implies no finite unpenalized MLE.
        design = np.column_stack((np.ones(len(x)), x))
        signed = design * (2 * labels.to_numpy()[:, None] - 1)
        separation = linprog(-signed.sum(axis=0), A_ub=-signed, b_ub=np.zeros(len(x)),
                             bounds=[(-1, 1)] * 2, method="highs")
        if separation.success and separation.fun >= -1e-9:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
                model.fit(x, labels)
            if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
                intercept, slope = model.intercept_[0], model.coef_[0, 0]
    return {
        "observations": len(frame), "prevalence": labels.mean(),
        "mean_probability": scores.mean(), "brier_score": brier_score_loss(labels, scores),
        "intercept": intercept, "slope": slope,
    }


def analyze_calibration(folds, predictions, experiments):
    results, bins, scored = [], [], []
    for experiment in experiments:
        paired = []
        for fold in folds:
            raw = predictions.loc[predictions.experiment_id.eq(experiment)
                                  & predictions.season.eq(fold.season)].copy()
            calibrator, count = fit_calibrator(fold.train, experiment, fold.season)
            results.append({
                "experiment_id": experiment, "scope": str(fold.season), "method": "raw",
                "calibration_observations": count, **calibration_metrics(raw),
            })
            if calibrator is None:
                continue
            calibrated = raw.assign(predicted_probability=calibrator.predict_proba(
                logits(raw.predicted_probability)
            )[:, 1])
            results.append({
                "experiment_id": experiment, "scope": str(fold.season), "method": "sigmoid",
                "calibration_observations": count, **calibration_metrics(calibrated),
            })
            paired.extend((raw.assign(method="raw"), calibrated.assign(method="sigmoid")))
        if not paired:
            continue
        for method, frame in pd.concat(paired).groupby("method"):
            results.append({
                "experiment_id": experiment, "scope": "pooled_paired", "method": method,
                **calibration_metrics(frame),
            })
            bins.append(reliability(frame).assign(experiment_id=experiment, method=method))
            scored.append(frame[["player_id", "season", "breakout", "predicted_probability",
                                 "experiment_id", "method"]])
    bin_columns = ["bin", "count", "mean_probability", "observed_frequency",
                   "experiment_id", "method"]
    score_columns = ["player_id", "season", "breakout", "predicted_probability",
                     "experiment_id", "method"]
    return (
        pd.DataFrame(results),
        pd.concat(bins, ignore_index=True) if bins else pd.DataFrame(columns=bin_columns),
        pd.concat(scored, ignore_index=True) if scored else pd.DataFrame(columns=score_columns),
    )


def threshold_analysis(predictions):
    rows = []
    for experiment, group in predictions.groupby("experiment_id"):
        for threshold in THRESHOLDS:
            metrics = evaluate_predictions(group, threshold)
            rows.append({
                "experiment_id": experiment, "threshold": threshold,
                "predicted_breakouts": metrics["tp"] + metrics["fp"],
                **{key: metrics[key] for key in ("precision", "recall", "f1", "fp", "fn")},
            })
    return pd.DataFrame(rows)

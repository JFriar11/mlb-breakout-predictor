"""Evaluate validation predictions or explicitly score the frozen finalist."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from .feature_sets import FEATURE_SETS
from .model_visualizations import plot_validation
from .split_data import DATASET, KEYS, split_data
from .train_models import (
    EXPERIMENTS, MANIFEST, METRIC_DIR, MODEL_DIR, PREDICTION_DIR, THRESHOLD,
    prediction_frame, probabilities, sha256, write_json,
)
from .validate_data import ROOT

FIGURE_DIR = ROOT / "results/figures/week2"
SELECTION = "week2_selection.json"
STATCAST_METRICS = ["ap", "roc_auc", "brier", "precision_at_20", "recall"]


def rank_predictions(frame):
    return frame.sort_values(
        ["predicted_probability", "player_id", "season"], ascending=[False, True, True]
    ).reset_index(drop=True)


def evaluate_predictions(frame, threshold=THRESHOLD):
    y = frame.breakout
    if set(y) != {0, 1}:
        raise ValueError("Evaluation requires both binary classes for ROC-AUC and Average Precision")
    scores = frame.predicted_probability
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")
    predicted = scores.ge(threshold)
    top = rank_predictions(frame)
    hits = int(top.head(20).breakout.sum())
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "observations": len(y), "prevalence": float(y.mean()), "threshold": threshold,
        "ap": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "accuracy": accuracy_score(y, predicted),
        "brier": brier_score_loss(y, scores),
        "precision_at_10": top.head(10).breakout.mean(),
        "precision_at_20": top.head(20).breakout.mean(),
        "recall_at_20": hits / int(y.sum()), "breakouts_in_top_20": hits,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_experiment(metrics):
    """Choose the highest validation AP, breaking exact ties by experiment ID."""
    candidates = metrics.loc[list(EXPERIMENTS)].sort_index()
    return candidates.ap.idxmax()


def verify_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact changed since it was recorded: {path}")


def load_pipeline(model_dir, experiment, metadata):
    path = model_dir / f"{experiment}.joblib"
    verify_hash(path, metadata["experiments"][experiment]["pipeline_sha256"])
    pipeline = joblib.load(path)
    if tuple(pipeline.feature_names_in_) != FEATURE_SETS[EXPERIMENTS[experiment][1]]:
        raise ValueError(f"Unexpected fitted predictors for {experiment}")
    return pipeline


def validation_predictions(model_dir, prediction_dir, metadata, threshold):
    frames = {}
    for experiment in (*EXPERIMENTS, "constant"):
        path = prediction_dir / f"validation_{experiment}.csv"
        if experiment == "constant":
            expected_hash = metadata["constant_predictions_sha256"]
        else:
            load_pipeline(model_dir, experiment, metadata)
            expected_hash = metadata["experiments"][experiment]["validation_predictions_sha256"]
        verify_hash(path, expected_hash)
        frame = pd.read_csv(path, float_precision="round_trip")
        if (not frame.split.eq("validation").all() or not frame.season.eq(2023).all()
                or frame.duplicated(KEYS).any()
                or not frame.predicted_class.eq(frame.predicted_probability.ge(threshold)).all()):
            raise ValueError(f"Invalid validation predictions for {experiment}")
        if frames:
            columns = KEYS + ["breakout"]
            if not frame[columns].equals(frames["M1"][columns]):
                raise ValueError("Experiments must use identical validation observations")
        frames[experiment] = frame
    return frames


def statcast_comparison(metrics):
    rows = []
    for family, traditional, statcast in [
        ("Logistic Regression", "M2", "M3"), ("Random Forest", "M4", "M5"),
    ]:
        before, after = metrics.loc[traditional], metrics.loc[statcast]
        rows.append({
            "model_name": family, "traditional": traditional, "with_statcast": statcast,
            **{f"delta_{key}": after[key] - before[key] for key in STATCAST_METRICS},
        })
    return pd.DataFrame(rows)


def markdown_table(frame):
    rows = frame.round(4).astype(str).itertuples(index=False, name=None)
    lines = [tuple(frame.columns), ("---",) * len(frame.columns), *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in lines)


def write_summary(path, metadata, metrics, statcast, selected):
    split = pd.DataFrame(metadata["splits"])
    split["prevalence"] = split.prevalence.map("{:.2%}".format)
    cols = ["model_name", "feature_set", "ap", "roc_auc", "precision", "recall", "f1", "precision_at_20"]
    view = metrics[cols].rename(columns=dict(zip(cols, [
        "Model", "Features", "AP", "ROC-AUC", "Precision", "Recall", "F1", "Precision@20",
    ]))).rename_axis("Experiment").reset_index()
    best, baseline, constant = metrics.loc[selected], metrics.loc["M1"], metrics.loc["constant"]
    logistic_delta = metrics.loc["M2", "ap"] - baseline.ap
    traditional_gain = "raised" if logistic_delta > 0 else "lowered" if logistic_delta < 0 else "left unchanged"
    statcast_text = []
    for row in statcast.itertuples():
        direction = "increased" if row.delta_ap > 0 else "decreased" if row.delta_ap < 0 else "was unchanged"
        statcast_text.append(
            f"- {row.model_name} ({row.traditional} → {row.with_statcast}) AP {direction} "
            f"(change {row.delta_ap:+.4f}). Precision@20 changed by {row.delta_precision_at_20:+.4f}."
        )
    zero_precision = ", ".join(metrics.index[(metrics.tp + metrics.fp).eq(0)]) or "none"
    groups = "\n".join(f"- **{name}** includes " + ", ".join(f"`{c}`" for c in cols)
                       for name, cols in metadata["feature_sets"].items())
    path.write_text(f"""# Week 2 validation report

## Dataset and methodology

The existing Week 1 binary label marks a following-season improvement of at least 20 points in unrounded FanGraphs wRC+. Each row represents a hitter with at least 300 PA in both seasons. There are {split.observations.sum():,} qualifying player-seasons.

{markdown_table(split)}

{groups}

The five pipelines fit on the training partition only. Each uses median imputation. Logistic regression additionally standardizes predictors and uses default regularization, a 2,000-iteration limit and seed 42. Random forests use 300 trees, depth at most 6, at least 5 observations per leaf and seed 42. The experiments use the original class frequencies and exclude identifiers and outcome fields via the Week 1 allowlist. Split membership is fixed by season and unique by player-season. The 2025 holdout contributed only split audit counts and row identifiers, with no model scoring or selection.

## Validation results

{markdown_table(view)}

Average Precision is computed with scikit-learn's `average_precision_score`. The constant score is the training prevalence ({metadata['training_prevalence']:.2%}); its AP equals the validation prevalence ({constant.prevalence:.2%}). Its top-k hit rates reflect the deterministic player-ID tie order. The metrics CSV includes Brier score, accuracy, confusion counts, top-k metrics and deltas versus M1.

Classification uses the prespecified 0.50 cutoff. Undefined precision when a model calls zero positives is recorded as zero ({zero_precision}). Recall has a positive denominator in this validation set. The top 20 are ranked by probability, then player ID and season. The leaderboard's correctness column uses the 0.50 class prediction, while Precision@20 counts hits if all 20 ranked hitters are treated as breakout candidates.

**{selected}** was selected on validation AP ({best.ap:.4f}), with exact ties resolved by experiment ID. The fitted training-only pipeline and 0.50 threshold are frozen for optional final evaluation.

## Incremental value of Statcast

Deltas are Statcast minus traditional on the same validation observations. Lower Brier scores are better.

{markdown_table(statcast)}

{chr(10).join(statcast_text)}

These comparisons come from one season with {int(best.prevalence * best.observations + .5)} breakouts. Small changes may be unstable across seasons, and a gain in AP can coexist with a decline at a specific cutoff or among the top 20.

## Baseball interpretation

Age and starting wRC+ jointly provide a useful reference. M1 had AP {baseline.ap:.4f} against the constant model's {constant.ap:.4f}. Adding the traditional bundle {traditional_gain} logistic regression AP (change {logistic_delta:+.4f}). The bundle combines plate discipline with playing time and other batting measures, so this experiment does not isolate the contribution of plate discipline. Statcast's incremental results are model-specific, as shown above.

{selected} ranked {int(best.breakouts_in_top_20)} actual breakouts among its first 20 hitters ({best.precision_at_20:.0%}, compared with {best.prevalence:.2%} prevalence). This is a historical shortlist within the qualifying cohort, and its hit rate may vary out of season.

## Limitations

Breakouts are a minority class, and there are only six observed feature seasons. Neither transition involving the shortened 2020 season qualifies under the 300-PA rule. Future playing time is required to define this retrospective cohort, limiting preseason applicability. Players can repeat across seasons. Regression to the mean and the use of starting wRC+ in the definition of improvement can contribute to the apparent signal from prior performance. A 20-point gain can still leave a hitter below league average (100 wRC+). The validation winner and calibration estimates may be unstable with a single validation season and sparse probability bins. These results describe predictive associations.

## Week 3 roadmap

Evaluate gradient boosting (including XGBoost), expanding-window backtesting, age-group performance, feature importance, false-positive and false-negative case studies, and calibration improvements.
""")


def evaluate_validation(model_dir, prediction_dir, metric_dir, figure_dir):
    manifest = model_dir / MANIFEST
    metadata = json.loads(manifest.read_text())
    threshold = metadata["threshold"]
    frames = validation_predictions(model_dir, prediction_dir, metadata, threshold)
    rows = []
    for experiment, frame in frames.items():
        rows.append({
            "experiment": experiment,
            "model_name": frame.model_name.iloc[0],
            "feature_set": frame.feature_set.iloc[0],
            **evaluate_predictions(frame, threshold),
        })
    metrics = pd.DataFrame(rows).set_index("experiment")
    for key in STATCAST_METRICS:
        metrics[f"delta_{key}_vs_m1"] = metrics[key] - metrics.loc["M1", key]
    selected = select_experiment(metrics)
    statcast = statcast_comparison(metrics)
    metrics.to_csv(metric_dir / "week2_model_comparison.csv")
    statcast.to_csv(metric_dir / "week2_statcast_comparison.csv", index=False)
    leaders = rank_predictions(frames[selected]).head(20).copy()
    leaders.insert(0, "rank", range(1, len(leaders) + 1))
    leaders.insert(1, "experiment", selected)
    leaders["correct_at_0_50"] = leaders.predicted_class.eq(leaders.breakout)
    leaders.to_csv(prediction_dir / "validation_top20.csv", index=False)
    plot_validation(frames, metrics, selected, figure_dir)
    write_summary(metric_dir / "week2_summary.md", metadata, metrics, statcast, selected)
    write_json(metric_dir / SELECTION, {
        "manifest_sha256": sha256(manifest),
        "experiment": selected,
        "validation_ap": metrics.loc[selected, "ap"],
        "selection_rule": "Highest validation Average Precision; ties by experiment ID",
        "threshold": threshold,
        "threshold_rule": "Prespecified 0.50 cutoff",
    })
    return metrics


def evaluate_final_test(dataset, model_dir, prediction_dir, metric_dir):
    selection_path = metric_dir / SELECTION
    if not selection_path.exists():
        raise ValueError("Run validation evaluation to freeze model selection before final testing")
    selection = json.loads(selection_path.read_text())
    manifest = model_dir / MANIFEST
    verify_hash(manifest, selection["manifest_sha256"])
    metadata = json.loads(manifest.read_text())
    if selection["threshold"] != metadata["threshold"]:
        raise ValueError("Frozen threshold does not match training metadata")
    verify_hash(dataset, metadata["dataset_sha256"])
    experiment = selection["experiment"]
    model = load_pipeline(model_dir, experiment, metadata)
    test = split_data(pd.read_csv(dataset, float_precision="round_trip"))["test"]
    name, features = EXPERIMENTS[experiment]
    frame = prediction_frame(test, probabilities(model, test, experiment),
                             name, features, "test", selection["threshold"])
    metrics = pd.DataFrame([{
        "experiment": experiment, "model_name": name, "feature_set": features,
        **evaluate_predictions(frame, selection["threshold"]),
    }])
    frame.to_csv(prediction_dir / f"test_{experiment}.csv", index=False)
    metrics.to_csv(metric_dir / "week2_final_test.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-test", action="store_true", help="Score the frozen model on 2025 outcomes")
    args = parser.parse_args()
    if args.final_test:
        print(evaluate_final_test(DATASET, MODEL_DIR, PREDICTION_DIR, METRIC_DIR).to_string(index=False))
    else:
        metrics = evaluate_validation(MODEL_DIR, PREDICTION_DIR, METRIC_DIR, FIGURE_DIR)
        print(metrics[["ap", "roc_auc", "f1", "precision_at_20"]].to_string())


if __name__ == "__main__":
    main()

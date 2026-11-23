"""Run all Week 3 analyses on historical feature seasons only."""
import hashlib
import math
import platform
from importlib.metadata import version

from threadpoolctl import threadpool_limits

from .advanced_models import EXPERIMENTS, STATCAST_PAIRS, available_experiments, make_experiment
from .backtesting import folds_for, load_history, run_backtests, summarize
from .calibration_analysis import analyze_calibration, threshold_analysis
from .error_analysis import analyze_errors, case_studies
from .feature_sets import FEATURE_SETS
from .model_interpretation import age_performance, importance, statcast_differences, summarize_importance
from .split_data import DATASET
from .train_models import write_json
from .validate_data import ROOT
from .week3_visualizations import plot_results

COMPARISON_COLUMNS = [
    "experiment_id", "model_name", "feature_set", "fold_count",
    "mean_average_precision", "std_average_precision", "mean_roc_auc", "mean_brier_score",
    "mean_precision_at_20", "mean_recall_at_20", "pooled_average_precision",
]


def run(dataset=DATASET, destination=ROOT / "results"):
    from .week3_report import write_report

    metrics = destination / "metrics"
    predictions_dir = destination / "predictions/week3"
    figures = destination / "figures/week3"
    for directory in (metrics, predictions_dir, figures):
        directory.mkdir(parents=True, exist_ok=True)

    history = load_history(dataset)
    folds = list(folds_for(history))
    experiments, blocker = available_experiments()
    if blocker:
        print(f"XGBoost unavailable: {blocker}")
    with threadpool_limits(limits=2):
        detail, predictions, fits = run_backtests(folds, experiments)
        summary = summarize(detail, predictions)
        ordered = summary.experiment_id.tolist()
        selected = ordered[0]
        best_new = next(model for model in ordered if int(model[1:]) >= 6)
        best_tree = next(model for model in ordered if int(model[1:]) >= 4)
        trees = [best_tree]
        trees.extend(model for pair in STATCAST_PAIRS if best_tree in pair
                     for model in pair if model != best_tree)
        importance_detail = importance(folds, fits, trees)
        calibration, bins, calibration_scores = analyze_calibration(
            folds, predictions, tuple(dict.fromkeys(("M1", "M2", selected, best_new)))
        )

    comparison = summary[COMPARISON_COLUMNS]
    ages = age_performance(predictions, ("M1", "M2", best_new))
    importance_summary = summarize_importance(importance_detail)
    errors = analyze_errors(predictions, history)
    thresholds = threshold_analysis(predictions)
    statcast = statcast_differences(detail, STATCAST_PAIRS)
    cases = case_studies(errors, selected)
    for name, frame in {
        "backtesting_results": detail, "backtesting_summary": summary, "model_comparison": comparison,
        "age_group_performance": ages, "feature_importance": importance_summary,
        "feature_importance_by_fold": importance_detail, "error_analysis": errors,
        "threshold_analysis": thresholds, "calibration_comparison": calibration,
        "calibration_bins": bins, "statcast_comparison": statcast,
    }.items():
        frame.to_csv(metrics / f"week3_{name}.csv", index=False)
    predictions.to_csv(predictions_dir / "backtesting_predictions.csv", index=False)
    calibration_scores.to_csv(predictions_dir / "calibration_predictions.csv", index=False)
    (metrics / "week3_case_studies.md").write_text(cases, encoding="utf-8")
    plot_results(figures, detail, summary, statcast, ages, importance_summary,
                 trees, bins, calibration_scores, thresholds, selected)
    write_report(metrics / "week3_summary.md", detail, comparison, ages, importance_summary,
                 statcast, errors, calibration, thresholds, selected, best_new, best_tree, blocker)
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
    if not blocker:
        packages.append("xgboost")
    write_json(metrics / "week3_run_metadata.json", {
        "python": platform.python_version(), "packages": {p: version(p) for p in packages},
        "historical_csv_sha256": hashlib.sha256(history.to_csv(index=False).encode()).hexdigest(),
        "historical_row_count": len(history), "selection_rule": "Mean fold AP; ties by experiment ID",
        "selected": selected, "best_new": best_new, "best_tree": best_tree,
        "feature_sets": FEATURE_SETS, "xgboost_blocker": blocker,
        "experiments": {experiment: {
            "model_name": EXPERIMENTS[experiment][0], "feature_set": EXPERIMENTS[experiment][1],
            "steps": {
                name: {key: ("NaN" if isinstance(value, float) and math.isnan(value) else value)
                       for key, value in estimator.get_params().items()}
                for name, estimator in make_experiment(experiment).steps
            },
        } for experiment in experiments},
    })
    return comparison


if __name__ == "__main__":
    print(run().round(4).to_string(index=False))

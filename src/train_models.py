"""Fit Week 2 pipelines and save retrospective validation predictions."""
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_sets import FEATURE_SETS, predictors
from .split_data import DATASET, split_audit, split_data
from .validate_data import ROOT

MODEL_DIR = ROOT / "models/week2"
PREDICTION_DIR = ROOT / "results/predictions/week2"
METRIC_DIR = ROOT / "results/metrics"
MANIFEST = "experiment_metadata.json"
SEED = 42
THRESHOLD = 0.50
EXPERIMENTS = {
    "M1": ("Logistic Regression", "baseline"),
    "M2": ("Logistic Regression", "traditional"),
    "M3": ("Logistic Regression", "traditional_statcast"),
    "M4": ("Random Forest", "traditional"),
    "M5": ("Random Forest", "traditional_statcast"),
}
PREDICTION_METADATA = [
    "player_id", "player_name", "season", "age", "wrc_plus",
    "next_season_wrc_plus", "wrc_plus_change", "breakout",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def make_pipeline(model_name):
    imputer = ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    if model_name == "Logistic Regression":
        return Pipeline([
            imputer,
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ])
    if model_name == "Random Forest":
        return Pipeline([
            imputer,
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=5,
                random_state=SEED, n_jobs=-1,
            )),
        ])
    raise ValueError(f"Unknown model: {model_name}")


def fit_experiment(train, experiment):
    model_name, feature_set = EXPERIMENTS[experiment]
    return make_pipeline(model_name).fit(
        predictors(train, FEATURE_SETS[feature_set]), train.breakout
    )


def probabilities(model, frame, experiment):
    _, feature_set = EXPERIMENTS[experiment]
    return model.predict_proba(predictors(frame, FEATURE_SETS[feature_set]))[:, 1]


def prediction_frame(frame, scores, model_name, feature_set, split, threshold=THRESHOLD):
    output = frame[PREDICTION_METADATA].copy()
    output["predicted_probability"] = scores
    output["predicted_class"] = output.predicted_probability.ge(threshold).astype(int)
    return output.assign(model_name=model_name, feature_set=feature_set, split=split)


def train_models(dataset, model_dir, prediction_dir, metric_dir):
    for directory in (model_dir, prediction_dir, metric_dir):
        directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(dataset, float_precision="round_trip")
    for columns in FEATURE_SETS.values():
        predictors(data, columns)
    partitions = split_data(data)
    summary, memberships = split_audit(partitions)
    summary.to_csv(metric_dir / "week2_split_summary.csv", index=False)
    memberships.to_csv(metric_dir / "week2_split_rows.csv", index=False)
    train, validation = partitions["train"], partitions["validation"]
    prevalence = float(train.breakout.mean())
    metadata = {
        "dataset_sha256": sha256(dataset),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib")},
        "feature_sets": FEATURE_SETS,
        "threshold": THRESHOLD,
        "training_prevalence": prevalence,
        "splits": summary.to_dict(orient="records"),
        "experiments": {},
    }
    for experiment, (name, features) in EXPERIMENTS.items():
        pipeline = fit_experiment(train, experiment)
        model_path = model_dir / f"{experiment}.joblib"
        prediction_path = prediction_dir / f"validation_{experiment}.csv"
        joblib.dump(pipeline, model_path, compress=3)
        prediction_frame(validation, probabilities(pipeline, validation, experiment),
                         name, features, "validation").to_csv(prediction_path, index=False)
        metadata["experiments"][experiment] = {
            "model_name": name,
            "feature_set": features,
            "steps": {step: estimator.get_params() for step, estimator in pipeline.steps},
            "pipeline_sha256": sha256(model_path),
            "validation_predictions_sha256": sha256(prediction_path),
        }
    baseline_path = prediction_dir / "validation_constant.csv"
    prediction_frame(validation, prevalence, "Constant prevalence", "none", "validation").to_csv(
        baseline_path, index=False
    )
    metadata["constant_predictions_sha256"] = sha256(baseline_path)
    write_json(model_dir / MANIFEST, metadata)
    return summary


def main():
    print(train_models(DATASET, MODEL_DIR, PREDICTION_DIR, METRIC_DIR).to_string(index=False))


if __name__ == "__main__":
    main()

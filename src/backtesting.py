"""Expanding-window evaluation with outcome availability checked explicitly."""
import csv
from io import StringIO
from dataclasses import dataclass

import pandas as pd

from .advanced_models import EXPERIMENTS, features, make_experiment
from .evaluate_models import evaluate_predictions
from .split_data import KEYS
from .train_models import prediction_frame
from .validate_data import unique_keys, validate_dataset

HISTORY = (2017, 2018, 2021, 2022, 2023)
VALIDATION_SEASONS = HISTORY[2:]
METRICS = {
    "ap": "average_precision", "roc_auc": "roc_auc", "brier": "brier_score",
    **{name: name for name in (
        "precision", "recall", "f1", "precision_at_10", "precision_at_20",
        "recall_at_20", "tn", "fp", "fn", "tp",
    )},
}


@dataclass
class Fold:
    train: pd.DataFrame
    validation: pd.DataFrame
    season: int

    def metadata(self):
        return {
            "validation_season": self.season,
            "validation_outcome_season": int(self.validation.next_season.max()),
            "training_feature_seasons": " ".join(map(str, sorted(self.train.season.unique()))),
            "training_outcome_max": int(self.train.next_season.max()),
            "training_observations": len(self.train),
            "training_prevalence": self.train.breakout.mean(),
            "validation_observations": len(self.validation),
            "validation_breakouts": int(self.validation.breakout.sum()),
            "validation_prevalence": self.validation.breakout.mean(),
        }


def load_history(path):
    # Filter on the feature season before passing any records to pandas.
    buffer = StringIO()
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = next(reader)
        season_index = header.index("season")
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(row for row in reader if int(row[season_index]) in HISTORY)
    buffer.seek(0)
    data = pd.read_csv(buffer, float_precision="round_trip")
    validate_dataset(data)
    return data


def folds_for(data):
    if set(data.season) != set(HISTORY):
        raise ValueError("Expected only the five qualifying historical feature seasons")
    unique_keys(data, KEYS)
    if not data.next_season.eq(data.season + 1).all():
        raise ValueError("Labels must be from the immediately following season")
    data = data.sort_values(["season", "player_id"])
    for year in VALIDATION_SEASONS:
        train = data.loc[data.season.lt(year)].copy()
        validation = data.loc[data.season.eq(year)].copy()
        if train.next_season.max() > year:
            raise ValueError("Training labels are unavailable at prediction time")
        if set(train.breakout) != {0, 1} or set(validation.breakout) != {0, 1}:
            raise ValueError("Both partitions require both binary classes")
        yield Fold(train, validation, year)


def run_backtests(folds, experiments):
    scores, rows, fits = [], [], {}
    for fold in folds:
        for experiment in experiments:
            name, feature_set = EXPERIMENTS[experiment]
            model = make_experiment(experiment).fit(
                features(fold.train, experiment), fold.train.breakout
            )
            fits[experiment, fold.season] = model
            probability = model.predict_proba(features(fold.validation, experiment))[:, 1]
            scored = prediction_frame(fold.validation, probability, name, feature_set, "backtest")
            scores.append(scored.assign(experiment_id=experiment, validation_season=fold.season))
            metrics = evaluate_predictions(scored)
            rows.append({
                "experiment_id": experiment, "model_name": name, "feature_set": feature_set,
                **fold.metadata(), **{label: metrics[key] for key, label in METRICS.items()},
                "threshold": metrics["threshold"],
            })
    return pd.DataFrame(rows), pd.concat(scores, ignore_index=True), fits


def summarize(fold_metrics, predictions):
    rows = []
    for experiment, group in fold_metrics.groupby("experiment_id"):
        scored = predictions.loc[predictions.experiment_id.eq(experiment)]
        pooled = evaluate_predictions(scored)
        rows.append({
            "experiment_id": experiment, "model_name": group.model_name.iloc[0],
            "feature_set": group.feature_set.iloc[0], "fold_count": len(group),
            **{f"mean_{col}": group[col].mean() for col in METRICS.values()},
            **{f"std_{col}": group[col].std() for col in METRICS.values()},
            **{f"pooled_{label}": pooled[key] for key, label in METRICS.items()},
        })
    summary = pd.DataFrame(rows)
    return summary.sort_values(["mean_average_precision", "experiment_id"], ascending=[False, True])

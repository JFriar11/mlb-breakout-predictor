import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluate_models import SELECTION, evaluate_final_test, evaluate_validation
from src.feature_sets import FEATURE_SETS, predictors
from src.split_data import KEYS, split_data
from src.train_models import EXPERIMENTS, THRESHOLD, fit_experiment, probabilities, train_models
from src.validate_data import TARGET_COLUMNS


@pytest.mark.parametrize("field", [*TARGET_COLUMNS, "player_id", "player_name", "season", "next_season_obp"])
def test_feature_boundary_rejects_leakage(player_seasons, field):
    with pytest.raises(ValueError, match="outside"):
        predictors(player_seasons, [*FEATURE_SETS["baseline"], field])


def test_missing_predictor_is_rejected(player_seasons):
    with pytest.raises(ValueError, match="Missing predictors"):
        predictors(player_seasons.drop(columns="age"), FEATURE_SETS["baseline"])


@pytest.mark.parametrize("experiment", ["M1", "M5"])
def test_pipeline_fits_preprocessing_on_train_only(player_seasons, experiment):
    parts = split_data(player_seasons)
    train, validation = parts["train"], parts["validation"]
    pipeline = fit_experiment(train, experiment)
    imputer = pipeline.named_steps["imputer"]
    assert imputer.statistics_[0] == train.age.median()
    assert imputer.statistics_[0] != player_seasons.age.median()
    if experiment == "M1":
        assert pipeline.named_steps["scaler"].mean_[0] == pytest.approx(
            train.age.fillna(train.age.median()).mean()
        )
    else:
        assert "scaler" not in pipeline.named_steps
    assert pipeline.named_steps["model"].random_state == 42
    scores = probabilities(pipeline, validation, experiment)
    assert len(scores) == len(validation)
    assert ((scores >= 0) & (scores <= 1)).all()
    other = fit_experiment(train, experiment)
    np.testing.assert_allclose(scores, probabilities(other, validation, experiment), atol=1e-14)


def test_pipeline_run_isolates_validation_inputs_and_test_outcomes(player_seasons, tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluate_models.plot_validation", lambda *args: None)
    dataset = tmp_path / "dataset.csv"
    model_dir, prediction_dir, metric_dir, figure_dir = [
        tmp_path / name for name in ("models", "predictions", "metrics", "figures")
    ]

    def run(frame):
        frame.to_csv(dataset, index=False)
        train_models(dataset, model_dir, prediction_dir, metric_dir)
        return evaluate_validation(model_dir, prediction_dir, metric_dir, figure_dir)

    with pytest.raises(ValueError, match="freeze model selection"):
        evaluate_final_test(dataset, model_dir, prediction_dir, metric_dir)
    initial = run(player_seasons)
    validation = split_data(player_seasons)["validation"]
    for experiment in EXPERIMENTS:
        saved = pd.read_csv(prediction_dir / f"validation_{experiment}.csv")
        pipeline = joblib.load(model_dir / f"{experiment}.joblib")
        pd.testing.assert_frame_equal(saved[KEYS], validation[KEYS])
        assert tuple(pipeline.feature_names_in_) == FEATURE_SETS[EXPERIMENTS[experiment][1]]
        assert saved.predicted_probability.between(0, 1).all()
        assert saved.predicted_class.eq(saved.predicted_probability.ge(THRESHOLD)).all()

    selected = json.loads((metric_dir / SELECTION).read_text())
    changed = player_seasons.copy()
    test = changed.season.eq(2024)
    changed.loc[test, "breakout"] = 1 - changed.loc[test, "breakout"]
    changed.loc[test, "wrc_plus_change"] = changed.loc[test, "breakout"].map({0: -5, 1: 25})
    changed.loc[test, "next_season_wrc_plus"] = changed.loc[test, "wrc_plus"] + changed.loc[test, "wrc_plus_change"]
    pd.testing.assert_frame_equal(run(changed), initial, check_exact=False, atol=1e-14)
    frozen = json.loads((metric_dir / SELECTION).read_text())
    assert frozen["experiment"] == selected["experiment"]
    assert frozen["threshold"] == selected["threshold"] == THRESHOLD
    assert not list(prediction_dir.glob("test_*.csv"))
    assert not (metric_dir / "week2_final_test.csv").exists()

    # A new validation distribution and labels must leave training-fitted behavior unchanged.
    before = {
        experiment: probabilities(joblib.load(model_dir / f"{experiment}.joblib"), validation, experiment)
        for experiment in EXPERIMENTS
    }
    validation_rows = changed.season.eq(2023)
    changed.loc[validation_rows, "age"] += 100
    changed.loc[validation_rows, "breakout"] = 1 - changed.loc[validation_rows, "breakout"]
    changed.to_csv(dataset, index=False)
    train_models(dataset, model_dir, prediction_dir, metric_dir)
    with pytest.raises(ValueError, match="Artifact changed"):
        evaluate_final_test(dataset, model_dir, prediction_dir, metric_dir)
    for experiment in EXPERIMENTS:
        pipeline = joblib.load(model_dir / f"{experiment}.joblib")
        np.testing.assert_allclose(probabilities(pipeline, validation, experiment), before[experiment], atol=1e-14)

    # The explicit evaluation succeeds against a newly frozen synthetic run.
    evaluate_validation(model_dir, prediction_dir, metric_dir, figure_dir)
    final = evaluate_final_test(dataset, model_dir, prediction_dir, metric_dir)
    assert final.observations.tolist() == [8]
    assert final.threshold.tolist() == [THRESHOLD]

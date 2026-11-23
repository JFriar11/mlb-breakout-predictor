import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.advanced_models import available_experiments, features, make_experiment
from src.backtesting import HISTORY, folds_for, load_history, run_backtests, summarize
from src.calibration_analysis import (
    analyze_calibration, calibration_metrics, fit_calibrator, reliability, threshold_analysis,
)
from src.error_analysis import analyze_errors
from src.feature_sets import FEATURE_SETS
from src.model_interpretation import importance
from src.split_data import KEYS


@pytest.fixture
def history(player_seasons):
    return player_seasons.loc[player_seasons.season.isin(HISTORY)].copy()


def test_loader_filters_holdout_before_schema_validation(tmp_path, player_seasons):
    dataset = tmp_path / "dataset.csv"
    player_seasons.loc[player_seasons.season.eq(2024), "breakout"] = -99
    player_seasons.to_csv(dataset, index=False)
    loaded = load_history(dataset)
    assert set(loaded.season) == set(HISTORY)
    assert set(loaded.breakout) == {0, 1}


def test_folds_temporal_membership(history, player_seasons):
    folds = list(folds_for(history))
    assert [fold.season for fold in folds] == [2021, 2022, 2023]
    for index, fold in enumerate(folds):
        assert tuple(fold.train.season.unique()) == HISTORY[:index + 2]
        assert list(fold.validation.season.unique()) == [fold.season]
        assert fold.train.next_season.max() <= fold.season
        assert set(map(tuple, fold.train[KEYS].to_numpy())).isdisjoint(
            map(tuple, fold.validation[KEYS].to_numpy())
        )
    with pytest.raises(ValueError, match="historical"):
        list(folds_for(player_seasons))
    with pytest.raises(ValueError, match="Duplicate"):
        list(folds_for(pd.concat([history, history.iloc[:1]])))
    broken = history.copy()
    broken.loc[broken.season.eq(2018), "next_season"] = 2022
    with pytest.raises(ValueError, match="immediately following"):
        list(folds_for(broken))


@pytest.fixture
def backtest(history):
    folds = list(folds_for(history))
    with threadpool_limits(limits=2):
        detail, predictions, fits = run_backtests(folds, ("M1", "M2", "M6"))
    return folds, detail, predictions, fits


def test_fresh_pipelines_and_equal_populations(backtest):
    folds, detail, predictions, fits = backtest
    assert len({id(fit) for fit in fits.values()}) == 9
    assert len({id(fit["imputer"]) for fit in fits.values()}) == 9
    for fold in folds:
        assert fits["M2", fold.season]["scaler"].n_samples_seen_ == len(fold.train)
    populations = [group[KEYS + ["breakout"]].reset_index(drop=True)
                   for _, group in predictions.groupby("experiment_id")]
    for group in populations[1:]:
        pd.testing.assert_frame_equal(group, populations[0])
    assert set(predictions.season) == {2021, 2022, 2023}
    assert summarize(detail, predictions).fold_count.eq(3).all()


@pytest.mark.parametrize("experiment,feature_set", [
    ("M6", "traditional"), ("M7", "traditional_statcast"),
    ("M8", "traditional"), ("M9", "traditional_statcast"),
])
def test_boosters_are_seeded_and_allowlisted(history, experiment, feature_set):
    experiments, blocker = available_experiments()
    if experiment not in experiments:
        pytest.skip(blocker)
    fold = next(folds_for(history))
    train, validation = features(fold.train, experiment), features(fold.validation, experiment)
    assert tuple(train.columns) == FEATURE_SETS[feature_set]
    labels = np.zeros(len(train), dtype=int)
    labels[:2] = 1
    with threadpool_limits(limits=2):
        first = make_experiment(experiment).fit(train, labels).predict_proba(validation)
        second = make_experiment(experiment).fit(train, labels).predict_proba(validation)
    assert first.shape == (len(validation), 2)
    assert np.isfinite(first).all() and ((0 <= first) & (first <= 1)).all()
    np.testing.assert_allclose(first.sum(axis=1), 1)
    np.testing.assert_allclose(first, second)


def test_missing_xgboost_does_not_disable_other_models(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "xgboost", None)
    experiments, error = available_experiments()
    assert error and experiments == tuple(f"M{i}" for i in range(1, 8))


def test_importance_uses_validation_and_aligned_names(backtest, monkeypatch):
    folds, _, _, fits = backtest

    def confirm_validation(model, frame, labels, **kwargs):
        from types import SimpleNamespace

        season = next(year for (experiment, year), fit in fits.items() if fit is model)
        fold = next(fold for fold in folds if fold.season == season)
        pd.testing.assert_series_equal(labels, fold.validation.breakout)
        pd.testing.assert_frame_equal(frame, features(fold.validation, "M6"))
        assert kwargs["scoring"] == "average_precision"
        return SimpleNamespace(importances_mean=np.arange(frame.shape[1]),
                               importances_std=np.zeros(frame.shape[1]))

    monkeypatch.setattr("src.model_interpretation.permutation_importance", confirm_validation)
    result = importance(folds, fits, ["M6"])
    for fold in folds:
        coefficients = result.loc[result.experiment_id.eq("M2")
                                  & result.validation_season.eq(fold.season)]
        assert tuple(coefficients.feature) == FEATURE_SETS["traditional"]
        np.testing.assert_array_equal(coefficients.value,
                                      fits["M2", fold.season]["model"].coef_[0])
    assert set(result.validation_season) == {2021, 2022, 2023}


def test_calibration_fit_uses_only_completed_inner_history(backtest, monkeypatch):
    folds, _, predictions, _ = backtest
    fitted_seasons = []

    class InspectModel:
        def fit(self, frame, labels):
            assert frame.index.equals(labels.index)
            self.last_index = frame.index
            return self

        def predict_proba(self, frame):
            assert self.last_index.intersection(frame.index).empty
            training = pd.concat([folds[-1].train, folds[-1].validation])
            fitted_seasons.append((training.loc[self.last_index, "next_season"].max(),
                                   training.loc[frame.index, "season"].min()))
            score = np.linspace(.15, .65, len(frame))
            return np.column_stack((1 - score, score))

    monkeypatch.setattr("src.calibration_analysis.make_experiment", lambda _: InspectModel())
    folds[0].validation["breakout"] = -99  # evaluation scores keep the original labels
    comparison, bins, scores = analyze_calibration(folds, predictions, ["M1"])
    assert fitted_seasons == [(2018, 2018), (2018, 2018), (2019, 2021),
                              (2018, 2018), (2019, 2021), (2022, 2022)]
    assert set(comparison.method) == {"raw", "sigmoid"}
    assert scores.predicted_probability.between(0, 1).all()
    assert bins.groupby("method")["count"].sum().eq(24).all()
    with pytest.raises(ValueError, match="completed"):
        fit_calibrator(folds[-1].train, "M1", 2022)
    calibrator, count = fit_calibrator(folds[0].train.query("season == 2017"), "M1", 2021)
    assert calibrator is None and count == 0


def test_sparse_bins_and_descriptive_threshold_grid(backtest):
    _, _, predictions, _ = backtest
    tiny = pd.DataFrame({"predicted_probability": [0., .1, 1.], "breakout": [0, 0, 1]})
    bins = reliability(tiny)
    assert list(bins["count"]) == [2, 0, 0, 0, 1]
    assert np.isnan(calibration_metrics(tiny)["slope"])
    thresholds = threshold_analysis(predictions)
    altered = predictions.assign(breakout=1 - predictions.breakout)
    assert thresholds.threshold.equals(threshold_analysis(altered).threshold)
    assert thresholds.groupby("experiment_id").predicted_breakouts.apply(
        lambda count: count.is_monotonic_decreasing
    ).all()


def test_error_order_and_inclusive_threshold(backtest, history):
    _, _, predictions, _ = backtest
    sample = predictions.loc[predictions.experiment_id.eq("M1")].head(8).copy()
    sample["predicted_probability"] = [.01, .5, .9, .2, .4, .8, .65, .55]
    result = analyze_errors(sample, history)
    assert result.predicted_probability.is_monotonic_decreasing
    assert result.loc[result.error_category.eq("false_positive"), "predicted_probability"].min() >= .5
    assert result.loc[result.error_category.eq("false_negative"), "predicted_probability"].max() < .5
    cutoff = result.loc[result.predicted_probability.eq(.5)].iloc[0]
    assert cutoff.predicted_class == 1
    assert result.season_rank.tolist() == list(range(1, 9))

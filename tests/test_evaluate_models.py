import pandas as pd
import pytest

from src.evaluate_models import evaluate_predictions, rank_predictions


def test_known_classification_ranking_and_probability_metrics():
    frame = pd.DataFrame({
        "player_id": [4, 3, 2, 1], "season": [2023] * 4,
        "breakout": [0, 0, 1, 1], "predicted_probability": [.1, .6, .4, .9],
    })
    result = evaluate_predictions(frame)
    for metric in ("precision", "recall", "f1", "accuracy", "precision_at_10", "precision_at_20"):
        assert result[metric] == .5
    assert [result[m] for m in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert result["roc_auc"] == .75
    assert result["ap"] == pytest.approx(5 / 6)
    assert result["brier"] == pytest.approx(.185)
    assert result["recall_at_20"] == 1
    assert result["breakouts_in_top_20"] == 2


def test_top_rank_breaks_probability_ties_by_player_then_season():
    frame = pd.DataFrame({
        "player_id": [2, 2, 1, 4], "season": [2024, 2023, 2024, 2023],
        "predicted_probability": [.3, .3, .3, .4],
    })
    ranked = rank_predictions(frame)
    assert ranked.player_id.tolist() == [4, 1, 2, 2]
    assert ranked.season.tolist() == [2023, 2024, 2023, 2024]


def test_constant_scores_and_no_predicted_positives():
    frame = pd.DataFrame({
        "player_id": [1, 2, 3], "season": [2023] * 3,
        "breakout": [1, 0, 0], "predicted_probability": [.16] * 3,
    })
    result = evaluate_predictions(frame)
    assert result["roc_auc"] == .5
    assert result["ap"] == pytest.approx(1 / 3)
    assert result["precision"] == result["recall"] == result["f1"] == 0


def test_single_class_metrics_raise_controlled_error():
    with pytest.raises(ValueError, match="both binary classes"):
        evaluate_predictions(pd.DataFrame({"breakout": [0]}))

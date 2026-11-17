import pytest

from src.split_data import KEYS, split_audit, split_data


def test_chronological_partitions_and_counts(player_seasons):
    parts = split_data(player_seasons)
    train, validation, test = parts.values()
    assert train.season.max() < validation.season.min() < test.season.min()
    assert train.next_season.max() < validation.next_season.min() < test.next_season.min()
    assert train.season.unique().tolist() == [2017, 2018, 2021, 2022]
    summary, membership = split_audit(parts)
    assert summary.observations.tolist() == [32, 8, 8]
    assert summary.positives.tolist() == [12, 3, 3]
    assert summary.prevalence.tolist() == [0.375] * 3
    assert len(membership) == len(player_seasons)
    assert not membership.duplicated(KEYS).any()
    assert all(part.season.is_monotonic_increasing for part in parts.values())


def test_duplicate_player_season_is_rejected(player_seasons):
    with pytest.raises(ValueError, match="Duplicate"):
        split_data(player_seasons.iloc[[0, *range(len(player_seasons))]])


def test_single_class_partition_is_rejected(player_seasons):
    player_seasons.loc[player_seasons.season.eq(2024), "breakout"] = 0
    with pytest.raises(ValueError, match="test requires both"):
        split_data(player_seasons)


def test_nonconsecutive_outcome_is_rejected(player_seasons):
    player_seasons.loc[player_seasons.season.eq(2022), "next_season"] = 2024
    with pytest.raises(ValueError, match="immediately following"):
        split_data(player_seasons)

"""Smoke-test every EDA artifact without persisting synthetic project results."""
import pandas as pd
from src.build_dataset import build_dataset
from src.validate_data import FEATURE_COLUMNS
import src.explore_data as eda


def test_eda_artifacts(tmp_path, monkeypatch):
    rows = []
    for player in range(15):
        for year in (2022, 2023):
            row = dict.fromkeys(FEATURE_COLUMNS, .1)
            row.update(player_id=player+1, player_name=f'Synthetic {player}', season=year,
                       age=22+player, pa=300+player,
                       wrc_plus=80+player+(year-2022)*(player*3),
                       avg_exit_velocity=85+player/2, avg_launch_angle=10+player/2)
            rows.append(row)
    monkeypatch.setattr(eda, 'ROOT', tmp_path)
    (tmp_path/'results/metrics').mkdir(parents=True)
    result = eda.explore(build_dataset(pd.DataFrame(rows)))
    assert len(list((tmp_path/'results/figures').glob('*.png'))) == 7
    assert result['age'].observations.sum() == 15
    assert (tmp_path/'results/metrics/week1_summary.md').exists()
    assert len(result['top']) == 15

import numpy as np
import pandas as pd
import pytest

from src.feature_sets import TRADITIONAL_FEATURES, STATCAST_FEATURES
from src.split_data import SPLIT_SEASONS


@pytest.fixture
def player_seasons():
    rng = np.random.default_rng(73)
    rows = []
    for season in (year for years in SPLIT_SEASONS.values() for year in years):
        for player_id in range(8):
            row = {c: rng.uniform(.1, .3) for c in (*TRADITIONAL_FEATURES, *STATCAST_FEATURES)}
            age = 21 + player_id if season < 2023 else 45 + player_id
            wrc = 75 + 6 * player_id
            breakout = player_id % 3 == 0
            change = 25 if breakout else -5
            rows.append(dict(row, player_id=player_id, player_name=f"Hitter {player_id}",
                             age=age, pa=320, next_season_pa=360, wrc_plus=wrc,
                             next_season_wrc_plus=wrc + change, wrc_plus_change=change,
                             season=season, next_season=season + 1, breakout=int(breakout)))
    frame = pd.DataFrame(rows)
    frame.loc[frame.age.eq(21), "age"] = np.nan
    return frame.iloc[::-1].reset_index(drop=True)

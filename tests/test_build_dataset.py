import pandas as pd
import pytest
from src.build_dataset import build_dataset
from src.collect_data import merge_sources
from src.validate_data import FEATURE_COLUMNS, TARGET_COLUMNS, validate_dataset


def seasons(pa=(300, 300), wrc=(85, 105), years=(2022, 2023)):
    rows = []
    for year, appearances, value in zip(years, pa, wrc):
        row = dict.fromkeys(FEATURE_COLUMNS, float('nan'))
        row.update(player_id=10, player_name='Test hitter', season=year, age=25, pa=appearances, wrc_plus=value)
        rows.append(row)
    return pd.DataFrame(rows)


def test_boundary_qualifies_and_twenty_is_breakout():
    result = build_dataset(seasons())
    assert len(result) == 1
    assert result.iloc[0].breakout == 1

@pytest.mark.parametrize('pa', [(299, 300), (300, 299)])
def test_either_low_pa_excluded(pa):
    assert build_dataset(seasons(pa=pa)).empty


def test_nineteen_is_not_breakout():
    assert build_dataset(seasons(wrc=(85, 104))).iloc[0].breakout == 0


def test_nonconsecutive_excluded():
    assert build_dataset(seasons(years=(2021, 2023))).empty


def test_duplicate_detected():
    data = seasons()
    with pytest.raises(ValueError, match='Duplicate'):
        build_dataset(pd.concat([data, data.iloc[[0]]]))


def test_missing_next_season_excluded():
    assert build_dataset(seasons().iloc[:1]).empty


def test_no_same_season_targets():
    result = build_dataset(seasons())
    assert (result.next_season == result.season + 1).all()
    result['next_season'] = result.season
    with pytest.raises(ValueError, match='following season'):
        validate_dataset(result)


def test_feature_allowlist_excludes_targets():
    assert not set(FEATURE_COLUMNS) & set(TARGET_COLUMNS)
    assert not any(c.startswith('next_') for c in FEATURE_COLUMNS)
    assert set(FEATURE_COLUMNS).issubset(build_dataset(seasons()).columns)


def test_future_features_do_not_replace_current_features():
    data = seasons()
    data.loc[1, 'age'] = 99
    assert build_dataset(data).iloc[0].age == 25


def sources():
    fg = pd.DataFrame({'IDfg': [1, 2], 'Name': ['Same Name', 'Same Name'], 'PA': [400, 500],
                       'wRC+': [90, 110], 'BB%': [0.10, 0.15]})
    ev = pd.DataFrame({'player_id': [202, 101], 'avg_hit_speed': [95., 85.],
                       'ev95percent': [50., 30.], 'brl_percent': [10., 5.]})
    expected = pd.DataFrame({'player_id': [101, 202], 'est_woba': [.300, .400]})
    ids = pd.DataFrame({'key_fangraphs': [1, 2], 'key_mlbam': [101, 202]})
    return fg, ev, expected, ids


def test_merges_preserve_identity_and_source_units():
    report = {}
    merged = merge_sources(*sources(), 2022, report).set_index('player_id')
    assert merged.loc[101, 'avg_exit_velocity'] == 85
    assert merged.loc[202, 'xwoba'] == .4
    assert merged.loc[101, 'bb_rate'] == .1
    assert merged.loc[101, 'hard_hit_rate'] == .3
    assert merged.xba.isna().all()
    assert any('avg_launch_angle' in v for v in report['unavailable'])


def test_ambiguous_id_mapping_rejected():
    fg, ev, expected, ids = sources()
    ids.loc[1, 'key_mlbam'] = 101
    with pytest.raises(ValueError, match='Duplicate'):
        merge_sources(fg, ev, expected, ids, 2022, {})


def test_missing_target_wrc_excluded():
    assert build_dataset(seasons(wrc=(85, float('nan')))).empty


def test_invalid_fraction_rejected():
    data = seasons()
    data['bb_rate'] = 12
    with pytest.raises(ValueError, match='fraction'):
        build_dataset(data)


def test_source_missing_numeric_values_remain_missing():
    from src.collect_data import numeric
    values = numeric(pd.Series([30., float('nan')]), scale=100)
    assert values.iloc[0] == .3
    assert pd.isna(values.iloc[1])


def test_verified_launch_angle_mapping():
    fg, ev, expected, ids = sources()
    ev['avg_hit_angle'] = [15., 10.]
    result = merge_sources(fg, ev, expected, ids, 2022, {}).set_index('player_id')
    assert result.loc[101, 'avg_launch_angle'] == 10


def test_2020_qualification_effect_is_explicit():
    from src.build_dataset import qualification_report
    data = pd.concat([seasons(years=(2019, 2020), pa=(400, 267)),
                      seasons(years=(2021, 2022), pa=(400, 400))], ignore_index=True)
    report = qualification_report(data).set_index('season')
    assert report.loc[2019, 'qualifying_pairs'] == 0
    assert report.loc[2020, 'qualifying_pairs'] == 0
    assert report.loc[2020, 'current_max_pa'] == 267
    assert report.loc[2021, 'qualifying_pairs'] == 1

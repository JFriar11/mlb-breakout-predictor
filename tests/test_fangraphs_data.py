import json
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest
import requests
import src.collect_data as collector
from src.fangraphs_data import import_exports, normalize_fangraphs, normalize_modern, modern_leaderboard
from src.http_access import AccessRestricted, HttpAccess, retry_after_seconds


def export(season=2024):
    return pd.DataFrame({'playerid': [1], 'Name': ['A'], 'Season': [season],
                         'PA': [400], 'wRC+': [100.2], 'BB%': ['10%']})


def test_single_and_combined_exports(tmp_path):
    data = pd.concat([export(), export(2025)], ignore_index=True)
    path = tmp_path/'combined.csv'
    data.to_csv(path, index=False)
    result, provenance = import_exports([path])
    assert result.Season.tolist() == [2024, 2025]
    assert result['BB%'].tolist() == [.1, .1]
    assert result['wRC+'].tolist() == [100.2, 100.2]
    assert len(provenance[0]['sha256']) == 64
    data = export().drop(columns='Season')
    path = tmp_path/'single.csv'
    data.to_csv(path, index=False)
    result, _ = import_exports([path], import_season=2024)
    assert result.iloc[0].Season == 2024


def test_directory_of_season_exports(tmp_path):
    for year in (2024, 2025):
        export(year).to_csv(tmp_path/f'{year}.csv', index=False)
    result, provenance = import_exports([tmp_path])
    assert len(result) == len(provenance) == 2


def test_missing_or_wrong_import_season_rejected():
    with pytest.raises(ValueError, match='Season column'):
        normalize_fangraphs(export().drop(columns='Season'))
    with pytest.raises(ValueError, match='other than'):
        normalize_fangraphs(export(), year=2025)


@pytest.mark.parametrize('column', ['playerid', 'wRC+', 'PA', 'Name'])
def test_missing_required_export_field_rejected(column):
    with pytest.raises(ValueError, match='required columns'):
        normalize_fangraphs(export().drop(columns=column))


def test_numeric_percentage_units_are_explicit():
    frame = export()
    frame['BB%'] = 10.
    with pytest.raises(ValueError, match='fg-rate-units'):
        normalize_fangraphs(frame)
    assert normalize_fangraphs(frame, rate_units='percent').iloc[0]['BB%'] == .1
    frame['BB%'] = .1
    assert normalize_fangraphs(frame).iloc[0]['BB%'] == .1


def test_team_splits_and_repeated_exports_rejected():
    frame = pd.concat([export(), export()])
    frame['Team'] = ['NYY', 'BOS']
    with pytest.raises(ValueError, match='Duplicate'):
        normalize_fangraphs(frame)


def payload():
    return {'data': [{'playerid': 1, 'xMLBAMID': 101, 'Season': 2024,
                      'PlayerName': 'A', 'TeamNameAbb': '2 Tms',
                      'Name': '<a>A</a>', 'PA': 400, 'wRC+': 100.25},
                     {'playerid': 2, 'xMLBAMID': 102, 'Season': 2024,
                      'PlayerName': 'B', 'TeamNameAbb': 'BOS',
                      'Name': '<a>B</a>', 'PA': 0, 'wRC+': None}], 'totalCount': 2}


def test_modern_response_provenance_and_zero_pa_exclusion():
    frame, audit = normalize_modern(payload(), 2024, 2024)
    assert len(frame) == 1
    assert frame.iloc[0].Name == 'A'
    assert frame.iloc[0].Team == '2 Tms'
    assert frame.iloc[0].MLBAMID == 101
    assert frame.iloc[0]['wRC+'] == 100.25
    assert audit['zero_pa_excluded_by_season'] == {2024: 1}


def test_incomplete_modern_response_rejected():
    data = payload()
    data['totalCount'] = 20
    with pytest.raises(ValueError, match='Incomplete'):
        normalize_modern(data, 2024, 2024)
    with pytest.raises(ValueError, match='requested seasons'):
        normalize_modern(payload(), 2023, 2024)


def test_current_endpoint_request_settings():
    get = Mock(return_value=Mock(json=lambda: payload(), url='https://www.fangraphs.com/test'))
    modern_leaderboard(2024, 2024, get)
    settings = get.call_args.kwargs['params']
    assert settings['qual'] == '0' and settings['ind'] == '1'
    assert settings['team'] == '0' and settings['stats'] == 'bat'


def test_identity_conflict_fails_with_report():
    fg = normalize_fangraphs(export())
    fg['MLBAMID'] = 101
    ids = pd.DataFrame({'key_fangraphs': [1], 'key_mlbam': [102]})
    report = {}
    with pytest.raises(ValueError, match='ID conflicts'):
        collector.verified_ids(fg, ids, report)
    assert len(report['id_conflicts']) == 1


def test_missing_identity_reported_and_not_name_matched():
    fg = normalize_fangraphs(export())
    ids = pd.DataFrame({'key_fangraphs': [9], 'key_mlbam': [101]})
    report = {}
    assert collector.verified_ids(fg, ids, report).empty
    from test_build_dataset import sources
    original, ev, expected, _ = sources()
    output = collector.merge_sources(original, ev, expected, ids, 2024, report)
    assert output.empty
    assert report['unmatched_records'] == 2
    assert report['matched_records'] == 0


def test_ambiguous_identity_reported():
    fg = normalize_fangraphs(export())
    ids = pd.DataFrame({'key_fangraphs': [1, 1], 'key_mlbam': [101, 102]})
    report = {}
    with pytest.raises(ValueError, match='Duplicate'):
        collector.verified_ids(fg, ids, report)
    assert report['ambiguous_records'] == 1


def test_429_stops_run_and_persists_retry_after(tmp_path):
    response = requests.Response()
    response.status_code = 429
    response.headers['Retry-After'] = '3600'
    original = Mock(return_value=response)
    access = HttpAccess(tmp_path/'cooldowns.json')
    for _ in range(2):
        with pytest.raises(AccessRestricted):
            access.request(original, None, 'GET', 'https://www.fangraphs.com/test')
    assert original.call_count == 1
    other = HttpAccess(tmp_path/'cooldowns.json')
    with pytest.raises(AccessRestricted, match='cooldown'):
        other.request(original, None, 'GET', 'https://www.fangraphs.com/another')
    assert original.call_count == 1


def test_retry_after_http_date():
    date = format_datetime(datetime.fromtimestamp(2000, timezone.utc), usegmt=True)
    assert retry_after_seconds(date, 1000) == 1000
    assert retry_after_seconds('10', 1000) == 10


def test_access_restrictions_are_never_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, 'ROOT', tmp_path)
    (tmp_path/'data/raw').mkdir(parents=True)
    fetch = Mock(side_effect=AccessRestricted('HTTP 429'))
    with pytest.raises(AccessRestricted):
        collector.cached('restricted', fetch, attempts=3)
    assert fetch.call_count == 1


def test_successful_cache_is_reused_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, 'ROOT', tmp_path)
    (tmp_path/'data/raw').mkdir(parents=True)
    frame = export()
    frame.to_csv(tmp_path/'data/raw/success.csv', index=False)
    fetch = Mock(side_effect=AssertionError('Network should not be called'))
    assert len(collector.cached('success', fetch)) == 1
    fetch.assert_not_called()


def test_transient_timeout_uses_backoff(monkeypatch):
    fetch = Mock(side_effect=[requests.Timeout('temporary'), 'success'])
    sleep = Mock()
    monkeypatch.setattr(collector.time, 'sleep', sleep)
    assert collector.retry_fetch(fetch, 3) == 'success'
    assert fetch.call_count == 2
    sleep.assert_called_once_with(1)


def test_schema_error_does_not_retry(monkeypatch):
    fetch = Mock(side_effect=ValueError('changed schema'))
    sleep = Mock()
    monkeypatch.setattr(collector.time, 'sleep', sleep)
    with pytest.raises(ValueError):
        collector.retry_fetch(fetch, 3)
    assert fetch.call_count == 1
    sleep.assert_not_called()

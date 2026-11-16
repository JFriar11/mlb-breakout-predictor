"""FanGraphs CSV import and current public leaderboard response normalization.

wRC+ is used exactly as supplied by FanGraphs. Percent strings are explicit;
bare numeric rates default to fractions, with an opt-in percent unit for exports.
"""
from pathlib import Path
import hashlib
import pandas as pd
from .validate_data import unique_keys

RATE_FIELDS = ['BB%', 'K%', 'O-Swing%', 'Contact%', 'SwStr%']
ALIASES = {'playerid': 'IDfg', 'PlayerId': 'IDfg', 'PlayerID': 'IDfg',
           'Player ID': 'IDfg', 'playerId': 'IDfg', 'NameASCII': 'Name', 'Season': 'Season',
           'season': 'Season', 'xMLBAMID': 'MLBAMID'}


def normalize_fangraphs(frame, year=None, rate_units='fraction'):
    """Normalize without guessing a year or combining team-split rate statistics."""
    data = frame.copy()
    data.columns = data.columns.str.strip().str.lstrip('\ufeff')
    for source, target in ALIASES.items():
        if source in data and target not in data:
            data = data.rename(columns={source: target})
    required = {'IDfg', 'Name', 'PA', 'wRC+'}
    if not required.issubset(data):
        raise ValueError(f'FanGraphs missing required columns: {sorted(required-set(data))}')
    if 'Season' not in data:
        if year is None:
            raise ValueError('Combined exports require a Season column; single-season imports require --import-season')
        data['Season'] = year
    for col in ['Season', 'IDfg', 'PA', 'wRC+', 'Age', 'wOBA', 'ISO', 'BABIP', 'MLBAMID']:
        if col in data:
            text = data[col].astype('string').str.strip().replace({'': pd.NA, '-': pd.NA, '--': pd.NA})
            data[col] = pd.to_numeric(text, errors='raise').astype(float)
    for col in ['Season', 'IDfg', 'PA']:
        if data[col].isna().any() or not ((data[col] % 1 == 0) & (data[col] >= 0)).all():
            raise ValueError(f'FanGraphs {col} must contain nonmissing nonnegative integers')
    if year is not None and not (data.Season == year).all():
        raise ValueError(f'Export contains a season other than {year}')
    if data['wRC+'].isna().any() or data.Name.isna().any():
        raise ValueError('FanGraphs requires nonmissing Name and wRC+')
    for col in RATE_FIELDS:
        if col in data:
            text = data[col].astype('string').str.strip().replace({'': pd.NA, '-': pd.NA, '--': pd.NA})
            explicit_percent = text.str.endswith('%').fillna(False)
            values = pd.to_numeric(text.str.rstrip('%'), errors='raise').astype(float)
            data[col] = values / (100 if rate_units == 'percent' else 1)
            if rate_units != 'percent':
                data.loc[explicit_percent, col] = values[explicit_percent]/100
            if not data[col].dropna().between(0, 1).all():
                raise ValueError(f'{col} outside [0, 1]; use --fg-rate-units percent for numeric percentages')
    # Repeated rows are rejected, including team splits. Request all-team aggregates.
    unique_keys(data, ['IDfg', 'Season'])
    return data


def import_exports(paths, import_season=None, rate_units='fraction'):
    frames, provenance = [], []
    for path in paths:
        path = Path(path)
        files = sorted(path.glob('*.csv')) if path.is_dir() else [path]
        for file in files:
            frames.append(normalize_fangraphs(pd.read_csv(file), import_season, rate_units))
            provenance.append({'path': str(file.resolve()), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    if not frames:
        raise ValueError('No CSV exports found')
    combined = pd.concat(frames, ignore_index=True)
    unique_keys(combined, ['IDfg', 'Season'])
    return combined, provenance


def modern_leaderboard(start_year, end_year, get):
    """Read the public JSON backing the current FanGraphs MLB leaderboard.

    No credentials, browser impersonation, alternate hosts, or challenge handling.
    The caller enforces status handling and persistent Retry-After cooldowns.
    """
    params = {'age': '', 'pos': 'all', 'stats': 'bat', 'lg': 'all', 'qual': '0',
              'season': end_year, 'season1': start_year, 'startdate': f'{start_year}-01-01',
              'enddate': f'{end_year}-12-31', 'month': '0', 'team': '0',
              'pageitems': '20000', 'pagenum': '1', 'ind': '1', 'type': '8',
              'sortdir': 'default', 'sortstat': 'WAR'}
    response = get('https://www.fangraphs.com/api/leaders/major-league/data', params=params)
    payload = response.json()
    validate_response(payload, start_year, end_year)
    return payload, response.url


def validate_response(payload, start_year, end_year):
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
        raise ValueError('Unexpected FanGraphs JSON schema')
    if not payload['data'] or len(payload['data']) != int(payload.get('totalCount', -1)):
        raise ValueError('Incomplete or empty FanGraphs response; pagination may have changed')
    frame = pd.DataFrame(payload['data'])
    if set(frame.Season) != set(range(start_year, end_year+1)):
        raise ValueError('FanGraphs response does not cover precisely the requested seasons')


def normalize_modern(payload, start_year, end_year):
    validate_response(payload, start_year, end_year)
    frame = pd.DataFrame(payload['data'])
    counts = frame.groupby('Season').size().to_dict()
    zero_pa = frame.loc[frame.PA == 0].groupby('Season').size().to_dict()
    # qual=0 also returns pitchers/players without any batting PA; these cannot
    # qualify and have no wRC+. Record their exclusion before normalizing hitters.
    frame = frame.loc[frame.PA > 0].copy()
    # Prefer the explicit text fields over HTML anchor display strings.
    for target, source in [('Name', 'PlayerName'), ('Team', 'TeamNameAbb')]:
        if source not in frame:
            raise ValueError(f'FanGraphs JSON missing plain-text {source}')
        frame[target] = frame[source]
    frame = normalize_fangraphs(frame)
    return frame, {'raw_records_by_season': counts, 'zero_pa_excluded_by_season': zero_pa,
                   'wrc_plus_precision': 'Unrounded FanGraphs API values; no rounding before label calculation'}

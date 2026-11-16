"""Cache season-level FanGraphs/Savant responses and audit exact returned schemas."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import hashlib
import requests
from unittest.mock import patch
import pandas as pd
from .validate_data import ROOT, TRADITIONAL, validate_seasons, unique_keys

from .fangraphs_data import import_exports, modern_leaderboard, normalize_modern, normalize_fangraphs
from .http_access import HttpAccess, AccessRestricted

FG = dict(zip(TRADITIONAL, ['wRC+', 'wOBA', 'ISO', 'BABIP', 'BB%', 'K%',
                            'O-Swing%', 'Contact%', 'SwStr%']))
EV = {'avg_exit_velocity': 'avg_hit_speed', 'hard_hit_rate': 'ev95percent',
      'barrel_rate': 'brl_percent', 'avg_launch_angle': 'avg_hit_angle'}
EXPECTED = {'xba': 'est_ba', 'xwoba': 'est_woba'}


def numeric(series, scale=1):
    # Explicit source units; never infer units from a season's observed maximum.
    text = series.astype('string').str.rstrip('%')
    return pd.to_numeric(text, errors='raise').astype(float) / scale


def extract(frame, mapping, report, source, scale_columns=()):
    report.setdefault('schemas', {})[source] = list(frame.columns)
    result = pd.DataFrame(index=frame.index)
    for target, column in mapping.items():
        if column not in frame:
            report.setdefault('unavailable', []).append(f'{source}: {column} ({target})')
            result[target] = float('nan')
        else:
            result[target] = numeric(frame[column], 100 if target in scale_columns else 1)
    return result


def merge_sources(fg, ev, expected, ids, year, report):
    required = {'IDfg', 'Name', 'PA', 'wRC+'}
    if not required.issubset(fg):
        raise ValueError(f'FanGraphs missing required columns: {required-set(fg)}')
    if 'Season' in fg and not (numeric(fg.Season) == year).all():
        raise ValueError('FanGraphs returned an unexpected season')
    unique_keys(fg, ['IDfg'])  # Never sum team splits or average rate statistics.
    unique_keys(ev, ['player_id'])
    unique_keys(expected, ['player_id'])
    ids = ids[['key_fangraphs', 'key_mlbam']].copy()
    for col in ids:
        ids[col] = pd.to_numeric(ids[col], errors='raise')
    ids = ids.loc[(ids.key_fangraphs > 0) & (ids.key_mlbam > 0)].drop_duplicates()
    ambiguous = ids.loc[ids.duplicated('key_fangraphs', keep=False) | ids.duplicated('key_mlbam', keep=False)]
    report['ambiguous_mappings'] = ambiguous.to_dict('records')
    report['ambiguous_records'] = int(fg.IDfg.isin(ambiguous.key_fangraphs).sum())
    unique_keys(ids, ['key_fangraphs'])
    unique_keys(ids, ['key_mlbam'])
    base = extract(fg, FG, report, 'fangraphs')
    base['fangraphs_id'] = numeric(fg.IDfg)
    base['player_name'] = fg.Name
    base['pa'] = numeric(fg.PA)
    base['age'] = numeric(fg.Age) if 'Age' in fg else float('nan')
    base['team'] = fg.Team if 'Team' in fg else pd.NA
    report['missing_metadata'] = [c for c in ['Age', 'Team'] if c not in fg]
    base['season'] = year
    base = base.merge(ids, left_on='fangraphs_id', right_on='key_fangraphs', how='left', validate='one_to_one')
    report['unmapped_fangraphs_ids'] = base.loc[base.key_mlbam.isna(), 'fangraphs_id'].tolist()
    report['unmatched_records'] = int(base.key_mlbam.isna().sum())
    report['matched_records'] = int(base.key_mlbam.notna().sum())
    report['input_records'] = len(base)
    report['multi_team_records'] = int(fg.Team.astype(str).str.contains(r'\d+ Tms|TOT|---', regex=True).sum()) if 'Team' in fg else 0
    base = base.dropna(subset=['key_mlbam']).rename(columns={'key_mlbam': 'player_id'})
    base['player_id'] = base.player_id.astype('int64')
    for frame, mapping, source in [(ev, EV, 'savant_exit_velocity'), (expected, EXPECTED, 'savant_expected')]:
        clean = extract(frame, mapping, report, source, ['hard_hit_rate', 'barrel_rate'])
        clean['player_id'] = numeric(frame.player_id)
        report.setdefault('statcast_matches', {})[source] = {
            'matched': int(base.player_id.isin(clean.player_id).sum()),
            'unmatched': int((~base.player_id.isin(clean.player_id)).sum())}
        base = base.merge(clean, on='player_id', how='left', validate='one_to_one')
    base = base.drop(columns=['key_fangraphs'])
    validate_seasons(base)
    report['rows'] = len(base)
    report['missing_values'] = base.isna().sum().to_dict()
    return base


def retry_fetch(fetch, attempts):
    for attempt in range(attempts):
        try:
            return fetch()
        except AccessRestricted:
            raise
        except Exception as exc:
            transient = isinstance(exc, (requests.Timeout, requests.ConnectionError)) or (
                isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code >= 500)
            if not transient or attempt == attempts-1:
                raise
            time.sleep(2**attempt)


def cached(name, fetch, refresh=False, attempts=3):
    path = ROOT/f'data/raw/{name}.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path)
    result = retry_fetch(fetch, attempts)
    if result.empty:
        raise ValueError(f'{name}: source returned no records')
    temp = path.with_suffix('.tmp')
    result.to_csv(temp, index=False)
    temp.replace(path)
    return result


def verified_ids(fg, register, report):
    """Cross-check source-supplied IDs against Chadwick; never match on names."""
    ids = register[['key_fangraphs', 'key_mlbam']].copy()
    for col in ids:
        ids[col] = pd.to_numeric(ids[col], errors='raise')
    ids = ids.loc[(ids.key_fangraphs > 0) & (ids.key_mlbam > 0)].drop_duplicates()
    relevant = ids.loc[ids.key_fangraphs.isin(fg.IDfg)].copy()
    ambiguous = relevant.loc[relevant.duplicated('key_fangraphs', keep=False) | relevant.duplicated('key_mlbam', keep=False)]
    report['ambiguous_mappings'] = ambiguous.to_dict('records')
    report['ambiguous_records'] = int(fg.IDfg.isin(ambiguous.key_fangraphs).sum())
    unique_keys(relevant, ['key_fangraphs'])
    unique_keys(relevant, ['key_mlbam'])
    if 'MLBAMID' not in fg:
        report['identity_method'] = 'Chadwick register: FanGraphs ID to MLBAM ID'
        return relevant
    direct = fg[['IDfg', 'MLBAMID']].drop_duplicates().rename(columns={'IDfg': 'key_fangraphs', 'MLBAMID': 'key_mlbam'})
    valid = direct.key_mlbam.notna() & (direct.key_mlbam > 0) & (direct.key_mlbam % 1 == 0)
    report['source_missing_mlbam_ids'] = direct.loc[~valid, 'key_fangraphs'].tolist()
    direct = direct.loc[valid]
    unique_keys(direct, ['key_fangraphs'])
    unique_keys(direct, ['key_mlbam'])
    comparison = direct.merge(relevant, on='key_fangraphs', suffixes=('_source', '_register'), how='left', validate='one_to_one')
    conflicts = comparison.loc[comparison.key_mlbam_register.notna() & (comparison.key_mlbam_source != comparison.key_mlbam_register)]
    report['id_conflicts'] = conflicts.to_dict('records')
    report['register_verified_records'] = int(comparison.key_mlbam_register.notna().sum())
    report['source_only_records'] = int(comparison.key_mlbam_register.isna().sum())
    report['identity_method'] = 'FanGraphs supplied MLBAM ID, independently checked against Chadwick when available'
    if not conflicts.empty:
        raise ValueError('FanGraphs/Chadwick ID conflicts; see merge-quality report')
    # Register can fill missing direct IDs, but never override a conflicting ID.
    result = pd.concat([direct, relevant.loc[~relevant.key_fangraphs.isin(direct.key_fangraphs)]], ignore_index=True)
    unique_keys(result, ['key_fangraphs'])
    unique_keys(result, ['key_mlbam'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-year', type=int, default=2017)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--refresh', action='store_true', help='Refresh FanGraphs only; preserve Savant caches')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--fg-source', choices=['current', 'pybaseball'], default='current')
    parser.add_argument('--import-fangraphs', type=Path, nargs='+', help='CSV files or a directory; no FanGraphs network requests')
    parser.add_argument('--import-season', type=int, help='Explicit season for exports without Season')
    parser.add_argument('--fg-rate-units', choices=['fraction', 'percent'], default='fraction')
    args = parser.parse_args()
    if not 2017 <= args.start_year <= args.end_year <= 2025 or args.attempts < 1:
        parser.error('Use 2017 <= start-year <= end-year <= 2025 and attempts >= 1')
    import pybaseball as pb
    original = requests.sessions.Session.request
    access = HttpAccess(ROOT/'data/raw/http_cooldowns.json')
    def bounded_request(self, method, url, **kwargs):
        return access.request(original, self, method, url, **kwargs)
    audit = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'pybaseball_version': pb.__version__,
             'fangraphs_method': 'manual_csv' if args.import_fangraphs else args.fg_source, 'seasons': {}}
    failures = []
    previous = ROOT/'results/metrics/collection_report.json'
    history = ROOT/'results/metrics/collection_report_initial_failure.json'
    if previous.exists() and not history.exists():
        history.write_bytes(previous.read_bytes())
    with patch.object(requests.sessions.Session, 'request', bounded_request):
        combined, preparation_error = None, None
        try:
            if args.import_fangraphs:
                combined, audit['import_provenance'] = import_exports(args.import_fangraphs, args.import_season, args.fg_rate_units)
                for year, frame in combined.groupby('Season'):
                    frame.to_csv(ROOT/f'data/raw/fangraphs_{int(year)}.csv', index=False)
            elif args.fg_source == 'current':
                raw = ROOT/f'data/raw/fangraphs_response_{args.start_year}_{args.end_year}.json'
                metadata = raw.with_suffix('.metadata.json')
                if raw.exists() and not args.refresh:
                    payload = json.loads(raw.read_text())
                    source_metadata = json.loads(metadata.read_text()) if metadata.exists() else {}
                else:
                    payload, url = retry_fetch(lambda: modern_leaderboard(args.start_year, args.end_year, requests.get), args.attempts)
                    raw.write_text(json.dumps(payload)+'\n')
                    source_metadata = {'url': url, 'retrieved_utc': audit['timestamp_utc']}
                    metadata.write_text(json.dumps(source_metadata, indent=2)+'\n')
                combined, audit['fangraphs_exclusions'] = normalize_modern(payload, args.start_year, args.end_year)
                audit['source_provenance'] = {**source_metadata, 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}
                for year, frame in combined.groupby('Season'):
                    frame.to_csv(ROOT/f'data/raw/fangraphs_{int(year)}.csv', index=False)
            if combined is not None:
                register = cached('player_id_crosswalk', lambda: pb.playerid_reverse_lookup(combined.IDfg.unique().tolist(), key_type='fangraphs'), attempts=args.attempts)
        except Exception as exc:
            preparation_error = str(exc)
        for year in range(args.start_year, args.end_year+1):
            report = audit['seasons'][str(year)] = {'unavailable': [], 'ambiguous_mappings': [], 'id_conflicts': []}
            normalized = ROOT/f'data/processed/seasons_{year}.csv'
            try:
                if preparation_error:
                    raise RuntimeError(preparation_error)
                if combined is None:
                    fg = normalize_fangraphs(cached(f'fangraphs_{year}', lambda: pb.batting_stats(year, year, qual=0, ind=1), args.refresh, args.attempts), year)
                    register = cached(f'player_ids_{year}', lambda: pb.playerid_reverse_lookup(fg.IDfg.tolist(), key_type='fangraphs'), attempts=args.attempts)
                else:
                    fg = combined.loc[combined.Season == year].copy()
                    if fg.empty:
                        raise ValueError(f'No imported FanGraphs records for {year}')
                ev = cached(f'exit_velocity_{year}', lambda: pb.statcast_batter_exitvelo_barrels(year, minBBE=0), attempts=args.attempts)
                expected = cached(f'expected_{year}', lambda: pb.statcast_batter_expected_stats(year, minPA=0), attempts=args.attempts)
                ids = verified_ids(fg, register, report)
                merged = merge_sources(fg, ev, expected, ids, year, report)
                merged.to_csv(normalized, index=False)
                report.update(status='success', raw_rows={'fangraphs': len(fg), 'exit_velocity': len(ev), 'expected': len(expected)},
                              qualified_300_pa=int((merged.pa >= 300).sum()), max_pa=float(merged.pa.max()))
                print(f'{year}: {len(merged)} normalized hitters; {report["matched_records"]} IDs matched, {report["unmatched_records"]} unmatched', flush=True)
            except Exception as exc:
                report.update(status='failed', error=str(exc))
                normalized.unlink(missing_ok=True)
                failures.append(year)
                print(f'{year}: {exc}', flush=True)
            previous.write_text(json.dumps(audit, indent=2)+'\n')
            (ROOT/'results/metrics/merge_quality_report.json').write_text(json.dumps(audit['seasons'], indent=2)+'\n')
    audit['merge_totals'] = {key: sum(record.get(key, 0) for record in audit['seasons'].values())
                             for key in ['input_records', 'matched_records', 'unmatched_records',
                                         'ambiguous_records', 'register_verified_records', 'source_only_records', 'multi_team_records']}
    audit['status'] = 'failed' if failures else 'success'
    previous.write_text(json.dumps(audit, indent=2)+'\n')
    (ROOT/'results/metrics/merge_quality_report.json').write_text(
        json.dumps({'totals': audit['merge_totals'], 'seasons': audit['seasons']}, indent=2)+'\n')
    if failures:
        raise SystemExit(f'Collection incomplete for {failures}; see results/metrics/collection_report.json')

if __name__ == '__main__':
    main()

"""Construct consecutive, retrospective 300-PA player-season pairs."""
import argparse
import json
import pandas as pd
from .validate_data import ROOT, FEATURE_COLUMNS, validate_seasons, validate_dataset


def build_dataset(seasons):
    validate_seasons(seasons)
    current = seasons.dropna(subset=['pa', 'wrc_plus']).copy()
    following = current[['player_id', 'season', 'pa', 'wrc_plus']].rename(columns={
        'season': 'next_season', 'pa': 'next_season_pa', 'wrc_plus': 'next_season_wrc_plus'})
    following['season'] = following.next_season - 1
    paired = current.merge(following, on=['player_id', 'season'], how='inner', validate='one_to_one')
    paired = paired.loc[(paired.pa >= 300) & (paired.next_season_pa >= 300)].copy()
    paired['wrc_plus_change'] = paired.next_season_wrc_plus - paired.wrc_plus
    paired['breakout'] = (paired.wrc_plus_change >= 20).astype(int)
    if not paired.empty:
        validate_dataset(paired)
    return paired.sort_values(['season', 'player_id']).reset_index(drop=True)


def qualification_report(seasons):
    """Quantify each consecutive transition without relaxing the PA threshold."""
    validate_seasons(seasons)
    rows = []
    for year in range(int(seasons.season.min()), int(seasons.season.max())):
        start = seasons.loc[seasons.season == year]
        end = seasons.loc[seasons.season == year+1]
        pairs = start[['player_id', 'pa', 'wrc_plus']].merge(
            end[['player_id', 'pa', 'wrc_plus']], on='player_id', suffixes=('_current', '_next'), validate='one_to_one')
        valid = pairs[['wrc_plus_current', 'wrc_plus_next', 'pa_current', 'pa_next']].notna().all(axis=1)
        qualifies = valid & (pairs.pa_current >= 300) & (pairs.pa_next >= 300)
        rows.append({'season': year, 'next_season': year+1,
                     'consecutive_players': len(pairs), 'current_300_pa': int((start.pa >= 300).sum()),
                     'next_300_pa': int((end.pa >= 300).sum()),
                     'current_max_pa': start.pa.max(), 'next_max_pa': end.pa.max(),
                     'missing_outcome_or_pa': int((~valid).sum()), 'qualifying_pairs': int(qualifies.sum()),
                     'excluded_by_300_pa_rule': int((valid & ~qualifies).sum())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-year', type=int, default=2017)
    parser.add_argument('--end-year', type=int, default=2025)
    args = parser.parse_args()
    paths = [ROOT/f'data/processed/seasons_{y}.csv' for y in range(args.start_year, args.end_year+1)]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise SystemExit('Collection incomplete; missing normalized seasons: ' + ', '.join(missing))
    seasons = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    data = build_dataset(seasons)
    validate_dataset(data)
    output = ROOT/'data/processed/mlb_breakout_dataset.csv'
    data.to_csv(output, index=False)
    qualification_report(seasons).to_csv(ROOT/'results/metrics/qualification_by_transition.csv', index=False)
    (output.parent/'feature_columns.json').write_text(json.dumps(FEATURE_COLUMNS, indent=2)+'\n')
    print(f'Saved {len(data)} observations; breakout rate {data.breakout.mean():.2%}')

if __name__ == '__main__':
    main()

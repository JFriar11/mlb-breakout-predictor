"""Shared schema and fail-fast validation; all rates are fractions in [0, 1]."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADITIONAL = ['wrc_plus', 'woba', 'iso', 'babip', 'bb_rate', 'k_rate',
               'chase_rate', 'contact_rate', 'swstr_rate']
STATCAST = ['avg_exit_velocity', 'hard_hit_rate', 'barrel_rate',
            'avg_launch_angle', 'xba', 'xwoba']
FEATURE_COLUMNS = ['age', 'pa'] + TRADITIONAL + STATCAST
TARGET_COLUMNS = ['next_season', 'next_season_pa', 'next_season_wrc_plus',
                  'wrc_plus_change', 'breakout']
RATE_COLUMNS = ['bb_rate', 'k_rate', 'chase_rate', 'contact_rate', 'swstr_rate',
                'hard_hit_rate', 'barrel_rate']


def unique_keys(df, keys):
    if df[keys].isna().any().any():
        raise ValueError(f'Missing identity keys: {keys}')
    if df.duplicated(keys).any():
        raise ValueError(f'Duplicate records for {keys}')


def validate_seasons(df):
    required = ['player_id', 'player_name', 'season'] + FEATURE_COLUMNS
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    unique_keys(df, ['player_id', 'season'])
    for col in RATE_COLUMNS:
        if not df[col].dropna().between(0, 1).all():
            raise ValueError(f'{col} must be a fraction in [0, 1]')
    for col in ['player_id', 'season', 'pa']:
        values = df[col].dropna()
        if not ((values % 1 == 0) & (values >= 0)).all():
            raise ValueError(f'{col} must contain nonnegative integers')


def validate_dataset(df):
    validate_seasons(df)
    if df.empty:
        raise ValueError('No qualifying observations')
    if df[TARGET_COLUMNS + ['pa', 'wrc_plus']].isna().any().any():
        raise ValueError('Missing qualification or target values')
    if not (df.next_season == df.season + 1).all():
        raise ValueError('Targets must come from the immediately following season')
    if not ((df.pa >= 300) & (df.next_season_pa >= 300)).all():
        raise ValueError('Both seasons require at least 300 PA')
    change = df.next_season_wrc_plus - df.wrc_plus
    if not (abs(change - df.wrc_plus_change) < 1e-8).all():
        raise ValueError('Incorrect wRC+ change')
    if not (df.breakout == (change >= 20).astype(int)).all():
        raise ValueError('Incorrect breakout labels')
    if set(FEATURE_COLUMNS) & set(TARGET_COLUMNS):
        raise ValueError('Target leakage in feature allowlist')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/processed/mlb_breakout_dataset.csv')
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f'Dataset not found: {args.input}. Run collection and build_dataset first.')
    validate_dataset(pd.read_csv(args.input))
    print('Dataset validation passed')

if __name__ == '__main__':
    main()

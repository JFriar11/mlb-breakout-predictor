"""Reproducible Week 1 figures and summary, also called by the notebook."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .validate_data import ROOT, FEATURE_COLUMNS, STATCAST, validate_dataset


def age_groups(data):
    return pd.cut(data.age, [-float('inf'), 24, 29, float('inf')],
                  labels=['24 and younger', '25–29', '30 and older'])


def table(frame):
    # Markdown without an extra tabulate dependency.
    out = frame.round(4).reset_index().astype('string').fillna('NA')
    return '| ' + ' | '.join(out.columns) + ' |\n| ' + ' | '.join(['---']*len(out.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in out.to_numpy())


def explore(data):
    validate_dataset(data)
    sns.set_theme(style='whitegrid')
    figures = ROOT/'results/figures'
    figures.mkdir(parents=True, exist_ok=True)
    def save(name):
        plt.tight_layout()
        plt.savefig(figures/f'{name}.png', dpi=160, bbox_inches='tight')
        plt.close()
    season = data.groupby('season').breakout.agg(observations='size', breakouts='sum', breakout_rate='mean')
    season['breakout_percent'] = season.pop('breakout_rate')*100
    seasons = range(int(data.season.min()), int(data.season.max())+1)
    season = season.reindex(seasons)
    season[['observations', 'breakouts']] = season[['observations', 'breakouts']].fillna(0).astype(int)
    age = data.groupby(age_groups(data), observed=False).breakout.agg(observations='size', breakout_rate='mean')
    age['breakout_percent'] = age.pop('breakout_rate')*100
    missing = data[FEATURE_COLUMNS].isna().sum().to_frame('missing')
    missing['missing_percent'] = missing.missing/len(data)*100
    coverage = data.groupby('season')[STATCAST].agg(lambda x: x.notna().mean()*100).reindex(seasons)
    top_cols = ['player_name', 'season', 'wrc_plus', 'next_season_wrc_plus', 'wrc_plus_change', 'pa', 'next_season_pa']
    top = data.nlargest(15, 'wrc_plus_change')[top_cols]
    data.breakout.value_counts().reindex([0, 1], fill_value=0).rename(index={0: 'Non-breakout', 1: 'Breakout'}).plot.bar(rot=0)
    plt.ylabel('Observations'); save('class_balance')
    season.breakout_percent.plot.bar(rot=0)
    plt.ylabel('Breakout (%)'); save('breakout_by_season')
    sns.histplot(data.wrc_plus_change, bins=35)
    plt.axvline(20, color='red', linestyle='--', label='+20 breakout threshold')
    plt.legend(); save('wrc_plus_change')
    age.breakout_percent.plot.bar(rot=0)
    plt.ylabel('Breakout (%)'); save('breakout_by_age')
    selected = ['wrc_plus', 'age', 'k_rate', 'bb_rate', 'avg_exit_velocity', 'hard_hit_rate', 'barrel_rate', 'xwoba']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, ax in zip(selected, axes.flat):
        if data[col].notna().any():
            sns.boxplot(data=data, x='breakout', y=col, ax=ax)
        else:
            ax.text(.5, .5, 'Unavailable', ha='center'); ax.set_title(col)
    save('feature_distributions')
    correlations = data[FEATURE_COLUMNS].corr()
    plt.figure(figsize=(13, 10))
    sns.heatmap(correlations, cmap='vlag', center=0, vmin=-1, vmax=1)
    save('predictor_correlations')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for col, ax in zip(['wrc_plus', 'avg_exit_velocity', 'xwoba'], axes):
        sns.scatterplot(data=data, x=col, y='wrc_plus_change', hue='breakout', alpha=.4, ax=ax)
        ax.axhline(20, color='red', linestyle='--')
    save('feature_change_relationships')
    relationships = data[FEATURE_COLUMNS].corrwith(data.wrc_plus_change).sort_values().to_frame('Pearson correlation with change')
    summaries = data.groupby('breakout')[selected].agg(['count', 'mean', 'median', 'std'])
    missing.to_csv(ROOT/'results/metrics/missing_predictors.csv')
    relationships.to_csv(ROOT/'results/metrics/feature_change_correlations.csv')
    season.to_csv(ROOT/'results/metrics/breakout_by_season.csv')
    age.to_csv(ROOT/'results/metrics/breakout_by_age.csv')
    quality_path = ROOT/'results/metrics/collection_report.json'
    quality_text = 'Collection audit unavailable for this input.'
    if quality_path.exists():
        quality = json.loads(quality_path.read_text())
        quality_rows = [{'season': int(y), **{k: d.get(k, 0) for k in
                        ['input_records', 'matched_records', 'unmatched_records', 'ambiguous_records', 'register_verified_records', 'multi_team_records']}}
                        for y, d in quality.get('seasons', {}).items()]
        if quality_rows:
            quality_text = table(pd.DataFrame(quality_rows).set_index('season'))
    qualification_path = ROOT/'results/metrics/qualification_by_transition.csv'
    qualification_text = table(pd.read_csv(qualification_path).set_index('season')) if qualification_path.exists() else 'Transition audit unavailable for this input.'
    summaries.to_csv(ROOT/'results/metrics/feature_group_summary.csv')
    coverage.to_csv(ROOT/'results/metrics/statcast_coverage.csv')
    top.to_csv(ROOT/'results/metrics/largest_breakouts.csv', index=False)
    rounded_labels = ((data.next_season_wrc_plus.round() - data.wrc_plus.round()) >= 20).astype(int)
    precision_differences = int((rounded_labels != data.breakout).sum())
    text = f'''# Week 1 summary

Total observations: {len(data)}
Unique hitters: {data.player_id.nunique()}
Breakout observations: {int(data.breakout.sum())}
Distinct hitters with at least one breakout: {data.loc[data.breakout == 1, "player_id"].nunique()}
Non-breakouts: {int((data.breakout == 0).sum())}
Breakout percentage: {data.breakout.mean()*100:.2f}%

## By starting season
{table(season)}

Seasons with zero qualifying observations have an undefined rate (NA), not a zero breakout rate.

## By age group
{table(age)}
Unknown age observations: {data.age.isna().sum()}

## Missing predictors
{table(missing)}

## Statcast coverage (% nonmissing per feature season)
{table(coverage)}

## Largest observed breakouts
{table(top.set_index('player_name'))}

## Exploratory relationships
{table(relationships)}

Correlations are descriptive associations, not predictive importance or causation.
Missing values use pairwise complete observations. Group summaries are in feature_group_summary.csv.

## Identity merge quality (player-season records)
{quality_text}

## Qualification by transition and the 2020 season
{qualification_text}

The primary threshold remains 300 PA in both seasons. In the collected data, the
2020 maximum is 267 PA, so both transitions involving 2020 have zero qualifying
observations. They are excluded by the stated rule, never filled or labeled negative.
No prorated sensitivity dataset was mixed into the primary dataset.

## Sources and precision
FanGraphs current public MLB leaderboard JSON provides actual wRC+, traditional
statistics, FanGraphs IDs, and MLBAM IDs. The supplied IDs were cross-checked
against the Chadwick Baseball Bureau register. Baseball Savant season-level
leaderboards provide Statcast statistics, joined by MLBAM ID and season.
FanGraphs API wRC+ values retain their unrounded precision for both features and
labels. Rounding both season wRC+ values to integers first would produce
{int(rounded_labels.sum())} positives and change {precision_differences} labels; the primary
dataset uses the unrounded values consistently. Manual exports with rounded wRC+
may therefore produce different labels. Full-precision metrics CSVs accompany the
rounded Markdown tables.
All rates are fractions; exit velocity is mph and launch angle is degrees.
Cached raw responses and source URL/hash metadata preserve provenance.

## Quality and limitations
Both seasons must have 300 PA: this is a retrospective qualifying population,
not a fully realistic preseason population. No missing outcomes are labeled negative.
The 2020 rule eliminates both transitions involving 2020; the dataset spans
feature seasons 2017–2024 but contains rows in only six starting seasons. Repeated hitters are not independent observations.
Regression to the mean can contribute to associations with starting wRC+.
Unmapped IDs are excluded, missing predictors are retained without imputation,
and unavailable statistics are never replaced with a different metric.
See collection_report.json for exact source columns, unmapped IDs, and failures.
No models have been trained. Next-season fields and targets are excluded from FEATURE_COLUMNS.
'''
    (ROOT/'results/metrics/week1_summary.md').write_text(text)
    return {'season': season, 'age': age, 'missing': missing, 'coverage': coverage,
            'top': top, 'relationships': relationships, 'group_summary': summaries}


def main():
    path = ROOT/'data/processed/mlb_breakout_dataset.csv'
    if not path.exists():
        report = ROOT/'results/metrics/collection_report.json'
        detail = report.read_text() if report.exists() else 'Collection has not succeeded or no report exists.'
        raw_rows = []
        for year in range(2017, 2026):
            for source in ['exit_velocity', 'expected']:
                raw = ROOT/f'data/raw/{source}_{year}.csv'
                if raw.exists():
                    frame = pd.read_csv(raw)
                    raw_rows.append({'season': year, 'source': source, 'rows': len(frame),
                                     'unique_player_ids': frame.player_id.nunique()})
        raw_text = table(pd.DataFrame(raw_rows).set_index(['season', 'source'])) if raw_rows else 'No cached Savant data.'
        (ROOT/'results/metrics/week1_summary.md').write_text(
            '# Week 1 summary — collection incomplete\n\nNo labeled dataset is available. '
            'Observation count, class balance, age rates, Statcast coverage, and breakout examples '
            'cannot be calculated. No values were fabricated.\n\n'
            'Run the commands in README.md to retry collection, build, validation, and exploration.\n\n'
            '## Cached Savant records (not qualifying labeled observations)\n'+raw_text+'\n\n'
            'Inspect collection_report.json and fangraphs_diagnosis.json for current diagnostics. '
            'The current endpoint and manual CSV import are documented in README.md.\n\n'
            '## Completion checklist\n'
            '- [x] Collection/cleaning/build/validation implementation\n'
            '- [x] Savant 2017–2025 downloaded and schemas verified\n'
            '- [x] Offline tests and notebook execution\n'
            '- [x] README and honest failure summary\n'
            '- [ ] FanGraphs 2017–2025 collected\n'
            '- [ ] Live ID crosswalk and merged seasons verified\n'
            '- [ ] Actual qualifying pairs, labels, and processed CSV\n'
            '- [ ] Actual class balance, missingness, age rates, figures, and breakout examples\n\n'
            '## Collection diagnostics\n```json\n'+detail+'\n```\n')
        print('No processed dataset; wrote collection failure summary.')
        return
    result = explore(pd.read_csv(path))
    print(result['season'].to_string())

if __name__ == '__main__':
    main()

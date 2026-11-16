# MLB Breakout Hitter Predictor

Graduate machine learning project at Illinois Institute of Technology. Predict whether a hitter improves their following-season **FanGraphs wRC+ by at least 20 points**. Week 1 covers collection, cleaning, consecutive-season labels, validation, and exploratory analysis. No models have been trained.

Each observation uses season t features and season t+1 outcomes. Both seasons require **at least 300 PA**. Missing outcomes are excluded, never labeled negative. This selects a retrospective qualifying population rather than a fully realistic preseason population, because future playing time is unknown.

## Current results

Week 1 is complete using actual data for 2017–2025:

| Measure | Result |
| --- | --- |
| Qualifying player-season observations | 1,206 |
| Unique hitters | 495 |
| Breakout observations | 188 (15.59%) |
| Unique hitters with at least one breakout | 161 |
| Non-breakout observations | 1,018 |
| Missing predictors in the labeled dataset | 0 |
| Statcast coverage in the labeled dataset | 100% for each of six features |
| Hitter-season ID mappings independently verified against Chadwick | 7,240 |
| Unmatched / ambiguous / conflicting mappings | 0 / 0 / 0 |
| Multi-team player-season aggregates retained | 717 |

There are observations in six feature seasons within 2017–2024. **2019→2020 and 2020→2021 have zero qualifying pairs**, since the maximum PA in the 60-game 2020 season is 267. Before the PA filter those transitions have 486 and 497 consecutive players, respectively. Their breakout percentages are undefined, not zero. The 300-PA rule was preserved; no prorated records were mixed into the primary dataset.

See [Week 1 summary](results/metrics/week1_summary.md), [collection report](results/metrics/collection_report.json), [merge-quality report](results/metrics/merge_quality_report.json), and [transition qualification counts](results/metrics/qualification_by_transition.csv).

## Reproduce

Use Python 3.11+ (verified with Python 3.12). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.collect_data --start-year 2017 --end-year 2025
python -m src.build_dataset
python -m src.validate_data
python -m src.explore_data
pytest -q
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/01_data_exploration.ipynb
```

The current checkout has cached data, so collection can rerun without network access. A clean checkout needs ordinary network access to FanGraphs, Baseball Savant, and the public Chadwick register. The collector saves raw responses in data/raw (ignored by Git), normalized seasons in data/processed, and collection/merge diagnostics in results/metrics. `--refresh` refreshes FanGraphs only; successful Savant and crosswalk caches are preserved. Remove an individual Savant cache explicitly only if you intentionally need to refresh that source.

All requests use bounded timeouts. Transient connection/timeouts and server errors use exponential backoff, up to `--attempts` (default 3). Empty responses and schema failures are not retried. HTTP 401, 403, and 429 stop further requests to that host in the run. For 429, Retry-After seconds or HTTP dates are respected via a persistent data/raw/http_cooldowns.json ledger; a 60-second default applies if absent or invalid. The collector does not wait in a retry loop, change identities, use alternate hosts, or bypass challenges. Retry manually after the recorded cooldown if necessary. The builder refuses incomplete season ranges.

## FanGraphs failure diagnosis and repair

The original `pybaseball.batting_stats(year, year, qual=0, ind=1)` uses `/leaders-legacy.aspx` and expects a legacy `rgMasterTable` HTML table. A diagnostic request confirmed **HTTP 308 redirects that URL to `/leaders/major-league`**, which returns HTTP 200 without that table. The HTML parser therefore produces an empty dataframe. The earlier HTTP 429 for 2025 was a separate rate-limit response; retrying it was not a schema fix.

PyPI's latest release was still **2.2.7** when checked, and upstream master still used the legacy adapter. Installing a newer supported release therefore offered no confirmed repair. Evidence is preserved in [fangraphs_diagnosis.json](results/metrics/fangraphs_diagnosis.json) and the original failure report in collection_report_initial_failure.json.

The repaired default uses the public JSON backing the current FanGraphs MLB leaderboard:

`https://www.fangraphs.com/api/leaders/major-league/data`

It requests MLB batting, all leagues/positions/teams, minimum 0 PA, individual seasons 2017–2025, full seasons, no monthly/roster filters, and 20,000 rows. It checks `totalCount` against returned rows and verifies every requested season. The successful single request returned 12,820 records; 5,580 zero-PA records have no wRC+ and are explicitly excluded, leaving 7,240 hitter-seasons. Traded players already appear as all-team aggregates (`2 Tms` etc.), with no duplicate player-season keys. No rate metrics are averaged or summed across team splits.

The endpoint supplies actual FanGraphs wRC+ and all requested traditional fields, plus FanGraphs and MLBAM IDs. Plain-text `PlayerName` and `TeamNameAbb` are used instead of HTML anchor display fields. The original JSON response, request URL, timestamp, and SHA-256 hash are cached. This is a public website endpoint, not a guaranteed versioned API contract; future schema or access changes can still require manual exports. The legacy adapter remains available explicitly with `--fg-source pybaseball` for diagnostics, subject to the same access protections.

## Manual CSV fallback

If public automated access becomes unavailable, use FanGraphs' own [leaderboard Export feature](https://library.fangraphs.com/how-to-use-fangraphs-leaderboards/) through your normal authorized browser session. The importer supports a combined multi-season CSV, multiple single-season CSVs containing a Season column, or a directory of such CSVs.

Set the [MLB major-league batting leaderboard](https://www.fangraphs.com/leaders/major-league) as follows:

1. Actual historical regular-season batting statistics, not projections, postseason, minor leagues, or team totals.
2. Seasons **2017 through 2025**, **individual season rows** enabled. Alternatively export each season separately.
3. All leagues, positions, and teams; include inactive players; minimum PA **0** (do not select the default qualified-only threshold). No month, handedness, or situational splits.
4. Player-season aggregates across all teams. Do not export separate team-stint rows; duplicate player-season records fail validation.
5. Include FanGraphs player ID (`playerid` or `IDfg`), Name, Season, Age, PA, **wRC+**, wOBA, ISO, BABIP, BB%, K%, O-Swing%, Contact%, and SwStr%. Team and MLBAMID are useful but optional. Player IDs are included in FanGraphs exports; never reconstruct them from names. Missing optional statistics are reported and retained as null. ID, Name, PA, and wRC+ are mandatory; Season must be present or supplied explicitly for a single-season import.
6. Export the full result set; do not copy only the visible page. Keep the original files unchanged. Custom-column exports may be necessary for plate-discipline statistics.

```bash
# One combined file with Season present:
python -m src.collect_data --import-fangraphs /absolute/path/all_seasons.csv

# One CSV per season, each with its own Season column:
python -m src.collect_data --import-fangraphs /absolute/path/exports_directory
# Or pass all nine filenames after --import-fangraphs.

# An individual file lacking Season, imported explicitly as 2024:
python -m src.collect_data --start-year 2024 --end-year 2024 \
  --import-fangraphs /absolute/path/2024.csv --import-season 2024
```

After importing all nine seasons, run the default builder, validator, exploration, and notebook commands above. Import mode does not contact FanGraphs; uncached Savant/register data may still need network access. Source paths and hashes are recorded. Overlapping duplicate records across files fail validation. The importer accepts aliases for common player ID column names and explicit percent strings such as `10%`. Bare numeric rate values default to fractions; use `--fg-rate-units percent` only for an export that stores numeric percentages such as `10.0`. Units are never guessed from maxima. Do not apply that option to an ordinary fraction export.

## Sources, identities, and units

- [FanGraphs current leaderboard](https://www.fangraphs.com/leaders/major-league): actual traditional season aggregates and wRC+.
- [pybaseball Statcast functions](https://github.com/jldbc/pybaseball/blob/master/pybaseball/statcast_batter.py): season aggregates from `statcast_batter_exitvelo_barrels(year, minBBE=0)` and `statcast_batter_expected_stats(year, minPA=0)`; no pitch-level downloads.
- [Chadwick Baseball Bureau register](https://github.com/chadwickbureau/register): independent FanGraphs-to-MLBAM crosswalk, downloaded through supported `pybaseball.playerid_reverse_lookup`. Chadwick data is made available under the Open Data Commons Attribution License. In this run all 7,240 source ID pairs agree with the register. Names are display metadata, never the sole join key. Ambiguous mappings and source/register conflicts fail with a report; absent mappings are reported and excluded. Direct source IDs not in the register are explicitly counted rather than described as independently verified.

Each Statcast source is joined to one normalized hitter season by MLBAM ID. Duplicate source keys fail validation. Reports record matched/unmatched counts separately for identity mapping and each Statcast table; all are fully matched in this run. A matched row does not imply every field is present: field-level missingness is also reported.

| Normalized field | Verified source column | Units |
| --- | --- | --- |
| wrc_plus, woba, iso, babip | wRC+, wOBA, ISO, BABIP | Source index/decimal |
| bb_rate, k_rate | BB%, K% | Fraction |
| chase_rate, contact_rate, swstr_rate | O-Swing%, Contact%, SwStr% | Fraction |
| avg_exit_velocity | avg_hit_speed | mph |
| avg_launch_angle | avg_hit_angle | Degrees |
| hard_hit_rate, barrel_rate | ev95percent, brl_percent | Savant percentage divided by 100 |
| xba, xwoba | est_ba, est_woba | Decimal |

Barrel rate uses batted-ball events, not PA (`brl_pa` is not substituted). All rates must lie in [0, 1]. All requested source fields were present. Every predictor is nonmissing in the qualifying dataset; unqualified raw/normalized records can have missing Statcast values.

**Precision:** the primary dataset retains unrounded wRC+ from FanGraphs' JSON and applies `next_wrc_plus - wrc_plus >= 20` without rounding. Rounding both seasons to integers first would give 191 positives instead of 188, changing three boundary labels. A rounded manual export can therefore yield slightly different results; retain the original response for exact reproduction. Markdown tables round for readability; CSVs retain full precision. No proxy such as OPS+ is used.

## Artifacts and leakage controls

`src/build_dataset.py` joins by MLBAM ID and exactly season+1, filters both PA values, and calculates the label. The primary dataset is `data/processed/mlb_breakout_dataset.csv`. The explicit `FEATURE_COLUMNS` allowlist and `TARGET_COLUMNS` exclusions live in src/validate_data.py; the builder writes feature_columns.json. IDs, names, team, season, next-season PA/wRC+, change, and label are excluded from modeling inputs. All retained predictors are season t values.

The notebook and `python -m src.explore_data` share EDA code. Seven figures cover class balance, seasonal rates, age groups, improvement distribution with the +20 threshold, feature distributions, predictor correlations, and relationships with improvement. Summary CSVs include group statistics, missingness, Statcast coverage, top 15 actual improvements, and correlations. Correlations use pairwise complete observations and describe associations, not causal effects or predictive importance. Repeated hitters and regression to the mean limit interpretation.

The offline test suite covers qualification boundaries, labels, consecutive seasons, duplicates, missing outcomes, feature leakage, identity conflicts, multi-team duplicate rejection, CSV imports/units, complete JSON responses, 429 cooldowns, transient backoff, cache reuse, the 2020 effect, and all EDA artifacts. Synthetic data is used only in tests, never in reported findings.

## Planned Week 2

Compare traditional and Statcast feature sets using season-based training/validation splits and class-imbalance-aware baselines. Fit preprocessing only on training seasons and exclude every target/evaluation field. Week 2 modeling is not part of this implementation.

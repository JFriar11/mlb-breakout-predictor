# Week 1 completion execution record

Executed on 2026-11-16 using Python 3.12 and the existing .venv. The initial failed run is retained in collection_report_initial_failure.json; it has been superseded by the successful current collection report.

## Diagnosis and live access

- Checked PyPI metadata: latest supported release remained pybaseball 2.2.7.
- Checked upstream master: the FanGraphs adapter still requests /leaders-legacy.aspx and expects rgMasterTable HTML.
- One legacy-page diagnostic confirmed HTTP 308 to the current leaderboard, returning HTTP 200 with no legacy table. This explains the empty dataframe independently of the earlier HTTP 429 rate limit.
- One ordinary GET to the current public /api/leaders/major-league/data endpoint succeeded with HTTP 200, 12,820 rows, all nine requested seasons, and totalCount matching the response. The complete response was preserved as the raw cache; no repeat download was needed.
- Downloaded the public Chadwick register once through pybaseball.playerid_reverse_lookup and cached the crosswalk.
- No authentication, rate-limit circumvention, browser impersonation, or alternate access route was used. The repaired code stops a host immediately on 401/403/429 and persists Retry-After cooldowns.

## Final commands and results

| Command | Outcome |
| --- | --- |
| `python -m src.collect_data --start-year 2017 --end-year 2025` | Success using the cached successful current FanGraphs JSON and all 18 existing Savant files; normalized 7,240 hitter-seasons |
| `python -m src.build_dataset` | Success: 1,206 qualifying observations, 188 positives, 15.59% breakout rate |
| `python -m src.validate_data` | Success |
| `python -m src.explore_data` | Success: seven real-data figures, summary and analysis CSVs |
| `pytest -q` | 39 passed; nine upstream seaborn/matplotlib deprecation warnings |
| `jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb` | Success with local Jupyter kernel port permission; real-data outputs saved |
| `python -m compileall -q src tests` | Success |
| `python -m pip check` | No broken requirements |
| `git diff --check` | Success; files are still uncommitted |

The actual shell used .venv/bin/python and .venv/bin/jupyter. MPLCONFIGDIR=/private/tmp/mlb-breakout-mpl was set for non-notebook plotting/tests because the home matplotlib cache directory is outside writable roots.

All 18 Savant cache files were SHA-256 checked before and after final collection: byte-for-byte unchanged. No Week 2 model training was performed. The full notebook was validated against the notebook schema and executed without cell errors. A generated feature-distribution figure was visually inspected.

## Outcomes and precision

- 495 unique hitters; 188 breakout observations across 161 distinct hitters; 1,018 non-breakout observations.
- 7,240 identity mappings agree with Chadwick, with zero unmatched, ambiguous, or conflicting records; both Savant sources match all hitter-season rows. There are 717 multi-team player-season aggregates.
- Zero missing predictors in the qualifying dataset; all six Statcast predictors have 100% coverage.
- 2019→2020 and 2020→2021 each have zero qualifying pairs. Maximum 2020 PA is 267; before qualification there are 486 and 497 consecutive-player pairs, respectively.
- Unrounded API wRC+ is used consistently. Integer rounding before label calculation would give 191 positives instead of 188, changing three labels. This is documented rather than silently changing the definition or precision.

## Reproduce

See README.md for exact setup, default collection/build/validation/exploration/test/notebook commands, source provenance, limitations, and the tested manual CSV fallback. No further data or approval is required from the user to use the completed local dataset.

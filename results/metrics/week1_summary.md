# Week 1 summary

Total observations: 1206
Unique hitters: 495
Breakout observations: 188
Distinct hitters with at least one breakout: 161
Non-breakouts: 1018
Breakout percentage: 15.59%

## By starting season
| season | observations | breakouts | breakout_percent |
| --- | --- | --- | --- |
| 2017 | 219 | 39 | 17.8082 |
| 2018 | 187 | 29 | 15.508 |
| 2019 | 0 | 0 | NA |
| 2020 | 0 | 0 | NA |
| 2021 | 192 | 35 | 18.2292 |
| 2022 | 202 | 25 | 12.3762 |
| 2023 | 207 | 36 | 17.3913 |
| 2024 | 199 | 24 | 12.0603 |

Seasons with zero qualifying observations have an undefined rate (NA), not a zero breakout rate.

## By age group
| age | observations | breakout_percent |
| --- | --- | --- |
| 24 and younger | 204 | 23.5294 |
| 25–29 | 620 | 15.1613 |
| 30 and older | 382 | 12.0419 |
Unknown age observations: 0

## Missing predictors
| index | missing | missing_percent |
| --- | --- | --- |
| age | 0 | 0.0 |
| pa | 0 | 0.0 |
| wrc_plus | 0 | 0.0 |
| woba | 0 | 0.0 |
| iso | 0 | 0.0 |
| babip | 0 | 0.0 |
| bb_rate | 0 | 0.0 |
| k_rate | 0 | 0.0 |
| chase_rate | 0 | 0.0 |
| contact_rate | 0 | 0.0 |
| swstr_rate | 0 | 0.0 |
| avg_exit_velocity | 0 | 0.0 |
| hard_hit_rate | 0 | 0.0 |
| barrel_rate | 0 | 0.0 |
| avg_launch_angle | 0 | 0.0 |
| xba | 0 | 0.0 |
| xwoba | 0 | 0.0 |

## Statcast coverage (% nonmissing per feature season)
| season | avg_exit_velocity | hard_hit_rate | barrel_rate | avg_launch_angle | xba | xwoba |
| --- | --- | --- | --- | --- | --- | --- |
| 2017 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2018 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2019 | NA | NA | NA | NA | NA | NA |
| 2020 | NA | NA | NA | NA | NA | NA |
| 2021 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2022 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2023 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 2024 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## Largest observed breakouts
| player_name | season | wrc_plus | next_season_wrc_plus | wrc_plus_change | pa | next_season_pa |
| --- | --- | --- | --- | --- | --- | --- |
| Mookie Betts | 2017 | 106.6364 | 184.997 | 78.3606 | 712.0 | 614.0 |
| George Springer | 2024 | 94.3246 | 166.0032 | 71.6786 | 614.0 | 586.0 |
| Bo Bichette | 2024 | 69.9248 | 133.6164 | 63.6916 | 336.0 | 628.0 |
| Jurickson Profar | 2023 | 77.5205 | 139.25 | 61.7294 | 521.0 | 668.0 |
| Carlos Correa | 2023 | 94.074 | 154.6177 | 60.5437 | 580.0 | 367.0 |
| Aaron Judge | 2021 | 149.9826 | 206.1548 | 56.1722 | 633.0 | 696.0 |
| Ronald Acuña Jr. | 2022 | 115.3872 | 171.4487 | 56.0615 | 533.0 | 735.0 |
| Corey Seager | 2022 | 116.5733 | 172.5218 | 55.9484 | 663.0 | 536.0 |
| Brenton Doyle | 2023 | 44.2823 | 97.7749 | 53.4925 | 431.0 | 603.0 |
| Bobby Witt Jr. | 2023 | 115.849 | 169.0945 | 53.2456 | 694.0 | 709.0 |
| Cody Bellinger | 2022 | 83.1713 | 135.2134 | 52.0421 | 550.0 | 556.0 |
| Mitch Garver | 2018 | 103.5552 | 155.3043 | 51.7491 | 335.0 | 359.0 |
| James McCann | 2018 | 56.0862 | 107.5572 | 51.471 | 457.0 | 476.0 |
| Marcell Ozuna | 2022 | 89.6651 | 140.1763 | 50.5112 | 507.0 | 592.0 |
| Christian Yelich | 2017 | 117.0107 | 167.0841 | 50.0734 | 695.0 | 651.0 |

## Exploratory relationships
| index | Pearson correlation with change |
| --- | --- |
| wrc_plus | -0.4538 |
| woba | -0.4456 |
| babip | -0.3223 |
| iso | -0.2788 |
| xwoba | -0.2641 |
| xba | -0.2201 |
| barrel_rate | -0.1542 |
| age | -0.1058 |
| bb_rate | -0.1054 |
| hard_hit_rate | -0.1037 |
| avg_exit_velocity | -0.0978 |
| pa | -0.0363 |
| swstr_rate | -0.0288 |
| avg_launch_angle | -0.0267 |
| chase_rate | 0.0035 |
| k_rate | 0.024 |
| contact_rate | 0.0371 |

Correlations are descriptive associations, not predictive importance or causation.
Missing values use pairwise complete observations. Group summaries are in feature_group_summary.csv.

## Identity merge quality (player-season records)
| season | input_records | matched_records | unmatched_records | ambiguous_records | register_verified_records | multi_team_records |
| --- | --- | --- | --- | --- | --- | --- |
| 2017 | 957 | 957 | 0 | 0 | 957 | 80 |
| 2018 | 990 | 990 | 0 | 0 | 990 | 103 |
| 2019 | 990 | 990 | 0 | 0 | 990 | 100 |
| 2020 | 581 | 581 | 0 | 0 | 581 | 34 |
| 2021 | 1049 | 1049 | 0 | 0 | 1049 | 125 |
| 2022 | 693 | 693 | 0 | 0 | 693 | 80 |
| 2023 | 656 | 656 | 0 | 0 | 656 | 51 |
| 2024 | 651 | 651 | 0 | 0 | 651 | 67 |
| 2025 | 673 | 673 | 0 | 0 | 673 | 77 |

## Qualification by transition and the 2020 season
| season | next_season | consecutive_players | current_300_pa | next_300_pa | current_max_pa | next_max_pa | missing_outcome_or_pa | qualifying_pairs | excluded_by_300_pa_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017 | 2018 | 694 | 287 | 278 | 725.0 | 745.0 | 0 | 219 | 475 |
| 2018 | 2019 | 734 | 278 | 273 | 745.0 | 747.0 | 0 | 187 | 547 |
| 2019 | 2020 | 486 | 273 | 0 | 747.0 | 267.0 | 0 | 0 | 486 |
| 2020 | 2021 | 497 | 0 | 262 | 267.0 | 724.0 | 0 | 0 | 497 |
| 2021 | 2022 | 515 | 262 | 277 | 724.0 | 724.0 | 0 | 192 | 323 |
| 2022 | 2023 | 528 | 277 | 293 | 724.0 | 753.0 | 0 | 202 | 326 |
| 2023 | 2024 | 515 | 293 | 286 | 753.0 | 735.0 | 0 | 207 | 308 |
| 2024 | 2025 | 536 | 286 | 277 | 735.0 | 732.0 | 0 | 199 | 337 |

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
191 positives and change 3 labels; the primary
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

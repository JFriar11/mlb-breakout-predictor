"""Historical classification errors and season-specific scouting shortlists."""
import numpy as np
import pandas as pd

from .evaluate_models import rank_predictions
from .split_data import KEYS
from .train_models import THRESHOLD

CONTEXT = ["babip", "bb_rate", "k_rate", "chase_rate", "contact_rate", "avg_exit_velocity",
           "hard_hit_rate", "barrel_rate", "xwoba", "woba"]


def analyze_errors(predictions, history):
    ranked = rank_predictions(predictions)
    predicted = ranked.predicted_probability.ge(THRESHOLD)
    actual = ranked.breakout.eq(1)
    ranked["predicted_class"] = predicted.astype(int)
    ranked["error_category"] = np.select(
        [predicted & actual, predicted & ~actual, ~predicted & actual],
        ["true_positive", "false_positive", "false_negative"], default="true_negative",
    )
    ranked["season_rank"] = ranked.groupby(["experiment_id", "season"]).cumcount() + 1
    top = ranked.season_rank.le(20)
    ranked["ranking_category"] = np.select(
        [top & actual, top & ~actual, ~top & actual],
        ["top20_hit", "top20_non_breakout", "breakout_outside_top20"], default="outside_top20",
    )
    return ranked.merge(history[KEYS + CONTEXT], on=KEYS, validate="many_to_one")


def case_studies(errors, selected):
    primary = errors.loc[errors.experiment_id.eq(selected)]
    lines = ["# Historical hitter case studies", "",
             f"Cases use out-of-fold probabilities from {selected} unless a different model is shown. "
             "Classification uses a 0.50 cutoff. Rates and expected stats describe the feature season.", ""]
    for category in ("false_positive", "false_negative", "true_positive"):
        candidates = primary.loc[primary.error_category.eq(category)]
        highest_first = category != "false_negative"
        cases = candidates.sort_values("predicted_probability", ascending=not highest_first,
                                       kind="stable").head(3)
        lines.extend([f"## {category.replace('_', ' ').title()}", "",
                      f"{selected} produced {len(candidates)} cases in this category."])
        if len(cases) < 3:
            alternate = errors.loc[errors.error_category.eq(category)
                                   & ~errors.experiment_id.eq(selected)]
            alternate = alternate.loc[~pd.MultiIndex.from_frame(alternate[KEYS]).isin(
                pd.MultiIndex.from_frame(cases[KEYS])
            )].sort_values("predicted_probability", ascending=not highest_first, kind="stable")
            cases = pd.concat([cases, alternate.drop_duplicates(KEYS).head(3 - len(cases))])
            lines.append("Additional examples use other historical models at the same cutoff.")
        lines.append("")
        for row in cases.itertuples():
            lines.append(
                f"- **{row.player_name} ({row.season}, age {row.age:g}, {row.experiment_id})** had "
                f"a {row.predicted_probability:.1%} score and moved from {row.wrc_plus:.1f} to "
                f"{row.next_season_wrc_plus:.1f} wRC+ ({row.wrc_plus_change:+.1f}). "
                f"{interpret_case(row)}"
            )
        lines.append("")
    lines.extend(["## Per-season top 20", ""])
    for season, group in primary.groupby("season"):
        hits = group.ranking_category.eq("top20_hit").sum()
        non_breakouts = group.loc[group.ranking_category.eq("top20_non_breakout")]
        missed = group.loc[group.ranking_category.eq("breakout_outside_top20")].nlargest(
            1, "wrc_plus_change"
        )
        line = f"- **{season}** yielded {hits}/{min(20, len(group))} hits."
        if not non_breakouts.empty:
            row = non_breakouts.iloc[0]
            line += (f" The highest-ranked non-breakout, {row.player_name}, ranked {row.season_rank} "
                     f"with a {row.wrc_plus_change:+.1f} change.")
        if not missed.empty:
            row = missed.iloc[0]
            line += f" {row.player_name} gained {row.wrc_plus_change:+.1f} from rank {row.season_rank}."
        lines.append(line)
    return "\n".join(lines) + "\n"


def interpret_case(row):
    gap = row.xwoba - row.woba
    if abs(gap) >= .020:
        direction = "above" if gap > 0 else "below"
        return (f"Starting xwOBA ({row.xwoba:.3f}) sat {abs(gap):.3f} {direction} observed "
                f"wOBA, indicating a gap between expected and observed production.")
    if row.k_rate >= .25:
        return (f"A {row.k_rate:.1%} strikeout rate limited balls in play, alongside "
                f"a {row.barrel_rate:.1%} barrel rate on batted balls.")
    if row.babip >= .330:
        return (f"A .{round(row.babip * 1000):03d} starting BABIP supported observed "
                f"production. The feature-season xwOBA was {row.xwoba:.3f}.")
    return (f"The starting contact rate was {row.contact_rate:.1%} and hard-hit rate "
            f"was {row.hard_hit_rate:.1%}, describing bat-to-ball frequency and contact quality.")

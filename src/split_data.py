"""Primary chronological player-season split and reproducibility audit."""
import pandas as pd

from .validate_data import ROOT, unique_keys

DATASET = ROOT / "data/processed/mlb_breakout_dataset.csv"
KEYS = ["player_id", "season"]
SPLIT_SEASONS = {
    "train": (2017, 2018, 2021, 2022),
    "validation": (2023,),
    "test": (2024,),
}


def split_data(data):
    missing = set(KEYS + ["next_season", "breakout"]) - set(data)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    unique_keys(data, KEYS)
    expected = {season for seasons in SPLIT_SEASONS.values() for season in seasons}
    if set(data.season) != expected:
        raise ValueError(f"Expected feature seasons {sorted(expected)}")
    if not data.next_season.eq(data.season + 1).all():
        raise ValueError("Every outcome must be from the immediately following season")

    ordered = data.sort_values(["season", "player_id"])
    partitions = {}
    assigned = set()
    previous_outcome = None
    for name, seasons in SPLIT_SEASONS.items():
        part = ordered.loc[ordered.season.isin(seasons)].reset_index(drop=True)
        keys = set(part[KEYS].itertuples(index=False, name=None))
        if assigned.intersection(keys):
            raise ValueError("Overlapping player-season partitions")
        assigned.update(keys)
        if previous_outcome is not None and previous_outcome >= part.next_season.min():
            raise ValueError("Outcome seasons must be strictly ordered across partitions")
        previous_outcome = part.next_season.max()
        if set(part.breakout) != {0, 1}:
            raise ValueError(f"{name} requires both binary classes")
        partitions[name] = part
    if len(assigned) != len(data):
        raise ValueError("Every observation must be assigned exactly once")
    return partitions


def split_audit(partitions):
    summary = []
    memberships = []
    for name, part in partitions.items():
        summary.append({
            "split": name,
            "feature_seasons": ", ".join(map(str, part.season.unique())),
            "outcome_seasons": ", ".join(map(str, part.next_season.unique())),
            "observations": len(part),
            "positives": int(part.breakout.sum()),
            "prevalence": float(part.breakout.mean()),
        })
        memberships.append(part[KEYS + ["next_season"]].assign(split=name))
    return pd.DataFrame(summary), pd.concat(memberships, ignore_index=True)

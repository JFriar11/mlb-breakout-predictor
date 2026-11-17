"""Week 2 predictor groups drawn from the Week 1 allowlist."""
from pandas.api.types import is_numeric_dtype

from .validate_data import FEATURE_COLUMNS, STATCAST, TARGET_COLUMNS, TRADITIONAL

BASELINE_FEATURES = ("age", "wrc_plus")
TRADITIONAL_FEATURES = ("age", "pa", *TRADITIONAL)
STATCAST_FEATURES = tuple(STATCAST)
FEATURE_SETS = {
    "baseline": BASELINE_FEATURES,
    "traditional": TRADITIONAL_FEATURES,
    "traditional_statcast": (*TRADITIONAL_FEATURES, *STATCAST_FEATURES),
}


def predictors(frame, columns):
    """Reject unauthorized features at every fit and prediction boundary."""
    forbidden = set(columns) - set(FEATURE_COLUMNS)
    forbidden.update(c for c in columns if c in TARGET_COLUMNS or c.startswith("next_"))
    if forbidden:
        raise ValueError(f"Predictors outside the season-t allowlist: {sorted(forbidden)}")
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing predictors: {sorted(missing)}")
    nonnumeric = [c for c in columns if not is_numeric_dtype(frame[c])]
    if nonnumeric:
        raise ValueError(f"Nonnumeric predictors: {nonnumeric}")
    return frame.loc[:, list(columns)]

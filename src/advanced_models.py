"""Fixed Week 3 model configurations using the Week 2 feature allowlist."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .feature_sets import FEATURE_SETS, predictors
from .train_models import EXPERIMENTS as WEEK2_EXPERIMENTS, SEED, make_pipeline

EXPERIMENTS = {
    **WEEK2_EXPERIMENTS,
    "M6": ("HistGradientBoosting", "traditional"),
    "M7": ("HistGradientBoosting", "traditional_statcast"),
    "M8": ("XGBoost", "traditional"),
    "M9": ("XGBoost", "traditional_statcast"),
}
STATCAST_PAIRS = [("M2", "M3"), ("M4", "M5"), ("M6", "M7"), ("M8", "M9")]


def make_experiment(experiment):
    name, _ = EXPERIMENTS[experiment]
    if experiment in WEEK2_EXPERIMENTS:
        return make_pipeline(name)
    if name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=SEED,
        )
    else:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=2, tree_method="hist",
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", model),
    ])


def available_experiments():
    try:
        make_experiment("M8")
    except (ImportError, OSError, ValueError) as error:
        return tuple(WEEK2_EXPERIMENTS) + ("M6", "M7"), str(error)
    return tuple(EXPERIMENTS), None


def features(frame, experiment):
    return predictors(frame, FEATURE_SETS[EXPERIMENTS[experiment][1]])

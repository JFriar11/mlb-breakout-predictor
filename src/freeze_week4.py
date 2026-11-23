"""Fit and freeze the two prespecified Week 4 pipelines using historical rows."""
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib

from .backtesting import HISTORY, load_history
from .feature_sets import FEATURE_SETS
from .split_data import DATASET, KEYS
from .train_models import EXPERIMENTS, fit_experiment, sha256, write_json
from .validate_data import ROOT

DESTINATION = ROOT / "models/week4"
PROTOCOL = ROOT / "results/metrics/week4_evaluation_protocol.md"
ROLES = {"M2": "primary", "M1": "comparison"}


def freeze(dataset=DATASET, destination=DESTINATION, protocol=PROTOCOL):
    if destination.exists():
        raise FileExistsError(f"Frozen directory already exists: {destination}")
    history = load_history(dataset).sort_values(["season", KEYS[0]])
    if tuple(history.season.unique()) != HISTORY or history.next_season.max() > 2024:
        raise ValueError("Freeze requires complete historical feature seasons and outcomes through 2024")
    manifest = {
        "protocol_sha256": sha256(protocol),
        "training_csv_sha256": hashlib.sha256(history.to_csv(index=False).encode()).hexdigest(),
        "training_feature_seasons": list(HISTORY),
        "latest_training_outcome_season": int(history.next_season.max()),
        "training_observations": len(history),
        "training_breakouts": int(history.breakout.sum()),
        "holdout_feature_season": 2024,
        "holdout_outcome_season": 2025,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in (
            "joblib", "numpy", "pandas", "scipy", "scikit-learn",
        )},
        "models": {},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging:
        # The staging directory lives on the same filesystem as its final destination.

        bundle = Path(staging) / destination.name
        bundle.mkdir()
        for model_id, role in ROLES.items():
            name, feature_set = EXPERIMENTS[model_id]
            pipeline = fit_experiment(history, model_id)
            filename = f"{model_id}.joblib"
            joblib.dump(pipeline, bundle / filename, compress=3)
            manifest["models"][model_id] = {
                "role": role,
                "model_name": name,
                "feature_set": feature_set,
                "predictors": FEATURE_SETS[feature_set],
                "pipeline": filename,
                "pipeline_sha256": sha256(bundle / filename),
            }
        write_json(bundle / "freeze_manifest.json", manifest)
        bundle.rename(destination)
    return manifest


if __name__ == "__main__":
    manifest = freeze()
    print(f"Froze M2 and M1 on {manifest['training_observations']:,} historical player-seasons.")

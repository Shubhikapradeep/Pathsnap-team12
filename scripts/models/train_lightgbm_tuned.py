"""Train PathSnap LightGBM using Bayesian-optimised parameters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lightgbm import LGBMClassifier
from pathsnap.evaluation import run_experiment


NAME = "lightgbm_tuned"
FAMILY = "gradient_boosting"

BEST_PARAMS_PATH = ROOT / "results" / "bayesian" / "best_params.json"

with BEST_PARAMS_PATH.open("r", encoding="utf-8") as file:
    result = json.load(file)

CONFIG = {
    **result["best_params"],
    "subsample_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def build_estimator():
    return LGBMClassifier(**CONFIG)


if __name__ == "__main__":
    print("Training Bayesian-optimised LightGBM...")
    print("Configuration:", CONFIG)

    run_experiment(
        name=NAME,
        family=FAMILY,
        config=CONFIG,
        estimator_factory=build_estimator,
        seed=42,
        max_train_rows=None,
    )

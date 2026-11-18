
"""Train and evaluate the LightGBM map-matching classifier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lightgbm import LGBMClassifier
from pathsnap.evaluation import run_experiment


NAME = "lightgbm"
FAMILY = "gradient_boosting"

CONFIG = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def build_estimator():
    return LGBMClassifier(**CONFIG)


if __name__ == "__main__":
    run_experiment(
        name=NAME,
        family=FAMILY,
        config=CONFIG,
        estimator_factory=build_estimator,
        seed=42,
        max_train_rows=None,
    )

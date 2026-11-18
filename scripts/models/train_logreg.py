"""Logistic Regression experiments for PathSnap candidate scoring.

Runs:
- logreg: C=1.0, max_iter=200
- logreg_c0p1: C=0.1, max_iter=200
- logreg_balanced: C=1.0, max_iter=200, class_weight='balanced'
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.evaluation import run_experiment

FAMILY = "linear"

EXPERIMENTS = [
    {
        "name": "logreg",
        "config": {"C": 1.0, "max_iter": 200, "class_weight": None, "random_state": 42},
    },
    {
        "name": "logreg_c0p1",
        "config": {"C": 0.1, "max_iter": 200, "class_weight": None, "random_state": 42},
    },
    {
        "name": "logreg_balanced",
        "config": {"C": 1.0, "max_iter": 200, "class_weight": "balanced", "random_state": 42},
    },
]


def build_estimator(config: dict):
    """Return a NEW unfitted estimator exposing fit / predict_proba."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config["C"],
            max_iter=config["max_iter"],
            class_weight=config.get("class_weight"),
            random_state=config.get("random_state", 42),
        ),
    )


def main() -> None:
    for exp in EXPERIMENTS:
        name = exp["name"]
        config = exp["config"]
        print(f"\n{'=' * 60}\nRunning experiment: {name}\n{'=' * 60}")
        run_experiment(
            name=name,
            family=FAMILY,
            config=config,
            estimator_factory=lambda cfg=config: build_estimator(cfg),
            seed=42,
            max_train_rows=None,
        )


if __name__ == "__main__":
    main()

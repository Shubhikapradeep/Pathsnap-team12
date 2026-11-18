
"""Train and evaluate the MLP map-matching classifier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.evaluation import run_experiment


NAME = "mlp"
FAMILY = "neural_network"

CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "alpha": 1e-4,
    "early_stopping": True,
    "max_iter": 100,
    "random_state": 42,
    "max_train_rows": 1_000_000,
}


def build_estimator():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=CONFIG["hidden_layer_sizes"],
            alpha=CONFIG["alpha"],
            early_stopping=CONFIG["early_stopping"],
            max_iter=CONFIG["max_iter"],
            random_state=CONFIG["random_state"],
        ),
    )


if __name__ == "__main__":
    run_experiment(
        NAME,
        FAMILY,
        CONFIG,
        build_estimator,
        seed=42,
        max_train_rows=CONFIG["max_train_rows"],
    )

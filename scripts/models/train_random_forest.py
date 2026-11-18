from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sklearn.ensemble import RandomForestClassifier
from pathsnap.evaluation import run_experiment

NAME = "random_forest"
FAMILY = "tree"
CONFIG = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
}

def build_estimator():
    return RandomForestClassifier(**CONFIG)

if __name__ == "__main__":
    run_experiment(
        NAME, FAMILY, CONFIG, build_estimator,
        seed=42, max_train_rows=800_000
    )

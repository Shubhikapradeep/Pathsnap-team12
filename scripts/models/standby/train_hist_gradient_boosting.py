"""Standby model: hist_gradient_boosting (boosting). CPU only, no GPU needed.

Run from the repo root:  .venv/bin/python scripts/models/standby/train_hist_gradient_boosting.py
Use only if a primary model fails, is too slow, or the team needs another comparison point.
Whoever runs it commits the script and its results under their own name.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from pathsnap.evaluation import run_experiment  # noqa: E402

NAME = "hist_gradient_boosting"
FAMILY = "boosting"
CONFIG = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31}
MAX_TRAIN_ROWS = 1_000_000


def build_estimator():
    """Return a NEW unfitted estimator exposing fit / predict_proba."""
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False, random_state=42)


if __name__ == "__main__":
    run_experiment(NAME, FAMILY, {**CONFIG, "max_train_rows": MAX_TRAIN_ROWS}, build_estimator,
                   seed=42, max_train_rows=MAX_TRAIN_ROWS)

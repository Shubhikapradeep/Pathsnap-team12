"""Standby model: decision_tree (tree). CPU only, no GPU needed.

Run from the repo root:  .venv/bin/python scripts/models/standby/train_decision_tree.py
Use only if a primary model fails, is too slow, or the team needs another comparison point.
Whoever runs it commits the script and its results under their own name.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from pathsnap.evaluation import run_experiment  # noqa: E402

NAME = "decision_tree"
FAMILY = "tree"
CONFIG = {"max_depth": 12, "min_samples_leaf": 20}
MAX_TRAIN_ROWS = None


def build_estimator():
    """Return a NEW unfitted estimator exposing fit / predict_proba."""
    return DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42)


if __name__ == "__main__":
    run_experiment(NAME, FAMILY, {**CONFIG, "max_train_rows": MAX_TRAIN_ROWS}, build_estimator,
                   seed=42, max_train_rows=MAX_TRAIN_ROWS)

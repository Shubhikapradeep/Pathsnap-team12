"""Standby model: xgboost (boosting). CPU only, no GPU needed.

Run from the repo root:  .venv/bin/python scripts/models/standby/train_xgboost.py
Use only if a primary model fails, is too slow, or the team needs another comparison point.
Whoever runs it commits the script and its results under their own name.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from xgboost import XGBClassifier  # noqa: E402

from pathsnap.evaluation import run_experiment  # noqa: E402

NAME = "xgboost"
FAMILY = "boosting"
CONFIG = {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "tree_method": "hist"}
MAX_TRAIN_ROWS = 1_000_000


def build_estimator():
    """Return a NEW unfitted estimator exposing fit / predict_proba."""
    return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42, eval_metric="logloss")


if __name__ == "__main__":
    run_experiment(NAME, FAMILY, {**CONFIG, "max_train_rows": MAX_TRAIN_ROWS}, build_estimator,
                   seed=42, max_train_rows=MAX_TRAIN_ROWS)

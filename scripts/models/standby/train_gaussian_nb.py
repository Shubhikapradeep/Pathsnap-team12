"""Standby model: gaussian_nb (linear and probabilistic). CPU only, no GPU needed.

Run from the repo root:  .venv/bin/python scripts/models/standby/train_gaussian_nb.py
Use only if a primary model fails, is too slow, or the team needs another comparison point.
Whoever runs it commits the script and its results under their own name.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from sklearn.naive_bayes import GaussianNB  # noqa: E402

from pathsnap.evaluation import run_experiment  # noqa: E402

NAME = "gaussian_nb"
FAMILY = "linear and probabilistic"
CONFIG = {"var_smoothing": 1e-9}
MAX_TRAIN_ROWS = None


def build_estimator():
    """Return a NEW unfitted estimator exposing fit / predict_proba."""
    return GaussianNB()


if __name__ == "__main__":
    run_experiment(NAME, FAMILY, {**CONFIG, "max_train_rows": MAX_TRAIN_ROWS}, build_estimator,
                   seed=42, max_train_rows=MAX_TRAIN_ROWS)

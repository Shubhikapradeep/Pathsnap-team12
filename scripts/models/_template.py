"""Copy this file to scripts/models/train_<model>.py and edit the three marked parts.

Run from the repo root:  .venv/bin/python scripts/models/train_<model>.py
Scale-sensitive models (linear, KNN, SVM, MLP) MUST be wrapped in a Pipeline with
StandardScaler so that scaling is fit inside each CV fold on train data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sklearn.linear_model import LogisticRegression  # noqa: E402  (1) your estimator import
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pathsnap.evaluation import run_experiment  # noqa: E402

NAME = "template_example"          # (2) unique short name, used for result/model filenames
FAMILY = "linear"                  # algorithm family for the report table
CONFIG = {"C": 1.0, "max_iter": 200}   # initial hyperparameters, as logged


def build_estimator():
    """(3) Return a NEW unfitted estimator exposing fit / predict_proba."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=CONFIG["C"], max_iter=CONFIG["max_iter"]))


if __name__ == "__main__":
    run_experiment(NAME, FAMILY, CONFIG, build_estimator, seed=42, max_train_rows=None)

"""Development stand-in classifier, used ONLY to build and tune the HMM decoder.

It is not one of the compared models and its numbers must not appear in the model
comparison table. It writes out-of-fold train probabilities and student_test
probabilities to data/predictions/_dev_hgb_*.parquet (gitignored). Once the real
models finish, point the HMM scripts at their prediction files instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from pathsnap.evaluation import run_experiment  # noqa: E402

if __name__ == "__main__":
    run_experiment(
        "_dev_hgb", "boosting (development stand-in)",
        {"max_iter": 150, "learning_rate": 0.1, "max_leaf_nodes": 31},
        lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                               early_stopping=False, random_state=42),
        seed=42, max_train_rows=600_000, results_dir=ROOT / "data" / "dev_runs", save_model=False)

"""Compare candidate models after HMM decoding (student_test), to choose the model for tuning and deployment.

Reads results/hmm_<model>/student_test_comparison.csv and best_params.json written by scripts/05_tune_hmm.py.
Usage: python scripts/10_hmm_model_comparison.py random_forest extra_trees lightgbm mlp
Writes results/hmm_model_comparison.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(models: list[str]) -> None:
    rows = []
    for model in models:
        folder = ROOT / "results" / f"hmm_{model}"
        table = pd.read_csv(folder / "student_test_comparison.csv").set_index("decoder")
        tuned, base = table.loc["classifier + HMM (tuned)"], table.loc["classifier only"]
        grid = pd.read_csv(folder / "tuning_grid.csv")
        rows.append({
            "model": model, "classifier_top1": base.top1_accuracy, "hmm_top1": tuned.top1_accuracy,
            "top1_gain": tuned.top1_accuracy - base.top1_accuracy,
            "classifier_continuity": base.route_continuity, "hmm_continuity": tuned.route_continuity,
            "classifier_arc_f1": base.route_arc_f1, "hmm_arc_f1": tuned.route_arc_f1,
            "tuning_trips_best_top1": grid.top1_accuracy.max(), "best_params": json.dumps(json.loads((folder / "best_params.json").read_text()))})
    out = pd.DataFrame(rows).sort_values("hmm_top1", ascending=False)
    out.to_csv(ROOT / "results" / "hmm_model_comparison.csv", index=False)
    print(out.drop(columns="best_params").round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])

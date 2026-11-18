"""Tune HMM decoder parameters and compare classifier-only vs classifier + HMM.

Protocol (no evaluator data involved):
  1. Grid-search decoder parameters on train OUT-OF-FOLD probabilities (every k-th train trip), so the
     probabilities were never produced by a model that saw those trips. Objective: observation top-1 accuracy,
     ties broken by route continuity.
  2. Evaluate the chosen parameters ONCE on student_test; report classifier-only vs classifier + HMM.

Usage: python scripts/05_tune_hmm.py --probs <model_name> [--every 3]
  <model_name> selects data/predictions/<model_name>_oof.parquet and _student_test.parquet.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.hmm import HmmParams  # noqa: E402
from pathsnap.hmm_eval import evaluate_routes, predict_partition  # noqa: E402

GRID = {
    "hop_penalty": [0.25, 0.5, 1.0],
    "break_penalty": [3.0, 6.0, 10.0],
    "max_hops": [3, 5, 8],
    "u_turn_penalty": [0.0, 2.0, 5.0],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs", required=True, help="model name whose saved probabilities to decode")
    parser.add_argument("--every", type=int, default=3, help="use every k-th train trip for tuning")
    args = parser.parse_args()
    out_dir = ROOT / "results" / f"hmm_{args.probs.lstrip('_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_train = pd.read_parquet(ROOT / "data" / "obs_train.parquet")
    pairs_train = pd.read_parquet(ROOT / "data" / "predictions" / f"{args.probs}_oof.parquet")
    tuning_trips = sorted(obs_train.trip_id.unique())[:: args.every]
    obs_tune = obs_train[obs_train.trip_id.isin(tuning_trips)]
    pairs_tune = pairs_train[pairs_train.trip_id.isin(tuning_trips)]
    print(f"tuning on {len(tuning_trips)} train trips ({len(obs_tune):,} points), grid size "
          f"{len(list(itertools.product(*GRID.values())))}")

    graphs: dict = {}
    baseline = evaluate_routes(obs_tune, predict_partition(pairs_tune, obs_tune, "train", None), "train")
    rows = []
    started = time.time()
    for values in itertools.product(*GRID.values()):
        params = HmmParams(**dict(zip(GRID, values)))
        metrics = evaluate_routes(obs_tune, predict_partition(pairs_tune, obs_tune, "train", params, graphs=graphs), "train")
        rows.append({**dict(zip(GRID, values)), **{k: metrics[k] for k in ("top1_accuracy", "route_continuity", "route_arc_f1")}})
    search = pd.DataFrame(rows).sort_values(["top1_accuracy", "route_continuity"], ascending=False)
    search.to_csv(out_dir / "tuning_grid.csv", index=False)
    best_row = search.iloc[0]
    best = HmmParams(hop_penalty=float(best_row.hop_penalty), break_penalty=float(best_row.break_penalty),
                     max_hops=int(best_row.max_hops), u_turn_penalty=float(best_row.u_turn_penalty))
    print(f"grid done in {time.time() - started:.0f}s; classifier-only on tuning trips: top1 "
          f"{baseline['top1_accuracy']:.4f}; best HMM top1 {best_row.top1_accuracy:.4f}  params {asdict(best)}")
    (out_dir / "best_params.json").write_text(json.dumps(asdict(best), indent=2))

    obs_test = pd.read_parquet(ROOT / "data" / "obs_student_test.parquet")
    pairs_test = pd.read_parquet(ROOT / "data" / "predictions" / f"{args.probs}_student_test.parquet")
    base_pred = predict_partition(pairs_test, obs_test, "student_test", None)
    hmm_pred = predict_partition(pairs_test, obs_test, "student_test", best)
    default_pred = predict_partition(pairs_test, obs_test, "student_test", HmmParams())
    comparison = pd.DataFrame([
        {"decoder": "classifier only", **evaluate_routes(obs_test, base_pred, "student_test")},
        {"decoder": "classifier + HMM (default params)", **evaluate_routes(obs_test, default_pred, "student_test")},
        {"decoder": "classifier + HMM (tuned)", **evaluate_routes(obs_test, hmm_pred, "student_test")},
    ])
    comparison.to_csv(out_dir / "student_test_comparison.csv", index=False)
    print(comparison.round(4).to_string(index=False))

    per_trip = []
    for trip_id in sorted(obs_test.trip_id.unique()):
        o = obs_test[obs_test.trip_id == trip_id]
        row = {"trip_id": trip_id, "n_obs": len(o)}
        for label, pred in (("classifier", base_pred), ("hmm", hmm_pred)):
            m = evaluate_routes(o, pred[pred.trip_id == trip_id], "student_test")
            row[f"{label}_top1"], row[f"{label}_continuity"] = m["top1_accuracy"], m["route_continuity"]
        row["top1_change"] = row["hmm_top1"] - row["classifier_top1"]
        per_trip.append(row)
    per_trip = pd.DataFrame(per_trip)
    per_trip.to_csv(out_dir / "student_test_per_trip.csv", index=False)
    print("\nper trip (HMM top1 change):")
    print(per_trip.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

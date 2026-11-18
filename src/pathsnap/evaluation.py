"""Shared experiment harness so every model is scored identically.

Usage (see scripts/models/_template.py):

    run_experiment(name="logreg", family="linear", config={...},
                   estimator_factory=lambda: make_pipeline(StandardScaler(), LogisticRegression(...)))

Protocol
- Fit/CV data: the 80 train trips (minus trips whose reference route does not
  cover the GPS track, see results/excluded_trips.csv).
- CV: 5-fold GroupKFold, groups = trip id, so no trip is split across folds.
- Student-test: the 10 student_test trips, model refit on ALL train rows.
- Evaluator-test trips are never touched here.

Metrics
- macro_f1 / log_loss: pair level (candidate is/is-not the reference arc), threshold 0.5.
- top1_accuracy: observation level. Per GPS point take the highest-scoring candidate and
  compare with the reference arc. Points whose reference arc is not in the candidate set
  count as wrong (so candidate misses are charged to the pipeline).
- route_continuity: share of consecutive-second point pairs whose predicted arcs are the
  same arc or head-to-tail connected (end node of one == start node of the next).
  truth_continuity is the same number for the reference arcs, i.e. the achievable level.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from pathsnap.data import DEFAULT_DATA_ROOT
from pathsnap.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
N_SPLITS = 5
DEFAULT_SEED = 42
_ARC_CACHE: dict[tuple[str, str], np.ndarray] = {}


def load_partition(partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pairs, obs) tables built by scripts/03_build_dataset.py."""
    pairs = pd.read_parquet(DATA_DIR / f"features_{partition}.parquet")
    obs = pd.read_parquet(DATA_DIR / f"obs_{partition}.parquet")
    return pairs, obs


def _arcs_for(partition: str, trip_id: str) -> np.ndarray:
    key = (partition, trip_id)
    if key not in _ARC_CACHE:
        path = Path(DEFAULT_DATA_ROOT) / partition / trip_id / f"{trip_id}.arcs"
        _ARC_CACHE[key] = np.loadtxt(path, dtype=np.int64, ndmin=2)
    return _ARC_CACHE[key]


def _continuity(frame: pd.DataFrame, arc_col: str, partition: str) -> float:
    """Share of consecutive observations whose arcs are identical or head-to-tail connected."""
    ok, total = 0, 0
    for trip_id, group in frame.sort_values(["trip_id", "obs_idx"]).groupby("trip_id"):
        arcs = _arcs_for(partition, trip_id)
        idx = group.obs_idx.to_numpy()
        arc = group[arc_col].to_numpy()
        consecutive = np.diff(idx) == 1
        a, b = arc[:-1][consecutive], arc[1:][consecutive]
        same = a == b
        connected = arcs[a, 1] == arcs[b, 0]
        ok += int((same | connected).sum())
        total += int(consecutive.sum())
    return ok / total if total else float("nan")


def score_predictions(pairs: pd.DataFrame, obs: pd.DataFrame, proba: np.ndarray,
                      partition: str) -> dict:
    """Compute all metrics for scored candidate pairs of one or more whole trips."""
    y = pairs.y.to_numpy()
    hard = (proba >= 0.5).astype(int)
    scored = pairs[["trip_id", "obs_idx", "arc_id", "y"]].copy()
    scored["p"] = proba
    best = scored.loc[scored.groupby(["trip_id", "obs_idx"]).p.idxmax()]
    top1 = float(best.y.sum()) / len(obs)
    pred = best.rename(columns={"arc_id": "pred_arc"})
    return {
        "n_pairs": int(len(pairs)), "n_obs": int(len(obs)),
        "macro_f1": float(f1_score(y, hard, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "top1_accuracy": top1,
        "candidate_miss_rate": float(1 - obs.truth_in_candidates.mean()),
        "route_continuity": _continuity(pred, "pred_arc", partition),
        "truth_continuity": _continuity(obs.assign(arc=obs.truth_arc), "arc", partition),
        "confusion_matrix": confusion_matrix(y, hard, labels=[0, 1]).tolist(),
    }


def _subsample_rows(pairs: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Random observation-level subsample (whole candidate sets kept together)."""
    if max_rows is None or len(pairs) <= max_rows:
        return pairs
    rng = np.random.default_rng(seed)
    keys = pairs[["trip_id", "obs_idx"]].drop_duplicates()
    keep_fraction = max_rows / len(pairs)
    chosen = keys[rng.random(len(keys)) < keep_fraction]
    return pairs.merge(chosen, on=["trip_id", "obs_idx"])


def run_experiment(name: str, family: str, config: dict, estimator_factory: Callable[[], object],
                   seed: int = DEFAULT_SEED, max_train_rows: int | None = None,
                   results_dir: Path = RESULTS_DIR, save_model: bool = True) -> dict:
    """Run CV on train and the student-test evaluation, then log everything.

    estimator_factory must return a NEW sklearn-style estimator (fit, predict_proba) each call.
    max_train_rows optionally subsamples the rows used for FITTING (whole candidate sets);
    evaluation always uses all rows. Record it in `config` when you use it.
    """
    import joblib

    train_pairs, train_obs = load_partition("train")
    test_pairs, test_obs = load_partition("student_test")
    groups = train_pairs.trip_id.to_numpy()
    print(f"[{name}] {len(train_pairs):,} train pairs, {train_pairs.trip_id.nunique()} trips; "
          f"fit-row cap = {max_train_rows}; {N_SPLITS}-fold GroupKFold")

    folds, oof = [], np.zeros(len(train_pairs))
    for fold, (fit_idx, val_idx) in enumerate(GroupKFold(n_splits=N_SPLITS).split(train_pairs, groups=groups)):
        fit_rows = _subsample_rows(train_pairs.iloc[fit_idx], max_train_rows, seed + fold)
        started = time.time()
        model = estimator_factory().fit(fit_rows[FEATURE_COLUMNS], fit_rows.y)
        fit_seconds = time.time() - started
        val_pairs = train_pairs.iloc[val_idx]
        val_obs = train_obs[train_obs.trip_id.isin(val_pairs.trip_id.unique())]
        proba = model.predict_proba(val_pairs[FEATURE_COLUMNS])[:, 1]
        oof[val_idx] = proba
        metrics = score_predictions(val_pairs, val_obs, proba, "train")
        metrics.update(fold=fold, fit_seconds=fit_seconds)
        folds.append(metrics)
        print(f"[{name}] fold {fold}: macro_f1={metrics['macro_f1']:.4f} top1={metrics['top1_accuracy']:.4f} "
              f"logloss={metrics['log_loss']:.4f} ({fit_seconds:.0f}s)")

    fit_rows = _subsample_rows(train_pairs, max_train_rows, seed)
    started = time.time()
    final = estimator_factory().fit(fit_rows[FEATURE_COLUMNS], fit_rows.y)
    final_fit_seconds = time.time() - started
    started = time.time()
    test_proba = final.predict_proba(test_pairs[FEATURE_COLUMNS])[:, 1]
    inference_seconds = time.time() - started
    test = score_predictions(test_pairs, test_obs, test_proba, "student_test")
    test.update(fit_seconds=final_fit_seconds, inference_seconds=inference_seconds,
                inference_ms_per_obs=1000 * inference_seconds / len(test_obs))

    fold_f1 = [f["macro_f1"] for f in folds]
    train_fit_proba = final.predict_proba(fit_rows[FEATURE_COLUMNS])[:, 1]
    record = {
        "name": name, "family": family, "config": config, "seed": seed,
        "max_train_rows": max_train_rows, "n_splits": N_SPLITS, "features": FEATURE_COLUMNS,
        "cv_folds": folds, "cv_macro_f1_mean": float(np.mean(fold_f1)), "cv_macro_f1_std": float(np.std(fold_f1)),
        "cv_top1_mean": float(np.mean([f["top1_accuracy"] for f in folds])),
        "cv_log_loss_mean": float(np.mean([f["log_loss"] for f in folds])),
        "cv_route_continuity_mean": float(np.mean([f["route_continuity"] for f in folds])),
        "train_pair_macro_f1": float(f1_score(fit_rows.y, train_fit_proba >= 0.5, average="macro")),
        "student_test": test,
        "run_at_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
    }
    results_dir = Path(results_dir)
    (results_dir / "experiments").mkdir(parents=True, exist_ok=True)
    (results_dir / "experiments" / f"{name}.json").write_text(json.dumps(record, indent=2))
    _upsert_comparison_row(results_dir / "model_comparison.csv", record)

    prediction_dir = DATA_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    keys = ["trip_id", "obs_idx", "arc_id", "y"]
    train_pairs[keys].assign(p=oof).to_parquet(prediction_dir / f"{name}_oof.parquet", index=False)
    test_pairs[keys].assign(p=test_proba).to_parquet(prediction_dir / f"{name}_student_test.parquet", index=False)
    if save_model:
        (ROOT / "models").mkdir(exist_ok=True)
        joblib.dump(final, ROOT / "models" / f"{name}.joblib")
    print(f"[{name}] CV macro F1 {record['cv_macro_f1_mean']:.4f}±{record['cv_macro_f1_std']:.4f} | "
          f"student-test macro F1 {test['macro_f1']:.4f}, top1 {test['top1_accuracy']:.4f}, "
          f"logloss {test['log_loss']:.4f}, continuity {test['route_continuity']:.4f}")
    return record


def _upsert_comparison_row(path: Path, record: dict) -> None:
    test = record["student_test"]
    row = {
        "model": record["name"], "family": record["family"], "key_config": json.dumps(record["config"], sort_keys=True),
        "cv_macro_f1": round(record["cv_macro_f1_mean"], 4), "cv_macro_f1_std": round(record["cv_macro_f1_std"], 4),
        "cv_top1": round(record["cv_top1_mean"], 4), "cv_route_continuity": round(record["cv_route_continuity_mean"], 4),
        "test_top1": round(test["top1_accuracy"], 4), "test_macro_f1": round(test["macro_f1"], 4),
        "test_log_loss": round(test["log_loss"], 4), "test_route_continuity": round(test["route_continuity"], 4),
        "train_macro_f1": round(record["train_pair_macro_f1"], 4),
        "fit_seconds": round(test["fit_seconds"], 1), "inference_ms_per_obs": round(test["inference_ms_per_obs"], 4),
    }
    table = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    table = table[table["model"] != record["name"]]
    pd.concat([table, pd.DataFrame([row])], ignore_index=True).sort_values("model").to_csv(path, index=False)

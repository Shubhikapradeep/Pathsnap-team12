"""FastAPI backend for PathSnap map-matching evaluator.

Agreed interface per docs/API_CONTRACT.md.
Serves Laptop 2 UI over personal hotspot on port 8000.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root and src/ are importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.schemas import (  # noqa: E402
    CONTRACT_VERSION,
    ErrorResponse,
    Health,
    Metrics,
    ModelInfo,
    PerPoint,
    PredictResponse,
    Route,
    RouteMetrics,
    Timing,
    TripSummary,
)
from pathsnap.data import Trip, load_trip  # noqa: E402
from pathsnap.evaluation import _continuity  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS, build_pair_features  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402
from pathsnap.hmm import ArcGraph, HmmParams, classifier_only_prediction, decode_trip  # noqa: E402
from pathsnap.labels import align_track_to_route  # noqa: E402
from pathsnap.route_geometry import arcs_to_segments, collapse_arc_sequence  # noqa: E402

# Setup loggers
LOG_FILE = ROOT / "api.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pathsnap.api")

# State
APP_STATE: dict = {
    "model": None,
    "model_name": "logreg",
    "git_version": "unknown",
    "data_dir": None,
    "graphs": {},
    "hmm_params": HmmParams(),
}


def get_git_revision_short_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT))
        return out.decode("ascii").strip()
    except Exception:
        return "v1.0"


def resolve_data_folder() -> tuple[Path, str]:
    """Return (base_folder, partition_name) for locating trip directories."""
    evaluator_env = os.environ.get("EVALUATOR_DATA")
    if evaluator_env:
        p = Path(evaluator_env)
        if p.exists():
            return p.parent, p.name

    pathsnap_data = os.environ.get("PATHSNAP_DATA", "../Student folder")
    base = Path(pathsnap_data)
    # Common layouts: <PATHSNAP_DATA>/student_test or <PATHSNAP_DATA>/Student folder/student_test
    if (base / "student_test").exists():
        return base, "student_test"
    if (base / "Student folder" / "student_test").exists():
        return base / "Student folder", "student_test"
    if (ROOT.parent / "Student folder" / "Student folder" / "student_test").exists():
        return ROOT.parent / "Student folder" / "Student folder", "student_test"
    return base, "student_test"


def stub_scorer(pairs: pd.DataFrame) -> np.ndarray:
    """Distance-based fallback scoring function (closer = higher score)."""
    dist = pairs.dist_m.to_numpy()
    return np.exp(-dist / 25.0)


def model_scorer(pairs: pd.DataFrame, model) -> np.ndarray:
    """Score pairs with the trained pipeline."""
    X = pairs[FEATURE_COLUMNS]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


@asynccontextmanager
async def lifespan(app: FastAPI):
    APP_STATE["git_version"] = get_git_revision_short_hash()
    base_dir, partition = resolve_data_folder()
    APP_STATE["data_dir"] = base_dir / partition
    logger.info(f"Resolved trip data folder: {APP_STATE['data_dir']}")

    use_stub = os.environ.get("USE_STUB", "0").lower() in ("1", "true", "yes")
    model_path = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "models" / "logreg.joblib"))

    if not use_stub and model_path.exists():
        try:
            APP_STATE["model"] = joblib.load(model_path)
            APP_STATE["model_name"] = model_path.stem
            logger.info(f"Loaded classifier model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model {model_path}: {e}. Falling back to stub.")
            APP_STATE["model"] = None
            APP_STATE["model_name"] = "stub_distance"
    else:
        APP_STATE["model"] = None
        APP_STATE["model_name"] = "stub_distance"
        logger.info("Using distance-based stub model for predictions")

    yield
    APP_STATE["graphs"].clear()


app = FastAPI(
    title="PathSnap Evaluator API",
    version=CONTRACT_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def list_available_trips() -> list[str]:
    data_dir = APP_STATE.get("data_dir")
    if not data_dir or not data_dir.exists():
        base, partition = resolve_data_folder()
        data_dir = base / partition
    if not data_dir.exists():
        return []
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists())


@app.get("/health", response_model=Health)
def health():
    return Health(
        status="ok",
        contract_version=CONTRACT_VERSION,
        model=APP_STATE["model_name"],
        version=APP_STATE["git_version"],
    )


@app.get("/trips", response_model=list[TripSummary])
def get_trips():
    trip_ids = list_available_trips()
    base, partition = resolve_data_folder()
    summaries = []
    for tid in trip_ids:
        track_file = base / partition / tid / f"{tid}.track"
        try:
            with open(track_file, "r") as f:
                n_lines = sum(1 for _ in f)
        except Exception:
            n_lines = 0
        summaries.append(TripSummary(trip_id=tid, n_points=n_lines))
    return summaries


def run_pipeline(trip_id: str, custom_scorer: Callable[[pd.DataFrame], np.ndarray] | None = None) -> PredictResponse:
    t_start = time.time()
    base, partition = resolve_data_folder()
    trip_folder = base / partition / trip_id
    if not trip_folder.exists() or not (trip_folder / f"{trip_id}.track").exists():
        raise HTTPException(
            status_code=404,
            detail=f"Trip '{trip_id}' not found in {base / partition}",
        )

    warnings: list[str] = []

    # 1. Load raw trip (route is loaded server-side only for metrics)
    trip = load_trip(trip_id, partition, root=base, load_route=True)
    n_points = len(trip.track_lonlat)

    # 2. Local projection
    track_xy, node_xy = project_trip(trip)

    # 3. Reference alignment check for warnings (if reference route exists)
    truth_arc = None
    if trip.route is not None:
        try:
            _, truth_arc, align_dist = align_track_to_route(
                track_xy, trip.timestamps, node_xy, trip.arcs, trip.route
            )
            far_fraction = float((align_dist > 50.0).mean())
            if far_fraction > 0.20:
                warnings.append("reference route covers only part of the GPS track; accuracy for this trip is unreliable")
        except Exception as e:
            logger.warning(f"Reference alignment warning: {e}")
            truth_arc = None

    # 4. Candidate pairs and features
    t_cand_start = time.time()
    pairs = build_pair_features(track_xy, trip.timestamps, node_xy, trip.arcs)
    candidates_s = time.time() - t_cand_start
    features_s = 0.0

    # 5. Model scoring
    t_clf_start = time.time()
    if custom_scorer is not None:
        proba = custom_scorer(pairs)
    elif APP_STATE["model"] is not None:
        proba = model_scorer(pairs, APP_STATE["model"])
    else:
        proba = stub_scorer(pairs)
    classifier_s = time.time() - t_clf_start

    pairs["p"] = proba

    # 6. Classifier-only prediction
    clf_predicted = classifier_only_prediction(pairs, n_points)

    # 7. HMM decoding
    t_hmm_start = time.time()
    if trip_id not in APP_STATE["graphs"]:
        APP_STATE["graphs"][trip_id] = ArcGraph(trip.arcs, 8)
    graph = APP_STATE["graphs"][trip_id]
    hmm_predicted = decode_trip(pairs, trip.timestamps, graph, APP_STATE["hmm_params"], n_points)
    hmm_s = time.time() - t_hmm_start

    # 8. Routes (geometry segments)
    clf_arc_seq = collapse_arc_sequence(clf_predicted)
    clf_segments = arcs_to_segments(clf_arc_seq, trip.arcs, trip.node_lonlat)

    hmm_arc_seq = collapse_arc_sequence(hmm_predicted)
    hmm_segments = arcs_to_segments(hmm_arc_seq, trip.arcs, trip.node_lonlat)

    # GPS coordinates in Leaflet [lat, lon, t_seconds]
    gps_list = [
        [round(float(lat), 6), round(float(lon), 6), float(t)]
        for (lon, lat), t in zip(trip.track_lonlat, trip.timestamps)
    ]

    # 9. Server-side metrics computation (truth_arc is NEVER exposed to UI)
    scored_mask = clf_predicted >= 0
    n_scored = int(scored_mask.sum())
    cand_miss_rate = float(1.0 - (n_scored / n_points)) if n_points > 0 else 0.0

    def compute_route_metrics(pred_arcs: np.ndarray) -> RouteMetrics:
        if truth_arc is None:
            # If no ground truth route provided, report 0.0
            return RouteMetrics(top1_accuracy=0.0, route_continuity=0.0, route_arc_f1=0.0)

        # Top-1 observation accuracy
        top1 = float((pred_arcs == truth_arc).mean())

        # Continuity
        valid = pred_arcs >= 0
        df_pred = pd.DataFrame({
            "trip_id": trip_id,
            "obs_idx": np.arange(n_points)[valid],
            "pred_arc": pred_arcs[valid],
        })
        cont = _continuity(df_pred, "pred_arc", partition) if len(df_pred) > 0 else 0.0

        # Arc-set F1
        truth_set = set(truth_arc.tolist())
        pred_set = set(pred_arcs[valid].tolist())
        hit = len(truth_set & pred_set)
        p = hit / len(pred_set) if pred_set else 0.0
        r = hit / len(truth_set) if truth_set else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return RouteMetrics(
            top1_accuracy=round(top1, 4),
            route_continuity=round(cont, 4),
            route_arc_f1=round(f1, 4),
        )

    hmm_metrics = compute_route_metrics(hmm_predicted)
    clf_metrics = compute_route_metrics(clf_predicted)

    total_s = time.time() - t_start

    response = PredictResponse(
        contract_version=CONTRACT_VERSION,
        trip_id=trip_id,
        n_points=n_points,
        gps=gps_list,
        classifier_route=Route(arc_ids=clf_arc_seq, segments=clf_segments),
        hmm_route=Route(arc_ids=hmm_arc_seq, segments=hmm_segments),
        per_point=PerPoint(
            hmm_arc=hmm_predicted.tolist(),
            classifier_arc=clf_predicted.tolist(),
        ),
        metrics=Metrics(
            top1_accuracy=hmm_metrics.top1_accuracy,
            route_continuity=hmm_metrics.route_continuity,
            route_arc_f1=hmm_metrics.route_arc_f1,
            n_points=n_points,
            n_scored_points=n_scored,
            candidate_miss_rate=round(cand_miss_rate, 4),
            classifier_only=clf_metrics,
        ),
        timing=Timing(
            seconds=round(total_s, 3),
            candidates_s=round(candidates_s, 3),
            features_s=round(features_s, 3),
            classifier_s=round(classifier_s, 3),
            hmm_s=round(hmm_s, 3),
        ),
        model=ModelInfo(name=APP_STATE["model_name"], version=APP_STATE["git_version"]),
        warnings=warnings,
        mock=False,
    )

    # Evaluator live log format specified in API_CONTRACT.md:
    # PREDICT trip=00000015 status=200 top1=0.9xx seconds=3.2
    logger.info(
        f"PREDICT trip={trip_id} status=200 top1={response.metrics.top1_accuracy:.4f} "
        f"continuity={response.metrics.route_continuity:.4f} seconds={total_s:.2f}"
    )

    return response


@app.post("/predict/{trip_id}", response_model=PredictResponse, responses={404: {"model": ErrorResponse}})
def predict_trip_post(trip_id: str):
    return run_pipeline(trip_id)


@app.get("/predict/{trip_id}", response_model=PredictResponse, responses={404: {"model": ErrorResponse}})
def predict_trip_get(trip_id: str):
    return run_pipeline(trip_id)

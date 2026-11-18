"""End-to-end inference: raw trip -> routes. This is what the API calls.

Prediction path (no labels anywhere in it):
    project -> candidate arcs + features -> classifier probabilities -> classifier-only route
                                                                     -> HMM/Viterbi route

`run_pipeline` takes a Trip loaded WITHOUT its reference route (`load_trip(..., load_route=False)`),
so ground truth cannot leak into prediction. Scoring against reference data is a separate step in
`pathsnap.scoring`, called only on the server side.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pathsnap.data import Trip
from pathsnap.features import FEATURE_COLUMNS, build_pair_features
from pathsnap.geo import project_trip
from pathsnap.hmm import ArcGraph, HmmParams, classifier_only_prediction, decode_trip
from pathsnap.route_geometry import arcs_to_segments, collapse_arc_sequence


@dataclass
class Artifacts:
    """Everything inference needs: the fitted classifier, decoder parameters and a name/version."""

    classifier: Any
    hmm_params: HmmParams
    model_name: str
    version: str = "unknown"


def load_artifacts(model_path: Path, hmm_params_path: Path, model_name: str, version: str = "unknown") -> Artifacts:
    """Load a joblib classifier and a best_params.json written by scripts/05_tune_hmm.py."""
    raw = json.loads(Path(hmm_params_path).read_text())
    allowed = HmmParams.__dataclass_fields__.keys()
    params = HmmParams(**{key: value for key, value in raw.items() if key in allowed})
    return Artifacts(joblib.load(model_path), params, model_name, version)


@dataclass
class PipelineResult:
    """Predictions for one trip. Arrays have one entry per GPS point; -1 means no candidate."""

    trip_id: str
    n_points: int
    gps: list[list[float]]                 # [lat, lon, t_seconds]
    classifier_arc: np.ndarray
    hmm_arc: np.ndarray
    classifier_route: dict                 # {"arc_ids": [...], "segments": [[[lat, lon], ...], ...]}
    hmm_route: dict
    candidates: pd.DataFrame               # obs_idx, arc_id (kept so scoring can compute candidate miss rate)
    timing: dict = field(default_factory=dict)
    track_xy: np.ndarray | None = None     # projected, metres (used by scoring)
    node_xy: np.ndarray | None = None


def _route_payload(per_point_arcs: np.ndarray, trip: Trip) -> dict:
    arc_ids = collapse_arc_sequence(per_point_arcs)
    return {"arc_ids": arc_ids, "segments": arcs_to_segments(arc_ids, trip.arcs, trip.node_lonlat)}


def run_pipeline(trip: Trip, artifacts: Artifacts) -> PipelineResult:
    """Run the full prediction pipeline on one trip."""
    n_points = len(trip.timestamps)
    timing: dict[str, float] = {}
    started = time.perf_counter()

    track_xy, node_xy = project_trip(trip)
    pairs = build_pair_features(track_xy, trip.timestamps, node_xy, trip.arcs)
    timing["features_s"] = time.perf_counter() - started

    stage = time.perf_counter()
    if len(pairs):
        pairs["p"] = artifacts.classifier.predict_proba(pairs[FEATURE_COLUMNS])[:, 1]
    else:
        pairs["p"] = pd.Series(dtype=float)
    timing["classifier_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    classifier_arc = classifier_only_prediction(pairs, n_points) if len(pairs) else np.full(n_points, -1)
    graph = ArcGraph(trip.arcs, depth=artifacts.hmm_params.max_hops)
    hmm_arc = decode_trip(pairs[["obs_idx", "arc_id", "p"]], trip.timestamps, graph, artifacts.hmm_params, n_points)
    timing["hmm_s"] = time.perf_counter() - stage
    timing["seconds"] = time.perf_counter() - started

    lat_lon = trip.track_lonlat[:, ::-1]
    gps = [[round(float(lat), 6), round(float(lon), 6), float(t)] for (lat, lon), t in zip(lat_lon, trip.timestamps)]
    return PipelineResult(
        trip_id=trip.trip_id, n_points=n_points, gps=gps, classifier_arc=classifier_arc, hmm_arc=hmm_arc,
        classifier_route=_route_payload(classifier_arc, trip), hmm_route=_route_payload(hmm_arc, trip),
        candidates=pairs[["obs_idx", "arc_id"]].copy(), timing={k: round(v, 4) for k, v in timing.items()},
        track_xy=track_xy, node_xy=node_xy)

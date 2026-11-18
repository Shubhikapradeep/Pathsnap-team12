"""Server-side scoring of a PipelineResult against the reference route.

Only the API (and offline evaluation) may call this. It needs the trip's `.route`, which must never reach
the prediction pipeline, the model or the UI. Metrics use in-memory arrays only (no disk lookups), and match
the definitions in `pathsnap.evaluation` / `pathsnap.hmm_eval`:

* top1_accuracy       share of ALL points whose predicted arc equals the reference arc (-1 counts as wrong)
* route_continuity    share of consecutive-second point pairs whose predicted arcs are the same or head-to-tail connected
* route_arc_f1        F1 between the SET of predicted arcs and the SET of reference arcs
* candidate_miss_rate share of points whose reference arc was not in the candidate set
"""
from __future__ import annotations

import numpy as np

from pathsnap.data import Trip
from pathsnap.labels import align_track_to_route
from pathsnap.pipeline import PipelineResult

UNRELIABLE_DISTANCE_M = 50.0
UNRELIABLE_FRACTION = 0.20
TRUNCATED_ROUTE_WARNING = ("reference route covers only part of the GPS track; accuracy for this trip is unreliable")


def _continuity(predicted: np.ndarray, arcs: np.ndarray) -> float:
    index = np.flatnonzero(predicted >= 0)
    consecutive = np.diff(index) == 1
    a, b = predicted[index[:-1]][consecutive], predicted[index[1:]][consecutive]
    if len(a) == 0:
        return float("nan")
    return float(((a == b) | (arcs[a, 1] == arcs[b, 0])).mean())


def _arc_set_f1(predicted: np.ndarray, truth: np.ndarray) -> float:
    pred_set, truth_set = set(predicted[predicted >= 0].tolist()), set(truth.tolist())
    hit = len(pred_set & truth_set)
    precision = hit / len(pred_set) if pred_set else 0.0
    recall = hit / len(truth_set) if truth_set else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def reference_arcs(trip: Trip, result: PipelineResult) -> tuple[np.ndarray, np.ndarray]:
    """Per-point reference arc and its alignment distance (metres), derived from trip.route."""
    if trip.route is None:
        raise ValueError("scoring needs the reference route; load the trip with load_route=True on the server")
    _, truth, align_distance = align_track_to_route(result.track_xy, trip.timestamps, result.node_xy,
                                                    trip.arcs, trip.route)
    return truth, align_distance


def score_result(trip: Trip, result: PipelineResult) -> dict:
    """Metrics for the HMM route and the classifier-only baseline, plus any warnings."""
    truth, align_distance = reference_arcs(trip, result)
    candidate_keys = set(zip(result.candidates.obs_idx.tolist(), result.candidates.arc_id.tolist()))
    in_candidates = np.fromiter(((i, int(a)) in candidate_keys for i, a in enumerate(truth)), dtype=bool, count=len(truth))

    def route_metrics(predicted: np.ndarray) -> dict:
        return {"top1_accuracy": round(float((predicted == truth).mean()), 4),
                "route_continuity": round(_continuity(predicted, trip.arcs), 4),
                "route_arc_f1": round(_arc_set_f1(predicted, truth), 4)}

    warnings = []
    if (align_distance > UNRELIABLE_DISTANCE_M).mean() > UNRELIABLE_FRACTION:
        warnings.append(TRUNCATED_ROUTE_WARNING)
    return {
        "metrics": {**route_metrics(result.hmm_arc), "n_points": result.n_points,
                    "n_scored_points": int((result.hmm_arc >= 0).sum()),
                    "candidate_miss_rate": round(float(1 - in_candidates.mean()), 4),
                    "classifier_only": route_metrics(result.classifier_arc)},
        "warnings": warnings,
    }

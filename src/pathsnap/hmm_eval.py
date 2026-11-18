"""Run the HMM decoder over many trips and score the resulting routes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.evaluation import _arcs_for, _continuity
from pathsnap.hmm import ArcGraph, HmmParams, classifier_only_prediction, decode_trip

GRAPH_DEPTH = 8  # must be >= any max_hops searched


def predict_partition(pairs: pd.DataFrame, obs: pd.DataFrame, partition: str, params: HmmParams | None,
                      trip_ids: list[str] | None = None, graphs: dict[str, ArcGraph] | None = None
                      ) -> pd.DataFrame:
    """Predicted arc per observation. params=None gives the classifier-only baseline.

    pairs needs columns trip_id, obs_idx, arc_id, p. Returns trip_id, obs_idx, pred_arc (-1 = no candidate).
    """
    graphs = graphs if graphs is not None else {}
    out = []
    trips = trip_ids if trip_ids is not None else sorted(obs.trip_id.unique())
    pair_groups = {t: g for t, g in pairs.groupby("trip_id")}
    for trip_id in trips:
        trip_obs = obs[obs.trip_id == trip_id].sort_values("obs_idx")
        n_obs = len(trip_obs)
        trip_pairs = pair_groups[trip_id]
        if params is None:
            predicted = classifier_only_prediction(trip_pairs, n_obs)
        else:
            if trip_id not in graphs:
                graphs[trip_id] = ArcGraph(_arcs_for(partition, trip_id), GRAPH_DEPTH)
            predicted = decode_trip(trip_pairs, trip_obs.timestamp.to_numpy(), graphs[trip_id], params, n_obs)
        out.append(pd.DataFrame({"trip_id": trip_id, "obs_idx": np.arange(n_obs), "pred_arc": predicted}))
    return pd.concat(out, ignore_index=True)


def evaluate_routes(obs: pd.DataFrame, predictions: pd.DataFrame, partition: str) -> dict:
    """Pointwise and sequence-level metrics for decoded routes.

    top1_accuracy       share of ALL points whose predicted arc equals the reference arc
    route_continuity    share of consecutive-second predicted pairs that are the same arc or connected
    route_arc_f1        per trip, F1 between the SET of predicted arcs and the SET of reference arcs
                        (sequence-level overlap, ignores order); averaged over trips
    """
    merged = obs[["trip_id", "obs_idx", "truth_arc"]].merge(predictions, on=["trip_id", "obs_idx"])
    scored = merged[merged.pred_arc >= 0]
    f1_per_trip = []
    for _, trip in merged.groupby("trip_id"):
        truth_set, pred_set = set(trip.truth_arc.tolist()), set(trip.pred_arc[trip.pred_arc >= 0].tolist())
        hit = len(truth_set & pred_set)
        precision = hit / len(pred_set) if pred_set else 0.0
        recall = hit / len(truth_set) if truth_set else 0.0
        f1_per_trip.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "n_obs": int(len(merged)),
        "top1_accuracy": float((merged.pred_arc == merged.truth_arc).mean()),
        "route_continuity": _continuity(scored, "pred_arc", partition),
        "truth_continuity": _continuity(merged.assign(arc=merged.truth_arc), "arc", partition),
        "route_arc_f1": float(np.mean(f1_per_trip)),
    }

"""Per-(observation, candidate arc) feature engineering.

Temporal availability: this is an offline map-matcher over a completed trip, so
features may use GPS points before and after observation i within a short
window (<= CONTEXT_OFFSETS seconds-ish). No route labels and no information
from any other trip are ever used. Candidate generation and every feature are
computed from the GPS track and the road network only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates
from pathsnap.geo import point_segment_distance

RADIUS_M = 100.0
MAX_CANDIDATES = 30
CONTEXT_OFFSETS = (-6, -3, 3, 6)
MAX_CONTEXT_SECONDS = 15.0
HEADING_HALF_WINDOW = 3
MIN_SPEED_FOR_HEADING_MS = 1.0
MAX_HEADING_WINDOW_SECONDS = 18.0

FEATURE_COLUMNS = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio_to_min", "n_candidates",
    "t_frac", "arc_len_m", "heading_cos", "heading_valid", "speed_ms", "motion_along_arc_m",
    "ctx_dist_mean", "ctx_dist_max", "ctx_valid_count",
    "start_in_degree", "end_out_degree", "has_reverse_arc", "dist_to_start_m", "dist_to_end_m",
]


def vehicle_motion(track_xy: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unit_heading (n,2), speed_ms (n,), valid (n,)) from a centred window."""
    n = len(track_xy)
    idx = np.arange(n)
    lo = np.clip(idx - HEADING_HALF_WINDOW, 0, n - 1)
    hi = np.clip(idx + HEADING_HALF_WINDOW, 0, n - 1)
    displacement = track_xy[hi] - track_xy[lo]
    elapsed = timestamps[hi] - timestamps[lo]
    distance = np.linalg.norm(displacement, axis=1)
    speed = np.divide(distance, elapsed, out=np.zeros(n), where=elapsed > 0)
    valid = (elapsed > 0) & (elapsed <= MAX_HEADING_WINDOW_SECONDS) & (speed >= MIN_SPEED_FOR_HEADING_MS)
    heading = np.divide(displacement, distance[:, None], out=np.zeros_like(displacement),
                        where=(distance > 0)[:, None])
    return heading, speed, valid


def build_pair_features(track_xy: np.ndarray, timestamps: np.ndarray, node_xy: np.ndarray,
                        arcs: np.ndarray, radius_m: float = RADIUS_M,
                        max_candidates: int = MAX_CANDIDATES) -> pd.DataFrame:
    """One row per (observation, candidate arc) with FEATURE_COLUMNS plus obs_idx, arc_id."""
    cand = generate_candidates(track_xy, node_xy, arcs, radius_m, max_candidates)
    obs, arc = cand.obs_idx.to_numpy(), cand.arc_id.to_numpy()
    start_xy, end_xy = node_xy[arcs[arc, 0]], node_xy[arcs[arc, 1]]
    arc_vec = end_xy - start_xy
    arc_len = np.linalg.norm(arc_vec, axis=1)
    arc_dir = np.divide(arc_vec, arc_len[:, None], out=np.zeros_like(arc_vec), where=(arc_len > 0)[:, None])

    heading, speed, heading_ok = vehicle_motion(track_xy, timestamps)
    heading_cos = np.where(heading_ok[obs], np.einsum("ij,ij->i", heading[obs], arc_dir), 0.0)
    n_obs = len(track_xy)
    window_dist = np.linalg.norm(track_xy[np.clip(obs + HEADING_HALF_WINDOW, 0, n_obs - 1)]
                                 - track_xy[np.clip(obs - HEADING_HALF_WINDOW, 0, n_obs - 1)], axis=1)
    motion_along_arc = np.where(heading_ok[obs], window_dist * heading_cos, 0.0)

    dist = cand.dist_m.to_numpy()
    min_dist = cand.groupby("obs_idx").dist_m.transform("min").to_numpy()
    n_cand = cand.groupby("obs_idx").dist_m.transform("size").to_numpy()

    context = []
    context_valid = []
    for offset in CONTEXT_OFFSETS:
        neighbour = np.clip(obs + offset, 0, n_obs - 1)
        ok = np.abs(timestamps[neighbour] - timestamps[obs]) <= MAX_CONTEXT_SECONDS
        d, _ = point_segment_distance(track_xy[neighbour], start_xy, end_xy)
        context.append(np.where(ok, d, np.nan))
        context_valid.append(ok)
    context = np.column_stack(context)
    valid_count = np.column_stack(context_valid).sum(axis=1)
    # Own distance is included so rows with no valid neighbour fall back to dist_m (flagged by ctx_valid_count == 0)
    with_self = np.column_stack([context, dist])
    ctx_mean = np.nanmean(with_self, axis=1)
    ctx_max = np.nanmax(with_self, axis=1)

    in_degree = np.bincount(arcs[:, 1], minlength=len(node_xy))
    out_degree = np.bincount(arcs[:, 0], minlength=len(node_xy))
    reverse_keys = set(zip(arcs[:, 1].tolist(), arcs[:, 0].tolist()))
    has_reverse_all = np.fromiter(((a, b) in reverse_keys for a, b in arcs.tolist()), dtype=bool, count=len(arcs))

    t = cand.t_frac.to_numpy()
    return pd.DataFrame({
        "obs_idx": obs, "arc_id": arc,
        "dist_m": dist, "dist_rank": cand.dist_rank.to_numpy(),
        "dist_minus_min": dist - min_dist, "dist_ratio_to_min": dist / (min_dist + 1.0),
        "n_candidates": n_cand, "t_frac": t, "arc_len_m": arc_len,
        "heading_cos": heading_cos, "heading_valid": heading_ok[obs].astype(int),
        "speed_ms": speed[obs], "motion_along_arc_m": motion_along_arc,
        "ctx_dist_mean": ctx_mean, "ctx_dist_max": ctx_max, "ctx_valid_count": valid_count,
        "start_in_degree": in_degree[arcs[arc, 0]], "end_out_degree": out_degree[arcs[arc, 1]],
        "has_reverse_arc": has_reverse_all[arc].astype(int),
        "dist_to_start_m": t * arc_len, "dist_to_end_m": (1 - t) * arc_len,
    })

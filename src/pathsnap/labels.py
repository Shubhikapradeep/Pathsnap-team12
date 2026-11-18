"""Derive per-observation reference arcs from the supplied route.

The .route file lists arcs in travel order with no timestamps, so a GPS row
does not map directly to a route arc. We recover the mapping with a monotone
dynamic-programming alignment: observation i is assigned a route *position*
p_i with p_i non-decreasing, minimising summed point-to-arc distance.
Skipping forward over route arcs carries a small penalty; after a long time gap
the vehicle may legitimately have travelled any distance, so gaps allow a
free-form forward jump (penalty proportional to the jump).

This is a derived reference label, not independently verified ground truth.
Its quality is reported through `align_dist` (distance of each observation to
its assigned arc).
"""
from __future__ import annotations

import numpy as np

from pathsnap.geo import point_segment_distance

DISTANCE_CAP_M = 200.0


def _distance_to_route_arcs(track_xy: np.ndarray, seg_start: np.ndarray,
                            seg_end: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """(n_obs, n_route) matrix of point-to-route-arc distances in metres."""
    n_obs, n_route = len(track_xy), len(seg_start)
    out = np.empty((n_obs, n_route))
    for lo in range(0, n_obs, chunk):
        hi = min(lo + chunk, n_obs)
        pts = np.repeat(track_xy[lo:hi], n_route, axis=0)
        a = np.tile(seg_start, (hi - lo, 1))
        b = np.tile(seg_end, (hi - lo, 1))
        out[lo:hi] = point_segment_distance(pts, a, b)[0].reshape(hi - lo, n_route)
    return out


def align_track_to_route(track_xy: np.ndarray, timestamps: np.ndarray, node_xy: np.ndarray,
                         arcs: np.ndarray, route: np.ndarray, max_skip: int = 40,
                         skip_penalty_m: float = 0.5, gap_seconds: float = 10.0
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (route_position, arc_id, align_dist_m), each of length n_obs."""
    seg_start = node_xy[arcs[route, 0]]
    seg_end = node_xy[arcs[route, 1]]
    cost = np.minimum(_distance_to_route_arcs(track_xy, seg_start, seg_end), DISTANCE_CAP_M)
    n_obs, n_route = cost.shape
    positions = np.arange(n_route, dtype=float)
    backpointer = np.zeros((n_obs, n_route), dtype=np.int32)
    accumulated = cost[0].copy()
    big = 1e18
    for i in range(1, n_obs):
        if timestamps[i] - timestamps[i - 1] > gap_seconds:
            shifted = accumulated - skip_penalty_m * positions
            running_min = np.minimum.accumulate(shifted)
            source = np.maximum.accumulate(np.where(shifted == running_min, np.arange(n_route), 0))
            best = running_min + skip_penalty_m * positions
        else:
            stacked = np.full((max_skip + 1, n_route), big)
            for k in range(max_skip + 1):
                penalty = skip_penalty_m * max(k - 1, 0)
                stacked[k, k:] = accumulated[: n_route - k] + penalty
            step = stacked.argmin(axis=0)
            best = stacked[step, np.arange(n_route)]
            source = np.arange(n_route) - step
        backpointer[i] = source
        accumulated = best + cost[i]
    path = np.empty(n_obs, dtype=np.int64)
    path[-1] = int(accumulated.argmin())
    for i in range(n_obs - 1, 0, -1):
        path[i - 1] = backpointer[i, path[i]]
    return path, route[path], cost[np.arange(n_obs), path]

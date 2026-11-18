"""Candidate road-arc generation for each GPS observation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from pathsnap.geo import point_segment_distance


def build_arc_lines(node_xy: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    """Shapely LineString per arc (index = arc_id)."""
    coords = np.stack([node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]], axis=1)
    return shapely.linestrings(coords)


def generate_candidates(track_xy: np.ndarray, node_xy: np.ndarray, arcs: np.ndarray,
                        radius_m: float, max_candidates: int) -> pd.DataFrame:
    """All arcs within radius_m of each observation, nearest max_candidates kept.

    Returns columns obs_idx, arc_id, dist_m, t_frac, dist_rank.
    """
    tree = STRtree(build_arc_lines(node_xy, arcs))
    obs_idx, arc_id = tree.query(shapely.points(track_xy), predicate="dwithin", distance=radius_m)
    dist, t_frac = point_segment_distance(track_xy[obs_idx], node_xy[arcs[arc_id, 0]],
                                          node_xy[arcs[arc_id, 1]])
    frame = pd.DataFrame({"obs_idx": obs_idx, "arc_id": arc_id, "dist_m": dist, "t_frac": t_frac})
    frame = frame.sort_values(["obs_idx", "dist_m"], kind="stable")
    frame["dist_rank"] = frame.groupby("obs_idx").cumcount()
    return frame[frame["dist_rank"] < max_candidates].reset_index(drop=True)

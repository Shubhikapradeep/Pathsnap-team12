"""Raw data loading for PathSnap trips.

Each trip lives in <root>/<partition>/<trip_id>/ with four headerless,
whitespace-separated files: .track (lon, lat, t), .nodes (lon, lat),
.arcs (from_node, to_node) and .route (arc_id). Row index is the node/arc id.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", "../Student folder"))
PARTITIONS = ("train", "student_test")


@dataclass
class Trip:
    trip_id: str
    partition: str
    track_lonlat: np.ndarray   # (n_obs, 2) longitude, latitude
    timestamps: np.ndarray     # (n_obs,) seconds
    node_lonlat: np.ndarray    # (n_nodes, 2)
    arcs: np.ndarray           # (n_arcs, 2) from_node, to_node
    route: np.ndarray | None   # (n_route,) arc ids, None if unavailable


def list_trip_ids(partition: str, root: Path = DEFAULT_DATA_ROOT) -> list[str]:
    """Return sorted trip ids found in a partition folder."""
    return sorted(p.name for p in (Path(root) / partition).iterdir() if p.is_dir())


def load_trip(trip_id: str, partition: str, root: Path = DEFAULT_DATA_ROOT,
              load_route: bool = True) -> Trip:
    """Load one trip. Set load_route=False for inference-side loading."""
    folder = Path(root) / partition / trip_id
    track = np.loadtxt(folder / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(folder / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(folder / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    route_path = folder / f"{trip_id}.route"
    route = None
    if load_route and route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int64, ndmin=1)
    return Trip(trip_id, partition, track[:, :2], track[:, 2], nodes, arcs, route)

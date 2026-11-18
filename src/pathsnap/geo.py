"""Projection and point-to-segment geometry.

Every trip has its own road network in a different part of the world, so each
trip is projected into its own local azimuthal-equidistant plane (metres)
centred on the GPS track. Distances inside a trip are then Euclidean metres.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

from pathsnap.data import Trip


def project_trip(trip: Trip) -> tuple[np.ndarray, np.ndarray]:
    """Return (track_xy, node_xy) in metres in a trip-local projection."""
    lon0, lat0 = trip.track_lonlat.mean(axis=0)
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    track_x, track_y = transformer.transform(trip.track_lonlat[:, 0], trip.track_lonlat[:, 1])
    node_x, node_y = transformer.transform(trip.node_lonlat[:, 0], trip.node_lonlat[:, 1])
    return np.column_stack([track_x, track_y]), np.column_stack([node_x, node_y])


def point_segment_distance(points: np.ndarray, seg_start: np.ndarray,
                           seg_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise distance from points[i] to segment (seg_start[i], seg_end[i]).

    Returns (distance_m, t) where t in [0, 1] is the clamped position of the
    closest point along the segment (0 = start node, 1 = end node).
    """
    direction = seg_end - seg_start
    length_sq = np.einsum("ij,ij->i", direction, direction)
    offset = points - seg_start
    raw_t = np.einsum("ij,ij->i", offset, direction) / np.where(length_sq > 0, length_sq, 1.0)
    t = np.clip(np.where(length_sq > 0, raw_t, 0.0), 0.0, 1.0)
    closest = seg_start + t[:, None] * direction
    return np.linalg.norm(points - closest, axis=1), t

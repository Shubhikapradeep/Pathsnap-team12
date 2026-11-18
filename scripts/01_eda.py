"""Dataset inventory and data-quality summary for train and student_test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.data import PARTITIONS, list_trip_ids, load_trip  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def summarise_trip(trip) -> dict:
    dt = np.diff(trip.timestamps)
    n_nodes, n_arcs = len(trip.node_lonlat), len(trip.arcs)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(trip.arcs.tolist())
    components = list(nx.connected_components(graph))
    lon, lat = trip.track_lonlat[:, 0], trip.track_lonlat[:, 1]
    route = trip.route
    return {
        "partition": trip.partition, "trip_id": trip.trip_id,
        "n_obs": len(trip.timestamps), "n_nodes": n_nodes, "n_arcs": n_arcs,
        "n_route_arcs": None if route is None else len(route),
        "route_unique_arcs": None if route is None else len(set(route.tolist())),
        "route_arc_id_max": None if route is None else int(route.max()),
        "nan_track_rows": int(np.isnan(trip.track_lonlat).any(axis=1).sum()),
        "time_not_increasing": int((dt <= 0).sum()),
        "dt_exactly_1s_pct": round(100 * float((dt == 1).mean()), 2),
        "dt_max_s": float(dt.max()), "n_gaps_gt_1s": int((dt > 1).sum()),
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
        "n_components": len(components),
        "largest_component_pct": round(100 * max(map(len, components)) / n_nodes, 2),
        "isolated_nodes": sum(1 for c in components if len(c) == 1),
        "arc_ids_valid": bool(trip.arcs.min() >= 0 and trip.arcs.max() < n_nodes),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for partition in PARTITIONS:
        for trip_id in list_trip_ids(partition):
            rows.append(summarise_trip(load_trip(trip_id, partition)))
    inventory = pd.DataFrame(rows)
    inventory.to_csv(RESULTS / "dataset_inventory.csv", index=False)
    pd.set_option("display.width", 200)
    print(inventory.groupby("partition").agg(
        trips=("trip_id", "count"), obs_min=("n_obs", "min"), obs_med=("n_obs", "median"),
        obs_max=("n_obs", "max"), arcs_med=("n_arcs", "median"),
        route_med=("n_route_arcs", "median"), gaps=("n_gaps_gt_1s", "sum"),
        dt_max=("dt_max_s", "max"), nan_rows=("nan_track_rows", "sum"),
        non_monotonic=("time_not_increasing", "sum"), comps_max=("n_components", "max"),
        isolated=("isolated_nodes", "sum"),
    ).T)
    print("\nlon range:", inventory.lon_min.min(), inventory.lon_max.max(),
          "| lat range:", inventory.lat_min.min(), inventory.lat_max.max())
    print("route arc id out of range:", int((inventory.route_arc_id_max >= inventory.n_arcs).sum()))
    print("invalid node refs in arcs:", int((~inventory.arc_ids_valid).sum()))


if __name__ == "__main__":
    main()

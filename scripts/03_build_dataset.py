"""Build the candidate-pair feature tables for train and student_test.

Writes (gitignored) data/features_<partition>.parquet and data/obs_<partition>.parquet.
Evaluator-test trips are never processed here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data import PARTITIONS, list_trip_ids, load_trip  # noqa: E402
from pathsnap.features import MAX_CANDIDATES, RADIUS_M, build_pair_features  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402
from pathsnap.labels import align_track_to_route  # noqa: E402

BAD_ALIGN_M, BAD_TRIP_FRACTION = 50.0, 0.20


def main() -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    excluded = []
    for partition in PARTITIONS:
        pair_frames, obs_frames = [], []
        started = time.time()
        for trip_id in list_trip_ids(partition):
            trip = load_trip(trip_id, partition)
            track_xy, node_xy = project_trip(trip)
            route_pos, truth_arc, align_dist = align_track_to_route(
                track_xy, trip.timestamps, node_xy, trip.arcs, trip.route)
            if (align_dist > BAD_ALIGN_M).mean() > BAD_TRIP_FRACTION:
                excluded.append({"partition": partition, "trip_id": trip_id,
                                 "pct_obs_far_from_route": round(100 * float((align_dist > BAD_ALIGN_M).mean()), 1),
                                 "reason": "reference route does not cover GPS track"})
                continue
            pairs = build_pair_features(track_xy, timestamps=trip.timestamps, node_xy=node_xy, arcs=trip.arcs)
            pairs["trip_id"] = trip_id
            pairs["y"] = (pairs.arc_id.to_numpy() == truth_arc[pairs.obs_idx.to_numpy()]).astype(int)
            in_candidates = pairs.groupby("obs_idx").y.max().reindex(range(len(truth_arc)), fill_value=0)
            obs_frames.append(pd.DataFrame({
                "trip_id": trip_id, "obs_idx": np.arange(len(truth_arc)), "timestamp": trip.timestamps,
                "truth_arc": truth_arc, "route_pos": route_pos, "align_dist_m": align_dist,
                "truth_in_candidates": in_candidates.to_numpy().astype(bool)}))
            pair_frames.append(pairs)
        pairs = pd.concat(pair_frames, ignore_index=True)
        obs = pd.concat(obs_frames, ignore_index=True)
        pairs.to_parquet(ROOT / "data" / f"features_{partition}.parquet", index=False)
        obs.to_parquet(ROOT / "data" / f"obs_{partition}.parquet", index=False)
        print(f"{partition}: {obs.trip_id.nunique()} trips, {len(obs)} obs, {len(pairs)} pairs, "
              f"positive rate {pairs.y.mean():.3f}, candidate miss rate {100 * (1 - obs.truth_in_candidates.mean()):.2f}%, "
              f"{time.time() - started:.0f}s (radius={RADIUS_M} m, max_candidates={MAX_CANDIDATES})")
    pd.DataFrame(excluded).to_csv(ROOT / "results" / "excluded_trips.csv", index=False)
    print("excluded trips:", excluded)


if __name__ == "__main__":
    main()

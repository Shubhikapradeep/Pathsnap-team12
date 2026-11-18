"""Choose the candidate radius: miss rate vs candidate-set size, on train trips only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.candidates import generate_candidates  # noqa: E402
from pathsnap.data import list_trip_ids, load_trip  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402
from pathsnap.labels import align_track_to_route  # noqa: E402

RADII = (30, 50, 75, 100, 150)
BAD_ALIGN_M, BAD_TRIP_FRACTION = 50.0, 0.20  # trip excluded if reference route does not cover its track
SAMPLE_EVERY = 4  # every 4th train trip -> 20 trips, avoids using student_test for design


def main() -> None:
    records = []
    align_rows = []
    for trip_id in list_trip_ids("train")[::SAMPLE_EVERY]:
        trip = load_trip(trip_id, "train")
        track_xy, node_xy = project_trip(trip)
        _, truth_arc, align_dist = align_track_to_route(
            track_xy, trip.timestamps, node_xy, trip.arcs, trip.route)
        align_rows.append({"trip_id": trip_id, "n_obs": len(truth_arc),
                           "align_median_m": np.median(align_dist),
                           "align_p95_m": np.percentile(align_dist, 95),
                           "align_gt50m_pct": 100 * (align_dist > 50).mean()})
        if (align_dist > BAD_ALIGN_M).mean() > BAD_TRIP_FRACTION:
            print(f"excluding trip {trip_id}: reference route does not cover track")
            continue
        label_ok = np.ones(len(truth_arc), dtype=bool)
        for radius in RADII:
            cand = generate_candidates(track_xy, node_xy, trip.arcs, radius, max_candidates=50)
            cand = cand[label_ok[cand.obs_idx.values]]
            hit = cand[cand.arc_id.values == truth_arc[cand.obs_idx.values]].obs_idx.nunique()
            records.append({"trip_id": trip_id, "radius_m": radius, "n_obs": int(label_ok.sum()),
                            "hits": hit, "n_pairs": len(cand),
                            "obs_with_no_candidate": int(label_ok.sum()) - cand.obs_idx.nunique()})
    align = pd.DataFrame(align_rows)
    study = pd.DataFrame(records).groupby("radius_m").agg(
        n_obs=("n_obs", "sum"), hits=("hits", "sum"), pairs=("n_pairs", "sum"),
        no_cand=("obs_with_no_candidate", "sum"))
    study["miss_rate_pct"] = 100 * (1 - study.hits / study.n_obs)
    study["avg_candidates"] = study.pairs / study.n_obs
    Path(__file__).resolve().parents[1].joinpath("results").mkdir(exist_ok=True)
    study.to_csv(Path(__file__).resolve().parents[1] / "results" / "radius_study.csv")
    align.to_csv(Path(__file__).resolve().parents[1] / "results" / "alignment_quality_sample.csv", index=False)
    print(study.round(2).to_string())
    print("\nalignment quality (20 sampled train trips):")
    print(align.describe().loc[["min", "50%", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()

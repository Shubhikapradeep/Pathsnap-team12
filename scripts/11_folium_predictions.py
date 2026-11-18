"""Folium map comparing GPS, road network, reference route, classifier-only route and classifier + HMM route.

Uses saved model probabilities (data/predictions/<model>_student_test.parquet, written by the model script) and the
tuned decoder parameters (results/hmm_<model>/best_params.json).
Usage: python scripts/11_folium_predictions.py <model> <trip_id>   e.g. lightgbm 00000077
Writes results/maps/student_test_<trip_id>_predictions.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.hmm import HmmParams  # noqa: E402
from pathsnap.hmm_eval import predict_partition  # noqa: E402
from pathsnap.route_geometry import arcs_to_segments, collapse_arc_sequence  # noqa: E402

PARTITION = "student_test"


def main(model: str, trip_id: str) -> None:
    trip = load_trip(trip_id, PARTITION)
    obs = pd.read_parquet(ROOT / "data" / f"obs_{PARTITION}.parquet", filters=[("trip_id", "==", trip_id)])
    pairs = pd.read_parquet(ROOT / "data" / "predictions" / f"{model}_{PARTITION}.parquet", filters=[("trip_id", "==", trip_id)])
    raw = json.loads((ROOT / "results" / f"hmm_{model}" / "best_params.json").read_text())
    params = HmmParams(**{k: v for k, v in raw.items() if k in HmmParams.__dataclass_fields__})
    classifier_arc = predict_partition(pairs, obs, PARTITION, None).pred_arc.to_numpy()
    hmm_arc = predict_partition(pairs, obs, PARTITION, params).pred_arc.to_numpy()
    truth = obs.sort_values("obs_idx").truth_arc.to_numpy()
    top1_classifier, top1_hmm = float((classifier_arc == truth).mean()), float((hmm_arc == truth).mean())

    lat_lon = trip.track_lonlat[:, ::-1]
    fmap = folium.Map(location=lat_lon.mean(axis=0).tolist(), zoom_start=16, tiles="OpenStreetMap")
    fmap.fit_bounds([lat_lon.min(axis=0).tolist(), lat_lon.max(axis=0).tolist()])

    candidate_arcs = np.unique(pairs.arc_id.to_numpy())
    network = folium.FeatureGroup(name="Road network near track", show=True)
    for start, end in trip.arcs[candidate_arcs]:
        folium.PolyLine([trip.node_lonlat[start][::-1].tolist(), trip.node_lonlat[end][::-1].tolist()],
                        color="#95a5a6", weight=2, opacity=0.7).add_to(network)
    network.add_to(fmap)

    def add_route(name: str, arcs: np.ndarray, color: str, weight: int, opacity: float, dash: str | None = None) -> None:
        group = folium.FeatureGroup(name=name, show=True)
        for segment in arcs_to_segments(collapse_arc_sequence(arcs), trip.arcs, trip.node_lonlat):
            folium.PolyLine(segment, color=color, weight=weight, opacity=opacity, dash_array=dash).add_to(group)
        group.add_to(fmap)

    add_route("Reference route (derived from .route)", np.asarray(trip.route), "#27ae60", 9, 0.55)
    add_route("Classifier-only route (fragmented)", classifier_arc, "#e67e22", 4, 0.95)
    add_route("Classifier + HMM route", hmm_arc, "#1f4e79", 4, 0.95, dash="8 6")

    gps = folium.FeatureGroup(name="Raw GPS observations", show=True)
    for index in range(0, len(lat_lon), max(1, len(lat_lon) // 300)):
        folium.CircleMarker(lat_lon[index].tolist(), radius=2.5, color="#c0392b", fill=True, fill_opacity=0.9,
                            tooltip=f"point {index}, t={trip.timestamps[index]:.0f}s").add_to(gps)
    gps.add_to(fmap)

    legend = (f"<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;"
              f"border:1px solid #999;font-size:12px;line-height:1.5'><b>Trip {trip_id} ({PARTITION}), model: {model}</b><br>"
              f"<span style='color:#c0392b'>&#9679;</span> raw GPS points<br>"
              f"<span style='color:#27ae60'>&#9644;</span> reference route<br>"
              f"<span style='color:#e67e22'>&#9644;</span> classifier only (top-1 {top1_classifier:.3f})<br>"
              f"<span style='color:#1f4e79'>&#9644;</span> classifier + HMM (top-1 {top1_hmm:.3f})<br>"
              f"<span style='color:#95a5a6'>&#9644;</span> road network (candidate arcs)</div>")
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    out = ROOT / "results" / "maps" / f"{PARTITION}_{trip_id}_predictions.html"
    fmap.save(str(out))
    print(f"{out.name}: {len(trip.timestamps)} points, top-1 classifier {top1_classifier:.4f} -> HMM {top1_hmm:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

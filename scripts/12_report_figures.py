"""Extra report figures from real data: trajectory + road network plots and confusion matrices.

  results/figures/trajectories_network.png   GPS track, road network and reference route for three trips
  results/figures/confusion_matrices.png     student_test pair-level confusion matrices for every model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402

TRIPS = [("train", "00000034"), ("student_test", "00000052"), ("student_test", "00000077")]


def trajectories() -> None:
    fig, axes = plt.subplots(1, len(TRIPS), figsize=(16, 5.4))
    for ax, (partition, trip_id) in zip(axes, TRIPS):
        trip = load_trip(trip_id, partition)
        track, nodes = project_trip(trip)
        pad = 300.0
        low, high = track.min(axis=0) - pad, track.max(axis=0) + pad
        inside = np.all((nodes[trip.arcs[:, 0]] > low) & (nodes[trip.arcs[:, 0]] < high), axis=1)
        segments = np.stack([nodes[trip.arcs[inside, 0]], nodes[trip.arcs[inside, 1]]], axis=1)
        ax.add_collection(LineCollection(segments, colors="#b0b7bd", linewidths=0.5))
        route = np.stack([nodes[trip.arcs[trip.route, 0]], nodes[trip.arcs[trip.route, 1]]], axis=1)
        ax.add_collection(LineCollection(route, colors="#27ae60", linewidths=2.2, label="reference route"))
        ax.scatter(track[:, 0], track[:, 1], s=2, color="#c0392b", label="raw GPS", zorder=3)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_aspect("equal")
        ax.set_title(f"trip {trip_id} ({partition}): {len(track):,} points", fontsize=10)
        ax.set_xlabel("x (m, trip-local projection)")
        ax.legend(loc="upper left", fontsize=8, markerscale=4)
    fig.suptitle("GPS observations, road network and reference route (local metric projection)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "trajectories_network.png", dpi=140)
    plt.close(fig)


def confusion_matrices() -> None:
    records = sorted((json.loads(p.read_text()) for p in (ROOT / "results" / "experiments").glob("*.json")),
                     key=lambda r: r["name"])
    records = [r for r in records if not r["name"].startswith("_")]
    cols = 3
    rows = int(np.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.9 * rows))
    for ax, record in zip(axes.ravel(), records):
        matrix = np.array(record["student_test"]["confusion_matrix"], dtype=float)
        normalised = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{normalised[i, j]:.3f}\n({int(matrix[i, j]):,})", ha="center", va="center",
                        color="white" if normalised[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks([0, 1], ["not ref.", "reference"])
        ax.set_yticks([0, 1], ["not ref.", "reference"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(record["name"], fontsize=10)
    for ax in axes.ravel()[len(records):]:
        ax.axis("off")
    fig.suptitle("Student-test confusion matrices (candidate-pair level, row-normalised; counts in brackets)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "confusion_matrices.png", dpi=140)
    plt.close(fig)


def nearest_candidate_baseline() -> None:
    """Baseline 0: pick the nearest candidate arc for every GPS point (candidate misses count as wrong)."""
    import pandas as pd
    rows = []
    for partition in ("train", "student_test"):
        pairs = pd.read_parquet(ROOT / "data" / f"features_{partition}.parquet", columns=["dist_rank", "y"])
        obs = pd.read_parquet(ROOT / "data" / f"obs_{partition}.parquet", columns=["trip_id"])
        rows.append({"partition": partition, "n_obs": len(obs), "nearest_candidate_top1": float(pairs[pairs.dist_rank == 0].y.sum() / len(obs))})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "baseline_nearest_candidate.csv", index=False)


if __name__ == "__main__":
    trajectories()
    confusion_matrices()
    nearest_candidate_baseline()
    print("wrote trajectories_network.png, confusion_matrices.png and baseline_nearest_candidate.csv")

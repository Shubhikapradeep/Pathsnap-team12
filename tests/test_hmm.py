"""Unit tests for the HMM/Viterbi decoder on tiny hand-built road networks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.hmm import (ArcGraph, HmmParams, classifier_only_prediction, decode_trip,  # noqa: E402
                          transition_matrix)

# Main road: nodes 0-1-2-3-4 -> arcs 0..3.  Parallel road (not connected): nodes 10..14 -> arcs 4..7.
ARCS = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (10, 11), (11, 12), (12, 13), (13, 14)])


def _pairs(main_p: list[float], parallel_p: list[float]) -> pd.DataFrame:
    rows = []
    for i, (pm, pp) in enumerate(zip(main_p, parallel_p)):
        rows.append({"obs_idx": i, "arc_id": i, "p": pm})
        rows.append({"obs_idx": i, "arc_id": 4 + i, "p": pp})
    return pd.DataFrame(rows)


def test_hmm_fixes_isolated_jump_to_disconnected_road():
    pairs = _pairs([0.6, 0.6, 0.4, 0.6], [0.4, 0.4, 0.6, 0.4])
    assert classifier_only_prediction(pairs, 4).tolist() == [0, 1, 6, 3]  # classifier jumps roads at point 2
    decoded = decode_trip(pairs, np.arange(4.0), ArcGraph(ARCS, depth=6), HmmParams(), n_obs=4)
    assert decoded.tolist() == [0, 1, 2, 3]


def test_points_without_candidates_get_minus_one():
    pairs = _pairs([0.6, 0.6, 0.4, 0.6], [0.4, 0.4, 0.6, 0.4])
    pairs = pairs[pairs.obs_idx != 2]
    decoded = decode_trip(pairs, np.arange(4.0), ArcGraph(ARCS, depth=6), HmmParams(), n_obs=4)
    assert decoded[2] == -1 and decoded[[0, 1, 3]].tolist() == [0, 1, 3]


def test_empty_trip_returns_all_minus_one():
    empty = pd.DataFrame({"obs_idx": [], "arc_id": [], "p": []})
    assert decode_trip(empty, np.arange(3.0), ArcGraph(ARCS, depth=6), HmmParams(), 3).tolist() == [-1, -1, -1]


def test_large_time_gap_cuts_chain_into_independent_segments():
    pairs = _pairs([0.6, 0.55, 0.45, 0.6], [0.4, 0.45, 0.55, 0.4])
    graph, params = ArcGraph(ARCS, depth=6), HmmParams()
    timestamps = np.array([0.0, 1.0, 100.0, 101.0])
    full = decode_trip(pairs, timestamps, graph, params, n_obs=4)
    first = decode_trip(pairs[pairs.obs_idx < 2], timestamps, graph, params, n_obs=4)
    second = decode_trip(pairs[pairs.obs_idx >= 2], timestamps, graph, params, n_obs=4)
    assert full.tolist() == np.where(first >= 0, first, second).tolist()


def test_immediate_u_turn_is_penalised():
    arcs = np.array([(0, 1), (1, 0), (1, 2)])  # arc 1 is the reverse twin of arc 0
    pairs = pd.DataFrame({"obs_idx": [0, 1, 1], "arc_id": [0, 1, 2], "p": [0.9, 0.5, 0.5]})
    decoded = decode_trip(pairs, np.array([0.0, 1.0]), ArcGraph(arcs, depth=3), HmmParams(), n_obs=2)
    assert decoded.tolist() == [0, 2]


def test_transition_matrix_scores():
    graph, params = ArcGraph(ARCS, depth=6), HmmParams()
    scores = transition_matrix(graph, np.array([0]), np.array([0, 1, 2, 5]), params)
    assert scores[0].tolist() == [0.0, -params.hop_penalty, -2 * params.hop_penalty, -params.break_penalty]


def test_hops_beyond_max_hops_count_as_break():
    graph = ArcGraph(ARCS, depth=6)
    near = transition_matrix(graph, np.array([0]), np.array([3]), HmmParams(max_hops=3))
    far = transition_matrix(graph, np.array([0]), np.array([3]), HmmParams(max_hops=2))
    assert near[0, 0] == -3 * 0.5 and far[0, 0] == -6.0

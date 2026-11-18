"""HMM / Viterbi route reconstruction on top of classifier probabilities.

Model
-----
* Hidden state at GPS point i: which of its candidate arcs the vehicle is on.
  The state set differs per point (its candidate set), and each state keeps its
  arc id, so the decoded path is directly a sequence of road arcs.
* Emission score of candidate b at point i: log(classifier probability), floored at
  `probability_floor` so a zero never makes a state impossible.
* Transition score from arc a (point i-1) to arc b (point i), in log space:
    - same arc                                  0
    - b reachable from a in h arc-steps (h<=max_hops)   -hop_penalty * h
      (h = 1 means "next arc after a"; an immediate U-turn onto the reverse twin
      additionally pays `u_turn_penalty`)
    - not reachable within max_hops             -break_penalty   (soft break, never -inf,
      so a path always exists; a break is merely expensive)
* Gaps: if two consecutive scored points are more than `gap_seconds` apart, road
  connectivity says nothing reliable about the unobserved stretch, so the chain is cut and
  each side is decoded independently (no transition constraint across the cut).
* Points with no candidate arc get prediction -1 and are skipped; the chain continues
  across them unless the time gap triggers a cut.

Viterbi recurrence (per segment):
    delta_0(b)  = emission_0(b)
    delta_t(b)  = emission_t(b) + max_a [ delta_{t-1}(a) + transition(a, b) ]
    psi_t(b)    = argmax_a [ ... ]     (back-pointer)
The best final state is traced back through psi. Cost O(T * K^2) for T points, K candidates.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HmmParams:
    """Decoder parameters. Tuned on train out-of-fold probabilities and student_test only."""

    hop_penalty: float = 0.5
    break_penalty: float = 6.0
    max_hops: int = 4
    u_turn_penalty: float = 2.0
    gap_seconds: float = 10.0
    probability_floor: float = 1e-6


class ArcGraph:
    """Arc-level successor graph with cached breadth-first hop distances.

    Arc a precedes arc b when the end node of a is the start node of b.
    """

    def __init__(self, arcs: np.ndarray, depth: int) -> None:
        self.arcs = arcs
        self.depth = depth
        self._outgoing: dict[int, list[int]] = defaultdict(list)
        for arc_id, start_node in enumerate(arcs[:, 0].tolist()):
            self._outgoing[start_node].append(arc_id)
        self._hops_cache: dict[int, dict[int, int]] = {}

    def hops_from(self, arc_id: int) -> dict[int, int]:
        """Map arc -> minimum number of arc steps from arc_id (0 = itself), up to depth."""
        cached = self._hops_cache.get(arc_id)
        if cached is not None:
            return cached
        reached = {arc_id: 0}
        frontier = [arc_id]
        for hop in range(1, self.depth + 1):
            next_frontier = []
            for current in frontier:
                for successor in self._outgoing.get(int(self.arcs[current, 1]), ()):
                    if successor not in reached:
                        reached[successor] = hop
                        next_frontier.append(successor)
            frontier = next_frontier
            if not frontier:
                break
        self._hops_cache[arc_id] = reached
        return reached

    def is_reverse_twin(self, arc_a: int, arc_b: int) -> bool:
        """True when b runs the opposite way along the same two nodes as a."""
        return bool(self.arcs[arc_b, 0] == self.arcs[arc_a, 1] and self.arcs[arc_b, 1] == self.arcs[arc_a, 0])


def transition_matrix(graph: ArcGraph, previous_arcs: np.ndarray, current_arcs: np.ndarray,
                      params: HmmParams) -> np.ndarray:
    """(K_prev, K_cur) matrix of log transition scores."""
    scores = np.full((len(previous_arcs), len(current_arcs)), -params.break_penalty)
    current_index = {int(arc): j for j, arc in enumerate(current_arcs)}
    for i, arc_a in enumerate(previous_arcs.tolist()):
        for arc_b, hops in graph.hops_from(arc_a).items():
            j = current_index.get(arc_b)
            if j is None or hops > params.max_hops:
                continue
            score = -params.hop_penalty * hops
            if hops == 1 and graph.is_reverse_twin(arc_a, arc_b):
                score -= params.u_turn_penalty
            scores[i, j] = score
    return scores


def decode_trip(pairs: pd.DataFrame, timestamps: np.ndarray, graph: ArcGraph,
                params: HmmParams, n_obs: int) -> np.ndarray:
    """Viterbi-decode one trip.

    pairs: columns obs_idx, arc_id, p (classifier probability) for this trip only.
    Returns an int array of length n_obs with the predicted arc id, or -1 where the
    point has no candidates.
    """
    predicted = np.full(n_obs, -1, dtype=np.int64)
    if len(pairs) == 0:
        return predicted
    pairs = pairs.sort_values("obs_idx", kind="stable")
    obs_column = pairs.obs_idx.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(obs_column)) + 1]
    arc_groups = np.split(pairs.arc_id.to_numpy(), starts[1:])
    emission_groups = np.split(np.log(np.clip(pairs.p.to_numpy(), params.probability_floor, 1.0)), starts[1:])
    obs_ids = obs_column[starts]

    back_pointers: dict[int, np.ndarray] = {}
    segment_start = 0
    delta = emission_groups[0]

    def trace_back(segment_end: int) -> None:
        state = int(np.argmax(delta))
        for t in range(segment_end, segment_start - 1, -1):
            predicted[obs_ids[t]] = arc_groups[t][state]
            if t > segment_start:
                state = int(back_pointers[t][state])

    for t in range(1, len(obs_ids)):
        elapsed = timestamps[obs_ids[t]] - timestamps[obs_ids[t - 1]]
        if elapsed > params.gap_seconds:
            trace_back(t - 1)
            segment_start = t
            delta = emission_groups[t]
            continue
        transition = transition_matrix(graph, arc_groups[t - 1], arc_groups[t], params)
        candidate_scores = delta[:, None] + transition
        best_previous = candidate_scores.argmax(axis=0)
        delta = candidate_scores[best_previous, np.arange(len(arc_groups[t]))] + emission_groups[t]
        back_pointers[t] = best_previous
    trace_back(len(obs_ids) - 1)
    return predicted


def classifier_only_prediction(pairs: pd.DataFrame, n_obs: int) -> np.ndarray:
    """Baseline: independent highest-probability candidate per point (-1 if none)."""
    predicted = np.full(n_obs, -1, dtype=np.int64)
    best = pairs.loc[pairs.groupby("obs_idx").p.idxmax()]
    predicted[best.obs_idx.to_numpy()] = best.arc_id.to_numpy()
    return predicted

"""Parity and leakage tests for the inference pipeline. Needs the local data (skipped on a fresh clone)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data import DEFAULT_DATA_ROOT, load_trip  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS, build_pair_features  # noqa: E402
from pathsnap.geo import project_trip  # noqa: E402
from pathsnap.hmm import HmmParams  # noqa: E402
from pathsnap.hmm_eval import evaluate_routes, predict_partition  # noqa: E402
from pathsnap.pipeline import Artifacts, run_pipeline  # noqa: E402
from pathsnap.scoring import score_result  # noqa: E402

TRIP_ID = "00000052"
FEATURES = ROOT / "data" / "features_student_test.parquet"
pytestmark = pytest.mark.skipif(not FEATURES.exists() or not (Path(DEFAULT_DATA_ROOT) / "student_test" / TRIP_ID).exists(),
                                reason="needs data/ and PATHSNAP_DATA (run scripts/03_build_dataset.py)")


@pytest.fixture(scope="module")
def artifacts() -> Artifacts:
    from sklearn.ensemble import HistGradientBoostingClassifier
    pairs = pd.read_parquet(ROOT / "data" / "features_student_test.parquet")
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(pairs[FEATURE_COLUMNS], pairs.y)
    return Artifacts(model, HmmParams(hop_penalty=1.0, break_penalty=10.0, max_hops=5, u_turn_penalty=5.0), "test-model")


def test_features_from_raw_files_match_training_features():
    trip = load_trip(TRIP_ID, "student_test", load_route=False)
    track_xy, node_xy = project_trip(trip)
    fresh = build_pair_features(track_xy, trip.timestamps, node_xy, trip.arcs).sort_values(["obs_idx", "arc_id"]).reset_index(drop=True)
    stored = pd.read_parquet(FEATURES, filters=[("trip_id", "==", TRIP_ID)]).sort_values(["obs_idx", "arc_id"]).reset_index(drop=True)
    assert len(fresh) == len(stored)
    assert (fresh.obs_idx.values == stored.obs_idx.values).all() and (fresh.arc_id.values == stored.arc_id.values).all()
    np.testing.assert_allclose(fresh[FEATURE_COLUMNS].to_numpy(float), stored[FEATURE_COLUMNS].to_numpy(float), rtol=1e-9, atol=1e-9)


def test_pipeline_metrics_equal_offline_evaluation(artifacts):
    trip = load_trip(TRIP_ID, "student_test", load_route=False)
    result = run_pipeline(trip, artifacts)
    online = score_result(load_trip(TRIP_ID, "student_test"), result)["metrics"]

    obs = pd.read_parquet(ROOT / "data" / "obs_student_test.parquet")
    obs = obs[obs.trip_id == TRIP_ID]
    pairs = pd.DataFrame({"trip_id": TRIP_ID, "obs_idx": result.candidates.obs_idx, "arc_id": result.candidates.arc_id})
    stored = pd.read_parquet(FEATURES, filters=[("trip_id", "==", TRIP_ID)])
    pairs = stored[["trip_id", "obs_idx", "arc_id"]].copy()
    pairs["p"] = artifacts.classifier.predict_proba(stored[FEATURE_COLUMNS])[:, 1]
    hmm_offline = evaluate_routes(obs, predict_partition(pairs, obs, "student_test", artifacts.hmm_params), "student_test")
    base_offline = evaluate_routes(obs, predict_partition(pairs, obs, "student_test", None), "student_test")

    for key in ("top1_accuracy", "route_continuity", "route_arc_f1"):
        assert online[key] == pytest.approx(hmm_offline[key], abs=1e-4), key
        assert online["classifier_only"][key] == pytest.approx(base_offline[key], abs=1e-4), key


def test_prediction_path_never_sees_the_reference_route(artifacts):
    trip = load_trip(TRIP_ID, "student_test", load_route=False)
    assert trip.route is None
    result = run_pipeline(trip, artifacts)
    assert result.n_points == len(trip.timestamps) and len(result.hmm_arc) == result.n_points
    with pytest.raises(ValueError):
        score_result(trip, result)            # scoring without the server-side route must fail loudly


def test_payload_shapes(artifacts):
    result = run_pipeline(load_trip(TRIP_ID, "student_test", load_route=False), artifacts)
    assert len(result.gps) == result.n_points and len(result.gps[0]) == 3
    assert set(result.hmm_route) == {"arc_ids", "segments"} and result.timing["seconds"] > 0

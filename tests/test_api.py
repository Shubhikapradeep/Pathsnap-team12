"""Unit tests for the PathSnap FastAPI backend using TestClient."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Force stub mode for fast, deterministic unit test
os.environ["USE_STUB"] = "1"
os.environ["PATHSNAP_DATA"] = str(ROOT.parent / "Student folder" / "Student folder")

from api.main import app  # noqa: E402


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["contract_version"] == "1.0"
    assert "model" in data
    assert "version" in data


def test_trips_endpoint(client: TestClient):
    response = client.get("/trips")
    assert response.status_code == 200
    trips = response.json()
    assert isinstance(trips, list)
    assert len(trips) > 0
    first = trips[0]
    assert "trip_id" in first
    assert "n_points" in first
    assert first["n_points"] > 0


def test_predict_endpoint(client: TestClient):
    trips_resp = client.get("/trips")
    trip_id = trips_resp.json()[0]["trip_id"]

    response = client.post(f"/predict/{trip_id}")
    assert response.status_code == 200
    data = response.json()

    # Contract schema validation
    assert data["trip_id"] == trip_id
    assert data["contract_version"] == "1.0"
    assert data["n_points"] > 0
    assert len(data["gps"]) == data["n_points"]

    # Check GPS coordinates are [lat, lon, t_seconds]
    first_gps = data["gps"][0]
    assert len(first_gps) == 3

    # Check routes
    assert "classifier_route" in data
    assert "hmm_route" in data
    assert "arc_ids" in data["hmm_route"]
    assert "segments" in data["hmm_route"]

    # Check per-point predictions
    assert len(data["per_point"]["hmm_arc"]) == data["n_points"]
    assert len(data["per_point"]["classifier_arc"]) == data["n_points"]

    # Check metrics
    metrics = data["metrics"]
    assert "top1_accuracy" in metrics
    assert "route_continuity" in metrics
    assert "classifier_only" in metrics

    # Hard rule: Reference labels must stay server-side and never leak to the UI
    assert "truth_arc" not in data
    assert "reference_route" not in data
    assert "route" not in data["gps"][0]


def test_predict_404(client: TestClient):
    response = client.post("/predict/non_existent_trip_99999999")
    assert response.status_code == 404


if __name__ == "__main__":
    with TestClient(app) as test_client:
        print("Running test_health_endpoint...")
        test_health_endpoint(test_client)
        print("Running test_trips_endpoint...")
        test_trips_endpoint(test_client)
        print("Running test_predict_endpoint...")
        test_predict_endpoint(test_client)
        print("Running test_predict_404...")
        test_predict_404(test_client)
        print("All API tests passed successfully!")


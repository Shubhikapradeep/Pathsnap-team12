# PathSnap – Team 12

Map-matching noisy GPS trajectories to a road network using machine learning and Hidden Markov Model (HMM) Viterbi decoding.

## Architecture
- **Pipeline**: GPS track -> Candidate arcs (radius 100m) -> Pairwise feature engineering -> Classifier scoring -> HMM Viterbi route reconstruction.
- **Laptop 1 (Backend API)**: FastAPI + Uvicorn server running the matching pipeline and internal evaluation.
- **Laptop 2 (Evaluator UI)**: Interactive Leaflet frontend visualizing GPS observations, disconnected classifier-only routes, and connected HMM routes.

---

## Running the API (Laptop 1)

### 1. Setup & Environment
Ensure requirements are installed in `.venv`:
```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set the path to the dataset folder (containing `student_test/` and `train/`):
```powershell
$env:PATHSNAP_DATA = "D:\ML Hackthon\Student folder\Student folder"
```

### 2. Start the Server
Bind to `0.0.0.0` on port `8000` so Laptop 2 can access the API across the mobile hotspot:
```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

To run with distance-based stub scoring (for testing without trained models):
```powershell
$env:USE_STUB = "1"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Verification & Tests
Run the API unit test suite:
```powershell
.\.venv\Scripts\python.exe tests/test_api.py
```

---

## Two-Laptop Hotspot Setup (Demo Mode)

1. Turn on **Personal Hotspot** on a mobile phone.
2. Connect **both Laptop 1 and Laptop 2** to this hotspot.
3. On Laptop 1 (Windows), open PowerShell and check IP:
   ```powershell
   ipconfig
   ```
   Note the IPv4 address under the Wi-Fi adapter (e.g. `192.168.43.50`).
4. Start the API on Laptop 1:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
5. On Laptop 2, open the UI and set the API Base URL to:
   ```
   http://<Laptop-1-IP>:8000
   ```
   Click **Test Connection** (`/health`).
6. Select any trip to run prediction. The evaluator can observe the live request logs in Laptop 1's console:
   ```text
   PREDICT trip=00000052 status=200 top1=0.9695 continuity=0.9617 seconds=0.18
   ```

---

## API Endpoints
- `GET /health` -> Health check, model name, and git revision.
- `GET /trips` -> List available trip IDs and point counts.
- `POST /predict/{trip_id}` (or `GET /predict/{trip_id}`) -> Runs full pipeline and returns predicted route polylines and server-side accuracy metrics.

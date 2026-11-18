# PathSnap – map-matching GPS traces to a road network (Team 12)

Noisy 1 Hz GPS points are matched to the arcs of a road network, and a connected route is reconstructed.

```
GPS point ──► candidate arcs (<=100 m) ──► classifier scores each (point, arc) pair ──► HMM / Viterbi ──► connected route
                                                                                              │
                                                   FastAPI (Laptop 1) ◄── UI (Laptop 2) ◄─────┘
```

* **Stage 1, classifier:** for every GPS point, list nearby candidate arcs and score "is this the true arc?" (19 features per pair).
* **Stage 2, HMM + Viterbi:** pick the arc sequence that is both well scored and connected on the road graph.
* Candidate generation, features, models and decoder are in `src/pathsnap/`; experiments are scripts in `scripts/`.

## Status
| Component | State |
|---|---|
| Data processing, labels, candidates, features | done |
| Model experiments (6 distinct models + variants), comparison | done |
| HMM / Viterbi decoder and tuning | done (final-model run pending) |
| Bayesian hyperparameter optimisation | in progress (`docs/BAYESIAN_OPT.md`) |
| Inference pipeline (`src/pathsnap/pipeline.py`) | done |
| FastAPI backend, evaluator UI | in progress (`docs/API_CONTRACT.md`, `docs/UI_SPEC.md`) |

Planning and ownership: `docs/PLAN.md`, `docs/REMAINING_WORK.md`.

## Installation
Developed and tested on Python 3.14; other recent Python 3 versions should work but are untested. No GPU is needed; all models are CPU.
```bash
git clone https://github.com/Shubhikapradeep/Pathsnap-team12.git
cd Pathsnap-team12
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
macOS: LightGBM/XGBoost need OpenMP: `brew install libomp`. Details and standby models: `docs/STANDBY_MODELS.md`.

## Data
The data is not in the repository. Point to the folder that contains `train/` and `student_test/` (one sub-folder per trip with
`.track`, `.nodes`, `.arcs`, `.route`):
```bash
export PATHSNAP_DATA="/path/to/Student folder"
```
The evaluator trips are never used for training, tuning or model choice, and must never be committed.

## Reproducing the results (run from the repo root)
```bash
# 1. dataset inventory and data-quality summary          -> results/dataset_inventory.csv
.venv/bin/python scripts/01_eda.py
# 2. candidate radius study                              -> results/radius_study.csv
.venv/bin/python scripts/02_radius_study.py
# 3. labels, candidates, features (writes data/*.parquet, not committed)
.venv/bin/python scripts/03_build_dataset.py
# 4. train and evaluate each model (5-fold GroupKFold by trip + student_test) -> results/experiments/<model>.json
.venv/bin/python scripts/models/train_logreg.py
.venv/bin/python scripts/models/train_knn.py
.venv/bin/python scripts/models/train_random_forest.py
.venv/bin/python scripts/models/train_extra_trees.py
.venv/bin/python scripts/models/train_lightgbm.py
.venv/bin/python scripts/models/train_mlp.py
# 5. comparison table and figure                         -> results/model_comparison_table.*, results/figures/
.venv/bin/python scripts/06_compare_models.py
# 6. HMM tuning and classifier-only vs +HMM comparison   -> results/hmm_<model>/
.venv/bin/python scripts/05_tune_hmm.py --probs lightgbm
.venv/bin/python scripts/07_plot_hmm_effect.py lightgbm mlp
# 7. feature evidence and Folium map
.venv/bin/python scripts/08_feature_figures.py
.venv/bin/python scripts/09_folium_trip.py student_test 00000052     # -> results/maps/
# tests
.venv/bin/python -m pytest -q tests
```
Seeds are fixed (42). Each model script logs its configuration, folds, timings and metrics in `results/experiments/<model>.json`.
Step 6 needs the saved probabilities that step 4 writes to `data/predictions/`.

## Repository layout
```
src/pathsnap/   data.py geo.py labels.py candidates.py features.py evaluation.py hmm.py hmm_eval.py
                route_geometry.py pipeline.py scoring.py
scripts/        numbered pipeline scripts, models/ (one script per model), models/standby/, plots/
api/            FastAPI backend (schemas.py = response contract)
ui/             evaluator interface (ui/mock = example responses)
tests/          unit and parity tests
results/        experiment logs, comparison tables, figures, Folium maps, HMM results
docs/           plan, task sheets, API contract, UI spec, Bayesian spec, per-model notes (docs/model_notes)
```

## Method in brief
* **Projection.** Each trip is a different city, so every trip is projected to its own local azimuthal-equidistant plane (metres).
* **Reference labels.** `.route` has no timestamps; per-point reference arcs are derived by a monotone dynamic-programming alignment
  (`src/pathsnap/labels.py`). These are derived reference labels, not independently verified ground truth.
* **Excluded trips.** Train trips 00000033 and 00000054 are excluded because their reference route covers only part of the GPS track
  (`results/excluded_trips.csv`).
* **Candidates.** All arcs within 100 m, nearest 30 kept (miss rate 0.16% in the radius study, `results/radius_study.csv`).
* **Validation.** 5-fold GroupKFold by trip (no trip is split across folds), then student_test. Metrics: macro F1, log loss,
  observation-level top-1 accuracy, route continuity; definitions in `src/pathsnap/evaluation.py`.
* **HMM.** States are the candidate arcs per point; emission = classifier probability; transition = penalty by hop distance on the arc graph;
  chains are cut at time gaps above 10 s (`src/pathsnap/hmm.py`).

## Known limitations
* Reference labels are derived and may be truncated or imperfect for some trips.
* Arcs are straight segments between nodes; curved roads are approximated.
* Some trips contain very large time gaps; the decoder cuts the chain there.
* Free-form results depend on library versions: tree ensembles can differ slightly across environments.

## Use of AI tools
[Team to complete according to the event's disclosure policy.]

## Team
Team 12: Shubhika, Atharva, Yashvantha, Tanish.

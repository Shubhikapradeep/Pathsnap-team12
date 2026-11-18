# Remaining work after the first review

Status snapshot: features, six distinct models, the comparison tooling and the HMM decoder are done and merged.
Still missing on GitHub: final model choice, Bayesian tuning, inference pipeline, API, UI, README, notes for logreg and KNN, the report.
Owners below are the current plan; change them in `docs/PLAN.md` if needed.

## Critical path (each step unblocks the next)
| # | Task | Owner | Needs | Done when |
|---|---|---|---|---|
| 1 | **Choose the final model**: run the HMM on the top candidates (random_forest, extra_trees, lightgbm, mlp) and compare top-1 *after decoding* | Shubhika | model probabilities | table in `results/` + a written reason |
| 2 | **Bayesian optimisation** of the chosen model (`docs/BAYESIAN_OPT.md`) | Yashvantha (fallback Tanish) | 1 | trials CSV, best params, tuned vs untuned table, notes |
| 3 | **Re-run HMM tuning** on the tuned model's probabilities; fix decoder params | Shubhika | 2 | `results/hmm_<model>_tuned/` |
| 4 | **Inference pipeline** `src/pathsnap/pipeline.py`: raw trip -> project -> candidates -> features -> classifier -> HMM -> routes. No labels inside | Shubhika | 1 (final model can be swapped in later) | unit test on student_test trips reproduces offline metrics |
| 5 | **Ship the model artifacts**: final model file, HMM params, feature list, versions, in `models/` (check size; ask before committing a big file) | Shubhika | 3 | API loads them with no training code |
| 6 | **API** `api/main.py` per `docs/API_CONTRACT.md`: wraps the pipeline, loads `.route` server-side for metrics, logs every request | Atharva | 4 (stub first) | all 10 student_test trips return valid `PredictResponse` |
| 7 | **UI** `ui/` per `docs/UI_SPEC.md` | Tanish | mock now, API later | all 10 trips work through the real API |
| 8 | **Two-laptop rehearsal on the hotspot** with the evaluator trips, API terminal visible | all | 6, 7 | all 10 evaluator trips run; screenshots saved |

## In parallel (no blockers)
* **Atharva**: notes `docs/model_notes/logreg.md` and `knn.md` (missing). Explain: KNN train macro F1 is 1.000 because each training point is its own neighbour (not a real fit signal); balanced class weights hurt logreg.
* **Tanish**: move `docs/lightgbm.md` and `docs/mlp.md` into `docs/model_notes/`; add the LightGBM search-space section.
* **Yashvantha**: confirm RF/ET notes cover A/B/C questions; start the Bayesian script skeleton.
* **Shubhika**: README, ablation (below), Folium maps with predicted routes, report skeleton.

## Report evidence still to produce
* **Ablation** (one feature group removed at a time: distance, heading, context, graph) with group-CV; supports feature claims.
* **Folium maps** with GPS, road network, reference route, classifier-only route and HMM route, legend and layer control, for 2-3 trips (one good, one hard such as 00000077, one where the HMM did not help). Code in `scripts/09_folium_trip.py` (extend with predicted layers after step 4).
* **Confusion matrices** per model from `results/experiments/*.json` (`student_test.confusion_matrix`).
* **HMM section**: classifier-only versus +HMM table, per-trip figure, and the failure cases.
* **Screenshots**: UI, and API request logs on Laptop 1 (report section 7).

## Report assembly (Shubhika), sections from `Deliverables.pdf`
1 Dataset and EDA (Folium, missingness, trajectories, road network) · 2 Model comparison table (>=5 models, actual numbers) ·
3 One subsection per model (A: suitability and limits, B: hyperparameters and loss, C: results and over/underfitting) ·
4 Model visualisations (CV F1, test top-1 and F1, log loss, confusion matrices, MLP loss curve) ·
5 Bayesian optimisation ·
6 HMM/Viterbi ·
7 Folium maps ·
8 API, UI and deployment screenshots.
Rule: every number must come from a file under `results/` made by committed code. Each person writes their own model sections in their own words.

## Repo requirements checklist (final push)
- [ ] README with install, run and deployment steps (two-laptop instructions included)
- [ ] `requirements.txt` complete; nothing hardcoded (API address, data paths are settings)
- [ ] All code for data processing, features, training, evaluation, Bayesian search, HMM, API, UI, Folium maps
- [ ] Results CSV/JSON and logs committed; saved final model if size allows
- [ ] No secrets, no `data/`, no evaluator data in the repo
- [ ] Repo set to **Public** shortly before submission
- [ ] Commit history shows each member's own work (own identity)

## Risks to watch
* An evaluator trip may have a truncated reference route (like train trips 33 and 54). The API must show a warning, not hide it or change the pipeline.
* Largest evaluator trip has 6,854 points: check API latency (target under 30 s).
* The final model file must reach Atharva's laptop outside git if it is gitignored or large.
* Hotspot IP changes: re-check before the demo.
* AI-assistance disclosure and viva readiness: everyone must be able to explain their own parts (see the private viva guide).

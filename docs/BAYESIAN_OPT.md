# Bayesian hyperparameter optimisation – spec

**Owner: Yashvantha** (non-Mac machine, lightest delivered load so far). **Fallback: Tanish**, who owns LightGBM and
could also run it: one LightGBM fit is only about 11 s on a Mac, so no GPU or special hardware is needed.
Tanish supplies the LightGBM search-space rationale in `docs/model_notes/lightgbm.md`; Yashvantha writes and runs the code.
Branch: `feat/bayesian-opt`. Commits under your own identity.

## Which model, and why  (decide with Shubhika before starting)
Do not start until the model is chosen. Pointwise results now put four models within about 1 point of top-1
(student-test): Random Forest 0.878, MLP 0.876, Extra Trees 0.873, LightGBM 0.870 (`results/experiments/*.json`).
Random Forest also has the best CV macro F1 (0.931) and log loss (0.032), but it overfits the most (train minus CV macro F1 = 0.038).
The ranking can change after HMM decoding: it already did for MLP versus LightGBM (LightGBM 0.928, MLP 0.908; `results/hmm_lightgbm/`,
`results/hmm_mlp/`). So the rule is:

1. Run `python scripts/05_tune_hmm.py --probs <model>` for the top candidates (random_forest, extra_trees, lightgbm, mlp).
2. Tune the model with the best top-1 **after HMM decoding**. If two are within noise, prefer the one that is cheaper to tune and has more
   regularisation knobs. LightGBM fits in about 11 s per fold; Random Forest on 800k rows takes much longer per trial.
3. Write the choice and the evidence in the report (model-selection section).

The search space below is written for LightGBM. If another model wins, replace it with that model's parameters (for Random Forest:
`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `max_samples`) and keep everything else in this spec unchanged.

## Hard rules
* Optimise on **group-CV over the train trips only** (same `GroupKFold(5)` by trip as `src/pathsnap/evaluation.py`).
* **Student-test is never used inside a trial.** It is used once, after the search, to compare tuned with untuned.
* Evaluator data is never touched.
* Fixed seeds everywhere (`TPESampler(seed=42)`, `random_state=42`). Keep the full trial record.

## Objective
**Minimise mean 5-fold CV log loss** (pair level). Why: the HMM uses the classifier's probabilities as emission scores, so probability
quality matters more than a 0.5-threshold F1, and log loss is a proper scoring rule. Record CV macro F1 and CV observation top-1 for every
trial as user attributes, and report all three. Final sanity check: re-run the HMM on the tuned probabilities and confirm top-1 after
decoding does not get worse (see Deliverables). If you prefer another objective, write the reason in the report.

## Search space (starting point; edit with a reason)
| Parameter | Range | Scale | Why |
|---|---|---|---|
| `num_leaves` | 15 to 255 | log | tree complexity; main overfitting lever |
| `learning_rate` | 0.01 to 0.2 | log | step size; trades off with `n_estimators` |
| `n_estimators` | 100 to 600 | int | number of boosting rounds |
| `min_child_samples` | 20 to 500 | log | larger = smoother, less overfit |
| `subsample` | 0.5 to 1.0 | linear | row sampling (keep `subsample_freq=1`) |
| `colsample_bytree` | 0.5 to 1.0 | linear | feature sampling |
| `reg_lambda` | 1e-3 to 10 | log | L2 regularisation |
| `reg_alpha` | 1e-3 to 10 | log | L1 regularisation |

Fixed: `random_state=42`, `n_jobs=-1`, `verbose=-1`, `subsample_freq=1`. Untuned baseline = `results/experiments/lightgbm.json`
(n_estimators 300, learning_rate 0.05, num_leaves 63, subsample 0.8, colsample_bytree 0.8).

## Method
* Library: **Optuna** (already in `requirements.txt`), `TPESampler(seed=42)` (Bayesian, tree-structured Parzen estimator), `direction="minimize"`.
* Script: `scripts/tune_bayesian.py`. Reuse `GroupKFold`, `load_partition`, `score_predictions` and `FEATURE_COLUMNS` from `pathsnap.evaluation` /
  `pathsnap.features`; do not call `run_experiment` inside the objective (it also evaluates on student-test).
* Report each fold's log loss with `trial.report(...)` and use `MedianPruner(n_startup_trials=8)` so bad trials stop early.
* Trials: **30** (about 40 minutes at roughly 75 s per trial, from 11 s per fold measured on Tanish's Mac). Cap with `timeout=5400`. Print the
  expected run scale before starting and record the actual total time in the report (cost of the search).
* Storage: SQLite file `results/bayesian/optuna_study.db` (`storage="sqlite:///..."`, `load_if_exists=True`) so an interrupted run resumes.

## Deliverables (all committed)
1. `scripts/tune_bayesian.py`
2. `results/bayesian/trials.csv` (`study.trials_dataframe()`: every trial, params, value, state, duration, user attributes)
3. `results/bayesian/best_params.json`
4. `results/bayesian/optuna_study.db` (the search record the policy asks us to retain; small)
5. Refit with the best params through the shared harness under a new name: `NAME = "lightgbm_tuned"` in `scripts/models/train_lightgbm_tuned.py`
   -> `results/experiments/lightgbm_tuned.json`. Same protocol and partitions as the untuned model.
6. Figures from real trial data: optimisation history (best value vs trial) and parameter importance
   (`optuna.importance.get_param_importances`) saved under `results/figures/`.
7. HMM check: `python scripts/05_tune_hmm.py --probs lightgbm_tuned` (needs `data/predictions/lightgbm_tuned_*.parquet`, which the harness writes).
8. `docs/model_notes/bayesian.md` answering, in your own words and from the results:
   1. Which hyperparameters were searched? 2. What ranges or distributions? 3. What objective? 4. How many trials?
   5. Best values? 6. Did tuned beat untuned? Table with the same metrics on the same partitions: CV macro F1, student-test top-1,
   student-test macro F1, log loss, inference time (and top-1 after HMM).

## What to expect (do not pre-write conclusions)
The untuned model is already strong, so gains may be small. A modest or null improvement is a valid result: report it honestly with the
numbers, and say what the search showed about which parameters mattered. Never tune on student-test results.

## Be ready to explain in the viva
Why Bayesian rather than grid or random search; what TPE does (models the density of good and bad trials and samples where good/bad is largest);
what the pruner does; why log loss; why the student-test set stays outside the loop; how many trials and why that many.

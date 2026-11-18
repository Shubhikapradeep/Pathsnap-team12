# Standby models and Mac notes

## No GPU needed
Everything here is tabular (about 19 features, about 3M candidate rows) and runs on CPU. Nobody needs CUDA.
Apple Silicon and Intel Macs are both fine. Do not install PyTorch/TensorFlow for this task.

## Mac setup gotchas
* **LightGBM and XGBoost need OpenMP on macOS.** If `import lightgbm` fails with a `libomp` error:
  `brew install libomp`, then re-run.
* Use the repo venv (`.venv/bin/python ...`), not the system Python.
* Keep the laptop plugged in and run long jobs with the lid open (macOS sleeps jobs). `caffeinate -i <command>`
  keeps it awake.
* If a run is too slow or runs out of memory, lower `max_train_rows` in the script and **write the new value in
  `CONFIG`** so the log is honest. Evaluation always uses all rows, so results stay comparable.
  Rough guide: KNN 200k, MLP 1M, Random Forest / Extra Trees 800k, boosting can use everything.
* Run one model at a time on a laptop; `n_jobs=-1` already uses every core.

## Standby models (CPU, tested to build and fit)
Located in `scripts/models/standby/`. Same harness, same metrics, so they slot straight into the comparison table.

| Script | Family | Use it when |
|---|---|---|
| `train_hist_gradient_boosting.py` | boosting (sklearn) | LightGBM will not install or run on a Mac |
| `train_xgboost.py` | boosting (XGBoost, `hist`) | a second boosting model is useful for comparison |
| `train_gaussian_nb.py` | probabilistic | a very fast extra baseline; likely poor probabilities because features are correlated |
| `train_decision_tree.py` | tree | a cheap, interpretable single tree; good for the over/underfitting discussion |

Rules:
* Primary assignments stay as in `TASKS_<name>.md`. Standby is for failures, or extra comparison points.
* The minimum is five models; we plan six. Everything you actually run must be documented in the report, so do
  not run standby models and hide them.
* Whoever runs a standby model commits the script, results and a `docs/model_notes/<model>.md` under their own
  name, and must be able to explain it.

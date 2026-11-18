
"""Bayesian hyperparameter optimisation for PathSnap.

Optimises LightGBM using 5-fold GroupKFold on training trips only.
Student-test and evaluator data are never accessed during trials.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from pathsnap.evaluation import load_partition, score_predictions
from pathsnap.features import FEATURE_COLUMNS


SEED = 42
N_SPLITS = 5
N_TRIALS = 30
TIMEOUT = 5400

RESULTS_DIR = ROOT / "results" / "bayesian"
DATABASE = RESULTS_DIR / "optuna_study.db"

STUDY_NAME = "pathsnap_lightgbm_bayesian"


def build_estimator(trial):
    """Create a LightGBM classifier using trial-suggested parameters."""

    params = {
        "num_leaves": trial.suggest_int(
            "num_leaves", 15, 255, log=True
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 600
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 20, 500, log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.5, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 10.0, log=True
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-3, 10.0, log=True
        ),
        "subsample_freq": 1,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    return LGBMClassifier(**params)


def objective(trial):
    """Minimise mean validation log loss across five trip-grouped folds."""

    train_pairs, train_obs = load_partition("train")

    groups = train_pairs["trip_id"].to_numpy()
    cv = GroupKFold(n_splits=N_SPLITS)

    fold_losses = []
    fold_f1 = []
    fold_top1 = []

    for fold, (fit_idx, val_idx) in enumerate(
        cv.split(train_pairs, groups=groups)
    ):
        fit_pairs = train_pairs.iloc[fit_idx]
        val_pairs = train_pairs.iloc[val_idx]

        val_obs = train_obs[
            train_obs.trip_id.isin(val_pairs.trip_id.unique())
        ]

        model = build_estimator(trial)

        model.fit(
            fit_pairs[FEATURE_COLUMNS],
            fit_pairs["y"],
        )

        probabilities = model.predict_proba(
            val_pairs[FEATURE_COLUMNS]
        )[:, 1]

        metrics = score_predictions(
            val_pairs,
            val_obs,
            probabilities,
            "train",
        )

        fold_losses.append(metrics["log_loss"])
        fold_f1.append(metrics["macro_f1"])
        fold_top1.append(metrics["top1_accuracy"])

        trial.report(metrics["log_loss"], step=fold)

        print(
            f"Trial {trial.number} | "
            f"Fold {fold + 1}/{N_SPLITS} | "
            f"Log loss: {metrics['log_loss']:.5f} | "
            f"Macro F1: {metrics['macro_f1']:.5f} | "
            f"Top-1: {metrics['top1_accuracy']:.5f}"
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_loss = float(np.mean(fold_losses))

    trial.set_user_attr("fold_log_losses", fold_losses)
    trial.set_user_attr("cv_macro_f1_mean", float(np.mean(fold_f1)))
    trial.set_user_attr("cv_top1_mean", float(np.mean(fold_top1)))
    trial.set_user_attr("cv_log_loss_std", float(np.std(fold_losses)))

    return mean_loss


def save_results(study, elapsed_seconds):
    """Save the complete study record and best parameters."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve every trial, including pruned and failed trials.
    trials = study.trials_dataframe()
    trials.to_csv(RESULTS_DIR / "trials.csv", index=False)

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
        print("No completed trials available yet.")
        return

    best = study.best_trial

    best_record = {
        "model": "lightgbm",
        "best_trial_number": best.number,
        "best_cv_log_loss": best.value,
        "best_params": best.params,
        "cv_macro_f1_mean": best.user_attrs.get("cv_macro_f1_mean"),
        "cv_top1_mean": best.user_attrs.get("cv_top1_mean"),
        "cv_log_loss_std": best.user_attrs.get("cv_log_loss_std"),
        "total_trials": len(study.trials),
        "completed_trials": len(completed),
        "elapsed_seconds": elapsed_seconds,
        "seed": SEED,
        "n_splits": N_SPLITS,
    }

    (RESULTS_DIR / "best_params.json").write_text(
        json.dumps(best_record, indent=2)
    )

    print("\nBest trial:", best.number)
    print("Best CV log loss:", best.value)
    print("Best parameters:")
    print(json.dumps(best.params, indent=2))


def save_figures(study):
    """Generate optimisation history and parameter importance figures."""

    import matplotlib.pyplot as plt
    from optuna.importance import get_param_importances

    figures_dir = ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
        return

    # Optimisation history.
    values = [trial.value for trial in completed]
    best_values = np.minimum.accumulate(values)

    plt.figure()
    plt.plot(range(1, len(best_values) + 1), best_values)
    plt.xlabel("Completed trial")
    plt.ylabel("Best CV log loss")
    plt.title("Bayesian Optimisation History")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "bayesian_optimisation_history.png",
        dpi=200,
    )
    plt.close()

    # Parameter importance.
    if len(completed) >= 2:
        importance = get_param_importances(study)

        names = list(importance.keys())
        scores = list(importance.values())

        plt.figure()
        plt.barh(names, scores)
        plt.xlabel("Importance")
        plt.title("Bayesian Hyperparameter Importance")
        plt.tight_layout()
        plt.savefig(
            figures_dir / "bayesian_parameter_importance.png",
            dpi=200,
        )
        plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PathSnap Bayesian Optimisation")
    print("Model: LightGBM")
    print(f"Maximum trials: {N_TRIALS}")
    print(f"Timeout: {TIMEOUT} seconds")
    print(f"CV: {N_SPLITS}-fold GroupKFold")
    print("Objective: Mean CV log loss")
    print("Student-test: NOT accessed during optimisation")
    print("=" * 60)

    print(
        "\nExpected scale: approximately 40 minutes "
        "at the specification's estimated 75 seconds per trial."
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{DATABASE}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Resume an interrupted study without exceeding 30 total trials.
    remaining = max(0, N_TRIALS - len(study.trials))

    if remaining == 0:
        print("\nThe study already contains 30 trials.")
        save_results(study, 0)
        save_figures(study)
        return

    print(f"\nStarting {remaining} remaining trials...")

    started = time.time()

    study.optimize(
        objective,
        n_trials=remaining,
        timeout=TIMEOUT,
        gc_after_trial=True,
    )

    elapsed = time.time() - started

    print(f"\nActual run time: {elapsed:.2f} seconds")

    save_results(study, elapsed)
    save_figures(study)

    print("\nBayesian optimisation finished.")


if __name__ == "__main__":
    main()

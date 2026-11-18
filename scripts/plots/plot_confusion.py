"""Plot confusion matrices from real student_test evaluation results.

Reads student_test.confusion_matrix from results/experiments/<model>.json
and saves visualization figures to results/figures/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "results" / "experiments"
OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_TO_PLOT = [
    ("logreg", "Logistic Regression (C=1.0)"),
    ("logreg_balanced", "Logistic Regression (Balanced)"),
    ("logreg_c0p1", "Logistic Regression (C=0.1)"),
    ("knn", "K-Nearest Neighbors (k=15)"),
    ("knn_k5", "K-Nearest Neighbors (k=5)"),
]


def load_confusion_matrix(model_name: str) -> tuple[np.ndarray, dict]:
    json_path = EXPERIMENTS_DIR / f"{model_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing experiment file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cm = np.array(data["student_test"]["confusion_matrix"])
    return cm, data


def plot_single_cm(cm: np.ndarray, title: str, filename: str) -> None:
    """Plot an individual confusion matrix with raw counts and row-normalised percentages."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = np.array([
        [f"{cm[0, 0]:,}\n({cm_norm[0, 0]:.2%})", f"{cm[0, 1]:,}\n({cm_norm[0, 1]:.2%})"],
        [f"{cm[1, 0]:,}\n({cm_norm[1, 0]:.2%})", f"{cm[1, 1]:,}\n({cm_norm[1, 1]:.2%})"],
    ])

    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
        annot_kws={"size": 11, "fontweight": "bold"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Reference Label", fontsize=11, labelpad=8)
    fig.tight_layout()

    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_comparison_grid() -> None:
    """Plot side-by-side comparison for LogReg, LogReg Balanced, and KNN."""
    selected = [
        ("logreg", "Logistic Regression (C=1.0)"),
        ("logreg_balanced", "LogReg (Class-Weighted Balanced)"),
        ("knn", "KNN (k=15, 200k cap)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (model_name, title) in zip(axes, selected):
        cm, data = load_confusion_matrix(model_name)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        macro_f1 = data["student_test"]["macro_f1"]
        top1 = data["student_test"]["top1_accuracy"]

        labels = np.array([
            [f"{cm[0, 0]:,}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]:,}\n({cm_norm[0, 1]:.1%})"],
            [f"{cm[1, 0]:,}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]:,}\n({cm_norm[1, 1]:.1%})"],
        ])

        sns.heatmap(
            cm_norm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Neg (0)", "Pos (1)"],
            yticklabels=["Neg (0)", "Pos (1)"],
            annot_kws={"size": 10},
        )
        ax.set_title(f"{title}\nF1: {macro_f1:.4f} | Top-1: {top1:.4f}", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    fig.suptitle("PathSnap Student-Test Confusion Matrices Comparison (366,311 candidate pairs)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "confusion_matrix_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure: {out_path}")


def main() -> None:
    for model_name, title in MODELS_TO_PLOT:
        try:
            cm, _ = load_confusion_matrix(model_name)
            plot_single_cm(cm, f"Confusion Matrix: {title}\n(student_test partition)", f"confusion_matrix_{model_name}.png")
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")

    plot_comparison_grid()


if __name__ == "__main__":
    main()

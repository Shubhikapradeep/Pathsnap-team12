"""Build the model comparison table and figures from results/experiments/*.json.

Reading the per-model JSON files (not model_comparison.csv) avoids merge conflicts when several
people add models on different branches. Runs whose name starts with "_" (development stand-ins) are skipped.

Usage: python scripts/06_compare_models.py [--experiments-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_records(experiments_dir: Path) -> list[dict]:
    records = [json.loads(p.read_text()) for p in sorted(experiments_dir.glob("*.json"))]
    return [r for r in records if not r["name"].startswith("_")]


def build_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        test = r["student_test"]
        rows.append({
            "model": r["name"], "family": r["family"],
            "cv_macro_f1": r["cv_macro_f1_mean"], "cv_macro_f1_std": r["cv_macro_f1_std"],
            "cv_top1": r["cv_top1_mean"], "cv_route_continuity": r["cv_route_continuity_mean"],
            "test_top1": test["top1_accuracy"], "test_macro_f1": test["macro_f1"],
            "test_log_loss": test["log_loss"], "test_route_continuity": test["route_continuity"],
            "train_macro_f1": r["train_pair_macro_f1"],
            "overfit_gap_train_minus_cv": r["train_pair_macro_f1"] - r["cv_macro_f1_mean"],
            "fit_seconds": test["fit_seconds"], "inference_ms_per_obs": test["inference_ms_per_obs"],
            "max_train_rows": r.get("max_train_rows"),
            "key_config": json.dumps(r["config"], sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)


def bar(ax, table: pd.DataFrame, column: str, title: str, error: str | None = None) -> None:
    """Horizontal dot plot. Dots (not bars) so the axis can zoom without exaggerating differences from a zero baseline."""
    positions = range(len(table))
    ax.errorbar(table[column], positions, xerr=table[error] if error else None, fmt="o", color="#3b6ea8",
                ecolor="#888888", capsize=3, markersize=7)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(table.model)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    margin = (table[column].max() - table[column].min()) * 0.15 or 0.01
    ax.set_xlim(table[column].min() - margin, table[column].max() + margin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, default=ROOT / "results" / "experiments")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    records = load_records(args.experiments_dir)
    if not records:
        raise SystemExit(f"no experiment JSON files in {args.experiments_dir}")
    table = build_table(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)
    table.to_csv(args.out_dir / "model_comparison_table.csv", index=False)
    view = table.drop(columns=["key_config"]).round(4)
    (args.out_dir / "model_comparison_table.md").write_text(view.to_markdown(index=False) if hasattr(view, "to_markdown") else view.to_string(index=False))
    print(view.to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    bar(axes[0, 0], table, "cv_macro_f1", "CV macro F1 (mean ± std over 5 trip folds)", "cv_macro_f1_std")
    bar(axes[0, 1], table, "test_top1", "Student-test top-1 accuracy")
    bar(axes[0, 2], table, "test_macro_f1", "Student-test macro F1")
    bar(axes[1, 0], table, "test_log_loss", "Student-test log loss (lower is better)")
    bar(axes[1, 1], table, "overfit_gap_train_minus_cv", "Train minus CV macro F1 (overfitting gap)")
    bar(axes[1, 2], table, "inference_ms_per_obs", "Inference time (ms per GPS point)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "model_comparison.png", dpi=150)
    print(f"\nwrote table + figure to {args.out_dir}")


if __name__ == "__main__":
    main()

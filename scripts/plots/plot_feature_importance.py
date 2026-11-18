"""Generate feature-importance plots for Random Forest and Extra Trees."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "results" / "experiments"
OUTPUT_DIR = ROOT / "results" / "feature_importance"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name in ("random_forest", "extra_trees"):
    model = joblib.load(MODEL_DIR / f"{name}.joblib")

    with open(REPORT_DIR / f"{name}.json") as f:
        report = json.load(f)

    features = report["features"]
    importance = model.feature_importances_

    if len(features) != len(importance):
        raise ValueError(
            f"{name}: {len(features)} feature names but "
            f"{len(importance)} importance values"
        )

    df = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values("importance", ascending=True)

    df.to_csv(OUTPUT_DIR / f"{name}_importance.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.barh(df["feature"], df["importance"])
    plt.xlabel("Feature importance")
    plt.title(f"{name.replace('_', ' ').title()} Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_importance.png", dpi=200)
    plt.close()

    print(f"Generated {name} feature-importance plot")


"""Extract and plot the actual MLP training history."""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "mlp.joblib"
OUTPUT_DIR = ROOT / "results" / "curves"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the fitted sklearn pipeline
model = joblib.load(MODEL_PATH)

# Extract the MLP from the pipeline
mlp = model.named_steps["mlpclassifier"]

loss = mlp.loss_curve_
validation = mlp.validation_scores_

# Store actual training history
history = pd.DataFrame({
    "epoch": range(1, len(loss) + 1),
    "training_loss": loss,
    "validation_score": validation,
})

csv_path = OUTPUT_DIR / "mlp_history.csv"
history.to_csv(csv_path, index=False)

# Plot training loss and validation score
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(
    history["epoch"],
    history["training_loss"],
    label="Training Loss",
)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(
    history["epoch"],
    history["validation_score"],
    label="Validation Score",
    linestyle="--",
)
ax2.set_ylabel("Validation Score")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("MLP Training Loss vs Validation Score")
fig.tight_layout()

plot_path = OUTPUT_DIR / "mlp_training_history.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)

print(f"Epochs recorded: {len(loss)}")
print(f"Final training loss: {loss[-1]:.6f}")
print(f"Final validation score: {validation[-1]:.6f}")
print(f"CSV saved: {csv_path}")
print(f"Plot saved: {plot_path}")

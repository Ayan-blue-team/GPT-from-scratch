from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

RUNS = ["S", "M", "L", "A1", "A2", "A3", "A4", "A5"]

# Figure 1: training and validation loss curves for all runs.
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

for run in RUNS:
    path = LOGS / f"{run}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing log: {path}")

    df = pd.read_csv(path)
    required = {"step", "train_loss", "val_loss"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing required columns: {required}")

    axes[0].plot(df["step"], df["train_loss"], label=run)
    axes[1].plot(df["step"], df["val_loss"], label=run)

axes[0].set_title("Training loss")
axes[1].set_title("Validation loss")

for ax in axes:
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (nats/token)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("GPT Training Experiments")
fig.tight_layout()
fig.savefig(FIGURES / "loss_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure 2: scaling comparison for S, M, and L.
summary_path = ROOT / "runs.csv"
summary = pd.read_csv(summary_path)
scaling = summary[summary["run_id"].isin(["S", "M", "L"])].copy()
scaling = scaling.sort_values("params")

if set(scaling["run_id"]) != {"S", "M", "L"}:
    raise ValueError("runs.csv must contain S, M, and L.")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    scaling["params"],
    scaling["best_val_loss"],
    marker="o",
    linewidth=2,
)

for _, row in scaling.iterrows():
    ax.annotate(
        row["run_id"],
        (row["params"], row["best_val_loss"]),
        xytext=(6, 6),
        textcoords="offset points",
    )

ax.set_xscale("log")
ax.set_xlabel("Model parameters (log scale)")
ax.set_ylabel("Best validation loss (nats/token)")
ax.set_title("Model Size vs. Best Validation Loss")
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES / "model_scaling.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved figures:")
print(FIGURES / "loss_curves.png")
print(FIGURES / "model_scaling.png")
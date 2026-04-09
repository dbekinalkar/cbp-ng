import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Config ----
CSV_FILE = "results2.csv"
OUTPUT_DIR = "local_predictor"
METRIC = "accuracy"   # Change to: extra_cycles, energy_per_inst, etc.
# ----------------

# Load data
df = pd.read_csv(CSV_FILE)

# Ensure numeric types
numeric_cols = df.columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore')
df["accuracy"] = 1 - (df["mispredictions"] / df["predictions"])

# Create output dir
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. Line plots (each parameter)
# -------------------------------
def plot_param_effect(param):
    grouped = df.groupby(param)[METRIC].mean().reset_index()

    plt.figure()
    sns.lineplot(data=grouped, x=param, y=METRIC, marker="o")
    plt.title(f"{METRIC} vs {param}")
    plt.grid(True)

    plt.savefig(f"{OUTPUT_DIR}/{METRIC}_vs_{param}.png")
    plt.close()

def plot_param_effect_split(param_x, param_split):
    grouped = (
        df.groupby([param_x, param_split])[METRIC]
        .mean()
        .reset_index()
    )

    plt.figure()
    palette = sns.color_palette("viridis", n_colors=grouped[param_split].nunique())
    sns.lineplot(
        data=grouped,
        x=param_x,
        y=METRIC,
        hue=param_split,
        style=param_split,
        markers=True,
        palette=palette
    )

    plt.title(f"{METRIC} vs {param_x} (split by {param_split})")
    plt.grid(True)

    # Move legend outside (to the right)
    plt.legend(
        title=param_split,
        bbox_to_anchor=(1.05, 1),  # push right
        loc="upper left",
        borderaxespad=0.
    )

    # Make room so it doesn't get cut off
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{METRIC}_vs_{param_x}_by_{param_split}.png", bbox_inches="tight")
    plt.close()
for p in ["BHT_SIZE_BITS","PHT_SIZE_BITS","HIST_LEN"]:
    plot_param_effect(p)

plot_param_effect_split("HIST_LEN", "BHT_SIZE_BITS")
plot_param_effect_split("HIST_LEN", "PHT_SIZE_BITS")
# -----------------------------------
# 2. Heatmaps (pairwise interactions)
# -----------------------------------
def plot_heatmap(p_x, p_y):
    pivot = df.pivot_table(
        values=METRIC,
        index=p_y,
        columns=p_x,
        aggfunc="mean"
    )

    plt.figure()
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
    plt.title(f"{METRIC}: {p_x} vs {p_y}")

    plt.savefig(f"{OUTPUT_DIR}/{METRIC}_{p_x}_vs_{p_y}.png")
    plt.close()

plot_heatmap("BHT_SIZE_BITS", "PHT_SIZE_BITS")
plot_heatmap("BHT_SIZE_BITS", "HIST_LEN")
plot_heatmap("PHT_SIZE_BITS", "HIST_LEN")

# -----------------------------------
# 3. Best configuration
# -----------------------------------
best = df.loc[df[METRIC].idxmin()]

print("\nBest configuration:")
print(best)

# -----------------------------------
# 4. Optional: 3D scatter
# -----------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df["BHT_SIZE_BITS"], df["PHT_SIZE_BITS"], df["HIST_LEN"],
                c=df[METRIC], cmap="viridis")

ax.set_xlabel("BHT_SIZE_BITS")
ax.set_ylabel("PHT_SIZE_BITS")
ax.set_zlabel("HIST_LEN")
ax.set_title(f"3D Scatter colored by {METRIC}")

fig.colorbar(sc, label=METRIC)

plt.savefig(f"{OUTPUT_DIR}/3d_scatter_{METRIC}.png")
plt.close()

print(f"\nPlots saved in '{OUTPUT_DIR}/'")
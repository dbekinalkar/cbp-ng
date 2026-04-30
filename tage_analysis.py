import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

# =========================
# CONFIG
# =========================
DB_PATH = "sqlite:///tage.db"
STUDY_NAME = "tage_optimization"

# Set this to whatever directory you want
BASE_OUTPUT_DIR = "optuna_results"

# minimize=True → lower is better
ascending = True


# =========================
# OUTPUT SETUP
# =========================
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Saving results to: {OUTPUT_DIR}")


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# =========================
# LOAD STUDY
# =========================
study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=DB_PATH
)

print("Study loaded.")
print(f"Number of trials: {len(study.trials)}")


# =========================
# BEST TRIAL
# =========================
best_trial = study.best_trial

print("\n=== BEST TRIAL ===")
print(f"Value (objective): {best_trial.value}")
print("Params:")
for k, v in best_trial.params.items():
    print(f"  {k}: {v}")

# Save best trial
with open(os.path.join(OUTPUT_DIR, "best_trial.json"), "w") as f:
    json.dump({
        "value": best_trial.value,
        "params": best_trial.params
    }, f, indent=2)


# =========================
# DATAFRAME VIEW
# =========================
df = study.trials_dataframe()

# Keep only completed trials
df = df[df["state"] == "COMPLETE"]

print("\nDataframe columns:")
print(df.columns)

# Rename value column for convenience
df = df.rename(columns={"value": "score"})


# =========================
# SORT BEST TRIALS
# =========================
df_sorted = df.sort_values("score", ascending=ascending)

print("\nTop 10 trials:")
print(df_sorted.head(10)[["score"] + [c for c in df.columns if "params_" in c]])

# Save CSV
df_sorted.to_csv(os.path.join(OUTPUT_DIR, "trials_sorted.csv"), index=False)


# =========================
# PARAM IMPORTANCE
# =========================
try:
    from optuna.importance import get_param_importances

    importances = get_param_importances(study)

    print("\n=== PARAM IMPORTANCE ===")
    for k, v in importances.items():
        print(f"{k}: {v:.4f}")

    plt.figure()
    sns.barplot(x=list(importances.values()), y=list(importances.keys()))
    plt.title("Parameter Importance")

    save_plot("param_importance.png")

except Exception as e:
    print("Param importance failed:", e)


# =========================
# PAIRPLOT
# =========================
param_cols = [c for c in df.columns if c.startswith("params_")]

if len(param_cols) > 0:
    sample_df = df_sorted.head(200)

    pairplot = sns.pairplot(sample_df[param_cols + ["score"]])
    pairplot.fig.suptitle("Pairplot", y=1.02)

    pairplot.savefig(os.path.join(OUTPUT_DIR, "pairplot.png"), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'pairplot.png')}")


# =========================
# CORRELATION HEATMAP
# =========================
if len(param_cols) > 0:
    corr_df = df[param_cols + ["score"]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")

    save_plot("correlation_matrix.png")
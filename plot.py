import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (replace with your file path if needed)
df = pd.read_csv("new_optuna_results/20260430_210223/trials_sorted.csv")

# Sort by trial number
df = df.sort_values("number")

x = df["number"]
y = df["score"]

# Compute running best (since lower score is better)
running_best = y.cummin()

plt.figure()

# Scatter points (all trials)
plt.scatter(x, y)

# Line = best score so far
plt.plot(x, running_best)

# Labels and title
plt.xlabel("Trial")
plt.ylabel("Objective 0")

# Remove surrounding box
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.grid(True)

plt.show()
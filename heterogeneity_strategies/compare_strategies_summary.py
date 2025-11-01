import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results_dir = "strategy_results"
strategies = {
    "FedAvg":   [f"server_history_{a}.json" for a in [0.1, 1, 10]],
    "FedProx":  [f"server_history_FedProx{a}.json" for a in [0.1, 1, 10]],
    "SCAFFOLD": [f"server_history_Scaffold{a}.json" for a in [0.1, 1, 10]],
}
alpha_values = [0.1, 1, 10]

summary = []

for strat, files in strategies.items():
    for alpha, fname in zip(alpha_values, files):
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        with open(path, "r") as f:
            history = json.load(f)
        acc_dict = history.get("FL_accuracy", {})
        rounds = sorted(acc_dict.keys(), key=lambda x: int(x))
        accs = [acc_dict[r] for r in rounds]
        if not accs:
            continue
        final_acc = accs[-1]
        threshold = 0.95 * final_acc
        conv_round = next((i+1 for i, a in enumerate(accs) if a >= threshold), len(accs))
        last10 = accs[-10:] if len(accs) >= 10 else accs
        stability = np.std(last10)
        summary.append({
            "Strategy": strat,
            "Alpha": alpha,
            "Final Accuracy": f"{final_acc:.6f}",
            "Convergence Round": conv_round,
            "Stability (std last 10)": f"{stability:.6f}"
        })

df = pd.DataFrame(summary)
print(df.to_markdown(index=False))

figures_dir = "strategy_figures"
os.makedirs(figures_dir, exist_ok=True)

row_colors = {
    "FedAvg": "#e6e6e6", 
    "FedProx": "#d0e0f0",    
    "SCAFFOLD": "#f0e0d0"    
}
row_color_list = [row_colors[row["Strategy"]] for _, row in df.iterrows()]

fig_height = 2 + 0.1 * len(df)
fig, ax = plt.subplots(figsize=(10, fig_height))
ax.axis('off')
tbl = ax.table(
    cellText=list(df.values),
    colLabels=list(df.columns),
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.auto_set_column_width(col=list(range(len(df.columns))))

for i in range(len(df)):
    for j in range(len(df.columns)):
        tbl[(i+1, j)].set_facecolor(row_color_list[i])  # +1 because row 0 is header

for j in range(len(df.columns)):
    tbl[(0, j)].set_facecolor("#cccccc")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "strategies_summary_table.png"), dpi=200, bbox_inches="tight")
plt.close()

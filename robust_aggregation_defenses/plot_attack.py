import json
import os
import matplotlib.pyplot as plt

# Settings
results_dir = "attack_simulation_results"  
ATTACK_TYPE = "data"  # "data" or "model"
alpha = 1
num_clients = 10
num_rounds = 30

ratios = [0, 0.25, 0.5]
labels = ["0% malicious", "25% malicious", "50% malicious"]
colors = ["#2b83ba", "#fdae61", "#d7191c"]

loss_curves = []
acc_curves = []

for ratio in ratios:
    fname = f"server_history_krum_{ATTACK_TYPE}_poisoning_{ratio}_ratio.json"
    path = os.path.join(results_dir, fname)
    if os.path.exists(path):
        path = path
    else:
        print(f"File not found: {path}")
        loss_curves.append(None)
        acc_curves.append(None)
        continue
    with open(path, "r") as f:
        history = json.load(f)
    loss = history.get("FL_loss", {})
    acc = history.get("FL_accuracy", {})
    rounds = sorted(loss.keys(), key=lambda x: int(x))
    loss_curves.append([loss[r] for r in rounds])
    acc_curves.append([acc[r] for r in rounds])

# Plot Loss
plt.figure(figsize=(10, 5))
for i, curve in enumerate(loss_curves):
    if curve is not None:
        plt.plot(curve, label=labels[i], color=colors[i])
plt.title(f"Krum - {ATTACK_TYPE.capitalize()} Poisoning: Training Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"Krum_{ATTACK_TYPE}_poisoning_loss_comparison.png"))
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 5))
for i, curve in enumerate(acc_curves):
    if curve is not None:
        plt.plot(curve, label=labels[i], color=colors[i])
plt.title(f"Krum - {ATTACK_TYPE.capitalize()} Poisoning: Test Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"Krum_{ATTACK_TYPE}_poisoning_accuracy_comparison.png"))
plt.show()  

import json
import os
import matplotlib.pyplot as plt

results_dir = "robustness_results"
ATTACK_TYPE = "data"
num_clients = 10
num_rounds = 15

alphas = [10, 1, 0.1]
alpha_labels = [r"$\alpha=10$", r"$\alpha=1$", r"$\alpha=0.1$"]
colors = ["#2b83ba", "#fdae61", "#d7191c"]

loss_curves = []
acc_curves = []

for alpha, label in zip(alphas, alpha_labels):
    fname = f"server_history_krum_{ATTACK_TYPE}_poisoning_{alpha}_alpha.json"
    path = os.path.join(results_dir, fname)
    if not os.path.exists(path):
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
        plt.plot(curve, label=alpha_labels[i], color=colors[i])
plt.title(f"Krum - {ATTACK_TYPE.capitalize()} Poisoning (25%): Training Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"Krum_{ATTACK_TYPE}_poisoning_loss_alpha_comparison.png"))
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 5))
for i, curve in enumerate(acc_curves):
    if curve is not None:
        plt.plot(curve, label=alpha_labels[i], color=colors[i])
plt.title(f"Krum - {ATTACK_TYPE.capitalize()} Poisoning (25%): Test Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"Krum_{ATTACK_TYPE}_poisoning_accuracy_alpha_comparison.png"))
plt.show()

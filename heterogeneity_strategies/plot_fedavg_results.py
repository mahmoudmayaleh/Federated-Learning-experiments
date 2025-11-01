import json
import matplotlib.pyplot as plt
import os

results_dir = "strategy_results"
figures_dir = "strategy_figures"
os.makedirs(figures_dir, exist_ok=True)

alpha_values = [0.1, 1, 10]  

loss_curves = {}
acc_curves = {}

for alpha in alpha_values:
    filename = os.path.join(results_dir, f"server_history_{alpha}.json")
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue
    with open(filename, "r") as f:
        history = json.load(f)
    loss = history.get("FL_loss", {})
    acc = history.get("FL_accuracy", {})
    rounds = sorted(loss.keys(), key=lambda x: int(x))
    loss_curves[alpha] = [loss[r] for r in rounds]
    acc_curves[alpha] = [acc[r] for r in rounds]

plt.figure(figsize=(10, 5))
for alpha, losses in loss_curves.items():
    plt.plot(losses, label=f"α={alpha}")
plt.title("FedAvg Training Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "fedavg_loss_curves.png"))
plt.show()

plt.figure(figsize=(10, 5))
for alpha, accs in acc_curves.items():
    plt.plot(accs, label=f"α={alpha}")
plt.title("FedAvg Test Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "fedavg_accuracy_curves.png"))
plt.show()

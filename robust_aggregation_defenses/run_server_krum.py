import json
from flwr.server import start_server, ServerConfig
from strategy import KrumStrategy
import os
from run_clients_auto import MALICIOUS_RATIO, ATTACK_TYPE

NUM_CLIENTS = 10
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.01
ALPHA_DIRICHLET = 0.1
DATASET_NAME = "Fashion"
SEED = 42
NUM_ROUNDS = 15

num_byzantine = int(NUM_CLIENTS * MALICIOUS_RATIO)

def main():
    strategy = KrumStrategy(
        num_clients=NUM_CLIENTS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        alpha_dirichlet=ALPHA_DIRICHLET,
        dataset_name=DATASET_NAME,
        seed=SEED,
        f=num_byzantine,
    )

    history = start_server(
        server_address="localhost:8081",
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    def list_to_dict(lst):
        if isinstance(lst, dict):
            return lst
        return {str(i + 1): v for i, v in enumerate(lst)}

    history_dict = {
        "losses_distributed": list_to_dict(history.losses_distributed),
        "metrics_distributed_fit": list_to_dict(history.metrics_distributed_fit),
        "metrics_distributed": list_to_dict(history.metrics_distributed),
    }

    def flatten_metric(metric_list):
        return {str(round_num): value for round_num, value in metric_list}

    if "metrics_distributed_fit" in history_dict:
        fit_metrics = history_dict["metrics_distributed_fit"]
        if "FL_loss" in fit_metrics:
            history_dict["FL_loss"] = flatten_metric(fit_metrics["FL_loss"])
        if "FL_accuracy" in fit_metrics:
            history_dict["FL_accuracy"] = flatten_metric(fit_metrics["FL_accuracy"])

    results_dir = "robustness_results"
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f"server_history_krum_{ATTACK_TYPE}_poisoning_{ALPHA_DIRICHLET}_alpha.json"), "w") as f:
        json.dump(history_dict, f, indent=4)

if __name__ == "__main__":
    main()

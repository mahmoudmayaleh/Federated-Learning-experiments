import json
from flwr.server import start_server, ServerConfig
from strategy import FedAvgStrategy, FedProxStrategy, ScaffoldStrategy
from client_manager import CustomClientManager
import time
import random
import numpy as np
import torch
import os

SEED = 42
NUM_ROUNDS = 50
NUM_CLIENTS = 10
EPOCHS = 3
DATASET_NAME = "Fashion"
ALPHA_DIRICHLET = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    server_address = "localhost:8081"
    num_rounds = NUM_ROUNDS

    client_manager = CustomClientManager()
    strategy = FedAvgStrategy(
        num_clients=NUM_CLIENTS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        alpha_dirichlet=ALPHA_DIRICHLET,
        dataset_name=DATASET_NAME,
        seed=SEED,
    )

    history = start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds),
        client_manager=client_manager,
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

    # Ensure the results directory exists
    results_dir = "baseline_results"
    os.makedirs(results_dir, exist_ok=True)

    # Save results in the baseline_results folder
    with open(os.path.join(results_dir, f"server_history_{ALPHA_DIRICHLET}.json"), "w") as f:
        json.dump(history_dict, f, indent=4)

if __name__ == "__main__":
    main()

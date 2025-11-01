# Baseline Federated Training

This module contains the reference implementation used to benchmark standard federated learning on Fashion-MNIST. It provides utilities to generate client datasets, run the server/client loop, and compare the impact of core hyperparameters such as learning rate, batch size, number of clients, epochs, and Dirichlet alpha.

## Contents

- `client.py`, `client_manager.py`, `strategy.py` – core Flower client and strategy implementations.
- `run_server.py`, `run_client.py`, `run_clients_auto.py` – scripts to start the coordinator and simulated clients.
- `generate_data.py`, `distributed_data/` – helpers for creating and storing non-IID client splits.
- `compare_*.py`, `analyze_results.py`, `results_visualizer.py` – analysis scripts for benchmarking different hyperparameters.
- `baseline_results/`, `baseline_figures/` – storage for metrics and visualisations produced by the analysis scripts.

## Usage

1. Install the shared dependencies listed at the repository root (`pip install -r requirements/base.txt`).
2. Generate the client datasets if you have not already done so:
   ```bash
   python generate_data.py --num-clients 10 --alpha 1.0
   ```
3. Start the server and clients:
   ```bash
   python run_server.py
   python run_client.py --cid 0
   ```
4. After training completes, run any of the comparison scripts to produce figures in `baseline_figures/`.

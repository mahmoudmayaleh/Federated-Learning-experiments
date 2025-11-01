# Federated Learning Experiment Suite

A consolidated collection of federated learning experiments covering baseline training, heterogeneity analysis, adversarial robustness, and an industrial-grade data pipeline.
## Repository Layout

- `baseline_federated_training/` – reference Flower implementation plus tooling to benchmark core hyperparameters.
- `heterogeneity_strategies/` – comparison of FedAvg, FedProx, and SCAFFOLD under non-IID data splits.
- `robust_aggregation_defenses/` – simulations of data/model poisoning attacks and robust aggregation rules.
- `industrial_pipeline/` – containerized data ingestion and federated orchestration pipeline used for large-scale experiments.
- `docs/` – PDFs summarising findings for each module.
- `requirements/` – Python dependencies for the experimentation modules.

## Getting Started

1. Create a Python environment (3.9–3.11 recommended).
2. Install the shared experimentation dependencies:
   ```bash
   pip install -r requirements/base.txt
   ```
3. Pick a module and follow the usage instructions in its local `README.md`.

The industrial pipeline relies on Docker and Compose; refer to the scripts under `industrial_pipeline/` for service definitions and helper commands.

## Documentation

Supplementary PDFs in `docs/` capture the main observations, experiment settings, and architectural notes for quick reference.

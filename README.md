# Federated Learning Experiment Suite

A consolidated collection of federated learning experiments covering baseline training, heterogeneity analysis, adversarial robustness, and an industrial-grade data pipeline. The code has been refactored into self-contained modules with descriptive names so the suite can be shared as a polished portfolio project.

## Repository Layout

- `baseline_federated_training/` – reference Flower implementation plus tooling to benchmark core hyperparameters.
- `heterogeneity_strategies/` – comparison of FedAvg, FedProx, and SCAFFOLD under non-IID data splits.
- `robust_aggregation_defenses/` – simulations of data/model poisoning attacks and robust aggregation rules.
- `industrial_pipeline/` – containerised data ingestion and federated orchestration pipeline used for large-scale experiments.
- `docs/` – curated PDFs summarising findings for each module.
- `requirements/` – Python dependencies for the experimentation modules.

## Getting Started

1. Create a Python environment (3.9–3.11 recommended).
2. Install the shared experimentation dependencies:
   ```bash
   pip install -r requirements/base.txt
   ```
3. Pick a module and follow the usage instructions in its local `README.md`.

The industrial pipeline relies on Docker and Compose; refer to the scripts under `industrial_pipeline/` for service definitions and helper commands.

## Naming Convention

All folders, scripts, and artefacts have been renamed to highlight what each experiment demonstrates rather than the original course iteration. References to academic task numbers (TP1–TP4) were removed in favour of descriptive project titles.

## Documentation

Supplementary PDFs in `docs/` capture the main observations, experiment settings, and architectural notes for quick reference. Feel free to replace them with updated reports as you extend the suite.

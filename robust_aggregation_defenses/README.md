# Robust Aggregation Experiments

This module explores secure aggregation techniques when a fraction of clients behave maliciously. It benchmarks **FedAvg**, **FedMedian**, and **Krum** under both data and model poisoning scenarios.

## Contents

- `run_server_attack.py`, `run_server_fedmedian.py`, `run_server_krum.py` – entry points for each aggregation strategy.
- `run_client_attack.py`, `run_clients_auto.py` – utilities for spawning honest and adversarial clients.
- `client_attack.py`, `strategy.py` – attack-aware client and server implementations.
- `robustness_results/`, `attack_simulation_results/` – JSON logs and generated artefacts for each scenario.
- `plot_attack.py`, `plot_attack_alpha.py` – visualisations contrasting defence performance across a and malicious ratios.

## Usage

```bash
python run_server_attack.py --malicious-ratio 0.3 --alpha 0.5
python run_client_attack.py data 0
```

Once all clients finish, generate figures:

```bash
python plot_attack.py
python plot_attack_alpha.py
```

## Notes

- Configure the malicious ratio, attack type, and a directly in the helper scripts before execution.
- Results are stored in `robustness_results/` while derived plots and summaries are written to `attack_simulation_results/`.

# Federated Learning Demo (Flower + PyTorch)

This repo provides a short, runnable demo focused on **Federated Learning** under **non-IID data**.

It includes:
- Non-IID data visualization (Dirichlet partition)
- Local-only training baselines (one model per client)
- Federated training (FedAvg with Flower) producing one global model
- Comparison plots: local-only vs federated vs centralized reference

## Repository layout

- `commands.txt`  
  Copy-paste friendly command list to set up and run the demo.

- `run_demo.sh`  
  Runs the demo in the right order: non-IID plot → local-only → federated run → centralized and comparison plot.

- `prepare_data.py`  
  Downloads MNIST into `./data/`.

- `common.py`  
  Shared utilities:
  - model definition (`SmallCNN`)
  - Dirichlet non-IID partitioning
  - dataloaders builder (`make_loaders`)
  - train/eval utilities
  - parameter conversion (PyTorch ↔ NumPy)
  - optional warning configuration (if enabled)

- `client.py`  
  Flower client implementation (PyTorch local training and evaluation).

- `server.py`  
  Flower server using FedAvg plus centralized evaluation on the global test set.

- `local_baselines.py`  
  Trains one local-only model per client and prints:
  - one line per client with per-class counts (0–9)
  - local-only accuracy on the global test set  
  Writes results to `logs/local_baselines.json`.

- `plot_non_iid.py`  
  Plots the non-IID partition (stacked bar chart: class distribution per client).

- `plot_compare.py`  
  Plots:
  - federated accuracy vs rounds (if available from logs)
  - final comparison bar chart (local models, local mean, FL final, centralized)

- `plot_logs.py`  
  Quick plot utility that reads `logs/server_metrics.jsonl` (if present) to display global accuracy.

- `logs/`  
  Generated outputs (server/client logs, JSON logs, history dumps).

## Quick start (recommended)

All commands are provided in `commands.txt`. The intended flow is:

1. Create and activate a virtual environment
2. Install dependencies
3. Download MNIST (recommended)
4. Run the demo
5. Plot global accuracy

To avoid duplicating command blocks here, open `commands.txt` and follow it top-to-bottom.

## `commands.txt` (included in this repo)

The file contains:

- 1) Create environment  
  `python3 -m venv .venv` and activate it `source .venv/bin/activate`

- 2) Install dependencies  
  `pip install -r requirements.txt`

- 3) Download MNIST (recommended)  
  `python prepare_data.py`

- 4) Run the demo  
  `chmod +x run_demo.sh` then `bash run_demo.sh`

- 5) Quick global accuracy plot  
  `python plot_logs.py logs/server_metrics.jsonl`

## Configuration (environment variables)

Defaults are set in `run_demo.sh`. You can override them before running the script.

### Non-IID partition parameters (shared across plots, local-only, and FL)
- `NUM_CLIENTS` (default: 3)
- `DIRICHLET_ALPHA` (default: 0.1)  
  Note: **alpha low means stronger non-IID** (more skewed distributions).  
  Suggested values:
  - 0.05 very strong heterogeneity
  - 0.1 strong heterogeneity (recommended for a short demo)
  - 0.5 moderate heterogeneity
  - 1.0+ near-IID
- `DIRICHLET_SEED` (default: 42)
- `MAX_SAMPLES_PER_CLIENT` (default: 12000)

### Federated training parameters
- `NUM_ROUNDS` (default: 15)
- `LOCAL_EPOCHS` (default: 1)
- `LR` (default: 0.05)
- `BATCH_SIZE` (default: 64)

### Local-only and centralized reference parameters
- `LOCAL_BASELINE_EPOCHS` (default: 2)
- `CENTRALIZED_EPOCHS` (default: 2)

## Recommended fair comparison (local-only vs federated)

To compare local-only vs federated under a similar **per-client compute budget**, match the number of local epochs per client:

- Federated per-client epochs ≈ `NUM_ROUNDS × LOCAL_EPOCHS`
- Local-only per-client epochs = `LOCAL_BASELINE_EPOCHS`

Recommended setting:
- `LOCAL_BASELINE_EPOCHS = NUM_ROUNDS × LOCAL_EPOCHS`

## Expected outputs

- A stacked bar chart showing class distribution per client (non-IID partition)
- Console output per client for local-only:
  - per-class counts (0–9) in a single line
  - local-only accuracy on the global test set
- Federated training logs (server and clients) written under `logs/`
- Comparison charts:
  - federated accuracy vs rounds (if the series can be extracted from logs)
  - final bar chart: local client accuracies, local mean, FL final, centralized reference

## Troubleshooting

- If MNIST is not found, run `prepare_data.py` first (step 3 in `commands.txt`).
- If `plot_logs.py` or `plot_compare.py` cannot find the federated accuracy series:
  - check `logs/server_metrics.jsonl`
  - check `logs/history_dump.json`
  - check `logs/server_stdout.log`  
  Log structures can differ slightly depending on the installed Flower version.

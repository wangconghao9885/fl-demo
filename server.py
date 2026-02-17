from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import flwr as fl
import torch

from common import DataConfig, SmallCNN, evaluate, make_loaders, read_partition_env, set_parameters

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # metrics: List[(num_examples, metrics_dict)]
    totals: Dict[str, float] = {}
    count = 0
    for n, m in metrics:
        count += n
        for k, v in m.items():
            if k in ("client_id",):
                continue
            totals[k] = totals.get(k, 0.0) + float(v) * n
    if count == 0:
        return {}
    return {k: v / count for k, v in totals.items()}


def on_fit_config(server_round: int) -> Dict[str, str]:
    # Esto llega a los clients en fit(config)
    return {"server_round": str(server_round)}


def get_evaluate_fn():
    num_clients, alpha, seed, max_samples = read_partition_env()
    _, testloader = make_loaders(
        client_id=1,
        data_cfg=DataConfig(batch_size=64),
        num_clients=num_clients,
        dirichlet_alpha=alpha,
        seed=seed,
        max_samples_per_client=max_samples,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SmallCNN().to(device)

    def evaluate_fn(server_round: int, parameters, config):
        set_parameters(net, parameters)
        loss, acc = evaluate(net, testloader, device)
        return float(loss), {"accuracy": float(acc)}

    return evaluate_fn



def main() -> None:
    os.makedirs("logs", exist_ok=True)
    stdout_log = os.path.join("logs", "server_stdout.log")
    metrics_log = os.path.join("logs", "server_metrics.jsonl")
    history_dump = os.path.join("logs", "history_dump.json")

    num_rounds = int(os.getenv("NUM_ROUNDS", "15"))

    # Importante para tolerar dropout:
    # - min_* = 2 para que siga con 2/3 clientes
    # - accept_failures=True para que una excepción en un cliente no tumbe la ronda
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        accept_failures=True,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Starting Flower server at 127.0.0.1:8080 for {num_rounds} rounds")
    print(f"Logs: {stdout_log}, {metrics_log}, {history_dump}")

    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Persistencia: dump “raw” del history para plotting offline
    try:
        with open(history_dump, "w", encoding="utf-8") as f:
            json.dump(history.__dict__, f, indent=2, default=str)
    except Exception as e:
        print(f"Could not dump history: {e}")

    # Log mínimo en JSONL: pérdidas centralizadas por ronda (si están)
    now = datetime.utcnow().isoformat()
    try:
        with open(metrics_log, "a", encoding="utf-8") as f:
            for rnd, loss in getattr(history, "losses_centralized", []):
                f.write(json.dumps({"ts": now, "round": int(rnd), "centralized_loss": float(loss)}) + "\n")
            # Si tu versión expone accuracy centralizada en metrics_centralized['accuracy']
            metrics_c = getattr(history, "metrics_centralized", {})
            acc_series = metrics_c.get("accuracy") if isinstance(metrics_c, dict) else None
            if acc_series:
                for rnd, acc in acc_series:
                    f.write(json.dumps({"ts": now, "round": int(rnd), "centralized_accuracy": float(acc)}) + "\n")
    except Exception as e:
        print(f"Could not write metrics log: {e}")

    print("Server finished.")


if __name__ == "__main__":
    main()

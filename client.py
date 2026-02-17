from __future__ import annotations

import os
from typing import Dict, Tuple

import flwr as fl
import torch

from common import (
    DataConfig,
    SmallCNN,
    evaluate,
    get_parameters,
    make_loaders,
    read_partition_env,
    set_parameters,
    train_one_epoch,
)


def parse_dropout_rounds(s: str) -> Tuple[int, int] | None:
    # Formato: "4-7"
    if not s or "-" not in s:
        return None
    a, b = s.split("-", 1)
    return int(a.strip()), int(b.strip())


class MnistClient(fl.client.NumPyClient):
    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SmallCNN().to(self.device)

        data_cfg = DataConfig(batch_size=int(os.getenv("BATCH_SIZE", "64")))

        num_clients, alpha, seed, max_samples = read_partition_env()
        self.trainloader, self.testloader = make_loaders(
	    client_id=client_id,
	    data_cfg=data_cfg,
	    num_clients=num_clients,
	    dirichlet_alpha=alpha,
	    seed=seed,
	    max_samples_per_client=max_samples,
        )

        self.dropout = parse_dropout_rounds(os.getenv("DROPOUT_ROUNDS", ""))
        self.lr = float(os.getenv("LR", "0.05"))
        self.local_epochs = int(os.getenv("LOCAL_EPOCHS", "1"))

    def get_parameters(self, config: Dict[str, str]):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        server_round = int(config.get("server_round", 0))

        # Dropout "soft": no mata el proceso
        if self.dropout is not None:
            lo, hi = self.dropout
            if lo <= server_round <= hi:
                metrics = {
                    "train_loss": float("nan"),
                    "train_acc": float("nan"),
                    "client_id": int(self.client_id),
                    "dropped": 1.0,
                }
                # Devuelve parámetros sin cambios y peso 0 (num_examples=0)
                return get_parameters(self.net), 0, metrics

        train_loss = 0.0
        train_acc = 0.0
        for _ in range(self.local_epochs):
            train_loss, train_acc = train_one_epoch(
                self.net, self.trainloader, self.device, lr=self.lr
            )

        metrics = {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "client_id": int(self.client_id),
            "dropped": 0.0,
        }
        return get_parameters(self.net), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, acc = evaluate(self.net, self.testloader, self.device)
        metrics = {"accuracy": float(acc), "client_id": int(self.client_id)}
        return float(loss), len(self.testloader.dataset), metrics


def main() -> None:
    client_id = int(os.environ["CLIENT_ID"])
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=MnistClient(client_id),
    )


if __name__ == "__main__":
    main()

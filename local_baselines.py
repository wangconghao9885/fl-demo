import os
import json
from datetime import datetime

import numpy as np
import torch

from common import DataConfig, SmallCNN, evaluate, make_loaders, train_one_epoch, read_partition_env


def class_counts_from_loader(trainloader) -> np.ndarray:
    # Cuenta etiquetas 0-9 recorriendo el dataset del DataLoader
    counts = np.zeros(10, dtype=int)
    ds = trainloader.dataset
    # Subset(MNIST) típico: ds.dataset es MNIST y ds.indices son los índices del subset
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        idx = np.array(ds.indices)
        targets = np.array(base.targets)[idx]  # type: ignore[attr-defined]
        for c in range(10):
            counts[c] = int(np.sum(targets == c))
        return counts

    # Fallback genérico (más lento): iterar batches
    for _, y in trainloader:
        y_np = y.numpy()
        for c in range(10):
            counts[c] += int(np.sum(y_np == c))
    return counts


def train_local(client_id: int, epochs: int = 10, lr: float = 0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SmallCNN().to(device)

    num_clients, alpha, seed, max_samples = read_partition_env()
    trainloader, testloader = make_loaders(
	    client_id=client_id,
	    data_cfg=DataConfig(batch_size=64),
	    num_clients=num_clients,
	    dirichlet_alpha=alpha,
	    seed=seed,
	    max_samples_per_client=max_samples,
    )

    trainloader, testloader = make_loaders(client_id=client_id, data_cfg=DataConfig(batch_size=64))

    # Print 1 línea: conteos por clase
    counts = class_counts_from_loader(trainloader)
    counts_str = " ".join([f"{i}:{counts[i]}" for i in range(10)])
    print(f"Client {client_id} class counts -> {counts_str}")

    for _ in range(epochs):
        train_one_epoch(net, trainloader, device, lr=lr)

    loss, acc = evaluate(net, testloader, device)
    return loss, acc


def main():
    os.makedirs("logs", exist_ok=True)
    out = {"ts": datetime.utcnow().isoformat(), "locals": []}

    for cid in [1, 2, 3]:
        loss, acc = train_local(cid)
        out["locals"].append({"client_id": cid, "test_loss": float(loss), "test_acc": float(acc)})
        print(f"Client {cid} local-only: acc={acc:.4f}, loss={loss:.4f}")

    with open("logs/local_baselines.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

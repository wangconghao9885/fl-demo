from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import SmallCNN, evaluate, train_one_epoch


def make_dirichlet_partitions(
    trainset: datasets.MNIST,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    max_samples_per_client: int | None = 12000,
) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    targets = trainset.targets.numpy()  # type: ignore[attr-defined]
    idx_by_class = {c: np.where(targets == c)[0] for c in range(10)}
    for c in range(10):
        rng.shuffle(idx_by_class[c])

    client_indices: Dict[int, List[int]] = {i + 1: [] for i in range(num_clients)}

    for c in range(10):
        idx_c = idx_by_class[c]
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        shards = np.split(idx_c, splits)
        for i, shard in enumerate(shards):
            client_indices[i + 1].extend(shard.tolist())

    if max_samples_per_client is not None:
        for cid in client_indices:
            if len(client_indices[cid]) > max_samples_per_client:
                rng.shuffle(client_indices[cid])
                client_indices[cid] = client_indices[cid][:max_samples_per_client]

    return client_indices


def load_fl_accuracy_series() -> List[Tuple[int, float]]:
    # Preferimos server_metrics.jsonl (si está), si no, history_dump.json.
    metrics_path = Path("logs/server_metrics.jsonl")
    if metrics_path.exists():
        series = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "centralized_accuracy" in obj and "round" in obj:
                series.append((int(obj["round"]), float(obj["centralized_accuracy"])))
        series = sorted(set(series), key=lambda x: x[0])
        if series:
            return series

    hist_path = Path("logs/history_dump.json")
    if hist_path.exists():
        data = json.loads(hist_path.read_text(encoding="utf-8"))
        mc = data.get("metrics_centralized", {})
        acc_series = mc.get("accuracy") if isinstance(mc, dict) else None
        if acc_series:
            return [(int(r), float(v)) for r, v in acc_series]

    return []


def train_centralized(
    trainset: datasets.MNIST,
    testset: datasets.MNIST,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    net = SmallCNN().to(device)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    for _ in range(epochs):
        train_one_epoch(net, trainloader, device, lr=lr)

    loss, acc = evaluate(net, testloader, device)
    return loss, acc


def train_local_only(
    trainset: datasets.MNIST,
    testset: datasets.MNIST,
    client_indices: Dict[int, List[int]],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> Dict[int, float]:
    local_acc: Dict[int, float] = {}
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    for cid, idx in client_indices.items():
        net = SmallCNN().to(device)
        subset = Subset(trainset, idx)
        trainloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)

        for _ in range(epochs):
            train_one_epoch(net, trainloader, device, lr=lr)

        _, acc = evaluate(net, testloader, device)
        local_acc[cid] = float(acc)

    return local_acc


def try_load_local_baselines() -> Dict[int, float]:
    p = Path("logs/local_baselines.json")
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[int, float] = {}
    for row in data.get("locals", []):
        out[int(row["client_id"])] = float(row["test_acc"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1, help="alpha bajo => no-IID fuerte")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=12000)

    ap.add_argument("--centralized-epochs", type=int, default=2)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--retrain-locals", action="store_true", help="Ignora logs/local_baselines.json y reentrena local-only")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

    parts = make_dirichlet_partitions(
        trainset=trainset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        max_samples_per_client=args.max_samples if args.max_samples > 0 else None,
    )

    # Local-only: de logs o reentrenar
    local_acc = {}
    if not args.retrain_locals:
        local_acc = try_load_local_baselines()
    if not local_acc or args.retrain_locals:
        local_acc = train_local_only(
            trainset=trainset,
            testset=testset,
            client_indices=parts,
            epochs=args.local_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
        )

    # Centralized
    _, acc_central = train_centralized(
        trainset=trainset,
        testset=testset,
        epochs=args.centralized_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )

    # FL series
    fl_series = load_fl_accuracy_series()
    fl_final = fl_series[-1][1] if fl_series else float("nan")

    # Prepare values
    local_vals = [local_acc.get(i + 1, float("nan")) for i in range(args.num_clients)]
    local_mean = float(np.nanmean(local_vals))

    # Plot 1: FL accuracy vs rounds (si hay logs)
    if fl_series:
        rounds = [r for r, _ in fl_series]
        accs = [a for _, a in fl_series]
        plt.figure()
        plt.plot(rounds, accs)
        plt.xlabel("Round")
        plt.ylabel("Centralized accuracy (test)")
        plt.title("Federated Learning global accuracy vs rounds")
        plt.tight_layout()
        plt.show()
    else:
        print("No encuentro serie de accuracy FL en logs/server_metrics.jsonl ni logs/history_dump.json")

    # Plot 2: Comparativa final (bars)
    labels = [f"local c{i+1}" for i in range(args.num_clients)] + ["local mean", "FL final", "centralized"]
    values = local_vals + [local_mean, fl_final, float(acc_central)]

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Test accuracy")
    plt.title(f"Compare: local-only vs FL vs centralized (Dirichlet alpha={args.alpha})")
    plt.tight_layout()
    plt.show()

    # Print resumen numérico para tu slide
    for i in range(args.num_clients):
        print(f"local client {i+1}: {local_vals[i]:.4f}")
    print(f"local mean: {local_mean:.4f}")
    print(f"FL final: {fl_final:.4f}")
    print(f"centralized: {acc_central:.4f}")


if __name__ == "__main__":
    main()

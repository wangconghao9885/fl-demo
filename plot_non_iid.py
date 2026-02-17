from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def make_dirichlet_partitions(
    trainset: datasets.MNIST,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    max_samples_per_client: int | None = None,
) -> Dict[int, List[int]]:
    """
    client_id (1..num_clients) -> indices del trainset
    Partición no-IID por clase usando Dirichlet(alpha).
    alpha bajo => más no-IID; alpha alto => más IID.
    """
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


def class_counts(trainset: datasets.MNIST, indices: List[int]) -> np.ndarray:
    targets = trainset.targets.numpy()  # type: ignore[attr-defined]
    y = targets[np.array(indices)]
    counts = np.zeros(10, dtype=int)
    for c in range(10):
        counts[c] = int(np.sum(y == c))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.1, help="alpha bajo => no-IID fuerte")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=12000)
    args = ap.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)

    parts = make_dirichlet_partitions(
        trainset=trainset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        max_samples_per_client=args.max_samples if args.max_samples > 0 else None,
    )

    counts_matrix = np.vstack([class_counts(trainset, parts[cid]) for cid in range(1, args.num_clients + 1)])

    # Print 1 línea por cliente (útil para tu charla)
    for i in range(args.num_clients):
        cid = i + 1
        counts_str = " ".join([f"{k}:{counts_matrix[i, k]}" for k in range(10)])
        print(f"Client {cid} class counts -> {counts_str}")

    # Plot: barras apiladas por cliente
    x = np.arange(args.num_clients)
    bottom = np.zeros(args.num_clients, dtype=int)

    plt.figure()
    for c in range(10):
        plt.bar(x, counts_matrix[:, c], bottom=bottom, label=str(c))
        bottom += counts_matrix[:, c]

    plt.xticks(x, [f"client {i+1}" for i in range(args.num_clients)])
    plt.xlabel("Client")
    plt.ylabel("Train samples")
    plt.title(f"Non-IID MNIST partition (Dirichlet alpha={args.alpha}, seed={args.seed})")
    plt.legend(title="Class", ncol=5, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

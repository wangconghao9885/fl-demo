from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ----------------------------
# Modelo pequeño (MNIST)
# ----------------------------
class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Datos
# ----------------------------
@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 64
    num_workers: int = 0  # estable en portátil


def get_datasets() -> Tuple[datasets.MNIST, datasets.MNIST]:
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)
    return trainset, testset


def make_dirichlet_partitions(
    trainset: datasets.MNIST,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    max_samples_per_client: Optional[int] = 12000,
) -> Dict[int, List[int]]:
    """
    client_id (1..num_clients) -> indices del trainset
    Dirichlet por clase. alpha bajo => no-IID fuerte.
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


def make_loaders(
    client_id: int,
    data_cfg: Optional[DataConfig] = None,
    num_clients: int = 3,
    dirichlet_alpha: float = 0.1,
    seed: int = 42,
    max_samples_per_client: Optional[int] = 12000,
) -> Tuple[DataLoader, DataLoader]:
    if data_cfg is None:
        data_cfg = DataConfig()

    trainset, testset = get_datasets()

    parts = make_dirichlet_partitions(
        trainset=trainset,
        num_clients=num_clients,
        alpha=dirichlet_alpha,
        seed=seed,
        max_samples_per_client=max_samples_per_client,
    )
    if client_id not in parts:
        raise ValueError(f"client_id={client_id} fuera de rango 1..{num_clients}")

    train_subset = Subset(trainset, parts[client_id])

    trainloader = DataLoader(
        train_subset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=False,
    )
    testloader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=False,
    )
    return trainloader, testloader


# ----------------------------
# Parámetros (Flower <-> PyTorch)
# ----------------------------
def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    keys = list(net.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    net.load_state_dict(state_dict, strict=True)


# ----------------------------
# Train/Eval
# ----------------------------
@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    net.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss = criterion(logits, y).item()
        loss_sum += loss * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
) -> Tuple[float, float]:
    net.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = net(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        loss_sum += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


# ----------------------------
# Helpers de configuración (env)
# ----------------------------
def read_partition_env() -> Tuple[int, float, int, Optional[int]]:
    num_clients = int(os.getenv("NUM_CLIENTS", "3"))
    alpha = float(os.getenv("DIRICHLET_ALPHA", "0.1"))
    seed = int(os.getenv("DIRICHLET_SEED", "42"))
    mspc_raw = os.getenv("MAX_SAMPLES_PER_CLIENT", "12000").strip()
    max_samples_per_client = int(mspc_raw) if mspc_raw else None
    return num_clients, alpha, seed, max_samples_per_client

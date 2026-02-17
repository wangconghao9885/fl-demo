from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    # Preferimos history_dump.json porque Flower cambia la estructura del history según versión.
    history_path = Path("logs/history_dump.json")
    if not history_path.exists():
        print("No encuentro logs/history_dump.json. Ejecuta primero la demo.")
        sys.exit(1)

    data = json.loads(history_path.read_text(encoding="utf-8"))

    # Intento extraer accuracy centralizada por ronda, si existe.
    # En muchas versiones: metrics_centralized = {'accuracy': [(1, 0.12), (2, 0.25), ...]}
    metrics = data.get("metrics_centralized", {})
    acc_series = metrics.get("accuracy")

    if not acc_series:
        print("No encuentro serie de accuracy en metrics_centralized. Mira logs/server_stdout.log.")
        sys.exit(1)

    rounds = [int(r) for r, _ in acc_series]
    acc = [float(v) for _, v in acc_series]

    plt.figure()
    plt.plot(rounds, acc)
    plt.xlabel("Round")
    plt.ylabel("Centralized accuracy (test)")
    plt.title("Flower FL demo")
    plt.show()


if __name__ == "__main__":
    main()

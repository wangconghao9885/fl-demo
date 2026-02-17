#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config demo (ajusta si quieres)
# -----------------------------
export NUM_CLIENTS="${NUM_CLIENTS:-3}"
export DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.2}"      # alpha bajo => non-IID fuerte
export DIRICHLET_SEED="${DIRICHLET_SEED:-42}"
export MAX_SAMPLES_PER_CLIENT="${MAX_SAMPLES_PER_CLIENT:-12000}"

export NUM_ROUNDS="${NUM_ROUNDS:-5}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
export LR="${LR:-0.05}"
export BATCH_SIZE="${BATCH_SIZE:-64}"

# para baseline y comparativa
export LOCAL_BASELINE_EPOCHS="${LOCAL_BASELINE_EPOCHS:-5}"
export CENTRALIZED_EPOCHS="${CENTRALIZED_EPOCHS:-5}"

mkdir -p logs

# -----------------------------
# 0) Datos (descarga si falta)
# -----------------------------
if [ ! -d "data/MNIST" ]; then
  python prepare_data.py
fi

# -----------------------------
# 1) Gráfica non-IID (partición)
# -----------------------------
python plot_non_iid.py \
  --num-clients "$NUM_CLIENTS" \
  --alpha "$DIRICHLET_ALPHA" \
  --seed "$DIRICHLET_SEED" \
  --max-samples "$MAX_SAMPLES_PER_CLIENT"

# -----------------------------
# 2) Entrenamiento local-only (3 modelos locales)
# -----------------------------
echo "=============================="
echo "Local isolated model training for 5 rounds"

python local_baselines.py

# -----------------------------
# 3) Entrenamiento federado (Flower)
# -----------------------------
cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT
echo "=============================="
echo "Federated Learning for 5 rounds"
echo "Starting server..."
python server.py 2>&1 | tee logs/server_stdout.log &
SERVER_PID=$!

sleep 2

echo "Starting clients..."
CLIENT_ID=1 python client.py 2>&1 | tee logs/client1.log &
CLIENT_ID=2 python client.py 2>&1 | tee logs/client2.log &
CLIENT_ID=3 python client.py 2>&1 | tee logs/client3.log &

# Espera a que termine el server (cuando complete NUM_ROUNDS)
wait "$SERVER_PID"

# -----------------------------
# 4) Comparativa final (local-only vs FL vs centralizado)
# -----------------------------
echo "=============================="
echo "Final centralized training for comparison"

python plot_compare.py \
  --num-clients "$NUM_CLIENTS" \
  --alpha "$DIRICHLET_ALPHA" \
  --seed "$DIRICHLET_SEED" \
  --max-samples "$MAX_SAMPLES_PER_CLIENT" \
  --centralized-epochs "$CENTRALIZED_EPOCHS" \
  --local-epochs "$LOCAL_BASELINE_EPOCHS" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE"

echo "Done. See logs/ and the plots."

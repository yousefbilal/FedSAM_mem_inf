#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG YOU CAN EDIT ----------
DATASET="cifar10"
N_RUNS="${1:-3}"        # how many runs to launch (default 3, override via arg)
ALPHA="${2:-0.5}"       # alpha value (default 0.5, override via arg)
LOGDIR="logs_fedopt_resnet18"
DEVICE="cuda:0"
# ---------------------------------------

mkdir -p "$LOGDIR"

echo "Launching ${N_RUNS} runs with random seeds (alpha=${ALPHA})"
echo "Logs -> ${LOGDIR}/"

for ((i=0; i<N_RUNS; i++)); do
    # Generate a random 32-bit unsigned integer as seed
    SEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')
    LOGFILE="${LOGDIR}/alpha_${ALPHA}_run_${i}_seed_${SEED}.log"

    echo "Run $i: seed=${SEED}, log=${LOGFILE}"

    nohup python main.py \
        -dataset "$DATASET" \
        --num-rounds 10000 \
        --eval-every 100 \
        --batch-size 64 \
        --num-epochs 1 \
        --clients-per-round 5 \
        -model resnet18 \
        -lr 0.01 \
        --weight-decay 0.0004 \
        -device "$DEVICE" \
        -algorithm fedopt \
        --server-opt sgd \
        --server-lr 1 \
        --num-workers 0 \
        --where-loading init \
        -alpha "$ALPHA" \
        --client-algorithm sam \
        -rho 0.1 \
        -eta 0 \
        --n-clients 100 \
        --influence_runs 20 \
        --seed "$SEED" \
        --imagenet-pretrained \
        > "$LOGFILE" 2>&1 &
done

echo "All ${N_RUNS} runs launched in background. You can safely exit the SSH session."
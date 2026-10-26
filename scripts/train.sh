#!/usr/bin/env bash
# Train one configuration with seeds 1111, 1112 and 1113.
# usage: bash scripts/train.sh configs/sims/main.yaml [gpu_id]
set -e
cd "$(dirname "$0")/.."
CONFIG=$1; GPU=${2:-0}
for SEED in 1111 1112 1113; do
  CUDA_VISIBLE_DEVICES=$GPU python -m semmsa.train --config "$CONFIG" --seed $SEED
done

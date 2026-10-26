#!/usr/bin/env bash
# Train every configuration of a dataset (3 seeds each) and run the robust evaluation.
# usage: bash scripts/reproduce.sh sims [gpu_id]
set -e
cd "$(dirname "$0")/.."
DATASET=$1; GPU=${2:-0}
for CONFIG in configs/$DATASET/*.yaml; do
  [ "$(basename "$CONFIG")" = "eval.yaml" ] && continue
  bash scripts/train.sh "$CONFIG" "$GPU"
done
bash scripts/evaluate.sh "$DATASET" "$GPU"

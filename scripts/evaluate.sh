#!/usr/bin/env bash
# Robust evaluation (missing rates 0.0-0.9) with the per-metric checkpoints listed in configs/<dataset>/eval.yaml.
# usage: bash scripts/evaluate.sh sims [gpu_id]
set -e
cd "$(dirname "$0")/.."
DATASET=$1; GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python -m semmsa.evaluate --spec "configs/$DATASET/eval.yaml" --out_dir results

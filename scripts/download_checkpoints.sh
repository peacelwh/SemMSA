#!/usr/bin/env bash
# Download the released checkpoints into checkpoints/ (currently CH-SIMS).
# usage: bash scripts/download_checkpoints.sh [sims]
set -e
cd "$(dirname "$0")/.."
DATASETS=${*:-"sims"}
python - $DATASETS <<'PY'
import sys
from huggingface_hub import snapshot_download
RELEASED = {"sims"}
for name in sys.argv[1:]:
    if name not in RELEASED:
        sys.exit(f"checkpoints for {name} are not released yet (available: {', '.join(sorted(RELEASED))})")
patterns = [f"{d}/*" for d in sys.argv[1:]]
snapshot_download(repo_id="Bluexxx/SemMSA", allow_patterns=patterns, local_dir="checkpoints")
print("checkpoints ->", "checkpoints/", patterns)
PY

#!/usr/bin/env bash
# Download the frozen encoders / LLM used by SemMSA into pretrained/.
set -e
cd "$(dirname "$0")/.."
mkdir -p pretrained
python - <<'PY'
from huggingface_hub import snapshot_download
for repo, local in [("google-bert/bert-base-uncased", "pretrained/bert-base-uncased"),
                    ("google-bert/bert-base-chinese", "pretrained/bert-base-chinese"),
                    ("Qwen/Qwen3-1.7B", "pretrained/Qwen3-1.7B")]:
    snapshot_download(repo_id=repo, local_dir=local, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "merges.txt", "vocab*"])
    print("downloaded", repo, "->", local)
PY

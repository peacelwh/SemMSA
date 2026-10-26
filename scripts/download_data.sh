#!/usr/bin/env bash
# Download the processed MOSI / MOSEI / SIMS features from the Hugging Face Hub into data/.
# usage: bash scripts/download_data.sh [mosi|mosei|sims ...]   (default: all three)
set -e
cd "$(dirname "$0")/.."
DATASETS=${*:-"mosi mosei sims"}
python - $DATASETS <<'PY'
import glob
import hashlib
import os
import shutil
import sys

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "Bluexxx/SemMSA-data"
FILES = {"mosi": "mosi_unaligned_50.pkl", "sims": "sims_unaligned_39.pkl"}
# MOSEI (13.7 GB) is stored as 1 GiB parts and reassembled here
MOSEI, MOSEI_SHA256 = "mosei_unaligned_50.pkl", "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


for name in sys.argv[1:]:
    if name in FILES:
        path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=FILES[name], local_dir="data")
    elif name == "mosei":
        snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=["mosei_unaligned_50/*"], local_dir="data")
        parts = sorted(glob.glob("data/mosei_unaligned_50/mosei_unaligned_50.pkl.part*"))
        path = os.path.join("data", MOSEI)
        with open(path, "wb") as out:
            for part in parts:
                with open(part, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 24)
        if sha256(path) != MOSEI_SHA256:
            sys.exit(f"checksum mismatch for {path}; delete data/mosei_unaligned_50/ and download again")
        shutil.rmtree("data/mosei_unaligned_50")
    else:
        sys.exit(f"unknown dataset: {name} (choose from mosi, mosei, sims)")
    print(name, "->", path)
PY

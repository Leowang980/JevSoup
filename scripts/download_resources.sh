#!/usr/bin/env bash
# Run in a clone; no reference to the research workspace is required.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
MODEL_SIZE="${1:-4b}"
case "$MODEL_SIZE" in 1.7b|4b|8b) ;; *) echo 'Usage: bash scripts/download_resources.sh 1.7b|4b|8b' >&2; exit 2 ;; esac
python -m jev_lora prepare
python -m jev_lora download --size "$MODEL_SIZE"
python -m jev_lora download-embedding --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
python -m jev_lora check --data artifacts/portal/data/eval.jsonl \
  --model-dir "models/portal-qwen3-$MODEL_SIZE" --embedding-dir models/qwen3-embedding-0.6b

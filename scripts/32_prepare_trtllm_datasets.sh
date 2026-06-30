#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/data/synthetic_requests"
DRY_RUN=0
NUM_REQUESTS="${NUM_REQUESTS:-100}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODEL_PATH="${MODEL_PATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
DATASETS=(isl128_osl128 isl512_osl128 isl1024_osl128 isl2048_osl256)

if [[ -n "${TRTLLM_PREPARE_DATASET_CMD_TEMPLATE:-}" ]]; then
  for dataset in "${DATASETS[@]}"; do
    isl="${dataset#isl}"; isl="${isl%%_osl*}"
    osl="${dataset##*_osl}"
    output="$OUT_DIR/${dataset}.jsonl"
    cmd="${TRTLLM_PREPARE_DATASET_CMD_TEMPLATE//\{dataset\}/$dataset}"
    cmd="${cmd//\{input_len\}/$isl}"
    cmd="${cmd//\{output_len\}/$osl}"
    cmd="${cmd//\{output\}/$output}"
    cmd="${cmd//\{num_requests\}/$NUM_REQUESTS}"
    echo "+ $cmd"
    [[ "$DRY_RUN" -eq 1 ]] || bash -lc "$cmd"
  done
  exit 0
fi

if command -v trtllm-bench >/dev/null 2>&1; then
  for dataset in "${DATASETS[@]}"; do
    isl="${dataset#isl}"; isl="${isl%%_osl*}"
    osl="${dataset##*_osl}"
    output="$OUT_DIR/${dataset}.json"
    global_args=(-m "$MODEL")
    if [[ -n "$MODEL_PATH" ]]; then
      global_args+=(--model_path "$MODEL_PATH")
    fi
    cmd=(trtllm-bench "${global_args[@]}" prepare-dataset --trust-remote-code --output "$output" token-unif-dist --num-requests "$NUM_REQUESTS" --input-min "$isl" --input-max "$isl" --output-min "$osl" --output-max "$osl")
    echo "+ ${cmd[*]}"
    [[ "$DRY_RUN" -eq 1 ]] || "${cmd[@]}"
  done
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Would generate fallback synthetic JSONL datasets in %s\n' "$OUT_DIR"
  exit 0
fi

"${PYTHON_BIN:-python3}" - "$OUT_DIR" "$NUM_REQUESTS" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
num_requests = int(sys.argv[2])
specs = [(128, 128), (512, 128), (1024, 128), (2048, 256)]
for input_len, output_len in specs:
    dataset = f"isl{input_len}_osl{output_len}"
    path = out_dir / f"{dataset}.jsonl"
    token = "benchmark"
    words = [token] * max(1, input_len)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(num_requests):
            record = {
                "id": f"{dataset}_{i:04d}",
                "dataset": dataset,
                "input_len": input_len,
                "output_len": output_len,
                "max_tokens": output_len,
                "prompt": " ".join(words),
                "synthetic": True,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(path)
PY

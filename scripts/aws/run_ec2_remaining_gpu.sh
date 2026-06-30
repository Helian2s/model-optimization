#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/model-optimization}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
S3_RESULT_PREFIX="${S3_RESULT_PREFIX:?Set S3_RESULT_PREFIX to an s3:// prefix for result upload.}"
SHUTDOWN_GUARD_MINUTES="${SHUTDOWN_GUARD_MINUTES:-240}"
TRTLLM_PROMPT_BACKEND="${TRTLLM_PROMPT_BACKEND:-examples}"
TRTLLM_PROMPT_TIMEOUT_SECONDS="${TRTLLM_PROMPT_TIMEOUT_SECONDS:-300}"
LLM_THINKING_MODE="${LLM_THINKING_MODE:-auto}"
HF_SYNTHETIC_REQUESTS="${HF_SYNTHETIC_REQUESTS:-20}"
HF_SYNTHETIC_WARMUP="${HF_SYNTHETIC_WARMUP:-2}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/workspace/huggingface-cache}"

cd "$ROOT_DIR"
mkdir -p results reports artifacts

upload_and_stop() {
  status=$?
  printf '{"status":%s,"finished_at":"%s","scope":"remaining_gpu"}\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/gpu_remaining_status.json || true
  aws s3 sync results "$S3_RESULT_PREFIX/results" --only-show-errors || true
  aws s3 sync reports "$S3_RESULT_PREFIX/reports" --only-show-errors || true
  sudo shutdown -h now || true
  exit "$status"
}
trap upload_and_stop EXIT

sudo shutdown -h +"$SHUTDOWN_GUARD_MINUTES" "Safety stop for TensorRT-LLM remaining GPU run"

run_container() {
  local command="$1"
  bash scripts/02_start_trtllm_container.sh --run "$command"
}

if [[ ! -f results/model_download.json ]]; then
  run_container "python3 scripts/10_download_model.py --model '$MODEL' --cache-dir '$HF_CACHE_DIR'"
fi

MODEL_PATH="$(
  python3 - <<'PY'
import json
from pathlib import Path
path = Path("results/model_download.json")
if path.exists():
    print(json.loads(path.read_text()).get("snapshot_path", ""))
PY
)"

if [[ ! -d artifacts/trtllm_engine_bf16_or_fp16 ]] || [[ -z "$(find artifacts/trtllm_engine_bf16_or_fp16 -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]]; then
  run_container "python3 scripts/30_build_trtllm_engine.py --model '$MODEL' --model-download-json results/model_download.json"
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="results/pre_remaining_gpu_${timestamp}"
mkdir -p "$backup_dir"
for file in \
  hf_baseline_raw.jsonl \
  hf_baseline_summary.json \
  trtllm_outputs_raw.jsonl \
  trtllm_prompt_summary.json \
  quality_regression.json \
  quality_regression.md \
  summary.csv \
  summary.json; do
  if [[ -f "results/$file" ]]; then
    cp "results/$file" "$backup_dir/$file"
  fi
done

rm -f \
  results/hf_baseline_raw.jsonl \
  results/hf_baseline_summary.json \
  results/trtllm_outputs_raw.jsonl \
  results/trtllm_prompt_summary.json \
  results/quality_regression.json \
  results/quality_regression.md

run_container "python3 scripts/21_run_hf_baseline.py --model '$MODEL' --cache-dir '$HF_CACHE_DIR' --thinking-mode '$LLM_THINKING_MODE'"
run_container "python3 scripts/22_run_hf_synthetic.py --model '$MODEL' --cache-dir '$HF_CACHE_DIR' --requests '$HF_SYNTHETIC_REQUESTS' --warmup-requests '$HF_SYNTHETIC_WARMUP' --datasets isl1024_osl128 isl2048_osl256"
run_container "python3 scripts/36_run_trtllm_prompt_set.py --model '$MODEL' --model-download-json results/model_download.json --backend '$TRTLLM_PROMPT_BACKEND' --examples-timeout-seconds '$TRTLLM_PROMPT_TIMEOUT_SECONDS' --thinking-mode '$LLM_THINKING_MODE'"

python3 scripts/40_quality_regression.py

run_container "MODEL_PATH='$MODEL_PATH' bash scripts/35_run_trtllm_quantized_or_kv_cache.sh --model '$MODEL' --model-path '$MODEL_PATH'"

if ! run_container "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH' bash scripts/60_profile_nsys.sh"; then
  mkdir -p results/nsys
  printf 'Nsight Systems profiling failed or nsys was unavailable. See profile_command.log if present.\n' > results/nsys/profile_failed.txt
fi

python3 scripts/50_parse_results.py

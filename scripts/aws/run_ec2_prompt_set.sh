#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/model-optimization}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
S3_RESULT_PREFIX="${S3_RESULT_PREFIX:?Set S3_RESULT_PREFIX to an s3:// prefix for result upload.}"
SHUTDOWN_GUARD_MINUTES="${SHUTDOWN_GUARD_MINUTES:-120}"
TRTLLM_PROMPT_BACKEND="${TRTLLM_PROMPT_BACKEND:-examples}"
TRTLLM_PROMPT_TIMEOUT_SECONDS="${TRTLLM_PROMPT_TIMEOUT_SECONDS:-300}"
LLM_THINKING_MODE="${LLM_THINKING_MODE:-auto}"

cd "$ROOT_DIR"
mkdir -p results reports

upload_and_stop() {
  status=$?
  printf '{"status":%s,"finished_at":"%s","scope":"prompt_set"}\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/gpu_prompt_set_status.json || true
  aws s3 sync results "$S3_RESULT_PREFIX/results" --only-show-errors || true
  aws s3 sync reports "$S3_RESULT_PREFIX/reports" --only-show-errors || true
  sudo shutdown -h now || true
  exit "$status"
}
trap upload_and_stop EXIT

sudo shutdown -h +"$SHUTDOWN_GUARD_MINUTES" "Safety stop for TensorRT-LLM prompt-set run"

bash scripts/02_start_trtllm_container.sh \
  --run "python3 scripts/36_run_trtllm_prompt_set.py --model '$MODEL' --model-download-json results/model_download.json --backend '$TRTLLM_PROMPT_BACKEND' --resume --examples-timeout-seconds '$TRTLLM_PROMPT_TIMEOUT_SECONDS' --thinking-mode '$LLM_THINKING_MODE'"

python3 scripts/40_quality_regression.py
python3 scripts/50_parse_results.py

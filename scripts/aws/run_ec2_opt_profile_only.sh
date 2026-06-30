#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/model-optimization}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
S3_RESULT_PREFIX="${S3_RESULT_PREFIX:?Set S3_RESULT_PREFIX to an s3:// prefix for result upload.}"
SHUTDOWN_GUARD_MINUTES="${SHUTDOWN_GUARD_MINUTES:-120}"

cd "$ROOT_DIR"
mkdir -p results reports

upload_and_stop() {
  status=$?
  printf '{"status":%s,"finished_at":"%s","scope":"opt_profile_only"}\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/gpu_opt_profile_status.json || true
  aws s3 sync results "$S3_RESULT_PREFIX/results" --only-show-errors || true
  aws s3 sync reports "$S3_RESULT_PREFIX/reports" --only-show-errors || true
  sudo shutdown -h now || true
  exit "$status"
}
trap upload_and_stop EXIT

sudo shutdown -h +"$SHUTDOWN_GUARD_MINUTES" "Safety stop for TensorRT-LLM optimization/profile run"

MODEL_PATH="$(
  python3 - <<'PY'
import json
from pathlib import Path
path = Path("results/model_download.json")
if path.exists():
    print(json.loads(path.read_text()).get("snapshot_path", ""))
PY
)"

run_container() {
  local command="$1"
  bash scripts/02_start_trtllm_container.sh --run "$command"
}

run_container "MODEL_PATH='$MODEL_PATH' bash scripts/35_run_trtllm_quantized_or_kv_cache.sh --model '$MODEL' --model-path '$MODEL_PATH'"

if ! run_container "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH' bash scripts/60_profile_nsys.sh"; then
  mkdir -p results/nsys
  printf 'Nsight Systems profiling failed or nsys was unavailable. See profile_command.log if present.\n' > results/nsys/profile_failed.txt
fi

python3 scripts/50_parse_results.py

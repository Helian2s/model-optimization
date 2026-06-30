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
  printf '{"status":%s,"finished_at":"%s","scope":"throughput_retry"}\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/gpu_throughput_retry_status.json || true
  aws s3 sync results "$S3_RESULT_PREFIX/results" --only-show-errors || true
  aws s3 sync reports "$S3_RESULT_PREFIX/reports" --only-show-errors || true
  sudo shutdown -h now || true
  exit "$status"
}
trap upload_and_stop EXIT

sudo shutdown -h +"$SHUTDOWN_GUARD_MINUTES" "Safety stop for TensorRT-LLM throughput retry"

MODEL_PATH="$(
  python3 - <<'PY'
import json
from pathlib import Path
path = Path("results/model_download.json")
if path.exists():
    print(json.loads(path.read_text()).get("snapshot_path", ""))
PY
)"

bash scripts/02_start_trtllm_container.sh \
  --run "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH' bash scripts/34_run_trtllm_throughput.sh --quick"

python3 scripts/50_parse_results.py

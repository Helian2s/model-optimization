#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/model-optimization}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODE="${MODE:-quick}"
S3_RESULT_PREFIX="${S3_RESULT_PREFIX:?Set S3_RESULT_PREFIX to an s3:// prefix for result upload.}"
SHUTDOWN_GUARD_MINUTES="${SHUTDOWN_GUARD_MINUTES:-480}"

cd "$ROOT_DIR"
mkdir -p results artifacts reports

upload_and_stop() {
  status=$?
  printf '{"status":%s,"finished_at":"%s"}\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/gpu_session_status.json || true
  aws s3 sync results "$S3_RESULT_PREFIX/results" --only-show-errors || true
  aws s3 sync reports "$S3_RESULT_PREFIX/reports" --only-show-errors || true
  aws s3 sync artifacts "$S3_RESULT_PREFIX/artifacts" --only-show-errors || true
  sudo shutdown -h now || true
  exit "$status"
}
trap upload_and_stop EXIT

bash scripts/90_run_all.sh \
  --gpu \
  "--$MODE" \
  --model "$MODEL" \
  --shutdown-guard-minutes "$SHUTDOWN_GUARD_MINUTES" \
  --no-final-shutdown \
  --skip-optimization \
  --skip-quality \
  --skip-profile

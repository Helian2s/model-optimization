#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="Qwen/Qwen3-1.7B"
MODE="quick"
LOCAL_ONLY=0
GPU=0
DRY_RUN=0
SHUTDOWN_GUARD_MINUTES=240
NO_SHUTDOWN_GUARD=0
FINAL_SHUTDOWN=1
SKIP_OPTIMIZATION=0
SKIP_QUALITY=0
SKIP_PROFILE=0
CONTAINER_IMAGE="${TRTLLM_CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/workspace/huggingface-cache}"
TRTLLM_PROMPT_BACKEND="${TRTLLM_PROMPT_BACKEND:-examples}"
TRTLLM_PROMPT_TIMEOUT_SECONDS="${TRTLLM_PROMPT_TIMEOUT_SECONDS:-300}"
LLM_THINKING_MODE="${LLM_THINKING_MODE:-auto}"
HF_TOKEN_VALUE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --quick) MODE="quick"; shift ;;
    --full) MODE="full"; shift ;;
    --local-only) LOCAL_ONLY=1; shift ;;
    --gpu) GPU=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --shutdown-guard-minutes) SHUTDOWN_GUARD_MINUTES="$2"; shift 2 ;;
    --no-shutdown-guard) NO_SHUTDOWN_GUARD=1; shift ;;
    --no-final-shutdown) FINAL_SHUTDOWN=0; shift ;;
    --skip-optimization) SKIP_OPTIMIZATION=1; shift ;;
    --skip-quality) SKIP_QUALITY=1; shift ;;
    --skip-profile) SKIP_PROFILE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No Python interpreter found." >&2
    exit 1
  fi
fi
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN

run() {
  echo "+ $*"
  "$@"
}

run_in_container() {
  local command="$1"
  echo "+ [container] $command"
  HF_TOKEN="$HF_TOKEN_VALUE" HUGGINGFACE_HUB_TOKEN="$HF_TOKEN_VALUE" \
    bash scripts/02_start_trtllm_container.sh --image "$CONTAINER_IMAGE" --run "$command"
}

is_ec2_instance() {
  token="$(curl -fsS --max-time 0.35 -X PUT \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
    http://169.254.169.254/latest/api/token 2>/dev/null || true)"
  [[ -n "$token" ]]
}

schedule_shutdown_guard() {
  if [[ "$NO_SHUTDOWN_GUARD" -eq 1 ]]; then
    echo "Refusing to run GPU mode without shutdown guard. Remove --no-shutdown-guard." >&2
    exit 1
  fi
  echo "+ sudo shutdown -h +$SHUTDOWN_GUARD_MINUTES"
  sudo shutdown -h +"$SHUTDOWN_GUARD_MINUTES" "Safety stop for TensorRT-LLM benchmark"
}

stop_instance_from_inside() {
  if [[ "$GPU" -eq 1 && "$FINAL_SHUTDOWN" -eq 1 ]]; then
    echo "Stopping EC2 instance from inside OS with sudo shutdown -h now."
    sudo shutdown -h now || true
  fi
}

if [[ "$LOCAL_ONLY" -eq 1 ]]; then
  run "$PYTHON_BIN" -m src.prompt_loader data/prompts.jsonl
  run bash scripts/00_check_env.sh --output-dir results/local_check
  run "$PYTHON_BIN" scripts/10_download_model.py --model "$MODEL" --dry-run --output-dir results/local_check
  run "$PYTHON_BIN" scripts/22_run_hf_synthetic.py --model "$MODEL" --dry-run --output-dir results/local_check
  run "$PYTHON_BIN" scripts/30_build_trtllm_engine.py --model "$MODEL" --dry-run --metadata-out results/local_check/trtllm_build_metadata.json
  run "$PYTHON_BIN" scripts/31_run_trtllm_smoke.py --dry-run --output results/local_check/trtllm_smoke.json
  run "$PYTHON_BIN" scripts/36_run_trtllm_prompt_set.py --model "$MODEL" --dry-run --output-dir results/local_check
  run bash scripts/32_prepare_trtllm_datasets.sh --dry-run
  run bash scripts/33_run_trtllm_latency.sh --quick --dry-run
  run bash scripts/34_run_trtllm_throughput.sh --quick --dry-run
  run env RESULTS_DIR="$ROOT_DIR/results/local_check" bash scripts/35_run_trtllm_quantized_or_kv_cache.sh --model "$MODEL" --dry-run
  run env OUT_DIR="$ROOT_DIR/results/local_check/nsys" bash scripts/60_profile_nsys.sh --dry-run
  run "$PYTHON_BIN" scripts/40_quality_regression.py --output-dir results/local_check
  run "$PYTHON_BIN" scripts/50_parse_results.py --fixtures-only --results-dir results/local_check --reports-dir reports/local_check
  echo "Local-only validation completed. EC2 was not started."
  exit 0
fi

if [[ "$GPU" -ne 1 ]]; then
  echo "Nothing to run. Use --local-only for local validation or --gpu when already on the EC2 GPU instance." >&2
  exit 2
fi

if ! is_ec2_instance && [[ "${I_AM_ON_EC2_GPU:-0}" != "1" ]]; then
  echo "GPU mode must run on the EC2 GPU instance. EC2 was not started by this script." >&2
  exit 1
fi

schedule_shutdown_guard
if [[ "$FINAL_SHUTDOWN" -eq 1 ]]; then
  trap stop_instance_from_inside EXIT
fi

if command -v aws >/dev/null 2>&1; then
  HF_TOKEN_VALUE="$(aws ssm get-parameter --name /finetuning/huggingface/token --with-decryption --region us-west-2 --query Parameter.Value --output text 2>/dev/null || true)"
fi

mkdir -p /mnt/workspace/huggingface-cache /mnt/workspace/model-optimization-artifacts 2>/dev/null || true

run bash scripts/00_check_env.sh
run_in_container "python3 scripts/10_download_model.py --model '$MODEL' --cache-dir '$HF_CACHE_DIR'"
run_in_container "python3 scripts/21_run_hf_baseline.py --model '$MODEL' --cache-dir '$HF_CACHE_DIR' --thinking-mode '$LLM_THINKING_MODE'"
run_in_container "python3 scripts/22_run_hf_synthetic.py --model '$MODEL' --cache-dir '$HF_CACHE_DIR' --requests 20 --warmup-requests 2 --datasets isl1024_osl128 isl2048_osl256"
run_in_container "python3 scripts/30_build_trtllm_engine.py --model '$MODEL' --model-download-json results/model_download.json"
run_in_container "python3 scripts/31_run_trtllm_smoke.py --model '$MODEL'"
run_in_container "python3 scripts/36_run_trtllm_prompt_set.py --model '$MODEL' --model-download-json results/model_download.json --backend '$TRTLLM_PROMPT_BACKEND' --resume --examples-timeout-seconds '$TRTLLM_PROMPT_TIMEOUT_SECONDS' --thinking-mode '$LLM_THINKING_MODE'"
MODEL_PATH_VALUE="$(python3 - <<'PY'
import json
from pathlib import Path
p = Path('results/model_download.json')
if p.exists():
    print(json.loads(p.read_text()).get('snapshot_path', ''))
PY
)"
run_in_container "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH_VALUE' bash scripts/32_prepare_trtllm_datasets.sh --num-requests 100"
if [[ "$MODE" == "quick" ]]; then
  run_in_container "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH_VALUE' bash scripts/33_run_trtllm_latency.sh --quick"
  run_in_container "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH_VALUE' bash scripts/34_run_trtllm_throughput.sh --quick"
else
  run_in_container "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH_VALUE' bash scripts/33_run_trtllm_latency.sh --requests 100"
  run_in_container "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH_VALUE' bash scripts/34_run_trtllm_throughput.sh"
fi
if [[ "$SKIP_OPTIMIZATION" -eq 0 ]]; then
  run_in_container "MODEL_PATH='$MODEL_PATH_VALUE' bash scripts/35_run_trtllm_quantized_or_kv_cache.sh --model '$MODEL' --model-path '$MODEL_PATH_VALUE'"
fi
if [[ "$SKIP_QUALITY" -eq 0 ]]; then
  run_in_container "python3 scripts/40_quality_regression.py"
fi
if [[ "$SKIP_PROFILE" -eq 0 ]]; then
  run_in_container "MODEL='$MODEL' MODEL_PATH='$MODEL_PATH_VALUE' bash scripts/60_profile_nsys.sh"
fi
run "$PYTHON_BIN" scripts/50_parse_results.py

if [[ "$FINAL_SHUTDOWN" -eq 1 ]]; then
  echo "GPU run completed. Shutdown will be triggered by EXIT trap."
else
  echo "GPU run completed. Final shutdown was delegated to the caller."
fi

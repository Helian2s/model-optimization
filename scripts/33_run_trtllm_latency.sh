#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="${ENGINE_DIR:-$ROOT_DIR/artifacts/trtllm_engine_bf16_or_fp16}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/synthetic_requests}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
REQUESTS="${REQUESTS:-20}"
WARMUP="${WARMUP:-5}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODEL_PATH="${MODEL_PATH:-}"
DRY_RUN=0
LOG_PREFIX="${LOG_PREFIX:-trtllm_latency}"
DATASETS=(isl128_osl128 isl512_osl128 isl1024_osl128 isl2048_osl256)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) DATASETS=(isl1024_osl128 isl2048_osl256); shift ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RESULTS_DIR"

if [[ -z "${TRTLLM_LATENCY_CMD_TEMPLATE:-}" ]] && ! command -v trtllm-bench >/dev/null 2>&1; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run: trtllm-bench not required locally."
  else
    echo "trtllm-bench not found. Run inside TensorRT-LLM container or set TRTLLM_LATENCY_CMD_TEMPLATE." >&2
    exit 1
  fi
fi

for dataset in "${DATASETS[@]}"; do
  dataset_path="$DATA_DIR/${dataset}.json"
  if [[ ! -f "$dataset_path" ]]; then
    dataset_path="$DATA_DIR/${dataset}.jsonl"
  fi
  log="$RESULTS_DIR/${LOG_PREFIX}_${dataset}.log"
  report_json="$RESULTS_DIR/${LOG_PREFIX}_${dataset}_report.json"
  iteration_log="$RESULTS_DIR/${LOG_PREFIX}_${dataset}_iterations.jsonl"
  if [[ -n "${TRTLLM_LATENCY_CMD_TEMPLATE:-}" ]]; then
    cmd="$TRTLLM_LATENCY_CMD_TEMPLATE"
    cmd="${cmd//\{engine_dir\}/$ENGINE_DIR}"
    cmd="${cmd//\{dataset\}/$dataset_path}"
    cmd="${cmd//\{requests\}/$REQUESTS}"
    cmd="${cmd//\{warmup\}/$WARMUP}"
  else
    model_args="-m '$MODEL'"
    if [[ -n "$MODEL_PATH" ]]; then
      model_args="$model_args --model_path '$MODEL_PATH'"
    fi
    cmd="trtllm-bench $model_args latency --backend tensorrt --engine_dir '$ENGINE_DIR' --dataset '$dataset_path' --num_requests '$REQUESTS' --warmup '$WARMUP' --concurrency 1 --report_json '$report_json' --iteration_log '$iteration_log'"
  fi
  echo "+ $cmd" | tee "$log"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    bash -lc "$cmd" >>"$log" 2>&1
  fi
done

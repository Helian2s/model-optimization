#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/nsys}"
ENGINE_DIR="${ENGINE_DIR:-$ROOT_DIR/artifacts/trtllm_engine_bf16_or_fp16}"
DATASET="${DATASET:-$ROOT_DIR/data/synthetic_requests/isl1024_osl128.jsonl}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODEL_PATH="${MODEL_PATH:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
if [[ ! -f "$DATASET" && "$DATASET" == *.jsonl ]]; then
  json_dataset="${DATASET%.jsonl}.json"
  if [[ -f "$json_dataset" ]]; then
    DATASET="$json_dataset"
  fi
fi
if [[ -n "${NSYS_PROFILE_CMD_TEMPLATE:-}" ]]; then
  target="$NSYS_PROFILE_CMD_TEMPLATE"
  target="${target//\{engine_dir\}/$ENGINE_DIR}"
  target="${target//\{dataset\}/$DATASET}"
else
  model_args="-m '$MODEL'"
  if [[ -n "$MODEL_PATH" ]]; then
    model_args="$model_args --model_path '$MODEL_PATH'"
  fi
  target="trtllm-bench $model_args latency --backend tensorrt --engine_dir '$ENGINE_DIR' --dataset '$DATASET' --num_requests 20 --warmup 5 --concurrency 1"
fi

cmd="nsys profile --force-overwrite=true --duration=90 --output '$OUT_DIR/trtllm_steady_state' bash -lc \"$target\""
echo "+ $cmd" | tee "$OUT_DIR/profile_command.log"
if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi
command -v nsys >/dev/null 2>&1 || { echo "nsys not found. Run inside an NVIDIA environment with Nsight Systems installed." >&2; exit 1; }
set +e
bash -lc "$cmd" >>"$OUT_DIR/profile_command.log" 2>&1
profile_status=$?
set -e
if [[ "$profile_status" -ne 0 && ! -f "$OUT_DIR/trtllm_steady_state.nsys-rep" ]]; then
  exit "$profile_status"
fi

{
  echo "# Nsight Systems Profiling"
  echo
  echo "- Target command: \`$target\`"
  echo "- Output prefix: \`$OUT_DIR/trtllm_steady_state\`"
  echo
  if command -v nsys >/dev/null 2>&1 && [[ -f "$OUT_DIR/trtllm_steady_state.nsys-rep" ]]; then
    if [[ "$profile_status" -ne 0 ]]; then
      echo "The `nsys profile` command returned status $profile_status, but a report file was generated. This can happen when the profiled benchmark re-parents worker processes during shutdown."
      echo
    fi
    echo "## Stats"
    echo
    echo "\`\`\`text"
    nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum "$OUT_DIR/trtllm_steady_state.nsys-rep" 2>&1 | head -200 || true
    echo "\`\`\`"
  else
    echo "The nsys report file was not found after profiling."
  fi
  echo
  echo "## Interpretation Checklist"
  echo
  echo "- CPU launch gaps: inspect CUDA API summary and timeline in Nsight Systems."
  echo "- GPU busy time: inspect kernel density in the `.nsys-rep` timeline."
  echo "- NCCL: single-GPU runs should not be NCCL dominated."
  echo "- Dominant kernels: inspect CUDA GPU kernel summary for attention/GEMM/decode operations."
} >"$OUT_DIR/profile_summary.md"

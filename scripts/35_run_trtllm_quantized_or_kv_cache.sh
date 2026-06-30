#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
BASE_ENGINE_DIR="${BASE_ENGINE_DIR:-$ROOT_DIR/artifacts/trtllm_engine_bf16_or_fp16}"
OPT_ENGINE_DIR="${OPT_ENGINE_DIR:-$ROOT_DIR/artifacts/trtllm_engine_fp8_kv}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-$ROOT_DIR/artifacts/trtllm_checkpoint_bf16_or_fp16}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/synthetic_requests}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODEL_PATH="${MODEL_PATH:-}"
DRY_RUN=0
REQUESTS="${REQUESTS:-20}"
WARMUP="${WARMUP:-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RESULTS_DIR" "$OPT_ENGINE_DIR"
attempt_log="$RESULTS_DIR/trtllm_optimization_attempt.json"
attempt_md="$RESULTS_DIR/trtllm_optimization_attempt.md"
status="not_attempted"
commands_json="$RESULTS_DIR/trtllm_optimization_commands.jsonl"
: >"$commands_json"

record_command() {
  local label="$1"
  local command="$2"
  "${PYTHON_BIN:-python3}" - "$commands_json" "$label" "$command" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"label": sys.argv[2], "command": sys.argv[3]}, sort_keys=True) + "\n")
PY
}

run_logged() {
  local label="$1"
  local log="$2"
  local command="$3"
  record_command "$label" "$command"
  echo "+ $command" | tee "$log"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  bash -lc "$command" >>"$log" 2>&1
}

model_args="-m '$MODEL'"
if [[ -n "$MODEL_PATH" ]]; then
  model_args="$model_args --model_path '$MODEL_PATH'"
fi

build_fp8_kv() {
  if [[ -n "${TRTLLM_OPT_BUILD_CMD_TEMPLATE:-}" ]]; then
    local cmd="$TRTLLM_OPT_BUILD_CMD_TEMPLATE"
    cmd="${cmd//\{model\}/$MODEL}"
    cmd="${cmd//\{engine_dir\}/$OPT_ENGINE_DIR}"
    cmd="${cmd//\{checkpoint_dir\}/$BASE_CKPT_DIR}"
    run_logged "custom_optimization_build" "$RESULTS_DIR/trtllm_opt_build.log" "$cmd"
    status="custom_template_attempted"
    return 0
  fi

  if ! command -v trtllm-build >/dev/null 2>&1; then
    status="no_trtllm_build_for_quantization"
    return 1
  fi

  local help_text
  help_text="$(trtllm-build --help 2>&1 || true)"
  local flag=""
  if grep -q -- "--kv_cache_dtype" <<<"$help_text"; then
    flag="--kv_cache_dtype"
  elif grep -q -- "--kv-cache-dtype" <<<"$help_text"; then
    flag="--kv-cache-dtype"
  else
    status="fp8_kv_flag_not_exposed"
    return 1
  fi

  local cmd="trtllm-build --checkpoint_dir '$BASE_CKPT_DIR' --output_dir '$OPT_ENGINE_DIR' $flag fp8"
  if run_logged "fp8_kv_build" "$RESULTS_DIR/trtllm_fp8_kv_build.log" "$cmd"; then
    status="fp8_kv_built"
    return 0
  fi
  status="fp8_kv_build_failed"
  return 1
}

run_optimized_benchmarks() {
  ENGINE_DIR="$OPT_ENGINE_DIR" RESULTS_DIR="$RESULTS_DIR" LOG_PREFIX="trtllm_latency_fp8_kv" \
    "$ROOT_DIR/scripts/33_run_trtllm_latency.sh" --quick --requests "$REQUESTS" --warmup "$WARMUP" --model "$MODEL" --model-path "$MODEL_PATH"
  ENGINE_DIR="$OPT_ENGINE_DIR" RESULTS_DIR="$RESULTS_DIR" LOG_PREFIX="trtllm_throughput_fp8_kv" \
    "$ROOT_DIR/scripts/34_run_trtllm_throughput.sh" --quick --model "$MODEL" --model-path "$MODEL_PATH"
}

runtime_kv_fallback() {
  if ! command -v trtllm-bench >/dev/null 2>&1; then
    status="${status}_and_no_trtllm_bench_for_runtime_fallback"
    return 1
  fi
  local help_text
  help_text="$(trtllm-bench latency --help 2>&1 || true)"
  local kv_flag=""
  local kv_value=""
  if grep -q -- "--kv_cache_free_gpu_mem_fraction" <<<"$help_text"; then
    kv_flag="--kv_cache_free_gpu_mem_fraction"
    kv_value="0.70"
  elif grep -q -- "--kv_cache_free_gpu_memory_fraction" <<<"$help_text"; then
    kv_flag="--kv_cache_free_gpu_memory_fraction"
    kv_value="0.70"
  elif grep -q -- "--kv_cache_percentage" <<<"$help_text"; then
    kv_flag="--kv_cache_percentage"
    kv_value="0.70"
  else
    status="${status}_and_runtime_kv_fraction_flag_not_exposed"
    return 1
  fi

  local fallback_status="runtime_kv_fraction_attempted"
  for dataset in isl1024_osl128 isl2048_osl256; do
    local isl="${dataset#isl}"
    isl="${isl%%_osl*}"
    local osl="${dataset##*_osl}"
    local dataset_path="$DATA_DIR/${dataset}.json"
    if [[ ! -f "$dataset_path" ]]; then
      dataset_path="$DATA_DIR/${dataset}.jsonl"
    fi
    local latency_report="$RESULTS_DIR/trtllm_latency_kv_runtime_${dataset}_report.json"
    local latency_log="$RESULTS_DIR/trtllm_latency_kv_runtime_${dataset}.log"
    local latency_cmd="trtllm-bench $model_args latency --backend tensorrt --engine_dir '$BASE_ENGINE_DIR' --dataset '$dataset_path' --num_requests '$REQUESTS' --warmup '$WARMUP' --concurrency 1 $kv_flag '$kv_value' --report_json '$latency_report'"
    if ! run_logged "runtime_kv_latency_${dataset}" "$latency_log" "$latency_cmd"; then
      fallback_status="runtime_kv_fraction_failed"
    fi

    local throughput_report="$RESULTS_DIR/trtllm_throughput_kv_runtime_${dataset}_report.json"
    local throughput_log="$RESULTS_DIR/trtllm_throughput_kv_runtime_${dataset}.log"
    local throughput_cmd="trtllm-bench $model_args throughput --backend tensorrt --engine_dir '$BASE_ENGINE_DIR' --dataset '$dataset_path' --num_requests 20 --warmup 5 --target_input_len '$isl' --target_output_len '$osl' $kv_flag '$kv_value' --report_json '$throughput_report'"
    if ! run_logged "runtime_kv_throughput_${dataset}" "$throughput_log" "$throughput_cmd"; then
      fallback_status="runtime_kv_fraction_failed"
    fi
  done
  status="$fallback_status"
}

if build_fp8_kv; then
  if [[ "$DRY_RUN" -eq 0 ]]; then
    run_optimized_benchmarks
  fi
else
  runtime_kv_fallback || true
fi

"${PYTHON_BIN:-python3}" - "$attempt_log" "$attempt_md" "$status" "$OPT_ENGINE_DIR" "$BASE_ENGINE_DIR" "$commands_json" <<'PY'
import json
import sys
from pathlib import Path

attempt_log = Path(sys.argv[1])
attempt_md = Path(sys.argv[2])
commands_path = Path(sys.argv[6])
commands = []
if commands_path.exists():
    commands = [json.loads(line) for line in commands_path.read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "status": sys.argv[3],
    "optimized_engine_dir": sys.argv[4],
    "base_engine_dir": sys.argv[5],
    "attempt_order": [
        "fp8_kv_cache",
        "fp8_weight_activation",
        "int8",
        "int4_awq_or_gptq",
        "documented_kv_cache_runtime_fallback",
    ],
    "commands": commands,
}
attempt_log.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# TensorRT-LLM Optimization Attempt",
    "",
    f"- Status: `{payload['status']}`",
    f"- Optimized engine dir: `{payload['optimized_engine_dir']}`",
    f"- Base engine dir: `{payload['base_engine_dir']}`",
    "",
    "## Attempt Order",
    "",
]
lines.extend(f"- `{item}`" for item in payload["attempt_order"])
lines.extend(["", "## Commands", ""])
if commands:
    lines.extend(f"- `{item['label']}`: `{item['command']}`" for item in commands)
else:
    lines.append("No executable optimization command was available in this environment.")
lines.append("")
attempt_md.write_text("\n".join(lines), encoding="utf-8")
PY

echo "Optimization attempt status: $status"

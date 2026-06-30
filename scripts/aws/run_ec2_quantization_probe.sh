#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/model-optimization}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
FALLBACK_MODEL="${FALLBACK_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TRTLLM_TEST_IMAGE="${TRTLLM_TEST_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.1}"
S3_RESULT_PREFIX="${S3_RESULT_PREFIX:?Set S3_RESULT_PREFIX to an s3:// prefix for result upload.}"
SHUTDOWN_GUARD_MINUTES="${SHUTDOWN_GUARD_MINUTES:-180}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/workspace/huggingface-cache}"
TRY_FP8_KV_SMOKE="${TRY_FP8_KV_SMOKE:-1}"

cd "$ROOT_DIR"
mkdir -p results reports artifacts

RUN_ID="${RUN_ID:-quant_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
PROBE_DIR="results/$RUN_ID"
REPORT_DIR="reports/$RUN_ID"
mkdir -p "$PROBE_DIR" "$REPORT_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$PROBE_DIR/engineer.log"
}

upload_and_stop() {
  status=$?
  log "Trap fired with status=$status; syncing results and stopping instance."
  printf '{"status":%s,"finished_at":"%s","scope":"quantization_probe","run_id":"%s","image":"%s"}\n' \
    "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "$TRTLLM_TEST_IMAGE" \
    > "$PROBE_DIR/gpu_quantization_probe_status.json" || true
  aws s3 sync "$PROBE_DIR" "$S3_RESULT_PREFIX/results/$RUN_ID" --only-show-errors || true
  aws s3 sync "$REPORT_DIR" "$S3_RESULT_PREFIX/reports/$RUN_ID" --only-show-errors || true
  sudo shutdown -h now || true
  exit "$status"
}
trap upload_and_stop EXIT

log "Starting TensorRT-LLM quantization probe."
log "Run ID: $RUN_ID"
log "Primary model: $MODEL"
log "Fallback model: $FALLBACK_MODEL"
log "Test image: $TRTLLM_TEST_IMAGE"

sudo shutdown -c || true
sudo shutdown -h +"$SHUTDOWN_GUARD_MINUTES" "Safety stop for TensorRT-LLM quantization probe"
log "Scheduled shutdown guard for $SHUTDOWN_GUARD_MINUTES minutes."

log "Pulling NVIDIA TensorRT-LLM image."
docker pull "$TRTLLM_TEST_IMAGE" 2>&1 | tee "$PROBE_DIR/docker_pull.log"
docker image inspect "$TRTLLM_TEST_IMAGE" --format '{{json .RepoDigests}}' > "$PROBE_DIR/container_repo_digests.json" 2>/dev/null || true

run_container() {
  local command="$1"
  log "Container command: $command"
  bash scripts/02_start_trtllm_container.sh --image "$TRTLLM_TEST_IMAGE" --run "$command"
}

log "Downloading or reusing primary model snapshot."
MODEL_DOWNLOAD_DIR="$PROBE_DIR/model_download"
MODEL_DOWNLOAD_JSON="$MODEL_DOWNLOAD_DIR/model_download.json"
if ! run_container "python3 scripts/10_download_model.py --model '$MODEL' --cache-dir '$HF_CACHE_DIR' --output-dir '$MODEL_DOWNLOAD_DIR'"; then
  log "Primary model download failed; trying fallback model."
  MODEL="$FALLBACK_MODEL"
  run_container "python3 scripts/10_download_model.py --model '$MODEL' --cache-dir '$HF_CACHE_DIR' --output-dir '$MODEL_DOWNLOAD_DIR'"
fi

MODEL_PATH="$(
  python3 - "$MODEL_DOWNLOAD_JSON" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if path.exists():
    print(json.loads(path.read_text()).get("snapshot_path", ""))
PY
)"
log "Resolved model path: ${MODEL_PATH:-not found}"

probe_flags=""
if [[ "$TRY_FP8_KV_SMOKE" == "1" ]]; then
  probe_flags="--try-fp8-kv"
fi

log "Inspecting quantization APIs, CLI flags, and examples."
run_container "python3 scripts/37_probe_trtllm_quantization.py --model '$MODEL' --model-path '$MODEL_PATH' --output '$PROBE_DIR/quantization_probe.json' $probe_flags"

log "Creating human-readable probe summary."
python3 - "$PROBE_DIR/quantization_probe.json" "$REPORT_DIR/quantization_probe_summary.md" <<'PY'
import json
import sys
from pathlib import Path

probe_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
data = json.loads(probe_path.read_text(encoding="utf-8"))
flags = data.get("flag_summary", {})
api = data.get("python_api", {})
smoke = data.get("fp8_kv_smoke", {})
versions = data.get("package_versions", {})
lines = [
    "# TensorRT-LLM Quantization Probe Summary",
    "",
    f"- Captured UTC: `{data.get('captured_at_utc')}`",
    f"- Model: `{data.get('model')}`",
    f"- Model path: `{data.get('model_path')}`",
    "",
    "## Package Versions",
    "",
]
for key, value in versions.items():
    lines.append(f"- `{key}`: `{value}`")
lines.extend(["", "## Python API", ""])
for key in [
    "tensorrt_llm_import",
    "tensorrt_llm_version_attr",
    "tensorrt_llm.LLM",
    "tensorrt_llm.llmapi.KvCacheConfig",
    "kv_cache_config_fp8_construct",
    "tensorrt_llm.quantization.QuantMode",
]:
    lines.append(f"- `{key}`: `{api.get(key)}`")
lines.extend(["", "## CLI Flag Summary", ""])
for command_name, command_flags in flags.items():
    lines.append(f"### {command_name}")
    for key, value in command_flags.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
lines.extend(["## Example Paths", ""])
for path in data.get("example_paths", [])[:80]:
    lines.append(f"- `{path}`")
if smoke:
    lines.extend(["", "## FP8 KV Smoke", "", f"```json\n{json.dumps(smoke, indent=2, sort_keys=True)}\n```"])
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

log "Quantization probe completed."

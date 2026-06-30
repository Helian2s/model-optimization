#!/usr/bin/env bash
set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-finetuning-local}"
AWS_REGION="${AWS_REGION:-us-west-2}"
INSTANCE_ID="${INSTANCE_ID:-i-0c769a18f50fd1fe6}"
ROOT_DIR="${ROOT_DIR:-/mnt/workspace/model-optimization}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
FALLBACK_MODEL="${FALLBACK_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TRTLLM_TEST_IMAGE="${TRTLLM_TEST_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.1}"
SHUTDOWN_GUARD_MINUTES="${SHUTDOWN_GUARD_MINUTES:-180}"
RUN_ID="${RUN_ID:-quant_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
S3_BASE_PREFIX="${S3_BASE_PREFIX:-s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning}"
S3_SOURCE_BUNDLE="${S3_SOURCE_BUNDLE:-$S3_BASE_PREFIX/source-bundles/model-optimization-$RUN_ID.tgz}"
S3_RESULT_PREFIX="${S3_RESULT_PREFIX:-$S3_BASE_PREFIX/results/trtllm-benchmark/$RUN_ID}"
RESULTS_DIR="${RESULTS_DIR:-results}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-14400}"

mkdir -p "$RESULTS_DIR"
LOCAL_COMMAND_JSON="$RESULTS_DIR/ssm_${RUN_ID}.json"
LOCAL_STATUS_JSON="$RESULTS_DIR/ec2_${RUN_ID}_status.json"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

aws_cli() {
  aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"
}

log "Checking AWS identity for profile=$AWS_PROFILE region=$AWS_REGION"
aws_cli sts get-caller-identity >/dev/null

bundle_path="$RESULTS_DIR/model-optimization-$RUN_ID.tgz"
log "Creating source bundle $bundle_path"
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='results' \
  --exclude='artifacts' \
  --exclude='__pycache__' \
  --exclude='.mypy_cache' \
  --exclude='.pytest_cache' \
  -czf "$bundle_path" .
log "Uploading source bundle to $S3_SOURCE_BUNDLE"
aws_cli s3 cp "$bundle_path" "$S3_SOURCE_BUNDLE" --only-show-errors

state="$(aws_cli ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)"
log "Current instance state: $state"

if [[ "$state" == "stopped" ]]; then
  log "Starting instance $INSTANCE_ID"
  aws_cli ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
  aws_cli ec2 wait instance-running --instance-ids "$INSTANCE_ID"
elif [[ "$state" != "running" ]]; then
  log "Waiting for instance to become running from state=$state"
  aws_cli ec2 wait instance-running --instance-ids "$INSTANCE_ID"
fi

log "Waiting for EC2 instance status checks."
aws_cli ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID" || true

log "Waiting for SSM managed instance registration."
deadline=$((SECONDS + 900))
while true; do
  ping_status="$(aws_cli ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || true)"
  if [[ "$ping_status" == "Online" ]]; then
    break
  fi
  if (( SECONDS > deadline )); then
    log "SSM did not become Online before timeout; stopping instance."
    aws_cli ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null || true
    exit 1
  fi
  sleep 15
done
log "SSM status: Online"

remote_script=$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
aws s3 cp '$S3_SOURCE_BUNDLE' /tmp/model-optimization-$RUN_ID.tgz --only-show-errors
rm -rf '$ROOT_DIR'
mkdir -p '$ROOT_DIR'
tar -xzf /tmp/model-optimization-$RUN_ID.tgz -C '$ROOT_DIR'
cd '$ROOT_DIR'
RUN_ID='$RUN_ID' \
MODEL='$MODEL' \
FALLBACK_MODEL='$FALLBACK_MODEL' \
TRTLLM_TEST_IMAGE='$TRTLLM_TEST_IMAGE' \
S3_RESULT_PREFIX='$S3_RESULT_PREFIX' \
SHUTDOWN_GUARD_MINUTES='$SHUTDOWN_GUARD_MINUTES' \
bash scripts/aws/run_ec2_quantization_probe.sh
EOF
)

log "Sending SSM command for run_id=$RUN_ID image=$TRTLLM_TEST_IMAGE"
parameters_file="$(mktemp)"
python3 - "$parameters_file" "$remote_script" <<'PY'
import json
import sys
from pathlib import Path

script = sys.argv[2]
payload = {
    "commands": [
        "cat > /tmp/run_trtllm_quant_probe.sh <<'SCRIPT'\n" + script + "\nSCRIPT",
        "bash /tmp/run_trtllm_quant_probe.sh",
    ]
}
Path(sys.argv[1]).write_text(json.dumps(payload), encoding="utf-8")
PY
aws_cli ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "TensorRT-LLM quantization probe $RUN_ID" \
  --parameters "file://$parameters_file" \
  --output json > "$LOCAL_COMMAND_JSON"
rm -f "$parameters_file"

command_id="$(python3 - "$LOCAL_COMMAND_JSON" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["Command"]["CommandId"])
PY
)"
log "SSM command id: $command_id"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
status="Pending"
while true; do
  invocation_json="$(aws_cli ssm get-command-invocation \
    --command-id "$command_id" \
    --instance-id "$INSTANCE_ID" \
    --output json 2>/dev/null || true)"
  if [[ -n "$invocation_json" ]]; then
    printf '%s\n' "$invocation_json" > "$LOCAL_STATUS_JSON"
    status="$(python3 - "$LOCAL_STATUS_JSON" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1])).get("Status", "Unknown"))
PY
)"
    log "SSM status: $status"
    case "$status" in
      Success|Failed|Cancelled|TimedOut|Cancelling)
        break
        ;;
    esac
  fi
  if (( SECONDS > deadline )); then
    log "Timed out waiting for SSM command; requesting EC2 stop."
    aws_cli ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null || true
    exit 1
  fi
  sleep 60
done

log "Best-effort stop request after probe completion."
aws_cli ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null || true
aws_cli ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" || true
final_state="$(aws_cli ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text 2>/dev/null || true)"
log "Final instance state: $final_state"

if [[ "$status" != "Success" ]]; then
  log "Probe did not finish successfully; inspect $LOCAL_STATUS_JSON"
  exit 1
fi

log "Probe completed successfully. Remote artifacts should be under $S3_RESULT_PREFIX"

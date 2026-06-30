#!/usr/bin/env bash
set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-finetuning-local}"
AWS_REGION="${AWS_REGION:-us-west-2}"
SOURCE_INSTANCE_ID="${SOURCE_INSTANCE_ID:-i-0c769a18f50fd1fe6}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.2xlarge}"
ROOT_VOLUME_GB="${ROOT_VOLUME_GB:-500}"
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
SSM_ONLINE_TIMEOUT_SECONDS="${SSM_ONLINE_TIMEOUT_SECONDS:-900}"
AZ_ORDER="${AZ_ORDER:-us-west-2b us-west-2a us-west-2d us-west-2c}"
DOWNLOAD_RESULTS="${DOWNLOAD_RESULTS:-1}"
ALLOW_TEMP_EC2="${ALLOW_TEMP_EC2:-0}"

mkdir -p "$RESULTS_DIR"
LOCAL_COMMAND_JSON="$RESULTS_DIR/ssm_${RUN_ID}.json"
LOCAL_STATUS_JSON="$RESULTS_DIR/ec2_${RUN_ID}_status.json"
LOCAL_LAUNCH_JSON="$RESULTS_DIR/ec2_${RUN_ID}_launch.json"
LOCAL_LAUNCH_ERRORS="$RESULTS_DIR/ec2_${RUN_ID}_launch_errors.log"

TEMP_INSTANCE_ID=""
BLOCK_DEVICE_FILE=""
NETWORK_FILE=""

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

aws_cli() {
  aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  rm -f "$BLOCK_DEVICE_FILE" "$NETWORK_FILE"
  if [[ -n "$TEMP_INSTANCE_ID" ]]; then
    log "Terminating temporary instance $TEMP_INSTANCE_ID."
    aws_cli ec2 terminate-instances --instance-ids "$TEMP_INSTANCE_ID" >/dev/null || true
    aws_cli ec2 wait instance-terminated --instance-ids "$TEMP_INSTANCE_ID" || true
    log "Temporary instance $TEMP_INSTANCE_ID terminated."
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

log "Checking AWS identity for profile=$AWS_PROFILE region=$AWS_REGION"
aws_cli sts get-caller-identity >/dev/null

if [[ "$ALLOW_TEMP_EC2" != "1" ]]; then
  log "Refusing to create a temporary EC2 instance without explicit approval."
  log "Set ALLOW_TEMP_EC2=1 only after the account owner approves a new temporary EC2 launch."
  exit 2
fi

source_instance_json="$(aws_cli ec2 describe-instances \
  --instance-ids "$SOURCE_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0]' \
  --output json)"

read_source_attr() {
  python3 - "$1" "$source_instance_json" <<'PY'
import json
import sys

key = sys.argv[1]
data = json.loads(sys.argv[2])
if key == "security_groups":
    print(" ".join(group["GroupId"] for group in data.get("SecurityGroups", [])))
else:
    value = data
    for part in key.split("."):
        value = value.get(part, {}) if isinstance(value, dict) else {}
    print(value if value not in ({}, None) else "")
PY
}

AMI_ID="${AMI_ID:-$(read_source_attr ImageId)}"
VPC_ID="${VPC_ID:-$(read_source_attr VpcId)}"
ROOT_DEVICE_NAME="${ROOT_DEVICE_NAME:-$(read_source_attr RootDeviceName)}"
IAM_PROFILE_ARN="${IAM_PROFILE_ARN:-$(read_source_attr IamInstanceProfile.Arn)}"
SECURITY_GROUP_IDS_TEXT="${SECURITY_GROUP_IDS:-$(read_source_attr security_groups)}"
read -r -a SECURITY_GROUP_IDS_ARRAY <<< "$SECURITY_GROUP_IDS_TEXT"

if [[ -z "$AMI_ID" || -z "$VPC_ID" || -z "$ROOT_DEVICE_NAME" || -z "$IAM_PROFILE_ARN" || "${#SECURITY_GROUP_IDS_ARRAY[@]}" -eq 0 ]]; then
  log "Could not derive required launch attributes from source instance $SOURCE_INSTANCE_ID."
  exit 1
fi

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

BLOCK_DEVICE_FILE="$(mktemp)"
NETWORK_FILE="$(mktemp)"

python3 - "$BLOCK_DEVICE_FILE" "$ROOT_DEVICE_NAME" "$ROOT_VOLUME_GB" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps([
        {
            "DeviceName": sys.argv[2],
            "Ebs": {
                "VolumeSize": int(sys.argv[3]),
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }
    ]),
    encoding="utf-8",
)
PY

: > "$LOCAL_LAUNCH_ERRORS"
for az in $AZ_ORDER; do
  subnet_id="$(aws_cli ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$az" \
    --query 'Subnets[?MapPublicIpOnLaunch==`true`]|[0].SubnetId' \
    --output text 2>/dev/null || true)"
  if [[ -z "$subnet_id" || "$subnet_id" == "None" ]]; then
    log "Skipping $az; no public subnet found in VPC $VPC_ID."
    continue
  fi

  python3 - "$NETWORK_FILE" "$subnet_id" "${SECURITY_GROUP_IDS_ARRAY[@]}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps([
        {
            "DeviceIndex": 0,
            "SubnetId": sys.argv[2],
            "Groups": sys.argv[3:],
            "AssociatePublicIpAddress": True,
        }
    ]),
    encoding="utf-8",
)
PY

  log "Trying temporary $INSTANCE_TYPE launch in $az subnet=$subnet_id"
  launch_stderr="$(mktemp)"
  if launch_json="$(aws_cli ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile "Arn=$IAM_PROFILE_ARN" \
    --network-interfaces "file://$NETWORK_FILE" \
    --block-device-mappings "file://$BLOCK_DEVICE_FILE" \
    --metadata-options "HttpEndpoint=enabled,HttpTokens=optional" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=trtllm-quant-probe-$RUN_ID},{Key=Project,Value=trtllm-optimization-benchmark},{Key=RunId,Value=$RUN_ID},{Key=ManagedBy,Value=codex}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=trtllm-quant-probe-$RUN_ID},{Key=Project,Value=trtllm-optimization-benchmark},{Key=RunId,Value=$RUN_ID},{Key=ManagedBy,Value=codex}]" \
    --output json 2>"$launch_stderr")"; then
    printf '%s\n' "$launch_json" > "$LOCAL_LAUNCH_JSON"
    TEMP_INSTANCE_ID="$(python3 - "$LOCAL_LAUNCH_JSON" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["Instances"][0]["InstanceId"])
PY
)"
    log "Launched temporary instance $TEMP_INSTANCE_ID in $az."
    rm -f "$launch_stderr"
    break
  fi

  {
    printf '[%s] Launch failed in %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$az"
    cat "$launch_stderr"
    printf '\n'
  } >> "$LOCAL_LAUNCH_ERRORS"
  cat "$launch_stderr" >&2
  rm -f "$launch_stderr"
done

if [[ -z "$TEMP_INSTANCE_ID" ]]; then
  log "Could not launch a temporary $INSTANCE_TYPE in AZ order: $AZ_ORDER"
  exit 1
fi

log "Waiting for temporary instance to run and pass status checks."
aws_cli ec2 wait instance-running --instance-ids "$TEMP_INSTANCE_ID"
aws_cli ec2 wait instance-status-ok --instance-ids "$TEMP_INSTANCE_ID" || true

log "Waiting for SSM managed instance registration."
deadline=$((SECONDS + SSM_ONLINE_TIMEOUT_SECONDS))
while true; do
  ping_status="$(aws_cli ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$TEMP_INSTANCE_ID" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || true)"
  if [[ "$ping_status" == "Online" ]]; then
    break
  fi
  if (( SECONDS > deadline )); then
    log "SSM did not become Online before timeout."
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
  --instance-ids "$TEMP_INSTANCE_ID" \
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
    --instance-id "$TEMP_INSTANCE_ID" \
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
    log "Timed out waiting for SSM command."
    exit 1
  fi
  sleep 60
done

if [[ "$DOWNLOAD_RESULTS" == "1" ]]; then
  log "Best-effort local download of remote S3 results."
  aws_cli s3 sync "$S3_RESULT_PREFIX" "$RESULTS_DIR/${RUN_ID}_s3" --only-show-errors || true
fi

if [[ "$status" != "Success" ]]; then
  log "Probe did not finish successfully; inspect $LOCAL_STATUS_JSON and $LOCAL_LAUNCH_ERRORS"
  exit 1
fi

log "Probe completed successfully. Remote artifacts are under $S3_RESULT_PREFIX"

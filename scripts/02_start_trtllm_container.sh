#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19"
NAME="trtllm-benchmark"
PULL=0
DETACH=0
DRY_RUN=0
EXEC_CMD=""
RUN_CMD=""
HF_TOKEN_VALUE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --pull) PULL=1; shift ;;
    --detach) DETACH=1; shift ;;
    --exec) EXEC_CMD="$2"; shift 2 ;;
    --run) RUN_CMD="$2"; shift 2 ;;
    --hf-token) HF_TOKEN_VALUE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

if [[ "$PULL" -eq 1 ]]; then
  run docker pull "$IMAGE"
fi

if [[ -z "$HF_TOKEN_VALUE" && -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_VALUE="$HF_TOKEN"
fi

container_env=(
  -e "TRTLLM_CONTAINER_IMAGE=$IMAGE"
  -e "HF_HOME=${HF_HOME:-/mnt/workspace/huggingface-cache}"
  -e "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/workspace/huggingface-cache}"
  -e "HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/workspace/huggingface-cache/hub}"
  -e "PYTHONPATH=/workspace"
)

if [[ -n "$HF_TOKEN_VALUE" ]]; then
  export HF_TOKEN="$HF_TOKEN_VALUE"
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN_VALUE"
  container_env+=(-e "HF_TOKEN" -e "HUGGINGFACE_HUB_TOKEN")
fi

container_mounts=(
  -v "$ROOT_DIR:/workspace"
)
if [[ -d /mnt/workspace ]]; then
  container_mounts+=(-v "/mnt/workspace:/mnt/workspace")
fi
if [[ -d /opt/dlami/nvme ]]; then
  container_mounts+=(-v "/opt/dlami/nvme:/opt/dlami/nvme")
fi

if [[ -n "$RUN_CMD" ]]; then
  run docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${container_mounts[@]}" \
    "${container_env[@]}" \
    -w /workspace \
    "$IMAGE" bash -lc "$RUN_CMD"
  exit 0
fi

if [[ -n "$EXEC_CMD" ]]; then
  run docker exec -it "$NAME" bash -lc "$EXEC_CMD"
  exit 0
fi

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$NAME"; then
  echo "Container $NAME is already running."
else
  docker_args=(
    run --gpus all --ipc=host
    --ulimit memlock=-1 --ulimit stack=67108864
    --name "$NAME"
    "${container_mounts[@]}"
    -w /workspace
    "${container_env[@]}"
  )
  if [[ "$DETACH" -eq 1 ]]; then
    docker_args+=(-d "$IMAGE" sleep infinity)
  else
    docker_args+=(--rm -it "$IMAGE" bash)
  fi
  run docker "${docker_args[@]}"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$ROOT_DIR/results"
  docker image inspect "$IMAGE" --format '{{json .RepoDigests}}' > "$ROOT_DIR/results/container_repo_digests.json" 2>/dev/null || true
fi

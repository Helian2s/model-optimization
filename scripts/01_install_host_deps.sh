#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

echo "Checking host dependencies for TensorRT-LLM benchmark."
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
command -v docker >/dev/null 2>&1 && docker --version || true
command -v nvidia-ctk >/dev/null 2>&1 && nvidia-ctk --version || true

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. No packages installed."
  exit 0
fi

if command -v apt-get >/dev/null 2>&1; then
  run sudo apt-get update
  run sudo apt-get install -y ca-certificates curl jq git python3-venv python3-pip docker.io
  run sudo systemctl enable --now docker
else
  echo "apt-get not found. This script currently supports Ubuntu/Debian hosts." >&2
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "NVIDIA Container Toolkit is not installed or not on PATH." >&2
  echo "Install it from NVIDIA's official documentation for the target Ubuntu release." >&2
  exit 1
fi

run sudo nvidia-ctk runtime configure --runtime=docker
run sudo systemctl restart docker
echo "Host dependency check completed."

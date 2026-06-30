"""Environment capture for local, EC2, and TensorRT-LLM container runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.gpu_monitor import gpu_snapshot_dict
from src.results_schema import ensure_dir, write_json


def run_command(command: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return {"command": command, "available": False, "returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.SubprocessError as exc:
        return {"command": command, "available": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def ec2_metadata(path: str, timeout: float = 0.35) -> str | None:
    token_request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        with urllib.request.urlopen(token_request, timeout=timeout) as response:
            token = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

    request = urllib.request.Request(
        f"http://169.254.169.254/latest/meta-data/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def disk_report(paths: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        usage = shutil.disk_usage(path)
        report[str(path)] = {
            "total_gb": round(usage.total / (1024**3), 3),
            "used_gb": round(usage.used / (1024**3), 3),
            "free_gb": round(usage.free / (1024**3), 3),
        }
    return report


def git_commit() -> str | None:
    result = run_command(["git", "rev-parse", "HEAD"], timeout=3.0)
    if result["available"] and result["returncode"] == 0 and result["stdout"]:
        return str(result["stdout"])
    return None


def docker_image_digest(image: str | None) -> str | None:
    if not image:
        return None
    result = run_command(["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"], timeout=10.0)
    if not (result["available"] and result["returncode"] == 0 and result["stdout"]):
        return None
    try:
        digests = json.loads(str(result["stdout"]))
    except json.JSONDecodeError:
        return None
    if not digests:
        return None
    return str(digests[0])


def collect_environment(container_image: str | None = None) -> dict[str, Any]:
    nvidia_smi = run_command(["nvidia-smi"], timeout=8.0)
    nvidia_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=8.0,
    )
    nvcc = run_command(["nvcc", "--version"], timeout=5.0)
    docker_version = run_command(["docker", "--version"], timeout=5.0)
    docker_info = run_command(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=5.0)
    nvidia_ctk = run_command(["nvidia-ctk", "--version"], timeout=5.0)

    env_image = (
        container_image
        or os.environ.get("TRTLLM_CONTAINER_IMAGE")
        or os.environ.get("TENSORRT_LLM_CONTAINER_IMAGE")
        or os.environ.get("NVIDIA_CONTAINER_IMAGE")
    )

    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "epoch_seconds": time.time(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "ec2": {
            "instance_type": ec2_metadata("instance-type"),
            "instance_id": ec2_metadata("instance-id"),
            "availability_zone": ec2_metadata("placement/availability-zone"),
        },
        "gpu": {
            "nvidia_smi_available": bool(nvidia_smi["available"] and nvidia_smi["returncode"] == 0),
            "nvidia_smi_query": nvidia_query,
            "snapshots": gpu_snapshot_dict(),
        },
        "nvidia": {
            "nvidia_smi": nvidia_smi,
            "nvcc_version": nvcc,
            "nvidia_container_toolkit": nvidia_ctk,
        },
        "docker": {
            "version": docker_version,
            "runtimes": docker_info,
        },
        "python_packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "tensorrt_llm": package_version("tensorrt_llm"),
            "tensorrt": package_version("tensorrt"),
            "pynvml": package_version("pynvml"),
        },
        "container": {
            "inside_container": Path("/.dockerenv").exists(),
            "image": env_image,
            "image_digest": docker_image_digest(env_image),
            "env": {
                key: os.environ.get(key)
                for key in [
                    "NVIDIA_VISIBLE_DEVICES",
                    "NVIDIA_DRIVER_CAPABILITIES",
                    "TRTLLM_CONTAINER_IMAGE",
                    "TENSORRT_LLM_CONTAINER_IMAGE",
                ]
                if os.environ.get(key) is not None
            },
        },
        "disk": {
            "paths": disk_report(["/", str(Path.cwd()), "/mnt", "/opt/dlami/nvme"]),
            "df_h": run_command(["df", "-hP"], timeout=5.0),
        },
        "git": {
            "commit": git_commit(),
            "status_short": run_command(["git", "status", "--short"], timeout=5.0),
        },
    }
    return report


def text_report(report: dict[str, Any]) -> str:
    gpu_names = [sample.get("name") for sample in report["gpu"].get("snapshots", [])]
    lines = [
        "TensorRT-LLM Benchmark Environment Report",
        f"Captured UTC: {report.get('captured_at_utc')}",
        f"Host: {report['host'].get('hostname')}",
        f"Platform: {report['host'].get('platform')}",
        f"Python: {report['host'].get('python_version')}",
        f"EC2 instance type: {report['ec2'].get('instance_type') or 'not detected'}",
        f"EC2 instance id: {report['ec2'].get('instance_id') or 'not detected'}",
        f"GPU count: {len(gpu_names)}",
        f"GPU names: {', '.join(gpu_names) if gpu_names else 'not detected'}",
        f"PyTorch version: {report['python_packages'].get('torch') or 'not installed'}",
        f"Transformers version: {report['python_packages'].get('transformers') or 'not installed'}",
        f"TensorRT-LLM version: {report['python_packages'].get('tensorrt_llm') or 'not installed'}",
        f"TensorRT version: {report['python_packages'].get('tensorrt') or 'not installed'}",
        f"Docker: {report['docker']['version'].get('stdout') or 'not available'}",
        f"NVIDIA Container Toolkit: {report['nvidia']['nvidia_container_toolkit'].get('stdout') or 'not available'}",
        f"Container image: {report['container'].get('image') or 'not detected'}",
        f"Container digest: {report['container'].get('image_digest') or 'not detected'}",
        f"Git commit: {report['git'].get('commit') or 'not detected'}",
        "",
        "Disk:",
    ]
    for path, usage in report["disk"].get("paths", {}).items():
        lines.append(
            f"  {path}: total={usage['total_gb']}GB used={usage['used_gb']}GB free={usage['free_gb']}GB"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture benchmark environment information.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--container-image", default=None)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    report = collect_environment(container_image=args.container_image)
    write_json(output_dir / "env_report.json", report)
    (output_dir / "env_report.txt").write_text(text_report(report), encoding="utf-8")
    print(f"Wrote {output_dir / 'env_report.json'}")
    print(f"Wrote {output_dir / 'env_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

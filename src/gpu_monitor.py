"""GPU telemetry helpers with NVIDIA-first behavior and CPU-only fallback."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GpuSample:
    timestamp: float
    index: int
    name: str
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    utilization_gpu_pct: float | None = None


def _run(command: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def query_nvidia_smi() -> list[GpuSample]:
    """Return a point-in-time NVIDIA GPU snapshot using nvidia-smi if available."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = _run(command)
    if result is None or result.returncode != 0:
        return []

    samples: list[GpuSample] = []
    now = time.time()
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            samples.append(
                GpuSample(
                    timestamp=now,
                    index=int(parts[0]),
                    name=parts[1],
                    memory_used_mb=float(parts[2]),
                    memory_total_mb=float(parts[3]),
                    utilization_gpu_pct=float(parts[4]),
                )
            )
        except ValueError:
            continue
    return samples


def gpu_snapshot_dict() -> list[dict[str, Any]]:
    return [sample.__dict__.copy() for sample in query_nvidia_smi()]


class GpuMonitor:
    """Lightweight background sampler used by benchmark scripts."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "GpuMonitor":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2, 1.0))
        return self.summary()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.extend(query_nvidia_smi())
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "sample_count": 0,
                "max_memory_used_mb": None,
                "max_memory_used_gb": None,
                "max_utilization_gpu_pct": None,
            }
        max_memory_mb = max(
            (sample.memory_used_mb for sample in self.samples if sample.memory_used_mb is not None),
            default=None,
        )
        max_util = max(
            (sample.utilization_gpu_pct for sample in self.samples if sample.utilization_gpu_pct is not None),
            default=None,
        )
        return {
            "sample_count": len(self.samples),
            "max_memory_used_mb": max_memory_mb,
            "max_memory_used_gb": None if max_memory_mb is None else round(max_memory_mb / 1024.0, 3),
            "max_utilization_gpu_pct": max_util,
        }

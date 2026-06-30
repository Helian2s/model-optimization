"""Shared result schemas and metric helpers for benchmark outputs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable


SUMMARY_COLUMNS = [
    "variant",
    "dataset",
    "input_len",
    "output_len",
    "num_requests",
    "avg_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "avg_ttft_ms",
    "avg_itl_ms",
    "request_throughput_rps",
    "total_tokens_per_sec",
    "generation_tokens_per_sec",
    "max_gpu_memory_gb",
    "quality_pass_rate",
    "notes",
]


@dataclass
class SummaryRow:
    """One normalized row for results/summary.csv and summary.json."""

    variant: str
    dataset: str
    input_len: int | None = None
    output_len: int | None = None
    num_requests: int | None = None
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p90_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    avg_ttft_ms: float | None = None
    avg_itl_ms: float | None = None
    request_throughput_rps: float | None = None
    total_tokens_per_sec: float | None = None
    generation_tokens_per_sec: float | None = None
    max_gpu_memory_gb: float | None = None
    quality_pass_rate: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {column: data.get(column) for column in SUMMARY_COLUMNS}

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SummaryRow":
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: data.get(key) for key in allowed if key in data})


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return records
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {file_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object in {file_path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def write_summary_csv(path: str | Path, rows: Iterable[SummaryRow]) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def mean(values: Iterable[float]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Return nearest-rank percentile for a non-empty iterable."""

    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return None
    if pct <= 0:
        return cleaned[0]
    if pct >= 100:
        return cleaned[-1]
    rank = math.ceil((pct / 100.0) * len(cleaned))
    return cleaned[max(rank - 1, 0)]


def safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def round_optional(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def latency_summary(latencies_ms: Iterable[float]) -> dict[str, float | None]:
    values = [float(value) for value in latencies_ms if value is not None]
    return {
        "avg_latency_ms": round_optional(mean(values)),
        "p50_latency_ms": round_optional(percentile(values, 50)),
        "p90_latency_ms": round_optional(percentile(values, 90)),
        "p95_latency_ms": round_optional(percentile(values, 95)),
        "p99_latency_ms": round_optional(percentile(values, 99)),
    }

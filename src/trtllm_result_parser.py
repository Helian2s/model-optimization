"""Parsers for Hugging Face summaries and TensorRT-LLM benchmark logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.results_schema import SummaryRow, read_json


DATASET_RE = re.compile(r"isl(?P<input>\d+)_osl(?P<output>\d+)")


def dataset_lengths(dataset: str) -> tuple[int | None, int | None]:
    match = DATASET_RE.search(dataset)
    if not match:
        return None, None
    return int(match.group("input")), int(match.group("output"))


def _number_after_label(text: str, labels: list[str]) -> float | None:
    for label in labels:
        pattern = re.compile(
            rf"{label}\s*(?:\([^)]*\))?\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _int_after_label(text: str, labels: list[str]) -> int | None:
    value = _number_after_label(text, labels)
    if value is None:
        return None
    return int(round(value))


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _log_has_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in [
            "traceback",
            "error during benchmarking",
            "valueerror:",
            "runtimeerror:",
            "[e] [",
        ]
    )


def _parse_trt_memory_gb(text: str) -> tuple[float | None, str | None]:
    """Estimate peak TensorRT-LLM GPU memory from runtime allocation log lines."""

    usage_matches = list(
        re.finditer(
            r"Memory usage when calculating max tokens in paged kv cache:\s*"
            r"total:\s*(?P<total>[-+]?\d+(?:\.\d+)?)\s*GiB,\s*"
            r"available:\s*(?P<available>[-+]?\d+(?:\.\d+)?)\s*GiB",
            text,
            re.IGNORECASE,
        )
    )
    kv_allocs = [
        float(match.group("allocated"))
        for match in re.finditer(
            r"Allocated\s+(?P<allocated>[-+]?\d+(?:\.\d+)?)\s+GiB\s+for max tokens in paged KV cache",
            text,
            re.IGNORECASE,
        )
    ]
    estimates: list[float] = []
    for index, usage in enumerate(usage_matches):
        total = float(usage.group("total"))
        available = float(usage.group("available"))
        allocated = kv_allocs[min(index, len(kv_allocs) - 1)] if kv_allocs else 0.0
        estimates.append((total - available) + allocated)
    if estimates:
        return max(estimates), "max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log"
    if kv_allocs:
        return max(kv_allocs), "max_gpu_memory_gb is TensorRT paged KV-cache allocation only"
    return None, None


def _summary_from_report(report_path: Path, variant: str, notes: list[str]) -> SummaryRow | None:
    data = read_json(report_path, default=None)
    if not isinstance(data, dict):
        return None

    dataset = dataset_from_filename(report_path.name)
    input_len, output_len = dataset_lengths(dataset)
    request_info = data.get("request_info") if isinstance(data.get("request_info"), dict) else {}
    performance = data.get("performance") if isinstance(data.get("performance"), dict) else {}
    streaming = data.get("streaming_metrics") if isinstance(data.get("streaming_metrics"), dict) else {}
    latency_percentiles = (
        performance.get("request_latency_percentiles_ms")
        if isinstance(performance.get("request_latency_percentiles_ms"), dict)
        else {}
    )

    if request_info.get("avg_input_length") is not None:
        input_len = round(request_info["avg_input_length"])
    if request_info.get("avg_output_length") is not None:
        output_len = round(request_info["avg_output_length"])

    return SummaryRow(
        variant=variant,
        dataset=dataset,
        input_len=input_len,
        output_len=output_len,
        num_requests=request_info.get("num_requests"),
        avg_latency_ms=performance.get("avg_request_latency_ms"),
        p50_latency_ms=latency_percentiles.get("p50"),
        p90_latency_ms=latency_percentiles.get("p90"),
        p95_latency_ms=latency_percentiles.get("p95"),
        p99_latency_ms=latency_percentiles.get("p99"),
        avg_ttft_ms=streaming.get("avg_ttft_ms"),
        avg_itl_ms=streaming.get("avg_tpot_ms"),
        request_throughput_rps=performance.get("request_throughput_req_s"),
        total_tokens_per_sec=performance.get("system_total_throughput_tok_s"),
        generation_tokens_per_sec=(
            performance.get("system_output_throughput_tok_s")
            or performance.get("output_throughput_per_gpu_tok_s")
            or streaming.get("token_output_speed_tok_s")
        ),
        max_gpu_memory_gb=_get_nested(data, "memory", "max_gpu_memory_gb"),
        notes="; ".join(notes),
    )


def parse_trtllm_log(path: str | Path, variant: str, mode: str) -> SummaryRow:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    dataset = dataset_from_filename(file_path.name)
    input_len, output_len = dataset_lengths(dataset)

    notes = []
    if _log_has_error(text):
        notes.append("log contains benchmark error")
    if mode:
        notes.append(mode)
    report_path = file_path.with_name(f"{file_path.stem}_report.json")
    if report_path.exists():
        row = _summary_from_report(report_path, variant=variant, notes=notes)
        if row is not None:
            if row.max_gpu_memory_gb is None:
                memory_gb, memory_note = _parse_trt_memory_gb(text)
                row.max_gpu_memory_gb = memory_gb
                if memory_note:
                    row.notes = "; ".join([part for part in [row.notes, memory_note] if part])
            return row

    memory_gb, memory_note = _parse_trt_memory_gb(text)
    if memory_note:
        notes.append(memory_note)
    return SummaryRow(
        variant=variant,
        dataset=dataset,
        input_len=input_len,
        output_len=output_len,
        num_requests=_int_after_label(text, ["num requests", "number of requests", "requests"]),
        avg_latency_ms=_number_after_label(text, ["avg latency", "average latency", "request latency"]),
        p50_latency_ms=_number_after_label(text, ["p50 latency", "latency p50", "50th percentile latency"]),
        p90_latency_ms=_number_after_label(text, ["p90 latency", "latency p90", "90th percentile latency"]),
        p95_latency_ms=_number_after_label(text, ["p95 latency", "latency p95", "95th percentile latency"]),
        p99_latency_ms=_number_after_label(text, ["p99 latency", "latency p99", "99th percentile latency"]),
        avg_ttft_ms=_number_after_label(text, ["avg ttft", "average ttft", "time to first token", "ttft"]),
        avg_itl_ms=_number_after_label(text, ["avg itl", "average itl", "inter token latency", "itl"]),
        request_throughput_rps=_number_after_label(text, ["request throughput", "requests/sec", "req/s"]),
        total_tokens_per_sec=_number_after_label(text, ["total token throughput", "total tokens/sec", "tokens/sec"]),
        generation_tokens_per_sec=_number_after_label(
            text,
            ["generation token throughput", "output token throughput", "generated tokens/sec"],
        ),
        max_gpu_memory_gb=memory_gb,
        notes="; ".join(notes),
    )


def dataset_from_filename(name: str) -> str:
    match = DATASET_RE.search(name)
    if match:
        return match.group(0)
    if "prompt" in name:
        return "prompt_set"
    return "unknown"


def variant_from_filename(path: str | Path) -> tuple[str, str]:
    name = Path(path).name.lower()
    mode = "latency" if "latency" in name else "throughput" if "throughput" in name else ""
    if any(token in name for token in ["fp8", "kv", "awq", "gptq", "int4", "int8", "quant"]):
        return "trtllm_optimized", mode
    return "trtllm_base", mode


def parse_hf_summary(path: str | Path) -> SummaryRow | None:
    data = read_json(path)
    if not data:
        return None
    return SummaryRow(
        variant="hf_baseline",
        dataset="prompt_set",
        input_len=round(data["avg_input_tokens"]) if data.get("avg_input_tokens") is not None else None,
        output_len=round(data["avg_output_tokens"]) if data.get("avg_output_tokens") is not None else None,
        num_requests=data.get("total_requests"),
        avg_latency_ms=data.get("avg_latency_ms"),
        p50_latency_ms=data.get("p50_latency_ms"),
        p90_latency_ms=data.get("p90_latency_ms"),
        p95_latency_ms=data.get("p95_latency_ms"),
        p99_latency_ms=data.get("p99_latency_ms"),
        request_throughput_rps=data.get("request_throughput_rps"),
        total_tokens_per_sec=data.get("output_tokens_per_sec"),
        generation_tokens_per_sec=data.get("generation_tokens_per_sec") or data.get("output_tokens_per_sec"),
        max_gpu_memory_gb=data.get("max_gpu_memory_gb"),
        notes=f"model={data.get('model', 'unknown')}; precision={data.get('precision', 'unknown')}",
    )


def parse_hf_synthetic_summary(path: str | Path) -> SummaryRow | None:
    data = read_json(path)
    if not data or data.get("variant") != "hf_synthetic":
        return None
    dataset = data.get("dataset") or dataset_from_filename(Path(path).name)
    input_len = data.get("input_len")
    output_len = data.get("output_len")
    if input_len is None and data.get("avg_input_tokens") is not None:
        input_len = round(data["avg_input_tokens"])
    if output_len is None and data.get("avg_output_tokens") is not None:
        output_len = round(data["avg_output_tokens"])
    return SummaryRow(
        variant="hf_synthetic",
        dataset=dataset,
        input_len=input_len,
        output_len=output_len,
        num_requests=data.get("total_requests"),
        avg_latency_ms=data.get("avg_latency_ms"),
        p50_latency_ms=data.get("p50_latency_ms"),
        p90_latency_ms=data.get("p90_latency_ms"),
        p95_latency_ms=data.get("p95_latency_ms"),
        p99_latency_ms=data.get("p99_latency_ms"),
        request_throughput_rps=data.get("request_throughput_rps"),
        total_tokens_per_sec=data.get("output_tokens_per_sec"),
        generation_tokens_per_sec=data.get("generation_tokens_per_sec") or data.get("output_tokens_per_sec"),
        max_gpu_memory_gb=data.get("max_gpu_memory_gb"),
        notes=f"model={data.get('model', 'unknown')}; precision={data.get('precision', 'unknown')}; length-matched synthetic HF baseline",
    )


def parse_trt_prompt_summary(path: str | Path) -> SummaryRow | None:
    data = read_json(path)
    if not data or data.get("status") == "dry_run":
        return None
    backend = data.get("backend", "unknown")
    notes = [
        f"model={data.get('model', 'unknown')}",
        f"backend={backend}",
        f"engine={data.get('engine_dir', 'unknown')}",
    ]
    if backend == "examples":
        notes.append("quality-output-only; examples backend includes per-prompt process startup")
    llmapi_error = data.get("llmapi_error_before_fallback")
    if llmapi_error:
        notes.append("LLM API fallback used")
    return SummaryRow(
        variant=data.get("variant", "trtllm_base"),
        dataset=data.get("dataset", "prompt_set"),
        input_len=round(data["avg_input_tokens"]) if data.get("avg_input_tokens") is not None else None,
        output_len=round(data["avg_output_tokens"]) if data.get("avg_output_tokens") is not None else None,
        num_requests=data.get("total_requests"),
        avg_latency_ms=data.get("avg_latency_ms"),
        p50_latency_ms=data.get("p50_latency_ms"),
        p90_latency_ms=data.get("p90_latency_ms"),
        p95_latency_ms=data.get("p95_latency_ms"),
        p99_latency_ms=data.get("p99_latency_ms"),
        request_throughput_rps=data.get("request_throughput_rps"),
        total_tokens_per_sec=data.get("output_tokens_per_sec"),
        generation_tokens_per_sec=data.get("generation_tokens_per_sec") or data.get("output_tokens_per_sec"),
        max_gpu_memory_gb=data.get("max_gpu_memory_gb"),
        notes="; ".join(notes),
    )


def quality_pass_rates(path: str | Path) -> dict[str, float | None]:
    data = read_json(path, default={})
    rates: dict[str, float | None] = {}
    for variant in data.get("variants", []):
        rates[variant.get("variant", "unknown")] = variant.get("pass_rate")
    return rates


def discover_trt_logs(results_dir: str | Path) -> list[Path]:
    root = Path(results_dir)
    if not root.exists():
        return []
    patterns = [
        "trtllm_latency_*.log",
        "trtllm_throughput_*.log",
        "trtllm_*latency*.log",
        "trtllm_*throughput*.log",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    return sorted(set(found))


def parse_all_results(results_dir: str | Path) -> list[SummaryRow]:
    root = Path(results_dir)
    rows: list[SummaryRow] = []
    hf = parse_hf_summary(root / "hf_baseline_summary.json")
    if hf:
        rows.append(hf)
    for hf_synthetic_path in sorted(root.glob("hf_synthetic_isl*_summary.json")):
        hf_synthetic = parse_hf_synthetic_summary(hf_synthetic_path)
        if hf_synthetic:
            rows.append(hf_synthetic)
    trt_prompt = parse_trt_prompt_summary(root / "trtllm_prompt_summary.json")
    if trt_prompt:
        rows.append(trt_prompt)
    for log_path in discover_trt_logs(root):
        variant, mode = variant_from_filename(log_path)
        rows.append(parse_trtllm_log(log_path, variant=variant, mode=mode))

    rates = quality_pass_rates(root / "quality_regression.json")
    if rates:
        for row in rows:
            if row.dataset == "prompt_set" and row.variant in rates:
                row.quality_pass_rate = rates[row.variant]
    return rows


def fixture_rows() -> list[SummaryRow]:
    return [
        SummaryRow(
            variant="hf_baseline",
            dataset="prompt_set",
            input_len=256,
            output_len=80,
            num_requests=50,
            avg_latency_ms=850.0,
            p50_latency_ms=820.0,
            p90_latency_ms=1100.0,
            p95_latency_ms=1250.0,
            p99_latency_ms=1400.0,
            generation_tokens_per_sec=95.0,
            max_gpu_memory_gb=8.5,
            quality_pass_rate=1.0,
            notes="fixture row for local parser validation",
        ),
        SummaryRow(
            variant="trtllm_base",
            dataset="isl1024_osl128",
            input_len=1024,
            output_len=128,
            num_requests=20,
            avg_latency_ms=320.0,
            p50_latency_ms=310.0,
            p90_latency_ms=390.0,
            p95_latency_ms=420.0,
            p99_latency_ms=460.0,
            avg_ttft_ms=45.0,
            avg_itl_ms=2.1,
            request_throughput_rps=3.1,
            total_tokens_per_sec=3600.0,
            generation_tokens_per_sec=410.0,
            max_gpu_memory_gb=6.2,
            notes="fixture row for local parser validation",
        ),
    ]

#!/usr/bin/env python3
"""Parse raw benchmark outputs and generate summary files and report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results_schema import SUMMARY_COLUMNS, SummaryRow, safe_div, write_json, write_summary_csv
from src.trtllm_result_parser import fixture_rows, parse_all_results


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: Iterable[SummaryRow]) -> str:
    columns = SUMMARY_COLUMNS
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        data = row.to_dict()
        lines.append("| " + " | ".join(_fmt(data.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def speedup_comparable_to_hf_prompt(row: SummaryRow) -> tuple[bool, str]:
    if row.dataset != "prompt_set":
        return False, "not compared to HF baseline because dataset/workload differs"
    lowered_notes = row.notes.lower()
    if "quality-output-only" in lowered_notes or "per-prompt process startup" in lowered_notes:
        return False, "same prompt records, but runner includes per-prompt example-process startup; use for quality only"
    return True, ""


def speedup_rows(rows: list[SummaryRow]) -> list[dict]:
    hf_synthetic_by_dataset = {
        row.dataset: row for row in rows if row.variant == "hf_synthetic"
    }
    speedups: list[dict] = []
    hf_prompt = next((row for row in rows if row.variant == "hf_baseline" and row.dataset == "prompt_set"), None)
    for row in rows:
        if row.variant.startswith("hf_"):
            continue
        comparable, note = speedup_comparable_to_hf_prompt(row)
        if comparable and hf_prompt:
            baseline = hf_prompt
        elif row.dataset in hf_synthetic_by_dataset:
            baseline = hf_synthetic_by_dataset[row.dataset]
            comparable = True
            note = "compared with length-matched HF synthetic baseline"
        else:
            baseline = None

        if comparable and baseline:
            latency_speedup = safe_div(hf_prompt.avg_latency_ms, row.avg_latency_ms)
            if row.dataset != "prompt_set":
                latency_speedup = safe_div(baseline.avg_latency_ms, row.avg_latency_ms)
            throughput_speedup = safe_div(row.generation_tokens_per_sec, baseline.generation_tokens_per_sec)
            memory_reduction = safe_div(baseline.max_gpu_memory_gb, row.max_gpu_memory_gb)
        else:
            latency_speedup = None
            throughput_speedup = None
            memory_reduction = None
        speedups.append(
            {
                "variant": row.variant,
                "dataset": row.dataset,
                "comparable_to_hf_prompt_set": comparable,
                "latency_speedup": latency_speedup,
                "throughput_speedup": throughput_speedup,
                "memory_reduction": memory_reduction,
                "notes": note,
            }
        )
    return speedups


def speedup_markdown(speedups: list[dict]) -> str:
    lines = [
        "| Variant | Dataset | Comparable | Latency speedup | Throughput speedup | Memory reduction | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in speedups:
        lines.append(
            f"| {item['variant']} | {item['dataset']} | {item['comparable_to_hf_prompt_set']} | "
            f"{_fmt(item['latency_speedup'])} | {_fmt(item['throughput_speedup'])} | "
            f"{_fmt(item['memory_reduction'])} | {item['notes']} |"
        )
    return "\n".join(lines)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _nested(data: dict, *keys: str) -> object:
    current: object = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_gpu(data: dict) -> dict:
    snapshots = _nested(data, "gpu", "snapshots")
    if isinstance(snapshots, list) and snapshots and isinstance(snapshots[0], dict):
        return snapshots[0]
    return {}


def _config_value(key: str) -> str:
    config_path = ROOT / "configs" / "benchmark_config.yaml"
    if not config_path.exists():
        return ""
    text = config_path.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)\s*$", text, re.MULTILINE)
    return match.group(1) if match else ""


def _first_log_match(results_dir: Path, pattern: str) -> str:
    compiled = re.compile(pattern, re.IGNORECASE)
    for log_path in sorted(results_dir.glob("trtllm_*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = compiled.search(text)
        if match:
            return match.group(1)
    return ""


def environment_markdown(results_dir: Path) -> str:
    env = _load_json(results_dir / "env_report.json")
    if not env:
        return "Environment report was not available."
    gpu = _first_gpu(env)
    nvidia_smi = str(_nested(env, "nvidia", "nvidia_smi", "stdout") or "")
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi)
    cuda_version = cuda_match.group(1) if cuda_match else ""
    toolkit_available = _nested(env, "nvidia", "nvidia_container_toolkit", "available")
    docker_version = str(_nested(env, "docker", "version", "stdout") or "").strip()
    container_image = _nested(env, "container", "image") or _config_value("image")
    container_digest = _nested(env, "container", "image_digest") or _config_value("validated_repo_digest")
    package_versions = _nested(env, "python_packages") or {}
    trtllm_version = ""
    tensorrt_version = ""
    if isinstance(package_versions, dict):
        trtllm_version = package_versions.get("tensorrt_llm") or ""
        tensorrt_version = package_versions.get("tensorrt") or ""
    trtllm_version = trtllm_version or _first_log_match(results_dir, r"TensorRT LLM version:\s*([^\s]+)")
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| Captured UTC | {_nested(env, 'captured_at_utc') or ''} |",
        f"| EC2 instance | {_nested(env, 'ec2', 'instance_type') or ''} / {_nested(env, 'ec2', 'instance_id') or ''} |",
        f"| Availability zone | {_nested(env, 'ec2', 'availability_zone') or ''} |",
        f"| GPU | {gpu.get('name', '')} x1 |",
        f"| GPU memory | {gpu.get('memory_total_mb', '')} MiB |",
        f"| NVIDIA driver | {str(_nested(env, 'gpu', 'nvidia_smi_query', 'stdout') or '').split(',')[-1].strip()} |",
        f"| CUDA version | {cuda_version} |",
        f"| Docker | {docker_version} |",
        f"| NVIDIA Container Toolkit | {toolkit_available} |",
        f"| Container image | {container_image} |",
        f"| Container digest | {container_digest} |",
        f"| PyTorch | {(package_versions.get('torch') or '') if isinstance(package_versions, dict) else ''} |",
        f"| Transformers | {(package_versions.get('transformers') or '') if isinstance(package_versions, dict) else ''} |",
        f"| TensorRT-LLM | {trtllm_version} |",
        f"| TensorRT | {tensorrt_version} |",
        f"| Git commit | {_nested(env, 'git', 'commit') or ''} |",
    ]
    return "\n".join(lines)


def model_markdown(results_dir: Path) -> str:
    model = _load_json(results_dir / "model_download.json")
    build = _load_json(results_dir / "trtllm_build_metadata.json")
    hf = _load_json(results_dir / "hf_baseline_summary.json")
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | {model.get('model') or build.get('model') or hf.get('model') or ''} |",
        f"| Snapshot path | {model.get('snapshot_path') or ''} |",
        f"| HF precision | {hf.get('precision') or ''} |",
        f"| TensorRT-LLM engine dtype | {build.get('dtype') or ''} |",
        f"| TensorRT-LLM engine dir | {build.get('engine_dir') or ''} |",
        f"| Engine status | {build.get('status') or ''} |",
    ]
    return "\n".join(lines)


def memory_markdown(rows: list[SummaryRow]) -> str:
    hf_synthetic_by_dataset = {row.dataset: row for row in rows if row.variant == "hf_synthetic"}
    hf_prompt = next((row for row in rows if row.variant == "hf_baseline" and row.dataset == "prompt_set"), None)
    lines = [
        "| Variant | Dataset | Mode/notes | Max GPU memory GB | HF/TRT memory ratio |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        baseline = hf_synthetic_by_dataset.get(row.dataset)
        if row.dataset == "prompt_set":
            baseline = hf_prompt
        ratio = None
        if baseline is not None and not row.variant.startswith("hf_"):
            ratio = safe_div(baseline.max_gpu_memory_gb, row.max_gpu_memory_gb)
        lines.append(
            f"| {row.variant} | {row.dataset} | {row.notes} | "
            f"{_fmt(row.max_gpu_memory_gb)} | {_fmt(ratio)} |"
        )
    return "\n".join(lines)


def quality_markdown(quality_path: Path) -> str:
    if not quality_path.exists():
        return "No quality regression file was found."
    data = json.loads(quality_path.read_text(encoding="utf-8"))
    variants = data.get("variants", [])
    if not variants:
        return "No quality regression variants were available."
    lines = [
        "| Variant | Total | Passed | Failed | Strict pass rate | Empty/refusal | Pass delta vs HF | New failures vs HF | Output match vs HF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in variants:
        pass_rate = variant.get("pass_rate")
        pass_rate_text = "" if pass_rate is None else f"{float(pass_rate):.3f}"
        pass_delta = variant.get("pass_delta_vs_baseline")
        pass_delta_text = "" if pass_delta is None else f"{int(pass_delta):+d}"
        new_failures = variant.get("newly_failed_vs_baseline_count")
        new_failures_text = "" if new_failures is None else str(new_failures)
        match_rate = variant.get("normalized_match_rate_vs_baseline")
        match_rate_text = "" if match_rate is None else f"{float(match_rate):.3f}"
        lines.append(
            f"| {variant.get('variant', 'unknown')} | {variant.get('total', '')} | "
            f"{variant.get('passed', '')} | {variant.get('failed', '')} | "
            f"{pass_rate_text} | {variant.get('empty_or_refusal_count', '')} | "
            f"{pass_delta_text} | {new_failures_text} | {match_rate_text} |"
        )
    return "\n".join(lines)


def quality_interpretation_lines(quality_path: Path) -> list[str]:
    data = _load_json(quality_path)
    variants = data.get("variants", [])
    if not isinstance(variants, list) or not variants:
        return ["Quality regression data was not available."]
    lines: list[str] = []
    for variant in variants:
        if not isinstance(variant, dict) or "baseline_variant" not in variant:
            continue
        new_failures = int(variant.get("newly_failed_vs_baseline_count") or 0)
        pass_delta = int(variant.get("pass_delta_vs_baseline") or 0)
        match_rate = variant.get("normalized_match_rate_vs_baseline")
        match_text = "" if match_rate is None else f" Normalized output match vs HF was {float(match_rate):.1%}."
        if new_failures == 0 and pass_delta >= 0:
            lines.append(
                f"No TensorRT-specific quality regression was detected for `{variant.get('variant')}`: "
                f"strict pass count delta vs HF was {pass_delta:+d}, with {new_failures} newly failed checks."
                f"{match_text}"
            )
        else:
            lines.append(
                f"Quality regression was detected for `{variant.get('variant')}`: "
                f"strict pass count delta vs HF was {pass_delta:+d}, with {new_failures} newly failed checks."
                f"{match_text}"
            )
    if not lines:
        lines.append("Only baseline quality data was available; no optimized variant could be compared.")
    return lines


def benchmark_interpretation(rows: list[SummaryRow], speedups: list[dict], quality_path: Path) -> list[str]:
    lines: list[str] = []
    latency_items = [
        item for item in speedups
        if item.get("latency_speedup") is not None and item["latency_speedup"] > 1.0
    ]
    throughput_items = [
        item for item in speedups
        if item.get("throughput_speedup") is not None and item["throughput_speedup"] > 1.0
    ]
    memory_items = [
        item for item in speedups
        if item.get("memory_reduction") is not None
    ]
    if latency_items:
        best_latency = max(latency_items, key=lambda item: item["latency_speedup"])
        lines.append(
            "Latency improved on length-matched measured rows. "
            f"The best latency speedup was {best_latency['latency_speedup']:.3f}x "
            f"for `{best_latency['dataset']}`."
        )
    else:
        lines.append("Latency improvement was not demonstrated by the parsed comparable rows.")
    if throughput_items:
        best_throughput = max(throughput_items, key=lambda item: item["throughput_speedup"])
        lines.append(
            "Throughput improved on length-matched measured rows. "
            f"The best generation-token throughput speedup was {best_throughput['throughput_speedup']:.3f}x "
            f"for `{best_throughput['dataset']}`."
        )
    else:
        lines.append("Throughput improvement was not demonstrated by the parsed comparable rows.")
    if memory_items:
        best_memory = max(memory_items, key=lambda item: item["memory_reduction"])
        if best_memory["memory_reduction"] and best_memory["memory_reduction"] > 1.0:
            lines.append(
                "Memory usage improved on at least one comparable row. "
                f"The best HF/TRT memory ratio was {best_memory['memory_reduction']:.3f}x."
            )
        else:
            lines.append(
                "Memory usage did not improve in the parsed comparable rows. "
                "TensorRT-LLM reserved a large paged KV-cache pool, so the HF/TRT memory ratio is below 1.0."
            )
    else:
        lines.append("Memory reduction is not claimed because comparable TensorRT memory measurements were unavailable.")
    lines.extend(quality_interpretation_lines(quality_path))
    if any(row.dataset == "prompt_set" and "quality-output-only" in row.notes for row in rows):
        lines.append(
            "The TensorRT prompt-set row is used for quality comparison only because that runner used the examples "
            "backend with per-prompt process startup; speedup claims use length-matched synthetic workloads."
        )
    return lines


def optional_file_section(title: str, path: Path, max_chars: int = 4000) -> list[str]:
    if not path.exists():
        return [f"{title}: `{path}` not found."]
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]\n"
    return [f"## {title}", "", text.rstrip(), ""]


def generate_report(rows: list[SummaryRow], results_dir: Path, fixture_mode: bool) -> str:
    env_path = results_dir / "env_report.json"
    hf_path = results_dir / "hf_baseline_summary.json"
    quality_path = results_dir / "quality_regression.json"
    speedups = speedup_rows(rows)
    lines = [
        "# TensorRT-LLM Optimization Benchmark Report",
        "",
        "## Status",
        "",
        "This report was generated from parsed local result files.",
        "",
    ]
    if fixture_mode:
        lines.extend([
            "Fixture mode was used for local parser validation. Do not interpret these rows as measured results.",
            "",
        ])
    lines.extend(
        [
            "## Inputs Found",
            "",
            f"- Environment report: `{env_path}` exists={env_path.exists()}",
            f"- HF summary: `{hf_path}` exists={hf_path.exists()}",
            f"- Quality report: `{quality_path}` exists={quality_path.exists()}",
            "",
            "## Environment",
            "",
            environment_markdown(results_dir),
            "",
            "## Model And Engine",
            "",
            model_markdown(results_dir),
            "",
            "## Summary Table",
            "",
            markdown_table(rows) if rows else "No benchmark rows were parsed.",
            "",
            "## Speedup Table",
            "",
            speedup_markdown(speedups) if speedups else "No optimized rows were available for speedup calculation.",
            "",
            "## Memory Table",
            "",
            memory_markdown(rows) if rows else "No benchmark rows were parsed.",
            "",
            "## Quality Regression",
            "",
            quality_markdown(quality_path),
            "",
            *optional_file_section("Optimization Attempt", results_dir / "trtllm_optimization_attempt.md"),
            *optional_file_section("Nsight Systems Summary", results_dir / "nsys" / "profile_summary.md"),
            "## Interpretation",
            "",
        ]
    )
    if not rows:
        lines.append("No measured benchmark results are available yet.")
    elif fixture_mode:
        lines.append("Only fixture data is present. Run the EC2 GPU phases to produce measured results.")
    else:
        comparable = [item for item in speedups if item["comparable_to_hf_prompt_set"]]
        prompt_comparable = [item for item in comparable if item["dataset"] == "prompt_set"]
        synthetic_comparable = [item for item in comparable if item["dataset"] != "prompt_set"]
        if prompt_comparable:
            lines.append("At least one optimized prompt-set row can be compared with the HF prompt baseline.")
        elif synthetic_comparable:
            lines.append(
                "Length-matched synthetic HF rows can be compared with TensorRT-LLM synthetic rows. "
                "Prompt-set TensorRT rows marked quality-only are not used for speedup claims."
            )
        else:
            lines.append(
                "No speedup is claimed against the HF baseline unless a matching prompt-set optimized row exists. "
                "Synthetic TensorRT-LLM rows are reported separately, and prompt-set rows marked quality-only are not "
                "used for speedup claims."
            )
        lines.extend(["", *benchmark_interpretation(rows, speedups, quality_path)])
    lines.extend(
        [
            "",
            "## Commands Used",
            "",
            "- `bash scripts/00_check_env.sh`",
            "- `python scripts/10_download_model.py --model <model>`",
            "- `bash scripts/20_run_hf_baseline.sh --model <model>`",
            "- `python scripts/22_run_hf_synthetic.py --model <model>`",
            "- `python scripts/30_build_trtllm_engine.py --model <model>`",
            "- `python scripts/31_run_trtllm_smoke.py`",
            "- `bash scripts/32_prepare_trtllm_datasets.sh`",
            "- `bash scripts/33_run_trtllm_latency.sh`",
            "- `bash scripts/34_run_trtllm_throughput.sh`",
            "- `bash scripts/35_run_trtllm_quantized_or_kv_cache.sh`",
            "- `python scripts/40_quality_regression.py`",
            "- `python scripts/50_parse_results.py`",
            "",
            "## Notes",
            "",
            "- EC2 must be stopped after GPU phases.",
            "- Speedups require matching dataset and output constraints.",
            "- TensorRT-LLM synthetic datasets are not directly compared to prompt-set HF baseline rows.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse benchmark results and generate summary/report files.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = fixture_rows() if args.fixtures_only else parse_all_results(results_dir)
    write_summary_csv(results_dir / "summary.csv", rows)
    summary_json = {
        "fixture_mode": args.fixtures_only,
        "rows": [row.to_dict() for row in rows],
        "speedups": speedup_rows(rows),
    }
    write_json(results_dir / "summary.json", summary_json)
    (reports_dir / "final_report.md").write_text(
        generate_report(rows, results_dir=results_dir, fixture_mode=args.fixtures_only),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "fixture_mode": args.fixtures_only}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

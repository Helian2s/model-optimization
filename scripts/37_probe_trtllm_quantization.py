#!/usr/bin/env python3
"""Probe TensorRT-LLM quantization support inside an NVIDIA container."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def command_output(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "available": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except FileNotFoundError:
        return {
            "available": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "not found",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "available": False,
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout}s\n{exc.stderr or ''}",
        }


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def find_examples() -> list[str]:
    roots = [
        Path("/app/tensorrt_llm/examples"),
        Path("/workspace/TensorRT-LLM/examples"),
        Path("/workspace/examples"),
    ]
    matches: list[str] = []
    needles = ("quant", "awq", "gptq", "fp8", "int4", "int8")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(matches) >= 200:
                break
            lowered = str(path).lower()
            if any(needle in lowered for needle in needles):
                matches.append(str(path))
    return sorted(matches)


def inspect_python_api() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        trtllm = importlib.import_module("tensorrt_llm")
        result["tensorrt_llm_import"] = True
        result["tensorrt_llm_version_attr"] = getattr(trtllm, "__version__", None)
    except Exception:
        result["tensorrt_llm_import"] = False
        result["tensorrt_llm_error"] = traceback.format_exc()
        return result

    for module_name, object_name in [
        ("tensorrt_llm", "LLM"),
        ("tensorrt_llm", "SamplingParams"),
        ("tensorrt_llm.llmapi", "KvCacheConfig"),
        ("tensorrt_llm.quantization", "QuantMode"),
    ]:
        key = f"{module_name}.{object_name}"
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, object_name)
            result[key] = {
                "available": True,
                "signature": str(inspect.signature(obj)) if callable(obj) else "",
            }
        except Exception:
            result[key] = {"available": False, "error": traceback.format_exc()}

    try:
        from tensorrt_llm.llmapi import KvCacheConfig

        result["kv_cache_config_fp8_construct"] = {
            "available": True,
            "repr": repr(KvCacheConfig(dtype="fp8")),
        }
    except Exception:
        result["kv_cache_config_fp8_construct"] = {
            "available": False,
            "error": traceback.format_exc(),
        }
    return result


def summarize_flags(help_text: str) -> dict[str, bool]:
    lowered = help_text.lower()
    return {
        "mentions_fp8": "fp8" in lowered,
        "mentions_int8": "int8" in lowered,
        "mentions_int4": "int4" in lowered,
        "mentions_awq": "awq" in lowered,
        "mentions_gptq": "gptq" in lowered,
        "mentions_kv_cache_dtype": "kv_cache_dtype" in lowered or "kv-cache-dtype" in lowered,
        "mentions_kv_cache_fraction": "kv_cache_free_gpu_mem_fraction" in lowered
        or "kv_cache_free_gpu_memory_fraction" in lowered
        or "kv_cache_percentage" in lowered,
        "mentions_quant": "quant" in lowered,
    }


def try_fp8_kv_smoke(model_path: str, max_new_tokens: int) -> dict[str, Any]:
    if not model_path:
        return {"attempted": False, "reason": "model_path not provided"}
    try:
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig

        llm = LLM(model=model_path, kv_cache_config=KvCacheConfig(dtype="fp8"))
        outputs = llm.generate(
            ["Return only the word ok."],
            SamplingParams(max_tokens=max_new_tokens, temperature=0),
        )
        return {
            "attempted": True,
            "success": True,
            "output": str(outputs),
        }
    except Exception:
        return {
            "attempted": True,
            "success": False,
            "error": traceback.format_exc(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe TensorRT-LLM quantization support.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output", default="results/quantization_probe.json")
    parser.add_argument("--try-fp8-kv", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    trtllm_build_help = command_output(["trtllm-build", "--help"])
    trtllm_bench_latency_help = command_output(["trtllm-bench", "latency", "--help"])
    trtllm_bench_throughput_help = command_output(["trtllm-bench", "throughput", "--help"])

    payload: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_path": args.model_path,
        "python": sys.version,
        "package_versions": {
            "tensorrt_llm": package_version("tensorrt_llm") or package_version("tensorrt-llm"),
            "tensorrt": package_version("tensorrt"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "modelopt": package_version("modelopt") or package_version("nvidia-modelopt"),
        },
        "python_api": inspect_python_api(),
        "example_paths": find_examples(),
        "commands": {
            "trtllm_build_help": trtllm_build_help,
            "trtllm_bench_latency_help": trtllm_bench_latency_help,
            "trtllm_bench_throughput_help": trtllm_bench_throughput_help,
        },
        "flag_summary": {
            "trtllm_build": summarize_flags(
                f"{trtllm_build_help.get('stdout', '')}\n{trtllm_build_help.get('stderr', '')}"
            ),
            "trtllm_bench_latency": summarize_flags(
                f"{trtllm_bench_latency_help.get('stdout', '')}\n"
                f"{trtllm_bench_latency_help.get('stderr', '')}"
            ),
            "trtllm_bench_throughput": summarize_flags(
                f"{trtllm_bench_throughput_help.get('stdout', '')}\n"
                f"{trtllm_bench_throughput_help.get('stderr', '')}"
            ),
        },
    }
    if args.try_fp8_kv:
        payload["fp8_kv_smoke"] = try_fp8_kv_smoke(args.model_path, args.max_new_tokens)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "try_fp8_kv": args.try_fp8_kv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

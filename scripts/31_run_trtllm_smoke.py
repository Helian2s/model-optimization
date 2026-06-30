#!/usr/bin/env python3
"""Run a minimal TensorRT-LLM smoke test using discovered or user-supplied commands."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results_schema import write_json
from src.results_schema import read_json


def find_run_script() -> Path | None:
    for root in [Path("/app/tensorrt_llm"), Path("/workspace/TensorRT-LLM"), ROOT]:
        if not root.exists():
            continue
        generic = root / "examples" / "run.py"
        if generic.exists():
            return generic
        candidates = sorted(root.glob("**/run.py"))
        for candidate in candidates:
            lowered = str(candidate).lower()
            if "qwen" in lowered or "llama" in lowered:
                return candidate
        for candidate in candidates:
            lowered = str(candidate).lower()
            if "layer_wise_benchmarks" not in lowered and "openai_triton" not in lowered:
                return candidate
    return None


def resolve_tokenizer_dir(model: str, model_download_json: str | None) -> str:
    if model_download_json:
        data = read_json(model_download_json, default={})
        snapshot_path = data.get("snapshot_path") if isinstance(data, dict) else None
        if snapshot_path and Path(snapshot_path).exists():
            return str(snapshot_path)
    direct = Path(model)
    if direct.exists():
        return str(direct)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a TensorRT-LLM engine.")
    parser.add_argument("--engine-dir", default="artifacts/trtllm_engine_bf16_or_fp16")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--model-download-json", default="results/model_download.json")
    parser.add_argument("--prompt", default="Explain TensorRT-LLM in one sentence.")
    parser.add_argument("--max-output-len", type=int, default=64)
    parser.add_argument("--output", default="results/trtllm_smoke.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--command-template", default=os.environ.get("TRTLLM_SMOKE_CMD_TEMPLATE"))
    args = parser.parse_args()

    if args.command_template:
        rendered = args.command_template.format(
            engine_dir=args.engine_dir,
            model=args.model,
            tokenizer_dir=args.tokenizer_dir or resolve_tokenizer_dir(args.model, args.model_download_json),
            prompt=args.prompt,
            max_output_len=args.max_output_len,
        )
        command = shlex.split(rendered)
    else:
        trtllm_serve = shutil.which("trtllm-serve")
        run_script = find_run_script()
        if run_script:
            tokenizer_dir = args.tokenizer_dir or resolve_tokenizer_dir(args.model, args.model_download_json)
            command = [
                sys.executable,
                str(run_script),
                "--engine_dir",
                args.engine_dir,
                "--tokenizer_dir",
                tokenizer_dir,
                "--input_text",
                args.prompt,
                "--max_output_len",
                str(args.max_output_len),
                "--temperature",
                "0.0",
                "--top_k",
                "1",
            ]
        elif trtllm_serve:
            command = [trtllm_serve, "--help"]
        else:
            result = {
                "status": "dry_run_no_runner_found" if args.dry_run else "no_runner_found",
                "notes": "Run inside the TensorRT-LLM container or provide TRTLLM_SMOKE_CMD_TEMPLATE.",
            }
            write_json(args.output, result)
            if args.dry_run:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            raise SystemExit(result["notes"])

    print("+ " + " ".join(command))
    if args.dry_run:
        result = {"status": "dry_run", "command": command}
    else:
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        result = {
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "command": command,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode != 0:
            write_json(args.output, result)
            raise SystemExit(proc.returncode)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

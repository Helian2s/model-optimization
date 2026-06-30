#!/usr/bin/env python3
"""Build a TensorRT-LLM engine with version-aware command discovery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results_schema import write_json
from src.results_schema import read_json


def run(command: list[str], dry_run: bool = False, cwd: Path | None = None) -> dict:
    record = {"command": command, "dry_run": dry_run, "returncode": None, "stdout": "", "stderr": ""}
    print("+ " + " ".join(command))
    if dry_run:
        return record
    result = subprocess.run(command, text=True, capture_output=True, cwd=cwd, check=False)
    record.update({"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with {result.returncode}: {' '.join(command)}\n{result.stderr}")
    return record


def find_converter(model: str) -> Path | None:
    search_roots = [Path("/app/tensorrt_llm"), Path("/workspace/TensorRT-LLM"), ROOT]
    model_lower = model.lower()
    preferred_tokens = ["qwen"] if "qwen" in model_lower else ["llama", "tinyllama"]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(root.glob("**/convert_checkpoint.py"))
    for token in preferred_tokens:
        for candidate in candidates:
            if token in str(candidate).lower():
                return candidate
    return candidates[0] if candidates else None


def build_commands(args: argparse.Namespace) -> tuple[list[list[str]], dict]:
    engine_dir = Path(args.engine_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    model_dir = args.model_dir or resolve_model_dir(args.model, args.model_download_json)
    metadata = {
        "model": args.model,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "engine_dir": str(engine_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "builder": None,
        "converter": None,
    }

    commands: list[list[str]] = []
    if args.command_template:
        commands.append(
            args.command_template.format(
                model=args.model,
                model_dir=model_dir,
                engine_dir=str(engine_dir),
                checkpoint_dir=str(checkpoint_dir),
                dtype=args.dtype,
                max_batch_size=args.max_batch_size,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
            ).split()
        )
        metadata["builder"] = "custom_template"
        return commands, metadata

    trtllm_build = shutil.which("trtllm-build")
    converter = find_converter(args.model)
    if converter:
        metadata["converter"] = str(converter)
        commands.append(
            [
                sys.executable,
                str(converter),
                "--model_dir",
                model_dir,
                "--output_dir",
                str(checkpoint_dir),
                "--dtype",
                args.dtype,
            ]
        )
    if trtllm_build:
        metadata["builder"] = trtllm_build
        if converter:
            commands.append(
                [
                    trtllm_build,
                    "--checkpoint_dir",
                    str(checkpoint_dir),
                    "--output_dir",
                    str(engine_dir),
                    "--max_batch_size",
                    str(args.max_batch_size),
                    "--max_input_len",
                    str(args.max_input_len),
                    "--max_seq_len",
                    str(args.max_input_len + args.max_output_len),
                    "--gemm_plugin",
                    args.dtype,
                ]
            )
        else:
            commands.append(
                [
                    trtllm_build,
                    "--model_dir",
                    model_dir,
                    "--output_dir",
                    str(engine_dir),
                    "--max_batch_size",
                    str(args.max_batch_size),
                    "--max_input_len",
                    str(args.max_input_len),
                    "--max_seq_len",
                    str(args.max_input_len + args.max_output_len),
                    "--gemm_plugin",
                    args.dtype,
                ]
            )
    return commands, metadata


def resolve_model_dir(model: str, model_download_json: str | None) -> str:
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
    parser = argparse.ArgumentParser(description="Build or reuse a TensorRT-LLM engine.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-dir", default=None, help="Local model snapshot path. Defaults to results/model_download.json snapshot_path when present.")
    parser.add_argument("--model-download-json", default="results/model_download.json")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "bf16", "fp16"])
    parser.add_argument("--engine-dir", default="artifacts/trtllm_engine_bf16_or_fp16")
    parser.add_argument("--checkpoint-dir", default="artifacts/trtllm_checkpoint_bf16_or_fp16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--max-output-len", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--command-template", default=os.environ.get("TRTLLM_BUILD_CMD_TEMPLATE"))
    parser.add_argument("--metadata-out", default="results/trtllm_build_metadata.json")
    args = parser.parse_args()

    engine_dir = Path(args.engine_dir)
    if engine_dir.exists() and any(engine_dir.iterdir()) and not args.force:
        metadata = {"status": "reused_existing_engine", "engine_dir": str(engine_dir), "model": args.model}
        write_json(args.metadata_out, metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    if args.dtype == "bf16":
        args.dtype = "bfloat16"
    if args.dtype == "fp16":
        args.dtype = "float16"

    engine_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    commands, metadata = build_commands(args)
    if not commands:
        metadata["status"] = "no_builder_found"
        metadata["notes"] = "Run inside the NVIDIA TensorRT-LLM container or provide TRTLLM_BUILD_CMD_TEMPLATE."
        write_json(args.metadata_out, metadata)
        if args.dry_run:
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 0
        raise SystemExit(metadata["notes"])

    command_records = []
    for command in commands:
        command_records.append(run(command, dry_run=args.dry_run, cwd=ROOT))
    metadata["status"] = "dry_run" if args.dry_run else "built"
    metadata["commands"] = command_records
    write_json(args.metadata_out, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

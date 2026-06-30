#!/usr/bin/env python3
"""Download model artifacts from Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hf_benchmark import resolve_secret
from src.results_schema import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a model snapshot from Hugging Face Hub.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME") or "artifacts/hf_cache")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        result = {
            "model": args.model,
            "revision": args.revision,
            "cache_dir": args.cache_dir,
            "local_dir": args.local_dir,
            "dry_run": True,
        }
        write_json(output_dir / "model_download.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to download models. Install requirements first.") from exc

    token = resolve_secret(
        ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
        ssm_name="/finetuning/huggingface/token",
        region=args.region,
    )
    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        token=token,
    )
    result = {
        "model": args.model,
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "local_dir": args.local_dir,
        "snapshot_path": path,
        "dry_run": False,
    }
    write_json(output_dir / "model_download.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

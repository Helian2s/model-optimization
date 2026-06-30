#!/usr/bin/env python3
"""Run the fixed prompt set through a TensorRT-LLM engine.

The output schema intentionally mirrors the Hugging Face baseline raw file so
quality checks and prompt-set speedup parsing can compare matching workloads.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gpu_monitor import GpuMonitor
from src.hf_benchmark import format_prompt
from src.prompt_loader import PromptRecord, load_prompts
from src.results_schema import latency_summary, mean, read_json, safe_div, write_json, write_jsonl


OUTPUT_RE = re.compile(r'Output \[Text \d+ Beam 0\]: "(?P<text>.*)"\s*$', re.DOTALL)


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


def find_run_script() -> Path | None:
    for root in [Path("/app/tensorrt_llm"), Path("/workspace/TensorRT-LLM"), ROOT]:
        generic = root / "examples" / "run.py"
        if generic.exists():
            return generic
    return None


def import_tokenizer(tokenizer_dir: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Transformers is required for prompt formatting and token counting.") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def token_count(tokenizer: Any, text: str) -> int:
    return int(len(tokenizer.encode(text, add_special_tokens=False)))


def maybe_synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def validate_lengths(
    *,
    record: PromptRecord,
    input_tokens: int,
    max_input_len: int,
    max_seq_len: int,
    skip_too_long: bool,
) -> bool:
    if input_tokens <= max_input_len and input_tokens + record.max_new_tokens <= max_seq_len:
        return True
    message = (
        f"{record.id} exceeds engine limits: input_tokens={input_tokens}, "
        f"max_new_tokens={record.max_new_tokens}, max_input_len={max_input_len}, max_seq_len={max_seq_len}"
    )
    if skip_too_long:
        print(f"Skipping: {message}", file=sys.stderr)
        return False
    raise ValueError(message)


def extract_llm_text(result: Any) -> str:
    item = result
    if isinstance(item, (list, tuple)):
        if not item:
            return ""
        item = item[0]
    outputs = getattr(item, "outputs", None)
    if outputs:
        first = outputs[0]
        for attr in ["text", "output_text", "generated_text"]:
            value = getattr(first, attr, None)
            if value is not None:
                return str(value)
    for attr in ["text", "output_text", "generated_text", "output"]:
        value = getattr(item, attr, None)
        if value is not None:
            return str(value)
    if isinstance(item, dict):
        for key in ["text", "output_text", "generated_text", "output"]:
            if key in item:
                return str(item[key])
    return str(item)


def make_sampling_params(max_tokens: int) -> Any:
    from tensorrt_llm import SamplingParams

    candidates = [
        {"max_tokens": max_tokens, "temperature": 0.0, "top_k": 1},
        {"max_tokens": max_tokens, "temperature": 0.0},
        {"max_tokens": max_tokens},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return SamplingParams(**kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to construct TensorRT-LLM SamplingParams: {last_error}")


def create_llm(*, model: str, engine_dir: str, tokenizer_dir: str, trust_remote_code: bool) -> tuple[Any, str]:
    from tensorrt_llm import LLM

    candidates = [
        ("llmapi_engine_backend", {"model": engine_dir, "tokenizer": tokenizer_dir, "trust_remote_code": trust_remote_code, "backend": "tensorrt"}),
        ("llmapi_engine", {"model": engine_dir, "tokenizer": tokenizer_dir, "trust_remote_code": trust_remote_code}),
        (
            "llmapi_model_with_engine",
            {
                "model": model,
                "tokenizer": tokenizer_dir,
                "trust_remote_code": trust_remote_code,
                "backend": "tensorrt",
                "engine_dir": engine_dir,
            },
        ),
    ]
    errors: list[str] = []
    for label, kwargs in candidates:
        try:
            return LLM(**kwargs), label
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError("Unable to initialize TensorRT-LLM LLM API:\n" + "\n".join(errors))


def run_llmapi_record(llm: Any, record: PromptRecord, prompt_text: str) -> tuple[str, float]:
    params = make_sampling_params(record.max_new_tokens)
    maybe_synchronize_cuda()
    start = time.perf_counter()
    output = llm.generate([prompt_text], sampling_params=params)
    if not isinstance(output, list):
        output = list(output)
    maybe_synchronize_cuda()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return extract_llm_text(output), latency_ms


def parse_examples_output(stdout: str) -> str:
    match = OUTPUT_RE.search(stdout)
    if not match:
        return ""
    return match.group("text")


def run_examples_record(
    *,
    run_script: Path,
    engine_dir: str,
    tokenizer_dir: str,
    record: PromptRecord,
    prompt_text: str,
    timeout_seconds: int,
) -> tuple[str, float, dict[str, Any]]:
    command = [
        sys.executable,
        str(run_script),
        "--engine_dir",
        engine_dir,
        "--tokenizer_dir",
        tokenizer_dir,
        "--input_text",
        prompt_text,
        "--max_output_len",
        str(record.max_new_tokens),
        "--no_prompt_template",
        "--temperature",
        "0.0",
        "--top_k",
        "1",
    ]
    maybe_synchronize_cuda()
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        raise RuntimeError(
            f"examples/run.py timed out for {record.id} after {latency_ms:.1f} ms "
            f"(timeout_seconds={timeout_seconds})"
        ) from exc
    maybe_synchronize_cuda()
    latency_ms = (time.perf_counter() - start) * 1000.0
    details = {
        "command": command,
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
    }
    if proc.returncode != 0:
        details["stdout_tail"] = proc.stdout[-2000:]
        raise RuntimeError(f"examples/run.py failed for {record.id}: {details}")
    return parse_examples_output(proc.stdout), latency_ms, details


def read_existing_raw(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def build_record(
    *,
    record: PromptRecord,
    prompt_text: str,
    input_tokens: int,
    output_text: str,
    output_tokens: int,
    latency_ms: float,
    backend_used: str,
) -> dict[str, Any]:
    latency_s = latency_ms / 1000.0
    return {
        "id": record.id,
        "category": record.category,
        "prompt": record.prompt,
        "formatted_prompt": prompt_text,
        "max_new_tokens": record.max_new_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "generation_tokens_per_sec": safe_div(output_tokens, latency_s),
        "output_text": output_text,
        "expected_contains": record.expected_contains,
        "expected_exact": record.expected_exact,
        "backend": backend_used,
    }


def run_prompt_set(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    tokenizer_dir = args.tokenizer_dir or resolve_tokenizer_dir(args.model, args.model_download_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        result = {
            "variant": "trtllm_base",
            "dataset": "prompt_set",
            "status": "dry_run",
            "model": args.model,
            "engine_dir": args.engine_dir,
            "tokenizer_dir": tokenizer_dir,
            "prompt_count": len(prompts),
            "backend": args.backend,
            "thinking_mode": args.thinking_mode,
        }
        write_json(output_dir / "trtllm_prompt_summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    tokenizer = import_tokenizer(tokenizer_dir, trust_remote_code=not args.no_trust_remote_code)
    raw_path = output_dir / "trtllm_outputs_raw.jsonl"
    if args.resume:
        raw_records = read_existing_raw(raw_path)
        completed_ids = {str(record.get("id")) for record in raw_records}
    else:
        raw_records = []
        completed_ids = set()
        raw_path.unlink(missing_ok=True)

    formatted: list[tuple[PromptRecord, str, int]] = []
    for record in prompts:
        if record.id in completed_ids:
            continue
        prompt_text = format_prompt(tokenizer, record, args.chat_template, thinking_mode=args.thinking_mode)
        input_tokens = token_count(tokenizer, prompt_text)
        if validate_lengths(
            record=record,
            input_tokens=input_tokens,
            max_input_len=args.max_input_len,
            max_seq_len=args.max_seq_len,
            skip_too_long=args.skip_too_long,
        ):
            formatted.append((record, prompt_text, input_tokens))

    llm: Any | None = None
    backend_used = args.backend
    llm_error: str | None = None
    if args.backend in {"auto", "llmapi"}:
        try:
            llm, backend_used = create_llm(
                model=args.model,
                engine_dir=args.engine_dir,
                tokenizer_dir=tokenizer_dir,
                trust_remote_code=not args.no_trust_remote_code,
            )
        except Exception as exc:
            llm_error = str(exc)
            if args.backend == "llmapi":
                raise
            backend_used = "examples"

    run_script = find_run_script()
    if backend_used == "examples" and run_script is None:
        raise SystemExit("Could not find TensorRT-LLM examples/run.py for fallback prompt runner.")

    warmup_records = formatted[: max(args.warmup_prompts, 0)]
    try:
        for record, prompt_text, _input_tokens in warmup_records:
            if llm is not None and backend_used.startswith("llmapi"):
                run_llmapi_record(llm, record, prompt_text)
            else:
                assert run_script is not None
                run_examples_record(
                    run_script=run_script,
                    engine_dir=args.engine_dir,
                    tokenizer_dir=tokenizer_dir,
                    record=record,
                    prompt_text=prompt_text,
                    timeout_seconds=args.examples_timeout_seconds,
                )
    except Exception as exc:
        if args.backend == "llmapi":
            raise
        llm_error = f"{llm_error or ''}\nLLM API warmup failed: {exc}".strip()
        llm = None
        backend_used = "examples"
        if run_script is None:
            raise SystemExit("LLM API failed and examples/run.py fallback was not found.") from exc
        for record, prompt_text, _input_tokens in warmup_records:
            run_examples_record(
                run_script=run_script,
                engine_dir=args.engine_dir,
                tokenizer_dir=tokenizer_dir,
                record=record,
                prompt_text=prompt_text,
                timeout_seconds=args.examples_timeout_seconds,
            )

    monitor = GpuMonitor(interval_seconds=0.25).start()
    total_start = time.perf_counter()
    for record, prompt_text, input_tokens in formatted:
        if llm is not None and backend_used.startswith("llmapi"):
            output_text, latency_ms = run_llmapi_record(llm, record, prompt_text)
        else:
            assert run_script is not None
            output_text, latency_ms, _details = run_examples_record(
                run_script=run_script,
                engine_dir=args.engine_dir,
                tokenizer_dir=tokenizer_dir,
                record=record,
                prompt_text=prompt_text,
                timeout_seconds=args.examples_timeout_seconds,
            )
        output_tokens = token_count(tokenizer, output_text)
        result_record = build_record(
            record=record,
            prompt_text=prompt_text,
            input_tokens=input_tokens,
            output_text=output_text,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            backend_used=backend_used,
        )
        raw_records.append(result_record)
        append_jsonl(raw_path, result_record)
    total_seconds = time.perf_counter() - total_start
    monitor_summary = monitor.stop()

    latencies = [record["latency_ms"] for record in raw_records]
    total_input_tokens = sum(int(record["input_tokens"]) for record in raw_records)
    total_output_tokens = sum(int(record["output_tokens"]) for record in raw_records)
    summary = {
        "variant": "trtllm_base",
        "dataset": "prompt_set",
        "status": "ok",
        "model": args.model,
        "engine_dir": args.engine_dir,
        "tokenizer_dir": tokenizer_dir,
        "prompt_file": args.prompts,
        "backend": backend_used,
        "llmapi_error_before_fallback": llm_error,
        "total_requests": len(raw_records),
        "warmup_requests": len(warmup_records),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput_rps": safe_div(len(raw_records), total_seconds),
        "output_tokens_per_sec": safe_div(total_output_tokens, total_seconds),
        "generation_tokens_per_sec": safe_div(total_output_tokens, total_seconds),
        "max_gpu_memory_gb": monitor_summary.get("max_memory_used_gb"),
        "gpu_monitor": monitor_summary,
        "deterministic_settings": {
            "chat_template": args.chat_template,
            "thinking_mode": args.thinking_mode,
            "do_sample": False,
            "temperature": 0.0,
        },
        **latency_summary(latencies),
        "avg_input_tokens": mean(record["input_tokens"] for record in raw_records),
        "avg_output_tokens": mean(record["output_tokens"] for record in raw_records),
        "elapsed_seconds": total_seconds,
    }
    write_jsonl(raw_path, raw_records)
    write_json(output_dir / "trtllm_prompt_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TensorRT-LLM on the fixed benchmark prompt set.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--engine-dir", default="artifacts/trtllm_engine_bf16_or_fp16")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--model-download-json", default="results/model_download.json")
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--backend", choices=["auto", "llmapi", "examples"], default="auto")
    parser.add_argument("--chat-template", choices=["auto", "none"], default="auto")
    parser.add_argument("--thinking-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--warmup-prompts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--skip-too-long", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip prompt IDs already present in trtllm_outputs_raw.jsonl.")
    parser.add_argument(
        "--examples-timeout-seconds",
        type=int,
        default=300,
        help="Per-prompt timeout for the examples/run.py fallback. Use 0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not shutil.which("nvidia-smi"):
        print("Warning: nvidia-smi not found; GPU memory monitoring may be empty.", file=sys.stderr)
    run_prompt_set(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

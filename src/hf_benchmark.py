"""Hugging Face Transformers baseline benchmark implementation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from src.gpu_monitor import GpuMonitor
from src.prompt_loader import PromptRecord, load_prompts
from src.results_schema import latency_summary, mean, safe_div, write_json, write_jsonl


def resolve_secret(env_names: list[str], ssm_name: str | None = None, region: str = "us-west-2") -> str | None:
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value
    if not ssm_name:
        return None
    try:
        result = subprocess.run(
            [
                "aws",
                "ssm",
                "get-parameter",
                "--name",
                ssm_name,
                "--with-decryption",
                "--region",
                region,
                "--query",
                "Parameter.Value",
                "--output",
                "text",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def import_hf_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and Transformers are required for HF baseline. "
            "Install requirements on the GPU instance before running this script."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def choose_dtype(torch: Any) -> Any:
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def model_device(model: Any, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def should_disable_thinking(tokenizer: Any, thinking_mode: str) -> bool:
    if thinking_mode == "off":
        return True
    if thinking_mode == "on":
        return False
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    tokenizer_class = tokenizer.__class__.__name__.lower()
    return "qwen3" in tokenizer_name or "qwen3" in tokenizer_class


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, thinking_mode: str) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if should_disable_thinking(tokenizer, thinking_mode):
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def format_prompt(tokenizer: Any, record: PromptRecord, mode: str, thinking_mode: str = "auto") -> str:
    if mode == "none":
        return record.prompt
    if mode == "auto" and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": record.prompt}]
        return apply_chat_template(tokenizer, messages, thinking_mode=thinking_mode)
    return record.prompt


def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    record: PromptRecord,
    chat_template: str,
    thinking_mode: str,
) -> dict[str, Any]:
    prompt_text = format_prompt(tokenizer, record, chat_template, thinking_mode=thinking_mode)
    device = model_device(model, torch)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    generation_kwargs = {
        **inputs,
        "max_new_tokens": record.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    try:
        output_ids = model.generate(**generation_kwargs)
    except Exception:
        generation_kwargs.pop("temperature", None)
        output_ids = model.generate(**generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    generated_ids = output_ids[0][input_tokens:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_tokens = int(generated_ids.numel())
    latency_s = end - start
    latency_ms = latency_s * 1000.0
    return {
        "id": record.id,
        "category": record.category,
        "prompt": record.prompt,
        "max_new_tokens": record.max_new_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "generation_tokens_per_sec": safe_div(output_tokens, latency_s),
        "output_text": output_text,
        "expected_contains": record.expected_contains,
        "expected_exact": record.expected_exact,
    }


def run_baseline(
    *,
    model_name: str,
    prompts_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | None = None,
    trust_remote_code: bool = True,
    warmup_prompts: int = 3,
    chat_template: str = "auto",
    thinking_mode: str = "auto",
    region: str = "us-west-2",
) -> dict[str, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer = import_hf_stack()
    prompts = load_prompts(prompts_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token = resolve_secret(
        ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
        ssm_name="/finetuning/huggingface/token",
        region=region,
    )
    dtype = choose_dtype(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model.eval()

    warmup_records = prompts[: max(warmup_prompts, 0)]
    with torch.inference_mode():
        for record in warmup_records:
            _ = generate_one(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                record=record,
                chat_template=chat_template,
                thinking_mode=thinking_mode,
            )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    monitor = GpuMonitor(interval_seconds=0.25).start()
    raw_records: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    with torch.inference_mode():
        for record in prompts:
            raw_records.append(
                generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    torch=torch,
                    record=record,
                    chat_template=chat_template,
                    thinking_mode=thinking_mode,
                )
            )
    total_end = time.perf_counter()
    monitor_summary = monitor.stop()

    cuda_peak_gb = None
    if torch.cuda.is_available():
        cuda_peak_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
    max_gpu_memory_gb = cuda_peak_gb or monitor_summary.get("max_memory_used_gb")

    latencies = [record["latency_ms"] for record in raw_records]
    total_output_tokens = sum(int(record["output_tokens"]) for record in raw_records)
    total_input_tokens = sum(int(record["input_tokens"]) for record in raw_records)
    total_seconds = total_end - total_start
    summary = {
        "variant": "hf_baseline",
        "model": model_name,
        "precision": "bf16" if dtype is torch.bfloat16 else "fp16",
        "prompt_file": str(prompts_path),
        "total_requests": len(raw_records),
        "warmup_requests": len(warmup_records),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_sec": safe_div(total_output_tokens, total_seconds),
        "generation_tokens_per_sec": safe_div(total_output_tokens, total_seconds),
        "request_throughput_rps": safe_div(len(raw_records), total_seconds),
        "max_gpu_memory_gb": max_gpu_memory_gb,
        "cuda_max_memory_allocated_gb": cuda_peak_gb,
        "gpu_monitor": monitor_summary,
        "deterministic_settings": {
            "do_sample": False,
            "temperature": 0.0,
            "chat_template": chat_template,
            "thinking_mode": thinking_mode,
        },
        **latency_summary(latencies),
        "avg_input_tokens": mean(record["input_tokens"] for record in raw_records),
        "avg_output_tokens": mean(record["output_tokens"] for record in raw_records),
        "elapsed_seconds": total_seconds,
    }
    write_jsonl(output_path / "hf_baseline_raw.jsonl", raw_records)
    write_json(output_path / "hf_baseline_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hugging Face Transformers GPU baseline.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE"))
    parser.add_argument("--warmup-prompts", type=int, default=3)
    parser.add_argument("--chat-template", choices=["auto", "none"], default="auto")
    parser.add_argument("--thinking-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    summary = run_baseline(
        model_name=args.model,
        prompts_path=args.prompts,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        trust_remote_code=not args.no_trust_remote_code,
        warmup_prompts=args.warmup_prompts,
        chat_template=args.chat_template,
        thinking_mode=args.thinking_mode,
        region=args.region,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

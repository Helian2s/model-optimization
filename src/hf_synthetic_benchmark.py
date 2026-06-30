"""Length-matched Hugging Face benchmark for TensorRT-LLM synthetic regimes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.gpu_monitor import GpuMonitor
from src.hf_benchmark import choose_dtype, import_hf_stack, resolve_secret
from src.results_schema import latency_summary, mean, safe_div, write_json, write_jsonl


@dataclass(frozen=True)
class SyntheticSpec:
    name: str
    input_len: int
    output_len: int


DEFAULT_SPECS = [
    SyntheticSpec("isl128_osl128", 128, 128),
    SyntheticSpec("isl512_osl128", 512, 128),
    SyntheticSpec("isl1024_osl128", 1024, 128),
    SyntheticSpec("isl2048_osl256", 2048, 256),
]


def parse_dataset_names(values: list[str]) -> list[SyntheticSpec]:
    selected = []
    by_name = {spec.name: spec for spec in DEFAULT_SPECS}
    for value in values:
        if value not in by_name:
            raise ValueError(f"Unknown dataset {value!r}. Expected one of: {', '.join(by_name)}")
        selected.append(by_name[value])
    return selected


def representative_token_id(tokenizer: Any) -> int:
    candidates = [" benchmark", " token", " inference", " model"]
    special_ids = {
        value
        for value in [
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
        ]
        if value is not None
    }
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for token_id in ids:
            if token_id not in special_ids:
                return int(token_id)
    return 100


def make_input_ids(tokenizer: Any, torch: Any, input_len: int, request_index: int, device: Any) -> Any:
    base_id = representative_token_id(tokenizer)
    separator_id = tokenizer.encode(str(request_index % 10), add_special_tokens=False)
    separator_id = int(separator_id[0]) if separator_id else base_id
    token_ids = [base_id] * input_len
    if input_len > 1:
        token_ids[-1] = separator_id
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def model_device(model: Any, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_synthetic_one(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    spec: SyntheticSpec,
    request_index: int,
) -> dict[str, Any]:
    device = model_device(model, torch)
    input_ids = make_input_ids(tokenizer, torch, spec.input_len, request_index, device)
    attention_mask = torch.ones_like(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": spec.output_len,
        "min_new_tokens": spec.output_len,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    output_ids = model.generate(**generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - start

    generated_ids = output_ids[0][spec.input_len :]
    output_text = tokenizer.decode(generated_ids[: min(32, generated_ids.numel())], skip_special_tokens=True)
    output_tokens = int(generated_ids.numel())
    return {
        "id": f"{spec.name}_{request_index:04d}",
        "dataset": spec.name,
        "input_tokens": spec.input_len,
        "output_tokens": output_tokens,
        "target_output_tokens": spec.output_len,
        "latency_ms": latency_s * 1000.0,
        "generation_tokens_per_sec": safe_div(output_tokens, latency_s),
        "output_preview": output_text,
    }


def summarize_dataset(
    *,
    model_name: str,
    precision: str,
    spec: SyntheticSpec,
    raw_records: list[dict[str, Any]],
    total_seconds: float,
    monitor_summary: dict[str, Any],
    cuda_peak_gb: float | None,
    warmup_requests: int,
) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in raw_records]
    total_output_tokens = sum(int(record["output_tokens"]) for record in raw_records)
    total_input_tokens = sum(int(record["input_tokens"]) for record in raw_records)
    return {
        "variant": "hf_synthetic",
        "dataset": spec.name,
        "model": model_name,
        "precision": precision,
        "input_len": spec.input_len,
        "output_len": spec.output_len,
        "total_requests": len(raw_records),
        "warmup_requests": warmup_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_sec": safe_div(total_output_tokens, total_seconds),
        "generation_tokens_per_sec": safe_div(total_output_tokens, total_seconds),
        "request_throughput_rps": safe_div(len(raw_records), total_seconds),
        "max_gpu_memory_gb": cuda_peak_gb or monitor_summary.get("max_memory_used_gb"),
        "cuda_max_memory_allocated_gb": cuda_peak_gb,
        "gpu_monitor": monitor_summary,
        "deterministic_settings": {
            "do_sample": False,
            "min_new_tokens_equals_max_new_tokens": True,
        },
        **latency_summary(latencies),
        "avg_input_tokens": mean(record["input_tokens"] for record in raw_records),
        "avg_output_tokens": mean(record["output_tokens"] for record in raw_records),
        "elapsed_seconds": total_seconds,
        "notes": "Synthetic length-matched HF baseline; prompts are repeated token IDs generated locally.",
    }


def run_hf_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = parse_dataset_names(args.datasets)
    if args.dry_run:
        result = {
            "variant": "hf_synthetic",
            "status": "dry_run",
            "model": args.model,
            "requests": args.requests,
            "warmup_requests": args.warmup_requests,
            "datasets": [spec.__dict__ for spec in specs],
        }
        write_json(output_dir / "hf_synthetic_dry_run.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    torch, AutoModelForCausalLM, AutoTokenizer = import_hf_stack()

    token = resolve_secret(
        ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
        ssm_name="/finetuning/huggingface/token",
        region=args.region,
    )
    dtype = choose_dtype(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=not args.no_trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=not args.no_trust_remote_code,
        token=token,
    )
    model.eval()

    aggregate: dict[str, Any] = {
        "variant": "hf_synthetic",
        "model": args.model,
        "precision": "bf16" if dtype is torch.bfloat16 else "fp16",
        "datasets": {},
    }

    with torch.inference_mode():
        for spec in specs:
            for warmup_index in range(max(args.warmup_requests, 0)):
                generate_synthetic_one(
                    model=model,
                    tokenizer=tokenizer,
                    torch=torch,
                    spec=spec,
                    request_index=warmup_index,
                )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            monitor = GpuMonitor(interval_seconds=0.25).start()
            raw_records = []
            total_start = time.perf_counter()
            for request_index in range(args.requests):
                raw_records.append(
                    generate_synthetic_one(
                        model=model,
                        tokenizer=tokenizer,
                        torch=torch,
                        spec=spec,
                        request_index=request_index,
                    )
                )
            total_seconds = time.perf_counter() - total_start
            monitor_summary = monitor.stop()
            cuda_peak_gb = None
            if torch.cuda.is_available():
                cuda_peak_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

            summary = summarize_dataset(
                model_name=args.model,
                precision=aggregate["precision"],
                spec=spec,
                raw_records=raw_records,
                total_seconds=total_seconds,
                monitor_summary=monitor_summary,
                cuda_peak_gb=cuda_peak_gb,
                warmup_requests=args.warmup_requests,
            )
            write_jsonl(output_dir / f"hf_synthetic_{spec.name}_raw.jsonl", raw_records)
            write_json(output_dir / f"hf_synthetic_{spec.name}_summary.json", summary)
            aggregate["datasets"][spec.name] = summary
            print(json.dumps(summary, indent=2, sort_keys=True))

    write_json(output_dir / "hf_synthetic_summary.json", aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HF synthetic length-regime baselines.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["isl1024_osl128", "isl2048_osl256"],
        help="Synthetic regimes to run.",
    )
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_hf_synthetic(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

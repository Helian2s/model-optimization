# TensorRT-LLM Optimization Benchmark Report

## Status

This report was generated from parsed local result files.

## Inputs Found

- Environment report: `results/remaining_20260629T043847Z/env_report.json` exists=True
- HF summary: `results/remaining_20260629T043847Z/hf_baseline_summary.json` exists=True
- Quality report: `results/remaining_20260629T043847Z/quality_regression.json` exists=True

## Environment

| Field | Value |
| --- | --- |
| Captured UTC | 2026-06-29T03:18:10.493931+00:00 |
| EC2 instance | g6e.2xlarge / i-0c769a18f50fd1fe6 |
| Availability zone | us-west-2c |
| GPU | NVIDIA L40S x1 |
| GPU memory | 46068.0 MiB |
| NVIDIA driver | 595.71.05 |
| CUDA version | 13.2 |
| Docker | Docker version 29.6.0, build fb59821 |
| NVIDIA Container Toolkit | True |
| Container image | nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19 |
| Container digest | nvcr.io/nvidia/tensorrt-llm/release@sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28 |
| PyTorch |  |
| Transformers |  |
| TensorRT-LLM | 1.3.0rc19 |
| TensorRT |  |
| Git commit |  |

## Model And Engine

| Field | Value |
| --- | --- |
| Model | Qwen/Qwen3-1.7B |
| Snapshot path | /mnt/workspace/huggingface-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| HF precision | bf16 |
| TensorRT-LLM engine dtype | bfloat16 |
| TensorRT-LLM engine dir | artifacts/trtllm_engine_bf16_or_fp16 |
| Engine status | built |

## Summary Table

| variant | dataset | input_len | output_len | num_requests | avg_latency_ms | p50_latency_ms | p90_latency_ms | p95_latency_ms | p99_latency_ms | avg_ttft_ms | avg_itl_ms | request_throughput_rps | total_tokens_per_sec | generation_tokens_per_sec | max_gpu_memory_gb | quality_pass_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hf_baseline | prompt_set | 41 | 70 | 50 | 1865.907 | 951.733 | 5076.641 | 5103.797 | 5914.950 |  |  | 0.536 | 37.662 | 37.662 | 3.251 | 0.520 | model=Qwen/Qwen3-1.7B; precision=bf16 |
| hf_synthetic | isl1024_osl128 | 1024 | 128 | 20 | 3415.315 | 3412.645 | 3428.285 | 3431.110 | 3433.596 |  |  | 0.293 | 37.473 | 37.473 | 3.374 |  | model=Qwen/Qwen3-1.7B; precision=bf16; length-matched synthetic HF baseline |
| hf_synthetic | isl2048_osl256 | 2048 | 256 | 20 | 6855.528 | 6860.488 | 6923.552 | 6931.466 | 6938.751 |  |  | 0.146 | 37.339 | 37.339 | 3.535 |  | model=Qwen/Qwen3-1.7B; precision=bf16; length-matched synthetic HF baseline |
| trtllm_base | prompt_set | 41 | 70 | 50 | 21988.442 | 21887.700 | 22546.210 | 22660.176 | 22849.105 |  |  | 0.045 | 3.184 | 3.184 | 40.506 | 0.520 | model=Qwen/Qwen3-1.7B; backend=examples; engine=artifacts/trtllm_engine_bf16_or_fp16; quality-output-only; examples backend includes per-prompt process startup |
| trtllm_base | isl1024_osl128 | 1024 | 128 | 20 | 829.752 | 829.954 | 833.556 | 834.455 | 834.455 | 22.953 | 6.353 | 1.205 | 1388.296 | 154.255 | 40.460 |  | latency; max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log |
| trtllm_base | isl2048_osl256 | 2048 | 256 | 20 | 1709.091 | 1710.436 | 1712.168 | 1712.311 | 1712.311 | 41.941 | 6.538 | 0.585 | 1348.049 | 149.783 | 40.460 |  | latency; max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log |
| trtllm_base | isl1024_osl128 | 1024 | 128 | 20 | 2133.307 | 2428.693 | 3371.188 | 3371.475 | 3371.475 |  |  | 5.924 | 6824.227 | 758.247 | 40.460 |  | throughput; max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log |
| trtllm_base | isl2048_osl256 | 2048 | 256 | 20 | 4980.614 | 5641.575 | 7732.250 | 7732.625 | 7732.625 |  |  | 2.585 | 5955.295 | 661.699 | 40.460 |  | throughput; max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log |

## Speedup Table

| Variant | Dataset | Comparable | Latency speedup | Throughput speedup | Memory reduction | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| trtllm_base | prompt_set | False |  |  |  | same prompt records, but runner includes per-prompt example-process startup; use for quality only |
| trtllm_base | isl1024_osl128 | True | 4.116 | 4.116 | 0.083 | compared with length-matched HF synthetic baseline |
| trtllm_base | isl2048_osl256 | True | 4.011 | 4.011 | 0.087 | compared with length-matched HF synthetic baseline |
| trtllm_base | isl1024_osl128 | True | 1.601 | 20.235 | 0.083 | compared with length-matched HF synthetic baseline |
| trtllm_base | isl2048_osl256 | True | 1.376 | 17.721 | 0.087 | compared with length-matched HF synthetic baseline |

## Memory Table

| Variant | Dataset | Mode/notes | Max GPU memory GB | HF/TRT memory ratio |
| --- | --- | --- | ---: | ---: |
| hf_baseline | prompt_set | model=Qwen/Qwen3-1.7B; precision=bf16 | 3.251 |  |
| hf_synthetic | isl1024_osl128 | model=Qwen/Qwen3-1.7B; precision=bf16; length-matched synthetic HF baseline | 3.374 |  |
| hf_synthetic | isl2048_osl256 | model=Qwen/Qwen3-1.7B; precision=bf16; length-matched synthetic HF baseline | 3.535 |  |
| trtllm_base | prompt_set | model=Qwen/Qwen3-1.7B; backend=examples; engine=artifacts/trtllm_engine_bf16_or_fp16; quality-output-only; examples backend includes per-prompt process startup | 40.506 | 0.080 |
| trtllm_base | isl1024_osl128 | latency; max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log | 40.460 | 0.083 |
| trtllm_base | isl2048_osl256 | latency; max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log | 40.460 | 0.087 |
| trtllm_base | isl1024_osl128 | throughput; max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log | 40.460 | 0.083 |
| trtllm_base | isl2048_osl256 | throughput; max_gpu_memory_gb estimated from TensorRT paged KV-cache allocation log | 40.460 | 0.087 |

## Quality Regression

| Variant | Total | Passed | Failed | Strict pass rate | Empty/refusal | Pass delta vs HF | New failures vs HF | Output match vs HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf_baseline | 25 | 13 | 12 | 0.520 | 1 |  |  |  |
| trtllm_base | 25 | 13 | 12 | 0.520 | 1 | +0 | 0 | 0.840 |

## Optimization Attempt

# TensorRT-LLM Optimization Attempt

- Status: `fp8_kv_flag_not_exposed_and_runtime_kv_fraction_flag_not_exposed`
- Optimized engine dir: `/workspace/artifacts/trtllm_engine_fp8_kv`
- Base engine dir: `/workspace/artifacts/trtllm_engine_bf16_or_fp16`

## Attempt Order

- `fp8_kv_cache`
- `fp8_weight_activation`
- `int8`
- `int4_awq_or_gptq`
- `documented_kv_cache_runtime_fallback`

## Commands

No executable optimization command was available in this environment.

## Nsight Systems Summary

# Nsight Systems Profiling

- Target command: `trtllm-bench -m 'Qwen/Qwen3-1.7B' --model_path '<snapshot>' latency --backend tensorrt --engine_dir '/workspace/artifacts/trtllm_engine_bf16_or_fp16' --dataset '/workspace/data/synthetic_requests/isl1024_osl128.json' --num_requests 20 --warmup 5 --concurrency 1`
- Report file: `results/opt_profile_20260629T052000Z/nsys/trtllm_steady_state.nsys-rep`
- Profile status: follow-up opt/profile run generated the `.nsys-rep`; the first remaining run used a missing `.jsonl` dataset path and is superseded by the follow-up profile.

## Observations From Captured Log

- The profiled workload was the steady-state TensorRT-LLM latency benchmark, not model download or engine build.
- TensorRT-LLM reported `nccl_plugin` as `None`; for this single-GPU run, NCCL is not expected to dominate.
- TensorRT-LLM reported `paged_kv_cache` enabled, `tokens_per_block=32`, and about `35.35 GiB` allocated for max tokens in paged KV cache.
- The engine log shows attention/GEMM plugin configuration and decode/runtime setup, but kernel-level dominance should be inspected directly in the `.nsys-rep` timeline or with `nsys stats` in the NVIDIA environment.

## Interpretation Checklist

- CPU launch gaps: inspect CUDA API rows in the `.nsys-rep`; no text stats were exported locally.
- GPU busy time: inspect kernel density in the `.nsys-rep`; the benchmark ran measured requests under Nsight.
- NCCL: not material for this single-GPU run based on `nccl_plugin=None`.
- Dominant kernels: expected to be attention/GEMM/decode related for TensorRT-LLM; confirm in Nsight GUI or `nsys stats`.

## Interpretation

Length-matched synthetic HF rows can be compared with TensorRT-LLM synthetic rows. Prompt-set TensorRT rows marked quality-only are not used for speedup claims.

Latency improved on length-matched measured rows. The best latency speedup was 4.116x for `isl1024_osl128`.
Throughput improved on length-matched measured rows. The best generation-token throughput speedup was 20.235x for `isl1024_osl128`.
Memory usage did not improve in the parsed comparable rows. TensorRT-LLM reserved a large paged KV-cache pool, so the HF/TRT memory ratio is below 1.0.
No TensorRT-specific quality regression was detected for `trtllm_base`: strict pass count delta vs HF was +0, with 0 newly failed checks. Normalized output match vs HF was 84.0%.
The TensorRT prompt-set row is used for quality comparison only because that runner used the examples backend with per-prompt process startup; speedup claims use length-matched synthetic workloads.

## Commands Used

- `bash scripts/00_check_env.sh`
- `python scripts/10_download_model.py --model <model>`
- `bash scripts/20_run_hf_baseline.sh --model <model>`
- `python scripts/22_run_hf_synthetic.py --model <model>`
- `python scripts/30_build_trtllm_engine.py --model <model>`
- `python scripts/31_run_trtllm_smoke.py`
- `bash scripts/32_prepare_trtllm_datasets.sh`
- `bash scripts/33_run_trtllm_latency.sh`
- `bash scripts/34_run_trtllm_throughput.sh`
- `bash scripts/35_run_trtllm_quantized_or_kv_cache.sh`
- `python scripts/40_quality_regression.py`
- `python scripts/50_parse_results.py`

## Notes

- EC2 must be stopped after GPU phases.
- Speedups require matching dataset and output constraints.
- TensorRT-LLM synthetic datasets are not directly compared to prompt-set HF baseline rows.

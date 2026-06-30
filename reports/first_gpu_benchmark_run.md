# First GPU Benchmark Run

Created: 2026-06-29

Instance:

- ID: `i-0c769a18f50fd1fe6`
- Type: `g6e.2xlarge`
- GPU: `NVIDIA L40S`
- Region: `us-west-2`
- Final state after work: `stopped`

## Scope

This was the first real GPU benchmark session after local implementation and EC2 validation.

Requested scope:

- Download `Qwen/Qwen3-1.7B`.
- Run Hugging Face baseline.
- Build TensorRT-LLM engine.
- Run TensorRT-LLM smoke test.
- Run quick TensorRT-LLM latency and throughput benchmarks.

Out of scope for this session:

- Quantization / KV-cache experiment.
- Nsight Systems profiling.
- Quality regression across TensorRT outputs.

## Safety

- EC2 was started only for GPU work.
- In-instance shutdown guards were scheduled.
- Each SSM wrapper uploaded results to S3 and called `sudo shutdown -h now`.
- The instance remained in `stopping` longer than expected after both GPU sessions, so an external EC2 `stop-instances --force` was issued.
- Final verified state: `stopped`.

## Container And Tooling

- Container tag: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`
- Container repo digest previously validated: `nvcr.io/nvidia/tensorrt-llm/release@sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28`
- TensorRT-LLM version: `1.3.0rc19`
- TensorRT version from validation: `10.15.1.29`
- Qwen converter: `/app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py`
- Generic engine runner for smoke: `/app/tensorrt_llm/examples/run.py`

## Result Locations

Primary local parsed result set:

- `results/throughput_retry_20260629T033539Z/summary.csv`
- `results/throughput_retry_20260629T033539Z/summary.json`
- `reports/throughput_retry_20260629T033539Z/final_report.md`

Intermediate local result set from the first GPU session:

- `results/gpu_quick_20260629T031648Z/`
- `reports/gpu_quick_20260629T031648Z/`

S3 result prefixes:

- `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/trtllm-benchmark/quick-run-20260629T031648Z/`
- `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/trtllm-benchmark/throughput-retry-20260629T033539Z/`

Large TensorRT artifacts were uploaded during the first session and left in S3:

- `quick-run-20260629T031648Z/artifacts/trtllm_checkpoint_bf16_or_fp16/`
- `quick-run-20260629T031648Z/artifacts/trtllm_engine_bf16_or_fp16/`

## Successful Outputs

Hugging Face baseline:

- Requests: `50`
- Average input tokens: `37.14`
- Average output tokens: `83.04`
- Average latency: `2174.966 ms`
- P50 / P90 / P95 / P99 latency: `1691.272 / 5003.612 / 5049.417 / 5889.815 ms`
- Generation throughput: `38.168 tok/s`
- Reported max allocated GPU memory: `3.25 GiB`
- GPU monitor peak used memory: `4.642 GiB`

TensorRT-LLM engine:

- Dtype: `bfloat16`
- Engine directory: `artifacts/trtllm_engine_bf16_or_fp16`
- Engine file: `rank0.engine`, about `4.09 GB`
- Checkpoint file: `rank0.safetensors`, about `4.11 GB`
- Build completed successfully.

TensorRT-LLM smoke:

- Status: `ok`
- Used `/app/tensorrt_llm/examples/run.py`
- Used the local Hugging Face snapshot as tokenizer directory.

TensorRT-LLM quick latency:

| dataset | requests | avg latency ms | p50 ms | p90 ms | p95 ms | p99 ms | avg TTFT ms | avg ITL ms | req/s | total tok/s | gen tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `isl1024_osl128` | 20 | 829.752 | 829.954 | 833.556 | 834.455 | 834.455 | 22.953 | 6.353 | 1.205 | 1388.296 | 154.255 |
| `isl2048_osl256` | 20 | 1709.091 | 1710.436 | 1712.168 | 1712.311 | 1712.311 | 41.941 | 6.538 | 0.585 | 1348.049 | 149.783 |

TensorRT-LLM quick throughput:

| dataset | requests | avg latency ms | p50 ms | p90 ms | p95 ms | p99 ms | req/s | total tok/s | gen tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `isl1024_osl128` | 20 | 2133.307 | 2428.693 | 3371.188 | 3371.475 | 3371.475 | 5.924 | 6824.227 | 758.247 |
| `isl2048_osl256` | 20 | 4980.614 | 5641.575 | 7732.250 | 7732.625 | 7732.625 | 2.585 | 5955.295 | 661.699 |

## Issues Found And Fixes

### Throughput `--iteration_log` bug

The first throughput attempt failed before producing a report:

```text
ValueError: _TrtLLM got invalid argument: enable_iter_perf_stats
```

Cause:

- In TensorRT-LLM `1.3.0rc19`, `trtllm-bench throughput --iteration_log ...` passes `enable_iter_perf_stats` into the TRT backend, which rejects it.

Fix:

- `scripts/34_run_trtllm_throughput.sh` no longer passes `--iteration_log` by default.
- A new opt-in flag `--iteration-log` keeps the old behavior available if a future TensorRT-LLM version supports it.
- The retry succeeded and produced both throughput report JSON files.

### Hugging Face token exposure risk

The first two container invocations printed environment arguments that included the Hugging Face token value in SSM command output.

Fix:

- `scripts/02_start_trtllm_container.sh` now passes `-e HF_TOKEN` and `-e HUGGINGFACE_HUB_TOKEN` without embedding values in the Docker command line.
- `scripts/90_run_all.sh` now passes the token through environment variables instead of a `--hf-token` CLI argument.

Recommended follow-up:

- Rotate the Hugging Face token stored in `/finetuning/huggingface/token`.

## Conclusions

- `g6e.2xlarge` is sufficient for the base BF16 TensorRT-LLM engine and quick latency/throughput benchmarks for `Qwen/Qwen3-1.7B`.
- The run completed the first benchmark scope after a throughput retry.
- TensorRT-LLM latency and throughput metrics are now available for two synthetic regimes.
- No speedup should be claimed against the HF prompt-set baseline yet because the HF baseline and TensorRT synthetic datasets are not the same workload.
- A comparable prompt-set TensorRT output path is still needed before making direct HF-vs-TensorRT speedup claims.

## Next Steps

1. Implement or adapt a TensorRT prompt-set runner so HF and TensorRT can be compared on identical prompts.
2. Run `scripts/35_run_trtllm_quantized_or_kv_cache.sh` for the required optimization experiment beyond the base TensorRT runtime.
3. Run `scripts/40_quality_regression.py` after TensorRT prompt-set outputs exist.
4. Run `scripts/60_profile_nsys.sh` for a representative steady-state TensorRT-LLM latency run.
5. Regenerate the final report after quantization/KV-cache, quality, and profiling are complete.

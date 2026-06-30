# TensorRT-LLM 1.2.1 Quantization Follow-Up

Run ID: `quant_probe_temp_20260629T232604Z`

Status: completed successfully; temporary EC2 instance was terminated.

## Infrastructure

- Existing stopped instance `i-0c769a18f50fd1fe6` in `us-west-2c` could not be started because of `InsufficientInstanceCapacity`.
- Temporary `g6e.2xlarge` instance `i-05900920ee0c9d687` was launched in `us-west-2b`.
- Shutdown guard was scheduled inside the instance for `2026-06-30 02:29:31 UTC`.
- Local cleanup trap terminated the temporary instance after SSM success.
- Final safety check after the run found no running/stopping GPU instances.

## NVIDIA Container

- Image tag: `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`
- Image digest: `nvcr.io/nvidia/tensorrt-llm/release@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9`

## Model

- Model: `Qwen/Qwen3-1.7B`
- Resolved snapshot in the temporary instance: `/mnt/workspace/huggingface-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

## Package Versions

| Package | Version |
|---|---:|
| TensorRT-LLM | `1.2.1` |
| TensorRT | `10.14.1.48` |
| NVIDIA ModelOpt | `0.37.0` |
| PyTorch | `2.10.0a0+b4e4ee81d3.nv25.12` |
| Transformers | `4.57.3` |

## Findings

| Area | Result |
|---|---|
| `trtllm-build` quantization flags | Mentions FP8, INT8, INT4, and quantization-related build/plugin options. |
| `trtllm-bench latency/throughput` quantization flags | No direct AWQ/GPTQ/INT4/INT8/FP8 flags. Runtime exposes KV cache memory fraction as `--kv_cache_free_gpu_mem_fraction`. |
| Python `KvCacheConfig(dtype="fp8")` | Constructor works in TensorRT-LLM 1.2.1. |
| FP8 KV smoke test | Failed at runtime for Qwen3-1.7B with an FMHA kernel assertion. This is not a measured optimization success. |
| AWQ/GPTQ | Not directly exposed in `trtllm-build --help`; example paths exist, including `examples/quantization/quantize.py` and a `gptneox/gptq_convert.sh` example. |
| ModelOpt compatibility warning | Container emitted a warning that `transformers==4.57.3` is incompatible with `nvidia-modelopt`; this may matter before attempting ModelOpt AWQ/GPTQ/INT4 conversion. |

## Conclusion

TensorRT-LLM `1.2.1` is a useful stable/non-rc container to inspect because it includes ModelOpt and quantization examples, but it did not make FP8 KV cache a clean runnable optimization for `Qwen/Qwen3-1.7B` on this L40S/g6e setup. The only directly actionable optimization fallback from the current benchmark scripts is the KV-cache runtime memory-fraction experiment using `--kv_cache_free_gpu_mem_fraction`.

This run does not prove an AWQ/GPTQ speedup. It only proves those paths require a separate ModelOpt quantization attempt, likely through `/app/tensorrt_llm/examples/quantization/quantize.py`, with careful handling of the container's ModelOpt/Transformers compatibility warning.

## Artifacts

- Local probe JSON: `results/quant_probe_temp_20260629T232604Z_s3/results/quant_probe_temp_20260629T232604Z/quantization_probe.json`
- Local probe markdown: `results/quant_probe_temp_20260629T232604Z_s3/reports/quant_probe_temp_20260629T232604Z/quantization_probe_summary.md`
- S3 prefix: `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/trtllm-benchmark/quant_probe_temp_20260629T232604Z`

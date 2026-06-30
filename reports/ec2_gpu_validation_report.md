# EC2 GPU Validation Report

Created: 2026-06-29

Instance:

- ID: `i-0c769a18f50fd1fe6`
- Type: `g6e.2xlarge`
- Region: `us-west-2`
- Access: SSM send-command
- Final state after validation: `stopped`

## Safety

- Confirmed instance-initiated shutdown behavior was `stop`.
- Started the instance only for GPU validation.
- Installed an in-instance shutdown guard immediately after SSM came online:
  - scheduled shutdown: `2026-06-29 06:34:09 UTC`
- Stopped the instance after validation and verified AWS state `stopped`.

## Host Validation

Validated from `results/ec2_host_validation.json`.

- Instance metadata reported `g6e.2xlarge`.
- GPU visible through `nvidia-smi`.
- GPU: `NVIDIA L40S`
- GPU memory: `46068 MiB`
- NVIDIA driver: `595.71.05`
- CUDA reported by `nvidia-smi`: `13.2`
- Docker: `Docker version 29.6.0`
- Docker runtimes include `nvidia`.
- NVIDIA Container Toolkit: `1.19.1`
- Hugging Face token parameter available in SSM.
- NGC API key parameter available in SSM.
- Internet access to Hugging Face worked.
- `nvcr.io` returned `401 Unauthorized`, which is expected before authenticated Docker login.

Storage:

- Root volume: about `96G`, `78G` free during validation.
- Persistent workspace volume mounted at `/mnt/workspace`: about `246G`, `205G` free before pulling the TensorRT-LLM container.
- Local DLAMI NVMe mounted at `/opt/dlami/nvme`: about `412G`, `391G` free.

After pulling the TensorRT-LLM container, the project env report saw `/mnt/workspace` with about `145G` free, so the container consumes substantial workspace/Docker storage. The instance still has enough space for this project, but benchmark artifacts should stay on `/mnt/workspace` or `/opt/dlami/nvme`, not root.

## TensorRT-LLM Container Validation

Validated from `results/ec2_ngc_validation.json`.

Image:

- Tag: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`
- Pulled digest: `sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28`
- Repo digest: `nvcr.io/nvidia/tensorrt-llm/release@sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28`

Inside the container:

- `import tensorrt_llm` succeeded.
- TensorRT-LLM version: `1.3.0rc19`
- TensorRT version: `10.15.1.29`
- `trtllm-build` found at `/usr/local/bin/trtllm-build`
- `trtllm-bench` found at `/usr/local/bin/trtllm-bench`
- Qwen converter found at `/app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py`
- Llama converter also present.

Warnings observed:

- `torchao` warning about incompatible torch version.
- `nvidia-modelopt` warning that `transformers 5.5.4` may be incompatible with ModelOpt HF workflows.

These warnings do not block TensorRT-LLM import or CLI discovery. They may matter if the quantization path uses ModelOpt/HF integration; the FP8 KV-cache path should be tried first.

## Project Environment Capture

The current local repo snapshot was copied to EC2 and `bash scripts/00_check_env.sh --container-image nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19` ran successfully from `/mnt/workspace/model-optimization`.

Copied back locally:

- `results/ec2_validation/env_report.json`
- `results/ec2_validation/env_report.txt`

S3 result prefix:

- `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/trtllm-benchmark/ec2-validation-20260629T024640Z`

## Conclusions

- `g6e.2xlarge` is validated for the next GPU benchmark phase.
- The NVIDIA container path is valid.
- TensorRT-LLM CLI tools needed by the scripts exist in the container.
- The Qwen conversion path exists in this TensorRT-LLM container.
- The next EC2 session can move from validation to model download, HF baseline, engine build, and quick TensorRT-LLM benchmark.

## Next Script Adjustments Before Benchmark Session

- Prefer `/mnt/workspace` for Hugging Face cache, Docker data, TensorRT checkpoints, and engines.
- Update build logic to use `/app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py` inside the container.
- Update benchmark wrappers if `trtllm-bench --help` shows syntax differences during the next session.
- Treat ModelOpt/HF warnings as a risk for quantization; try FP8 KV cache first.

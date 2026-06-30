# TensorRT-LLM Optimization Benchmark

Reproducible AWS/EC2 benchmark project for optimizing a small LLM with NVIDIA TensorRT-LLM.

The intended workflow is:

1. Capture the EC2/GPU/container environment.
2. Run a Hugging Face Transformers baseline.
3. Build and benchmark a TensorRT-LLM engine.
4. Attempt an additional optimization such as FP8 KV cache or quantization.
5. Parse benchmark logs into a final report with measured speedups.

Primary model: `Qwen/Qwen3-1.7B`

Fallback models:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

See [reports/aws_infrastructure_assessment.md](reports/aws_infrastructure_assessment.md) for the AWS infrastructure assessment and recommended EC2 target.

See [reports/final_implementation_plan.md](reports/final_implementation_plan.md) for the detailed implementation plan, execution sequence, and time estimate.

## Current Status

The required benchmark implementation and first measured AWS GPU run are complete.

Authoritative local outputs:

- [reports/final_report.md](reports/final_report.md)
- [results/summary.csv](results/summary.csv)
- [results/summary.json](results/summary.json)
- [reports/repository_packaging_checklist.md](reports/repository_packaging_checklist.md)

Measured conclusion from the saved run:

- TensorRT-LLM improved latency on length-matched synthetic workloads. Best parsed latency speedup: `4.116x` on `isl1024_osl128`.
- TensorRT-LLM improved generation-token throughput. Best parsed throughput speedup: `20.235x` on `isl1024_osl128`.
- Memory usage did not improve in the parsed comparable rows because TensorRT-LLM reserved a large paged KV-cache pool.
- No TensorRT-specific quality regression was detected by the simple deterministic checks.

The TensorRT-LLM prompt-set row is used for quality comparison only because that runner used the examples backend with per-prompt process startup. Speedup claims use length-matched synthetic workloads.

## NVIDIA-First Policy

This project should use NVIDIA platform components wherever practical:

- official NGC TensorRT-LLM containers for optimized inference
- CUDA/NVIDIA Container Toolkit for GPU container execution
- TensorRT-LLM APIs and CLIs before custom inference code
- Nsight Systems for profiling
- NVML, `nvidia-smi`, and NVIDIA tooling for GPU telemetry
- TensorRT/TensorRT-LLM reported versions and engine metadata in results

Hugging Face Transformers is used only as the required baseline path. AWS services should provide infrastructure only; the benchmark should not use SageMaker, Inferentia, Trainium, Neuron, or non-NVIDIA inference runtimes unless explicitly added later for a separate comparison.

## Local-First Execution Policy

Use the local CPU-only workstation for everything that does not require a GPU:

- repository editing and validation
- prompt/config generation
- AWS inventory and planning
- result parsing
- final report generation
- lightweight unit tests and schema checks

Use EC2 only for GPU-required work:

- Hugging Face GPU baseline measurement
- TensorRT-LLM container validation
- TensorRT-LLM engine build
- TensorRT-LLM latency/throughput benchmarks
- quantization or KV-cache experiments
- Nsight Systems GPU profiling
- target EC2 environment capture

The EC2 GPU instance must be started only for those GPU phases and stopped immediately after the GPU work finishes or fails. Non-GPU preparation and post-processing should happen locally before starting the instance and after stopping it.

Before any long GPU work, schedule an in-instance shutdown guard such as `sudo shutdown -h +240` so the instance stops even if AWS API access, internet connectivity, or the Codex session is lost.

This keeps AWS cost controlled and makes GPU runs shorter and more reproducible.

## Local Python Environment

A local virtual environment should live at `.venv/` and is ignored by git.

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorRT-LLM itself should be run from the official NGC TensorRT-LLM container rather than installed into the host virtual environment.

## Local Validation

These commands do not require an EC2 GPU instance:

```bash
source .venv/bin/activate
python -m compileall src scripts
python scripts/50_parse_results.py --results-dir results/remaining_20260629T043847Z --reports-dir reports
```

The parser refreshes:

- `results/remaining_20260629T043847Z/summary.csv`
- `results/remaining_20260629T043847Z/summary.json`
- `reports/final_report.md`

The top-level acceptance summaries are:

- `results/summary.csv`
- `results/summary.json`

## GPU Reproduction Path

Run GPU phases only on the EC2 instance and only after local changes are ready:

```bash
bash scripts/90_run_all.sh --model Qwen/Qwen3-1.7B --quick
```

The project uses the validated NVIDIA NGC TensorRT-LLM container:

```text
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19
nvcr.io/nvidia/tensorrt-llm/release@sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28
```

The preferred EC2 target from the saved assessment is `g6e.2xlarge` in `us-west-2` with an NVIDIA L40S GPU.

## Artifact Policy

Commit the source, configs, prompts, final summaries, and final report. Keep these heavy artifacts local or in S3:

- `artifacts/`
- raw run folders under `results/<run-id>/`
- `*.nsys-rep`
- `*_iterations.jsonl`
- model/container/cache directories

Before publishing, rotate the Hugging Face token used by the EC2/SSM workflow because an early GPU-session command path may have exposed it before the Docker launcher was patched.

# Codex Session History

Updated: 2026-06-29

This file is a structured history of the Codex work performed in this repository. It is not a raw chat transcript export; the Codex UI transcript is not directly available to write from the workspace. This summary captures the useful technical decisions, commands, outcomes, and next steps needed to continue the project safely.

## Initial Task

Build a reproducible AWS/EC2 TensorRT-LLM benchmark project for NVIDIA Gen AI LLM certification preparation.

Required outcomes:

- Capture EC2/GPU/container environment.
- Run Hugging Face baseline.
- Build and benchmark TensorRT-LLM optimized inference.
- Attempt FP8 KV-cache, FP8, INT8, INT4/AWQ/GPTQ, or a documented KV-cache/runtime fallback.
- Run quality regression checks.
- Produce `results/summary.csv`, `results/summary.json`, and `reports/final_report.md`.
- Use raw EC2, not SageMaker, Inferentia, Trainium, or Neuron.

Primary model:

- `Qwen/Qwen3-1.7B`

Fallbacks:

1. `Qwen/Qwen2.5-1.5B-Instruct`
2. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Policies Established

### NVIDIA-First

Use NVIDIA platform components wherever practical:

- NGC TensorRT-LLM container.
- NVIDIA Container Toolkit.
- CUDA/TensorRT/TensorRT-LLM.
- `nvidia-smi` and NVML telemetry.
- Nsight Systems profiling.

Hugging Face/PyTorch are used only for the required baseline.

### Local-First

Use the local CPU workstation for:

- repository editing
- config/prompt generation
- AWS inventory
- parsing and final report generation
- lightweight validation

Use EC2 only for:

- GPU environment capture
- HF GPU baseline timing
- TensorRT-LLM container validation
- engine build
- latency/throughput benchmarks
- quantization or KV-cache experiment
- Nsight Systems profiling

### EC2 Cost Safety

Start EC2 only when GPU work is ready. Stop it immediately after GPU work finishes or fails.

Before any long GPU work, schedule an in-instance shutdown guard:

```bash
sudo shutdown -h +240 "Safety stop for TensorRT-LLM benchmark"
```

For longer runs:

```bash
sudo shutdown -h +480 "Safety stop for TensorRT-LLM benchmark"
```

The target instance has instance-initiated shutdown behavior set to `stop`, so OS shutdown stops the instance instead of terminating it.

## AWS Inventory Findings

AWS profile used for inventory:

- `finetuning-local`

Region:

- `us-west-2`

Existing GPU instance:

- Instance ID: `i-0c769a18f50fd1fe6`
- Type: `g6e.2xlarge`
- AZ: `us-west-2c`
- AMI: `ami-002a4c6455e1ec9cb`
- AMI name: `Deep Learning Base AMI with Single CUDA (Ubuntu 24.04) 20260619`
- Access: SSM Session Manager
- Security group: no ingress, outbound allowed
- Instance profile: `FinetuningGpuInstanceRole`
- Shutdown behavior: `stop`
- Termination protection: enabled

Storage:

- Root EBS: 100 GiB gp3
- Persistent data EBS: 250 GiB gp3 at `/mnt/workspace`
- DLAMI local NVMe: `/opt/dlami/nvme`

S3 bucket:

- `finetuning-lab-1-037678282394-us-west-2-an`
- Private, encrypted, versioning enabled, no lifecycle policy.
- Use only for small logs/reports or temporary source bundles unless lifecycle cleanup is added.

SSM parameters available to the instance:

- `/finetuning/huggingface/token`
- `/finetuning/ngc/api-key`

Quotas:

- G/VT On-Demand vCPU quota: `8`
- P instance On-Demand quota: `0`
- GPU Spot quota: `0`

Implications:

- `g6e.2xlarge` is usable.
- `g6e.4xlarge`, `p5.4xlarge`, and GPU Spot are currently blocked by quota.

Pricing snapshot:

- `g6e.2xlarge`: `$2.24208/hr`
- `g6e.4xlarge`: `$3.00424/hr`
- `p5.4xlarge`: `$6.88/hr`
- Existing 350 GiB gp3 storage: about `$28/month`

## Repository Work Completed

Created required structure:

```text
configs/
scripts/
src/
data/
data/synthetic_requests/
results/
reports/
artifacts/
```

Created local venv:

```text
.venv/
```

Created and updated:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `configs/benchmark_config.yaml`
- `reports/aws_infrastructure_assessment.md`
- `reports/final_implementation_plan.md`
- `reports/local_implementation_progress.md`
- `reports/ec2_gpu_validation_report.md`

Implemented prompt dataset:

- `data/prompts.jsonl`
- 50 prompts total.
- 25 deterministic prompts with `expected_contains`.
- Categories include short chat, summarization, coding, reasoning, long context, quality, and deterministic longer outputs.

Implemented local modules:

- `src/results_schema.py`
- `src/prompt_loader.py`
- `src/gpu_monitor.py`
- `src/env_report.py`
- `src/hf_benchmark.py`
- `src/trtllm_result_parser.py`
- `src/quality_checks.py`

Implemented scripts:

- `scripts/00_check_env.sh`
- `scripts/01_install_host_deps.sh`
- `scripts/02_start_trtllm_container.sh`
- `scripts/10_download_model.py`
- `scripts/20_run_hf_baseline.sh`
- `scripts/21_run_hf_baseline.py`
- `scripts/30_build_trtllm_engine.py`
- `scripts/31_run_trtllm_smoke.py`
- `scripts/32_prepare_trtllm_datasets.sh`
- `scripts/33_run_trtllm_latency.sh`
- `scripts/34_run_trtllm_throughput.sh`
- `scripts/35_run_trtllm_quantized_or_kv_cache.sh`
- `scripts/40_quality_regression.py`
- `scripts/50_parse_results.py`
- `scripts/60_profile_nsys.sh`
- `scripts/90_run_all.sh`

## Local Validation Completed

Command:

```bash
bash scripts/90_run_all.sh --local-only --model Qwen/Qwen3-1.7B --quick
```

Validated locally:

- prompt loading and schema checks
- local environment capture
- model download dry-run
- TensorRT engine dry-run/discovery path
- TensorRT smoke dry-run
- synthetic dataset dry-run
- latency/throughput dry-runs
- optimization dry-run
- Nsight dry-run
- quality regression missing-output reporting
- fixture report generation

Local generated outputs:

- `results/local_check/env_report.json`
- `results/local_check/env_report.txt`
- `results/local_check/model_download.json`
- `results/local_check/quality_regression.json`
- `results/local_check/quality_regression.md`
- `results/local_check/summary.csv`
- `results/local_check/summary.json`
- `results/local_check/trtllm_build_metadata.json`
- `results/local_check/trtllm_smoke.json`
- `reports/local_check/final_report.md`

These generated files are ignored by git.

## EC2 GPU Validation Completed

The instance was started only for validation and stopped afterward.

Final verified state:

```text
i-0c769a18f50fd1fe6: stopped
```

## First GPU Benchmark Session Completed

Date: 2026-06-29

Final verified EC2 state:

```text
i-0c769a18f50fd1fe6: stopped
```

Run report:

- `reports/first_gpu_benchmark_run.md`

Primary local result set:

- `results/throughput_retry_20260629T033539Z/summary.csv`
- `results/throughput_retry_20260629T033539Z/summary.json`
- `reports/throughput_retry_20260629T033539Z/final_report.md`

S3 result prefixes:

- `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/trtllm-benchmark/quick-run-20260629T031648Z/`
- `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/trtllm-benchmark/throughput-retry-20260629T033539Z/`

Completed:

- Model download for `Qwen/Qwen3-1.7B`.
- HF baseline on GPU.
- TensorRT-LLM Qwen checkpoint conversion.
- TensorRT-LLM BF16 engine build.
- TensorRT-LLM smoke test using `/app/tensorrt_llm/examples/run.py`.
- Synthetic dataset preparation.
- Quick latency benchmarks for `isl1024_osl128` and `isl2048_osl256`.
- Quick throughput benchmarks for `isl1024_osl128` and `isl2048_osl256`.

Key measured rows:

- HF prompt-set baseline: `2174.966 ms` average latency, `38.168` generated tok/s.
- TensorRT latency `isl1024_osl128`: `829.752 ms`, `154.255` generated tok/s.
- TensorRT latency `isl2048_osl256`: `1709.091 ms`, `149.783` generated tok/s.
- TensorRT throughput `isl1024_osl128`: `5.924 req/s`, `758.247` generated tok/s.
- TensorRT throughput `isl2048_osl256`: `2.585 req/s`, `661.699` generated tok/s.

Fixes made during the session:

- `scripts/31_run_trtllm_smoke.py` now uses the generic TensorRT-LLM `examples/run.py` path and resolves the local tokenizer snapshot.
- `scripts/34_run_trtllm_throughput.sh` no longer passes `--iteration_log` by default because TensorRT-LLM `1.3.0rc19` rejects the resulting `enable_iter_perf_stats` argument for the TRT backend.
- `scripts/02_start_trtllm_container.sh` and `scripts/90_run_all.sh` were patched to avoid putting the Hugging Face token value into Docker command output.
- `src/trtllm_result_parser.py` now parses TensorRT-LLM `*_report.json` files directly for latency/throughput metrics.

Security note:

- The first two container invocations in the initial GPU session printed the Hugging Face token value in SSM command output before the launcher was patched. Rotate the token stored in `/finetuning/huggingface/token`.

Remaining major work:

- Implement a TensorRT prompt-set runner for direct HF-vs-TensorRT comparison on identical prompts.
- Run the required quantization / KV-cache experiment.
- Run quality regression once TensorRT prompt-set outputs exist.
- Run Nsight Systems profiling.
- Regenerate the final report after those remaining GPU phases.

## Current Continuation Point

User requested: save chat history and start the next step.

Next step selected:

- Build the TensorRT prompt-set runner locally before starting EC2 again.

Reason:

- The first GPU session produced valid TensorRT synthetic latency/throughput results.
- Direct speedup and quality claims still require running TensorRT on the same fixed prompts used by the Hugging Face baseline.
- This runner should write `results/trtllm_outputs_raw.jsonl` so `scripts/40_quality_regression.py` can compare HF baseline and TensorRT output.
- EC2 should remain stopped while this runner is implemented and locally validated.

Safety:

- Confirmed shutdown behavior was `stop`.
- Installed shutdown guard immediately after SSM came online.
- Stopped the instance after validation and waited until AWS reported `stopped`.

Host validation:

- GPU: `NVIDIA L40S`
- GPU memory: `46068 MiB`
- Driver: `595.71.05`
- CUDA reported by `nvidia-smi`: `13.2`
- Docker: `29.6.0`
- Docker runtimes include `nvidia`
- NVIDIA Container Toolkit: `1.19.1`
- `/mnt/workspace` mounted and usable
- `/opt/dlami/nvme` available
- HF and NGC SSM parameters available

NGC TensorRT-LLM container validation:

- Image: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`
- Actual pulled digest: `sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28`
- Repo digest: `nvcr.io/nvidia/tensorrt-llm/release@sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28`
- `import tensorrt_llm` succeeded
- TensorRT-LLM version: `1.3.0rc19`
- TensorRT version: `10.15.1.29`
- `trtllm-build` found
- `trtllm-bench` found
- Qwen converter found:
  - `/app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py`

Validation reports copied locally:

- `results/ec2_validation/env_report.json`
- `results/ec2_validation/env_report.txt`

S3 result prefix:

- `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/trtllm-benchmark/ec2-validation-20260629T024640Z`

## Important Observations

The actual NGC digest differed from the earlier catalog lookup. The config was updated to the digest pulled on EC2:

```text
nvcr.io/nvidia/tensorrt-llm/release@sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28
```

The TensorRT-LLM container emitted warnings:

- `torchao` warning about incompatible torch version.
- `nvidia-modelopt` warning that `transformers 5.5.4` may be incompatible with ModelOpt HF workflows.

These did not block TensorRT-LLM import or CLI discovery. They may affect quantization paths, so FP8 KV cache should be attempted first.

After pulling the TensorRT-LLM container, `/mnt/workspace` free space dropped to about `145G`. Continue using `/mnt/workspace` and `/opt/dlami/nvme` for heavy artifacts, not root.

## Current Project State

EC2:

- stopped

Local implementation:

- local scaffolding and validation complete
- GPU validation complete
- not yet run real HF baseline
- not yet built real TensorRT-LLM engine
- not yet run real TensorRT-LLM benchmark
- not yet run real quantization/KV-cache experiment
- not yet generated final measured report

## Next Steps

Before next EC2 benchmark session:

1. Adjust TensorRT build logic to use the confirmed Qwen converter path:
   - `/app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py`
2. Prefer `/mnt/workspace` for:
   - Hugging Face cache
   - TensorRT checkpoints
   - TensorRT engines
   - result artifacts
3. Check `trtllm-build --help` and `trtllm-bench --help` inside the container if syntax mismatches appear.

Next EC2 session:

1. Start EC2.
2. Schedule shutdown guard.
3. Run model download.
4. Run HF GPU baseline.
5. Build TensorRT-LLM BF16/FP16 engine.
6. Smoke test generation.
7. Prepare synthetic datasets.
8. Run quick latency/throughput benchmarks.
9. Stop EC2.
10. Parse results locally and decide whether to proceed to full benchmark/optimization.

Do not leave EC2 running for local parsing or report editing.

## Prompt-Set GPU Run - 2026-06-29

User requested: save chat history and start the next step.

Work completed:

- Implemented TensorRT prompt-set runner hardening:
  - `scripts/36_run_trtllm_prompt_set.py` now supports incremental raw output writes.
  - Added `--resume`.
  - Added `--examples-timeout-seconds`.
- Updated prompt-set EC2 wrapper:
  - `scripts/aws/run_ec2_prompt_set.sh` now defaults to `TRTLLM_PROMPT_BACKEND=examples`.
  - Uses resume and a per-prompt timeout.
- Updated top-level GPU runner:
  - `scripts/90_run_all.sh` now passes prompt backend, resume, and timeout settings.
- Updated reporting:
  - `src/trtllm_result_parser.py` marks `examples` prompt-set rows as quality-output-only.
  - `scripts/50_parse_results.py` no longer treats per-prompt `examples/run.py` output as fair speedup data.
  - `scripts/50_parse_results.py` now includes a quality regression table in the final report.
- Added Qwen3 thinking-mode control after the prompt-set run exposed short-output quality failures:
  - `src/hf_benchmark.py` passes `enable_thinking=False` for Qwen3 in `thinking-mode=auto`.
  - `scripts/21_run_hf_baseline.py` supports `--thinking-mode auto|on|off`.
  - `scripts/36_run_trtllm_prompt_set.py` supports `--thinking-mode auto|on|off`.
  - `scripts/90_run_all.sh` and `scripts/aws/run_ec2_prompt_set.sh` pass `LLM_THINKING_MODE`.

GPU run:

- EC2 instance: `i-0c769a18f50fd1fe6`
- Instance type: `g6e.2xlarge`
- SSM command: `39363d17-5ef7-43a9-b300-ae022099bd29`
- S3 result prefix: `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/results/trtllm-benchmark/prompt-set-20260629T035306Z`
- Runtime: `21m15.954s`
- Final EC2 state verified: `stopped`

Local synced files:

- `results/prompt_set_20260629T035306Z/`
- `reports/prompt_set_20260629T035306Z/`
- `reports/prompt_set_run_20260629T035306Z.md`

Prompt-set TensorRT result:

- Backend used: `examples`
- Total requests: `50`
- Average latency: `21982.832 ms`
- P50 latency: `21854.962 ms`
- P95 latency: `22686.289 ms`
- Generation throughput: `3.777 tokens/sec`
- Max observed GPU memory: `40.51 GB`

Important interpretation:

- The prompt-set TensorRT outputs are useful for quality comparison.
- The prompt-set latency/throughput numbers are not valid optimized-runtime speedup numbers because the fallback used `examples/run.py` once per prompt, including process and engine startup overhead.
- Synthetic TensorRT-LLM latency/throughput rows from `trtllm-bench` remain the valid optimized runtime benchmark results.

LLM API fallback reason:

```text
Unrecognized model in artifacts/trtllm_engine_bf16_or_fp16. Should have a `model_type` key in its config.json.
```

Quality result:

- HF baseline: `1/25` deterministic checks passed.
- TensorRT-LLM base: `1/25` deterministic checks passed.
- This is not a TensorRT-specific regression. Qwen3 emitted `<think>` reasoning text, and short deterministic `max_new_tokens` values truncated output before expected substrings.

Next continuation point:

1. Start EC2 only for a short rerun of HF baseline and TensorRT prompt-set quality with `LLM_THINKING_MODE=auto` or `off`.
2. Stop EC2 immediately after that GPU rerun.
3. Sync results locally.
4. Rerun quality regression and final report parsing.

## Remaining GPU Phases - 2026-06-29

Completed remaining measured phases:

- HF baseline rerun with Qwen3 thinking control.
- HF synthetic baselines for `isl1024_osl128` and `isl2048_osl256`.
- TensorRT prompt-set output rerun with Qwen3 thinking control.
- Quality regression.
- Optimization/KV-cache attempt.
- Nsight Systems profile attempt and follow-up profile capture.
- Final parsing and report generation.

AWS safety:

- Main SSM command: `90430080-0f74-498d-ac88-2c7ea4ab0eec`
- Follow-up opt/profile SSM command: `a3907a68-3d0c-48bc-98c4-227392256533`
- EC2 final state was verified as `stopped` before local AWS credentials expired.

Local result folders:

- `results/remaining_20260629T043847Z/`
- `reports/remaining_20260629T043847Z/`
- `results/opt_profile_20260629T052000Z/`
- `reports/opt_profile_20260629T052000Z/`

Top-level acceptance files refreshed:

- `results/summary.csv`
- `results/summary.json`
- `reports/final_report.md`

Key results:

- HF synthetic `isl1024_osl128`: `3415.315 ms`, `37.473 tok/s`
- TRT latency `isl1024_osl128`: `829.752 ms`, `154.255 tok/s`
- HF synthetic `isl2048_osl256`: `6855.528 ms`, `37.339 tok/s`
- TRT latency `isl2048_osl256`: `1709.091 ms`, `149.783 tok/s`
- TRT throughput `isl1024_osl128`: `758.247 generation tok/s`
- TRT throughput `isl2048_osl256`: `661.699 generation tok/s`

Speedup summary:

- `isl1024_osl128` latency mode: `4.116x`
- `isl2048_osl256` latency mode: `4.011x`
- `isl1024_osl128` throughput mode generation TPS: `20.235x`
- `isl2048_osl256` throughput mode generation TPS: `17.721x`

Quality:

- HF baseline: `12/25`
- TensorRT base: `12/25`
- No TensorRT-specific regression detected by simple pass/fail counts.

Optimization/KV:

- Status: `fp8_kv_flag_not_exposed_and_runtime_kv_fraction_flag_not_exposed`
- TensorRT-LLM 1.3.0rc19 did not expose the checked FP8 KV or runtime KV fraction flags.
- Nsight/TensorRT logs document `paged_kv_cache=True`, `tokens_per_block=32`, and about `35.35 GiB` allocated for paged KV cache.

Nsight:

- Captured report: `results/opt_profile_20260629T052000Z/nsys/trtllm_steady_state.nsys-rep`
- Summary: `results/opt_profile_20260629T052000Z/nsys/profile_summary.md`

AWS upload caveat:

- EC2-side S3 sync completed.
- Local AWS credentials expired while uploading corrected report files after local report regeneration.
- Corrected authoritative files are saved locally in the repository.

## Chat History Save - 2026-06-29

User asked to save the chat history after the benchmark work. This file is the repository-local running handoff log for the Codex session.

Current authoritative state:

- Final local acceptance artifacts exist:
  - `results/summary.csv`
  - `results/summary.json`
  - `reports/final_report.md`
- Completed GPU benchmark artifacts are preserved locally under:
  - `results/remaining_20260629T043847Z/`
  - `reports/remaining_20260629T043847Z/`
  - `results/opt_profile_20260629T052000Z/`
  - `reports/opt_profile_20260629T052000Z/`
- EC2 instance `i-029fba4015ac40a73` was verified stopped before AWS credentials expired.
- AWS credentials later expired during local-to-S3 upload of corrected regenerated report files, so any further AWS status checks or S3 uploads require re-authentication.
- No active GPU work was running from the last verified state.

Important conclusions to carry forward:

- The project now has measured HF baseline, TensorRT-LLM latency, TensorRT-LLM throughput, documented KV-cache/optimization attempt, Nsight profile output, summary files, and final report.
- TensorRT-LLM improved measured latency by about `4.116x` on `isl1024_osl128` and `4.011x` on `isl2048_osl256`.
- TensorRT-LLM throughput mode improved measured generation tokens/sec by about `20.235x` on `isl1024_osl128` and `17.721x` on `isl2048_osl256`.
- Simple deterministic quality checks showed equal pass counts for HF baseline and TensorRT base (`12/25`), so no TensorRT-specific regression was detected by that simple method.
- TensorRT-LLM `1.3.0rc19` did not expose the checked FP8 KV/runtime KV fraction flags in the available in-container CLI path; logs still document paged KV-cache behavior.

Useful next steps, if continuing:

- Review `reports/final_report.md` for wording and presentation.
- Optionally refresh AWS authentication and upload corrected local report files to S3.
- Optionally improve quality prompt expected answers or add semantic checks.
- Optionally test another official NVIDIA TensorRT-LLM container tag if explicit FP8 KV-cache CLI support is required.

## Local Report Polish - 2026-06-29

User asked to proceed with the next steps. Work stayed local; no EC2 instance was started.

Completed:

- Updated `src/quality_checks.py` to report both strict deterministic correctness and regression-vs-HF baseline.
- Added Unicode normalization for simple expected answers such as `H2O` vs `H₂O`.
- Regenerated quality reports for:
  - `results/remaining_20260629T043847Z/quality_regression.json`
  - `results/remaining_20260629T043847Z/quality_regression.md`
  - `results/opt_profile_20260629T052000Z/quality_regression.json`
  - `results/opt_profile_20260629T052000Z/quality_regression.md`
- Updated `src/trtllm_result_parser.py` to estimate TensorRT-LLM GPU memory from paged KV-cache allocation logs.
- Updated `scripts/50_parse_results.py` to add environment, model/engine, memory, richer quality, and explicit interpretation sections to `reports/final_report.md`.
- Regenerated:
  - `results/remaining_20260629T043847Z/summary.csv`
  - `results/remaining_20260629T043847Z/summary.json`
  - `results/summary.csv`
  - `results/summary.json`
  - `reports/final_report.md`

Updated quality conclusion:

- HF baseline strict deterministic checks: `13/25` after Unicode normalization.
- TensorRT base strict deterministic checks: `13/25`.
- TensorRT base introduced `0` newly failed checks vs HF baseline.
- Normalized TensorRT output match vs HF baseline: `84.00%`.
- Report now states no TensorRT-specific quality regression was detected.

Updated performance/memory conclusion:

- Latency improved on comparable synthetic rows, best speedup `4.116x` on `isl1024_osl128`.
- Generation-token throughput improved on comparable synthetic rows, best speedup `20.235x` on `isl1024_osl128`.
- Memory usage did not improve in the parsed comparable rows because TensorRT-LLM reserved a large paged KV-cache pool; report now states this explicitly.
- Final report now records the validated NGC image `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`, digest `sha256:0082fd4f9934326e46bc77c7eb1e14404b168f2c8fd830d926a2da5a092bac28`, and TensorRT-LLM version `1.3.0rc19`.

## Repository Cleanup And Packaging - 2026-06-29

User asked to start the next steps and save chat history. Work stayed local; no EC2 instance was started.

Completed:

- Ran masked secret scans for Hugging Face token literals, AWS key/session patterns, private key headers, bearer/authorization strings, and generic password/secret/token labels.
- No literal AWS or Hugging Face credential values were found in files intended for commit.
- Confirmed matches were code references to environment variables, documentation about credential handling, or prompt text.
- Added `reports/repository_packaging_checklist.md` with the secret scan result, large artifact policy, current final result state, and before-publishing reminders.
- Updated `.gitignore` so lightweight final deliverables are commit-ready:
  - `results/summary.csv`
  - `results/summary.json`
  - `reports/final_report.md`
  - `reports/repository_packaging_checklist.md`
  - selected planning/history reports
- Kept heavy/generated artifacts ignored:
  - `artifacts/`
  - raw run folders under `results/<run-id>/`
  - `*.nsys-rep`
  - `*.qdrep`
  - `*_iterations.jsonl`
  - `trtllm_*.log`
  - `.venv/`
- Updated `README.md` to reflect the completed benchmark state, measured conclusions, local validation path, GPU reproduction command, NGC container tag/digest, and artifact policy.

Important reminder:

- Rotate the Hugging Face token used by the EC2/SSM workflow before publishing or sharing broadly. Earlier notes indicate an initial GPU-session command path may have exposed the token before the Docker launcher was patched.

## Quantization Follow-Up Probe Attempt - 2026-06-29

User re-authenticated AWS and asked to continue the TensorRT-LLM quantization follow-up.

Local preparation completed:

- Added `scripts/37_probe_trtllm_quantization.py`.
- Added `scripts/aws/run_ec2_quantization_probe.sh`.
- Added `scripts/aws/start_ec2_quantization_probe.sh`.
- Added detailed engineer log: `reports/quantization_followup_engineer_log_20260629T221338Z.md`.
- Patched the local launcher to upload the current repo source bundle to S3 before invoking SSM, so EC2 runs the latest local scripts.

Launch attempt:

- `RUN_ID=quant_probe_20260629T222002Z`.
- Source bundle uploaded to `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/source-bundles/model-optimization-quant_probe_20260629T222002Z.tgz`.
- Tried to start existing `g6e.2xlarge` instance `i-0c769a18f50fd1fe6` in `us-west-2c`.
- AWS returned `InsufficientInstanceCapacity`.
- Retried the same stopped instance once; AWS returned `InsufficientInstanceCapacity` again.
- Verified final instance state: `stopped`.

Current status:

- No GPU work ran.
- No EC2 instance is running.
- The quantization probe is ready to run once capacity is available or once the user approves a topology/cost change such as temporarily switching the stopped instance to `g6e.4xlarge` or launching a replacement in another AZ.

## Quantization Follow-Up Completion - 2026-06-29

User reported EC2 was starting and asked to start implementation of the plan.

Infrastructure and safety work:

- Existing instance `i-0c769a18f50fd1fe6` was checked and later remained `stopped`.
- First SSM launcher bug fixed: command parameters are now sent through a JSON parameter file.
- Second SSM launcher bug fixed: remote payload is written to `/tmp/run_trtllm_quant_probe.sh` and invoked with `bash`, because AWS-RunShellScript command strings run under `/bin/sh`.
- Added `scripts/aws/start_temp_ec2_quantization_probe.sh` to launch a temporary same-AMI GPU instance in alternate AZs and terminate it after the probe.
- Temporary launch attempt `quant_probe_temp_20260629T230558Z` created `i-006c3ad4d6cf1e049` in `us-west-2a`; it failed due a local wrapper path bug, then the cleanup trap terminated it.
- Patched `scripts/aws/run_ec2_quantization_probe.sh` to use `10_download_model.py --output-dir` correctly.
- Successful run `quant_probe_temp_20260629T232604Z` created `i-05900920ee0c9d687` in `us-west-2b`, ran the probe, synced S3 artifacts locally, and terminated the instance.
- Final AWS safety check found no GPU instances in `pending`, `running`, `stopping`, or `shutting-down`.

Successful probe facts:

- NVIDIA image: `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`
- Digest: `nvcr.io/nvidia/tensorrt-llm/release@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9`
- Model: `Qwen/Qwen3-1.7B`
- TensorRT-LLM: `1.2.1`
- TensorRT: `10.14.1.48`
- NVIDIA ModelOpt: `0.37.0`
- PyTorch: `2.10.0a0+b4e4ee81d3.nv25.12`
- Transformers: `4.57.3`

Quantization conclusion:

- `KvCacheConfig(dtype="fp8")` exists in TensorRT-LLM 1.2.1, but FP8 KV smoke failed for Qwen3-1.7B with an FMHA kernel assertion. This is not a measured successful optimization.
- `trtllm-build --help` mentions FP8, INT8, INT4, and quantization-related build/plugin options.
- `trtllm-bench latency/throughput` do not expose direct AWQ/GPTQ/INT4/INT8/FP8 benchmark flags.
- Runtime KV-cache memory fraction is exposed as `--kv_cache_free_gpu_mem_fraction`; patched `scripts/35_run_trtllm_quantized_or_kv_cache.sh` and `scripts/37_probe_trtllm_quantization.py` to recognize this spelling.
- AWQ/GPTQ were not completed; example paths exist, especially `/app/tensorrt_llm/examples/quantization/quantize.py`, but a separate ModelOpt quantization attempt is still needed.
- Container warned that `transformers==4.57.3` is incompatible with `nvidia-modelopt`, which should be handled before a serious AWQ/GPTQ/INT4 attempt.

Saved artifacts:

- `reports/quantization_followup_engineer_log_20260629T221338Z.md`
- `reports/quantization_followup_20260629T232604Z.md`
- `results/quantization_followup_summary_20260629T232604Z.json`
- `results/quant_probe_temp_20260629T232604Z_s3/`

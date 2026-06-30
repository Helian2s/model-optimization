# Final Implementation Plan

Created: 2026-06-29

Project: TensorRT-LLM optimization benchmark on AWS EC2

Primary target:

- Local workstation for all CPU-only work.
- Existing stopped EC2 GPU instance `i-0c769a18f50fd1fe6` only for GPU-required phases.
- Instance type: `g6e.2xlarge`
- GPU: 1x NVIDIA L40S, 48 GB class
- Region: `us-west-2`
- Access: AWS Systems Manager Session Manager

## Goal

Build a reproducible benchmark project that demonstrates NVIDIA LLM optimization knowledge:

1. Capture the full runtime environment.
2. Measure a Hugging Face Transformers baseline.
3. Build and benchmark a TensorRT-LLM optimized runtime.
4. Attempt at least one additional optimization, preferably FP8 KV cache.
5. Run quality checks against deterministic prompts.
6. Produce `results/summary.csv`, `results/summary.json`, and `reports/final_report.md`.

The project must not claim speedups unless the matching workload was measured.

## Operating Rules

Use NVIDIA tooling wherever practical:

- NGC TensorRT-LLM container for optimized inference.
- NVIDIA Container Toolkit for Docker GPU access.
- TensorRT-LLM APIs/CLIs for engine build and benchmarks.
- `nvidia-smi` and NVML for telemetry.
- Nsight Systems for profiling.

Use AWS only as raw EC2 infrastructure:

- No SageMaker.
- No Inferentia, Trainium, or Neuron.
- No AWS-managed ML runtime.

Use the local CPU-only workstation whenever possible:

- repo editing
- prompt/config generation
- AWS inventory
- parsing and report generation
- lightweight tests

Start EC2 only for GPU phases, and stop it immediately after the GPU phase completes or fails.

## Model Strategy

Primary model:

- `Qwen/Qwen3-1.7B`

Fallback order:

1. `Qwen/Qwen2.5-1.5B-Instruct`
2. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Do not switch models silently. Any fallback must be recorded in:

- `results/env_report.json`
- `results/summary.json`
- `reports/final_report.md`

## Benchmark Strategy

Two benchmark tracks are needed because the requested artifacts mix real prompt workloads and synthetic fixed-length workloads.

Prompt-set track:

- Uses `data/prompts.jsonl`.
- Used for Hugging Face baseline.
- Used for TensorRT-LLM smoke and quality comparison where the runtime path supports prompt-based generation.
- Speedup claims are valid only when HF and TensorRT-LLM use the same prompt records and output limits.

Synthetic fixed-length track:

- Uses TensorRT-LLM synthetic datasets under `data/synthetic_requests/`.
- Required regimes:
  - `isl128_osl128`
  - `isl512_osl128`
  - `isl1024_osl128`
  - `isl2048_osl256`
- Used for TensorRT-LLM latency and throughput benchmarks.
- Synthetic TensorRT results must not be compared against prompt-set HF baseline results unless a matching HF synthetic baseline is implemented.

## Detailed Step Breakdown

This table is the working execution plan. Local steps should be completed before EC2 starts. GPU steps should be grouped into one or a few short EC2 sessions, each protected by an in-instance shutdown guard.

| Step | Location | Work | Main outputs | Estimate |
| ---: | --- | --- | --- | ---: |
| 1 | local | Finalize project policies, scaffold, and planning files | README/config/reports updated | 0.5 h |
| 2 | local | Build the real prompt dataset with 50+ prompts and 20+ deterministic checks | `data/prompts.jsonl` | 1-2 h |
| 3 | local | Implement shared schemas and prompt loader | `src/results_schema.py`, `src/prompt_loader.py` | 1.5-2.5 h |
| 4 | local | Implement environment capture logic with local and EC2-aware fallbacks | `src/env_report.py`, `scripts/00_check_env.sh` | 2-3 h |
| 5 | local | Implement result parser/report skeleton with fixture support | `src/trtllm_result_parser.py`, `scripts/50_parse_results.py` | 3-5 h |
| 6 | local | Implement model download and Hugging Face baseline scripts | `scripts/10_download_model.py`, `scripts/21_run_hf_baseline.py`, wrapper | 3-5 h |
| 7 | local | Implement quality checks | `src/quality_checks.py`, `scripts/40_quality_regression.py` | 1.5-3 h |
| 8 | local | Implement TensorRT-LLM container and benchmark script wrappers with version discovery | scripts `02`, `30`-`35`, `60` | 4-7 h |
| 9 | local | Implement top-level orchestration and EC2 lifecycle guard hooks | `scripts/90_run_all.sh` | 2-4 h |
| 10 | local | Dry-run config loading, prompt validation, parser fixtures, and report generation | fixture report and clean local test run | 1.5-3 h |
| 11 | EC2 | Start instance, immediately install shutdown guard, validate SSM/GPU/Docker/NVIDIA container | env reports and container validation log | 1.5-3 h |
| 12 | EC2 | Run model download and Hugging Face GPU baseline | HF raw and summary JSON | 1-2 h |
| 13 | EC2 | Build base TensorRT-LLM BF16/FP16 engine and smoke test | base engine artifacts and smoke output | 1.5-3 h |
| 14 | EC2 | Prepare synthetic datasets and run quick latency/throughput benchmarks | at least two dataset logs | 1-2 h |
| 15 | EC2 | Run full benchmark set if quick path is stable | all required dataset logs | 2-4 h |
| 16 | EC2 | Attempt FP8 KV-cache or quantization experiment | optimized artifact/logs or documented fallback | 2-5 h |
| 17 | EC2 | Run quality regression generation and Nsight Systems steady-state profile | quality files and `results/nsys/` | 1-2 h |
| 18 | EC2/local | Stop instance, verify stopped state, sync only small result artifacts | instance stopped, local `results/` populated | 0.25-0.75 h |
| 19 | local | Parse final results and generate report | `results/summary.csv`, `summary.json`, `reports/final_report.md` | 2-4 h |
| 20 | local | Final acceptance pass and documentation cleanup | final checked repo | 1-2 h |

Expected critical path:

- Local implementation before first EC2 run: 20-32 hours.
- First EC2 validation session: 1.5-3 hours.
- Main EC2 benchmark session: 6-12 hours if TensorRT-LLM paths behave normally.
- Local final reporting: 2-4 hours.

## EC2 Safety and Failure Plan

The expensive failure mode is leaving `i-0c769a18f50fd1fe6` running after Codex loses AWS API access, loses internet, or the agent session is interrupted. The plan must not rely on a final Codex cleanup command as the only stop mechanism.

### Mandatory Before Any GPU Work

Before starting long GPU work:

1. Confirm the instance is still configured with instance-initiated shutdown behavior `stop`.
2. Start the instance only when the local scripts and configs needed for that GPU phase are ready.
3. Connect through Session Manager.
4. Immediately install an in-instance shutdown guard before downloads, engine builds, or benchmarks.

Recommended guard for a quick GPU session:

```bash
sudo shutdown -h +240 "Safety stop for TensorRT-LLM benchmark"
```

Recommended guard for a full GPU session:

```bash
sudo shutdown -h +480 "Safety stop for TensorRT-LLM benchmark"
```

Because this instance has instance-initiated shutdown behavior set to `stop`, an OS shutdown should stop the EC2 instance instead of terminating it. If that setting is not `stop`, do not run GPU work until it is corrected.

If more time is genuinely needed while actively connected:

```bash
sudo shutdown -c
sudo shutdown -h +240 "Extended safety stop for active TensorRT-LLM benchmark"
```

Do not cancel the shutdown guard unless replacing it with a new guard.

### Normal Stop Procedure

At the end of each GPU phase:

1. Stop active benchmark processes cleanly.
2. Sync or preserve result files.
3. Stop from inside the instance if SSM is available:

```bash
sudo shutdown -h now
```

4. Also call AWS stop from the local workstation if AWS API access is available:

```bash
aws ec2 stop-instances --profile finetuning-local --region us-west-2 --instance-ids i-0c769a18f50fd1fe6
```

5. Verify stopped state before treating the GPU phase as complete:

```bash
aws ec2 wait instance-stopped --profile finetuning-local --region us-west-2 --instance-ids i-0c769a18f50fd1fe6
```

### Failure Modes

| Failure case | Preferred action | Backup |
| --- | --- | --- |
| Codex loses AWS API authorization but SSM session still works | run `sudo shutdown -h now` inside EC2 | pre-scheduled shutdown guard stops instance |
| Codex loses local internet entirely | no live action possible | pre-scheduled shutdown guard stops instance |
| SSM disconnects but local AWS API still works | run `aws ec2 stop-instances` locally | pre-scheduled shutdown guard stops instance |
| TensorRT build hangs | local runner should timeout and stop instance | pre-scheduled shutdown guard stops instance |
| Codex process dies mid-run | no live action possible | pre-scheduled shutdown guard stops instance |
| AWS credentials expire before final cleanup | use active SSM shell if present | pre-scheduled shutdown guard stops instance |
| Both AWS API and SSM are unavailable | no live action possible | pre-scheduled shutdown guard stops instance |

### Runner Design Requirements

The future `scripts/90_run_all.sh` should:

- refuse GPU phases unless `--gpu` or an equivalent explicit flag is provided
- print the target instance ID, region, and planned maximum runtime before start
- verify instance-initiated shutdown behavior is `stop`
- set the in-instance shutdown guard as the first command after SSM access
- use shell `trap` cleanup where possible
- stop the instance on success
- stop the instance on failure
- verify `instance-stopped` before exiting a GPU phase
- write an EC2 lifecycle log under `results/ec2_lifecycle.log`

If the runner cannot set the in-instance shutdown guard, it must abort before running downloads, builds, or benchmarks.

## Phase Plan

### Phase 0: Repository Scaffold and Planning

Location: local workstation

Status: mostly complete

Deliverables:

- required directory structure
- `.venv`
- infrastructure assessment
- NVIDIA-first policy
- local-first policy
- EC2 start/stop policy
- final implementation plan

Remaining work:

- continue replacing placeholders with real implementation.

Estimated remaining time: 0.5 hour

### Phase 1: Local Core Implementation

Location: local workstation

Implement shared Python modules:

- `src/results_schema.py`
  - typed result records
  - summary table column definitions
  - validation helpers
- `src/prompt_loader.py`
  - JSONL loader
  - prompt schema validation
  - category filtering
- `src/gpu_monitor.py`
  - NVML helpers when available
  - graceful CPU-only fallback for local dry runs
- `src/env_report.py`
  - environment collection
  - JSON and text output writers
- `src/hf_benchmark.py`
  - baseline timing helpers
  - percentile and throughput calculations
- `src/trtllm_result_parser.py`
  - parser for TensorRT-LLM logs
  - tolerant regex/table parsing for CLI version drift
- `src/quality_checks.py`
  - normalized contains checks
  - exact match if expected output is present
  - empty/refusal detection

Implement scripts:

- `scripts/00_check_env.sh`
- `scripts/10_download_model.py`
- `scripts/21_run_hf_baseline.py`
- `scripts/40_quality_regression.py`
- `scripts/50_parse_results.py`

Create real prompt set:

- at least 50 prompts total
- at least 20 deterministic/simple prompts with `expected_contains`
- fixed IDs and stable categories

Estimated time: 5-8 hours

### Phase 2: Local Dry Runs and Validation

Location: local workstation

Run without GPU:

- syntax checks
- prompt schema validation
- parser tests using small sample logs
- report generation using fixture data
- config loading
- CLI help paths

Expected commands:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/50_parse_results.py --fixtures-only
```

Deliverables:

- scripts fail clearly when GPU/runtime artifacts are missing
- parser/report path works with fixture inputs
- `reports/final_report.md` template can be generated from partial data

Estimated time: 1.5-3 hours

### Phase 3: EC2 GPU Bootstrap and Environment Validation

Location: EC2 GPU instance

Start instance immediately before this phase:

```bash
aws ec2 start-instances --profile finetuning-local --region us-west-2 --instance-ids i-0c769a18f50fd1fe6
```

Connect with Session Manager.

Validate:

- SSM access works
- public egress works
- `nvidia-smi` sees L40S
- driver/CUDA versions are sane
- Docker is installed
- NVIDIA Container Toolkit works
- 250 GiB data volume is mounted
- Hugging Face and NGC secrets are retrievable from SSM
- Docker/model/cache directories are on the data volume if needed

Pull and verify NVIDIA container:

- `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`
- record digest actually pulled
- run `python3 -c "import tensorrt_llm"`

Run:

- `bash scripts/00_check_env.sh`

Stop instance immediately if setup fails or after validation if the next GPU phase will not run right away:

```bash
aws ec2 stop-instances --profile finetuning-local --region us-west-2 --instance-ids i-0c769a18f50fd1fe6
```

Estimated time: 1.5-3 hours

AWS compute estimate: `$3.36-$6.73`

### Phase 4: Baseline and Base TensorRT-LLM Benchmark

Location: EC2 GPU instance

Start instance only when ready to run the whole GPU sequence.

Run in order:

1. `scripts/00_check_env.sh`
2. `scripts/10_download_model.py`
3. `scripts/20_run_hf_baseline.sh`
4. `scripts/30_build_trtllm_engine.py`
5. `scripts/31_run_trtllm_smoke.py`
6. `scripts/32_prepare_trtllm_datasets.sh`
7. `scripts/33_run_trtllm_latency.sh`
8. `scripts/34_run_trtllm_throughput.sh`

Outputs:

- `results/env_report.json`
- `results/env_report.txt`
- `results/hf_baseline_raw.jsonl`
- `results/hf_baseline_summary.json`
- `artifacts/trtllm_engine_bf16_or_fp16/`
- `results/trtllm_latency_<dataset>.log`
- `results/trtllm_throughput_<dataset>.log`

Quick mode should benchmark at least two synthetic regimes and at least 20 measured requests.

Full mode should benchmark all four regimes and 100 measured requests where practical.

Estimated time:

- quick path: 3-5 hours
- full path: 5-8 hours

AWS compute estimate:

- quick path: `$6.73-$11.21`
- full path: `$11.21-$17.94`

### Phase 5: Quantization or KV-Cache Experiment

Location: EC2 GPU instance

Preferred attempt order:

1. FP8 KV cache
2. FP8 weight/activation quantization
3. INT8
4. INT4/AWQ/GPTQ
5. documented KV-cache/runtime configuration fallback

Run:

- `scripts/35_run_trtllm_quantized_or_kv_cache.sh`

Required datasets:

- `isl1024_osl128`
- `isl2048_osl256`

Outputs:

- separate artifact directory, for example `artifacts/trtllm_engine_fp8_kv/`
- separate benchmark logs
- clear notes if a quantization path is unsupported

Estimated time: 2-5 hours

AWS compute estimate: `$4.48-$11.21`

### Phase 6: Quality Regression and Profiling

Location: EC2 GPU instance for GPU-required generation/profile, then local workstation for parsing/reporting.

Run on EC2:

- `scripts/40_quality_regression.py` if it needs live TensorRT generation
- `scripts/60_profile_nsys.sh`

Nsight profile only steady-state inference, not download or engine build.

Outputs:

- `results/quality_regression.json`
- `results/quality_regression.md`
- `results/nsys/`

Stop EC2 immediately after this phase.

Estimated GPU time: 1-2 hours

AWS compute estimate: `$2.24-$4.48`

### Phase 7: Local Parsing, Report, and Final Verification

Location: local workstation

Run locally after stopping EC2:

- copy/sync small result files back from EC2 if needed
- `scripts/50_parse_results.py`
- inspect `results/summary.csv`
- inspect `results/summary.json`
- inspect `reports/final_report.md`

Verify acceptance criteria:

1. environment reports exist
2. HF baseline ran
3. TensorRT-LLM inference ran
4. at least two synthetic regimes benchmarked
5. latency and throughput logs captured
6. extra optimization attempted
7. `results/summary.csv` exists
8. `reports/final_report.md` exists
9. report states whether latency, throughput, or memory improved
10. report states quality regression status

Estimated time: 2-4 hours

## End-to-End Time Estimate

Best-case implementation time:

- 15-20 hands-on hours
- 2 focused working days
- 6-9 EC2 running hours

Expected implementation time:

- 22-32 hands-on hours
- 3-4 working days
- 9-15 EC2 running hours

Conservative/debug-heavy estimate:

- 35-50 hands-on hours
- 5-7 working days
- 15-25 EC2 running hours

The largest uncertainty is TensorRT-LLM version behavior: model support, CLI names, benchmark dataset tooling, and FP8/KV-cache options can change across releases.

## AWS Cost Estimate

Compute instance:

- `g6e.2xlarge`: `$2.24208/hr`

Estimated compute cost:

| Scenario | EC2 running hours | Compute cost |
| --- | ---: | ---: |
| Best case | 6-9 | `$13.45-$20.18` |
| Expected | 9-15 | `$20.18-$33.63` |
| Debug-heavy | 15-25 | `$33.63-$56.05` |

Storage:

- existing 350 GiB gp3 EBS is roughly `$28/month`
- no extra cost expected for local NVMe
- avoid repeated S3 uploads of large artifacts because versioning is enabled and no lifecycle policy exists

Network:

- pulling NGC container and Hugging Face model uses internet egress/ingress paths
- expected network cost should be small for this project, but model/container downloads can add time

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| TensorRT-LLM CLI/API differs from examples | scripts fail or require adaptation | implement version discovery and multiple command-path fallbacks |
| Qwen3 path unsupported or tokenizer issue | primary model blocked | fallback to Qwen2.5-1.5B, then TinyLlama, and document reason |
| root volume fills | Docker/model pulls fail | use/mount 250 GiB data volume for caches and artifacts |
| FP8 KV cache unsupported in selected path | optimization criterion at risk | try documented fallback runtime/KV-cache configuration and record exact reason |
| EC2 left running | cost overrun | enforce stop-on-failure in runner and manually verify instance state after GPU phases |
| mismatched benchmark comparisons | invalid speedup claims | compare only matching datasets and label synthetic TensorRT-only results separately |
| S3 versioning accumulates large files | storage cost growth | store large artifacts locally; upload only small logs/reports |
| launch template v3 lacks public IP | new instance cannot pull models/containers | use existing instance or version 2 behavior; do not launch replacement from v3 without NAT/endpoints |

## Implementation Order by File

First local implementation batch:

1. `data/prompts.jsonl`
2. `src/results_schema.py`
3. `src/prompt_loader.py`
4. `src/env_report.py`
5. `scripts/00_check_env.sh`
6. `scripts/50_parse_results.py`

Second local implementation batch:

1. `src/hf_benchmark.py`
2. `scripts/10_download_model.py`
3. `scripts/21_run_hf_baseline.py`
4. `scripts/20_run_hf_baseline.sh`
5. `src/quality_checks.py`
6. `scripts/40_quality_regression.py`

GPU/TensorRT implementation batch:

1. `scripts/02_start_trtllm_container.sh`
2. `scripts/30_build_trtllm_engine.py`
3. `scripts/31_run_trtllm_smoke.py`
4. `scripts/32_prepare_trtllm_datasets.sh`
5. `scripts/33_run_trtllm_latency.sh`
6. `scripts/34_run_trtllm_throughput.sh`
7. `scripts/35_run_trtllm_quantized_or_kv_cache.sh`
8. `scripts/60_profile_nsys.sh`

Final orchestration batch:

1. `scripts/90_run_all.sh`
2. final parser/report polish
3. README usage instructions
4. acceptance criteria checklist

## Stop Conditions

Stop and reassess before continuing if:

- EC2 cannot reach SSM after start.
- `nvidia-smi` does not show L40S.
- Docker cannot run with `--gpus all`.
- NGC TensorRT-LLM container cannot import `tensorrt_llm`.
- storage free space is below 100 GiB before engine build.
- the primary model fails and both fallback models also fail.
- a benchmark produces metrics that cannot be tied to a specific dataset and variant.

## Final Definition of Done

The project is complete when:

- all required scripts exist and run with clear resumable behavior
- EC2 GPU runs are captured and the instance is stopped afterward
- baseline and optimized results are present
- at least one extra optimization path was attempted and documented
- quality checks are summarized
- `results/summary.csv` and `results/summary.json` are generated
- `reports/final_report.md` explains measured latency, throughput, memory, quality, and limitations

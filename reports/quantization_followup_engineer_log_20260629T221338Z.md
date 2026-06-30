# TensorRT-LLM Quantization Follow-Up Engineer Log

Started: 2026-06-29 22:13:38 UTC

Objective:

- Try to complete workflow step 2, quantization, using an official NVIDIA TensorRT-LLM container different from the previous `1.3.0rc19` pre-release candidate.
- Preferred first probe: `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`.
- Determine whether FP8 KV cache, FP8 quantization, AWQ, GPTQ, INT8, or INT4 paths are practically available for the small model benchmark.
- Use EC2 only for GPU/container inspection and benchmark work.
- Stop EC2 after the GPU phase or if the probe fails.

Safety controls:

- Do local preparation first.
- Verify AWS credentials and instance state before starting EC2.
- Schedule an in-instance shutdown guard immediately after the instance is reachable.
- Upload/sync probe results before shutdown where possible.
- Do not claim quantization speedup unless a quantized engine is built and measured on matching datasets.

## Action Log

- 2026-06-29 22:13:38 UTC: Created this engineer log before touching AWS infrastructure.
- 2026-06-29 22:14 UTC: Read `configs/benchmark_config.yaml`; target remains `g6e.2xlarge` instance `i-0c769a18f50fd1fe6` in `us-west-2`, using profile `finetuning-local`.
- 2026-06-29 22:14 UTC: Checked AWS identity with `aws sts get-caller-identity --profile finetuning-local`; AWS CLI returned `Your session has expired. Please reauthenticate using 'aws login'`.
- 2026-06-29 22:14 UTC: Did not start EC2 because credentials are expired. Continuing with local preparation only.
- 2026-06-29 22:18 UTC: Added `scripts/37_probe_trtllm_quantization.py` to inspect TensorRT-LLM package versions, CLI flags, Python APIs, quantization example paths, and optionally run an FP8 KV-cache LLM API smoke test.
- 2026-06-29 22:19 UTC: Added instance-side script `scripts/aws/run_ec2_quantization_probe.sh`. It pulls `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`, schedules an in-instance shutdown guard, runs the quantization probe in the container, syncs result/report artifacts to S3, and shuts down the EC2 instance through a trap.
- 2026-06-29 22:20 UTC: Added local launcher `scripts/aws/start_ec2_quantization_probe.sh`. It checks AWS identity, starts `i-0c769a18f50fd1fe6` if stopped, waits for SSM, sends the probe command, polls SSM status, then issues a best-effort stop and waits for the stopped state.
- 2026-06-29 22:20 UTC: Marked new scripts executable and ran local syntax checks:
  - `bash -n scripts/aws/run_ec2_quantization_probe.sh`
  - `bash -n scripts/aws/start_ec2_quantization_probe.sh`
  - `.venv/bin/python -m py_compile scripts/37_probe_trtllm_quantization.py`
- 2026-06-29 22:20 UTC: Local preparation is complete. AWS re-authentication is required before the EC2 probe can be launched.
- 2026-06-29 22:21 UTC: Updated `.gitignore` to allow `reports/quantization_followup_engineer_log_*.md` so this probe log can be committed with the follow-up scripts.
- 2026-06-29 22:25 UTC: AWS re-authentication verified. Instance `i-0c769a18f50fd1fe6` is currently `stopped`.
- 2026-06-29 22:25 UTC: Patched `scripts/aws/start_ec2_quantization_probe.sh` to create a source bundle from the current local repo, upload it to the existing project S3 prefix, download it on EC2, and run the probe from that fresh source tree. This avoids running stale scripts already present on the instance.
- 2026-06-29 22:20 UTC: Launched `RUN_ID=quant_probe_20260629T222002Z`. Source bundle uploaded to `s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/source-bundles/model-optimization-quant_probe_20260629T222002Z.tgz`.
- 2026-06-29 22:20 UTC: EC2 start failed with `InsufficientInstanceCapacity` for existing `g6e.2xlarge` instance in `us-west-2c`.
- 2026-06-29 22:20 UTC: Verified instance state after failed start: `stopped`. No GPU work started and no shutdown guard was needed because the instance never entered `running`.
- 2026-06-29 22:20 UTC: Retried the same `RUN_ID=quant_probe_20260629T222002Z` against the same stopped instance. Start failed again with `InsufficientInstanceCapacity`.
- 2026-06-29 22:21 UTC: Verified instance state after retry: `stopped`.
- 2026-06-29 22:21 UTC: Checked instance type offerings in `us-west-2`. AWS lists `g6e.2xlarge`, `g6e.4xlarge`, and `p5.4xlarge` offerings across multiple AZs, including `g6e.4xlarge` in `us-west-2c`, but offerings do not guarantee immediate capacity.
- 2026-06-29 22:21 UTC: Pausing before changing instance type or launching a replacement because that changes cost/topology. Current state is safe: no GPU instance is running.
- 2026-06-29 22:49 UTC: User reported the EC2 instance was starting. Verified configured instance `i-0c769a18f50fd1fe6` is `running`.
- 2026-06-29 22:49 UTC: Launched probe runner with `RUN_ID=quant_probe_20260629T224957Z`; uploaded source bundle to S3 and waited for EC2/SSM readiness.
- 2026-06-29 22:51 UTC: SSM was online, but the local launcher failed before sending the probe due to AWS CLI parsing of a multiline `--parameters commands=...` argument.
- 2026-06-29 22:51 UTC: Immediately sent emergency shutdown guard through SSM command `8c95564a-d9cb-45d7-bef5-dc37d3bc0718`. Verified command success; shutdown scheduled for 2026-06-30 01:51:44 UTC.
- 2026-06-29 22:52 UTC: Patched `scripts/aws/start_ec2_quantization_probe.sh` to send SSM command parameters through a JSON parameter file instead of raw multiline CLI text.
- 2026-06-29 22:52 UTC: Patched `scripts/aws/run_ec2_quantization_probe.sh` to cancel any existing shutdown request before scheduling its own guard. This lets the probe replace the emergency guard cleanly.
- 2026-06-29 22:52 UTC: Re-ran shell syntax checks for both patched scripts; both passed.
- 2026-06-29 22:52 UTC: Re-launched with `RUN_ID=quant_probe_20260629T225250Z`. SSM accepted the command, but remote execution failed immediately with `/var/lib/amazon/ssm/.../_script.sh: 1: set: Illegal option -o pipefail`.
- 2026-06-29 22:58 UTC: Launcher issued a best-effort EC2 stop after the failed SSM command and waited until the instance reached `stopped`.
- 2026-06-29 23:00 UTC: Confirmed instance `i-0c769a18f50fd1fe6` is `stopped`; no GPU instance is running while applying the next fix.
- 2026-06-29 23:00 UTC: Root cause identified: AWS Systems Manager runs `AWS-RunShellScript` command strings under `/bin/sh`, so Bash-only `set -o pipefail` failed before the downloaded repo script could start.
- 2026-06-29 23:00 UTC: Patched `scripts/aws/start_ec2_quantization_probe.sh` to send two SSM commands: first write the real script to `/tmp/run_trtllm_quant_probe.sh` via a heredoc, then run it explicitly with `bash`.
- 2026-06-29 23:00 UTC: Re-ran `bash -n scripts/aws/start_ec2_quantization_probe.sh` and `bash -n scripts/aws/run_ec2_quantization_probe.sh`; both passed.
- 2026-06-29 23:01 UTC: Retried the existing stopped `g6e.2xlarge`; EC2 again returned `InsufficientInstanceCapacity` in `us-west-2c`. The instance remained `stopped`, and no SSM probe command was sent.
- 2026-06-29 23:03 UTC: Confirmed `g6e.2xlarge` and `g6e.4xlarge` are offered in all four `us-west-2` AZs, and the current VPC has public subnets in `us-west-2a`, `us-west-2b`, `us-west-2c`, and `us-west-2d`.
- 2026-06-29 23:08 UTC: Added `scripts/aws/start_temp_ec2_quantization_probe.sh` to launch a temporary same-AMI `g6e.2xlarge` in alternate AZs, run the S3-backed quantization probe, download S3 results locally when available, and terminate the temporary instance through a local cleanup trap.
- 2026-06-29 23:08 UTC: Validated shell syntax for `scripts/aws/start_temp_ec2_quantization_probe.sh`, `scripts/aws/start_ec2_quantization_probe.sh`, and `scripts/aws/run_ec2_quantization_probe.sh`.
- 2026-06-29 23:06 UTC: Temporary launcher tried `us-west-2b`; AWS returned `InsufficientInstanceCapacity`. It then launched temporary `g6e.2xlarge` instance `i-006c3ad4d6cf1e049` in `us-west-2a`.
- 2026-06-29 23:09 UTC: SSM command started on `i-006c3ad4d6cf1e049`; shutdown guard scheduled for 2026-06-30 02:09:35 UTC. Docker pull of `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` began.
- 2026-06-29 23:16 UTC: Docker pull completed. Captured digest from pull output: `sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9`.
- 2026-06-29 23:19 UTC: Probe failed after model download because the wrapper passed `--output $PROBE_DIR/model_download.json`; `scripts/10_download_model.py` treats this as an output directory, so the wrapper tried to read a directory as JSON.
- 2026-06-29 23:24 UTC: Local cleanup trap terminated temporary instance `i-006c3ad4d6cf1e049`; verified state `terminated`.
- 2026-06-29 23:26 UTC: Patched `scripts/aws/run_ec2_quantization_probe.sh` to call `10_download_model.py --output-dir "$PROBE_DIR/model_download"` and read `$PROBE_DIR/model_download/model_download.json`. Re-ran shell syntax checks successfully.
- 2026-06-29 23:26 UTC: Re-launched temporary probe. AWS accepted `g6e.2xlarge` in `us-west-2b` and created temporary instance `i-05900920ee0c9d687`.
- 2026-06-29 23:29 UTC: SSM command started on `i-05900920ee0c9d687`; shutdown guard scheduled for 2026-06-30 02:29:31 UTC.
- 2026-06-29 23:36 UTC: Docker pull completed for `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`; digest recorded as `sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9`.
- 2026-06-29 23:38 UTC: Model download completed for `Qwen/Qwen3-1.7B`; snapshot path `/mnt/workspace/huggingface-cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- 2026-06-29 23:40 UTC: Quantization probe completed. TensorRT-LLM 1.2.1 exposes `KvCacheConfig(dtype="fp8")`, but the FP8 KV smoke test failed with an FMHA kernel assertion for this Qwen3/L40S/container combination.
- 2026-06-29 23:41 UTC: SSM command reported `Success`; S3 artifacts were downloaded locally under `results/quant_probe_temp_20260629T232604Z_s3`.
- 2026-06-29 23:46 UTC: Local cleanup trap terminated temporary instance `i-05900920ee0c9d687`; verified no GPU instances remained in `pending`, `running`, `stopping`, or `shutting-down` states.
- 2026-06-29 23:49 UTC: Patched `scripts/35_run_trtllm_quantized_or_kv_cache.sh` and `scripts/37_probe_trtllm_quantization.py` to recognize TensorRT-LLM 1.2.1's actual runtime flag spelling: `--kv_cache_free_gpu_mem_fraction`.
- 2026-06-29 23:50 UTC: Saved concise follow-up report to `reports/quantization_followup_20260629T232604Z.md` and machine-readable summary to `results/quantization_followup_summary_20260629T232604Z.json`.

## Resume Command After AWS Re-Authentication

After running `aws login` for profile `finetuning-local`, launch the controlled probe with:

```bash
AWS_PROFILE=finetuning-local \
AWS_REGION=us-west-2 \
INSTANCE_ID=i-0c769a18f50fd1fe6 \
TRTLLM_TEST_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
MODEL=Qwen/Qwen3-1.7B \
FALLBACK_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
bash scripts/aws/start_ec2_quantization_probe.sh
```

Expected remote output prefix defaults to:

```text
s3://trtllm-optimization-benchmark/<RUN_ID>
```

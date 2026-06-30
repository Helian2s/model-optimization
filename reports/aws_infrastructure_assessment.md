# AWS Infrastructure Assessment

Captured from read-only AWS CLI inventory on 2026-06-29 UTC / 2026-06-28 America/Denver.

This file intentionally records resource identifiers that are useful for the benchmark work. It does not contain secret values, API keys, or decrypted SSM parameters.

## Recommendation

Use the existing stopped EC2 instance:

- Instance ID: `i-0c769a18f50fd1fe6`
- Instance type: `g6e.2xlarge`
- Availability zone: `us-west-2c`
- Access method: AWS Systems Manager Session Manager
- AMI: `ami-002a4c6455e1ec9cb`
- AMI name: `Deep Learning Base AMI with Single CUDA (Ubuntu 24.04) 20260619`

This is the best current target because it already exists, matches the task preference, has the right GPU class, and fits the current EC2 quota.

## NVIDIA-First Platform Requirement

Implementation should use NVIDIA platform components wherever practical:

- NGC TensorRT-LLM release container for optimized inference.
- NVIDIA Container Toolkit for Docker GPU access.
- TensorRT-LLM APIs and CLIs before custom runtime code.
- Nsight Systems for steady-state inference profiling.
- NVML and `nvidia-smi` for GPU telemetry and environment capture.
- TensorRT/TensorRT-LLM metadata in engine build logs and final results.

The Hugging Face Transformers path is retained as the required baseline only. AWS should be used as raw EC2 infrastructure, not as an alternate ML runtime. Do not use SageMaker, Inferentia, Trainium, Neuron, or non-NVIDIA inference stacks unless a future task explicitly asks for a separate comparison.

## Local-First Execution Requirement

Use the local CPU-only workstation as much as possible. EC2 should be used only when GPU access or target EC2 environment capture is required.

Run locally:

- repository edits
- prompt and config preparation
- AWS account inventory and planning
- result parsing
- CSV/JSON/Markdown report generation
- schema validation and lightweight tests

Run on EC2:

- `nvidia-smi` and GPU environment capture
- Hugging Face baseline timing on GPU
- TensorRT-LLM container startup and validation
- TensorRT-LLM engine builds
- TensorRT-LLM latency/throughput benchmarks
- FP8/quantization/KV-cache experiments
- Nsight Systems profiling

Start `i-0c769a18f50fd1fe6` only immediately before these GPU phases. Stop it immediately after the GPU phase completes or fails. All preparation that can run locally should happen before starting EC2, and all parsing/reporting should happen after stopping EC2.

This policy reduces AWS spend, shortens GPU sessions, and keeps repeated non-GPU iteration local.

## EC2 and GPU

The stopped instance is tagged for the existing lab environment:

- `Name=ft-exp00`
- `Project=NCP-GENL`
- `Experiment=FT-EXP-00`
- `Environment=Lab`
- `Owner=Val`
- `AutoStop=true`

Hardware characteristics for `g6e.2xlarge` in this account/region:

- GPU: 1x NVIDIA L40S
- GPU memory reported by EC2 metadata: 45,776 MiB
- vCPU: 8
- Host memory: 64 GiB
- Local instance storage: 1x 450 GB NVMe SSD
- Network: up to 20 Gbit

The L40S is suitable for this benchmark. It has enough memory for the selected 1-2B parameter models and supports FP8-class experiments, making FP8 KV-cache the preferred first additional optimization attempt.

## Storage

Attached EBS volumes:

- Root volume: `vol-0c4703d8ee8b9e1e7`
  - 100 GiB gp3
  - encrypted
  - delete on termination: true
  - attached as `/dev/sda1`
- Data volume: `vol-09a2fd5650d8c763e`
  - 250 GiB gp3
  - encrypted
  - delete on termination: false
  - attached as `/dev/sdf`

The 250 GiB data volume meets the minimum task requirement. Before running heavy container/model workflows, verify it is mounted and place Docker cache, Hugging Face cache, model downloads, and TensorRT-LLM engines there where practical. If engine builds or multiple model variants make space tight, resizing this volume to 500 GiB is the cleanest upgrade.

No self-owned EBS snapshots and no Elastic IPs were found during the read-only scan.

## Network and Access

The instance is intentionally SSM-only:

- No SSH key is attached to the instance.
- Security group: `sg-0797f3b8520d4efa9`
- Security group name: `ft-exp00-ssm-only`
- Ingress: none
- Egress: all outbound to `0.0.0.0/0`
- Subnet: `subnet-0e52a0c3ae6a086e8`
- VPC: `vpc-0480da876d161ad22`
- Subnet maps public IPs on launch.
- Main route table has `0.0.0.0/0` through internet gateway `igw-0e047616498b2b0fa`.

This is compatible with Session Manager and internet-based model/container pulls as long as the instance has public egress when running.

Important caveat: launch template version 3 disables public IP assignment, but no NAT gateway or SSM VPC endpoints were observed. A new instance launched from that version would likely lose outbound internet access and may not be reachable through SSM. The existing stopped instance was launched from version 2, which associates a public IP.

## IAM and Secrets

Instance profile:

- `FinetuningGpuInstanceRole`

Attached policies:

- `AmazonSSMManagedInstanceCore`
- custom `FinetuningGpuS3Access`

Inline policy:

- `FinetuningGpuParameterAccess`

Expected SSM SecureString parameters exist:

- `/finetuning/huggingface/token`
- `/finetuning/ngc/api-key`

These should let the instance retrieve Hugging Face and NGC credentials without committing secrets to the repository.

## S3

One bucket was found:

- `finetuning-lab-1-037678282394-us-west-2-an`
- Region: `us-west-2`
- Encryption: SSE-S3 (`AES256`) with bucket key enabled
- Public access block: enabled
- Versioning: enabled
- Lifecycle policy: none

Use this bucket only for small logs, reports, and result archives unless a lifecycle policy is added. Because versioning is enabled and no lifecycle cleanup exists, large model files or TensorRT engines would accumulate cost quickly if uploaded repeatedly.

The current instance role S3 policy allows reads/writes under existing `finetuning/` prefixes. If this benchmark needs S3 sync, either use a subdirectory under `finetuning/results/` or add a dedicated `trtllm-benchmark/` prefix to the policy later.

## Quotas

Relevant quotas in `us-west-2`:

- Running On-Demand G and VT instances: 8 vCPU
- Running On-Demand P instances: 0 vCPU
- G and VT Spot Instance Requests: 0
- P Spot Instance Requests: 0

Implications:

- One `g6e.2xlarge` is allowed and uses the full current G/VT quota.
- `g6e.4xlarge` is blocked because it needs 16 G/VT vCPU.
- `p5.4xlarge` is blocked because P-instance quota is 0.
- GPU Spot is currently blocked.

Do not plan around `g6e.4xlarge`, `p5.4xlarge`, or Spot unless quotas are increased first.

## Live Pricing Snapshot

Prices pulled from the AWS Pricing API for Linux on-demand instances in `us-west-2`:

| Instance type | GPU | Price |
| --- | --- | ---: |
| `g6e.2xlarge` | 1x L40S 48 GB class | `$2.24208/hr` |
| `g6e.4xlarge` | 1x L40S 48 GB class | `$3.00424/hr` |
| `p5.4xlarge` | 1x H100 80 GB | `$6.88/hr` |

gp3 pricing snapshot for `us-west-2`:

- Storage: `$0.08/GB-month`
- Additional IOPS: `$0.005/provisioned IOPS-month`
- Additional throughput: `$0.04/provisioned MiBps-month`

The existing 350 GiB of gp3 storage is roughly `$28/month` before any extra provisioned IOPS/throughput.

The account has a project budget named `NCP-GENL-Monthly-Cost` with a `$100/month` limit. At the `g6e.2xlarge` rate, this leaves roughly 40-45 compute hours per month after storage and incidental costs.

## Launch Template Caveat

Launch template:

- ID: `lt-00a77678a8796caf5`
- Name: `ft-exp00-g6e-v1`
- Default version: 2
- Latest version: 3

Version 2 associates a public IP. Version 3 disables public IP assignment.

Both versions reference this user-data bootstrap path:

`s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/source-bundles/bootstrap_instance.sh`

That object was not present during the inventory check. This is not a blocker for starting the existing stopped instance, but it is a blocker/risk for launching replacement instances from the template.

## Execution Plan

1. Start the existing `g6e.2xlarge` only when ready to validate runtime setup.
2. Connect with Session Manager, not SSH.
3. Verify `nvidia-smi`, Docker, and NVIDIA Container Toolkit.
4. Verify the 250 GiB volume mount and move caches/artifacts there if needed.
5. Pull `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19` and record the digest actually pulled.
6. Run the repository environment capture before any benchmark.
7. Keep large generated artifacts local unless an S3 lifecycle policy is added.
8. Stop the instance immediately after benchmark runs.

## Benchmark Design Notes

The final report must not compare mismatched workloads. Hugging Face prompt-set baseline results should only be compared with TensorRT-LLM results for the same prompt set and output limits. Synthetic TensorRT-LLM datasets such as `isl1024_osl128` require either matching Hugging Face synthetic baselines or clearly separate TensorRT-only latency/throughput reporting.

Recommended first optimization experiment:

1. Base TensorRT-LLM BF16/FP16 engine.
2. FP8 KV-cache variant.
3. If FP8 KV-cache is unsupported by the selected TensorRT-LLM version/model path, document the failure and run a KV-cache/runtime configuration fallback.

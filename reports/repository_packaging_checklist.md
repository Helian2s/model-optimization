# Repository Packaging Checklist

Generated locally on 2026-06-29 after the measured GPU benchmark runs. No EC2 instance was started for this packaging pass.

## Secret Scan

Masked scans were run for common high-risk patterns:

- Hugging Face token literals such as `hf_...`
- AWS access key IDs and session/secret key names
- private key headers
- bearer/authorization strings
- generic password, secret, token, and credential labels

Result:

- No literal AWS or Hugging Face credential values were found in the files intended for commit.
- Matches were code references to environment variables such as `HF_TOKEN`, documentation about credential handling, or prompt text mentioning credentials.
- Earlier GPU-session notes still recommend rotating the Hugging Face token because an early SSM command path may have exposed it before the Docker launcher was patched.

## Large Local Artifacts

Large benchmark artifacts are intentionally kept out of git:

- TensorRT-LLM engines under `artifacts/`
- raw benchmark logs under ignored `results/` run directories
- latency iteration JSONL files such as `*_iterations.jsonl`
- Nsight Systems reports such as `*.nsys-rep`
- local virtual environment under `.venv/`
- model, Hugging Face, and container caches

The important lightweight deliverables are commit-ready:

- `results/summary.csv`
- `results/summary.json`
- `reports/final_report.md`
- project source under `scripts/` and `src/`
- prompt/config files under `data/` and `configs/`
- planning/history reports explicitly allowed by `.gitignore`

## Current Final Result State

- EC2 was previously verified stopped before AWS credentials expired.
- Local authoritative report: `reports/final_report.md`
- Local authoritative summary files:
  - `results/summary.csv`
  - `results/summary.json`
- Current conclusion:
  - TensorRT-LLM improved measured latency and throughput on length-matched synthetic workloads.
  - Memory usage did not improve because the TensorRT-LLM run reserved a large paged KV-cache pool.
  - No TensorRT-specific quality regression was detected by the simple deterministic checks.

## Before Publishing

- Rotate the Hugging Face token referenced by the EC2/SSM workflow.
- Refresh AWS credentials only if uploading final artifacts to S3 or verifying EC2 state again.
- Keep raw `results/<run-id>/` folders local or in S3 unless explicitly needed in a separate artifact bundle.
- Run `git status --short` and review all tracked/untracked files before committing.

# Remaining GPU Run Summary

This run completed the remaining measured phases for the TensorRT-LLM benchmark project.

## AWS Safety State

- Instance: `i-0c769a18f50fd1fe6`
- Instance type: `g6e.2xlarge`
- Region: `us-west-2`
- Main remaining run SSM command: `90430080-0f74-498d-ac88-2c7ea4ab0eec`
- Follow-up opt/profile SSM command: `a3907a68-3d0c-48bc-98c4-227392256533`
- Main remaining run duration: `28m13.936s`
- Follow-up opt/profile run duration: `3m4.438s`
- EC2 final state was verified as `stopped` before AWS credentials expired.

## Local Result Folders

- `results/remaining_20260629T043847Z/`
- `reports/remaining_20260629T043847Z/`
- `results/opt_profile_20260629T052000Z/`
- `reports/opt_profile_20260629T052000Z/`

Top-level acceptance files were refreshed from the corrected remaining-run result:

- `results/summary.csv`
- `results/summary.json`
- `reports/final_report.md`

## Key Results

HF synthetic baseline:

- `isl1024_osl128`: avg latency `3415.315 ms`, generation throughput `37.473 tok/s`
- `isl2048_osl256`: avg latency `6855.528 ms`, generation throughput `37.339 tok/s`

TensorRT-LLM synthetic latency:

- `isl1024_osl128`: avg latency `829.752 ms`, generation throughput `154.255 tok/s`
- `isl2048_osl256`: avg latency `1709.091 ms`, generation throughput `149.783 tok/s`

TensorRT-LLM synthetic throughput:

- `isl1024_osl128`: request throughput `5.924 req/s`, generation throughput `758.247 tok/s`
- `isl2048_osl256`: request throughput `2.585 req/s`, generation throughput `661.699 tok/s`

Speedups versus length-matched HF synthetic baseline:

- Latency mode, `isl1024_osl128`: `4.116x` latency speedup, `4.116x` generation throughput speedup
- Latency mode, `isl2048_osl256`: `4.011x` latency speedup, `4.011x` generation throughput speedup
- Throughput mode, `isl1024_osl128`: `20.235x` generation throughput speedup
- Throughput mode, `isl2048_osl256`: `17.721x` generation throughput speedup

Prompt-set TensorRT output:

- Completed 50 prompts using the `examples` backend.
- This path is marked quality-only because it includes per-prompt process/engine startup.
- It is not used for speedup claims.

Quality regression:

- HF baseline: `12/25` deterministic checks passed.
- TensorRT-LLM base: `12/25` deterministic checks passed.
- No TensorRT-specific regression was detected by these simple checks because pass/fail counts matched.
- Remaining failures are mostly strict expected-substring issues and model behavior on very short deterministic prompts.

Optimization/KV-cache attempt:

- Status: `fp8_kv_flag_not_exposed_and_runtime_kv_fraction_flag_not_exposed`
- The TensorRT-LLM 1.3.0rc19 CLI in this container did not expose the FP8 KV-cache build flag or runtime KV fraction flag through the checked `trtllm-build` / `trtllm-bench latency` help output.
- The profile log documents that the base engine already runs with `paged_kv_cache=True`, `tokens_per_block=32`, and about `35.35 GiB` allocated for max tokens in paged KV cache.

Nsight Systems:

- Captured `.nsys-rep`: `results/opt_profile_20260629T052000Z/nsys/trtllm_steady_state.nsys-rep`
- Summary: `results/opt_profile_20260629T052000Z/nsys/profile_summary.md`
- The profile command returned nonzero because the target benchmark re-parented worker processes during shutdown, but the report file was generated.

## S3 Upload Note

Initial S3 sync from EC2 completed for both GPU runs. After local report correction, AWS credentials expired locally, so the latest corrected top-level files are authoritative in the repository:

- `reports/final_report.md`
- `results/summary.csv`
- `results/summary.json`


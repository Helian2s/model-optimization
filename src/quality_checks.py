"""Simple deterministic quality regression checks."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.prompt_loader import PromptRecord, deterministic_prompts, load_prompts
from src.results_schema import read_jsonl, write_json


REFUSAL_PATTERNS = [
    re.compile(r"\bi cannot\b", re.IGNORECASE),
    re.compile(r"\bi can't\b", re.IGNORECASE),
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bi am unable\b", re.IGNORECASE),
]

SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(SUBSCRIPT_TRANSLATION)
    normalized = normalized.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"^[`'\"\s]+|[`'\"\s]+$", "", normalized)
    return normalized


def is_empty_or_refusal(value: str) -> bool:
    if not value or not value.strip():
        return True
    return any(pattern.search(value) for pattern in REFUSAL_PATTERNS)


def load_output_map(path: str | Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    output_map: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = record.get("id")
        if record_id is None:
            continue
        output_map[str(record_id)] = record
    return output_map


def output_text(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    for key in ["output_text", "text", "generated_text", "output"]:
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def check_prompt(prompt: PromptRecord, variant_record: dict[str, Any] | None, baseline_text: str | None = None) -> dict:
    text = output_text(variant_record)
    normalized = normalize_text(text)
    expected_contains = normalize_text(prompt.expected_contains or "")
    expected_exact = normalize_text(prompt.expected_exact or "")
    contains_pass = True
    exact_pass = True
    if expected_contains:
        contains_pass = expected_contains in normalized
    if expected_exact:
        exact_pass = normalized == expected_exact
    empty_or_refusal = is_empty_or_refusal(text)
    baseline_norm = normalize_text(baseline_text or "")
    length_diff = None
    normalized_matches_baseline = None
    if baseline_text is not None:
        length_diff = len(text) - len(baseline_text)
        normalized_matches_baseline = normalized == baseline_norm
    passed = contains_pass and exact_pass and not empty_or_refusal
    return {
        "id": prompt.id,
        "category": prompt.category,
        "expected_contains": prompt.expected_contains,
        "expected_exact": prompt.expected_exact,
        "output_text": text,
        "contains_pass": contains_pass,
        "exact_pass": exact_pass,
        "empty_or_refusal": empty_or_refusal,
        "normalized_matches_baseline": normalized_matches_baseline,
        "output_length_chars": len(text),
        "baseline_length_diff_chars": length_diff,
        "passed": passed,
    }


def evaluate_variant(
    *,
    variant: str,
    prompts: list[PromptRecord],
    outputs: dict[str, dict[str, Any]],
    baseline_outputs: dict[str, dict[str, Any]] | None = None,
    baseline_checks: dict[str, dict[str, Any]] | None = None,
) -> dict:
    checks = []
    for prompt in prompts:
        baseline_text = None
        if baseline_outputs is not None:
            baseline_text = output_text(baseline_outputs.get(prompt.id))
        checks.append(check_prompt(prompt, outputs.get(prompt.id), baseline_text=baseline_text))
    total = len(checks)
    passed = sum(1 for check in checks if check["passed"])
    empty_or_refusal = sum(1 for check in checks if check["empty_or_refusal"])
    failed_ids = [check["id"] for check in checks if not check["passed"]]
    result = {
        "variant": variant,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": None if total == 0 else passed / total,
        "empty_or_refusal_count": empty_or_refusal,
        "failed_ids": failed_ids,
        "checks": checks,
    }
    if baseline_checks is not None:
        baseline_passed = sum(1 for check in baseline_checks.values() if check["passed"])
        newly_failed = [
            check["id"]
            for check in checks
            if baseline_checks.get(check["id"], {}).get("passed") and not check["passed"]
        ]
        recovered = [
            check["id"]
            for check in checks
            if baseline_checks.get(check["id"]) and not baseline_checks[check["id"]]["passed"] and check["passed"]
        ]
        newly_empty = [
            check["id"]
            for check in checks
            if not baseline_checks.get(check["id"], {}).get("empty_or_refusal") and check["empty_or_refusal"]
        ]
        changed = [
            check["id"]
            for check in checks
            if check.get("normalized_matches_baseline") is False
        ]
        match_count = sum(1 for check in checks if check.get("normalized_matches_baseline") is True)
        result.update(
            {
                "baseline_variant": "hf_baseline",
                "baseline_passed": baseline_passed,
                "pass_delta_vs_baseline": passed - baseline_passed,
                "newly_failed_vs_baseline_count": len(newly_failed),
                "newly_failed_vs_baseline_ids": newly_failed,
                "recovered_vs_baseline_count": len(recovered),
                "recovered_vs_baseline_ids": recovered,
                "newly_empty_or_refusal_vs_baseline_count": len(newly_empty),
                "newly_empty_or_refusal_vs_baseline_ids": newly_empty,
                "normalized_match_count_vs_baseline": match_count,
                "normalized_match_rate_vs_baseline": None if total == 0 else match_count / total,
                "changed_output_count_vs_baseline": len(changed),
                "changed_output_ids_vs_baseline": changed,
                "quality_regression_detected": bool(newly_failed or newly_empty),
            }
        )
    return result


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Quality Regression",
        "",
        "| Variant | Total | Passed | Failed | Pass rate | Empty/refusal |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in result["variants"]:
        pass_rate = variant["pass_rate"]
        rate_text = "n/a" if pass_rate is None else f"{pass_rate:.2%}"
        pass_delta = variant.get("pass_delta_vs_baseline")
        new_failures = variant.get("newly_failed_vs_baseline_count")
        match_rate = variant.get("normalized_match_rate_vs_baseline")
        pass_delta_text = "" if pass_delta is None else f"{pass_delta:+d}"
        new_failures_text = "" if new_failures is None else str(new_failures)
        match_rate_text = "" if match_rate is None else f"{match_rate:.2%}"
        lines.append(
            f"| {variant['variant']} | {variant['total']} | {variant['passed']} | "
            f"{variant['failed']} | {rate_text} | {variant['empty_or_refusal_count']} | "
            f"{pass_delta_text} | {new_failures_text} | {match_rate_text} |"
        )
    lines[2] = (
        "| Variant | Total | Passed | Failed | Strict pass rate | Empty/refusal | "
        "Pass delta vs HF | New failures vs HF | Output match vs HF |"
    )
    lines[3] = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines.extend(["", "## Regression Against HF Baseline", ""])
    baseline_compared = [variant for variant in result["variants"] if "baseline_variant" in variant]
    if not baseline_compared:
        lines.append("No HF baseline comparison was available.")
    else:
        for variant in baseline_compared:
            if variant.get("quality_regression_detected"):
                lines.append(
                    f"- `{variant['variant']}` introduced "
                    f"{variant['newly_failed_vs_baseline_count']} newly failed checks vs HF baseline: "
                    f"{', '.join(variant['newly_failed_vs_baseline_ids']) or 'none'}."
                )
            else:
                lines.append(
                    f"- `{variant['variant']}` introduced no newly failed deterministic checks vs HF baseline. "
                    f"Normalized output match rate: {variant['normalized_match_rate_vs_baseline']:.2%}."
                )
    lines.extend(["", "## Failures", ""])
    failures = [
        (variant["variant"], check)
        for variant in result["variants"]
        for check in variant["checks"]
        if not check["passed"]
    ]
    if not failures:
        lines.append("No deterministic quality failures detected by the simple checks.")
    else:
        for variant_name, check in failures:
            lines.append(
                f"- `{variant_name}` `{check['id']}` failed: "
                f"contains_pass={check['contains_pass']} exact_pass={check['exact_pass']} "
                f"empty_or_refusal={check['empty_or_refusal']}"
            )
    lines.append("")
    return "\n".join(lines)


def run_quality_regression(
    *,
    prompts_path: str | Path,
    variant_files: dict[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    prompts = deterministic_prompts(load_prompts(prompts_path))
    loaded = {
        variant: load_output_map(path)
        for variant, path in variant_files.items()
        if path and Path(path).exists()
    }
    baseline = loaded.get("hf_baseline")
    variants = []
    baseline_checks = None
    if baseline is not None:
        baseline_result = evaluate_variant(variant="hf_baseline", prompts=prompts, outputs=baseline)
        variants.append(baseline_result)
        baseline_checks = {check["id"]: check for check in baseline_result["checks"]}
    for variant, outputs in loaded.items():
        if variant == "hf_baseline":
            continue
        variants.append(
            evaluate_variant(
                variant=variant,
                prompts=prompts,
                outputs=outputs,
                baseline_outputs=baseline,
                baseline_checks=baseline_checks,
            )
        )
    missing = {
        variant: str(path)
        for variant, path in variant_files.items()
        if not path or not Path(path).exists()
    }
    result = {
        "prompt_file": str(prompts_path),
        "deterministic_prompt_count": len(prompts),
        "missing_variant_files": missing,
        "variants": variants,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(output_path / "quality_regression.json", result)
    (output_path / "quality_regression.md").write_text(markdown_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic quality regression checks.")
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--hf", default="results/hf_baseline_raw.jsonl")
    parser.add_argument("--trt", default="results/trtllm_outputs_raw.jsonl")
    parser.add_argument("--optimized", default="results/trtllm_optimized_outputs_raw.jsonl")
    args = parser.parse_args()

    result = run_quality_regression(
        prompts_path=args.prompts,
        output_dir=args.output_dir,
        variant_files={
            "hf_baseline": args.hf,
            "trtllm_base": args.trt,
            "trtllm_optimized": args.optimized,
        },
    )
    print(json.dumps({k: v for k, v in result.items() if k != "variants"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

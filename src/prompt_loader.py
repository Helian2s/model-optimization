"""Prompt dataset loading and validation helpers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PromptRecord:
    id: str
    category: str
    prompt: str
    max_new_tokens: int
    expected_contains: str | None = None
    expected_exact: str | None = None

    @classmethod
    def from_json(cls, data: dict, path: Path, line_number: int) -> "PromptRecord":
        required = ["id", "category", "prompt", "max_new_tokens"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path}:{line_number} missing required keys: {', '.join(missing)}")

        record = cls(
            id=str(data["id"]),
            category=str(data["category"]),
            prompt=str(data["prompt"]),
            max_new_tokens=int(data["max_new_tokens"]),
            expected_contains=(None if data.get("expected_contains") is None else str(data["expected_contains"])),
            expected_exact=(None if data.get("expected_exact") is None else str(data["expected_exact"])),
        )
        record.validate(path, line_number)
        return record

    def validate(self, path: Path, line_number: int) -> None:
        if not self.id:
            raise ValueError(f"{path}:{line_number} id must not be empty")
        if not self.category:
            raise ValueError(f"{path}:{line_number} category must not be empty")
        if not self.prompt:
            raise ValueError(f"{path}:{line_number} prompt must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError(f"{path}:{line_number} max_new_tokens must be positive")

    def to_generation_prompt(self) -> str:
        return self.prompt

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.expected_contains is not None:
            data["expected_contains"] = self.expected_contains
        if self.expected_exact is not None:
            data["expected_exact"] = self.expected_exact
        return data


def load_prompts(path: str | Path = "data/prompts.jsonl") -> list[PromptRecord]:
    file_path = Path(path)
    prompts: list[PromptRecord] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {file_path}:{line_number}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{file_path}:{line_number} expected JSON object")
            prompts.append(PromptRecord.from_json(data, file_path, line_number))
    validate_prompt_set(prompts, file_path)
    return prompts


def validate_prompt_set(prompts: Iterable[PromptRecord], path: Path) -> None:
    prompt_list = list(prompts)
    seen: set[str] = set()
    duplicates: list[str] = []
    for prompt in prompt_list:
        if prompt.id in seen:
            duplicates.append(prompt.id)
        seen.add(prompt.id)
    if duplicates:
        raise ValueError(f"{path} duplicate prompt ids: {', '.join(sorted(set(duplicates)))}")


def prompts_by_category(prompts: Iterable[PromptRecord], category: str) -> list[PromptRecord]:
    return [prompt for prompt in prompts if prompt.category == category]


def deterministic_prompts(prompts: Iterable[PromptRecord]) -> list[PromptRecord]:
    return [
        prompt
        for prompt in prompts
        if prompt.expected_contains is not None or prompt.expected_exact is not None
    ]


def summarize_prompts(prompts: Iterable[PromptRecord]) -> dict:
    prompt_list = list(prompts)
    categories = Counter(prompt.category for prompt in prompt_list)
    expected_count = len(deterministic_prompts(prompt_list))
    return {
        "total_prompts": len(prompt_list),
        "deterministic_prompts": expected_count,
        "categories": dict(sorted(categories.items())),
        "max_new_tokens_total": sum(prompt.max_new_tokens for prompt in prompt_list),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize the benchmark prompt JSONL file.")
    parser.add_argument("path", nargs="?", default="data/prompts.jsonl")
    parser.add_argument("--min-prompts", type=int, default=50)
    parser.add_argument("--min-deterministic", type=int, default=20)
    args = parser.parse_args()

    prompts = load_prompts(args.path)
    summary = summarize_prompts(prompts)
    if summary["total_prompts"] < args.min_prompts:
        raise SystemExit(f"Expected at least {args.min_prompts} prompts, found {summary['total_prompts']}")
    if summary["deterministic_prompts"] < args.min_deterministic:
        raise SystemExit(
            f"Expected at least {args.min_deterministic} deterministic prompts, "
            f"found {summary['deterministic_prompts']}"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

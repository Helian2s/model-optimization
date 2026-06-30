#!/usr/bin/env python3
"""Run a Hugging Face baseline for synthetic ISL/OSL benchmark regimes."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hf_synthetic_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())

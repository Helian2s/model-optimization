#!/usr/bin/env python3
"""CLI entrypoint for the Hugging Face baseline runner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hf_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())

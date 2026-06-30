#!/usr/bin/env python3
"""CLI entrypoint for quality regression checks."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quality_checks import main


if __name__ == "__main__":
    raise SystemExit(main())

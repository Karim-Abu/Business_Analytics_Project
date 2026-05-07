#!/usr/bin/env python3
"""Repository-root entry point for the data-preparation pipeline.

Usage
-----
    python scripts/run_pipeline.py
        Run sample pipeline (small synthetic data, fast).

    python scripts/run_pipeline.py --sample
        Same as above, explicit.

    python scripts/run_pipeline.py --full
        Run on real data from data/raw/ (must be supplied separately).

    python scripts/run_pipeline.py --mode safe_plus_conditional
        Use the larger build mode (adds conditional features + Orange CSVs).

    python scripts/run_pipeline.py --output-dir artifacts/my_run
        Custom output folder.

    python scripts/run_pipeline.py --no-orange-export
        Skip Orange CSV export.

This wrapper makes the pipeline runnable from any working directory by
adding ``Feature Engineering/`` to ``sys.path`` before importing the
pipeline module.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure pipeline prints (which contain unicode arrows / box chars) work on
# Windows consoles that default to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FE_DIR = PROJECT_ROOT / "Feature Engineering"

if not FE_DIR.exists():
    print(
        f"[run_pipeline] Could not locate the 'Feature Engineering' folder "
        f"at:\n  {FE_DIR}\n"
        "Are you running this from a fresh clone of the repository?",
        file=sys.stderr,
    )
    sys.exit(2)

# Make Feature Engineering importable as a flat module set.
sys.path.insert(0, str(FE_DIR))

from main_build_datasets import main  # noqa: E402


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except FileNotFoundError as exc:
        # load_raw_data already prints a friendly multi-line message.
        print(str(exc), file=sys.stderr)
        sys.exit(1)

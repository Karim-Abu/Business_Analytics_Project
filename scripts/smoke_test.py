#!/usr/bin/env python3
"""Smoke test: verify that the sample pipeline runs end-to-end on a fresh clone.

Exits 0 with ``SMOKE TEST PASSED`` on success, non-zero with
``SMOKE TEST FAILED`` plus a reason on failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "smoke_test"

REQUIRED_FILES = [
    "RUN_MANIFEST.md",
    "run_summary.json",
]
REQUIRED_DIRS = [
    "datasets",
    "audit",
    "metadata",
]


def fail(reason: str) -> None:
    print()
    print("SMOKE TEST FAILED")
    print(f"Reason: {reason}")
    sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("Smoke test: sample pipeline")
    print("=" * 60)

    # 1. Sample data must exist
    train_csv = SAMPLE_DIR / "train.csv"
    items_csv = SAMPLE_DIR / "items.csv"
    if not train_csv.exists() or not items_csv.exists():
        fail(
            f"Sample data missing. Expected:\n"
            f"  {train_csv}\n  {items_csv}"
        )

    # 2. Clean previous smoke output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # 3. Run sample pipeline (safe_only, no Orange export -> fast)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--sample",
        "--mode", "safe_only",
        "--output-dir", str(OUTPUT_DIR),
    ]
    print("Running:", " ".join(cmd))
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        fail(f"Pipeline exited with code {result.returncode}.")

    # 4. Verify required artefacts
    for name in REQUIRED_FILES:
        if not (OUTPUT_DIR / name).exists():
            fail(f"Missing required file: {name}")
    for d in REQUIRED_DIRS:
        path = OUTPUT_DIR / d
        if not path.exists() or not any(path.iterdir()):
            fail(f"Missing or empty directory: {d}")

    # 5. Manifest must mention some files
    manifest = (OUTPUT_DIR / "RUN_MANIFEST.md").read_text(encoding="utf-8")
    if "Generated files" not in manifest:
        fail("RUN_MANIFEST.md is malformed (no 'Generated files' section).")

    print()
    print("SMOKE TEST PASSED")
    print(f"Output: {OUTPUT_DIR}")
    sys.exit(0)


if __name__ == "__main__":
    main()

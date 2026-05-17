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

import pandas as pd

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
    "orange_exports",
]

PHARMFORM_EXPORTS = [
    "cls_pharmform_ablation_train_full.csv",
    "cls_pharmform_ablation_test.csv",
    "cls_pharmform_ablation_val.csv",
]
PHARMFORM_COLUMNS = [
    "pharmform_group",
    "pharmform_missing_flag",
    "pharmform_unmapped_flag",
]


def fail(reason: str) -> None:
    print()
    print("SMOKE TEST FAILED")
    print(f"Reason: {reason}")
    sys.exit(1)


def verify_sample_data() -> None:
    train_csv = SAMPLE_DIR / "train_sample.csv"
    items_csv = SAMPLE_DIR / "items_sample.csv"
    if not train_csv.exists() or not items_csv.exists():
        fail(
            f"Sample data missing. Expected:\n"
            f"  {train_csv}\n  {items_csv}"
        )


def clean_previous_output() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def run_sample_pipeline() -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--sample",
        "--mode", "safe_plus_conditional",
        "--output-dir", str(OUTPUT_DIR),
    ]
    print("Running:", " ".join(cmd))
    env = {**__import__("os").environ,
           "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        fail(f"Pipeline exited with code {result.returncode}.")


def verify_required_artifacts() -> None:
    for name in REQUIRED_FILES:
        if not (OUTPUT_DIR / name).exists():
            fail(f"Missing required file: {name}")
    for d in REQUIRED_DIRS:
        path = OUTPUT_DIR / d
        if not path.exists() or not any(path.iterdir()):
            fail(f"Missing or empty directory: {d}")


def verify_manifest() -> None:
    manifest = (OUTPUT_DIR / "RUN_MANIFEST.md").read_text(encoding="utf-8")
    if "Generated files" not in manifest:
        fail("RUN_MANIFEST.md is malformed (no 'Generated files' section).")


def verify_pharmform_exports() -> None:
    orange_dir = OUTPUT_DIR / "orange_exports"
    for name in PHARMFORM_EXPORTS:
        path = orange_dir / name
        if not path.exists():
            fail(f"Missing PharmForm ablation union export: {name}")

    train_union = pd.read_csv(
        orange_dir / "cls_pharmform_ablation_train_full.csv")
    missing_columns = [
        c for c in PHARMFORM_COLUMNS if c not in train_union.columns]
    if missing_columns:
        fail(
            f"Missing PharmForm columns in ablation export: {missing_columns}")

    for col in ["pharmform_missing_flag", "pharmform_unmapped_flag"]:
        values = set(train_union[col].dropna().unique())
        if not values <= {0, 1}:
            fail(f"{col} is not binary in ablation export: {values}")


def main() -> None:
    print("=" * 60)
    print("Smoke test: sample pipeline")
    print("=" * 60)

    verify_sample_data()
    clean_previous_output()
    run_sample_pipeline()
    verify_required_artifacts()
    verify_manifest()
    verify_pharmform_exports()

    print()
    print("SMOKE TEST PASSED")
    print(f"Output: {OUTPUT_DIR}")
    sys.exit(0)


if __name__ == "__main__":
    main()

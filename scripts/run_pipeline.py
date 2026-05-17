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

    python scripts/run_pipeline.py --full --benchmark
        Run the data-preparation pipeline and then a quick model benchmark.

This wrapper makes the pipeline runnable from any working directory by
adding ``Feature Engineering/`` to ``sys.path`` before importing the
pipeline module.
"""

from __future__ import annotations

import json
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

from main_build_datasets import main, write_run_manifest  # noqa: E402
from run_model_benchmark import run_benchmark  # noqa: E402


def _split_pipeline_and_benchmark_args(argv: list[str]) -> tuple[list[str], dict]:
    """Extract wrapper-level benchmark flags before delegating to the pipeline."""
    pipeline_args: list[str] = []
    benchmark_args = {
        "enabled": False,
        "max_train_rows": 200_000,
        "max_eval_rows": 100_000,
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--benchmark":
            benchmark_args["enabled"] = True
            i += 1
        elif arg == "--benchmark-max-train-rows":
            if i + 1 >= len(argv):
                raise ValueError("--benchmark-max-train-rows needs a value")
            benchmark_args["max_train_rows"] = int(argv[i + 1])
            i += 2
        elif arg.startswith("--benchmark-max-train-rows="):
            benchmark_args["max_train_rows"] = int(arg.split("=", 1)[1])
            i += 1
        elif arg == "--benchmark-max-eval-rows":
            if i + 1 >= len(argv):
                raise ValueError("--benchmark-max-eval-rows needs a value")
            benchmark_args["max_eval_rows"] = int(argv[i + 1])
            i += 2
        elif arg.startswith("--benchmark-max-eval-rows="):
            benchmark_args["max_eval_rows"] = int(arg.split("=", 1)[1])
            i += 1
        else:
            pipeline_args.append(arg)
            i += 1
    return pipeline_args, benchmark_args


def main_with_optional_benchmark(argv: list[str]) -> None:
    """Run the data pipeline, then optionally run the model benchmark."""
    pipeline_args, benchmark_args = _split_pipeline_and_benchmark_args(argv)
    main(pipeline_args)
    if not benchmark_args["enabled"]:
        return

    output_dir: Path | None = None
    for i, arg in enumerate(pipeline_args):
        if arg == "--output-dir" and i + 1 < len(pipeline_args):
            output_dir = Path(pipeline_args[i + 1]).resolve()
            break
        if arg.startswith("--output-dir="):
            output_dir = Path(arg.split("=", 1)[1]).resolve()
            break

    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / (
            "full_run" if "--full" in pipeline_args else "sample_run"
        )

    benchmark_summary = run_benchmark(
        output_dir=output_dir,
        max_train_rows=benchmark_args["max_train_rows"],
        max_eval_rows=benchmark_args["max_eval_rows"],
    )
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("timestamp_utc", None)
        summary.pop("files", None)
        summary["model_benchmark"] = {
            "status": "completed",
            "best_models": benchmark_summary.get("best_models", []),
        }
        notes = list(summary.get("notes", []))
        notes.append("Model benchmark results are available in benchmark/.")
        summary["notes"] = notes
        write_run_manifest(output_dir, summary)


if __name__ == "__main__":
    try:
        main_with_optional_benchmark(sys.argv[1:])
    except ValueError as exc:
        print(f"[run_pipeline] {exc}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as exc:
        # load_raw_data already prints a friendly multi-line message.
        print(str(exc), file=sys.stderr)
        sys.exit(1)

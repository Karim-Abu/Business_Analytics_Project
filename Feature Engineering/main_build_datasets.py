#!/usr/bin/env python3
"""
main_build_datasets.py — orchestrate the full data-preparation pipeline.

Usage
-----
    python main_build_datasets.py --mode safe_only
    python main_build_datasets.py --mode safe_plus_conditional

Modes
-----
safe_only              : load → clean → split → pid_segment → safe FE →
                         matrices → sampling → audit → export
safe_plus_conditional  : (not yet implemented) extends safe_only with
                         time-aware OOF and cumulative features
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import config as cfg
from io_utils import (
    load_raw_data, load_pharmform_mapping, merge_train_items, ensure_output_dirs,
    save_parquet, save_csv, save_text_report,
)
from Preprocessing.preprocessing import run_all_preprocessing
from Preprocessing.validation import assert_preprocessing_integrity
from Preprocessing.feature_engineering_safe import run_all_safe_features
from Preprocessing.feature_engineering_conditional import run_all_conditional_features
from Preprocessing.audit import run_full_audit
from Sampling.split import run_split
from Sampling.pid_segment import (
    fit_pid_segment,
    apply_pid_segment,
    save_pid_segment_map,
)
from Sampling.sampling import run_sampling
from feature_sets import (
    build_safe_feature_matrices, build_conditional_feature_matrices,
    summarize_feature_sets, export_matrices,
)
from orange_export import export_orange_csvs


# ═══════════════════════════════════════════════════════════════════════════
# safe_only pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_safe_only(run_context: dict | None = None) -> dict:
    """Execute the complete safe-only pipeline.

    Returns a dict of run metadata used for the run manifest.
    """
    t0 = time.time()
    ctx = run_context or {}

    # ── 1. Output dirs ───────────────────────────────────────────────────
    ensure_output_dirs()

    # ── 2. Load raw data ─────────────────────────────────────────────────
    df_train_raw, df_items = load_raw_data()
    pharmform_mapping = load_pharmform_mapping()

    # ── 3. Join ──────────────────────────────────────────────────────────
    df_merged = merge_train_items(df_train_raw, df_items)

    # ── 4. Preprocessing ─────────────────────────────────────────────────
    df_merged = run_all_preprocessing(
        df_merged, pharmform_mapping=pharmform_mapping)
    assert_preprocessing_integrity(
        df_merged, context="safe_only/post-preprocessing")

    # ── 5. Chronological split ───────────────────────────────────────────
    df_train, df_val, df_test = run_split(df_merged)

    # ── 6. pid_segment (before sampling) ─────────────────────────────────
    pid_map = fit_pid_segment(df_train)
    for d in (df_train, df_val, df_test):
        apply_pid_segment(d, pid_map)
    save_pid_segment_map(
        pid_map, cfg.OUTPUT_METADATA_DIR / "pid_segment_map.csv")

    # ── 7. Safe feature engineering ──────────────────────────────────────
    df_train, df_val, df_test, fe_metadata = run_all_safe_features(
        df_train, df_val, df_test
    )

    # ── 8. Build feature matrices ────────────────────────────────────────
    matrices = build_safe_feature_matrices(df_train, df_val, df_test)

    # ── 9. Sampling (prototyping) ────────────────────────────────────────
    sampling_result = run_sampling(df_train)

    # ── 10. Audit ────────────────────────────────────────────────────────
    reports = run_full_audit(
        df_train_raw=df_train_raw,
        df_items=df_items,
        df_merged=df_merged,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        feature_matrices=matrices,
    )

    # ``safe_only`` writes matrices/audits/metadata, but no Orange CSV export.
    # The final Orange files are intentionally reserved for the conditional run.
    # ── 11. Export ───────────────────────────────────────────────────────
    _export_all(matrices, sampling_result, reports, fe_metadata)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE  (safe_only)  -  {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Outputs in: {cfg.OUTPUT_DIR}")
    print(f"  Matrices:   {cfg.OUTPUT_DATASETS_DIR}")
    print(f"  Audits:     {cfg.OUTPUT_AUDIT_DIR}")
    print(f"  Metadata:   {cfg.OUTPUT_METADATA_DIR}")

    return {
        "mode": "safe_only",
        "elapsed_s": round(elapsed, 2),
        "rows": {
            "train_raw": len(df_train_raw),
            "items": len(df_items),
            "merged": len(df_merged),
            "train": len(df_train),
            "test": len(df_test),
            "validation": len(df_val),
        },
        "orange_export": "skipped (only available in safe_plus_conditional)",
        **ctx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# safe_plus_conditional pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_safe_plus_conditional(
    run_context: dict | None = None,
    orange_export: bool = True,
) -> dict:
    """Execute the full safe + conditional pipeline.

    Steps 1-7 are identical to ``safe_only``.  After the safe feature
    matrices are built, conditional features are computed and assembled
    into additional matrices.
    """
    t0 = time.time()
    ctx = run_context or {}

    # ── 1. Output dirs ───────────────────────────────────────────────────
    ensure_output_dirs()

    # ── 2. Load raw data ─────────────────────────────────────────────────
    df_train_raw, df_items = load_raw_data()
    pharmform_mapping = load_pharmform_mapping()

    # ── 3. Join ──────────────────────────────────────────────────────────
    df_merged = merge_train_items(df_train_raw, df_items)

    # ── 4. Preprocessing ─────────────────────────────────────────────────
    df_merged = run_all_preprocessing(
        df_merged, pharmform_mapping=pharmform_mapping)
    assert_preprocessing_integrity(
        df_merged, context="safe_plus_conditional/post-preprocessing")

    # ── 5. Chronological split ───────────────────────────────────────────
    df_train, df_val, df_test = run_split(df_merged)

    # ── 6. pid_segment (before sampling) ─────────────────────────────────
    pid_map = fit_pid_segment(df_train)
    for d in (df_train, df_val, df_test):
        apply_pid_segment(d, pid_map)
    save_pid_segment_map(
        pid_map, cfg.OUTPUT_METADATA_DIR / "pid_segment_map.csv")

    # ── 7. Safe feature engineering ──────────────────────────────────────
    df_train, df_val, df_test, fe_metadata = run_all_safe_features(
        df_train, df_val, df_test
    )

    # ── 8. Build safe feature matrices ───────────────────────────────────
    safe_matrices = build_safe_feature_matrices(df_train, df_val, df_test)

    # ── 9. Conditional feature engineering ────────────────────────────────
    df_train, df_val, df_test, cond_metadata = run_all_conditional_features(
        df_train, df_val, df_test
    )

    # ── 10. Build conditional feature matrices ───────────────────────────
    cond_matrices = build_conditional_feature_matrices(
        df_train, df_val, df_test
    )

    # Downstream audit/export should see the full union of safe + conditional
    # matrices because this mode represents the final end-to-end dataset build.
    all_matrices = {**safe_matrices, **cond_matrices}

    # ── 11. Sampling (prototyping) ───────────────────────────────────────
    sampling_result = run_sampling(df_train)

    # ── 12. Audit ────────────────────────────────────────────────────────
    reports = run_full_audit(
        df_train_raw=df_train_raw,
        df_items=df_items,
        df_merged=df_merged,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        feature_matrices=all_matrices,
    )

    # ── 13. Export ───────────────────────────────────────────────────────
    merged_metadata = {**fe_metadata, **cond_metadata}
    _export_all(all_matrices, sampling_result, reports, merged_metadata)

    # Orange consumes the fully engineered row-level datasets, not the parquet
    # matrices above. Export happens only after all validations succeeded.
    # ── 14. Orange CSV export ────────────────────────────────────────────
    if orange_export:
        export_orange_csvs(
            df_train, df_val, df_test,
            sampling_result,
            build_mode="safe_plus_conditional",
        )
        orange_status = "exported"
    else:
        print("[orange] export skipped (--no-orange-export)")
        orange_status = "skipped (--no-orange-export)"

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE  (safe_plus_conditional)  -  {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Outputs in: {cfg.OUTPUT_DIR}")
    print(f"  Matrices:   {cfg.OUTPUT_DATASETS_DIR}")
    print(f"  Audits:     {cfg.OUTPUT_AUDIT_DIR}")
    print(f"  Metadata:   {cfg.OUTPUT_METADATA_DIR}")
    if orange_export:
        print(f"  Orange:     {cfg.OUTPUT_ORANGE_EXPORTS_DIR}")

    return {
        "mode": "safe_plus_conditional",
        "elapsed_s": round(elapsed, 2),
        "rows": {
            "train_raw": len(df_train_raw),
            "items": len(df_items),
            "merged": len(df_merged),
            "train": len(df_train),
            "test": len(df_test),
            "validation": len(df_val),
        },
        "orange_export": orange_status,
        **ctx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Export helpers
# ═══════════════════════════════════════════════════════════════════════════

def _export_all(
    matrices: dict,
    sampling_result: dict,
    reports: dict[str, pd.DataFrame],
    fe_metadata: dict,
) -> None:
    """Persist all artefacts."""
    print(f"\n{'='*60}")
    print("EXPORTING")
    print(f"{'='*60}")

    # Parquet matrices are the Python-side modelling artefacts and remain
    # separate from the Orange CSV export that is written afterwards.
    # Feature matrices (parquet)
    export_matrices(matrices, cfg.OUTPUT_DATASETS_DIR)

    # Sampling samples (parquet)
    for key in ("train_cls_sample", "train_reg_sample"):
        if key in sampling_result:
            save_parquet(
                sampling_result[key],
                cfg.OUTPUT_DATASETS_DIR / f"{key}.parquet",
            )

    # Sampling audits (CSV)
    for key in ("audit_cls", "audit_reg"):
        if key in sampling_result:
            save_csv(
                sampling_result[key],
                cfg.OUTPUT_AUDIT_DIR / f"sampling_{key}.csv",
            )

    # Audit reports (CSV)
    for name, df in reports.items():
        save_csv(df, cfg.OUTPUT_AUDIT_DIR / f"{name}.csv")

    # Feature-set summary (also as readable text)
    summary = summarize_feature_sets(matrices)
    save_text_report(
        summary.to_string(index=False),
        cfg.OUTPUT_AUDIT_DIR / "feature_matrix_summary.txt",
    )

    # Metadata
    if "bin_edges" in fe_metadata:
        import json
        edges_serialisable = {
            k: v.tolist() for k, v in fe_metadata["bin_edges"].items()
        }
        path = cfg.OUTPUT_METADATA_DIR / "binning_edges.json"
        path.write_text(json.dumps(edges_serialisable, indent=2))
        print(f"[save] Metadata → binning_edges.json")

    # Conditional metadata (aggregation maps, OOF encodings, fold info)
    if "global_aggregation_maps" in fe_metadata:
        import json
        maps = fe_metadata["global_aggregation_maps"]
        serialisable = {}
        for name, m in maps.items():
            serialisable[name] = {
                "group_col": m["group_col"],
                "group_means": {str(k): v for k, v in m["group_means"].items()},
                "global_mean": m["global_mean"],
            }
        path = cfg.OUTPUT_METADATA_DIR / "conditional_aggregation_maps.json"
        path.write_text(json.dumps(serialisable, indent=2))
        print("[save] Metadata → conditional_aggregation_maps.json")

    if "full_train_encodings" in fe_metadata:
        import json
        encs = fe_metadata["full_train_encodings"]
        serialisable = {}
        for name, enc in encs.items():
            serialisable[name] = {
                "group_means": {str(k): v for k, v in enc["group_means"].items()},
                "global_mean": enc["global_mean"],
            }
        path = cfg.OUTPUT_METADATA_DIR / "conditional_oof_encodings.json"
        path.write_text(json.dumps(serialisable, indent=2))
        print("[save] Metadata → conditional_oof_encodings.json")

    if "oof_fold_info" in fe_metadata:
        import json
        path = cfg.OUTPUT_METADATA_DIR / "oof_fold_info.json"
        path.write_text(json.dumps(fe_metadata["oof_fold_info"], indent=2))
        print("[save] Metadata → oof_fold_info.json")


# ═══════════════════════════════════════════════════════════════════════════
# Run manifest
# ═══════════════════════════════════════════════════════════════════════════

_FILE_PURPOSES = {
    "RUN_MANIFEST.md": "Human-readable summary of this run.",
    "run_summary.json": "Machine-readable summary of this run.",
}

_DIR_PURPOSES = {
    "datasets": "Feature matrices and sampled training subsets (parquet).",
    "audit": "Audit reports (join quality, missingness, sampling, summary).",
    "metadata": "Reproducibility artefacts (pid_segment map, binning edges, encodings).",
    "orange_exports": "Orange-ready CSV exports (only in safe_plus_conditional).",
    "feature_selection": "Feature-selection reports (legacy, populated by separate script).",
}


def _scan_outputs(output_dir: Path) -> list[dict]:
    """Recursively list real output files for the manifest."""
    files: list[dict] = []
    for p in sorted(output_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(output_dir).as_posix()
        size = p.stat().st_size
        files.append({"path": rel, "size_bytes": size})
    return files


def write_run_manifest(output_dir: Path, summary: dict) -> None:
    """Write RUN_MANIFEST.md (human) and run_summary.json (machine)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _scan_outputs(output_dir)
    summary_full = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **summary,
        "files": files,
    }

    # JSON
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary_full, indent=2), encoding="utf-8"
    )

    # Markdown
    lines: list[str] = []
    lines.append(f"# Run Manifest")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{summary_full['timestamp_utc']}`")
    lines.append(f"- Run type: **{summary.get('run_type', 'unknown')}**")
    lines.append(f"- Build mode: `{summary.get('mode', 'unknown')}`")
    lines.append(f"- Elapsed: `{summary.get('elapsed_s', '?')} s`")
    lines.append(f"- Orange export: {summary.get('orange_export', 'n/a')}")
    lines.append("")
    lines.append("## Inputs")
    inputs = summary.get("inputs", {})
    for key, value in inputs.items():
        lines.append(f"- {key}: `{value}`")
    rows = summary.get("rows", {})
    if rows:
        lines.append("")
        lines.append("## Row counts")
        for key, value in rows.items():
            lines.append(f"- {key}: {value:,}")
    lines.append("")
    lines.append("## Output directory")
    lines.append(f"`{output_dir}`")
    lines.append("")
    lines.append("## What is in this folder?")
    for sub, desc in _DIR_PURPOSES.items():
        if (output_dir / sub).exists():
            lines.append(f"- `{sub}/` - {desc}")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    lines.append("| Path | Size (KB) |")
    lines.append("|------|-----------|")
    for f in files:
        if f["path"] in ("RUN_MANIFEST.md", "run_summary.json"):
            continue
        kb = f["size_bytes"] / 1024
        lines.append(f"| `{f['path']}` | {kb:.1f} |")
    lines.append("")
    lines.append("## Notes")
    notes = summary.get("notes", [])
    if not notes:
        lines.append("- (none)")
    else:
        for n in notes:
            lines.append(f"- {n}")
    lines.append("")
    lines.append("## Suggested next steps")
    for s in summary.get("next_steps", []):
        lines.append(f"- {s}")

    (output_dir / "RUN_MANIFEST.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        f"[manifest] Wrote RUN_MANIFEST.md and run_summary.json -> {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the requested dataset build pipeline."""
    parser = argparse.ArgumentParser(
        description="Build feature matrices for Dynamic Pricing project."
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--sample", action="store_true",
        help="Use small sample data from data/sample/ (default).",
    )
    data_group.add_argument(
        "--full", action="store_true",
        help="Use full real data from data/raw/ (not in repo).",
    )
    parser.add_argument(
        "--mode",
        choices=cfg.BUILD_MODES,
        default=cfg.BUILD_MODE_DEFAULT,
        help=f"Build mode (default: {cfg.BUILD_MODE_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write artefacts (default: artifacts/sample_run or artifacts/full_run).",
    )
    parser.add_argument(
        "--no-orange-export", action="store_true",
        help="Skip Orange CSV export in safe_plus_conditional mode.",
    )
    args = parser.parse_args(argv)

    # Default to sample if neither flag was given.
    run_type = "full" if args.full else "sample"

    if run_type == "sample":
        data_dir = cfg.SAMPLE_DATA_DIR
        default_out = cfg.PROJECT_ROOT / "artifacts" / "sample_run"
        train_filename = cfg.SAMPLE_TRAIN_FILENAME
        items_filename = cfg.SAMPLE_ITEMS_FILENAME
    else:
        data_dir = cfg.RAW_DATA_DIR
        default_out = cfg.PROJECT_ROOT / "artifacts" / "full_run"
        train_filename = "train.csv"
        items_filename = "items.csv"

    output_dir = Path(args.output_dir).resolve(
    ) if args.output_dir else default_out
    cfg.configure_runtime(data_dir, output_dir, train_filename, items_filename)

    print(f"\n{'#'*60}")
    print(f"  Dynamic Pricing - Data Preparation Pipeline")
    print(f"  Run type:   {run_type}")
    print(f"  Mode:       {args.mode}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"{'#'*60}\n")

    run_context = {
        "run_type": run_type,
        "inputs": {
            "data_dir": str(data_dir),
            "train_csv": str(cfg.TRAIN_CSV),
            "items_csv": str(cfg.ITEMS_CSV),
        },
        "next_steps": [
            "Inspect feature matrices in `datasets/`",
            "Inspect audit reports in `audit/`",
            "For real data, run: python scripts/run_pipeline.py --full",
        ],
        "notes": [
            "Sample run uses a tiny synthetic dataset; results are NOT meaningful for modelling."
            if run_type == "sample" else
            "Full run uses the real dataset from data/raw/.",
        ],
    }

    if args.mode == "safe_only":
        summary = run_safe_only(run_context=run_context)
    else:
        summary = run_safe_plus_conditional(
            run_context=run_context,
            orange_export=not args.no_orange_export,
        )

    write_run_manifest(output_dir, summary)


if __name__ == "__main__":
    main()

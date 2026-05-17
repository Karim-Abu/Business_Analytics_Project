#!/usr/bin/env python3
"""Quick sklearn benchmark for generated pipeline feature matrices."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREFERRED_VARIANTS = ("conditional", "expanded", "base")
CLASSIFICATION_METRIC = "f1"
REGRESSION_METRIC = "mae"
DIAGNOSTIC_WINNERS_LABEL = "Quick benchmark winners (diagnostic only)"
DIAGNOSTIC_NOTE = (
    "Note: This quick benchmark uses capped training/evaluation data and is "
    "not the final model selection reported in the paper."
)
FINAL_MODEL_SELECTION_NOTE = (
    "Final model selection is documented in the separate modelling "
    "documentation: CLS final: HistGradientBoosting, Threshold 0.22. "
    "REG final: Always-1 / DummyMedian, because median quantity = 1."
)
DUMMY_MEDIAN_ALWAYS_ONE_NOTE = (
    "For this dataset, DummyMedian is equivalent to the documented Always-1 "
    "baseline because the training median of quantity is 1."
)


def _utc_timestamp() -> str:
    """Return a compact timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _normalise_limit(value: int | None) -> int | None:
    """Return None for non-positive row limits."""
    if value is None or value <= 0:
        return None
    return value


def _read_target(path: Path, task: str) -> pd.Series:
    """Load a parquet target file as a clean Series."""
    target = pd.read_parquet(path).iloc[:, 0]
    target = pd.to_numeric(target, errors="coerce")
    if task == "cls":
        target = target.astype("Int64")
    return target


def _drop_missing_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove rows that have no usable target value."""
    mask = target.notna()
    return (
        features.loc[mask].reset_index(drop=True),
        target.loc[mask].reset_index(drop=True),
    )


def _coerce_numeric_pair(
    x_train: pd.DataFrame,
    x_eval: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert non-numeric columns with train-fitted category codes."""
    x_train = x_train.copy()
    x_eval = x_eval.copy()

    missing_eval_cols = [c for c in x_train.columns if c not in x_eval.columns]
    if missing_eval_cols:
        raise ValueError(
            f"Evaluation matrix is missing columns: {missing_eval_cols}")
    x_eval = x_eval[x_train.columns]

    for col in x_train.columns:
        if is_numeric_dtype(x_train[col]) and is_numeric_dtype(x_eval[col]):
            continue

        train_values = x_train[col].astype("string")
        eval_values = x_eval[col].astype("string")
        categories = pd.Index(train_values.dropna().unique())
        mapping = {value: code for code, value in enumerate(categories)}

        x_train[col] = train_values.map(mapping).astype("float64")
        x_eval[col] = eval_values.map(mapping).astype("float64")

    return x_train, x_eval


def _sample_pair(
    features: pd.DataFrame,
    target: pd.Series,
    max_rows: int | None,
    task: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Limit matrix size for a quick benchmark."""
    if max_rows is None or len(features) <= max_rows:
        return features.reset_index(drop=True), target.reset_index(drop=True)

    stratify = None
    if task == "cls":
        counts = target.value_counts(dropna=False)
        if len(counts) > 1 and counts.min() >= 2 and max_rows >= len(counts):
            stratify = target

    x_sample, _, y_sample, _ = train_test_split(
        features,
        target,
        train_size=max_rows,
        random_state=random_state,
        stratify=stratify,
    )
    return x_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def _available_variants(datasets_dir: Path, task: str) -> list[str]:
    """Return available matrix variants in a stable preference order."""
    variants: set[str] = set()
    for path in datasets_dir.glob(f"X_train_{task}_*.parquet"):
        variants.add(path.stem.replace(f"X_train_{task}_", "", 1))

    ordered = [variant for variant in PREFERRED_VARIANTS if variant in variants]
    ordered.extend(sorted(variants - set(ordered)))
    return ordered


def _evaluation_split(datasets_dir: Path, task: str, variant: str) -> str:
    """Use validation for model selection, then fall back to test."""
    for split in ("validation", "test"):
        if (
            datasets_dir / f"X_{split}_{task}_{variant}.parquet"
        ).exists() and (datasets_dir / f"y_{split}_{task}.parquet").exists():
            return split
    raise FileNotFoundError(
        f"No validation/test matrix found for task={task}, variant={variant}."
    )


def _classification_models(random_state: int) -> dict[str, Any]:
    """Return lightweight classification benchmark models."""
    return {
        "DummyMostFrequent": DummyClassifier(
            strategy="most_frequent",
            random_state=random_state,
        ),
        "LogisticRegressionBalanced": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            memory=None,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            random_state=random_state,
        ),
    }


def _regression_models(random_state: int) -> dict[str, Any]:
    """Return lightweight regression benchmark models."""
    return {
        "DummyMedian": DummyRegressor(strategy="median"),
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
            memory=None,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            random_state=random_state,
        ),
    }


def _positive_scores(model: Any, x_eval: pd.DataFrame) -> np.ndarray | None:
    """Return positive-class scores when the model exposes them."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_eval)
        if probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return None
    if hasattr(model, "decision_function"):
        return model.decision_function(x_eval)
    return None


def _empty_metric_row(
    task: str,
    variant: str,
    model_name: str,
    split: str,
    n_train: int,
    n_eval: int,
) -> dict[str, Any]:
    """Create a stable result row schema."""
    return {
        "task": task,
        "variant": variant,
        "model": model_name,
        "split": split,
        "n_train": n_train,
        "n_eval": n_eval,
        "selection_metric": CLASSIFICATION_METRIC if task == "cls" else REGRESSION_METRIC,
        "selection_value": np.nan,
        "training_target_median": np.nan,
        "status": "ok",
        "error": "",
        "accuracy": np.nan,
        "f1": np.nan,
        "pr_auc": np.nan,
        "roc_auc": np.nan,
        "mae": np.nan,
        "median_ae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "elapsed_s": np.nan,
    }


def _evaluate_classification(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    variant: str,
    split: str,
) -> dict[str, Any]:
    """Fit and score one classification model."""
    row = _empty_metric_row(
        "cls", variant, model_name, split, len(x_train), len(x_eval)
    )
    started = time.time()
    try:
        model.fit(x_train, y_train.astype(int))
        predictions = model.predict(x_eval)
        row["accuracy"] = accuracy_score(y_eval.astype(int), predictions)
        row["f1"] = f1_score(y_eval.astype(int), predictions, zero_division=0)

        scores = _positive_scores(model, x_eval)
        if scores is not None and y_eval.nunique() > 1:
            row["pr_auc"] = average_precision_score(y_eval.astype(int), scores)
            row["roc_auc"] = roc_auc_score(y_eval.astype(int), scores)
        row["selection_value"] = row[CLASSIFICATION_METRIC]
    except Exception as exc:  # pragma: no cover - retained in CSV for diagnosis
        row["status"] = "failed"
        row["error"] = str(exc)
    finally:
        row["elapsed_s"] = round(time.time() - started, 3)
    return row


def _evaluate_regression(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    variant: str,
    split: str,
) -> dict[str, Any]:
    """Fit and score one regression model."""
    row = _empty_metric_row(
        "reg", variant, model_name, split, len(x_train), len(x_eval)
    )
    started = time.time()
    try:
        model.fit(x_train, y_train.astype(float))
        predictions = model.predict(x_eval)
        row["mae"] = mean_absolute_error(y_eval.astype(float), predictions)
        row["median_ae"] = median_absolute_error(
            y_eval.astype(float), predictions)
        row["rmse"] = float(
            np.sqrt(mean_squared_error(y_eval.astype(float), predictions))
        )
        row["r2"] = r2_score(y_eval.astype(float), predictions)
        row["selection_value"] = row[REGRESSION_METRIC]
    except Exception as exc:  # pragma: no cover - retained in CSV for diagnosis
        row["status"] = "failed"
        row["error"] = str(exc)
    finally:
        row["elapsed_s"] = round(time.time() - started, 3)
    return row


def _load_task_data(
    datasets_dir: Path,
    task: str,
    variant: str,
    split: str,
    max_train_rows: int | None,
    max_eval_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load, clean, sample and numeric-coerce matrices for one task."""
    x_train = pd.read_parquet(
        datasets_dir / f"X_train_{task}_{variant}.parquet")
    y_train = _read_target(datasets_dir / f"y_train_{task}.parquet", task)
    x_eval = pd.read_parquet(
        datasets_dir / f"X_{split}_{task}_{variant}.parquet")
    y_eval = _read_target(datasets_dir / f"y_{split}_{task}.parquet", task)

    x_train, y_train = _drop_missing_target(x_train, y_train)
    x_eval, y_eval = _drop_missing_target(x_eval, y_eval)
    x_train, y_train = _sample_pair(
        x_train, y_train, max_train_rows, task, random_state
    )
    x_eval, y_eval = _sample_pair(
        x_eval, y_eval, max_eval_rows, task, random_state + 1
    )
    x_train, x_eval = _coerce_numeric_pair(x_train, x_eval)
    return x_train, y_train, x_eval, y_eval


def _median_quantity_note(row: pd.Series) -> str | None:
    """Return the Always-1 note for the matching DummyMedian winner."""
    if row.get("task") != "reg" or row.get("model") != "DummyMedian":
        return None
    median_value = row.get("training_target_median")
    if pd.isna(median_value):
        return None
    if np.isclose(float(median_value), 1.0):
        return DUMMY_MEDIAN_ALWAYS_ONE_NOTE
    return None


def _summary_winner_lines(row: pd.Series) -> list[str]:
    """Return markdown lines for one diagnostic benchmark winner."""
    if row["task"] == "cls":
        return [
            "- CLS: "
            f"{row['model']} on {row['variant']} "
            f"({row['split']}), F1={row['f1']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}, ROC-AUC={row['roc_auc']:.4f}"
        ]

    lines = [
        "- REG: "
        f"{row['model']} on {row['variant']} "
        f"({row['split']}), MAE={row['mae']:.4f}, "
        f"MedAE={row['median_ae']:.4f}, RMSE={row['rmse']:.4f}"
    ]
    median_note = _median_quantity_note(row)
    if median_note:
        lines.append(f"  - {median_note}")
    return lines


def _print_winner(row: pd.Series) -> None:
    """Print one diagnostic benchmark winner to stdout."""
    if row["task"] == "cls":
        print(
            f"    CLS: {row['model']} "
            f"F1={row['f1']:.4f} PR-AUC={row['pr_auc']:.4f}"
        )
        return

    print(
        f"    REG: {row['model']} "
        f"MAE={row['mae']:.4f} RMSE={row['rmse']:.4f}"
    )
    median_note = _median_quantity_note(row)
    if median_note:
        print(f"    {median_note}")


def _select_best(results: pd.DataFrame) -> pd.DataFrame:
    """Select best CLS and REG rows from successful benchmark results."""
    best_rows: list[pd.Series] = []
    for task, metric, ascending in (
        ("cls", CLASSIFICATION_METRIC, False),
        ("reg", REGRESSION_METRIC, True),
    ):
        subset = results[
            (results["task"] == task)
            & (results["status"] == "ok")
            & (results[metric].notna())
        ]
        if subset.empty:
            continue
        best_rows.append(subset.sort_values(
            metric, ascending=ascending).iloc[0])

    if not best_rows:
        return pd.DataFrame(columns=results.columns)
    return pd.DataFrame(best_rows).reset_index(drop=True)


def _write_text_summary(
    path: Path,
    output_dir: Path,
    max_train_rows: int | None,
    max_eval_rows: int | None,
    results: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    """Write a human-readable benchmark summary."""
    lines: list[str] = []
    lines.append("# Model Benchmark Summary")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{_utc_timestamp()}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Train row limit: `{max_train_rows or 'none'}`")
    lines.append(f"- Evaluation row limit: `{max_eval_rows or 'none'}`")
    lines.append("- Selection: CLS uses highest F1; REG uses lowest MAE.")
    lines.append(f"- {DIAGNOSTIC_NOTE}")
    lines.append(f"- {FINAL_MODEL_SELECTION_NOTE}")
    lines.append("")
    lines.append(f"## {DIAGNOSTIC_WINNERS_LABEL}")

    if best.empty:
        lines.append("- No successful model benchmark result.")
    else:
        for _, row in best.iterrows():
            lines.extend(_summary_winner_lines(row))

    failed = results[results["status"] != "ok"]
    if not failed.empty:
        lines.append("")
        lines.append("## Failed Runs")
        for _, row in failed.iterrows():
            lines.append(f"- {row['task']} {row['model']}: {row['error']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""
    cleaned = frame.replace({np.nan: None})
    return json.loads(cleaned.to_json(orient="records"))


def _models_for_task(task: str, random_state: int) -> dict[str, Any]:
    """Return benchmark models for one task."""
    if task == "cls":
        return _classification_models(random_state)
    return _regression_models(random_state)


def _evaluate_task_model(
    task: str,
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    variant: str,
    split: str,
) -> dict[str, Any]:
    """Evaluate one model for either CLS or REG."""
    if task == "cls":
        return _evaluate_classification(
            model_name, model, x_train, y_train, x_eval, y_eval, variant, split
        )
    return _evaluate_regression(
        model_name, model, x_train, y_train, x_eval, y_eval, variant, split
    )


def _benchmark_variant(
    datasets_dir: Path,
    task: str,
    variant: str,
    max_train_rows: int | None,
    max_eval_rows: int | None,
    random_state: int,
) -> list[dict[str, Any]]:
    """Run all benchmark models for one task and matrix variant."""
    split = _evaluation_split(datasets_dir, task, variant)
    print(f"  [{task}] matrix={variant}, split={split}")
    x_train, y_train, x_eval, y_eval = _load_task_data(
        datasets_dir,
        task,
        variant,
        split,
        max_train_rows,
        max_eval_rows,
        random_state,
    )
    result_rows = [
        _evaluate_task_model(
            task,
            model_name,
            model,
            x_train,
            y_train,
            x_eval,
            y_eval,
            variant,
            split,
        )
        for model_name, model in _models_for_task(task, random_state).items()
    ]
    if task == "reg":
        training_target_median = float(y_train.astype(float).median())
        for result_row in result_rows:
            result_row["training_target_median"] = training_target_median
    return result_rows


def _collect_benchmark_rows(
    datasets_dir: Path,
    max_train_rows: int | None,
    max_eval_rows: int | None,
    random_state: int,
) -> list[dict[str, Any]]:
    """Run all task and variant benchmarks and collect result rows."""
    rows: list[dict[str, Any]] = []
    for task in ("cls", "reg"):
        variants = _available_variants(datasets_dir, task)
        if not variants:
            print(f"  [{task}] skipped: no feature matrix found")
            continue

        for variant in variants:
            rows.extend(
                _benchmark_variant(
                    datasets_dir,
                    task,
                    variant,
                    max_train_rows,
                    max_eval_rows,
                    random_state,
                )
            )
    return rows


def run_benchmark(
    output_dir: Path | str,
    max_train_rows: int | None = 200_000,
    max_eval_rows: int | None = 100_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the quick model benchmark for a pipeline output directory."""
    output_dir = Path(output_dir).resolve()
    datasets_dir = output_dir / "datasets"
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Missing datasets directory: {datasets_dir}")

    max_train_rows = _normalise_limit(max_train_rows)
    max_eval_rows = _normalise_limit(max_eval_rows)

    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("MODEL BENCHMARK")
    print(f"{'=' * 60}")
    print(f"  Input:      {datasets_dir}")
    print(f"  Output:     {benchmark_dir}")
    print(f"  Train cap:  {max_train_rows or 'none'}")
    print(f"  Eval cap:   {max_eval_rows or 'none'}")

    started = time.time()
    rows = _collect_benchmark_rows(
        datasets_dir,
        max_train_rows,
        max_eval_rows,
        random_state,
    )

    results = pd.DataFrame(rows)
    best = _select_best(results) if not results.empty else pd.DataFrame()

    results_path = benchmark_dir / "model_benchmark_results.csv"
    winners_path = benchmark_dir / "quick_benchmark_winners.csv"
    json_path = benchmark_dir / "model_benchmark_summary.json"
    text_path = benchmark_dir / "model_benchmark_summary.txt"

    results.to_csv(results_path, index=False)
    best.to_csv(winners_path, index=False)
    _write_text_summary(
        text_path,
        output_dir,
        max_train_rows,
        max_eval_rows,
        results,
        best,
    )

    summary = {
        "timestamp_utc": _utc_timestamp(),
        "output_dir": str(output_dir),
        "benchmark_dir": str(benchmark_dir),
        "max_train_rows": max_train_rows,
        "max_eval_rows": max_eval_rows,
        "selection": {
            "cls": f"highest {CLASSIFICATION_METRIC}",
            "reg": f"lowest {REGRESSION_METRIC}",
        },
        "elapsed_s": round(time.time() - started, 2),
        "benchmark_is_final_model_selection": False,
        "quick_benchmark_winners_diagnostic_only": _json_records(best),
        "files": {
            "results_csv": str(results_path),
            "quick_benchmark_winners_csv": str(winners_path),
            "summary_json": str(json_path),
            "summary_txt": str(text_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  {DIAGNOSTIC_WINNERS_LABEL}:")
    print(f"  {DIAGNOSTIC_NOTE}")
    if best.empty:
        print("    (none)")
    else:
        for _, row in best.iterrows():
            _print_winner(row)
    print(f"  Results: {results_path}")
    print(f"  Summary: {text_path}")
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for standalone benchmark runs."""
    parser = argparse.ArgumentParser(
        description="Run a quick sklearn benchmark on pipeline output matrices."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(PROJECT_ROOT / "artifacts" / "sample_run"),
        help="Pipeline output directory (default: artifacts/sample_run).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=200_000,
        help="Maximum training rows per task; 0 means no cap.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=100_000,
        help="Maximum validation/test rows per task; 0 means no cap.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and models.",
    )
    args = parser.parse_args(argv)

    run_benchmark(
        output_dir=args.output_dir,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[benchmark] failed: {exc}", file=sys.stderr)
        sys.exit(1)

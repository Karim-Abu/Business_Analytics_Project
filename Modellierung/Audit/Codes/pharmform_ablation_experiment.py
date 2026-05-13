#!/usr/bin/env python3
"""Run the controlled CLS PharmForm ablation experiment.

The script consumes the single union export created by the feature-engineering
pipeline and builds the four variants internally. It does not change the
project split logic or tune model hyperparameters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FE_DIR = PROJECT_ROOT / "Feature Engineering"
if str(FE_DIR) not in sys.path:
    sys.path.insert(0, str(FE_DIR))

import config as cfg  # noqa: E402
from feature_sets import get_feature_list  # noqa: E402
from Preprocessing.validation import (  # noqa: E402
    assert_no_duplicate_features,
    assert_no_forbidden_features,
)


UNION_FILES = {
    "train": "cls_pharmform_ablation_train_full.csv",
    "test": "cls_pharmform_ablation_test.csv",
    "val": "cls_pharmform_ablation_val.csv",
}

VARIANTS = [
    "CLS_PHARMFORM_V0_BASELINE",
    "CLS_PHARMFORM_V1_CLEAN",
    "CLS_PHARMFORM_V2_GROUP",
    "CLS_PHARMFORM_V3_CLEAN_AND_GROUP",
]

VARIANT_LABELS = {
    "CLS_PHARMFORM_V0_BASELINE": "V0 baseline without pharmForm_norm",
    "CLS_PHARMFORM_V1_CLEAN": "V1 with pharmForm_norm",
    "CLS_PHARMFORM_V2_GROUP": "V2 with pharmform_group and flags",
    "CLS_PHARMFORM_V3_CLEAN_AND_GROUP": "V3 with clean and group features",
}

KNOWN_CATEGORICAL = {
    "category_norm",
    "campaignIndex_norm",
    "pharmForm_norm",
    "pharmForm_clean",
    "pharmform_group",
    "pid_segment",
    "group12",
    "group34",
    "price_diff_bin",
    "discount_bin",
}

THRESHOLDS = np.round(np.arange(5, 81) / 100, 2)
CODE_FENCE = "```text"
MODEL_LOGREG = "logreg"
MODEL_HISTGB = "histgb"
MODEL_BOTH = "both"
MODEL_NAME_BY_KEY = {
    MODEL_LOGREG: "LogisticRegression",
    MODEL_HISTGB: "HistGradientBoostingClassifier",
}
HISTGB_NATIVE_CATEGORICAL_SUPPORTED = (
    "categorical_features" in inspect.signature(
        HistGradientBoostingClassifier).parameters
)
HISTGB_NATIVE_CATEGORICAL_NOTE = (
    "Finalmodellnaeherer nichtlinearer Ablation-Check mit ordinal kodierten "
    "Kategorien und nativer categorical_features-Maske fuer Spalten mit "
    "maximal 255 Auspraegungen."
)
HISTGB_NO_NATIVE_CATEGORICAL_NOTE = (
    "HistGB nutzt ordinal-kodierte Kategorien ohne native categorical_features; "
    "Ergebnisse sind finalmodellnah, aber nicht perfekt kategorial behandelt."
)
MODEL_NOTES = {
    MODEL_LOGREG: "Linearer Ablation-Check, nicht identisch mit finalem CLS-Modell.",
    MODEL_HISTGB: HISTGB_NATIVE_CATEGORICAL_NOTE
    if HISTGB_NATIVE_CATEGORICAL_SUPPORTED
    else HISTGB_NO_NATIVE_CATEGORICAL_NOTE,
}
TECHNICAL_SAMPLE_DECISION = "nur technischer Sample-Run, keine fachliche Entscheidung"
TECHNICAL_SAMPLE_MESSAGE = (
    "Diese Auswertung dient nur der technischen Validierung. "
    "Die fachliche Entscheidung erfolgt erst mit dem Full-Run."
)
MIN_TEST_ROWS_FOR_DECISION = 10_000
MIN_MAPPING_EVENTS_FOR_DECISION = 100_000
MIN_DECISION_DELTA = 0.002
FIXED_HISTGB_THRESHOLD = 0.22
PRIMARY_THRESHOLD_STRATEGY = "test_f1_opt"
HISTGB_MAX_NATIVE_CATEGORIES = 255
FULL_LOGREG_ROW_THRESHOLD = 500_000
FULL_LOGREG_WARNING = (
    "Full-Scale LogisticRegression can be slow because of one-hot encoded "
    "categorical features."
)
FULL_LOGREG_SKIP_MESSAGE = (
    "LogisticRegression skipped on full-scale run due to runtime risk."
)
PARTIAL_METRICS_CSV = "pharmform_ablation_metrics_partial.csv"
PARTIAL_METRICS_JSONL = "pharmform_ablation_metrics_partial.jsonl"
PARTIAL_ERROR_LOG = "partial_error_log.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CLS PharmForm ablation experiment."
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        required=True,
        help="Directory containing the cls_pharmform_ablation_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Modellierung" /
        "Audit" / "Daten" / "pharmform_ablation",
        help="Directory for metrics CSV/JSON and Markdown report.",
    )
    parser.add_argument(
        "--model",
        choices=[MODEL_LOGREG, MODEL_HISTGB, MODEL_BOTH],
        default=MODEL_LOGREG,
        help="Model family to run: logreg, histgb, or both (default: logreg).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print timestamped progress messages.",
    )
    parser.add_argument(
        "--allow-full-logreg",
        action="store_true",
        help="Allow LogisticRegression when TRAIN has more than 500,000 rows.",
    )
    return parser.parse_args()


def selected_model_keys(model_arg: str) -> list[str]:
    if model_arg == MODEL_BOTH:
        return [MODEL_LOGREG, MODEL_HISTGB]
    return [model_arg]


def timestamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log_verbose(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[{timestamp()}] {message}", flush=True)


def print_warning(message: str) -> None:
    print(f"[{timestamp()}] WARNING: {message}", flush=True)


def reset_partial_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in [PARTIAL_METRICS_CSV, PARTIAL_METRICS_JSONL, PARTIAL_ERROR_LOG]:
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def select_models_for_run(
    requested_model_keys: list[str],
    train_n_rows: int,
    allow_full_logreg: bool,
) -> tuple[list[str], list[dict[str, object]]]:
    selected: list[str] = []
    skipped: list[dict[str, object]] = []
    for model_key in requested_model_keys:
        if (
            model_key == MODEL_LOGREG
            and train_n_rows > FULL_LOGREG_ROW_THRESHOLD
            and not allow_full_logreg
        ):
            print_warning(FULL_LOGREG_WARNING)
            print_warning(FULL_LOGREG_SKIP_MESSAGE)
            skipped.append({
                "model": MODEL_NAME_BY_KEY[model_key],
                "reason": FULL_LOGREG_SKIP_MESSAGE,
                "train_n_rows": train_n_rows,
                "threshold_n_rows": FULL_LOGREG_ROW_THRESHOLD,
            })
            continue
        selected.append(model_key)
    return selected, skipped


def load_union_exports(exports_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split, file_name in UNION_FILES.items():
        path = exports_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Required union export missing: {path}")
        frame = pd.read_csv(path)
        if "order" not in frame.columns:
            raise ValueError(f"Target column 'order' missing in {path}")
        frames[split] = frame

    train_cols = list(frames["train"].columns)
    for split, frame in frames.items():
        if list(frame.columns) != train_cols:
            raise ValueError(
                f"Column mismatch between train and {split} union export."
            )
    return frames


def get_variant_features(variant: str, available_columns: set[str]) -> list[str]:
    features = get_feature_list(variant)
    assert_no_duplicate_features(features, variant)
    assert_no_forbidden_features(features, cfg.VERBOTEN_CLS, variant)
    missing = [feature for feature in features if feature not in available_columns]
    if missing:
        raise ValueError(f"{variant} missing feature columns: {missing}")
    return features


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical: list[str] = []
    numeric: list[str] = []
    for feature in features:
        dtype = str(df[feature].dtype)
        if (
            feature in KNOWN_CATEGORICAL
            or dtype == "object"
            or dtype.startswith("string")
            or dtype.startswith("category")
        ):
            categorical.append(feature)
        else:
            numeric.append(feature)
    return numeric, categorical


def build_logreg_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler(with_mean=False)),
            ], memory=None),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ], memory=None),
            categorical_features,
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = LogisticRegression(
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        random_state=cfg.SEED,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", classifier),
    ], memory=None)


def build_histgb_model(
    numeric_features: list[str],
    categorical_features: list[str],
    native_categorical_mask: list[bool] | None = None,
) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ], memory=None),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent", keep_empty_features=True)),
                ("ordinal", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                )),
            ], memory=None),
            categorical_features,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.0,
    )
    if HISTGB_NATIVE_CATEGORICAL_SUPPORTED:
        if native_categorical_mask is None:
            native_categorical_mask = [True] * len(categorical_features)
        classifier = HistGradientBoostingClassifier(
            random_state=cfg.SEED,
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=31,
            categorical_features=(
                [False] * len(numeric_features) + native_categorical_mask
            ),
        )
    else:
        classifier = HistGradientBoostingClassifier(
            random_state=cfg.SEED,
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=31,
        )
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", classifier),
    ], memory=None)


def build_model(
    model_key: str,
    numeric_features: list[str],
    categorical_features: list[str],
    native_categorical_mask: list[bool] | None = None,
) -> Pipeline:
    if model_key == MODEL_LOGREG:
        return build_logreg_model(numeric_features, categorical_features)
    if model_key == MODEL_HISTGB:
        return build_histgb_model(
            numeric_features,
            categorical_features,
            native_categorical_mask=native_categorical_mask,
        )
    raise ValueError(f"Unknown model key: {model_key}")


def histgb_native_categorical_plan(
    train: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[list[bool], list[dict[str, object]]]:
    native_mask: list[bool] = []
    high_cardinality: list[dict[str, object]] = []
    for feature in categorical_features:
        cardinality = int(train[feature].nunique(dropna=True))
        use_native = cardinality <= HISTGB_MAX_NATIVE_CATEGORIES
        native_mask.append(use_native)
        if not use_native:
            high_cardinality.append({
                "feature": feature,
                "cardinality": cardinality,
                "max_native_categories": HISTGB_MAX_NATIVE_CATEGORIES,
            })
    return native_mask, high_cardinality


def plan_histgb_categoricals_for_variant(
    model_key: str,
    train: pd.DataFrame,
    categorical_features: list[str],
    variant: str,
    verbose: bool,
) -> tuple[list[bool] | None, list[dict[str, object]]]:
    if model_key != MODEL_HISTGB or not HISTGB_NATIVE_CATEGORICAL_SUPPORTED:
        return None, []

    native_mask, high_cardinality = histgb_native_categorical_plan(
        train, categorical_features
    )
    log_verbose(
        verbose,
        "HistGB native categorical plan "
        f"{variant}: native={sum(native_mask)}, "
        f"ordinal_numeric={len(high_cardinality)}",
    )
    for item in high_cardinality:
        log_verbose(
            verbose,
            "HistGB high-cardinality categorical fallback "
            f"{variant}: {item['feature']} cardinality={item['cardinality']} "
            "treated as ordinal numeric",
        )
    return native_mask, high_cardinality


def positive_proba(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise ValueError("Trained classifier has no positive class 1.")
    positive_index = classes.index(1)
    return model.predict_proba(frame)[:, positive_index]


def metric_dict(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (proba >= threshold).astype(int)
    unique_classes = set(pd.Series(y_true).dropna().unique())
    roc_auc = np.nan
    pr_auc = np.nan
    if unique_classes == {0, 1}:
        roc_auc = float(roc_auc_score(y_true, proba))
        pr_auc = float(average_precision_score(y_true, proba))

    return {
        "n_rows": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def choose_test_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    rows = []
    for threshold in THRESHOLDS:
        metrics = metric_dict(y_true, proba, float(threshold))
        rows.append({
            "threshold": float(threshold),
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
    grid = pd.DataFrame(rows).sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    )
    return float(grid.iloc[0]["threshold"])


def threshold_strategies(model_key: str, test_threshold: float) -> list[tuple[str, float]]:
    strategies = [
        ("default_0_5", 0.5),
        (PRIMARY_THRESHOLD_STRATEGY, test_threshold),
    ]
    if model_key == MODEL_HISTGB:
        strategies.append(("fixed_0_22", FIXED_HISTGB_THRESHOLD))
    return strategies


def write_partial_outputs(
    output_dir: Path,
    all_rows: list[dict],
    completed_rows: list[dict],
    model_key: str,
    variant: str,
    verbose: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / PARTIAL_METRICS_CSV
    jsonl_path = output_dir / PARTIAL_METRICS_JSONL

    log_verbose(
        verbose,
        f"Schreiben der Zwischenresultate: {csv_path.name}, {jsonl_path.name}",
    )
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    entry = {
        "completed_at": timestamp(),
        "model": MODEL_NAME_BY_KEY[model_key],
        "model_key": model_key,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "n_metric_rows": len(completed_rows),
        "metrics": json.loads(pd.DataFrame(completed_rows).to_json(orient="records")),
    }
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            entry, ensure_ascii=False, allow_nan=True) + "\n")


def record_variant_error(
    output_dir: Path,
    model_key: str,
    variant: str,
    exc: BaseException,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    error = {
        "timestamp": timestamp(),
        "model": MODEL_NAME_BY_KEY[model_key],
        "model_key": model_key,
        "variant": variant,
        "variant_label": VARIANT_LABELS.get(variant, variant),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    path = output_dir / PARTIAL_ERROR_LOG
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{error['timestamp']}] {error['model']} / {variant}\n")
        handle.write(f"{error['error_type']}: {error['error_message']}\n")
        handle.write(str(error["traceback"]))
        handle.write("\n")
    return error


def evaluate_variant(
    model_key: str,
    variant: str,
    frames: dict[str, pd.DataFrame],
    features: list[str],
    verbose: bool = False,
) -> list[dict]:
    train = frames["train"]
    y_train = train["order"].astype(int)
    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            f"Training target for {variant} must contain both classes.")

    numeric_features, categorical_features = split_feature_types(
        train, features)
    log_verbose(verbose, f"Anzahl Features {variant}: {len(features)}")
    log_verbose(
        verbose,
        "Anzahl numerischer/kategorialer Features "
        f"{variant}: numeric={len(numeric_features)}, categorical={len(categorical_features)}",
    )
    native_categorical_mask, high_cardinality_categoricals = (
        plan_histgb_categoricals_for_variant(
            model_key, train, categorical_features, variant, verbose
        )
    )
    model = build_model(
        model_key,
        numeric_features,
        categorical_features,
        native_categorical_mask=native_categorical_mask,
    )
    train_start = time.perf_counter()
    log_verbose(
        verbose, f"Trainingsstart {MODEL_NAME_BY_KEY[model_key]} / {variant}")
    model.fit(train[features], y_train)
    training_seconds = time.perf_counter() - train_start
    log_verbose(
        verbose,
        "Trainingsende "
        f"{MODEL_NAME_BY_KEY[model_key]} / {variant} nach {training_seconds:.2f}s",
    )

    log_verbose(
        verbose, f"Evaluation TEST {MODEL_NAME_BY_KEY[model_key]} / {variant}")
    test_proba = positive_proba(model, frames["test"][features])
    test_threshold = choose_test_threshold(
        frames["test"]["order"].astype(int), test_proba)

    rows = []
    for split in ["test", "val"]:
        log_verbose(
            verbose,
            f"Evaluation {split.upper()} {MODEL_NAME_BY_KEY[model_key]} / {variant}",
        )
        y_true = frames[split]["order"].astype(int)
        proba = test_proba if split == "test" else positive_proba(
            model, frames[split][features])
        for threshold_label, threshold in threshold_strategies(model_key, test_threshold):
            metrics = metric_dict(y_true, proba, threshold)
            rows.append({
                "model": MODEL_NAME_BY_KEY[model_key],
                "model_key": model_key,
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "eval_split": split.upper(),
                "threshold_strategy": threshold_label,
                "n_features": len(features),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_histgb_native_categorical_features": (
                    int(sum(native_categorical_mask))
                    if native_categorical_mask is not None
                    else np.nan
                ),
                "n_histgb_ordinal_numeric_categorical_features": (
                    len(high_cardinality_categoricals)
                    if native_categorical_mask is not None
                    else np.nan
                ),
                "histgb_high_cardinality_categorical_features": ";".join(
                    f"{item['feature']}={item['cardinality']}"
                    for item in high_cardinality_categoricals
                ),
                "training_seconds": training_seconds,
                **metrics,
            })
    return rows


def load_mapping_coverage(exports_dir: Path) -> pd.DataFrame:
    coverage_path = exports_dir.parent / "audit" / "pharmform_mapping_coverage.csv"
    if coverage_path.exists():
        return pd.read_csv(coverage_path)
    return pd.DataFrame()


def _coverage_events_total(coverage: pd.DataFrame) -> int | None:
    if coverage.empty or "events_total" not in coverage.columns:
        return None
    value = pd.to_numeric(coverage.loc[0, "events_total"], errors="coerce")
    if pd.isna(value):
        return None
    return int(value)


def _technical_sample_reasons(test_n_rows: int, coverage: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    if test_n_rows < MIN_TEST_ROWS_FOR_DECISION:
        reasons.append(
            f"TEST n_rows={test_n_rows:,} < {MIN_TEST_ROWS_FOR_DECISION:,}"
        )
    events_total = _coverage_events_total(coverage)
    if events_total is not None and events_total < MIN_MAPPING_EVENTS_FOR_DECISION:
        reasons.append(
            "mapping_coverage events_total="
            f"{events_total:,} < {MIN_MAPPING_EVENTS_FOR_DECISION:,}"
        )
    return reasons


def _safe_delta(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return np.nan
    return float(left - right)


def _model_note_by_name(model_name: str) -> str:
    for key, name in MODEL_NAME_BY_KEY.items():
        if name == model_name:
            return MODEL_NOTES[key]
    return ""


def _primary_model_name(model_names: list[str]) -> str:
    histgb_name = MODEL_NAME_BY_KEY[MODEL_HISTGB]
    if histgb_name in model_names:
        return histgb_name
    return model_names[0]


def _best_summary_for_model(opt: pd.DataFrame, model_name: str) -> dict[str, object]:
    model_opt = opt[opt["model"] == model_name].copy()
    test_rows = model_opt[model_opt["eval_split"] == "TEST"].sort_values(
        ["f1", "precision", "recall"], ascending=[False, False, False]
    )
    if test_rows.empty:
        raise ValueError(f"No TEST metrics available for {model_name}.")

    best = test_rows.iloc[0]
    baseline_test = test_rows[
        test_rows["variant"] == "CLS_PHARMFORM_V0_BASELINE"
    ].iloc[0]
    val_rows = model_opt[model_opt["eval_split"] == "VAL"].set_index("variant")
    baseline_val = val_rows.loc["CLS_PHARMFORM_V0_BASELINE"]
    best_val = val_rows.loc[best["variant"]]

    return {
        "model": model_name,
        "model_note": _model_note_by_name(model_name),
        "best_variant_by_test_f1": str(best["variant"]),
        "best_variant_label": str(best["variant_label"]),
        "test_f1": float(best["f1"]),
        "baseline_test_f1": float(baseline_test["f1"]),
        "delta_test_f1_vs_v0": _safe_delta(best["f1"], baseline_test["f1"]),
        "val_f1": float(best_val["f1"]),
        "baseline_val_f1": float(baseline_val["f1"]),
        "delta_val_f1_vs_v0": _safe_delta(best_val["f1"], baseline_val["f1"]),
        "test_pr_auc": float(best["pr_auc"]),
        "baseline_test_pr_auc": float(baseline_test["pr_auc"]),
        "delta_test_pr_auc_vs_v0": _safe_delta(best["pr_auc"], baseline_test["pr_auc"]),
        "val_pr_auc": float(best_val["pr_auc"]),
        "baseline_val_pr_auc": float(baseline_val["pr_auc"]),
        "delta_val_pr_auc_vs_v0": _safe_delta(best_val["pr_auc"], baseline_val["pr_auc"]),
        "test_precision": float(best["precision"]),
        "baseline_test_precision": float(baseline_test["precision"]),
        "delta_test_precision_vs_v0": _safe_delta(best["precision"], baseline_test["precision"]),
        "val_precision": float(best_val["precision"]),
        "baseline_val_precision": float(baseline_val["precision"]),
        "delta_val_precision_vs_v0": _safe_delta(best_val["precision"], baseline_val["precision"]),
        "test_recall": float(best["recall"]),
        "baseline_test_recall": float(baseline_test["recall"]),
        "delta_test_recall_vs_v0": _safe_delta(best["recall"], baseline_test["recall"]),
        "val_recall": float(best_val["recall"]),
        "baseline_val_recall": float(baseline_val["recall"]),
        "delta_val_recall_vs_v0": _safe_delta(best_val["recall"], baseline_val["recall"]),
    }


def _variant_deltas_for_model(opt: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model_opt = opt[opt["model"] == model_name].copy()
    baseline = model_opt[model_opt["variant"] == "CLS_PHARMFORM_V0_BASELINE"]
    baseline_test = baseline[baseline["eval_split"] == "TEST"].iloc[0]
    baseline_val = baseline[baseline["eval_split"] == "VAL"].iloc[0]

    rows = []
    for variant in [v for v in VARIANTS if v != "CLS_PHARMFORM_V0_BASELINE"]:
        test_row = model_opt[(model_opt["variant"] == variant)
                             & (model_opt["eval_split"] == "TEST")].iloc[0]
        val_row = model_opt[(model_opt["variant"] == variant)
                            & (model_opt["eval_split"] == "VAL")].iloc[0]
        rows.append({
            "model": model_name,
            "variant": variant,
            "variant_label": str(test_row["variant_label"]),
            "delta_test_f1": _safe_delta(test_row["f1"], baseline_test["f1"]),
            "delta_val_f1": _safe_delta(val_row["f1"], baseline_val["f1"]),
            "delta_test_pr_auc": _safe_delta(test_row["pr_auc"], baseline_test["pr_auc"]),
            "delta_val_pr_auc": _safe_delta(val_row["pr_auc"], baseline_val["pr_auc"]),
            "delta_test_precision": _safe_delta(test_row["precision"], baseline_test["precision"]),
            "delta_val_precision": _safe_delta(val_row["precision"], baseline_val["precision"]),
            "delta_test_recall": _safe_delta(test_row["recall"], baseline_test["recall"]),
            "delta_val_recall": _safe_delta(val_row["recall"], baseline_val["recall"]),
        })
    return pd.DataFrame(rows)


def _histgb_challenger_decision(opt: pd.DataFrame) -> dict[str, object]:
    model_name = MODEL_NAME_BY_KEY[MODEL_HISTGB]
    deltas = _variant_deltas_for_model(opt, model_name)
    deltas["passes_f1"] = (
        (deltas["delta_test_f1"] >= MIN_DECISION_DELTA)
        & (deltas["delta_val_f1"] >= 0)
    )
    deltas["passes_pr_auc"] = (
        (deltas["delta_test_pr_auc"] >= MIN_DECISION_DELTA)
        & (deltas["delta_val_pr_auc"] >= 0)
    )
    qualified = deltas[deltas["passes_f1"] | deltas["passes_pr_auc"]].copy()
    if qualified.empty:
        return {
            "decision": "nicht ins finale CLS-Modell übernehmen; nur interpretativ dokumentieren",
            "winning_variant": None,
            "candidate_deltas": json.loads(deltas.to_json(orient="records")),
        }

    qualified["decision_score"] = qualified[[
        "delta_test_f1", "delta_test_pr_auc",
    ]].max(axis=1)
    winner = qualified.sort_values(
        ["decision_score", "delta_val_f1", "delta_val_pr_auc", "delta_test_recall"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return {
        "decision": "als Challenger-Feature prüfen / potenziell behalten",
        "winning_variant": str(winner["variant"]),
        "winning_variant_label": str(winner["variant_label"]),
        "candidate_deltas": json.loads(deltas.to_json(orient="records")),
    }


def decide(metrics: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, object]:
    opt = metrics[metrics["threshold_strategy"]
                  == PRIMARY_THRESHOLD_STRATEGY].copy()
    if opt.empty:
        raise ValueError(
            "No optimized-threshold metrics available for decision.")

    model_names = sorted(opt["model"].unique().tolist())
    primary_model = _primary_model_name(model_names)
    test_n_rows = int(opt[opt["eval_split"] == "TEST"]["n_rows"].min())
    sample_reasons = _technical_sample_reasons(test_n_rows, coverage)
    best_by_model = [_best_summary_for_model(opt, model_name)
                     for model_name in model_names]

    if sample_reasons:
        decision_text = TECHNICAL_SAMPLE_DECISION
        histgb_decision = None
    elif MODEL_NAME_BY_KEY[MODEL_HISTGB] in model_names:
        histgb_decision = _histgb_challenger_decision(opt)
        decision_text = histgb_decision["decision"]
    else:
        histgb_decision = None
        decision_text = "nur linearer Ablation-Check, keine finale CLS-Feature-Entscheidung"

    primary_summary = next(item for item in best_by_model
                           if item["model"] == primary_model)
    return {
        "decision": decision_text,
        "is_technical_sample_run": bool(sample_reasons),
        "technical_sample_reasons": sample_reasons,
        "test_n_rows": test_n_rows,
        "mapping_events_total": _coverage_events_total(coverage),
        "models_run": model_names,
        "primary_decision_model": primary_model,
        "histgb_native_categorical_supported": HISTGB_NATIVE_CATEGORICAL_SUPPORTED,
        "model": primary_model,
        "model_note": _model_note_by_name(primary_model),
        "model_notes": {name: _model_note_by_name(name) for name in model_names},
        "best_by_model": best_by_model,
        "histgb_challenger": histgb_decision,
        "best_variant_by_test_f1": primary_summary["best_variant_by_test_f1"],
        "best_variant_label": primary_summary["best_variant_label"],
        "best_test_f1": primary_summary["test_f1"],
        "baseline_test_f1": primary_summary["baseline_test_f1"],
        "delta_test_f1_vs_v0": primary_summary["delta_test_f1_vs_v0"],
        "best_val_f1": primary_summary["val_f1"],
        "baseline_val_f1": primary_summary["baseline_val_f1"],
        "delta_val_f1_vs_v0": primary_summary["delta_val_f1_vs_v0"],
        "delta_test_pr_auc_vs_v0": primary_summary["delta_test_pr_auc_vs_v0"],
        "delta_val_pr_auc_vs_v0": primary_summary["delta_val_pr_auc_vs_v0"],
    }


def _append_table(lines: list[str], frame: pd.DataFrame, cols: list[str]) -> None:
    if frame.empty:
        lines.append("No rows available.")
        return
    lines.append(CODE_FENCE)
    lines.append(frame.loc[:, cols].to_string(index=False))
    lines.append("```")


def _append_report_notes(lines: list[str], model_names: list[str]) -> None:
    lines.extend([
        "",
        "## Notes",
        "",
        "- LogisticRegression uses median numeric imputation, sparse one-hot categoricals, scaling, and class_weight=balanced.",
        "- HistGradientBoostingClassifier uses median numeric imputation and ordinal categorical encoding; no dense one-hot matrix is created.",
        "- HistGB parameters: max_iter=100, learning_rate=0.1, max_leaf_nodes=31, random_state=cfg.SEED.",
    ])
    if HISTGB_NATIVE_CATEGORICAL_SUPPORTED:
        lines.append(
            "- HistGB native categorical_features: enabled for ordinal-encoded categorical columns with cardinality <= 255; higher-cardinality categorical columns are ordinal-encoded and treated as numeric because sklearn HistGB cannot use them as native categorical features."
        )
    else:
        lines.append(f"- {HISTGB_NO_NATIVE_CATEGORICAL_NOTE}")
    for model_name in model_names:
        note = _model_note_by_name(model_name)
        if note:
            lines.append(f"- {model_name}: {note}")
    lines.extend([
        "- No hyperparameter search was run.",
        "- Existing train/test/validation split is used as exported by the pipeline.",
    ])


def _append_skipped_models_section(
    lines: list[str], skipped_models: list[dict[str, object]]
) -> None:
    if not skipped_models:
        return
    lines.extend(["", "## Skipped Models", ""])
    for skipped in skipped_models:
        lines.append(f"- {skipped['model']}: {skipped['reason']}")


def _append_errors_section(
    lines: list[str], errors: list[dict[str, object]]
) -> None:
    if not errors:
        return
    lines.extend([
        "",
        "## Errors",
        "",
        f"Details were written to `{PARTIAL_ERROR_LOG}`.",
        "",
    ])
    for error in errors:
        lines.append(
            "- "
            f"{error['model']} / {error['variant']}: "
            f"{error['error_type']}: {error['error_message']}"
        )


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    decision: dict[str, object],
    errors: list[dict[str, object]] | None = None,
    skipped_models: list[dict[str, object]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = errors or []
    skipped_models = skipped_models or []

    metrics_path = output_dir / "pharmform_ablation_metrics.csv"
    json_path = output_dir / "pharmform_ablation_metrics.json"
    report_path = output_dir / "pharmform_ablation_report.md"

    metrics.to_csv(metrics_path, index=False)
    payload = {
        "decision": decision,
        "skipped_models": skipped_models,
        "errors": errors,
        "partial_outputs": {
            "metrics_csv": str(output_dir / PARTIAL_METRICS_CSV),
            "metrics_jsonl": str(output_dir / PARTIAL_METRICS_JSONL),
            "error_log": str(output_dir / PARTIAL_ERROR_LOG),
        },
        "mapping_coverage": json.loads(coverage.to_json(orient="records"))
        if not coverage.empty else [],
        "metrics": json.loads(metrics.to_json(orient="records")),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    opt = metrics[metrics["threshold_strategy"]
                  == PRIMARY_THRESHOLD_STRATEGY].copy()
    metric_cols = [
        "model", "variant", "eval_split", "threshold", "n_features",
        "precision", "recall", "f1", "roc_auc", "pr_auc",
    ]
    summary = pd.DataFrame(decision["best_by_model"])
    summary_cols = [
        "model", "best_variant_by_test_f1", "delta_test_f1_vs_v0",
        "delta_val_f1_vs_v0", "delta_test_pr_auc_vs_v0", "delta_val_pr_auc_vs_v0",
        "delta_test_precision_vs_v0", "delta_test_recall_vs_v0",
    ]

    lines = [
        "# PharmForm Ablation Report",
        "",
        "## Decision",
        "",
        f"- Decision: {decision['decision']}",
        f"- Primary decision model: {decision['primary_decision_model']}",
        f"- Models run: {', '.join(decision['models_run'])}",
        f"- Best variant by TEST F1 on primary model: {decision['best_variant_by_test_f1']}",
        f"- TEST F1 delta vs V0: {decision['delta_test_f1_vs_v0']:.6f}",
        f"- VAL F1 delta vs V0: {decision['delta_val_f1_vs_v0']:.6f}",
        f"- TEST PR-AUC delta vs V0: {decision['delta_test_pr_auc_vs_v0']:.6f}",
        f"- VAL PR-AUC delta vs V0: {decision['delta_val_pr_auc_vs_v0']:.6f}",
    ]
    _append_skipped_models_section(lines, skipped_models)
    _append_errors_section(lines, errors)
    if decision["is_technical_sample_run"]:
        lines.extend([
            f"- {TECHNICAL_SAMPLE_MESSAGE}",
            f"- Technical sample reason(s): {'; '.join(decision['technical_sample_reasons'])}",
        ])
    elif decision.get("histgb_challenger"):
        challenger = decision["histgb_challenger"]
        if challenger.get("winning_variant"):
            lines.append(
                f"- HistGB challenger variant: {challenger['winning_variant']}")

    lines.extend([
        "",
        "## Partial Outputs",
        "",
        f"- Partial metrics CSV: {PARTIAL_METRICS_CSV}",
        f"- Partial metrics JSONL: {PARTIAL_METRICS_JSONL}",
        f"- Partial error log: {PARTIAL_ERROR_LOG}",
        "",
        "## Best Variant By Model",
        "",
    ])
    _append_table(lines, summary, summary_cols)

    challenger = decision.get("histgb_challenger")
    if challenger and challenger.get("candidate_deltas"):
        delta_frame = pd.DataFrame(challenger["candidate_deltas"])
        delta_cols = [
            "variant", "delta_test_f1", "delta_val_f1",
            "delta_test_pr_auc", "delta_val_pr_auc",
            "delta_test_precision", "delta_test_recall",
        ]
        lines.extend([
            "",
            "## HistGB Candidate Deltas",
            "",
            f"Decision threshold: TEST F1 or TEST PR-AUC delta >= {MIN_DECISION_DELTA:.3f}; corresponding VAL delta must be non-negative.",
            "",
        ])
        _append_table(lines, delta_frame, delta_cols)

    lines.extend([
        "",
        "## Mapping Coverage",
        "",
    ])
    if coverage.empty:
        lines.append("Mapping coverage audit not found next to exports.")
    else:
        lines.append(CODE_FENCE)
        lines.append(coverage.to_string(index=False))
        lines.append("```")

    lines.extend([
        "",
        "## Variant Metrics",
        "",
        "Threshold strategy `test_f1_opt` optimizes on TEST and applies the same threshold to VAL.",
    ])
    for model_name in decision["models_run"]:
        lines.extend([
            "",
            f"### {model_name}",
            "",
        ])
        _append_table(lines, opt[opt["model"] == model_name], metric_cols)

    lines.extend([
        "",
        "## Default Threshold Check",
        "",
    ])
    _append_table(
        lines,
        metrics[metrics["threshold_strategy"] == "default_0_5"],
        metric_cols,
    )

    fixed = metrics[metrics["threshold_strategy"] == "fixed_0_22"]
    if not fixed.empty:
        lines.extend([
            "",
            "## Fixed Threshold 0.22 Check",
            "",
        ])
        _append_table(lines, fixed, metric_cols)

    _append_report_notes(lines, decision["models_run"])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved metrics CSV:  {metrics_path}")
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved report:       {report_path}")


def write_no_metrics_report(
    output_dir: Path,
    coverage: pd.DataFrame,
    errors: list[dict[str, object]],
    skipped_models: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "pharmform_ablation_metrics.csv"
    json_path = output_dir / "pharmform_ablation_metrics.json"
    report_path = output_dir / "pharmform_ablation_report.md"

    pd.DataFrame().to_csv(metrics_path, index=False)
    payload = {
        "decision": "No ablation metrics were produced.",
        "skipped_models": skipped_models,
        "errors": errors,
        "partial_outputs": {
            "metrics_csv": str(output_dir / PARTIAL_METRICS_CSV),
            "metrics_jsonl": str(output_dir / PARTIAL_METRICS_JSONL),
            "error_log": str(output_dir / PARTIAL_ERROR_LOG),
        },
        "mapping_coverage": json.loads(coverage.to_json(orient="records"))
        if not coverage.empty else [],
        "metrics": [],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# PharmForm Ablation Report",
        "",
        "## Decision",
        "",
        "- No ablation metrics were produced.",
    ]
    _append_skipped_models_section(lines, skipped_models)
    _append_errors_section(lines, errors)
    lines.extend([
        "",
        "## Mapping Coverage",
        "",
    ])
    if coverage.empty:
        lines.append("Mapping coverage audit not found next to exports.")
    else:
        lines.append(CODE_FENCE)
        lines.append(coverage.to_string(index=False))
        lines.append("```")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved metrics CSV:  {metrics_path}")
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved report:       {report_path}")


def main() -> None:
    args = parse_args()
    exports_dir = args.exports_dir.resolve()
    output_dir = args.output_dir.resolve()
    reset_partial_outputs(output_dir)

    log_verbose(args.verbose, f"CSV-Ladevorgang startet: {exports_dir}")
    frames = load_union_exports(exports_dir)
    log_verbose(
        args.verbose,
        "CSV-Ladevorgang abgeschlossen: "
        f"TRAIN={frames['train'].shape}, TEST={frames['test'].shape}, "
        f"VAL={frames['val'].shape}",
    )
    available_columns = set(frames["train"].columns)
    coverage = load_mapping_coverage(exports_dir)
    if coverage.empty:
        log_verbose(args.verbose, "Mapping coverage: nicht gefunden")
    else:
        log_verbose(
            args.verbose,
            "Mapping coverage geladen: "
            f"events_total={_coverage_events_total(coverage)}",
        )

    requested_model_keys = selected_model_keys(args.model)
    model_keys, skipped_models = select_models_for_run(
        requested_model_keys,
        len(frames["train"]),
        args.allow_full_logreg,
    )

    rows = []
    errors: list[dict[str, object]] = []
    for model_key in model_keys:
        model_start = time.perf_counter()
        log_verbose(
            args.verbose, f"Start Modell {MODEL_NAME_BY_KEY[model_key]}")
        for variant in VARIANTS:
            variant_start = time.perf_counter()
            try:
                features = get_variant_features(variant, available_columns)
                log_verbose(
                    args.verbose,
                    f"Start Variante {MODEL_NAME_BY_KEY[model_key]} / {variant}: "
                    f"{len(features)} Features",
                )
                completed_rows = evaluate_variant(
                    model_key,
                    variant,
                    frames,
                    features,
                    verbose=args.verbose,
                )
                rows.extend(completed_rows)
                write_partial_outputs(
                    output_dir,
                    rows,
                    completed_rows,
                    model_key,
                    variant,
                    args.verbose,
                )
                log_verbose(
                    args.verbose,
                    "Ende Variante "
                    f"{MODEL_NAME_BY_KEY[model_key]} / {variant} nach "
                    f"{time.perf_counter() - variant_start:.2f}s",
                )
            except Exception as exc:  # noqa: BLE001
                error = record_variant_error(
                    output_dir, model_key, variant, exc)
                errors.append(error)
                print_warning(
                    f"{MODEL_NAME_BY_KEY[model_key]} / {variant} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
        log_verbose(
            args.verbose,
            f"Ende Modell {MODEL_NAME_BY_KEY[model_key]} nach "
            f"{time.perf_counter() - model_start:.2f}s",
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        write_no_metrics_report(output_dir, coverage, errors, skipped_models)
        raise RuntimeError(
            "No ablation metrics were produced. See partial_error_log.txt "
            "and skipped model notes in the output directory."
        )
    decision = decide(metrics, coverage)
    write_report(
        output_dir,
        metrics,
        coverage,
        decision,
        errors=errors,
        skipped_models=skipped_models,
    )


if __name__ == "__main__":
    main()

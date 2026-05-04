from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 20)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "Daten" / "reg_test_predictions_orange.csv"
OUTPUT_DIR = BASE_DIR

TARGET_COLUMN = "quantity"
LOAD_SKIPROWS = [1, 2]
PLOT_MAIN_LIMIT = 30.0

MODEL_SPECS = {
    "constant_baseline": {
        "label": "Orange Constant baseline",
        "exact_names": ["Constant baseline (Orange: mean) - see Note for 'always 1'"],
        "token_groups": [["constant", "baseline"], ["orange", "mean"]],
        "source_type": "baseline_export",
        "clip_if_below_one": False,
        "scatter_file": None,
    },
    "random_forest": {
        "label": "Random Forest",
        "exact_names": ["Random Forest"],
        "token_groups": [["random", "forest"]],
        "source_type": "orange_model",
        "clip_if_below_one": True,
        "scatter_file": "reg_orange_scatter_random_forest.png",
    },
    "gradient_boosting": {
        "label": "Gradient Boosting Regressor",
        "exact_names": ["Gradient Boosting Regressor (sklearn default)", "Gradient Boosting Regressor"],
        "token_groups": [["gradient", "boosting"], ["boosting", "regressor"]],
        "source_type": "orange_model",
        "clip_if_below_one": True,
        "scatter_file": "reg_orange_scatter_gradient_boosting.png",
    },
    "ridge": {
        "label": "Ridge Regression",
        "exact_names": ["Ridge Regression", "Ridge"],
        "token_groups": [["ridge", "regression"], ["ridge"]],
        "source_type": "orange_model",
        "clip_if_below_one": True,
        "scatter_file": "reg_orange_scatter_ridge.png",
    },
}

SEGMENT_DEFINITIONS = {
    "all": "Alle Beobachtungen",
    "quantity_eq_1": "Nur quantity = 1",
    "quantity_gt_1": "Nur quantity > 1",
    "quantity_ge_3": "Nur quantity >= 3",
}

OVERALL_METRICS_FILE = OUTPUT_DIR / "reg_orange_metrics_overall.csv"
SEGMENTED_METRICS_FILE = OUTPUT_DIR / "reg_orange_metrics_segmented.csv"
BASELINE_COMPARISON_FILE = OUTPUT_DIR / "reg_orange_baseline_comparison.csv"
DIAGNOSTICS_FILE = OUTPUT_DIR / "reg_orange_prediction_diagnostics.csv"


def normalize_name(name):
    text = str(name).strip().lower()
    for char in ["(", ")", "[", "]", "{", "}", "-", "_", ":", ",", ".", "/", "\\", "'", '"']:
        text = text.replace(char, " ")
    return " ".join(text.split())


def detect_orange_metadata(df):
    if df.empty:
        return False

    preview = df.head(3).astype(str).apply(
        lambda col: col.str.strip().str.lower())
    metadata_hits = int(preview.isin(
        ["continuous", "discrete", "class", "meta"]).to_numpy().sum())

    quantity_preview_non_numeric = False
    if TARGET_COLUMN in df.columns:
        quantity_preview = pd.to_numeric(
            df[TARGET_COLUMN].head(3), errors="coerce")
        quantity_preview_non_numeric = bool(quantity_preview.isna().any())

    return metadata_hits >= 2 or quantity_preview_non_numeric


def load_orange_predictions(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print("Lade nur die Orange TEST-Prediction-Datei:")
    print(file_path)

    df_plain = pd.read_csv(file_path)
    strategy = "pd.read_csv(...)"
    fallback_reason = None

    if detect_orange_metadata(df_plain):
        fallback_reason = "Orange-Metadaten in den ersten Zeilen erkannt."
    elif TARGET_COLUMN not in df_plain.columns:
        fallback_reason = f"Target-Spalte '{TARGET_COLUMN}' fehlt nach plain read_csv()."
    else:
        try:
            pd.to_numeric(df_plain[TARGET_COLUMN], errors="raise")
        except Exception as exc:
            fallback_reason = (
                f"Numerische Konvertierung von '{TARGET_COLUMN}' nach plain read_csv() fehlgeschlagen: {exc}"
            )

    if fallback_reason is not None:
        print("Fallback-Laden aktiviert:")
        print(f"- {fallback_reason}")
        df_loaded = pd.read_csv(file_path, skiprows=LOAD_SKIPROWS)
        strategy = f"pd.read_csv(..., skiprows={LOAD_SKIPROWS})"
    else:
        df_loaded = df_plain

    print(f"Verwendete Ladestrategie: {strategy}")
    print(f"df.shape: {df_loaded.shape}")
    print("df.columns.tolist():")
    print(df_loaded.columns.tolist())
    print("df.head():")
    print(df_loaded.head())

    return df_loaded, strategy


def validate_target(df, column_name):
    if column_name not in df.columns:
        raise KeyError(
            f"Target-Spalte '{column_name}' fehlt in der geladenen Datei.")

    target = pd.to_numeric(df[column_name], errors="raise").astype(float)
    if target.isna().any():
        raise ValueError(
            f"Target-Spalte '{column_name}' enthaelt fehlende Werte nach der Konvertierung.")

    invalid_mask = target < 1
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        invalid_examples = target[invalid_mask].head().tolist()
        raise ValueError(
            f"Target-Spalte '{column_name}' enthaelt {invalid_count} Werte < 1. Beispiele: {invalid_examples}"
        )

    summary = {
        "share_quantity_eq_1": float((target == 1).mean()),
        "share_quantity_gt_1": float((target > 1).mean()),
        "median_quantity": float(target.median()),
        "mean_quantity": float(target.mean()),
        "max_quantity": float(target.max()),
    }

    print("Target-Validierung fuer quantity:")
    print(f"- Spalte vorhanden: {column_name in df.columns}")
    print("- Numerisch: Ja")
    print("- Keine Werte < 1: Ja")
    print(
        f"- Anteil quantity = 1: {summary['share_quantity_eq_1']:.6f} ({summary['share_quantity_eq_1'] * 100:.2f}%)")
    print(
        f"- Anteil quantity > 1: {summary['share_quantity_gt_1']:.6f} ({summary['share_quantity_gt_1'] * 100:.2f}%)")
    print(f"- Median quantity: {summary['median_quantity']:.6f}")
    print(f"- Mean quantity: {summary['mean_quantity']:.6f}")
    print(f"- Max quantity: {summary['max_quantity']:.6f}")

    return target, summary


def resolve_single_column(columns, exact_names, token_groups):
    for exact_name in exact_names:
        exact_matches = [column for column in columns if column == exact_name]
        if len(exact_matches) == 1:
            return exact_matches[0], "exact"

        normalized_matches = [
            column for column in columns if normalize_name(column) == normalize_name(exact_name)
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0], "normalized_exact"
        if len(normalized_matches) > 1:
            return None, f"ambiguous_normalized_exact={normalized_matches}"

    heuristic_matches = []
    for column in columns:
        normalized_column = normalize_name(column)
        for tokens in token_groups:
            if all(token in normalized_column for token in tokens):
                heuristic_matches.append(column)
                break

    heuristic_matches = list(dict.fromkeys(heuristic_matches))
    if len(heuristic_matches) == 1:
        return heuristic_matches[0], "heuristic"
    if len(heuristic_matches) > 1:
        return None, f"ambiguous_heuristic={heuristic_matches}"

    return None, "not_found"


def resolve_prediction_columns(columns):
    resolved = {}
    resolution_notes = {}

    for model_key, spec in MODEL_SPECS.items():
        column_name, note = resolve_single_column(
            columns, spec["exact_names"], spec["token_groups"])
        resolved[model_key] = column_name
        resolution_notes[model_key] = note

    detected_model_count = int(
        sum(resolved[key] is not None for key in [
            "random_forest", "gradient_boosting", "ridge"])
    )
    if detected_model_count == 0:
        raise KeyError(
            "Keine der erwarteten Orange-REG-Prediction-Spalten wurde erkannt.")

    print("Verwendete Prediction-Spalten (Dictionary):")
    print(resolved)
    print("Aufloesungsnotizen:")
    for model_key, note in resolution_notes.items():
        print(f"- {model_key}: {note}")

    return resolved


def coerce_prediction_series(df, column_name):
    if column_name not in df.columns:
        raise KeyError(
            f"Prediction-Spalte '{column_name}' fehlt in der Datei.")
    return pd.to_numeric(df[column_name], errors="raise").astype(float)


def compute_mape_percent(y_true, y_pred):
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true_array - y_pred_array) / y_true_array)) * 100.0)


def compute_overall_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = compute_mape_percent(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(np.asarray(y_pred, dtype=float) -
                 np.asarray(y_true, dtype=float)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "bias": bias,
    }


def build_segment_masks(y_true):
    return {
        "all": pd.Series(True, index=y_true.index),
        "quantity_eq_1": y_true == 1,
        "quantity_gt_1": y_true > 1,
        "quantity_ge_3": y_true >= 3,
    }


def evaluate_entries(y_true, entries):
    rows = []
    for entry in entries:
        metric_values = compute_overall_metrics(y_true, entry["predictions"])
        rows.append(
            {
                "model_key": entry["model_key"],
                "model_label": entry["model_label"],
                "evaluation_label": entry["evaluation_label"],
                "source_type": entry["source_type"],
                "prediction_column": entry["prediction_column"],
                "evaluation_variant": entry["evaluation_variant"],
                "n_rows": int(len(y_true)),
                "clip_floor": entry["clip_floor"],
                **metric_values,
            }
        )
    return pd.DataFrame(rows)


def evaluate_segments(y_true, entries):
    segment_masks = build_segment_masks(y_true)
    rows = []

    for entry in entries:
        for segment_key, mask in segment_masks.items():
            actual_segment = y_true[mask]
            pred_segment = entry["predictions"][mask]
            n_rows = int(mask.sum())

            if n_rows == 0:
                mae = np.nan
                rmse = np.nan
                bias = np.nan
            else:
                mse = float(mean_squared_error(actual_segment, pred_segment))
                mae = float(mean_absolute_error(actual_segment, pred_segment))
                rmse = float(np.sqrt(mse))
                bias = float(np.mean(np.asarray(
                    pred_segment, dtype=float) - np.asarray(actual_segment, dtype=float)))

            rows.append(
                {
                    "model_key": entry["model_key"],
                    "model_label": entry["model_label"],
                    "evaluation_label": entry["evaluation_label"],
                    "source_type": entry["source_type"],
                    "prediction_column": entry["prediction_column"],
                    "evaluation_variant": entry["evaluation_variant"],
                    "segment": segment_key,
                    "segment_description": SEGMENT_DEFINITIONS[segment_key],
                    "n_rows": n_rows,
                    "mae": mae,
                    "rmse": rmse,
                    "bias": bias,
                }
            )

    return pd.DataFrame(rows)


def classify_delta(delta_value, higher_is_better=False, tolerance=1e-12):
    if pd.isna(delta_value):
        return "not_applied"
    if abs(delta_value) <= tolerance:
        return "no_change"
    if higher_is_better:
        return "improved" if delta_value > 0 else "worsened"
    return "improved" if delta_value < 0 else "worsened"


def build_prediction_diagnostics(y_true, entries, overall_df):
    overall_lookup = overall_df.set_index(["model_key", "evaluation_variant"])
    rows = []

    for entry in entries:
        if entry["evaluation_variant"] != "raw":
            continue

        predictions = entry["predictions"]
        row = {
            "model_key": entry["model_key"],
            "model_label": entry["model_label"],
            "evaluation_label": entry["evaluation_label"],
            "source_type": entry["source_type"],
            "prediction_column": entry["prediction_column"],
            "n_rows": int(len(predictions)),
            "min_prediction": float(predictions.min()),
            "max_prediction": float(predictions.max()),
            "mean_prediction": float(predictions.mean()),
            "count_prediction_lt_zero": int((predictions < 0).sum()),
            "share_prediction_lt_zero": float((predictions < 0).mean()),
            "count_prediction_lt_one": int((predictions < 1).sum()),
            "share_prediction_lt_one": float((predictions < 1).mean()),
            "raw_bias": float(np.mean(np.asarray(predictions, dtype=float) - np.asarray(y_true, dtype=float))),
            "clipping_applied": False,
            "delta_mse_clipped_minus_raw": np.nan,
            "delta_rmse_clipped_minus_raw": np.nan,
            "delta_mae_clipped_minus_raw": np.nan,
            "delta_mape_clipped_minus_raw": np.nan,
            "delta_r2_clipped_minus_raw": np.nan,
            "clip_effect_mse": "not_applied",
            "clip_effect_rmse": "not_applied",
            "clip_effect_mae": "not_applied",
            "clip_effect_mape": "not_applied",
            "clip_effect_r2": "not_applied",
        }

        if entry["source_type"] == "orange_model" and (entry["model_key"], "clipped_at_1") in overall_lookup.index:
            raw_metrics = overall_lookup.loc[(entry["model_key"], "raw")]
            clipped_metrics = overall_lookup.loc[(
                entry["model_key"], "clipped_at_1")]

            delta_mse = float(clipped_metrics["mse"] - raw_metrics["mse"])
            delta_rmse = float(clipped_metrics["rmse"] - raw_metrics["rmse"])
            delta_mae = float(clipped_metrics["mae"] - raw_metrics["mae"])
            delta_mape = float(clipped_metrics["mape"] - raw_metrics["mape"])
            delta_r2 = float(clipped_metrics["r2"] - raw_metrics["r2"])

            row.update(
                {
                    "clipping_applied": True,
                    "delta_mse_clipped_minus_raw": delta_mse,
                    "delta_rmse_clipped_minus_raw": delta_rmse,
                    "delta_mae_clipped_minus_raw": delta_mae,
                    "delta_mape_clipped_minus_raw": delta_mape,
                    "delta_r2_clipped_minus_raw": delta_r2,
                    "clip_effect_mse": classify_delta(delta_mse, higher_is_better=False),
                    "clip_effect_rmse": classify_delta(delta_rmse, higher_is_better=False),
                    "clip_effect_mae": classify_delta(delta_mae, higher_is_better=False),
                    "clip_effect_mape": classify_delta(delta_mape, higher_is_better=False),
                    "clip_effect_r2": classify_delta(delta_r2, higher_is_better=True),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def determine_plot_limit(y_true, y_pred):
    max_value = max(float(np.max(y_true)), float(np.max(y_pred)))
    return float(min(PLOT_MAIN_LIMIT, max(5.0, np.ceil(max_value))))


def save_scatterplot(y_true, y_pred, model_label, output_path):
    plot_limit = determine_plot_limit(y_true, y_pred)
    outlier_count = int((y_true > PLOT_MAIN_LIMIT).sum())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=18, alpha=0.22,
               color="#1f77b4", edgecolors="none")
    ax.plot([0, plot_limit], [0, plot_limit], linestyle="--",
            linewidth=1.2, color="black", label="y = x")

    ax.set_xlim(0, plot_limit)
    ax.set_ylim(0, plot_limit)
    ax.set_xlabel("Actual quantity")
    ax.set_ylabel("Predicted quantity")
    ax.set_title(f"Orange REG: {model_label}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    note_text = f"Actual quantity > 30: {outlier_count}"
    ax.text(
        0.02,
        0.98,
        note_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white",
              "alpha": 0.85, "edgecolor": "#999999"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatterplot gespeichert: {output_path.name}")


def save_requested_scatterplots(y_true, entries):
    for entry in entries:
        if entry["evaluation_variant"] != "raw":
            continue
        if entry["model_key"] not in MODEL_SPECS:
            continue

        scatter_file = MODEL_SPECS[entry["model_key"]]["scatter_file"]
        if scatter_file is None:
            continue

        output_path = OUTPUT_DIR / scatter_file
        save_scatterplot(y_true, entry["predictions"],
                         entry["model_label"], output_path)


def round_float_columns(df, digits=6):
    rounded = df.copy()
    float_columns = rounded.select_dtypes(
        include=["float16", "float32", "float64"]).columns
    rounded[float_columns] = rounded[float_columns].round(digits)
    return rounded


def build_baseline_comparison(overall_df):
    orange_candidates = overall_df[
        (overall_df["source_type"] == "orange_model") & (
            overall_df["evaluation_variant"] == "raw")
    ].copy()
    if orange_candidates.empty:
        raise ValueError(
            "Keine raw Orange-Modelle fuer den Baseline-Vergleich verfuegbar.")

    orange_candidates = orange_candidates.sort_values(
        by=["rmse", "mae", "mse", "mape", "r2"],
        ascending=[True, True, True, True, False],
    )
    best_orange_raw = orange_candidates.iloc[0]

    baselines = overall_df[
        (overall_df["source_type"].isin(
            ["baseline_export", "control_baseline"]))
        & (overall_df["evaluation_variant"] == "raw")
    ].copy()

    rows = []
    for _, baseline_row in baselines.iterrows():
        rows.append(
            {
                "best_orange_model": best_orange_raw["model_label"],
                "best_orange_prediction_column": best_orange_raw["prediction_column"],
                "best_orange_rmse": float(best_orange_raw["rmse"]),
                "best_orange_mae": float(best_orange_raw["mae"]),
                "best_orange_r2": float(best_orange_raw["r2"]),
                "baseline_label": baseline_row["model_label"],
                "baseline_prediction_column": baseline_row["prediction_column"],
                "baseline_rmse": float(baseline_row["rmse"]),
                "baseline_mae": float(baseline_row["mae"]),
                "baseline_r2": float(baseline_row["r2"]),
                "delta_rmse_best_orange_minus_baseline": float(best_orange_raw["rmse"] - baseline_row["rmse"]),
                "delta_mae_best_orange_minus_baseline": float(best_orange_raw["mae"] - baseline_row["mae"]),
                "best_orange_beats_baseline_rmse": bool(best_orange_raw["rmse"] < baseline_row["rmse"]),
                "best_orange_beats_baseline_mae": bool(best_orange_raw["mae"] < baseline_row["mae"]),
            }
        )

    return pd.DataFrame(rows), best_orange_raw


def print_clipping_summary(diagnostics_df):
    clipped_rows = diagnostics_df[diagnostics_df["clipping_applied"]].copy()
    if clipped_rows.empty:
        print(
            "Keine Orange-Modelle mit Predictions < 1 gefunden. Kein Clipping ausgewertet.")
        return

    print("Clipping-Zusammenfassung fuer Modelle mit Predictions < 1:")
    for _, row in clipped_rows.iterrows():
        print(
            "- "
            f"{row['model_label']}: "
            f"RMSE {row['clip_effect_rmse']}, "
            f"MAE {row['clip_effect_mae']}, "
            f"MAPE {row['clip_effect_mape']}, "
            f"R2 {row['clip_effect_r2']}"
        )


def print_interpretation(best_orange_raw, overall_df, target_summary, diagnostics_df):
    always_one_row = overall_df[
        (overall_df["model_key"] == "always_1") & (
            overall_df["evaluation_variant"] == "raw")
    ].iloc[0]

    best_any_variant = overall_df[overall_df["source_type"] == "orange_model"].sort_values(
        by=["rmse", "mae", "mse", "mape", "r2"],
        ascending=[True, True, True, True, False],
    ).iloc[0]

    print("\nAbschlussinterpretation:")
    print(
        "- Bestes Orange-Modell insgesamt (raw): "
        f"{best_orange_raw['model_label']} "
        f"mit RMSE={best_orange_raw['rmse']:.6f}, MAE={best_orange_raw['mae']:.6f}, R2={best_orange_raw['r2']:.6f}."
    )
    print(
        "- Schlaegt es Always-1 nach MAE? "
        f"{'Ja' if best_orange_raw['mae'] < always_one_row['mae'] else 'Nein'} "
        f"(Orange={best_orange_raw['mae']:.6f}, Always-1={always_one_row['mae']:.6f})."
    )
    print(
        "- Schlaegt es Always-1 nach RMSE? "
        f"{'Ja' if best_orange_raw['rmse'] < always_one_row['rmse'] else 'Nein'} "
        f"(Orange={best_orange_raw['rmse']:.6f}, Always-1={always_one_row['rmse']:.6f})."
    )

    if best_any_variant["evaluation_variant"] != "raw":
        print(
            "- Beste Orange-Variante inklusive Clipping: "
            f"{best_any_variant['evaluation_label']} "
            f"mit RMSE={best_any_variant['rmse']:.6f}, MAE={best_any_variant['mae']:.6f}, R2={best_any_variant['r2']:.6f}."
        )

    print(
        "- Warum ist R2 niedrig? Die Zielverteilung ist stark konzentriert auf kleine Mengen, "
        f"vor allem auf quantity = 1 ({target_summary['share_quantity_eq_1'] * 100:.2f}%), "
        "waehrend wenige Mehrfachkaeufe und Ausreisser die Restvarianz dominieren. "
        "Schon kleine Modellfehler im Tail belasten daher R2 stark."
    )
    print(
        "- Warum ist REG schwieriger als CLS? CLS trennt Kauf gegen Nicht-Kauf, "
        "REG muss innerhalb der Kaufereignisse die exakte Menge treffen. "
        "Das ist eine feinere, deutlich tail-lastigere Aufgabe."
    )
    print(
        "- Was bedeutet die Dominanz von quantity = 1? Eine naive Baseline deckt bereits viele typische Faelle ab. "
        "Ein Modell ist nur dann praktisch relevant, wenn es die Mehrfachkaeufe besser erfasst, "
        "ohne den starken quantity=1-Block zu verschlechtern."
    )
    print(
        "- Konsequenz fuer Python-Full-Scale-REG: Always-1 muss als Pflicht-Benchmark bleiben. "
        "Die spaetere Modellierung sollte Overall- und Segmentmetriken gemeinsam berichten, "
        "Tail-Faelle separat pruefen und Positivitaets-Constraints oder Clipping bewusst evaluieren."
    )

    print_clipping_summary(diagnostics_df)


def main():
    print("Orange REG Test-Audit startet. Es wird nur die exportierte Prediction-Datei geladen.")
    print(f"Output-Verzeichnis: {OUTPUT_DIR}")

    df, _ = load_orange_predictions(INPUT_PATH)
    y_true, target_summary = validate_target(df, TARGET_COLUMN)
    resolved_columns = resolve_prediction_columns(df.columns.tolist())

    entries = []
    for model_key, spec in MODEL_SPECS.items():
        column_name = resolved_columns[model_key]
        if column_name is None:
            print(
                f"Prediction-Spalte fuer {model_key} nicht gefunden. Wird uebersprungen.")
            continue

        raw_predictions = coerce_prediction_series(df, column_name)
        entries.append(
            {
                "model_key": model_key,
                "model_label": spec["label"],
                "evaluation_label": spec["label"],
                "source_type": spec["source_type"],
                "prediction_column": column_name,
                "evaluation_variant": "raw",
                "clip_floor": np.nan,
                "predictions": raw_predictions,
            }
        )

        if spec["clip_if_below_one"] and bool((raw_predictions < 1).any()):
            entries.append(
                {
                    "model_key": model_key,
                    "model_label": spec["label"],
                    "evaluation_label": f"{spec['label']} (clipped at 1)",
                    "source_type": spec["source_type"],
                    "prediction_column": column_name,
                    "evaluation_variant": "clipped_at_1",
                    "clip_floor": 1.0,
                    "predictions": raw_predictions.clip(lower=1.0),
                }
            )

    entries.extend(
        [
            {
                "model_key": "always_1",
                "model_label": "Always-1 baseline",
                "evaluation_label": "Always-1 baseline",
                "source_type": "control_baseline",
                "prediction_column": "generated::always_1",
                "evaluation_variant": "raw",
                "clip_floor": np.nan,
                "predictions": pd.Series(1.0, index=y_true.index, dtype=float),
            },
            {
                "model_key": "test_median",
                "model_label": "TEST-Median baseline",
                "evaluation_label": "TEST-Median baseline",
                "source_type": "control_baseline",
                "prediction_column": "generated::test_median",
                "evaluation_variant": "raw",
                "clip_floor": np.nan,
                "predictions": pd.Series(float(y_true.median()), index=y_true.index, dtype=float),
            },
            {
                "model_key": "test_mean",
                "model_label": "TEST-Mean baseline",
                "evaluation_label": "TEST-Mean baseline",
                "source_type": "control_baseline",
                "prediction_column": "generated::test_mean",
                "evaluation_variant": "raw",
                "clip_floor": np.nan,
                "predictions": pd.Series(float(y_true.mean()), index=y_true.index, dtype=float),
            },
        ]
    )

    overall_df = evaluate_entries(y_true, entries)
    segmented_df = evaluate_segments(y_true, entries)
    diagnostics_df = build_prediction_diagnostics(y_true, entries, overall_df)
    baseline_comparison_df, best_orange_raw = build_baseline_comparison(
        overall_df)

    save_requested_scatterplots(y_true, entries)

    round_float_columns(overall_df).to_csv(OVERALL_METRICS_FILE, index=False)
    round_float_columns(segmented_df).to_csv(
        SEGMENTED_METRICS_FILE, index=False)
    round_float_columns(baseline_comparison_df).to_csv(
        BASELINE_COMPARISON_FILE, index=False)
    round_float_columns(diagnostics_df).to_csv(DIAGNOSTICS_FILE, index=False)

    print("CSV-Artefakte gespeichert:")
    print(f"- {OVERALL_METRICS_FILE.name}")
    print(f"- {SEGMENTED_METRICS_FILE.name}")
    print(f"- {BASELINE_COMPARISON_FILE.name}")
    print(f"- {DIAGNOSTICS_FILE.name}")

    print_interpretation(best_orange_raw, overall_df,
                         target_summary, diagnostics_df)


if __name__ == "__main__":
    main()

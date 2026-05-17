"""
Configuration constants for the Dynamic Pricing feature-engineering pipeline.

Single source of truth for paths, split boundaries, feature lists,
and forbidden-feature guards.  Imported as ``import config as cfg``.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

_FE_DIR = Path(__file__).resolve().parent          # Feature Engineering/
_PROJECT_ROOT = _FE_DIR.parent                     # Analytics Project Code/

PROJECT_ROOT = _PROJECT_ROOT

# Default data locations.
SAMPLE_DATA_DIR = _PROJECT_ROOT / "data" / "sample"
RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"

# DATA_DIR is the active data directory. configure_runtime() rewrites it.
DATA_DIR = RAW_DATA_DIR
TRAIN_CSV = DATA_DIR / "train.csv"
ITEMS_CSV = DATA_DIR / "items.csv"
SAMPLE_TRAIN_FILENAME = "train_sample.csv"
SAMPLE_ITEMS_FILENAME = "items_sample.csv"

PHARMFORM_MAPPING_FILE = "pharmform_mapping.csv.xlsx"
PHARMFORM_MAPPING_CANDIDATES = [
    _PROJECT_ROOT / "data" / "mappings" / PHARMFORM_MAPPING_FILE,
    _PROJECT_ROOT / PHARMFORM_MAPPING_FILE,
    _PROJECT_ROOT / "data" / PHARMFORM_MAPPING_FILE,
]

DEFAULT_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts" / "sample_run"

OUTPUT_DIR = _FE_DIR / "outputs"
OUTPUT_DATASETS_DIR = OUTPUT_DIR / "datasets"
OUTPUT_AUDIT_DIR = OUTPUT_DIR / "audit"
OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"
OUTPUT_FEATURE_SELECTION_DIR = OUTPUT_DIR / "feature_selection"
OUTPUT_ORANGE_EXPORTS_DIR = OUTPUT_DIR / "orange_exports"


def configure_runtime(
    data_dir: Path | str,
    output_dir: Path | str,
    train_filename: str = "train.csv",
    items_filename: str = "items.csv",
) -> None:
    """Reroute all data + output paths at runtime.

    Used by ``scripts/run_pipeline.py`` to switch between sample run
    (``data/sample/`` -> ``artifacts/<name>/``) and full run
    (``data/raw/`` -> ``artifacts/<name>/``) without code changes.
    """
    global DATA_DIR, TRAIN_CSV, ITEMS_CSV
    global OUTPUT_DIR, OUTPUT_DATASETS_DIR, OUTPUT_AUDIT_DIR
    global OUTPUT_METADATA_DIR, OUTPUT_FEATURE_SELECTION_DIR
    global OUTPUT_ORANGE_EXPORTS_DIR

    DATA_DIR = Path(data_dir).resolve()
    TRAIN_CSV = DATA_DIR / train_filename
    ITEMS_CSV = DATA_DIR / items_filename

    OUTPUT_DIR = Path(output_dir).resolve()
    OUTPUT_DATASETS_DIR = OUTPUT_DIR / "datasets"
    OUTPUT_AUDIT_DIR = OUTPUT_DIR / "audit"
    OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"
    OUTPUT_FEATURE_SELECTION_DIR = OUTPUT_DIR / "feature_selection"
    OUTPUT_ORANGE_EXPORTS_DIR = OUTPUT_DIR / "orange_exports"

# ── Split boundaries ────────────────────────────────────────────────────────
# Naming convention used in this project: Train -> Test -> Validation.


TRAIN_DAY_START = 26
TRAIN_DAY_END = 70
TEST_DAY_START = 71
TEST_DAY_END = 81
VAL_DAY_START = 82
VAL_DAY_END = 92

# ── Reproducibility ─────────────────────────────────────────────────────────

SEED = 42

# ── Sampling ─────────────────────────────────────────────────────────────────

SAMPLE_FRAC_CLS = 0.30
SAMPLE_FRAC_REG = 0.30

# ── Build modes ──────────────────────────────────────────────────────────────

BUILD_MODES = ["safe_only", "safe_plus_conditional"]
BUILD_MODE_DEFAULT = "safe_only"

# ── Feature lists: SAFE ──────────────────────────────────────────────────────

_SAFE_ALL: list[str] = [
    "day", "day_7", "day_14", "day_30",
    "adFlag", "availability",
    "price", "rrp", "competitorPrice", "competitorPrice_missing",
    "genericProduct", "salesIndex",
    "category_norm", "pharmForm_norm", "campaignIndex_norm",
    "has_campaign",
    "manufacturer_freq",
    "group12", "group34",
    "is_multipack", "pack_n", "pack_size", "pack_total_size",
    "price_diff", "price_discount", "competitorPrice_discount",
    "price_discount_diff",
    "is_lower_price", "is_discount", "is_greater_discount",
    "rrp_per_unit", "price_per_unit", "competitorPrice_per_unit",
    "price_diff_bin", "discount_bin",
]

CLS_BASE_SAFE: list[str] = [
    f for f in _SAFE_ALL if f not in ("price_diff_bin", "discount_bin")
]
CLS_EXPANDED_SAFE: list[str] = list(_SAFE_ALL)

REG_BASE_SAFE: list[str] = [
    f for f in _SAFE_ALL if f not in ("price_diff_bin", "discount_bin")
]
REG_EXPANDED_SAFE: list[str] = list(_SAFE_ALL)

# ── Feature lists: CONDITIONAL ───────────────────────────────────────────────

CLS_CONDITIONAL: list[str] = [
    # cumulative (ab Tag 26)
    "pid_total_events", "click_time", "basket_time", "order_time",
    "num_pid_order",
    # train-basierte Aggregationen
    "group12_order", "group34_order", "week_order",
    # time-aware OOF
    "pid_prob", "availability_likelihood", "day_7_likelihood",
    # pid_segment als Modellfeature
    "pid_segment",
]

REG_CONDITIONAL: list[str] = [
    # cumulative (ab Tag 26, OHNE num_pid_order)
    "pid_total_events", "click_time", "basket_time", "order_time",
    # train-basierte Aggregationen
    "group12_order", "group34_order", "week_order",
    # time-aware OOF (day_7_qty_mean_oof statt day_7_likelihood)
    "pid_prob", "availability_likelihood", "day_7_qty_mean_oof",
    # pid_segment als Modellfeature
    "pid_segment",
]

# ── Final feature sets (Round 4, for Orange export) ─────────────────────────

CLS_FINAL: list[str] = [
    "day", "day_7", "day_14", "day_30",
    "adFlag", "availability",
    "price", "competitorPrice",
    "salesIndex",
    "category_norm", "pharmForm_norm",
    "has_campaign",
    "group12", "group34",
    "is_greater_discount",
    "price_per_unit",
    "price_diff_bin", "discount_bin",
    "pid_total_events", "click_time", "basket_time", "order_time",
    "group12_order", "group34_order",
    "pid_prob", "availability_likelihood", "day_7_likelihood",
    "pid_segment",
]

REG_FINAL: list[str] = [
    "day", "day_7", "day_14", "day_30",
    "price", "competitorPrice",
    "genericProduct", "salesIndex",
    "category_norm", "pharmForm_norm", "campaignIndex_norm",
    "manufacturer_freq",
    "group34",
    "is_multipack",
    "price_diff", "price_discount",
    "price_diff_bin", "discount_bin",
    "pid_total_events", "click_time", "basket_time", "order_time",
    "group12_order", "group34_order",
    "pid_prob",
    "pid_segment",
]

# ── Conservative variant sets (for Orange sensitivity comparison) ────────────

CLS_FINAL_NO_GROUPS: list[str] = [
    f for f in CLS_FINAL if f not in ("group12", "group34")
]

REG_FINAL_NO_GROUP34: list[str] = [
    f for f in REG_FINAL if f != "group34"
]

# ── PharmForm ablation feature sets (CLS only) ──────────────────────────────

CLS_PHARMFORM_V0_BASELINE: list[str] = [
    f for f in CLS_FINAL if f != "pharmForm_norm"
]

CLS_PHARMFORM_V1_CLEAN: list[str] = list(CLS_FINAL)

CLS_PHARMFORM_V2_GROUP: list[str] = CLS_PHARMFORM_V0_BASELINE + [
    "pharmform_group", "pharmform_missing_flag", "pharmform_unmapped_flag",
]

CLS_PHARMFORM_V3_CLEAN_AND_GROUP: list[str] = CLS_FINAL + [
    "pharmform_group", "pharmform_missing_flag", "pharmform_unmapped_flag",
]

# Columns to cast as string before Orange CSV export
CATEGORICAL_AS_STRING: list[str] = [
    "pid_segment", "category_norm", "campaignIndex_norm", "pharmForm_norm",
    "pharmform_group", "pharmForm_clean", "group12",
]

# Export-only prefix for numeric-looking categorical columns.
# Applied in _build_orange_df() so Orange reliably treats them as discrete.
# Does NOT change preprocessing or feature engineering — export layer only.
ORANGE_DISCRETE_PREFIX: dict[str, str] = {
    "category_norm": "C_",
}

# ── Forbidden features ───────────────────────────────────────────────────────

VERBOTEN_CLS: list[str] = [
    "revenue", "lineID",
    "click", "basket",
    "quantity", "quantity_class", "qty_suspicious",
    "order", "q_raw",
    "pid_likelihood",
    "day_7_qty_mean_oof",   # REG-only
]

VERBOTEN_REG: list[str] = [
    "revenue", "lineID",
    "click", "basket",
    "quantity", "quantity_class", "qty_suspicious",
    "order", "q_raw",
    "pid_likelihood",
    "num_pid_order",         # Leakage: zählt Bestellungen ≈ quantity
    "day_7_likelihood",      # CLS-only
]

# ── OOF configuration ───────────────────────────────────────────────────────

# Maximale Unsicherheit (kein Informationsgehalt)
OOF_COLD_START_PROB: float = 0.5
OOF_COLD_START_QTY: float = 1.0    # Konservative Einzelmengenannahme

# ── Feature families (for selection reports) ─────────────────────────────────

FEATURE_FAMILIES: dict[str, list[str]] = {
    "price_absolute": [
        "price", "rrp", "competitorPrice", "competitorPrice_missing",
    ],
    "price_relative": [
        "price_diff", "price_discount", "competitorPrice_discount",
        "price_discount_diff",
        "is_lower_price", "is_discount", "is_greater_discount",
        "price_diff_bin", "discount_bin",
    ],
    "per_unit": [
        "rrp_per_unit", "price_per_unit", "competitorPrice_per_unit",
    ],
    "time": [
        "day", "day_7", "day_14", "day_30",
    ],
    "campaign_ad": [
        "adFlag", "campaignIndex_norm", "has_campaign", "salesIndex",
    ],
    "product_master": [
        "genericProduct", "category_norm", "pharmForm_norm",
        "manufacturer_freq", "group12", "group34", "availability",
    ],
    "pack_structure": [
        "is_multipack", "pack_n", "pack_size", "pack_total_size",
    ],
    "conditional_cumulative": [
        "pid_total_events", "click_time", "basket_time",
        "order_time", "num_pid_order",
    ],
    "conditional_aggregation": [
        "group12_order", "group34_order", "week_order",
    ],
    "conditional_oof": [
        "pid_prob", "availability_likelihood",
        "day_7_likelihood", "day_7_qty_mean_oof",
    ],
    "conditional_segment": [
        "pid_segment",
    ],
}

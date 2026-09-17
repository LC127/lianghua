from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)


# ------------------------------------------------------------
# Previous research day outputs
#
# Research Day 4 was stored under M1_day3.
# ------------------------------------------------------------

PREVIOUS_DAY_ROOT = (
    OUTPUT_ROOT
    / "M1_day4"
)


# ------------------------------------------------------------
# Previous Day Step 3:
# Structural diagnostics / node-level metrics
# ------------------------------------------------------------

PREVIOUS_STEP3_DIR = (
    PREVIOUS_DAY_ROOT
    / "03_step3_structural_diagnostics"
)

NODE_METRIC_ROOT = (
    PREVIOUS_STEP3_DIR
    / "node_metrics"
)


RAW_NETWORK_ID = (
    "B0_RAW_POSITIVE"
)

RESIDUAL_NETWORK_ID = (
    "M1_RESIDUAL_POSITIVE"
)


# ------------------------------------------------------------
# Day 5 root
# ------------------------------------------------------------

DAY5_ROOT = (
    OUTPUT_ROOT
    / "M1_day5"
)


# ------------------------------------------------------------
# Day 5 Step 1:
# Frozen factor design
# ------------------------------------------------------------

STEP1_DIR = (
    DAY5_ROOT
    / "01_step1_factor_design"
)

FACTOR_CONFIG_PATH = (
    STEP1_DIR
    / "factor_design_config.json"
)

ANALYSIS_DATES_PATH = (
    STEP1_DIR
    / "factor_analysis_dates.csv"
)

FACTOR_DICTIONARY_PATH = (
    STEP1_DIR
    / "network_factor_dictionary.csv"
)


# ------------------------------------------------------------
# Day 5 Step 2 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY5_ROOT
    / "02_step2_network_feature_panel"
)

PARTITION_DIR = (
    OUTPUT_DIR
    / "panel_parts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTITION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_PANEL_PATH = (
    OUTPUT_DIR
    / "base_node_feature_panel.parquet"
)

JOB_QA_PATH = (
    OUTPUT_DIR
    / "base_node_feature_panel_job_qa.csv"
)

COVERAGE_PATH = (
    OUTPUT_DIR
    / "base_node_feature_coverage_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "base_node_feature_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step2_base_node_feature_panel_metadata.json"
)


# ============================================================
# 1. Settings
# ============================================================

WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# Percentile convention
# ------------------------------------------------------------

PERCENTILE_TIE_METHOD = (
    "average"
)


# ------------------------------------------------------------
# Strict tolerance for DEGREE percentile reconstruction.
#
# Degree is integer-valued, so this should normally be
# numerically identical.
# ------------------------------------------------------------

DEGREE_PERCENTILE_TOLERANCE = 1e-10


# ------------------------------------------------------------
# Strength validity tolerance
#
# Positive-edge networks should have nonnegative strength.
# ------------------------------------------------------------

STRENGTH_TOLERANCE = 1e-10


# ------------------------------------------------------------
# Save each Date × W partition for debugging / later reuse.
# ------------------------------------------------------------

SAVE_JOB_PARTITIONS = True


# ------------------------------------------------------------
# Formal QA failures immediately stop execution.
# ------------------------------------------------------------

STRICT_QA = True


# ============================================================
# 2. Required Frozen Factor Definitions
# ============================================================

REQUIRED_FROZEN_FACTORS = {

    "residual_degree",

    "residual_strength",

    "residual_degree_percentile",

    "residual_strength_percentile",

    "delta_degree_percentile",

    "delta_strength_percentile",
}


# ============================================================
# 3. Output Panel Columns
# ============================================================

PANEL_COLUMNS = [

    # --------------------------------------------------------
    # Panel identifiers
    # --------------------------------------------------------

    "analysis_date",
    "window",
    "master_index",

    # --------------------------------------------------------
    # Stock identity
    # --------------------------------------------------------

    "security_id",
    "stock_code",
    "stock_name",
    "exchange",

    # --------------------------------------------------------
    # PIT economic labels / market exposure
    # --------------------------------------------------------

    "industry_id1",
    "industry_id2",

    "market_beta",
    "market_r2",

    # --------------------------------------------------------
    # Raw benchmark network
    # --------------------------------------------------------

    "raw_degree",
    "raw_strength",

    "raw_degree_percentile",
    "raw_strength_percentile",

    # --------------------------------------------------------
    # Primary residual network
    # --------------------------------------------------------

    "residual_degree",
    "residual_strength",

    "residual_degree_percentile",
    "residual_strength_percentile",

    # --------------------------------------------------------
    # Raw -> Residual re-ranking
    # --------------------------------------------------------

    "delta_degree_percentile",
    "delta_strength_percentile",

    # --------------------------------------------------------
    # Auxiliary raw-level changes
    # --------------------------------------------------------

    "residual_minus_raw_degree",
    "residual_minus_raw_strength",

    # --------------------------------------------------------
    # Node status
    # --------------------------------------------------------

    "raw_is_isolated",
    "residual_is_isolated",

    "raw_is_giant_component",
    "residual_is_giant_component",
]


# ============================================================
# 4. Utility Functions
# ============================================================

def load_json(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json(
    obj,
    path: Path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def percentile_rank(
    values,
):

    """
    Cross-sectional percentile rank.

    The definition is intentionally kept identical to
    the Step 3 convention for degree reconstruction.
    """

    return (
        pd.Series(
            values
        )
        .rank(
            method=PERCENTILE_TIE_METHOD,
            pct=True,
        )
        .to_numpy(
            dtype=np.float64
        )
    )


# ============================================================
# 5. Load Frozen Factor Design
# ============================================================

def load_frozen_design():

    config = load_json(
        FACTOR_CONFIG_PATH
    )

    if (
        config.get(
            "design_status"
        )
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Factor design is not marked FROZEN."
        )

    design_hash = (
        config.get(
            "design_hash"
        )
    )

    if not design_hash:

        raise RuntimeError(
            "Frozen design has no design_hash."
        )

    network_design = (
        config[
            "network_design"
        ]
    )

    if (
        network_design[
            "primary_network"
        ]
        !=
        RESIDUAL_NETWORK_ID
    ):

        raise RuntimeError(
            "Primary network in frozen design "
            "does not match expected Residual network."
        )

    if (
        network_design[
            "benchmark_network"
        ]
        !=
        RAW_NETWORK_ID
    ):

        raise RuntimeError(
            "Benchmark network in frozen design "
            "does not match expected Raw network."
        )

    frozen_windows = sorted(
        network_design[
            "windows"
        ]
    )

    if (
        frozen_windows
        !=
        sorted(
            WINDOWS
        )
    ):

        raise RuntimeError(
            "Frozen windows differ from "
            f"{WINDOWS}."
        )

    return (
        config,
        design_hash,
    )


# ============================================================
# 6. Validate Frozen Factor Dictionary
# ============================================================

def validate_factor_dictionary():

    if not FACTOR_DICTIONARY_PATH.exists():

        raise FileNotFoundError(
            "Missing factor dictionary:\n"
            f"{FACTOR_DICTIONARY_PATH}"
        )

    df = pd.read_csv(
        FACTOR_DICTIONARY_PATH
    )

    if (
        "factor_name"
        not in df.columns
    ):

        raise ValueError(
            "network_factor_dictionary.csv "
            "has no factor_name column."
        )

    existing = set(
        df[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )

    missing = (
        REQUIRED_FROZEN_FACTORS
        -
        existing
    )

    if missing:

        raise RuntimeError(
            "Frozen factor dictionary misses:\n"
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# 7. Load Frozen Analysis Dates
# ============================================================

def load_analysis_dates():

    if not ANALYSIS_DATES_PATH.exists():

        raise FileNotFoundError(
            "Missing analysis dates:\n"
            f"{ANALYSIS_DATES_PATH}"
        )

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

    required = [

        "analysis_date",
        "window",

        "common_node_count",
        "edge_budget_k",

        "previous_analysis_date",
        "next_analysis_date",

        "dynamic_factor_eligible",
        "forward_label_eligible",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "factor_analysis_dates.csv "
            f"missing columns: {missing}"
        )

    df[
        "analysis_date"
    ] = pd.to_datetime(
        df[
            "analysis_date"
        ],
        errors="raise",
    )

    df[
        "window"
    ] = pd.to_numeric(
        df[
            "window"
        ],
        errors="raise",
    ).astype(int)

    return (
        df.sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 8. Resolve Node Metric File
# ============================================================

def node_metric_path(
    network_id,
    window,
    date,
):

    date_string = (
        pd.Timestamp(
            date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    return (
        NODE_METRIC_ROOT
        /
        network_id
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )


# ============================================================
# 9. Load One Node Metric File
# ============================================================

def load_node_metrics(
    network_id,
    window,
    date,
):

    path = node_metric_path(

        network_id=
            network_id,

        window=
            window,

        date=
            date,
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing node metric:\n{path}"
        )

    df = pd.read_parquet(
        path
    )

    required = [

        "master_index",

        "security_id",
        "stock_code",
        "stock_name",
        "exchange",

        "market_beta",
        "market_r2",

        "industry_id1",
        "industry_id2",

        "degree",
        "strength",

        "degree_percentile",
        "strength_percentile",

        "is_giant_component",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{network_id} node metrics "
            f"missing columns:\n{missing}"
        )

    if (
        df[
            "master_index"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate master_index found:\n"
            f"{path}"
        )

    return (
        df.sort_values(
            "master_index"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 10. Validate Same Node Universe
# ============================================================

def validate_same_node_universe(
    raw,
    residual,
):

    raw_ids = (
        raw[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    residual_ids = (
        residual[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        raw_ids,
        residual_ids,
    ):

        raw_set = set(
            raw_ids.tolist()
        )

        residual_set = set(
            residual_ids.tolist()
        )

        only_raw = (
            raw_set
            -
            residual_set
        )

        only_residual = (
            residual_set
            -
            raw_set
        )

        raise RuntimeError(
            "Raw and Residual node universes differ.\n"
            f"Only Raw: {len(only_raw)}\n"
            f"Only Residual: {len(only_residual)}"
        )


# ============================================================
# 11. Validate Node Metadata Consistency
# ============================================================

def validate_metadata_consistency(
    raw,
    residual,
):

    columns = [

        "security_id",
        "stock_code",
        "stock_name",
        "exchange",

        "industry_id1",
        "industry_id2",
    ]

    problems = {}

    for col in columns:

        a = (
            raw[
                col
            ]
            .astype(
                "string"
            )
        )

        b = (
            residual[
                col
            ]
            .astype(
                "string"
            )
        )

        mismatch = ~(
            a.fillna(
                "<NA>"
            )
            ==
            b.fillna(
                "<NA>"
            )
        )

        n_bad = int(
            mismatch.sum()
        )

        if n_bad > 0:

            problems[
                col
            ] = n_bad

    if problems:

        raise RuntimeError(
            "Raw / Residual node metadata "
            f"are inconsistent: {problems}"
        )


# ============================================================
# 12. Build One Date × Window Panel
# ============================================================

def build_one_panel(
    raw,
    residual,
    analysis_date,
    window,
):

    # --------------------------------------------------------
    # Raw / Residual must correspond to exactly the same
    # formal Step-2 node universe.
    # --------------------------------------------------------

    validate_same_node_universe(
        raw,
        residual,
    )

    validate_metadata_consistency(
        raw,
        residual,
    )

    # --------------------------------------------------------
    # Residual table is used as canonical node metadata.
    # --------------------------------------------------------

    result = residual[
        [
            "master_index",

            "security_id",
            "stock_code",
            "stock_name",
            "exchange",

            "industry_id1",
            "industry_id2",

            "market_beta",
            "market_r2",
        ]
    ].copy()

    result.insert(
        0,
        "window",
        int(
            window
        ),
    )

    result.insert(
        0,
        "analysis_date",
        pd.Timestamp(
            analysis_date
        ),
    )

    # ========================================================
    # Raw network degree / strength
    # ========================================================

    result[
        "raw_degree"
    ] = (
        raw[
            "degree"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    result[
        "raw_strength"
    ] = (
        raw[
            "strength"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # ========================================================
    # Residual network degree / strength
    # ========================================================

    result[
        "residual_degree"
    ] = (
        residual[
            "degree"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    result[
        "residual_strength"
    ] = (
        residual[
            "strength"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # ========================================================
    # IMPORTANT FIX
    #
    # Use the percentile rankings that were formally computed
    # and saved in Day-4 Step 3.
    #
    # Reason:
    #
    # Step 3 computed strength percentile using the original
    # float64 strength values. The stored strength column may
    # subsequently have been compressed to float32.
    #
    # Re-ranking the stored float32 values can alter near-ties.
    #
    # Therefore Step-3 percentile fields are authoritative.
    # ========================================================

    result[
        "raw_degree_percentile"
    ] = (
        raw[
            "degree_percentile"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    result[
        "raw_strength_percentile"
    ] = (
        raw[
            "strength_percentile"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    result[
        "residual_degree_percentile"
    ] = (
        residual[
            "degree_percentile"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    result[
        "residual_strength_percentile"
    ] = (
        residual[
            "strength_percentile"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # ========================================================
    # Core Raw -> Residual re-ranking factors
    # ========================================================

    result[
        "delta_degree_percentile"
    ] = (

        result[
            "residual_degree_percentile"
        ]

        -

        result[
            "raw_degree_percentile"
        ]
    )

    result[
        "delta_strength_percentile"
    ] = (

        result[
            "residual_strength_percentile"
        ]

        -

        result[
            "raw_strength_percentile"
        ]
    )

    # ========================================================
    # Auxiliary raw-level changes
    # ========================================================

    result[
        "residual_minus_raw_degree"
    ] = (

        result[
            "residual_degree"
        ]

        -

        result[
            "raw_degree"
        ]
    )

    result[
        "residual_minus_raw_strength"
    ] = (

        result[
            "residual_strength"
        ]

        -

        result[
            "raw_strength"
        ]
    )

    # ========================================================
    # Isolation status
    # ========================================================

    result[
        "raw_is_isolated"
    ] = (
        result[
            "raw_degree"
        ]
        ==
        0
    )

    result[
        "residual_is_isolated"
    ] = (
        result[
            "residual_degree"
        ]
        ==
        0
    )

    # ========================================================
    # Giant component status
    # ========================================================

    result[
        "raw_is_giant_component"
    ] = (
        raw[
            "is_giant_component"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    result[
        "residual_is_giant_component"
    ] = (
        residual[
            "is_giant_component"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    # ========================================================
    # Stable output column order
    # ========================================================

    result = result[
        PANEL_COLUMNS
    ]

    return result


# ============================================================
# 13. Formal QA for One Panel
# ============================================================

def qa_one_panel(
    panel,
    raw_source,
    residual_source,
    expected_node_count,
    expected_edge_count,
    analysis_date,
    window,
):

    errors = []

    n = len(
        panel
    )

    # ========================================================
    # 13.1 Node count
    # ========================================================

    node_count_ok = (
        n
        ==
        int(
            expected_node_count
        )
    )

    if not node_count_ok:

        errors.append(
            "node_count_mismatch"
        )

    # ========================================================
    # 13.2 Unique panel key
    # ========================================================

    key_unique = (
        not panel[
            [
                "analysis_date",
                "window",
                "master_index",
            ]
        ]
        .duplicated()
        .any()
    )

    if not key_unique:

        errors.append(
            "duplicate_panel_key"
        )

    # ========================================================
    # 13.3 Handshake lemma
    #
    # sum_i d_i = 2 |E|
    # ========================================================

    expected_degree_sum = (
        2
        *
        int(
            expected_edge_count
        )
    )

    raw_degree_sum = int(
        panel[
            "raw_degree"
        ]
        .sum()
    )

    residual_degree_sum = int(
        panel[
            "residual_degree"
        ]
        .sum()
    )

    raw_degree_sum_ok = (
        raw_degree_sum
        ==
        expected_degree_sum
    )

    residual_degree_sum_ok = (
        residual_degree_sum
        ==
        expected_degree_sum
    )

    if not raw_degree_sum_ok:

        errors.append(
            "raw_degree_handshake_failure"
        )

    if not residual_degree_sum_ok:

        errors.append(
            "residual_degree_handshake_failure"
        )

    # ========================================================
    # 13.4 Degree validity
    # ========================================================

    degree_nonnegative = bool(

        (
            panel[
                [
                    "raw_degree",
                    "residual_degree",
                ]
            ]
            >=
            0
        )
        .all()
        .all()
    )

    if not degree_nonnegative:

        errors.append(
            "negative_degree"
        )

    # ========================================================
    # 13.5 Positive-edge strength validity
    # ========================================================

    strength_nonnegative = bool(

        (
            panel[
                [
                    "raw_strength",
                    "residual_strength",
                ]
            ]
            >=
            -
            STRENGTH_TOLERANCE
        )
        .all()
        .all()
    )

    if not strength_nonnegative:

        errors.append(
            "negative_strength"
        )

    # ========================================================
    # 13.6 Saved percentile validity
    #
    # All percentile values should lie in (0, 1].
    # ========================================================

    percentile_columns = [

        "raw_degree_percentile",
        "raw_strength_percentile",

        "residual_degree_percentile",
        "residual_strength_percentile",
    ]

    percentile_valid = True

    for col in percentile_columns:

        x = pd.to_numeric(
            panel[
                col
            ],
            errors="coerce",
        )

        ok = bool(

            x.notna().all()

            and

            (
                x
                >
                0
            ).all()

            and

            (
                x
                <=
                1.0
                +
                DEGREE_PERCENTILE_TOLERANCE
            ).all()
        )

        if not ok:

            percentile_valid = False

    if not percentile_valid:

        errors.append(
            "invalid_percentiles"
        )

    # ========================================================
    # 13.7 STRICT Degree-percentile reconstruction
    #
    # Degree is integer-valued and can be reconstructed
    # exactly from the stored node metric.
    # ========================================================

    recomputed_raw_degree_pct = (
        percentile_rank(
            panel[
                "raw_degree"
            ]
        )
    )

    recomputed_residual_degree_pct = (
        percentile_rank(
            panel[
                "residual_degree"
            ]
        )
    )

    # 按上游百分位字段的存储精度比较，避免 float32 舍入误报。
    recomputed_raw_degree_pct = (
        recomputed_raw_degree_pct
        .astype(raw_source["degree_percentile"].to_numpy().dtype)
        .astype(np.float64)
        )

    recomputed_residual_degree_pct = (
        recomputed_residual_degree_pct
        .astype(residual_source["degree_percentile"].to_numpy().dtype)
        .astype(np.float64)
        )

    raw_degree_pct_diff = float(
        np.nanmax(
            np.abs(

                recomputed_raw_degree_pct

                -

                panel[
                    "raw_degree_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    residual_degree_pct_diff = float(
        np.nanmax(
            np.abs(

                recomputed_residual_degree_pct

                -

                panel[
                    "residual_degree_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    degree_percentile_match = bool(

        max(
            raw_degree_pct_diff,
            residual_degree_pct_diff,
        )

        <=
        DEGREE_PERCENTILE_TOLERANCE
    )

    if not degree_percentile_match:

        errors.append(
            "degree_percentile_mismatch"
        )

    # ========================================================
    # 13.8 Strength percentile diagnostic only
    #
    # IMPORTANT:
    #
    # Do not use this as a FAIL criterion.
    #
    # Stored strength may be float32 while Step 3 percentile
    # was computed from original float64 strength.
    # ========================================================

    recomputed_raw_strength_pct = (
        percentile_rank(
            panel[
                "raw_strength"
            ]
        )
    )

    recomputed_residual_strength_pct = (
        percentile_rank(
            panel[
                "residual_strength"
            ]
        )
    )

    raw_strength_float32_rank_diff = float(
        np.nanmax(
            np.abs(

                recomputed_raw_strength_pct

                -

                panel[
                    "raw_strength_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    residual_strength_float32_rank_diff = float(
        np.nanmax(
            np.abs(

                recomputed_residual_strength_pct

                -

                panel[
                    "residual_strength_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    # ========================================================
    # 13.9 Explicit source-copy validation
    #
    # Since percentiles now come directly from Step 3,
    # verify that the panel did not alter them.
    # ========================================================

    raw_degree_source_copy_diff = float(
        np.nanmax(
            np.abs(

                panel[
                    "raw_degree_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )

                -

                raw_source[
                    "degree_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    raw_strength_source_copy_diff = float(
        np.nanmax(
            np.abs(

                panel[
                    "raw_strength_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )

                -

                raw_source[
                    "strength_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    residual_degree_source_copy_diff = float(
        np.nanmax(
            np.abs(

                panel[
                    "residual_degree_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )

                -

                residual_source[
                    "degree_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    residual_strength_source_copy_diff = float(
        np.nanmax(
            np.abs(

                panel[
                    "residual_strength_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )

                -

                residual_source[
                    "strength_percentile"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )
        )
    )

    percentile_source_copy_ok = bool(

        max(

            raw_degree_source_copy_diff,

            raw_strength_source_copy_diff,

            residual_degree_source_copy_diff,

            residual_strength_source_copy_diff,

        )
        <=
        DEGREE_PERCENTILE_TOLERANCE
    )

    if not percentile_source_copy_ok:

        errors.append(
            "percentile_source_copy_failure"
        )

    # ========================================================
    # 13.10 Re-ranking descriptive statistics
    # ========================================================

    mean_delta_degree_percentile = float(
        panel[
            "delta_degree_percentile"
        ]
        .mean()
    )

    mean_delta_strength_percentile = float(
        panel[
            "delta_strength_percentile"
        ]
        .mean()
    )

    median_abs_delta_degree_percentile = float(
        panel[
            "delta_degree_percentile"
        ]
        .abs()
        .median()
    )

    median_abs_delta_strength_percentile = float(
        panel[
            "delta_strength_percentile"
        ]
        .abs()
        .median()
    )

    # ========================================================
    # 13.11 Final QA state
    # ========================================================

    all_pass = (
        len(
            errors
        )
        ==
        0
    )

    return {

        "analysis_date":
            pd.Timestamp(
                analysis_date
            ),

        "window":
            int(
                window
            ),

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        "expected_node_count":
            int(
                expected_node_count
            ),

        "actual_node_count":
            int(
                n
            ),

        "expected_edge_count":
            int(
                expected_edge_count
            ),

        "expected_degree_sum":
            int(
                expected_degree_sum
            ),

        "raw_degree_sum":
            raw_degree_sum,

        "residual_degree_sum":
            residual_degree_sum,

        # ----------------------------------------------------
        # Formal QA
        # ----------------------------------------------------

        "node_count_ok":
            node_count_ok,

        "key_unique":
            key_unique,

        "raw_degree_sum_ok":
            raw_degree_sum_ok,

        "residual_degree_sum_ok":
            residual_degree_sum_ok,

        "degree_nonnegative":
            degree_nonnegative,

        "strength_nonnegative":
            strength_nonnegative,

        "percentile_valid":
            percentile_valid,

        "degree_percentile_match":
            degree_percentile_match,

        "percentile_source_copy_ok":
            percentile_source_copy_ok,

        # ----------------------------------------------------
        # Degree percentile strict diagnostics
        # ----------------------------------------------------

        "max_raw_degree_percentile_diff":
            raw_degree_pct_diff,

        "max_residual_degree_percentile_diff":
            residual_degree_pct_diff,

        # ----------------------------------------------------
        # Strength percentile FLOAT32 diagnostic
        #
        # Not a failure condition.
        # ----------------------------------------------------

        "raw_strength_float32_rank_diff":
            raw_strength_float32_rank_diff,

        "residual_strength_float32_rank_diff":
            residual_strength_float32_rank_diff,

        # ----------------------------------------------------
        # Source-copy diagnostics
        # ----------------------------------------------------

        "raw_degree_source_copy_diff":
            raw_degree_source_copy_diff,

        "raw_strength_source_copy_diff":
            raw_strength_source_copy_diff,

        "residual_degree_source_copy_diff":
            residual_degree_source_copy_diff,

        "residual_strength_source_copy_diff":
            residual_strength_source_copy_diff,

        # ----------------------------------------------------
        # Descriptive network state
        # ----------------------------------------------------

        "raw_isolated_share":
            float(
                panel[
                    "raw_is_isolated"
                ]
                .mean()
            ),

        "residual_isolated_share":
            float(
                panel[
                    "residual_is_isolated"
                ]
                .mean()
            ),

        "mean_delta_degree_percentile":
            mean_delta_degree_percentile,

        "mean_delta_strength_percentile":
            mean_delta_strength_percentile,

        "median_abs_delta_degree_percentile":
            median_abs_delta_degree_percentile,

        "median_abs_delta_strength_percentile":
            median_abs_delta_strength_percentile,

        # ----------------------------------------------------
        # Final
        # ----------------------------------------------------

        "qa_pass":
            all_pass,

        "qa_errors":
            "|".join(
                errors
            ),
    }


# ============================================================
# 14. Save One Date × Window Partition
# ============================================================

def save_job_partition(
    panel,
    window,
    date,
):

    date_string = (
        pd.Timestamp(
            date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    path = (
        PARTITION_DIR
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = Path(
        str(
            path
        )
        +
        ".tmp"
    )

    if temp_path.exists():

        temp_path.unlink()

    panel.to_parquet(
        temp_path,
        index=False,
        compression="zstd",
    )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# 15. Build Coverage Summary
# ============================================================

def build_coverage_summary(
    qa,
):

    rows = []

    for window, group in (
        qa.groupby(
            "window"
        )
    ):

        rows.append(
            {

                "window":
                    int(
                        window
                    ),

                "analysis_date_count":
                    int(
                        len(
                            group
                        )
                    ),

                "first_analysis_date":
                    group[
                        "analysis_date"
                    ]
                    .min(),

                "last_analysis_date":
                    group[
                        "analysis_date"
                    ]
                    .max(),

                "mean_node_count":
                    float(
                        group[
                            "actual_node_count"
                        ]
                        .mean()
                    ),

                "min_node_count":
                    int(
                        group[
                            "actual_node_count"
                        ]
                        .min()
                    ),

                "max_node_count":
                    int(
                        group[
                            "actual_node_count"
                        ]
                        .max()
                    ),

                "qa_pass_share":
                    float(
                        group[
                            "qa_pass"
                        ]
                        .mean()
                    ),

                "mean_raw_isolated_share":
                    float(
                        group[
                            "raw_isolated_share"
                        ]
                        .mean()
                    ),

                "mean_residual_isolated_share":
                    float(
                        group[
                            "residual_isolated_share"
                        ]
                        .mean()
                    ),

                "mean_median_abs_delta_degree_percentile":
                    float(
                        group[
                            "median_abs_delta_degree_percentile"
                        ]
                        .mean()
                    ),

                "mean_median_abs_delta_strength_percentile":
                    float(
                        group[
                            "median_abs_delta_strength_percentile"
                        ]
                        .mean()
                    ),

                # ------------------------------------------------
                # Useful diagnostic:
                # how much float32 strength changes percentile.
                # ------------------------------------------------

                "max_raw_strength_float32_rank_diff":
                    float(
                        group[
                            "raw_strength_float32_rank_diff"
                        ]
                        .max()
                    ),

                "max_residual_strength_float32_rank_diff":
                    float(
                        group[
                            "residual_strength_float32_rank_diff"
                        ]
                        .max()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. Descriptive Factor Summary
#
# This is only panel QA.
#
# NO future returns / Alpha information is used here.
# ============================================================

def summarize_panel_partition(
    panel,
):

    factor_columns = [

        "raw_degree",
        "raw_strength",

        "raw_degree_percentile",
        "raw_strength_percentile",

        "residual_degree",
        "residual_strength",

        "residual_degree_percentile",
        "residual_strength_percentile",

        "delta_degree_percentile",
        "delta_strength_percentile",

        "residual_minus_raw_degree",
        "residual_minus_raw_strength",
    ]

    rows = []

    analysis_date = (
        panel[
            "analysis_date"
        ]
        .iloc[
            0
        ]
    )

    window = int(
        panel[
            "window"
        ]
        .iloc[
            0
        ]
    )

    for factor in factor_columns:

        x = pd.to_numeric(
            panel[
                factor
            ],
            errors="coerce",
        )

        valid = (
            x.dropna()
        )

        if len(
            valid
        ) > 0:

            quantiles = (
                valid.quantile(
                    [
                        0.01,
                        0.05,
                        0.50,
                        0.95,
                        0.99,
                    ]
                )
            )

            mean_value = float(
                valid.mean()
            )

            std_value = float(
                valid.std(
                    ddof=0
                )
            )

            p01 = float(
                quantiles.loc[
                    0.01
                ]
            )

            p05 = float(
                quantiles.loc[
                    0.05
                ]
            )

            median = float(
                quantiles.loc[
                    0.50
                ]
            )

            p95 = float(
                quantiles.loc[
                    0.95
                ]
            )

            p99 = float(
                quantiles.loc[
                    0.99
                ]
            )

            zero_share = float(
                (
                    valid
                    ==
                    0
                )
                .mean()
            )

        else:

            mean_value = np.nan
            std_value = np.nan

            p01 = np.nan
            p05 = np.nan
            median = np.nan
            p95 = np.nan
            p99 = np.nan

            zero_share = np.nan

        rows.append(
            {

                "analysis_date":
                    analysis_date,

                "window":
                    window,

                "factor_name":
                    factor,

                "n":
                    int(
                        len(
                            x
                        )
                    ),

                "nonmissing_count":
                    int(
                        len(
                            valid
                        )
                    ),

                "missing_share":
                    float(
                        x.isna()
                        .mean()
                    ),

                "zero_share":
                    zero_share,

                "mean":
                    mean_value,

                "std":
                    std_value,

                "p01":
                    p01,

                "p05":
                    p05,

                "median":
                    median,

                "p95":
                    p95,

                "p99":
                    p99,
            }
        )

    return rows


# ============================================================
# 17. Save Final Metadata
# ============================================================

def save_metadata(
    design_hash,
    panel_row_count,
    qa,
    factor_summary,
):

    payload = {

        "research_day":
            5,

        "step":
            "Step2_Build_Base_Node_Feature_Panel",

        "design_hash":
            design_hash,

        "design_source":
            str(
                FACTOR_CONFIG_PATH
            ),

        "output_panel":
            str(
                FINAL_PANEL_PATH
            ),

        "panel_key": [

            "analysis_date",

            "window",

            "master_index",
        ],

        "primary_network":
            RESIDUAL_NETWORK_ID,

        "benchmark_network":
            RAW_NETWORK_ID,

        "windows":
            WINDOWS,

        "row_count":
            int(
                panel_row_count
            ),

        "percentile_definition": {

            "degree_percentile_source":
                (
                    "Formal Step-3 degree percentile; "
                    "strictly reconstructed from integer "
                    "degree as QA."
                ),

            "strength_percentile_source":
                (
                    "Formal Step-3 strength percentile."
                ),

            "strength_percentile_reason":
                (
                    "Step 3 may have computed strength "
                    "percentiles from float64 strength before "
                    "storing strength itself as float32. "
                    "Therefore re-ranking stored strength is "
                    "diagnostic only."
                ),

            "percentile_tie_method":
                PERCENTILE_TIE_METHOD,

            "percentile_scope":
                (
                    "Cross-sectional within each "
                    "analysis_date x window."
                ),
        },

        "core_factor_definitions": {

            "delta_degree_percentile":
                (
                    "Residual degree percentile "
                    "- Raw degree percentile"
                ),

            "delta_strength_percentile":
                (
                    "Residual strength percentile "
                    "- Raw strength percentile"
                ),
        },

        "important_interpretation": [

            (
                "Raw and Residual degree levels are directly "
                "comparable because the two networks use the "
                "same node support and the same edge budget."
            ),

            (
                "Raw and Residual strength absolute levels "
                "are not interpreted as perfectly comparable "
                "economic scales because residualization "
                "changes the correlation distribution."
            ),

            (
                "Therefore percentile-based Raw-Residual "
                "re-ranking is the main differential-factor "
                "definition."
            ),

            (
                "No future return, future volatility, or "
                "other forward outcome is used in Step 2."
            ),

            (
                "Industry decomposition, dynamic network "
                "features, and community features are "
                "intentionally deferred to later steps."
            ),
        ],

        "formal_qa": {

            "job_count":
                int(
                    len(
                        qa
                    )
                ),

            "all_jobs_pass":
                bool(
                    qa[
                        "qa_pass"
                    ]
                    .all()
                ),

            "failed_job_count":
                int(
                    (
                        ~qa[
                            "qa_pass"
                        ]
                    )
                    .sum()
                ),

            "strength_float32_rank_difference_is_failure":
                False,
        },

        "factor_summary_row_count":
            int(
                len(
                    factor_summary
                )
            ),
    }

    save_json(
        payload,
        METADATA_PATH,
    )


# ============================================================
# 18. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 5 - Step 2"
    )

    print(
        "Build Base Node Feature Panel"
    )

    print("=" * 80)

    # ========================================================
    # 18.1 Load frozen research design
    # ========================================================

    print()
    print(
        "[1] Load frozen factor design"
    )

    (
        config,
        design_hash,
    ) = load_frozen_design()

    validate_factor_dictionary()

    analysis_dates = (
        load_analysis_dates()
    )

    print(
        "Design hash:"
    )

    print(
        design_hash
    )

    print(
        f"Date-window jobs: "
        f"{len(analysis_dates):,}"
    )

    # ========================================================
    # 18.2 Prepare final streaming parquet
    # ========================================================

    temp_final_path = Path(
        str(
            FINAL_PANEL_PATH
        )
        +
        ".tmp"
    )

    if temp_final_path.exists():

        temp_final_path.unlink()

    writer = None

    qa_rows = []

    summary_rows = []

    total_rows = 0

    # ========================================================
    # 18.3 Date × Window loop
    # ========================================================

    try:

        total_jobs = len(
            analysis_dates
        )

        for job_no, row in enumerate(

            analysis_dates.itertuples(
                index=False
            ),

            start=1,
        ):

            date = pd.Timestamp(
                row.analysis_date
            )

            window = int(
                row.window
            )

            expected_nodes = int(
                row.common_node_count
            )

            expected_edges = int(
                row.edge_budget_k
            )

            print()
            print(
                f"[{job_no}/{total_jobs}] "
                f"W={window} | "
                f"{date.date()} | "
                f"N={expected_nodes:,} | "
                f"E={expected_edges:,}"
            )

            # =================================================
            # Load Raw node metrics
            # =================================================

            raw = (
                load_node_metrics(

                    network_id=
                        RAW_NETWORK_ID,

                    window=
                        window,

                    date=
                        date,
                )
            )

            # =================================================
            # Load Residual node metrics
            # =================================================

            residual = (
                load_node_metrics(

                    network_id=
                        RESIDUAL_NETWORK_ID,

                    window=
                        window,

                    date=
                        date,
                )
            )

            # =================================================
            # Build base node feature panel
            # =================================================

            panel = (
                build_one_panel(

                    raw=
                        raw,

                    residual=
                        residual,

                    analysis_date=
                        date,

                    window=
                        window,
                )
            )

            # =================================================
            # Formal QA
            # =================================================

            qa = (
                qa_one_panel(

                    panel=
                        panel,

                    raw_source=
                        raw,

                    residual_source=
                        residual,

                    expected_node_count=
                        expected_nodes,

                    expected_edge_count=
                        expected_edges,

                    analysis_date=
                        date,

                    window=
                        window,
                )
            )

            qa_rows.append(
                qa
            )

            # -------------------------------------------------
            # Print strength precision diagnostic only when
            # there is a nonzero discrepancy.
            # -------------------------------------------------

            if (
                qa[
                    "raw_strength_float32_rank_diff"
                ]
                >
                0
                or
                qa[
                    "residual_strength_float32_rank_diff"
                ]
                >
                0
            ):

                print(
                    "  Strength percentile diagnostic:"
                )

                print(
                    "    Raw float32 re-rank max diff      = "
                    f"{qa['raw_strength_float32_rank_diff']:.12g}"
                )

                print(
                    "    Residual float32 re-rank max diff = "
                    f"{qa['residual_strength_float32_rank_diff']:.12g}"
                )

            if (
                STRICT_QA
                and
                not qa[
                    "qa_pass"
                ]
            ):

                raise RuntimeError(
                    "QA failed for "
                    f"W={window}, "
                    f"{date.date()}:\n"
                    f"{qa['qa_errors']}"
                )

            # =================================================
            # Descriptive panel QA
            # =================================================

            summary_rows.extend(
                summarize_panel_partition(
                    panel
                )
            )

            # =================================================
            # Save Date × W partition
            # =================================================

            if SAVE_JOB_PARTITIONS:

                save_job_partition(

                    panel=
                        panel,

                    window=
                        window,

                    date=
                        date,
                )

            # =================================================
            # Stream into final parquet
            # =================================================

            table = pa.Table.from_pandas(
                panel,
                preserve_index=False,
            )

            if writer is None:

                writer = pq.ParquetWriter(

                    temp_final_path,

                    table.schema,

                    compression="zstd",
                )

            else:

                # ------------------------------------------------
                # Defensive schema validation
                # ------------------------------------------------

                if (
                    table.schema
                    !=
                    writer.schema
                ):

                    raise RuntimeError(
                        "Parquet schema changed across jobs.\n"
                        f"W={window}, date={date.date()}"
                    )

            writer.write_table(
                table
            )

            total_rows += len(
                panel
            )

    finally:

        if writer is not None:

            writer.close()

    # ========================================================
    # 18.4 Atomic finalize
    # ========================================================

    if not temp_final_path.exists():

        raise RuntimeError(
            "Temporary final panel "
            "was not created."
        )

    os.replace(
        temp_final_path,
        FINAL_PANEL_PATH,
    )

    # ========================================================
    # 18.5 Save formal QA
    # ========================================================

    qa_df = pd.DataFrame(
        qa_rows
    )

    qa_df = (
        qa_df.sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    qa_df.to_csv(

        JOB_QA_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 18.6 Coverage summary
    # ========================================================

    coverage = (
        build_coverage_summary(
            qa_df
        )
    )

    coverage.to_csv(

        COVERAGE_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 18.7 Factor descriptive QA
    # ========================================================

    factor_summary = pd.DataFrame(
        summary_rows
    )

    factor_summary = (
        factor_summary.sort_values(
            [
                "window",
                "factor_name",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    factor_summary.to_csv(

        SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 18.8 Save metadata
    # ========================================================

    save_metadata(

        design_hash=
            design_hash,

        panel_row_count=
            total_rows,

        qa=
            qa_df,

        factor_summary=
            factor_summary,
    )

    # ========================================================
    # 18.9 Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Base Node Feature Panel Summary"
    )

    print("=" * 80)

    print(
        f"Total rows: "
        f"{total_rows:,}"
    )

    print(
        f"QA jobs: "
        f"{len(qa_df):,}"
    )

    print(
        f"QA pass: "
        f"{int(qa_df['qa_pass'].sum()):,}"
        f"/{len(qa_df):,}"
    )

    print()

    print(
        coverage.to_string(
            index=False
        )
    )

    print()

    print(
        "Final panel:"
    )

    print(
        FINAL_PANEL_PATH
    )

    print()

    print(
        "Formal QA:"
    )

    print(
        JOB_QA_PATH
    )

    print()

    print(
        "Coverage summary:"
    )

    print(
        COVERAGE_PATH
    )

    print()

    print(
        "Factor descriptive summary:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    print("=" * 80)

    print(
        "Day 5 Step 2 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()




DEGREE_PERCENTILE_TOLERANCE = 1e-10
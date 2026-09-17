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
# Previous Research Day:
# Day 4 network outputs
# ------------------------------------------------------------

PREVIOUS_DAY_ROOT = (
    OUTPUT_ROOT
    / "M1_day4"
)


# ------------------------------------------------------------
# Previous Day Step 2:
# matched-density edge masters
# ------------------------------------------------------------

NETWORK_STEP2_DIR = (
    PREVIOUS_DAY_ROOT
    / "02_step2_matched_networks"
)

EDGE_ROOT = (
    NETWORK_STEP2_DIR
    / "edge_master"
)


RESIDUAL_NETWORK_ID = (
    "M1_RESIDUAL_POSITIVE"
)


# ------------------------------------------------------------
# Day 5
# ------------------------------------------------------------

DAY5_ROOT = (
    OUTPUT_ROOT
    / "M1_day5"
)


# ------------------------------------------------------------
# Day 5 Step 1:
# frozen factor design
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
# Day 5 Step 2:
# base node feature panel
# ------------------------------------------------------------

STEP2_DIR = (
    DAY5_ROOT
    / "02_step2_network_feature_panel"
)

BASE_PANEL_PATH = (
    STEP2_DIR
    / "base_node_feature_panel.parquet"
)

BASE_PANEL_PARTS_DIR = (
    STEP2_DIR
    / "panel_parts"
)

BASE_PANEL_METADATA_PATH = (
    STEP2_DIR
    / "step2_base_node_feature_panel_metadata.json"
)


# ------------------------------------------------------------
# Day 5 Step 3 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY5_ROOT
    / "03_step3_industry_decomposition"
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
    / "industry_decomposed_feature_panel.parquet"
)

JOB_QA_PATH = (
    OUTPUT_DIR
    / "industry_decomposition_job_qa.csv"
)

COVERAGE_PATH = (
    OUTPUT_DIR
    / "industry_decomposition_coverage_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "industry_decomposition_feature_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step3_industry_decomposition_metadata.json"
)


# ============================================================
# 1. Settings
# ============================================================

PRIMARY_DENSITY = 0.01

WINDOWS = [
    60,
    120,
    252,
]

MAIN_INDUSTRY_COLUMN = (
    "industry_id1"
)

SAVE_JOB_PARTITIONS = True

STRICT_QA = True


# ------------------------------------------------------------
# Strength reconstruction:
#
# Step 2 / Step 3 intermediate files may involve float32
# storage, while sums here are accumulated in float64.
#
# Therefore use allclose instead of exact equality.
# ------------------------------------------------------------

STRENGTH_ATOL = 1e-4
STRENGTH_RTOL = 1e-5

# Strength ratios mix float64 edge sums with values previously stored as
# float32. Two float32 epsilons cover rounding of numerator/denominator
# near one. Degree ratios retain the original, tighter bounds.
# Keep source denominators and unmodified ratios for the frozen design.
STRENGTH_RATIO_ATOL = float(2 * np.finfo(np.float32).eps)


# ------------------------------------------------------------
# Percentile convention
# ------------------------------------------------------------

PERCENTILE_TIE_METHOD = (
    "average"
)


# ============================================================
# 2. Frozen Factor Definitions Required by Step 3
# ============================================================

REQUIRED_FROZEN_FACTORS = {

    "same_industry_degree",

    "cross_industry_degree",

    "same_industry_strength",

    "cross_industry_strength",

    "cross_industry_degree_ratio",

    "cross_industry_strength_ratio",
}


# ============================================================
# 3. New Columns
# ============================================================

NEW_FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # Degree decomposition
    # --------------------------------------------------------

    "same_industry_degree",

    "cross_industry_degree",

    "unknown_industry_degree",

    "industry_labeled_degree",

    # --------------------------------------------------------
    # Degree ratios
    # --------------------------------------------------------

    "same_industry_degree_ratio",

    "cross_industry_degree_ratio",

    "unknown_industry_degree_ratio",

    "industry_labeled_degree_share",

    # --------------------------------------------------------
    # Strength decomposition
    # --------------------------------------------------------

    "same_industry_strength",

    "cross_industry_strength",

    "unknown_industry_strength",

    "industry_labeled_strength",

    # --------------------------------------------------------
    # Strength ratios
    # --------------------------------------------------------

    "same_industry_strength_ratio",

    "cross_industry_strength_ratio",

    "unknown_industry_strength_ratio",

    "industry_labeled_strength_share",

    # --------------------------------------------------------
    # Cross-sectional percentile factors
    # --------------------------------------------------------

    "same_industry_degree_percentile",

    "cross_industry_degree_percentile",

    "same_industry_strength_percentile",

    "cross_industry_strength_percentile",
]


# ============================================================
# 4. Utilities
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

    return (
        pd.Series(
            values
        )
        .rank(
            method=
                PERCENTILE_TIE_METHOD,
            pct=True,
        )
        .to_numpy(
            dtype=np.float64
        )
    )


def safe_ratio(
    numerator,
    denominator,
):

    numerator = np.asarray(
        numerator,
        dtype=np.float64,
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64,
    )

    result = np.full(
        len(numerator),
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            numerator
        )
        &
        np.isfinite(
            denominator
        )
        &
        (
            denominator
            >
            0
        )
    )

    result[
        valid
    ] = (

        numerator[
            valid
        ]

        /

        denominator[
            valid
        ]
    )

    return result


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
            "Factor design is not FROZEN."
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
            "Frozen primary network is not "
            f"{RESIDUAL_NETWORK_ID}."
        )

    if not np.isclose(
        float(
            network_design[
                "primary_density"
            ]
        ),
        PRIMARY_DENSITY,
    ):

        raise RuntimeError(
            "Frozen primary density differs from "
            f"{PRIMARY_DENSITY}."
        )

    if (
        sorted(
            network_design[
                "windows"
            ]
        )
        !=
        sorted(
            WINDOWS
        )
    ):

        raise RuntimeError(
            "Frozen windows are inconsistent."
        )

    if (
        network_design[
            "main_industry_level"
        ]
        !=
        MAIN_INDUSTRY_COLUMN
    ):

        raise RuntimeError(
            "Frozen industry definition is inconsistent."
        )

    return (
        config,
        design_hash,
    )


# ============================================================
# 6. Validate Step 2 Data Lineage
# ============================================================

def validate_step2_lineage(
    design_hash,
):

    metadata = load_json(
        BASE_PANEL_METADATA_PATH
    )

    step2_hash = (
        metadata.get(
            "design_hash"
        )
    )

    if (
        step2_hash
        !=
        design_hash
    ):

        raise RuntimeError(
            "Step 2 base panel was generated from "
            "a different frozen design.\n"
            f"Step1 hash: {design_hash}\n"
            f"Step2 hash: {step2_hash}"
        )

    formal_qa = metadata.get(
        "formal_qa",
        {},
    )

    if not formal_qa.get(
        "all_jobs_pass",
        False,
    ):

        raise RuntimeError(
            "Step 2 base panel did not pass all QA."
        )

    return metadata


# ============================================================
# 7. Validate Factor Dictionary
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
            "factor dictionary has no factor_name."
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
# 8. Load Analysis Dates
# ============================================================

def load_analysis_dates():

    if not ANALYSIS_DATES_PATH.exists():

        raise FileNotFoundError(
            f"Missing:\n{ANALYSIS_DATES_PATH}"
        )

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

    required = [

        "analysis_date",

        "window",

        "common_node_count",

        "edge_budget_k",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "factor_analysis_dates missing "
            f"columns: {missing}"
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
# 9. Resolve Base Panel Partition
# ============================================================

def base_panel_partition_path(
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
        BASE_PANEL_PARTS_DIR
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )


def load_base_panel_partition(
    window,
    date,
):

    path = (
        base_panel_partition_path(
            window,
            date,
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            "Base panel partition not found:\n"
            f"{path}"
        )

    df = pd.read_parquet(
        path
    )

    required = [

        "analysis_date",
        "window",
        "master_index",

        "security_id",

        MAIN_INDUSTRY_COLUMN,

        "residual_degree",
        "residual_strength",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Base panel partition missing "
            f"columns: {missing}"
        )

    if (
        df[
            "master_index"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate master_index in base panel."
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
# 10. Load Residual Edge Master
# ============================================================

def residual_edge_path(
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
        EDGE_ROOT
        /
        RESIDUAL_NETWORK_ID
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )


def load_residual_edges(
    window,
    date,
    edge_budget,
):

    path = (
        residual_edge_path(
            window,
            date,
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            "Residual edge master not found:\n"
            f"{path}"
        )

    edges = pd.read_parquet(
        path
    )

    required = [

        "master_index_i",

        "master_index_j",

        "correlation",

        "rank",
    ]

    missing = [
        col
        for col in required
        if col not in edges.columns
    ]

    if missing:

        raise ValueError(
            "Residual edge master missing "
            f"columns: {missing}"
        )

    edges = (
        edges[
            edges[
                "rank"
            ]
            <=
            int(
                edge_budget
            )
        ]
        .sort_values(
            "rank"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            edges
        )
        !=
        int(
            edge_budget
        )
    ):

        raise RuntimeError(
            f"Edge count mismatch: "
            f"{len(edges):,} != "
            f"{edge_budget:,}"
        )

    return edges


# ============================================================
# 11. Industry Label Encoding
#
# Convert arbitrary historical industry IDs to integer codes.
#
# Invalid / missing label -> -1.
# ============================================================

def encode_industry_labels(
    panel,
):

    raw_label = (
        panel[
            MAIN_INDUSTRY_COLUMN
        ]
    )

    label_string = (
        raw_label
        .astype(
            "string"
        )
    )

    invalid_strings = {

        "",

        "-1",

        "nan",

        "None",

        "<NA>",
    }

    valid = (

        raw_label.notna()

        &

        ~label_string.isin(
            invalid_strings
        )
    )

    codes = np.full(
        len(
            panel
        ),
        -1,
        dtype=np.int32,
    )

    if (
        valid.sum()
        >
        0
    ):

        valid_codes, _ = (
            pd.factorize(
                label_string[
                    valid
                ],
                sort=True,
            )
        )

        codes[
            valid.to_numpy()
        ] = (
            valid_codes.astype(
                np.int32
            )
        )

    return (
        codes,
        valid.to_numpy(
            dtype=bool
        ),
    )


# ============================================================
# 12. Build Direct master_index -> local row map
# ============================================================

def build_local_index_map(
    panel,
):

    master_index = (
        panel[
            "master_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if len(
        master_index
    ) == 0:

        raise RuntimeError(
            "Empty node panel."
        )

    max_master = int(
        master_index.max()
    )

    local_map = np.full(
        max_master + 1,
        -1,
        dtype=np.int32,
    )

    local_map[
        master_index
    ] = np.arange(
        len(
            master_index
        ),
        dtype=np.int32,
    )

    return (
        master_index,
        local_map,
    )


# ============================================================
# 13. Decompose Residual Network
# ============================================================

def decompose_industry_network(
    base_panel,
    edges,
):

    n = len(
        base_panel
    )

    (
        master_index,
        local_map,
    ) = (
        build_local_index_map(
            base_panel
        )
    )

    industry_codes, industry_valid = (
        encode_industry_labels(
            base_panel
        )
    )

    edge_i = (
        edges[
            "master_index_i"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    edge_j = (
        edges[
            "master_index_j"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    weights = (
        edges[
            "correlation"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # --------------------------------------------------------
    # Endpoint range QA
    # --------------------------------------------------------

    if (
        edge_i.max(
            initial=-1
        )
        >=
        len(
            local_map
        )
        or
        edge_j.max(
            initial=-1
        )
        >=
        len(
            local_map
        )
    ):

        raise RuntimeError(
            "Edge endpoint master_index exceeds "
            "current node mapping."
        )

    u = local_map[
        edge_i
    ]

    v = local_map[
        edge_j
    ]

    if (
        np.any(
            u < 0
        )
        or
        np.any(
            v < 0
        )
    ):

        raise RuntimeError(
            "Residual edge endpoint not found "
            "in base node panel."
        )

    # --------------------------------------------------------
    # Endpoint industries
    # --------------------------------------------------------

    industry_i = (
        industry_codes[
            u
        ]
    )

    industry_j = (
        industry_codes[
            v
        ]
    )

    labeled_edge = (

        (
            industry_i
            >=
            0
        )

        &

        (
            industry_j
            >=
            0
        )
    )

    same_edge = (

        labeled_edge

        &

        (
            industry_i
            ==
            industry_j
        )
    )

    cross_edge = (

        labeled_edge

        &

        (
            industry_i
            !=
            industry_j
        )
    )

    unknown_edge = (
        ~labeled_edge
    )

    # ========================================================
    # Allocate arrays
    # ========================================================

    same_degree = np.zeros(
        n,
        dtype=np.int32,
    )

    cross_degree = np.zeros(
        n,
        dtype=np.int32,
    )

    unknown_degree = np.zeros(
        n,
        dtype=np.int32,
    )

    same_strength = np.zeros(
        n,
        dtype=np.float64,
    )

    cross_strength = np.zeros(
        n,
        dtype=np.float64,
    )

    unknown_strength = np.zeros(
        n,
        dtype=np.float64,
    )

    # ========================================================
    # Helper: accumulate one edge category
    # ========================================================

    def accumulate(
        mask,
        degree_array,
        strength_array,
    ):

        uu = u[
            mask
        ]

        vv = v[
            mask
        ]

        ww = weights[
            mask
        ]

        np.add.at(
            degree_array,
            uu,
            1,
        )

        np.add.at(
            degree_array,
            vv,
            1,
        )

        np.add.at(
            strength_array,
            uu,
            ww,
        )

        np.add.at(
            strength_array,
            vv,
            ww,
        )

    # ========================================================
    # Same industry
    # ========================================================

    accumulate(
        same_edge,
        same_degree,
        same_strength,
    )

    # ========================================================
    # Cross industry
    # ========================================================

    accumulate(
        cross_edge,
        cross_degree,
        cross_strength,
    )

    # ========================================================
    # Unknown industry
    # ========================================================

    accumulate(
        unknown_edge,
        unknown_degree,
        unknown_strength,
    )

    # ========================================================
    # Derived totals
    # ========================================================

    labeled_degree = (
        same_degree
        +
        cross_degree
    )

    labeled_strength = (
        same_strength
        +
        cross_strength
    )

    total_degree = (
        same_degree
        +
        cross_degree
        +
        unknown_degree
    )

    total_strength = (
        same_strength
        +
        cross_strength
        +
        unknown_strength
    )

    # ========================================================
    # Ratios
    #
    # Frozen factor definition uses total residual degree /
    # strength as denominator.
    # ========================================================

    residual_degree = (
        base_panel[
            "residual_degree"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    residual_strength = (
        base_panel[
            "residual_strength"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    same_degree_ratio = safe_ratio(
        same_degree,
        residual_degree,
    )

    cross_degree_ratio = safe_ratio(
        cross_degree,
        residual_degree,
    )

    unknown_degree_ratio = safe_ratio(
        unknown_degree,
        residual_degree,
    )

    labeled_degree_share = safe_ratio(
        labeled_degree,
        residual_degree,
    )

    same_strength_ratio = safe_ratio(
        same_strength,
        residual_strength,
    )

    cross_strength_ratio = safe_ratio(
        cross_strength,
        residual_strength,
    )

    unknown_strength_ratio = safe_ratio(
        unknown_strength,
        residual_strength,
    )

    labeled_strength_share = safe_ratio(
        labeled_strength,
        residual_strength,
    )

    # ========================================================
    # Cross-sectional percentile features
    # ========================================================

    same_degree_pct = percentile_rank(
        same_degree
    )

    cross_degree_pct = percentile_rank(
        cross_degree
    )

    same_strength_pct = percentile_rank(
        same_strength
    )

    cross_strength_pct = percentile_rank(
        cross_strength
    )

    # ========================================================
    # Attach to base panel
    # ========================================================

    result = (
        base_panel.copy()
    )

    result[
        "same_industry_degree"
    ] = same_degree

    result[
        "cross_industry_degree"
    ] = cross_degree

    result[
        "unknown_industry_degree"
    ] = unknown_degree

    result[
        "industry_labeled_degree"
    ] = labeled_degree


    result[
        "same_industry_degree_ratio"
    ] = same_degree_ratio

    result[
        "cross_industry_degree_ratio"
    ] = cross_degree_ratio

    result[
        "unknown_industry_degree_ratio"
    ] = unknown_degree_ratio

    result[
        "industry_labeled_degree_share"
    ] = labeled_degree_share


    result[
        "same_industry_strength"
    ] = same_strength

    result[
        "cross_industry_strength"
    ] = cross_strength

    result[
        "unknown_industry_strength"
    ] = unknown_strength

    result[
        "industry_labeled_strength"
    ] = labeled_strength


    result[
        "same_industry_strength_ratio"
    ] = same_strength_ratio

    result[
        "cross_industry_strength_ratio"
    ] = cross_strength_ratio

    result[
        "unknown_industry_strength_ratio"
    ] = unknown_strength_ratio

    result[
        "industry_labeled_strength_share"
    ] = labeled_strength_share


    result[
        "same_industry_degree_percentile"
    ] = same_degree_pct

    result[
        "cross_industry_degree_percentile"
    ] = cross_degree_pct

    result[
        "same_industry_strength_percentile"
    ] = same_strength_pct

    result[
        "cross_industry_strength_percentile"
    ] = cross_strength_pct

    # ========================================================
    # Edge-level diagnostics
    # ========================================================

    diagnostics = {

        "edge_count":
            int(
                len(
                    edges
                )
            ),

        "same_industry_edge_count":
            int(
                same_edge.sum()
            ),

        "cross_industry_edge_count":
            int(
                cross_edge.sum()
            ),

        "unknown_industry_edge_count":
            int(
                unknown_edge.sum()
            ),

        "labeled_edge_count":
            int(
                labeled_edge.sum()
            ),

        "same_industry_edge_share_all":
            float(
                same_edge.mean()
            ),

        "cross_industry_edge_share_all":
            float(
                cross_edge.mean()
            ),

        "unknown_industry_edge_share":
            float(
                unknown_edge.mean()
            ),

        "same_industry_edge_share_labeled":
            (
                float(
                    same_edge.sum()
                    /
                    labeled_edge.sum()
                )
                if (
                    labeled_edge.sum()
                    >
                    0
                )
                else np.nan
            ),

        "node_industry_label_coverage":
            float(
                industry_valid.mean()
            ),

        "_reconstructed_degree":
            total_degree,

        "_reconstructed_strength":
            total_strength,
    }

    return (
        result,
        diagnostics,
    )


# ============================================================
# 14. Formal QA
# ============================================================

def qa_one_job(
    panel,
    diagnostics,
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
    # 14.1 Node count
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
    # 14.2 Edge count
    # ========================================================

    edge_count_ok = (
        diagnostics[
            "edge_count"
        ]
        ==
        int(
            expected_edge_count
        )
    )

    if not edge_count_ok:

        errors.append(
            "edge_count_mismatch"
        )

    # ========================================================
    # 14.3 Unique panel key
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
    # 14.4 Edge-category identity
    # ========================================================

    category_edge_sum = (

        diagnostics[
            "same_industry_edge_count"
        ]

        +

        diagnostics[
            "cross_industry_edge_count"
        ]

        +

        diagnostics[
            "unknown_industry_edge_count"
        ]
    )

    edge_category_identity_ok = (
        category_edge_sum
        ==
        int(
            expected_edge_count
        )
    )

    if not edge_category_identity_ok:

        errors.append(
            "edge_category_identity_failure"
        )

    # ========================================================
    # 14.5 Degree decomposition identity
    #
    # Same + Cross + Unknown must EXACTLY reconstruct
    # residual_degree.
    # ========================================================

    reconstructed_degree = (

        panel[
            "same_industry_degree"
        ]

        +

        panel[
            "cross_industry_degree"
        ]

        +

        panel[
            "unknown_industry_degree"
        ]
    )

    source_degree = (
        panel[
            "residual_degree"
        ]
    )

    degree_diff = (

        reconstructed_degree

        -

        source_degree
    )

    max_abs_degree_diff = int(
        degree_diff
        .abs()
        .max()
    )

    degree_reconstruction_ok = (
        max_abs_degree_diff
        ==
        0
    )

    if not degree_reconstruction_ok:

        errors.append(
            "degree_reconstruction_failure"
        )

    # ========================================================
    # 14.6 Degree handshake
    # ========================================================

    degree_sum = int(
        reconstructed_degree.sum()
    )

    expected_degree_sum = (
        2
        *
        int(
            expected_edge_count
        )
    )

    degree_handshake_ok = (
        degree_sum
        ==
        expected_degree_sum
    )

    if not degree_handshake_ok:

        errors.append(
            "degree_handshake_failure"
        )

    # ========================================================
    # 14.7 Strength reconstruction
    # ========================================================

    reconstructed_strength = (

        panel[
            "same_industry_strength"
        ]

        +

        panel[
            "cross_industry_strength"
        ]

        +

        panel[
            "unknown_industry_strength"
        ]
    ).to_numpy(
        dtype=np.float64
    )

    source_strength = (
        panel[
            "residual_strength"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    strength_diff = (

        reconstructed_strength

        -

        source_strength
    )

    max_abs_strength_diff = float(
        np.nanmax(
            np.abs(
                strength_diff
            )
        )
    )

    strength_reconstruction_ok = bool(
        np.allclose(

            reconstructed_strength,

            source_strength,

            atol=
                STRENGTH_ATOL,

            rtol=
                STRENGTH_RTOL,

            equal_nan=True,
        )
    )

    if not strength_reconstruction_ok:

        errors.append(
            "strength_reconstruction_failure"
        )

    # ========================================================
    # 14.8 Ratio range
    # ========================================================

    ratio_columns = [

        "same_industry_degree_ratio",

        "cross_industry_degree_ratio",

        "unknown_industry_degree_ratio",

        "industry_labeled_degree_share",

        "same_industry_strength_ratio",

        "cross_industry_strength_ratio",

        "unknown_industry_strength_ratio",

        "industry_labeled_strength_share",
    ]

    ratio_valid = True

    invalid_ratio_columns = []
    max_ratio_upper_excess = 0.0

    for col in ratio_columns:

        upper_tolerance = (
            STRENGTH_RATIO_ATOL if "strength" in col else 1e-8
        )

        x = pd.to_numeric(
            panel[
                col
            ],
            errors="coerce",
        )

        valid_x = (
            x.dropna()
        )

        if len(
            valid_x
        ) > 0:

            max_ratio_upper_excess = max(
                max_ratio_upper_excess,
                float(max(0.0, valid_x.max() - 1.0)),
            )

            if (

                (
                    valid_x
                    <
                    -1e-10
                ).any()

                or

                (
                    valid_x
                    >
                    1.0
                    +
                    upper_tolerance
                ).any()
            ):

                ratio_valid = False
                invalid_ratio_columns.append(col)

    if not ratio_valid:

        errors.append(
            "invalid_ratio_range"
        )

    # ========================================================
    # 14.9 Degree-ratio identity
    #
    # On active nodes:
    #
    # same + cross + unknown = 1
    # ========================================================

    active = (
        panel[
            "residual_degree"
        ]
        >
        0
    )

    degree_ratio_sum = (

        panel[
            "same_industry_degree_ratio"
        ]

        +

        panel[
            "cross_industry_degree_ratio"
        ]

        +

        panel[
            "unknown_industry_degree_ratio"
        ]
    )

    if (
        active.sum()
        >
        0
    ):

        max_degree_ratio_identity_diff = float(
            np.nanmax(
                np.abs(

                    degree_ratio_sum[
                        active
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )

                    -

                    1.0
                )
            )
        )

    else:

        max_degree_ratio_identity_diff = (
            np.nan
        )

    degree_ratio_identity_ok = bool(

        (
            active.sum()
            ==
            0
        )

        or

        (
            max_degree_ratio_identity_diff
            <=
            1e-10
        )
    )

    if not degree_ratio_identity_ok:

        errors.append(
            "degree_ratio_identity_failure"
        )

    # ========================================================
    # 14.10 Isolated nodes must have zero decomposed degree
    # ========================================================

    isolated = (
        panel[
            "residual_degree"
        ]
        ==
        0
    )

    isolated_degree_ok = bool(

        (
            panel.loc[
                isolated,
                [
                    "same_industry_degree",
                    "cross_industry_degree",
                    "unknown_industry_degree",
                ]
            ]
            ==
            0
        )
        .all()
        .all()
    )

    if not isolated_degree_ok:

        errors.append(
            "isolated_degree_failure"
        )

    # ========================================================
    # 14.11 Final
    # ========================================================

    qa_pass = (
        len(
            errors
        )
        ==
        0
    )

    labeled_degree_share = (
        panel.loc[
            active,
            "industry_labeled_degree_share"
        ]
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

        # ----------------------------------------------------
        # Industry coverage
        # ----------------------------------------------------

        "node_industry_label_coverage":
            diagnostics[
                "node_industry_label_coverage"
            ],

        "same_industry_edge_count":
            diagnostics[
                "same_industry_edge_count"
            ],

        "cross_industry_edge_count":
            diagnostics[
                "cross_industry_edge_count"
            ],

        "unknown_industry_edge_count":
            diagnostics[
                "unknown_industry_edge_count"
            ],

        "same_industry_edge_share_all":
            diagnostics[
                "same_industry_edge_share_all"
            ],

        "cross_industry_edge_share_all":
            diagnostics[
                "cross_industry_edge_share_all"
            ],

        "unknown_industry_edge_share":
            diagnostics[
                "unknown_industry_edge_share"
            ],

        "same_industry_edge_share_labeled":
            diagnostics[
                "same_industry_edge_share_labeled"
            ],

        # ----------------------------------------------------
        # Node-level industry coverage
        # ----------------------------------------------------

        "mean_industry_labeled_degree_share_active":
            (
                float(
                    labeled_degree_share.mean()
                )
                if len(
                    labeled_degree_share
                ) > 0
                else np.nan
            ),

        "min_industry_labeled_degree_share_active":
            (
                float(
                    labeled_degree_share.min()
                )
                if len(
                    labeled_degree_share
                ) > 0
                else np.nan
            ),

        # ----------------------------------------------------
        # Reconstruction diagnostics
        # ----------------------------------------------------

        "max_abs_degree_reconstruction_diff":
            max_abs_degree_diff,

        "max_abs_strength_reconstruction_diff":
            max_abs_strength_diff,

        "max_degree_ratio_identity_diff":
            max_degree_ratio_identity_diff,

        # ----------------------------------------------------
        # Formal QA
        # ----------------------------------------------------

        "node_count_ok":
            node_count_ok,

        "edge_count_ok":
            edge_count_ok,

        "key_unique":
            key_unique,

        "edge_category_identity_ok":
            edge_category_identity_ok,

        "degree_reconstruction_ok":
            degree_reconstruction_ok,

        "degree_handshake_ok":
            degree_handshake_ok,

        "strength_reconstruction_ok":
            strength_reconstruction_ok,

        "ratio_valid":
            ratio_valid,

        "invalid_ratio_columns":
            "|".join(invalid_ratio_columns),

        "max_ratio_upper_excess":
            max_ratio_upper_excess,

        "strength_ratio_atol":
            STRENGTH_RATIO_ATOL,

        "degree_ratio_identity_ok":
            degree_ratio_identity_ok,

        "isolated_degree_ok":
            isolated_degree_ok,

        "qa_pass":
            qa_pass,

        "qa_errors":
            "|".join(
                errors
            ),
    }


# ============================================================
# 15. Save One Job Partition
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
# 16. Descriptive Summary for One Panel
# ============================================================

def summarize_one_panel(
    panel,
):

    factor_columns = [

        "same_industry_degree",

        "cross_industry_degree",

        "same_industry_degree_ratio",

        "cross_industry_degree_ratio",

        "same_industry_strength",

        "cross_industry_strength",

        "same_industry_strength_ratio",

        "cross_industry_strength_ratio",

        "same_industry_degree_percentile",

        "cross_industry_degree_percentile",

        "same_industry_strength_percentile",

        "cross_industry_strength_percentile",

        "industry_labeled_degree_share",

        "industry_labeled_strength_share",
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

            quantiles = valid.quantile(
                [
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                ]
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
# 17. Coverage Summary
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

                "qa_pass_share":
                    float(
                        group[
                            "qa_pass"
                        ]
                        .mean()
                    ),

                "mean_node_industry_label_coverage":
                    float(
                        group[
                            "node_industry_label_coverage"
                        ]
                        .mean()
                    ),

                "mean_same_industry_edge_share_all":
                    float(
                        group[
                            "same_industry_edge_share_all"
                        ]
                        .mean()
                    ),

                "mean_cross_industry_edge_share_all":
                    float(
                        group[
                            "cross_industry_edge_share_all"
                        ]
                        .mean()
                    ),

                "mean_unknown_industry_edge_share":
                    float(
                        group[
                            "unknown_industry_edge_share"
                        ]
                        .mean()
                    ),

                "mean_same_industry_edge_share_labeled":
                    float(
                        group[
                            "same_industry_edge_share_labeled"
                        ]
                        .mean()
                    ),

                "mean_industry_labeled_degree_share_active":
                    float(
                        group[
                            "mean_industry_labeled_degree_share_active"
                        ]
                        .mean()
                    ),

                "max_degree_reconstruction_error":
                    int(
                        group[
                            "max_abs_degree_reconstruction_diff"
                        ]
                        .max()
                    ),

                "max_strength_reconstruction_error":
                    float(
                        group[
                            "max_abs_strength_reconstruction_diff"
                        ]
                        .max()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. Metadata
# ============================================================

def save_metadata(
    design_hash,
    row_count,
    qa,
    summary,
):

    payload = {

        "research_day":
            5,

        "step":
            "Step3_Industry_Backbone_Decomposition",

        "design_hash":
            design_hash,

        "input_base_panel":
            str(
                BASE_PANEL_PATH
            ),

        "input_residual_edge_root":
            str(
                EDGE_ROOT
                /
                RESIDUAL_NETWORK_ID
            ),

        "output_panel":
            str(
                FINAL_PANEL_PATH
            ),

        "primary_network":
            RESIDUAL_NETWORK_ID,

        "primary_density":
            PRIMARY_DENSITY,

        "industry_definition":
            MAIN_INDUSTRY_COLUMN,

        "panel_key": [

            "analysis_date",

            "window",

            "master_index",
        ],

        "row_count":
            int(
                row_count
            ),

        "decomposition_identity": {

            "degree":
                (
                    "ResidualDegree = "
                    "SameIndustryDegree + "
                    "CrossIndustryDegree + "
                    "UnknownIndustryDegree"
                ),

            "strength":
                (
                    "ResidualStrength approximately equals "
                    "SameIndustryStrength + "
                    "CrossIndustryStrength + "
                    "UnknownIndustryStrength"
                ),
        },

        "factor_definitions": {

            "same_industry_degree":
                (
                    "Number of residual-network neighbors "
                    "with the same historical industry_id1."
                ),

            "cross_industry_degree":
                (
                    "Number of residual-network neighbors "
                    "with a different historical industry_id1."
                ),

            "cross_industry_degree_ratio":
                (
                    "CrossIndustryDegree / ResidualDegree "
                    "for ResidualDegree > 0."
                ),

            "same_industry_strength":
                (
                    "Sum of residual edge correlations "
                    "to same-industry neighbors."
                ),

            "cross_industry_strength":
                (
                    "Sum of residual edge correlations "
                    "to cross-industry neighbors."
                ),

            "cross_industry_strength_ratio":
                (
                    "CrossIndustryStrength / "
                    "ResidualStrength for "
                    "ResidualStrength > 0."
                ),
        },

        "missing_industry_policy":
            (
                "Any edge with at least one missing/invalid "
                "industry label is classified as Unknown. "
                "It is never silently treated as Cross."
            ),

        "ratio_policy":
            (
                "Ratios are NaN when total residual degree "
                "or strength is zero. Original source denominators and "
                "computed ratios are retained without clipping. Strength "
                "ratio QA allows two float32 epsilons above one for "
                "storage rounding; degree-ratio bounds are unchanged."
            ),

        "strength_ratio_qa_atol": STRENGTH_RATIO_ATOL,

        "interpretation_notes": [

            (
                "Same-industry features characterize the "
                "industry backbone identified in Day 4."
            ),

            (
                "Cross-industry features characterize "
                "residual associations that extend beyond "
                "conventional industry boundaries."
            ),

            (
                "Cross-industry connectivity is not assumed "
                "to be Alpha-positive; its predictive sign "
                "will be evaluated only in later steps."
            ),

            (
                "Industry labels are contemporaneous "
                "historically indexed labels already stored "
                "in the PIT node panel."
            ),

            (
                "No future return or future-risk outcome is "
                "used in Step 3."
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
        },

        "summary_row_count":
            int(
                len(
                    summary
                )
            ),
    }

    save_json(
        payload,
        METADATA_PATH,
    )


# ============================================================
# 19. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 5 - Step 3"
    )

    print(
        "Industry Backbone Decomposition"
    )

    print("=" * 80)

    # ========================================================
    # 19.1 Frozen design
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

    validate_step2_lineage(
        design_hash
    )

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
    # 19.2 Prepare streaming parquet
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
    # 19.3 Loop
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
            # Load Day 5 Step 2 base panel
            # =================================================

            base_panel = (
                load_base_panel_partition(
                    window=
                        window,

                    date=
                        date,
                )
            )

            # =================================================
            # Load primary Residual edges
            # =================================================

            edges = (
                load_residual_edges(

                    window=
                        window,

                    date=
                        date,

                    edge_budget=
                        expected_edges,
                )
            )

            # =================================================
            # Industry decomposition
            # =================================================

            (
                panel,
                diagnostics,
            ) = (
                decompose_industry_network(

                    base_panel=
                        base_panel,

                    edges=
                        edges,
                )
            )

            # =================================================
            # QA
            # =================================================

            qa = (
                qa_one_job(

                    panel=
                        panel,

                    diagnostics=
                        diagnostics,

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

            print(
                "  Same-edge share "
                f"(labeled) = "
                f"{qa['same_industry_edge_share_labeled']:.4f}"
            )

            print(
                "  Industry label coverage = "
                f"{qa['node_industry_label_coverage']:.4f}"
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
            # Summary
            # =================================================

            summary_rows.extend(
                summarize_one_panel(
                    panel
                )
            )

            # =================================================
            # Save partition
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
            # Stream final panel
            # =================================================

            table = (
                pa.Table.from_pandas(
                    panel,
                    preserve_index=False,
                )
            )

            if writer is None:

                writer = (
                    pq.ParquetWriter(

                        temp_final_path,

                        table.schema,

                        compression="zstd",
                    )
                )

            else:

                if (
                    table.schema
                    !=
                    writer.schema
                ):

                    raise RuntimeError(
                        "Parquet schema changed across jobs.\n"
                        f"W={window}, "
                        f"date={date.date()}"
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
    # 19.4 Atomic finalization
    # ========================================================

    if not temp_final_path.exists():

        raise RuntimeError(
            "Temporary panel was not created."
        )

    os.replace(
        temp_final_path,
        FINAL_PANEL_PATH,
    )

    # ========================================================
    # 19.5 QA output
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
    # 19.6 Coverage summary
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
    # 19.7 Factor descriptive summary
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = (
        summary_df.sort_values(
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

    summary_df.to_csv(

        SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 19.8 Metadata
    # ========================================================

    save_metadata(

        design_hash=
            design_hash,

        row_count=
            total_rows,

        qa=
            qa_df,

        summary=
            summary_df,
    )

    # ========================================================
    # 19.9 Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Industry Backbone Decomposition Summary"
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

    print("=" * 80)

    print(
        "Day 5 Step 3 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()

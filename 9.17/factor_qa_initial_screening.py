from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

DAY5_ROOT = (
    OUTPUT_ROOT
    / "M1_day5"
)


# ------------------------------------------------------------
# Day 5 Step 1
# ------------------------------------------------------------

STEP1_DIR = (
    DAY5_ROOT
    / "01_step1_factor_design"
)

FACTOR_CONFIG_PATH = (
    STEP1_DIR
    / "factor_design_config.json"
)

FACTOR_DICTIONARY_PATH = (
    STEP1_DIR
    / "network_factor_dictionary.csv"
)

ANALYSIS_DATES_PATH = (
    STEP1_DIR
    / "factor_analysis_dates.csv"
)


# ------------------------------------------------------------
# Day 5 Step 5
# ------------------------------------------------------------

STEP5_DIR = (
    DAY5_ROOT
    / "05_step5_community_features"
)

STEP5_PANEL_PATH = (
    STEP5_DIR
    / "community_bridge_feature_panel.parquet"
)

STEP5_PARTITION_DIR = (
    STEP5_DIR
    / "panel_parts"
)

STEP5_METADATA_PATH = (
    STEP5_DIR
    / "step5_community_bridge_metadata.json"
)


# ------------------------------------------------------------
# Day 5 Step 6
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY5_ROOT
    / "06_step6_factor_qa_screening"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MONTHLY_QA_PATH = (
    OUTPUT_DIR
    / "factor_monthly_qa.csv"
)

QA_SUMMARY_PATH = (
    OUTPUT_DIR
    / "factor_qa_summary.csv"
)

TEMPORAL_PATH = (
    OUTPUT_DIR
    / "factor_temporal_stability.csv"
)

PAIR_MONTHLY_PATH = (
    OUTPUT_DIR
    / "factor_pair_monthly_spearman.csv"
)

PAIR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "factor_pair_correlation_summary.csv"
)

REDUNDANCY_PATH = (
    OUTPUT_DIR
    / "factor_redundancy_pairs.csv"
)

SCREENING_PATH = (
    OUTPUT_DIR
    / "factor_screening_decisions.csv"
)

CORE_SET_PATH = (
    OUTPUT_DIR
    / "recommended_preoutcome_core_set.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "step6_factor_qa_screening_metadata.json"
)


# ============================================================
# 1. Global Settings
# ============================================================

WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# Minimum cross-sectional observations required for
# pairwise / temporal Spearman correlation.
# ------------------------------------------------------------

MIN_PAIR_OBS = 100

MIN_TEMPORAL_OBS = 100


# ------------------------------------------------------------
# QA thresholds
# ------------------------------------------------------------

MIN_ELIGIBLE_NONMISSING_SHARE = 0.80

MIN_VARYING_JOB_SHARE = 0.90

STD_EPS = 1e-12


# ------------------------------------------------------------
# Redundancy threshold
#
# A pair is flagged if:
#
# median |monthly Spearman| >= 0.95
#
# AND
#
# at least 80% of valid months have |rho| >= 0.90.
# ------------------------------------------------------------

REDUNDANCY_MEDIAN_ABS_RHO = 0.95

REDUNDANCY_MONTH_SHARE_THRESHOLD = 0.80

REDUNDANCY_MONTH_ABS_RHO = 0.90


# ============================================================
# 2. Candidate Factor Specification
#
# priority:
# smaller number = preferred representative when two factors
# are almost redundant.
#
# tier:
# core      -> preferred economic factor
# auxiliary -> useful robustness / interpretation factor
#
# lower_bound / upper_bound:
# used only for QA.
# ============================================================

FACTOR_SPECS = {

    # ========================================================
    # A. Overall Residual Centrality
    # ========================================================

    "residual_degree_percentile": {
        "family": "centrality",
        "tier": "core",
        "priority": 10,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Residual-network degree cross-sectional percentile.",
    },

    "residual_strength_percentile": {
        "family": "centrality",
        "tier": "auxiliary",
        "priority": 20,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Residual-network weighted strength percentile.",
    },

    "delta_degree_percentile": {
        "family": "market_residualization",
        "tier": "core",
        "priority": 30,
        "lower_bound": -1.0,
        "upper_bound": 1.0,
        "meaning": "Residual minus Raw degree percentile.",
    },

    "delta_strength_percentile": {
        "family": "market_residualization",
        "tier": "auxiliary",
        "priority": 40,
        "lower_bound": -1.0,
        "upper_bound": 1.0,
        "meaning": "Residual minus Raw strength percentile.",
    },


    # ========================================================
    # B. Industry Decomposition
    # ========================================================

    "same_industry_degree_percentile": {
        "family": "industry",
        "tier": "auxiliary",
        "priority": 50,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Same-industry degree percentile.",
    },

    "same_industry_strength_percentile": {
        "family": "industry",
        "tier": "auxiliary",
        "priority": 60,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Same-industry strength percentile.",
    },

    "cross_industry_degree_percentile": {
        "family": "industry",
        "tier": "core",
        "priority": 70,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Cross-industry degree percentile.",
    },

    "cross_industry_strength_percentile": {
        "family": "industry",
        "tier": "auxiliary",
        "priority": 80,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Cross-industry strength percentile.",
    },

    "cross_industry_degree_ratio": {
        "family": "industry",
        "tier": "core",
        "priority": 90,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Fraction of residual degree going across industries.",
    },

    "cross_industry_strength_ratio": {
        "family": "industry",
        "tier": "auxiliary",
        "priority": 100,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Fraction of residual strength going across industries.",
    },


    # ========================================================
    # C. Dynamic Network Features
    # ========================================================

    "delta_residual_degree_percentile_1m": {
        "family": "dynamic",
        "tier": "core",
        "priority": 110,
        "lower_bound": -1.0,
        "upper_bound": 1.0,
        "meaning": "Adjacent-period change in residual degree percentile.",
    },

    "delta_residual_strength_percentile_1m": {
        "family": "dynamic",
        "tier": "auxiliary",
        "priority": 120,
        "lower_bound": -1.0,
        "upper_bound": 1.0,
        "meaning": "Adjacent-period change in residual strength percentile.",
    },

    "delta_same_industry_degree_percentile_1m": {
        "family": "dynamic_industry",
        "tier": "auxiliary",
        "priority": 130,
        "lower_bound": -1.0,
        "upper_bound": 1.0,
        "meaning": "Change in same-industry degree percentile.",
    },

    "delta_cross_industry_degree_percentile_1m": {
        "family": "dynamic_industry",
        "tier": "core",
        "priority": 140,
        "lower_bound": -1.0,
        "upper_bound": 1.0,
        "meaning": "Change in cross-industry degree percentile.",
    },

    "delta_cross_industry_degree_ratio_1m": {
        "family": "dynamic_industry",
        "tier": "auxiliary",
        "priority": 150,
        "lower_bound": -1.0,
        "upper_bound": 1.0,
        "meaning": "Change in cross-industry degree share.",
    },

    "neighbor_jaccard_1m": {
        "family": "dynamic_neighbor",
        "tier": "core",
        "priority": 160,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Common-node-conditioned adjacent-period neighbor Jaccard.",
    },

    "neighbor_retention_1m": {
        "family": "dynamic_neighbor",
        "tier": "core",
        "priority": 170,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Fraction of previous neighbors retained.",
    },

    "neighbor_turnover_1m": {
        "family": "dynamic_neighbor",
        "tier": "auxiliary",
        "priority": 999,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "1 minus neighbor Jaccard; algebraically redundant.",
    },

    "new_neighbor_count_1m": {
        "family": "dynamic_neighbor",
        "tier": "auxiliary",
        "priority": 180,
        "lower_bound": 0.0,
        "upper_bound": None,
        "meaning": "Number of newly formed common-node-conditioned links.",
    },

    "lost_neighbor_count_1m": {
        "family": "dynamic_neighbor",
        "tier": "auxiliary",
        "priority": 190,
        "lower_bound": 0.0,
        "upper_bound": None,
        "meaning": "Number of lost common-node-conditioned links.",
    },


    # ========================================================
    # D. Community / Bridge Features
    # ========================================================

    "within_community_degree_percentile": {
        "family": "community",
        "tier": "auxiliary",
        "priority": 200,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Within-community degree percentile.",
    },

    "within_community_strength_percentile": {
        "family": "community",
        "tier": "auxiliary",
        "priority": 210,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Within-community strength percentile.",
    },

    "outside_community_degree_percentile": {
        "family": "community",
        "tier": "core",
        "priority": 220,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Outside-community degree percentile.",
    },

    "outside_community_strength_percentile": {
        "family": "community",
        "tier": "auxiliary",
        "priority": 230,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Outside-community strength percentile.",
    },

    "outside_community_degree_ratio": {
        "family": "community",
        "tier": "auxiliary",
        "priority": 240,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Share of residual degree outside own community.",
    },

    "participation_coefficient_percentile": {
        "family": "community_bridge",
        "tier": "core",
        "priority": 250,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Cross-sectional percentile of participation coefficient.",
    },

    "weighted_participation_coefficient_percentile": {
        "family": "community_bridge",
        "tier": "auxiliary",
        "priority": 260,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "meaning": "Weighted participation coefficient percentile.",
    },

    "external_community_count": {
        "family": "community_bridge",
        "tier": "auxiliary",
        "priority": 270,
        "lower_bound": 0.0,
        "upper_bound": None,
        "meaning": "Number of distinct external communities reached.",
    },

    "community_diversity_count": {
        "family": "community_bridge",
        "tier": "auxiliary",
        "priority": 280,
        "lower_bound": 0.0,
        "upper_bound": None,
        "meaning": "Number of distinct neighbor communities.",
    },
}


FACTOR_NAMES = list(
    FACTOR_SPECS.keys()
)


# ============================================================
# 3. Helper Columns Required for Eligibility Definitions
# ============================================================

HELPER_COLUMNS = [

    "analysis_date",

    "window",

    "master_index",

    "residual_degree",

    "residual_strength",

    "dynamic_comparable",

    "previous_common_neighbor_count_1m",

    "current_common_neighbor_count_1m",

    "retained_neighbor_count_1m",
]


# ============================================================
# 4. Utilities
# ============================================================

def load_json(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
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


# ============================================================
# 5. Load Frozen Design
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

    return (
        config,
        design_hash,
    )


# ============================================================
# 6. Validate Step 5 Lineage
# ============================================================

def validate_step5_lineage(
    design_hash,
):

    metadata = load_json(
        STEP5_METADATA_PATH
    )

    if (
        metadata.get(
            "design_hash"
        )
        !=
        design_hash
    ):

        raise RuntimeError(
            "Step 5 was generated from "
            "a different frozen design."
        )

    formal_qa = (
        metadata.get(
            "formal_qa",
            {}
        )
    )

    if not formal_qa.get(
        "all_jobs_pass",
        False,
    ):

        raise RuntimeError(
            "Step 5 did not pass all formal QA."
        )

    return metadata


# ============================================================
# 7. Frozen Dictionary Annotation
# ============================================================

def load_factor_dictionary():

    if not FACTOR_DICTIONARY_PATH.exists():

        return set()

    df = pd.read_csv(
        FACTOR_DICTIONARY_PATH
    )

    if (
        "factor_name"
        not in df.columns
    ):

        return set()

    return set(
        df[
            "factor_name"
        ]
        .astype(str)
        .tolist()
    )


# ============================================================
# 8. Load Analysis Dates
# ============================================================

def load_analysis_dates():

    df = pd.read_csv(
        ANALYSIS_DATES_PATH
    )

    required = [

        "analysis_date",

        "window",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "factor_analysis_dates.csv "
            f"missing {missing}"
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
# 9. Partition Paths
# ============================================================

def partition_path(
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
        STEP5_PARTITION_DIR
        /
        f"W{window}"
        /
        f"{date_string}.parquet"
    )


# ============================================================
# 10. Load One Partition
# ============================================================

def load_partition(
    window,
    date,
):

    path = partition_path(
        window,
        date,
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing Step-5 partition:\n{path}"
        )

    # --------------------------------------------------------
    # Read schema first so missing candidate columns can be
    # reported clearly.
    # --------------------------------------------------------

    import pyarrow.parquet as pq

    schema_names = (
        pq.ParquetFile(
            path
        )
        .schema_arrow
        .names
    )

    missing_factors = [
        factor
        for factor in FACTOR_NAMES
        if factor not in schema_names
    ]

    if missing_factors:

        raise RuntimeError(
            "Step-5 panel does not contain "
            "the following candidate factors:\n"
            f"{missing_factors}"
        )

    required_columns = list(
        dict.fromkeys(
            HELPER_COLUMNS
            +
            FACTOR_NAMES
        )
    )

    missing_helpers = [
        col
        for col in required_columns
        if col not in schema_names
    ]

    if missing_helpers:

        raise RuntimeError(
            "Missing required columns:\n"
            f"{missing_helpers}"
        )

    df = pd.read_parquet(
        path,
        columns=required_columns,
    )

    df[
        "analysis_date"
    ] = pd.to_datetime(
        df[
            "analysis_date"
        ]
    )

    if (
        df[
            "master_index"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate master_index in "
            f"W={window}, date={date}"
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
# 11. Eligibility Definition
#
# This is important because some missing values are
# mathematically undefined, not data errors.
# ============================================================

def get_eligible_mask(
    df,
    factor,
):

    n = len(
        df
    )

    # --------------------------------------------------------
    # Neighbor Jaccard / turnover:
    #
    # union must be non-empty.
    # --------------------------------------------------------

    if factor in {

        "neighbor_jaccard_1m",

        "neighbor_turnover_1m",
    }:

        union_size = (

            pd.to_numeric(
                df[
                    "previous_common_neighbor_count_1m"
                ],
                errors="coerce",
            )

            +

            pd.to_numeric(
                df[
                    "current_common_neighbor_count_1m"
                ],
                errors="coerce",
            )

            -

            pd.to_numeric(
                df[
                    "retained_neighbor_count_1m"
                ],
                errors="coerce",
            )
        )

        return (

            df[
                "dynamic_comparable"
            ]
            .fillna(
                False
            )
            .to_numpy(
                dtype=bool
            )

            &

            (
                union_size
                >
                0
            )
            .fillna(
                False
            )
            .to_numpy(
                dtype=bool
            )
        )

    # --------------------------------------------------------
    # Retention:
    #
    # previous neighbor set must be non-empty.
    # --------------------------------------------------------

    if factor == "neighbor_retention_1m":

        return (

            df[
                "dynamic_comparable"
            ]
            .fillna(
                False
            )
            .to_numpy(
                dtype=bool
            )

            &

            (
                pd.to_numeric(
                    df[
                        "previous_common_neighbor_count_1m"
                    ],
                    errors="coerce",
                )
                >
                0
            )
            .fillna(
                False
            )
            .to_numpy(
                dtype=bool
            )
        )

    # --------------------------------------------------------
    # New / lost neighbor counts are defined for every
    # dynamically comparable stock.
    # --------------------------------------------------------

    if factor in {

        "new_neighbor_count_1m",

        "lost_neighbor_count_1m",
    }:

        return (
            df[
                "dynamic_comparable"
            ]
            .fillna(
                False
            )
            .to_numpy(
                dtype=bool
            )
        )

    # --------------------------------------------------------
    # All delta_*_1m factors require t-1 observation.
    # --------------------------------------------------------

    if (
        factor.startswith(
            "delta_"
        )
        and
        factor.endswith(
            "_1m"
        )
    ):

        return (
            df[
                "dynamic_comparable"
            ]
            .fillna(
                False
            )
            .to_numpy(
                dtype=bool
            )
        )

    # --------------------------------------------------------
    # Degree-ratio / participation variables require
    # non-isolated residual nodes.
    # --------------------------------------------------------

    if factor in {

        "cross_industry_degree_ratio",

        "outside_community_degree_ratio",

        "participation_coefficient_percentile",

        "weighted_participation_coefficient_percentile",
    }:

        return (
            pd.to_numeric(
                df[
                    "residual_degree"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
            >
            0
        )

    # --------------------------------------------------------
    # Strength ratio requires positive total strength.
    # --------------------------------------------------------

    if factor == "cross_industry_strength_ratio":

        return (
            pd.to_numeric(
                df[
                    "residual_strength"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
            >
            0
        )

    # --------------------------------------------------------
    # Other static percentile/count factors:
    # all current nodes are eligible.
    # --------------------------------------------------------

    return np.ones(
        n,
        dtype=bool,
    )


# ============================================================
# 12. Monthly Factor QA
# ============================================================

def monthly_factor_qa(
    df,
    factor,
    window,
    date,
):

    spec = (
        FACTOR_SPECS[
            factor
        ]
    )

    x = pd.to_numeric(
        df[
            factor
        ],
        errors="coerce",
    )

    eligible = get_eligible_mask(
        df,
        factor,
    )

    eligible_count = int(
        eligible.sum()
    )

    x_eligible = (
        x[
            eligible
        ]
    )

    finite = np.isfinite(
        x_eligible
        .to_numpy(
            dtype=np.float64
        )
    )

    finite_values = (
        x_eligible.iloc[
            np.flatnonzero(
                finite
            )
        ]
    )

    nonmissing_count = int(
        len(
            finite_values
        )
    )

    if eligible_count > 0:

        nonmissing_share = (
            nonmissing_count
            /
            eligible_count
        )

    else:

        nonmissing_share = np.nan

    if nonmissing_count > 0:

        mean_value = float(
            finite_values.mean()
        )

        std_value = float(
            finite_values.std(
                ddof=0
            )
        )

        q01 = float(
            finite_values.quantile(
                0.01
            )
        )

        q25 = float(
            finite_values.quantile(
                0.25
            )
        )

        q50 = float(
            finite_values.quantile(
                0.50
            )
        )

        q75 = float(
            finite_values.quantile(
                0.75
            )
        )

        q99 = float(
            finite_values.quantile(
                0.99
            )
        )

        iqr = (
            q75
            -
            q25
        )

        unique_count = int(
            finite_values.nunique(
                dropna=True
            )
        )

        zero_share = float(
            (
                finite_values
                ==
                0
            )
            .mean()
        )

    else:

        mean_value = np.nan

        std_value = np.nan

        q01 = np.nan
        q25 = np.nan
        q50 = np.nan
        q75 = np.nan
        q99 = np.nan

        iqr = np.nan

        unique_count = 0

        zero_share = np.nan

    varying = bool(

        (
            nonmissing_count
            >=
            2
        )

        and

        (
            unique_count
            >=
            2
        )

        and

        np.isfinite(
            std_value
        )

        and

        (
            std_value
            >
            STD_EPS
        )
    )

    # ========================================================
    # Range validation
    # ========================================================

    range_violation_count = 0

    lower_bound = (
        spec[
            "lower_bound"
        ]
    )

    upper_bound = (
        spec[
            "upper_bound"
        ]
    )

    if nonmissing_count > 0:

        if lower_bound is not None:

            range_violation_count += int(
                (
                    finite_values
                    <
                    lower_bound
                    -
                    1e-10
                )
                .sum()
            )

        if upper_bound is not None:

            range_violation_count += int(
                (
                    finite_values
                    >
                    upper_bound
                    +
                    1e-10
                )
                .sum()
            )

    return {

        "analysis_date":
            pd.Timestamp(
                date
            ),

        "window":
            int(
                window
            ),

        "factor_name":
            factor,

        "family":
            spec[
                "family"
            ],

        "tier":
            spec[
                "tier"
            ],

        "row_count":
            int(
                len(
                    df
                )
            ),

        "eligible_count":
            eligible_count,

        "nonmissing_count":
            nonmissing_count,

        "eligible_nonmissing_share":
            nonmissing_share,

        "mean":
            mean_value,

        "std":
            std_value,

        "q01":
            q01,

        "q25":
            q25,

        "median":
            q50,

        "q75":
            q75,

        "q99":
            q99,

        "iqr":
            iqr,

        "unique_count":
            unique_count,

        "zero_share":
            zero_share,

        "varying":
            varying,

        "range_violation_count":
            int(
                range_violation_count
            ),
    }


# ============================================================
# 13. Monthly Pairwise Spearman
#
# IMPORTANT:
#
# We compute correlations within each Date × Window first.
#
# We DO NOT simply pool all stock-month observations together,
# because pooled correlations can be driven by time variation.
# ============================================================

def monthly_pairwise_correlations(
    df,
    window,
    date,
):

    factor_df = pd.DataFrame(
        {
            factor:
                pd.to_numeric(
                    df[
                        factor
                    ],
                    errors="coerce",
                )

            for factor in FACTOR_NAMES
        }
    )

    correlation = factor_df.corr(
        method="spearman",
        min_periods=
            MIN_PAIR_OBS,
    )

    finite_matrix = np.isfinite(
        factor_df.to_numpy(
            dtype=np.float64
        )
    ).astype(
        np.int32
    )

    pair_count = (
        finite_matrix.T
        @
        finite_matrix
    )

    rows = []

    k = len(
        FACTOR_NAMES
    )

    for i in range(
        k
    ):

        for j in range(
            i + 1,
            k
        ):

            factor_a = (
                FACTOR_NAMES[
                    i
                ]
            )

            factor_b = (
                FACTOR_NAMES[
                    j
                ]
            )

            rho = correlation.loc[
                factor_a,
                factor_b,
            ]

            n_pair = int(
                pair_count[
                    i,
                    j
                ]
            )

            if not np.isfinite(
                rho
            ):

                continue

            rows.append(
                {

                    "analysis_date":
                        pd.Timestamp(
                            date
                        ),

                    "window":
                        int(
                            window
                        ),

                    "factor_a":
                        factor_a,

                    "factor_b":
                        factor_b,

                    "n_pair":
                        n_pair,

                    "spearman_rho":
                        float(
                            rho
                        ),

                    "abs_spearman_rho":
                        float(
                            abs(
                                rho
                            )
                        ),
                }
            )

    return rows


# ============================================================
# 14. Adjacent-period Temporal Stability
#
# For each factor:
#
# Spearman(
#     Factor_{i,t-1},
#     Factor_{i,t}
# )
#
# on stocks observed at both dates.
# ============================================================

def temporal_stability(
    current,
    previous,
    window,
    date,
    previous_date,
):

    if previous is None:

        return []

    current_index = (
        current[
            [
                "master_index"
            ]
            +
            FACTOR_NAMES
        ]
        .set_index(
            "master_index"
        )
    )

    previous_index = (
        previous[
            [
                "master_index"
            ]
            +
            FACTOR_NAMES
        ]
        .set_index(
            "master_index"
        )
    )

    common = (
        current_index.index
        .intersection(
            previous_index.index
        )
    )

    rows = []

    for factor in FACTOR_NAMES:

        x_prev = pd.to_numeric(
            previous_index.loc[
                common,
                factor
            ],
            errors="coerce",
        )

        x_curr = pd.to_numeric(
            current_index.loc[
                common,
                factor
            ],
            errors="coerce",
        )

        valid = (

            np.isfinite(
                x_prev.to_numpy(
                    dtype=np.float64
                )
            )

            &

            np.isfinite(
                x_curr.to_numpy(
                    dtype=np.float64
                )
            )
        )

        n_valid = int(
            valid.sum()
        )

        if (
            n_valid
            <
            MIN_TEMPORAL_OBS
        ):

            continue

        x1 = (
            x_prev.iloc[
                np.flatnonzero(
                    valid
                )
            ]
        )

        x2 = (
            x_curr.iloc[
                np.flatnonzero(
                    valid
                )
            ]
        )

        if (
            x1.nunique()
            <
            2
            or
            x2.nunique()
            <
            2
        ):

            continue

        rho = x1.corr(
            x2,
            method="spearman",
        )

        if not np.isfinite(
            rho
        ):

            continue

        rows.append(
            {

                "analysis_date":
                    pd.Timestamp(
                        date
                    ),

                "previous_analysis_date":
                    pd.Timestamp(
                        previous_date
                    ),

                "window":
                    int(
                        window
                    ),

                "factor_name":
                    factor,

                "n_common":
                    n_valid,

                "temporal_spearman":
                    float(
                        rho
                    ),
            }
        )

    return rows


# ============================================================
# 15. Aggregate QA Summary
# ============================================================

def build_qa_summary(
    monthly_qa,
):

    rows = []

    for (
        factor,
        window
    ), group in monthly_qa.groupby(
        [
            "factor_name",
            "window",
        ]
    ):

        eligible_jobs = (
            group[
                group[
                    "eligible_count"
                ]
                >
                0
            ]
        )

        if len(
            eligible_jobs
        ) > 0:

            mean_nonmissing = float(
                eligible_jobs[
                    "eligible_nonmissing_share"
                ]
                .mean()
            )

            min_nonmissing = float(
                eligible_jobs[
                    "eligible_nonmissing_share"
                ]
                .min()
            )

            median_nonmissing = float(
                eligible_jobs[
                    "eligible_nonmissing_share"
                ]
                .median()
            )

            varying_share = float(
                eligible_jobs[
                    "varying"
                ]
                .mean()
            )

            median_std = float(
                eligible_jobs[
                    "std"
                ]
                .median()
            )

            median_iqr = float(
                eligible_jobs[
                    "iqr"
                ]
                .median()
            )

            median_unique = float(
                eligible_jobs[
                    "unique_count"
                ]
                .median()
            )

        else:

            mean_nonmissing = np.nan

            min_nonmissing = np.nan

            median_nonmissing = np.nan

            varying_share = np.nan

            median_std = np.nan

            median_iqr = np.nan

            median_unique = np.nan

        rows.append(
            {

                "factor_name":
                    factor,

                "window":
                    int(
                        window
                    ),

                "family":
                    FACTOR_SPECS[
                        factor
                    ][
                        "family"
                    ],

                "tier":
                    FACTOR_SPECS[
                        factor
                    ][
                        "tier"
                    ],

                "job_count":
                    int(
                        len(
                            group
                        )
                    ),

                "eligible_job_count":
                    int(
                        len(
                            eligible_jobs
                        )
                    ),

                "mean_eligible_nonmissing_share":
                    mean_nonmissing,

                "median_eligible_nonmissing_share":
                    median_nonmissing,

                "min_eligible_nonmissing_share":
                    min_nonmissing,

                "varying_job_share":
                    varying_share,

                "median_monthly_std":
                    median_std,

                "median_monthly_iqr":
                    median_iqr,

                "median_monthly_unique_count":
                    median_unique,

                "total_range_violation_count":
                    int(
                        group[
                            "range_violation_count"
                        ]
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. Aggregate Temporal Stability
# ============================================================

def build_temporal_summary(
    temporal_monthly,
):

    if len(
        temporal_monthly
    ) == 0:

        return pd.DataFrame()

    rows = []

    for (
        factor,
        window
    ), group in temporal_monthly.groupby(
        [
            "factor_name",
            "window",
        ]
    ):

        rho = group[
            "temporal_spearman"
        ]

        rows.append(
            {

                "factor_name":
                    factor,

                "window":
                    int(
                        window
                    ),

                "transition_count":
                    int(
                        len(
                            group
                        )
                    ),

                "mean_temporal_spearman":
                    float(
                        rho.mean()
                    ),

                "median_temporal_spearman":
                    float(
                        rho.median()
                    ),

                "p10_temporal_spearman":
                    float(
                        rho.quantile(
                            0.10
                        )
                    ),

                "p90_temporal_spearman":
                    float(
                        rho.quantile(
                            0.90
                        )
                    ),

                "mean_common_obs":
                    float(
                        group[
                            "n_common"
                        ]
                        .mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. Aggregate Pairwise Correlations
# ============================================================

def build_pair_summary(
    pair_monthly,
):

    rows = []

    for (
        factor_a,
        factor_b,
        window
    ), group in pair_monthly.groupby(
        [
            "factor_a",
            "factor_b",
            "window",
        ]
    ):

        rho = (
            group[
                "spearman_rho"
            ]
        )

        abs_rho = (
            group[
                "abs_spearman_rho"
            ]
        )

        rows.append(
            {

                "factor_a":
                    factor_a,

                "factor_b":
                    factor_b,

                "window":
                    int(
                        window
                    ),

                "valid_month_count":
                    int(
                        len(
                            group
                        )
                    ),

                "mean_rho":
                    float(
                        rho.mean()
                    ),

                "median_rho":
                    float(
                        rho.median()
                    ),

                "mean_abs_rho":
                    float(
                        abs_rho.mean()
                    ),

                "median_abs_rho":
                    float(
                        abs_rho.median()
                    ),

                "p90_abs_rho":
                    float(
                        abs_rho.quantile(
                            0.90
                        )
                    ),

                "share_abs_rho_ge_090":
                    float(
                        (
                            abs_rho
                            >=
                            REDUNDANCY_MONTH_ABS_RHO
                        )
                        .mean()
                    ),

                "median_pair_obs":
                    float(
                        group[
                            "n_pair"
                        ]
                        .median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. Build Robust Redundancy Pairs
#
# Require the redundancy condition to hold in at least
# TWO of the three rolling-window specifications.
# ============================================================

def build_redundancy_pairs(
    pair_summary,
):

    if len(
        pair_summary
    ) == 0:

        return pd.DataFrame()

    temp = (
        pair_summary.copy()
    )

    temp[
        "window_redundant"
    ] = (

        (
            temp[
                "median_abs_rho"
            ]
            >=
            REDUNDANCY_MEDIAN_ABS_RHO
        )

        &

        (
            temp[
                "share_abs_rho_ge_090"
            ]
            >=
            REDUNDANCY_MONTH_SHARE_THRESHOLD
        )
    )

    rows = []

    for (
        factor_a,
        factor_b
    ), group in temp.groupby(
        [
            "factor_a",
            "factor_b",
        ]
    ):

        redundant_windows = (
            group.loc[
                group[
                    "window_redundant"
                ],
                "window"
            ]
            .astype(int)
            .tolist()
        )

        robust_redundant = (
            len(
                redundant_windows
            )
            >=
            2
        )

        if not robust_redundant:

            continue

        priority_a = (
            FACTOR_SPECS[
                factor_a
            ][
                "priority"
            ]
        )

        priority_b = (
            FACTOR_SPECS[
                factor_b
            ][
                "priority"
            ]
        )

        if (
            priority_a
            <
            priority_b
        ):

            preferred = factor_a

            secondary = factor_b

        elif (
            priority_b
            <
            priority_a
        ):

            preferred = factor_b

            secondary = factor_a

        else:

            preferred = ""

            secondary = ""

        rows.append(
            {

                "factor_a":
                    factor_a,

                "factor_b":
                    factor_b,

                "redundant_window_count":
                    int(
                        len(
                            redundant_windows
                        )
                    ),

                "redundant_windows":
                    "|".join(
                        str(x)
                        for x in redundant_windows
                    ),

                "mean_window_median_abs_rho":
                    float(
                        group[
                            "median_abs_rho"
                        ]
                        .mean()
                    ),

                "mean_window_mean_abs_rho":
                    float(
                        group[
                            "mean_abs_rho"
                        ]
                        .mean()
                    ),

                "preferred_representative":
                    preferred,

                "secondary_factor":
                    secondary,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "mean_window_median_abs_rho"
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
        if rows
        else
        pd.DataFrame()
    )


# ============================================================
# 19. Factor-level Screening Decisions
# ============================================================

def build_screening_decisions(
    qa_summary,
    temporal_summary,
    redundancy_pairs,
    frozen_factor_names,
):

    # --------------------------------------------------------
    # Aggregate QA across windows
    # --------------------------------------------------------

    factor_level = (
        qa_summary.groupby(
            "factor_name"
        )
        .agg(

            min_window_mean_nonmissing=(
                "mean_eligible_nonmissing_share",
                "min",
            ),

            min_window_min_nonmissing=(
                "min_eligible_nonmissing_share",
                "min",
            ),

            min_varying_job_share=(
                "varying_job_share",
                "min",
            ),

            max_range_violation_count=(
                "total_range_violation_count",
                "max",
            ),

            median_monthly_std=(
                "median_monthly_std",
                "median",
            ),

            median_monthly_iqr=(
                "median_monthly_iqr",
                "median",
            ),

        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Add temporal persistence
    # --------------------------------------------------------

    if len(
        temporal_summary
    ) > 0:

        temporal_factor = (
            temporal_summary.groupby(
                "factor_name"
            )
            .agg(

                median_temporal_spearman=(
                    "median_temporal_spearman",
                    "median",
                ),

                min_window_temporal_spearman=(
                    "median_temporal_spearman",
                    "min",
                ),

            )
            .reset_index()
        )

        factor_level = factor_level.merge(

            temporal_factor,

            on="factor_name",

            how="left",
        )

    else:

        factor_level[
            "median_temporal_spearman"
        ] = np.nan

        factor_level[
            "min_window_temporal_spearman"
        ] = np.nan

    # ========================================================
    # Build redundancy lookup
    # ========================================================

    redundant_with = {
        factor: []
        for factor in FACTOR_NAMES
    }

    if len(
        redundancy_pairs
    ) > 0:

        for row in redundancy_pairs.itertuples(
            index=False
        ):

            if (
                row.secondary_factor
                and
                row.preferred_representative
            ):

                redundant_with[
                    row.secondary_factor
                ].append(
                    row.preferred_representative
                )

    # ========================================================
    # Decisions
    # ========================================================

    rows = []

    for factor in FACTOR_NAMES:

        row = (
            factor_level[
                factor_level[
                    "factor_name"
                ]
                ==
                factor
            ]
        )

        if len(
            row
        ) != 1:

            raise RuntimeError(
                "Factor-level QA result missing for "
                f"{factor}"
            )

        row = row.iloc[
            0
        ]

        spec = (
            FACTOR_SPECS[
                factor
            ]
        )

        reasons = []

        # ----------------------------------------------------
        # Algebraic redundancy is deterministic.
        # ----------------------------------------------------

        if (
            factor
            ==
            "neighbor_turnover_1m"
        ):

            status = (
                "DROP_ALGEBRAIC_REDUNDANCY"
            )

            reasons.append(
                "neighbor_turnover_1m "
                "= 1 - neighbor_jaccard_1m"
            )

        elif (
            row[
                "max_range_violation_count"
            ]
            >
            0
        ):

            status = (
                "REVIEW_QA"
            )

            reasons.append(
                "range violation detected"
            )

        elif (
            row[
                "min_window_mean_nonmissing"
            ]
            <
            MIN_ELIGIBLE_NONMISSING_SHARE
        ):

            status = (
                "REVIEW_COVERAGE"
            )

            reasons.append(
                "eligible nonmissing share "
                "below threshold"
            )

        elif (
            row[
                "min_varying_job_share"
            ]
            <
            MIN_VARYING_JOB_SHARE
        ):

            status = (
                "REVIEW_LOW_VARIATION"
            )

            reasons.append(
                "insufficient cross-sectional variation"
            )

        elif redundant_with[
            factor
        ]:

            status = (
                "REVIEW_REDUNDANCY"
            )

            reasons.append(
                "highly correlated with: "
                +
                "|".join(
                    redundant_with[
                        factor
                    ]
                )
            )

        else:

            if (
                spec[
                    "tier"
                ]
                ==
                "core"
            ):

                status = (
                    "KEEP_CORE"
                )

            else:

                status = (
                    "KEEP_AUXILIARY"
                )

        rows.append(
            {

                "factor_name":
                    factor,

                "family":
                    spec[
                        "family"
                    ],

                "tier":
                    spec[
                        "tier"
                    ],

                "priority":
                    int(
                        spec[
                            "priority"
                        ]
                    ),

                "in_frozen_factor_dictionary":
                    (
                        factor
                        in
                        frozen_factor_names
                    ),

                "min_window_mean_nonmissing":
                    float(
                        row[
                            "min_window_mean_nonmissing"
                        ]
                    ),

                "min_window_min_nonmissing":
                    float(
                        row[
                            "min_window_min_nonmissing"
                        ]
                    ),

                "min_varying_job_share":
                    float(
                        row[
                            "min_varying_job_share"
                        ]
                    ),

                "max_range_violation_count":
                    int(
                        row[
                            "max_range_violation_count"
                        ]
                    ),

                "median_monthly_std":
                    float(
                        row[
                            "median_monthly_std"
                        ]
                    ),

                "median_monthly_iqr":
                    float(
                        row[
                            "median_monthly_iqr"
                        ]
                    ),

                "median_temporal_spearman":
                    (
                        float(
                            row[
                                "median_temporal_spearman"
                            ]
                        )
                        if pd.notna(
                            row[
                                "median_temporal_spearman"
                            ]
                        )
                        else np.nan
                    ),

                "redundant_with":
                    "|".join(
                        redundant_with[
                            factor
                        ]
                    ),

                "screening_status":
                    status,

                "screening_reason":
                    "; ".join(
                        reasons
                    ),

                "economic_meaning":
                    spec[
                        "meaning"
                    ],
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "priority"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 20. Metadata
# ============================================================

def save_metadata(
    design_hash,
    monthly_qa,
    qa_summary,
    temporal_summary,
    pair_monthly,
    pair_summary,
    redundancy_pairs,
    screening,
):

    payload = {

        "research_day":
            5,

        "step":
            "Step6_Factor_QA_and_Initial_Screening",

        "design_hash":
            design_hash,

        "input_panel":
            str(
                STEP5_PANEL_PATH
            ),

        "output_directory":
            str(
                OUTPUT_DIR
            ),

        "screening_is_preoutcome":
            True,

        "future_return_used":
            False,

        "future_risk_used":
            False,

        "windows":
            WINDOWS,

        "candidate_factor_count":
            int(
                len(
                    FACTOR_NAMES
                )
            ),

        "candidate_factors":
            FACTOR_NAMES,

        "screening_thresholds": {

            "minimum_eligible_nonmissing_share":
                MIN_ELIGIBLE_NONMISSING_SHARE,

            "minimum_varying_job_share":
                MIN_VARYING_JOB_SHARE,

            "minimum_pair_observations":
                MIN_PAIR_OBS,

            "minimum_temporal_observations":
                MIN_TEMPORAL_OBS,

            "redundancy_median_abs_spearman":
                REDUNDANCY_MEDIAN_ABS_RHO,

            "redundancy_month_abs_spearman":
                REDUNDANCY_MONTH_ABS_RHO,

            "redundancy_month_share":
                REDUNDANCY_MONTH_SHARE_THRESHOLD,
        },

        "methodology": {

            "distribution_qa":
                (
                    "Computed separately within each "
                    "Date x Window cross-section."
                ),

            "pairwise_correlation":
                (
                    "Monthly cross-sectional Spearman "
                    "correlations are computed first; "
                    "they are not estimated by pooling "
                    "all stock-month observations."
                ),

            "temporal_stability":
                (
                    "Adjacent-date Spearman correlation "
                    "on common stocks within each window."
                ),

            "redundancy":
                (
                    "A pair is flagged only when the "
                    "high-correlation condition holds "
                    "robustly in at least two windows."
                ),

            "screening":
                (
                    "Screening uses only data quality, "
                    "variation, persistence, redundancy, "
                    "and predefined economic priority. "
                    "No future return information is used."
                ),
        },

        "output_row_counts": {

            "factor_monthly_qa":
                int(
                    len(
                        monthly_qa
                    )
                ),

            "factor_qa_summary":
                int(
                    len(
                        qa_summary
                    )
                ),

            "factor_temporal_stability":
                int(
                    len(
                        temporal_summary
                    )
                ),

            "factor_pair_monthly_spearman":
                int(
                    len(
                        pair_monthly
                    )
                ),

            "factor_pair_correlation_summary":
                int(
                    len(
                        pair_summary
                    )
                ),

            "factor_redundancy_pairs":
                int(
                    len(
                        redundancy_pairs
                    )
                ),

            "factor_screening_decisions":
                int(
                    len(
                        screening
                    )
                ),
        },
    }

    save_json(
        payload,
        METADATA_PATH,
    )


# ============================================================
# 21. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 5 - Step 6"
    )

    print(
        "Factor QA & Initial Screening"
    )

    print("=" * 80)

    # ========================================================
    # 21.1 Design / lineage
    # ========================================================

    (
        config,
        design_hash,
    ) = load_frozen_design()

    validate_step5_lineage(
        design_hash
    )

    frozen_factor_names = (
        load_factor_dictionary()
    )

    analysis_dates = (
        load_analysis_dates()
    )

    print()
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

    print(
        f"Candidate factors: "
        f"{len(FACTOR_NAMES):,}"
    )

    # ========================================================
    # 21.2 Containers
    # ========================================================

    monthly_qa_rows = []

    pair_monthly_rows = []

    temporal_rows = []

    previous_by_window = {}

    previous_date_by_window = {}

    # ========================================================
    # 21.3 Main Date × Window loop
    # ========================================================

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

        print(
            f"[{job_no}/{total_jobs}] "
            f"W={window} | "
            f"{date.date()}"
        )

        df = load_partition(
            window=
                window,

            date=
                date,
        )

        # ====================================================
        # Monthly factor QA
        # ====================================================

        for factor in FACTOR_NAMES:

            monthly_qa_rows.append(

                monthly_factor_qa(

                    df=
                        df,

                    factor=
                        factor,

                    window=
                        window,

                    date=
                        date,
                )
            )

        # ====================================================
        # Monthly cross-sectional factor correlations
        # ====================================================

        pair_monthly_rows.extend(

            monthly_pairwise_correlations(

                df=
                    df,

                window=
                    window,

                date=
                    date,
            )
        )

        # ====================================================
        # Adjacent-period temporal stability
        # ====================================================

        previous = (
            previous_by_window.get(
                window
            )
        )

        previous_date = (
            previous_date_by_window.get(
                window
            )
        )

        temporal_rows.extend(

            temporal_stability(

                current=
                    df,

                previous=
                    previous,

                window=
                    window,

                date=
                    date,

                previous_date=
                    previous_date,
            )
        )

        previous_by_window[
            window
        ] = df

        previous_date_by_window[
            window
        ] = date

    # ========================================================
    # 21.4 Monthly QA table
    # ========================================================

    monthly_qa = (
        pd.DataFrame(
            monthly_qa_rows
        )
        .sort_values(
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

    monthly_qa.to_csv(

        MONTHLY_QA_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.5 QA summary
    # ========================================================

    qa_summary = (
        build_qa_summary(
            monthly_qa
        )
    )

    qa_summary.to_csv(

        QA_SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.6 Temporal stability
    # ========================================================

    temporal_monthly = (
        pd.DataFrame(
            temporal_rows
        )
    )

    temporal_summary = (
        build_temporal_summary(
            temporal_monthly
        )
    )

    temporal_summary.to_csv(

        TEMPORAL_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.7 Monthly pairwise correlations
    # ========================================================

    pair_monthly = (
        pd.DataFrame(
            pair_monthly_rows
        )
    )

    pair_monthly.to_csv(

        PAIR_MONTHLY_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.8 Pairwise correlation summary
    # ========================================================

    pair_summary = (
        build_pair_summary(
            pair_monthly
        )
    )

    pair_summary.to_csv(

        PAIR_SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.9 Redundancy detection
    # ========================================================

    redundancy_pairs = (
        build_redundancy_pairs(
            pair_summary
        )
    )

    redundancy_pairs.to_csv(

        REDUNDANCY_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.10 Screening decisions
    # ========================================================

    screening = (
        build_screening_decisions(

            qa_summary=
                qa_summary,

            temporal_summary=
                temporal_summary,

            redundancy_pairs=
                redundancy_pairs,

            frozen_factor_names=
                frozen_factor_names,
        )
    )

    screening.to_csv(

        SCREENING_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.11 Recommended pre-outcome core factor set
    # ========================================================

    recommended_core = (
        screening[
            screening[
                "screening_status"
            ]
            ==
            "KEEP_CORE"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    recommended_core.to_csv(

        CORE_SET_PATH,

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 21.12 Metadata
    # ========================================================

    save_metadata(

        design_hash=
            design_hash,

        monthly_qa=
            monthly_qa,

        qa_summary=
            qa_summary,

        temporal_summary=
            temporal_summary,

        pair_monthly=
            pair_monthly,

        pair_summary=
            pair_summary,

        redundancy_pairs=
            redundancy_pairs,

        screening=
            screening,
    )

    # ========================================================
    # 21.13 Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Factor QA & Initial Screening Summary"
    )

    print("=" * 80)

    print(
        f"Candidate factors: "
        f"{len(screening)}"
    )

    print()

    print(
        screening[
            [
                "screening_status"
            ]
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Robust redundancy pairs:"
    )

    print(
        len(
            redundancy_pairs
        )
    )

    print()

    print(
        "Recommended pre-outcome core set:"
    )

    if len(
        recommended_core
    ) > 0:

        print(
            recommended_core[
                [
                    "factor_name",
                    "family",
                ]
            ]
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No factor currently satisfies KEEP_CORE."
        )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print("=" * 80)

    print(
        "Day 5 Step 6 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)


# ------------------------------------------------------------
# Previous research day:
#
# Day 4 research outputs were stored under M1_day3.
# ------------------------------------------------------------

PREVIOUS_DAY_ROOT = (
    OUTPUT_ROOT
    / "M1_day4"
)


# ------------------------------------------------------------
# Step 2:
# Formal matched-density networks
# ------------------------------------------------------------

STEP2_DIR = (
    PREVIOUS_DAY_ROOT
    / "02_step2_matched_networks"
)

MATCHED_SUMMARY_PATH = (
    STEP2_DIR
    / "matched_density_network_summary.csv"
)

STEP2_METADATA_PATH = (
    STEP2_DIR
    / "step2_network_construction_metadata.json"
)

EDGE_ROOT = (
    STEP2_DIR
    / "edge_master"
)

NODE_ROOT = (
    STEP2_DIR
    / "node_universe"
)


# ------------------------------------------------------------
# Step 3:
# Node-level structural metrics
# ------------------------------------------------------------

STEP3_DIR = (
    PREVIOUS_DAY_ROOT
    / "03_step3_structural_diagnostics"
)

NODE_METRIC_ROOT = (
    STEP3_DIR
    / "node_metrics"
)


# ------------------------------------------------------------
# Step 4:
# Community memberships
# ------------------------------------------------------------

STEP4_DIR = (
    PREVIOUS_DAY_ROOT
    / "04_step4_industry_community_validation"
)

COMMUNITY_ROOT = (
    STEP4_DIR
    / "community_membership"
)


# ------------------------------------------------------------
# Step 5:
# Dynamic stability
#
# This directory is mainly used as a validation dependency.
# Node-level dynamic factors will later be reconstructed
# directly from adjacent node/edge panels.
# ------------------------------------------------------------

STEP5_DIR = (
    PREVIOUS_DAY_ROOT
    / "05_step5_dynamic_stability"
)


# ------------------------------------------------------------
# Day 5 Step 1 output
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day5"
    / "01_step1_factor_design"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Frozen Research Design
# ============================================================

PRIMARY_NETWORK = (
    "M1_RESIDUAL_POSITIVE"
)

BENCHMARK_NETWORK = (
    "B0_RAW_POSITIVE"
)


WINDOWS = [
    60,
    120,
    252,
]


PRIMARY_DENSITY = 0.01


ROBUSTNESS_DENSITIES = [
    0.002,
    0.005,
    0.010,
    0.020,
]


MAIN_INDUSTRY_LEVEL = (
    "industry_id1"
)

ROBUSTNESS_INDUSTRY_LEVEL = (
    "industry_id2"
)


# ------------------------------------------------------------
# Factor construction timing
# ------------------------------------------------------------

FEATURE_FREQUENCY = (
    "month_end_network_date"
)

FEATURE_INFORMATION_CUTOFF = (
    "analysis_date_close"
)


# ------------------------------------------------------------
# Future outcome convention
#
# Features at t:
#   use data through analysis_date t only.
#
# Future target:
#   starts strictly after t and ends at the next analysis date.
#
# This is a predictive-label definition, not yet a detailed
# transaction-price/backtest convention.
# ------------------------------------------------------------

OUTCOME_START_RULE = (
    "first_trading_day_strictly_after_analysis_date"
)

OUTCOME_END_RULE = (
    "next_analysis_date"
)


# ------------------------------------------------------------
# Missing-value / universe principles
# ------------------------------------------------------------

DYNAMIC_FIRST_DATE_POLICY = (
    "missing_not_zero"
)

FUTURE_OUTCOME_FILTER_POLICY = (
    "never_use_future_outcome_availability_to_define_feature_universe"
)


# ------------------------------------------------------------
# Strict input validation
# ------------------------------------------------------------

STRICT_INPUT_CHECK = True


NETWORK_IDS = [
    BENCHMARK_NETWORK,
    PRIMARY_NETWORK,
]


# ============================================================
# 2. Utility Functions
# ============================================================

def save_json(
    obj,
    path: Path,
) -> None:

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


def canonical_json_hash(
    obj,
) -> str:

    text = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# 3. Load Step 2 Metadata
# ============================================================

def load_step2_metadata():

    if not STEP2_METADATA_PATH.exists():

        raise FileNotFoundError(
            "Step 2 metadata not found:\n"
            f"{STEP2_METADATA_PATH}"
        )

    with open(
        STEP2_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# 4. Load Matched-density Summary
# ============================================================

def load_matched_summary():

    if not MATCHED_SUMMARY_PATH.exists():

        raise FileNotFoundError(
            "Matched-density summary not found:\n"
            f"{MATCHED_SUMMARY_PATH}"
        )

    df = pd.read_csv(
        MATCHED_SUMMARY_PATH
    )

    required = [
        "analysis_date",
        "window",
        "density_fraction",
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
            "matched_density_network_summary.csv "
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

    df[
        "density_fraction"
    ] = pd.to_numeric(
        df[
            "density_fraction"
        ],
        errors="raise",
    )

    return (
        df.sort_values(
            [
                "window",
                "density_fraction",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 5. Validate Frozen Network Design
# ============================================================

def validate_network_design(
    matched_summary,
    metadata,
):

    errors = []
    warnings = []

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    available_windows = sorted(
        matched_summary[
            "window"
        ]
        .unique()
        .tolist()
    )

    for window in WINDOWS:

        if window not in available_windows:

            errors.append(
                f"W={window} absent from Step 2."
            )

    # --------------------------------------------------------
    # Primary q
    # --------------------------------------------------------

    for window in WINDOWS:

        subset = matched_summary[
            matched_summary[
                "window"
            ]
            ==
            window
        ]

        if not np.any(
            np.isclose(
                subset[
                    "density_fraction"
                ],
                PRIMARY_DENSITY,
            )
        ):

            errors.append(
                f"W={window}: primary q="
                f"{PRIMARY_DENSITY} missing."
            )

    # --------------------------------------------------------
    # Robustness q-grid
    # --------------------------------------------------------

    for window in WINDOWS:

        subset = matched_summary[
            matched_summary[
                "window"
            ]
            ==
            window
        ]

        actual = subset[
            "density_fraction"
        ].to_numpy()

        for q in ROBUSTNESS_DENSITIES:

            if not np.any(
                np.isclose(
                    actual,
                    q,
                )
            ):

                warnings.append(
                    f"W={window}: robustness q="
                    f"{q} unavailable."
                )

    # --------------------------------------------------------
    # Step 2 metadata consistency
    # --------------------------------------------------------

    if (
        "primary_density"
        in metadata
    ):

        metadata_q = float(
            metadata[
                "primary_density"
            ]
        )

        if not np.isclose(
            metadata_q,
            PRIMARY_DENSITY,
        ):

            errors.append(
                "Step 2 metadata primary_density "
                f"={metadata_q}, but Day 5 design "
                f"freezes q={PRIMARY_DENSITY}."
            )

    return (
        errors,
        warnings,
    )


# ============================================================
# 6. Build Analysis-date Table
# ============================================================

def build_analysis_dates(
    matched_summary,
):

    primary = (
        matched_summary[
            np.isclose(
                matched_summary[
                    "density_fraction"
                ],
                PRIMARY_DENSITY,
            )
        ]
        [
            [
                "analysis_date",
                "window",
                "common_node_count",
                "edge_budget_k",
            ]
        ]
        .copy()
    )

    primary = (
        primary.sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    primary[
        "previous_analysis_date"
    ] = (
        primary.groupby(
            "window"
        )[
            "analysis_date"
        ]
        .shift(1)
    )

    primary[
        "next_analysis_date"
    ] = (
        primary.groupby(
            "window"
        )[
            "analysis_date"
        ]
        .shift(-1)
    )

    primary[
        "has_previous_analysis_date"
    ] = (
        primary[
            "previous_analysis_date"
        ]
        .notna()
    )

    primary[
        "has_next_analysis_date"
    ] = (
        primary[
            "next_analysis_date"
        ]
        .notna()
    )

    # --------------------------------------------------------
    # Dynamic factor eligibility
    #
    # Dynamic factor requires t-1.
    # --------------------------------------------------------

    primary[
        "dynamic_factor_eligible"
    ] = (
        primary[
            "has_previous_analysis_date"
        ]
    )

    # --------------------------------------------------------
    # Forward label eligibility
    #
    # Alpha/Risk future outcome requires t+1.
    # --------------------------------------------------------

    primary[
        "forward_label_eligible"
    ] = (
        primary[
            "has_next_analysis_date"
        ]
    )

    primary[
        "days_since_previous_network"
    ] = (

        primary[
            "analysis_date"
        ]

        -

        primary[
            "previous_analysis_date"
        ]

    ).dt.days

    primary[
        "days_to_next_network"
    ] = (

        primary[
            "next_analysis_date"
        ]

        -

        primary[
            "analysis_date"
        ]

    ).dt.days

    return primary


# ============================================================
# 7. Input File Inventory
# ============================================================

def build_input_inventory(
    analysis_dates,
):

    rows = []

    for row in analysis_dates.itertuples(
        index=False
    ):

        date = pd.Timestamp(
            row.analysis_date
        )

        window = int(
            row.window
        )

        date_string = (
            date.strftime(
                "%Y-%m-%d"
            )
        )

        paths = {

            "step2_node_universe":
                (
                    NODE_ROOT
                    /
                    f"W{window}"
                    /
                    f"{date_string}.parquet"
                ),

            "step2_raw_edge_master":
                (
                    EDGE_ROOT
                    /
                    BENCHMARK_NETWORK
                    /
                    f"W{window}"
                    /
                    f"{date_string}.parquet"
                ),

            "step2_residual_edge_master":
                (
                    EDGE_ROOT
                    /
                    PRIMARY_NETWORK
                    /
                    f"W{window}"
                    /
                    f"{date_string}.parquet"
                ),

            "step3_raw_node_metrics":
                (
                    NODE_METRIC_ROOT
                    /
                    BENCHMARK_NETWORK
                    /
                    f"W{window}"
                    /
                    f"{date_string}.parquet"
                ),

            "step3_residual_node_metrics":
                (
                    NODE_METRIC_ROOT
                    /
                    PRIMARY_NETWORK
                    /
                    f"W{window}"
                    /
                    f"{date_string}.parquet"
                ),

            "step4_raw_community":
                (
                    COMMUNITY_ROOT
                    /
                    BENCHMARK_NETWORK
                    /
                    f"W{window}"
                    /
                    f"{date_string}.parquet"
                ),

            "step4_residual_community":
                (
                    COMMUNITY_ROOT
                    /
                    PRIMARY_NETWORK
                    /
                    f"W{window}"
                    /
                    f"{date_string}.parquet"
                ),
        }

        result = {

            "analysis_date":
                date,

            "window":
                window,
        }

        all_present = True

        for name, path in (
            paths.items()
        ):

            exists = (
                path.exists()
            )

            result[
                f"has_{name}"
            ] = bool(
                exists
            )

            result[
                f"path_{name}"
            ] = str(
                path
            )

            all_present &= exists

        result[
            "all_required_sources_present"
        ] = bool(
            all_present
        )

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 8. Inventory Summary
# ============================================================

def build_inventory_summary(
    inventory,
):

    source_columns = [
        col
        for col in inventory.columns
        if col.startswith("has_")
    ]

    rows = []

    for window, group in (
        inventory.groupby(
            "window"
        )
    ):

        row = {

            "window":
                int(window),

            "analysis_date_count":
                int(
                    len(group)
                ),

            "all_sources_complete_dates":
                int(
                    group[
                        "all_required_sources_present"
                    ]
                    .sum()
                ),

            "all_sources_complete_share":
                float(
                    group[
                        "all_required_sources_present"
                    ]
                    .mean()
                ),
        }

        for column in source_columns:

            row[
                f"{column}_count"
            ] = int(
                group[
                    column
                ]
                .sum()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 9. Factor Dictionary
# ============================================================

def build_factor_dictionary():

    rows = [

        # ====================================================
        # A. Basic residual centrality
        # ====================================================

        {
            "factor_name":
                "residual_degree",

            "family":
                "centrality",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                "d_i^R = sum_j A_ij^R",

            "required_source":
                "Step3 residual node_metrics",

            "economic_interpretation":
                (
                    "Number of strongest residual "
                    "association links."
                ),

            "default_transform":
                "raw_and_cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "residual_strength",

            "family":
                "centrality",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                "s_i^R = sum_j rho_ij^R A_ij^R",

            "required_source":
                "Step3 residual node_metrics",

            "economic_interpretation":
                (
                    "Total intensity of strongest "
                    "residual associations."
                ),

            "default_transform":
                "raw_and_cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "residual_degree_percentile",

            "family":
                "centrality",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                "cross-sectional percentile of residual_degree",

            "required_source":
                "Step3 residual node_metrics",

            "economic_interpretation":
                "Relative residual-network centrality.",

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "residual_strength_percentile",

            "family":
                "centrality",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                "cross-sectional percentile of residual_strength",

            "required_source":
                "Step3 residual node_metrics",

            "economic_interpretation":
                "Relative weighted residual centrality.",

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        # ====================================================
        # B. Raw vs Residual differential
        # ====================================================

        {
            "factor_name":
                "delta_degree_percentile",

            "family":
                "raw_residual_differential",

            "core_factor":
                True,

            "source_network":
                "Raw_and_Residual",

            "formula":
                (
                    "DegreePct_i^Residual "
                    "- DegreePct_i^Raw"
                ),

            "required_source":
                "Step3 raw + residual node_metrics",

            "economic_interpretation":
                (
                    "Change in relative centrality after "
                    "removing the market mode."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "delta_strength_percentile",

            "family":
                "raw_residual_differential",

            "core_factor":
                True,

            "source_network":
                "Raw_and_Residual",

            "formula":
                (
                    "StrengthPct_i^Residual "
                    "- StrengthPct_i^Raw"
                ),

            "required_source":
                "Step3 raw + residual node_metrics",

            "economic_interpretation":
                (
                    "Weighted-centrality re-ranking after "
                    "market residualization."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        # ====================================================
        # C. Industry backbone decomposition
        # ====================================================

        {
            "factor_name":
                "same_industry_degree",

            "family":
                "industry_decomposition",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "sum_j A_ij^R "
                    "1{Industry_i=Industry_j}"
                ),

            "required_source":
                "Step2 residual edges + PIT industry_id1",

            "economic_interpretation":
                "Industry-backbone connectivity.",

            "default_transform":
                "raw_and_cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "cross_industry_degree",

            "family":
                "industry_decomposition",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "sum_j A_ij^R "
                    "1{Industry_i!=Industry_j}"
                ),

            "required_source":
                "Step2 residual edges + PIT industry_id1",

            "economic_interpretation":
                (
                    "Cross-industry network connectivity."
                ),

            "default_transform":
                "raw_and_cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "same_industry_strength",

            "family":
                "industry_decomposition",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "sum_j rho_ij^R A_ij^R "
                    "1{Industry_i=Industry_j}"
                ),

            "required_source":
                "Step2 residual edges + PIT industry_id1",

            "economic_interpretation":
                "Weighted industry-backbone exposure.",

            "default_transform":
                "raw_and_cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "cross_industry_strength",

            "family":
                "industry_decomposition",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "sum_j rho_ij^R A_ij^R "
                    "1{Industry_i!=Industry_j}"
                ),

            "required_source":
                "Step2 residual edges + PIT industry_id1",

            "economic_interpretation":
                (
                    "Intensity of cross-industry "
                    "residual associations."
                ),

            "default_transform":
                "raw_and_cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "cross_industry_degree_ratio",

            "family":
                "industry_decomposition",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "CrossDegree_i / Degree_i "
                    "for Degree_i>0"
                ),

            "required_source":
                "derived",

            "economic_interpretation":
                (
                    "Fraction of connectivity extending "
                    "outside the stock's own industry."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "cross_industry_strength_ratio",

            "family":
                "industry_decomposition",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "CrossStrength_i / Strength_i "
                    "for Strength_i>0"
                ),

            "required_source":
                "derived",

            "economic_interpretation":
                (
                    "Fraction of weighted association "
                    "outside own industry."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        # ====================================================
        # D. Dynamic features
        # ====================================================

        {
            "factor_name":
                "delta_residual_degree_percentile_1m",

            "family":
                "dynamic",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "DegreePct_i,t^R "
                    "- DegreePct_i,t-1^R"
                ),

            "required_source":
                "adjacent Step3 residual node_metrics",

            "economic_interpretation":
                (
                    "Recent change in residual-network "
                    "centrality."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "delta_residual_strength_percentile_1m",

            "family":
                "dynamic",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "StrengthPct_i,t^R "
                    "- StrengthPct_i,t-1^R"
                ),

            "required_source":
                "adjacent Step3 residual node_metrics",

            "economic_interpretation":
                (
                    "Recent change in weighted "
                    "residual centrality."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "neighbor_jaccard_1m",

            "family":
                "dynamic",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "|N_i,t intersect N_i,t-1| "
                    "/ |N_i,t union N_i,t-1|"
                ),

            "required_source":
                "adjacent Step2 residual edge masters",

            "economic_interpretation":
                (
                    "Stability of the stock's "
                    "network neighborhood."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "neighbor_retention_1m",

            "family":
                "dynamic",

            "core_factor":
                False,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "|N_i,t intersect N_i,t-1| "
                    "/ |N_i,t-1|"
                ),

            "required_source":
                "adjacent Step2 residual edge masters",

            "economic_interpretation":
                (
                    "Fraction of previous neighbors "
                    "retained."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },

        # ====================================================
        # E. Community / bridge features
        # ====================================================

        {
            "factor_name":
                "community_size",

            "family":
                "community",

            "core_factor":
                False,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "number of active nodes assigned "
                    "to stock i's Louvain community"
                ),

            "required_source":
                "Step4 residual community membership",

            "economic_interpretation":
                (
                    "Size of endogenous network group."
                ),

            "default_transform":
                "cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "within_community_degree",

            "family":
                "community",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "sum_j A_ij^R "
                    "1{C_i=C_j}"
                ),

            "required_source":
                (
                    "Step2 residual edges + "
                    "Step4 community membership"
                ),

            "economic_interpretation":
                (
                    "Connectivity inside endogenous "
                    "community."
                ),

            "default_transform":
                "raw_and_cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "outside_community_degree",

            "family":
                "community",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "sum_j A_ij^R "
                    "1{C_i!=C_j}"
                ),

            "required_source":
                (
                    "Step2 residual edges + "
                    "Step4 community membership"
                ),

            "economic_interpretation":
                (
                    "Cross-community bridge connectivity."
                ),

            "default_transform":
                "raw_and_cross_sectional_percentile",

            "future_role":
                "alpha_and_risk",
        },

        {
            "factor_name":
                "participation_coefficient",

            "family":
                "community",

            "core_factor":
                True,

            "source_network":
                PRIMARY_NETWORK,

            "formula":
                (
                    "P_i = 1 - sum_c "
                    "(k_ic / k_i)^2"
                ),

            "required_source":
                (
                    "Step2 residual edges + "
                    "Step4 community membership"
                ),

            "economic_interpretation":
                (
                    "Extent to which a stock connects "
                    "multiple endogenous communities."
                ),

            "default_transform":
                "none",

            "future_role":
                "alpha_and_risk",
        },
    ]

    df = pd.DataFrame(
        rows
    )

    df.insert(
        0,
        "factor_id",
        np.arange(
            1,
            len(df) + 1,
        ),
    )

    return df


# ============================================================
# 10. Future Outcome Dictionary
# ============================================================

def build_target_dictionary():

    rows = [

        {
            "target_name":
                "next_period_return",

            "research_role":
                "alpha",

            "start_rule":
                OUTCOME_START_RULE,

            "end_rule":
                OUTCOME_END_RULE,

            "conceptual_formula":
                (
                    "product_{s in (t,t_next]} "
                    "(1+r_i,s) - 1"
                ),

            "status":
                "planned_for_later_step",

            "note":
                (
                    "Predictive label only; detailed "
                    "portfolio execution convention will "
                    "be frozen before backtesting."
                ),
        },

        {
            "target_name":
                "next_period_realized_volatility",

            "research_role":
                "risk",

            "start_rule":
                OUTCOME_START_RULE,

            "end_rule":
                OUTCOME_END_RULE,

            "conceptual_formula":
                (
                    "sqrt(sum future daily returns^2)"
                ),

            "status":
                "planned_for_later_step",

            "note":
                (
                    "Primary future-risk candidate."
                ),
        },

        {
            "target_name":
                "next_period_downside_volatility",

            "research_role":
                "risk",

            "start_rule":
                OUTCOME_START_RULE,

            "end_rule":
                OUTCOME_END_RULE,

            "conceptual_formula":
                (
                    "sqrt(sum min(r_i,s,0)^2)"
                ),

            "status":
                "planned_for_later_step",

            "note":
                (
                    "Downside-risk candidate."
                ),
        },

        {
            "target_name":
                "next_period_max_drawdown",

            "research_role":
                "risk",

            "start_rule":
                OUTCOME_START_RULE,

            "end_rule":
                OUTCOME_END_RULE,

            "conceptual_formula":
                (
                    "maximum peak-to-trough drawdown "
                    "over next network period"
                ),

            "status":
                "planned_for_later_step",

            "note":
                (
                    "Tail/downside-risk candidate."
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. Freeze Design Config
# ============================================================

def build_design_config(
    analysis_dates,
):

    config = {

        "research_day":
            5,

        "research_theme":
            (
                "Network Feature Engineering "
                "and Factor Panel Construction"
            ),

        "step":
            "Step1_Freeze_Factor_Design",

        "created_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        # ====================================================
        # Network definition
        # ====================================================

        "network_design": {

            "primary_network":
                PRIMARY_NETWORK,

            "benchmark_network":
                BENCHMARK_NETWORK,

            "primary_density":
                PRIMARY_DENSITY,

            "robustness_densities":
                ROBUSTNESS_DENSITIES,

            "windows":
                WINDOWS,

            "positive_edges_only":
                True,

            "main_industry_level":
                MAIN_INDUSTRY_LEVEL,

            "robustness_industry_level":
                ROBUSTNESS_INDUSTRY_LEVEL,
        },

        # ====================================================
        # Timing
        # ====================================================

        "timing_design": {

            "feature_frequency":
                FEATURE_FREQUENCY,

            "feature_information_cutoff":
                FEATURE_INFORMATION_CUTOFF,

            "outcome_start_rule":
                OUTCOME_START_RULE,

            "outcome_end_rule":
                OUTCOME_END_RULE,

            "dynamic_first_date_policy":
                DYNAMIC_FIRST_DATE_POLICY,

            "future_outcome_filter_policy":
                FUTURE_OUTCOME_FILTER_POLICY,
        },

        # ====================================================
        # Panel identifiers
        # ====================================================

        "panel_key": [

            "analysis_date",

            "window",

            "master_index",
        ],

        # ====================================================
        # Factor families
        # ====================================================

        "factor_families": [

            "centrality",

            "raw_residual_differential",

            "industry_decomposition",

            "dynamic",

            "community",
        ],

        # ====================================================
        # Preprocessing
        # ====================================================

        "preprocessing": {

            "store_raw_values":
                True,

            "store_cross_sectional_percentiles":
                True,

            "fill_missing_dynamic_factor_with_zero":
                False,

            "winsorization_at_panel_construction":
                False,

            "industry_neutralization_at_panel_construction":
                False,

            "size_neutralization_at_panel_construction":
                False,

            "neutralization_stage":
                (
                    "later_factor_diagnostics_and_validation"
                ),
        },

        # ====================================================
        # Look-ahead protection
        # ====================================================

        "lookahead_controls": {

            "features_may_use_information_after_analysis_date":
                False,

            "future_outcomes_may_define_current_universe":
                False,

            "industry_definition":
                "historically_indexed_vendor_classification",

            "current_network_universe_rule":
                (
                    "use contemporaneous Step2 "
                    "node universe only"
                ),
        },

        # ====================================================
        # Research interpretation
        # ====================================================

        "research_principles": {

            "primary_goal":
                (
                    "measure whether validated network "
                    "structure contains useful stock-level "
                    "information"
                ),

            "do_not_optimize_factor_sign_in_step1":
                True,

            "do_not_select_best_window_in_step1":
                True,

            "do_not_select_best_density_using_future_returns":
                True,

            "raw_network_role":
                "benchmark_control",

            "residual_network_role":
                "primary_information_network",
        },

        # ====================================================
        # Date coverage
        # ====================================================

        "date_coverage": {

            str(window): {

                "first_date":
                    (
                        analysis_dates[
                            analysis_dates[
                                "window"
                            ]
                            ==
                            window
                        ][
                            "analysis_date"
                        ]
                        .min()
                        .strftime(
                            "%Y-%m-%d"
                        )
                    ),

                "last_date":
                    (
                        analysis_dates[
                            analysis_dates[
                                "window"
                            ]
                            ==
                            window
                        ][
                            "analysis_date"
                        ]
                        .max()
                        .strftime(
                            "%Y-%m-%d"
                        )
                    ),

                "network_date_count":
                    int(
                        (
                            analysis_dates[
                                "window"
                            ]
                            ==
                            window
                        )
                        .sum()
                    ),
            }

            for window in WINDOWS
        },

        # ====================================================
        # Source locations
        # ====================================================

        "source_directories": {

            "step2":
                str(
                    STEP2_DIR
                ),

            "step3":
                str(
                    STEP3_DIR
                ),

            "step4":
                str(
                    STEP4_DIR
                ),

            "step5":
                str(
                    STEP5_DIR
                ),
        },
    }

    return config


# ============================================================
# 12. Human-readable Design Summary
# ============================================================

def write_design_summary(
    config,
    factor_dictionary,
    inventory_summary,
):

    core_count = int(
        factor_dictionary[
            "core_factor"
        ]
        .sum()
    )

    total_count = len(
        factor_dictionary
    )

    text = f"""# Day 5 Step 1 — Frozen Network Factor Design

## Research objective

Convert the validated full-market stock network into a reproducible stock-level factor panel for subsequent Alpha and Risk research.

## Primary network

- Primary network: `{PRIMARY_NETWORK}`
- Benchmark network: `{BENCHMARK_NETWORK}`
- Primary density: `{100*PRIMARY_DENSITY:.1f}%`
- Robustness density grid: `{[100*q for q in ROBUSTNESS_DENSITIES]}%`
- Windows: `{WINDOWS}`
- Main industry definition: `{MAIN_INDUSTRY_LEVEL}`

## Timing rule

All factor values at analysis date `t` may use information available through `t` only.

Future outcomes start strictly after `t` and end at the next analysis date.

Therefore the empirical relation will follow:

\[
X_{{i,t}} \\rightarrow Y_{{i,t+1}}
\]

rather than contemporaneous feature-outcome matching.

## Factor families

1. Centrality
2. Raw–Residual differential
3. Industry backbone / cross-industry decomposition
4. Dynamic network features
5. Community / bridge features

Total predefined candidate factors: **{total_count}**

Core factors: **{core_count}**

## Important restrictions

- No factor sign is selected using future returns.
- No best window is selected in Step 1.
- No best density is selected using future outcomes.
- Missing lagged features are kept missing, not filled with zero.
- Current stock universe is never defined using future return availability.
- Industry and size neutralization are deferred to formal factor diagnostics.
- The Residual network remains an association network, not a conditional-independence network.

## Input coverage

{inventory_summary.to_markdown(index=False)}

## Design hash

`{config["design_hash"]}`

This hash should be stored with later factor-panel outputs to confirm that all subsequent analyses use the same frozen design.
"""

    path = (
        OUTPUT_DIR
        /
        "factor_design_summary.md"
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# 13. Main
# ============================================================

def main():

    print("=" * 80)
    print(
        "M1 A-Share Full-Market Stock Network"
    )
    print(
        "Day 5 - Step 1"
    )
    print(
        "Freeze Factor Design"
    )
    print("=" * 80)

    # ========================================================
    # 1. Load previous outputs
    # ========================================================

    print()
    print(
        "[1] Load previous network outputs"
    )

    metadata = (
        load_step2_metadata()
    )

    matched_summary = (
        load_matched_summary()
    )

    # ========================================================
    # 2. Validate design
    # ========================================================

    print()
    print(
        "[2] Validate frozen network specifications"
    )

    errors, warnings = (
        validate_network_design(
            matched_summary,
            metadata,
        )
    )

    for message in warnings:

        print(
            f"[WARNING] {message}"
        )

    if errors:

        for message in errors:

            print(
                f"[ERROR] {message}"
            )

        raise RuntimeError(
            "Frozen factor design validation failed."
        )

    # ========================================================
    # 3. Analysis dates
    # ========================================================

    print()
    print(
        "[3] Freeze network analysis dates"
    )

    analysis_dates = (
        build_analysis_dates(
            matched_summary
        )
    )

    analysis_dates.to_csv(

        OUTPUT_DIR
        /
        "factor_analysis_dates.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 4. Input inventory
    # ========================================================

    print()
    print(
        "[4] Validate required source files"
    )

    inventory = (
        build_input_inventory(
            analysis_dates
        )
    )

    inventory.to_csv(

        OUTPUT_DIR
        /
        "factor_input_inventory.csv",

        index=False,

        encoding="utf-8-sig",
    )

    inventory_summary = (
        build_inventory_summary(
            inventory
        )
    )

    inventory_summary.to_csv(

        OUTPUT_DIR
        /
        "factor_input_inventory_summary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    missing_inventory = (
        inventory[
            ~inventory[
                "all_required_sources_present"
            ]
        ]
    )

    if not missing_inventory.empty:

        print()

        print(
            f"[WARNING] "
            f"{len(missing_inventory):,} "
            f"date-window jobs have missing inputs."
        )

        if STRICT_INPUT_CHECK:

            raise RuntimeError(
                "Missing Step2/3/4 factor inputs. "
                "See factor_input_inventory.csv."
            )

    # ========================================================
    # 5. Factor dictionary
    # ========================================================

    print()
    print(
        "[5] Freeze factor dictionary"
    )

    factor_dictionary = (
        build_factor_dictionary()
    )

    factor_dictionary.to_csv(

        OUTPUT_DIR
        /
        "network_factor_dictionary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 6. Future target dictionary
    # ========================================================

    target_dictionary = (
        build_target_dictionary()
    )

    target_dictionary.to_csv(

        OUTPUT_DIR
        /
        "future_target_dictionary.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # ========================================================
    # 7. Freeze formal config
    # ========================================================

    print()
    print(
        "[6] Freeze formal factor design config"
    )

    config = (
        build_design_config(
            analysis_dates
        )
    )

    # --------------------------------------------------------
    # Hash the design before adding the hash itself.
    # --------------------------------------------------------

    design_hash = (
        canonical_json_hash(
            config
        )
    )

    config[
        "design_hash"
    ] = design_hash

    config[
        "design_status"
    ] = "FROZEN"

    save_json(

        config,

        OUTPUT_DIR
        /
        "factor_design_config.json",
    )

    # ========================================================
    # 8. Human-readable summary
    # ========================================================

    write_design_summary(

        config=
            config,

        factor_dictionary=
            factor_dictionary,

        inventory_summary=
            inventory_summary,
    )

    # ========================================================
    # 9. Console summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Frozen Factor Design"
    )

    print("=" * 80)

    print(
        f"Primary network: "
        f"{PRIMARY_NETWORK}"
    )

    print(
        f"Benchmark network: "
        f"{BENCHMARK_NETWORK}"
    )

    print(
        f"Primary q: "
        f"{100*PRIMARY_DENSITY:.2f}%"
    )

    print(
        f"Windows: "
        f"{WINDOWS}"
    )

    print(
        f"Candidate factors: "
        f"{len(factor_dictionary)}"
    )

    print(
        f"Core factors: "
        f"{factor_dictionary['core_factor'].sum()}"
    )

    print(
        f"Design hash:\n"
        f"{design_hash}"
    )

    print()
    print(
        "Input coverage:"
    )

    print(
        inventory_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)

    print(
        "Day 5 Step 1 Complete"
    )

    print("=" * 80)

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
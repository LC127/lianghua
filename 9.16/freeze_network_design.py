from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "01_step1_network_design"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Day 2 Step 1:
# baseline data / rolling-window design
# ------------------------------------------------------------

DAY2_STEP1_DIR_CANDIDATES = [

    OUTPUT_ROOT
    / "M1_day2"
    / "01_step1_baseline_config",

    Path(
        r"D:\M1_day2\01_step1_baseline_config"
    ),
]


def resolve_day2_step1_dir() -> Path:

    for path in (
        DAY2_STEP1_DIR_CANDIDATES
    ):

        if (
            path.exists()
            and
            (
                path
                / "day2_analysis_config.json"
            ).exists()
        ):

            return path

    raise FileNotFoundError(
        "未找到 Day 2 Step 1 baseline config。"
    )


DAY2_STEP1_DIR = (
    resolve_day2_step1_dir()
)

DAY2_CONFIG_PATH = (
    DAY2_STEP1_DIR
    / "day2_analysis_config.json"
)

ANALYSIS_DATES_PATH = (
    DAY2_STEP1_DIR
    / "analysis_dates_by_window.csv"
)


# ------------------------------------------------------------
# Day 2 Step 6:
# corrected Raw / Residual diagnostics
# ------------------------------------------------------------

STEP6_DIR = (
    OUTPUT_ROOT
    / "M1_day3"
    / "06_step6_market_residual"
)

STEP6_SUMMARY_PATH = (
    STEP6_DIR
    / "market_residual_summary.csv"
)

STEP6_WINDOW_PATH = (
    STEP6_DIR
    / "market_residual_window_comparison.csv"
)


# ============================================================
# 1. Formal Day 3 Network Design
# ============================================================

MAIN_RETURN_COLUMN = (
    "return_network"
)

ALTERNATIVE_RETURN_COLUMN = (
    "return_last_trade_log"
)


# ------------------------------------------------------------
# Rolling windows
#
# 最终会与 Day 2 config 做一致性检查
# ------------------------------------------------------------

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# Main pairwise-validity rules
# ------------------------------------------------------------

BASELINE_STOCK_VALID_RATIO = 0.90

BASELINE_PAIR_VALID_RATIO = 0.90


# ------------------------------------------------------------
# Global fixed-density network
#
# q = fraction of all comparable valid unordered pairs
# retained as edges.
# ------------------------------------------------------------

GLOBAL_DENSITY_GRID = [
    0.002,   # 0.2%
    0.005,   # 0.5%
    0.010,   # 1.0%
    0.020,   # 2.0%
]

PRIMARY_GLOBAL_DENSITY = 0.010


# ------------------------------------------------------------
# Per-node local-neighborhood robustness
# ------------------------------------------------------------

PER_NODE_TOPK_GRID = [
    10,
    25,
    50,
]

TOPK_SYMMETRIZATION = "OR"


# ------------------------------------------------------------
# Main network:
#
# strongest positive correlations
#
# Signed robustness:
#
# strongest absolute correlations,
# while retaining original correlation sign as weight.
# ------------------------------------------------------------

PRIMARY_EDGE_SCORE = (
    "positive_correlation"
)

SIGNED_ROBUSTNESS_SCORE = (
    "absolute_correlation"
)


# ------------------------------------------------------------
# Industry analysis
# ------------------------------------------------------------

MAIN_INDUSTRY_LEVEL = (
    "industry_id1"
)

ROBUSTNESS_INDUSTRY_LEVEL = (
    "industry_id2"
)


# ------------------------------------------------------------
# Deterministic edge selection
#
# Important when multiple pairs have equal scores.
# ------------------------------------------------------------

EDGE_TIE_BREAK_RULE = (
    "score_desc_then_security_id_i_then_security_id_j"
)


# ============================================================
# 2. Load Previous Design
# ============================================================

def load_day2_config():

    with open(
        DAY2_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    return config


def extract_day2_design(
    config,
):

    rolling = config[
        "rolling_design"
    ]

    return_info = config[
        "return"
    ]

    windows = [
        int(x)
        for x
        in rolling[
            "windows"
        ]
    ]

    stock_valid_ratio = float(
        rolling[
            "baseline_valid_ratio"
        ]
    )

    pair_valid_ratio = float(
        rolling[
            "pairwise_min_common_ratio"
        ]
    )

    baseline_return = (
        return_info[
            "baseline_column"
        ]
    )

    return (
        windows,
        stock_valid_ratio,
        pair_valid_ratio,
        baseline_return,
    )


# ============================================================
# 3. Load Analysis Dates
# ============================================================

def load_analysis_dates():

    if not ANALYSIS_DATES_PATH.exists():

        raise FileNotFoundError(
            f"不存在：{ANALYSIS_DATES_PATH}"
        )

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
            f"analysis_dates_by_window 缺字段："
            f"{missing}"
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
    ] = (
        pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        )
        .astype(int)
    )

    df = (
        df[
            df[
                "window"
            ]
            .isin(
                EXPECTED_WINDOWS
            )
        ]
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# 4. Load Step 6 for Universe Diagnostics
# ============================================================

def load_step6_summary():

    if not STEP6_SUMMARY_PATH.exists():

        raise FileNotFoundError(
            f"不存在：{STEP6_SUMMARY_PATH}"
        )

    df = pd.read_csv(
        STEP6_SUMMARY_PATH
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
    ] = (
        pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        )
        .astype(int)
    )

    if (
        "industry_level"
        in df.columns
    ):

        df = (
            df[
                df[
                    "industry_level"
                ]
                .eq(
                    MAIN_INDUSTRY_LEVEL
                )
            ]
            .copy()
        )

    return df


# ============================================================
# 5. Design Validation
# ============================================================

def validate_design_consistency(
    day2_windows,
    day2_stock_valid_ratio,
    day2_pair_valid_ratio,
    day2_return_column,
):

    rows = []

    def add_check(
        check_name,
        actual,
        expected,
        passed,
        note,
    ):

        rows.append(
            {
                "check":
                    check_name,

                "actual":
                    str(actual),

                "expected":
                    str(expected),

                "passed":
                    bool(passed),

                "note":
                    note,
            }
        )

    add_check(
        "Rolling windows",
        sorted(
            day2_windows
        ),
        EXPECTED_WINDOWS,
        sorted(
            day2_windows
        )
        ==
        EXPECTED_WINDOWS,
        (
            "Day 3 should retain the same "
            "60/120/252-day windows used in "
            "Day 2 diagnostics."
        ),
    )

    add_check(
        "Stock validity ratio",
        day2_stock_valid_ratio,
        BASELINE_STOCK_VALID_RATIO,
        np.isclose(
            day2_stock_valid_ratio,
            BASELINE_STOCK_VALID_RATIO,
        ),
        (
            "Main network construction should "
            "use the same stock-coverage rule."
        ),
    )

    add_check(
        "Pair validity ratio",
        day2_pair_valid_ratio,
        BASELINE_PAIR_VALID_RATIO,
        np.isclose(
            day2_pair_valid_ratio,
            BASELINE_PAIR_VALID_RATIO,
        ),
        (
            "Main Raw / Residual comparison "
            "should use the same pairwise "
            "common-observation rule."
        ),
    )

    add_check(
        "Baseline return definition",
        day2_return_column,
        MAIN_RETURN_COLUMN,
        (
            day2_return_column
            ==
            MAIN_RETURN_COLUMN
        ),
        (
            "Formal network should inherit "
            "the validated return_network baseline."
        ),
    )

    add_check(
        "Primary density belongs to grid",
        PRIMARY_GLOBAL_DENSITY,
        GLOBAL_DENSITY_GRID,
        (
            PRIMARY_GLOBAL_DENSITY
            in
            GLOBAL_DENSITY_GRID
        ),
        (
            "Primary q must be one of the "
            "pre-specified robustness densities."
        ),
    )

    add_check(
        "Top-k symmetrization",
        TOPK_SYMMETRIZATION,
        "OR",
        (
            TOPK_SYMMETRIZATION
            ==
            "OR"
        ),
        (
            "OR rule is frozen as the main "
            "per-node Top-k robustness design."
        ),
    )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# 6. Formal Comparable Universe Definition
# ============================================================

def build_universe_definition():

    rows = [

        {
            "level":
                "Node",

            "rule_id":
                "N1",

            "rule":
                (
                    "Security is in the point-in-time "
                    "SSE/SZSE listed universe at analysis date t."
                ),

            "purpose":
                "Avoid look-ahead universe selection.",
        },

        {
            "level":
                "Node",

            "rule_id":
                "N2",

            "rule":
                (
                    "Stock has at least ceil(0.90*W) "
                    "valid return_network observations "
                    "inside the rolling window."
                ),

            "purpose":
                (
                    "Maintain sufficient information "
                    "for rolling correlation estimation."
                ),
        },

        {
            "level":
                "Node",

            "rule_id":
                "N3",

            "rule":
                (
                    "Market-factor regression used for "
                    "residualization is estimable and "
                    "produces finite residuals."
                ),

            "purpose":
                (
                    "Raw and Residual networks must be "
                    "compared on the same node universe."
                ),
        },

        {
            "level":
                "Pair",

            "rule_id":
                "P1",

            "rule":
                (
                    "Both stocks belong to the common "
                    "node universe."
                ),

            "purpose":
                "Matched node support.",
        },

        {
            "level":
                "Pair",

            "rule_id":
                "P2",

            "rule":
                (
                    "Raw correlation and residual "
                    "correlation are both finite."
                ),

            "purpose":
                (
                    "Avoid comparing networks over "
                    "different candidate-pair sets."
                ),
        },

        {
            "level":
                "Pair",

            "rule_id":
                "P3",

            "rule":
                (
                    "The pair satisfies the minimum "
                    "common-observation requirement "
                    "ceil(0.90*W)."
                ),

            "purpose":
                "Common pairwise data quality rule.",
        },

        {
            "level":
                "Industry",

            "rule_id":
                "I1",

            "rule":
                (
                    "Industry label is NOT required "
                    "for inclusion in the network."
                ),

            "purpose":
                (
                    "Industry is an external structural "
                    "validation label, not a graph "
                    "construction filter."
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 7. Candidate Network Definitions
# ============================================================

def build_network_candidate_design():

    rows = [

        # ----------------------------------------------------
        # Raw baseline
        # ----------------------------------------------------

        {
            "network_id":
                "B0_RAW_POSITIVE",

            "network_family":
                "Raw Pearson",

            "role":
                "Unconditional baseline",

            "correlation_object":
                "rho_raw",

            "edge_score":
                "rho_raw",

            "edge_weight":
                "rho_raw",

            "edge_sign":
                "positive only",

            "sparsification":
                "Global Top-q",

            "primary":
                False,

            "interpretation":
                (
                    "Strongest unconditional positive "
                    "co-movement relationships."
                ),
        },

        # ----------------------------------------------------
        # Residual primary
        # ----------------------------------------------------

        {
            "network_id":
                "M1_RESIDUAL_POSITIVE",

            "network_family":
                "Market-Residual Pearson",

            "role":
                "Primary association candidate",

            "correlation_object":
                "rho_residual",

            "edge_score":
                "rho_residual",

            "edge_weight":
                "rho_residual",

            "edge_sign":
                "positive only",

            "sparsification":
                "Global Top-q",

            "primary":
                True,

            "interpretation":
                (
                    "Strongest positive associations "
                    "after removing the broad market mode."
                ),
        },

        # ----------------------------------------------------
        # Raw absolute / signed robustness
        # ----------------------------------------------------

        {
            "network_id":
                "R1_RAW_ABS_SIGNED",

            "network_family":
                "Raw Pearson",

            "role":
                "Signed-edge robustness",

            "correlation_object":
                "rho_raw",

            "edge_score":
                "abs(rho_raw)",

            "edge_weight":
                "rho_raw",

            "edge_sign":
                "retain original sign",

            "sparsification":
                "Global Top-q by absolute correlation",

            "primary":
                False,

            "interpretation":
                (
                    "Strongest raw associations "
                    "regardless of direction."
                ),
        },

        # ----------------------------------------------------
        # Residual absolute / signed robustness
        # ----------------------------------------------------

        {
            "network_id":
                "R2_RESIDUAL_ABS_SIGNED",

            "network_family":
                "Market-Residual Pearson",

            "role":
                "Signed-edge robustness",

            "correlation_object":
                "rho_residual",

            "edge_score":
                "abs(rho_residual)",

            "edge_weight":
                "rho_residual",

            "edge_sign":
                "retain original sign",

            "sparsification":
                "Global Top-q by absolute correlation",

            "primary":
                False,

            "interpretation":
                (
                    "Strongest market-adjusted "
                    "associations including negative edges."
                ),
        },

        # ----------------------------------------------------
        # Local Top-k robustness
        # ----------------------------------------------------

        {
            "network_id":
                "R3_RESIDUAL_TOPK",

            "network_family":
                "Market-Residual Pearson",

            "role":
                "Local-neighborhood robustness",

            "correlation_object":
                "rho_residual",

            "edge_score":
                "rho_residual",

            "edge_weight":
                "rho_residual",

            "edge_sign":
                "positive only",

            "sparsification":
                "Per-node Top-k + OR symmetrization",

            "primary":
                False,

            "interpretation":
                (
                    "Each stock retains its strongest "
                    "local positive residual neighbors."
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 8. Global Density Grid
# ============================================================

def build_density_grid():

    rows = []

    for q in (
        GLOBAL_DENSITY_GRID
    ):

        rows.append(
            {
                "density_fraction":
                    q,

                "density_percent":
                    100 * q,

                "is_primary":
                    (
                        q
                        ==
                        PRIMARY_GLOBAL_DENSITY
                    ),

                "selection_rule":
                    (
                        "K = floor("
                        "q * M_common_valid_pairs)"
                    ),

                "tie_break":
                    EDGE_TIE_BREAK_RULE,

                "purpose":
                    (
                        "Primary"
                        if (
                            q
                            ==
                            PRIMARY_GLOBAL_DENSITY
                        )
                        else
                        "Density robustness"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 9. Per-node Top-k Grid
# ============================================================

def build_topk_grid():

    rows = []

    for k in (
        PER_NODE_TOPK_GRID
    ):

        rows.append(
            {
                "top_k":
                    k,

                "symmetrization":
                    TOPK_SYMMETRIZATION,

                "edge_rule":
                    (
                        "Edge exists if i selects j "
                        "OR j selects i."
                    ),

                "score":
                    "positive residual correlation",

                "purpose":
                    "Local-neighborhood robustness",
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. Estimate Network Scale
#
# Important:
#
# This is an approximate edge budget based on
# C(p,2).
#
# The exact K at each date will be determined in
# Day 3 Step 2 using M_common_valid_pairs.
# ============================================================

def build_expected_network_scale(
    step6_summary,
):

    required = [
        "analysis_date",
        "window",
        "eligible_stock_count",
    ]

    missing = [
        col
        for col in required
        if col not in (
            step6_summary.columns
        )
    ]

    if missing:

        raise ValueError(
            f"Step 6 summary 缺字段：{missing}"
        )

    rows = []

    for row in (
        step6_summary[
            required
        ]
        .drop_duplicates(
            subset=[
                "analysis_date",
                "window",
            ]
        )
        .itertuples(
            index=False
        )
    ):

        p = int(
            row.eligible_stock_count
        )

        possible_pairs = (
            p
            *
            (
                p - 1
            )
            //
            2
        )

        for q in (
            GLOBAL_DENSITY_GRID
        ):

            approximate_edges = int(
                math.floor(
                    q
                    *
                    possible_pairs
                )
            )

            approximate_mean_degree = (
                2.0
                *
                approximate_edges
                /
                p
                if p > 0
                else np.nan
            )

            rows.append(
                {
                    "analysis_date":
                        row.analysis_date,

                    "window":
                        int(
                            row.window
                        ),

                    "eligible_stock_count":
                        p,

                    "possible_unordered_pairs":
                        possible_pairs,

                    "density_fraction":
                        q,

                    "density_percent":
                        100 * q,

                    "approximate_edge_count":
                        approximate_edges,

                    "approximate_mean_degree":
                        approximate_mean_degree,

                    "is_primary_density":
                        (
                            q
                            ==
                            PRIMARY_GLOBAL_DENSITY
                        ),

                    "important_note":
                        (
                            "Approximation only. "
                            "Step 2 will use the exact "
                            "common valid pair universe."
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. Window-level Expected Scale
# ============================================================

def summarize_expected_scale(
    scale_df,
):

    result = (
        scale_df.groupby(
            [
                "window",
                "density_fraction",
                "density_percent",
                "is_primary_density",
            ],
            as_index=False,
        )
        .agg(

            analysis_date_count=
                (
                    "analysis_date",
                    "count",
                ),

            mean_stock_count=
                (
                    "eligible_stock_count",
                    "mean",
                ),

            min_stock_count=
                (
                    "eligible_stock_count",
                    "min",
                ),

            max_stock_count=
                (
                    "eligible_stock_count",
                    "max",
                ),

            mean_approx_edge_count=
                (
                    "approximate_edge_count",
                    "mean",
                ),

            min_approx_edge_count=
                (
                    "approximate_edge_count",
                    "min",
                ),

            max_approx_edge_count=
                (
                    "approximate_edge_count",
                    "max",
                ),

            mean_approx_degree=
                (
                    "approximate_mean_degree",
                    "mean",
                ),
        )
    )

    return result


# ============================================================
# 12. Formal Edge Selection Rules
# ============================================================

def build_edge_selection_rules():

    rows = [

        {
            "rule_id":
                "E1",

            "rule":
                (
                    "Graph is undirected, simple, "
                    "with no self-loops."
                ),

            "reason":
                (
                    "Pearson correlation is symmetric."
                ),
        },

        {
            "rule_id":
                "E2",

            "rule":
                (
                    "Raw and Residual networks use "
                    "exactly the same common valid "
                    "candidate-pair universe."
                ),

            "reason":
                (
                    "Prevent apparent topology differences "
                    "from being caused by different "
                    "missing-data support."
                ),
        },

        {
            "rule_id":
                "E3",

            "rule":
                (
                    "At density q, both Raw and Residual "
                    "networks retain exactly "
                    "K=floor(q*M_common) edges."
                ),

            "reason":
                (
                    "Guarantee matched density and "
                    "matched edge budget."
                ),
        },

        {
            "rule_id":
                "E4",

            "rule":
                (
                    "Primary positive network ranks "
                    "pairs by correlation itself, "
                    "not absolute correlation."
                ),

            "reason":
                (
                    "Primary economic object is strong "
                    "positive synchronization after "
                    "market adjustment."
                ),
        },

        {
            "rule_id":
                "E5",

            "rule":
                (
                    "If fewer than K positive candidate "
                    "pairs are available, construction "
                    "must fail with a diagnostic warning "
                    "rather than fill the graph with "
                    "negative edges."
                ),

            "reason":
                (
                    "Preserve the meaning of the "
                    "positive-association network."
                ),
        },

        {
            "rule_id":
                "E6",

            "rule":
                (
                    "Signed robustness ranks by "
                    "absolute correlation and retains "
                    "the original sign as edge weight."
                ),

            "reason":
                (
                    "Separate magnitude-based signed "
                    "association from the primary "
                    "positive network."
                ),
        },

        {
            "rule_id":
                "E7",

            "rule":
                (
                    "Ties are resolved deterministically "
                    "by score descending, followed by "
                    "security_id_i and security_id_j."
                ),

            "reason":
                (
                    "Ensure exact reproducibility of "
                    "edge sets."
                ),
        },

        {
            "rule_id":
                "E8",

            "rule":
                (
                    "Industry labels are never used "
                    "to select edges."
                ),

            "reason":
                (
                    "Industry is a validation target; "
                    "using it during construction would "
                    "make later enrichment analysis circular."
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 13. Validation Metric Plan
# ============================================================

def build_validation_metric_plan():

    rows = [

        # ----------------------------------------------------
        # Step 3
        # ----------------------------------------------------

        {
            "future_step":
                "Day3-Step3",

            "metric_group":
                "Basic topology",

            "metric":
                "Node count N",

            "main_question":
                "Is node support comparable?",
        },

        {
            "future_step":
                "Day3-Step3",

            "metric_group":
                "Basic topology",

            "metric":
                "Edge count E / Density",

            "main_question":
                (
                    "Is matched-density construction "
                    "implemented correctly?"
                ),
        },

        {
            "future_step":
                "Day3-Step3",

            "metric_group":
                "Degree",

            "metric":
                "Degree distribution",

            "main_question":
                (
                    "How heterogeneous are local "
                    "network neighborhoods?"
                ),
        },

        {
            "future_step":
                "Day3-Step3",

            "metric_group":
                "Degree",

            "metric":
                "Weighted degree / Strength",

            "main_question":
                (
                    "Which stocks have the strongest "
                    "aggregate association exposure?"
                ),
        },

        {
            "future_step":
                "Day3-Step3",

            "metric_group":
                "Connectivity",

            "metric":
                "Giant component share",

            "main_question":
                (
                    "Does sparsification fragment "
                    "the market network?"
                ),
        },

        {
            "future_step":
                "Day3-Step3",

            "metric_group":
                "Concentration",

            "metric":
                "Top 1% / 5% degree share",

            "main_question":
                (
                    "Is network connectivity dominated "
                    "by a small set of hubs?"
                ),
        },

        # ----------------------------------------------------
        # Step 4
        # ----------------------------------------------------

        {
            "future_step":
                "Day3-Step4",

            "metric_group":
                "Industry",

            "metric":
                "Same-industry edge share",

            "main_question":
                (
                    "How much of the sparse network "
                    "lies within industries?"
                ),
        },

        {
            "future_step":
                "Day3-Step4",

            "metric_group":
                "Industry",

            "metric":
                "Industry enrichment",

            "main_question":
                (
                    "Are within-industry edges "
                    "over-represented relative to "
                    "the feasible-pair baseline?"
                ),
        },

        {
            "future_step":
                "Day3-Step4",

            "metric_group":
                "Industry",

            "metric":
                "Industry assortativity",

            "main_question":
                (
                    "Does the graph exhibit "
                    "attribute assortative mixing?"
                ),
        },

        {
            "future_step":
                "Day3-Step4",

            "metric_group":
                "Community",

            "metric":
                "Modularity / NMI with industry",

            "main_question":
                (
                    "Do endogenous network communities "
                    "align with economic industries?"
                ),
        },

        # ----------------------------------------------------
        # Step 5
        # ----------------------------------------------------

        {
            "future_step":
                "Day3-Step5",

            "metric_group":
                "Dynamic stability",

            "metric":
                "Adjacent edge Jaccard",

            "main_question":
                (
                    "How stable are exact edge sets "
                    "across adjacent month-ends?"
                ),
        },

        {
            "future_step":
                "Day3-Step5",

            "metric_group":
                "Dynamic stability",

            "metric":
                "Edge retention",

            "main_question":
                (
                    "What fraction of current edges "
                    "survive to the next observation?"
                ),
        },

        {
            "future_step":
                "Day3-Step5",

            "metric_group":
                "Node stability",

            "metric":
                "Degree rank Spearman",

            "main_question":
                (
                    "Are central stocks stable over time?"
                ),
        },

        {
            "future_step":
                "Day3-Step5",

            "metric_group":
                "Node stability",

            "metric":
                "Strength rank Spearman",

            "main_question":
                (
                    "Are weighted network positions "
                    "stable over time?"
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 14. Input Manifest
# ============================================================

def build_input_manifest():

    paths = [

        (
            "Day2 baseline config",
            DAY2_CONFIG_PATH,
        ),

        (
            "Day2 analysis dates",
            ANALYSIS_DATES_PATH,
        ),

        (
            "Step6 residual summary",
            STEP6_SUMMARY_PATH,
        ),

        (
            "Step6 window comparison",
            STEP6_WINDOW_PATH,
        ),
    ]

    rows = []

    for name, path in paths:

        rows.append(
            {
                "input_name":
                    name,

                "path":
                    str(path),

                "exists":
                    path.exists(),

                "size_bytes":
                    (
                        path.stat().st_size
                        if path.exists()
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. Save Master JSON Config
# ============================================================

def save_master_config(
    day2_config,
    analysis_dates,
):

    config = {

        "project":
            "M1_Stock_Association_Network",

        "stage":
            (
                "Day3_Step1_"
                "Freeze_Formal_Network_Construction_Design"
            ),

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        # ----------------------------------------------------
        # Data inherited from Day 2
        # ----------------------------------------------------

        "data_design": {

            "return_column":
                MAIN_RETURN_COLUMN,

            "alternative_return_column":
                ALTERNATIVE_RETURN_COLUMN,

            "windows":
                EXPECTED_WINDOWS,

            "stock_valid_ratio":
                BASELINE_STOCK_VALID_RATIO,

            "pair_valid_ratio":
                BASELINE_PAIR_VALID_RATIO,

            "analysis_frequency":
                "month_end_actual_trading_day",

            "market_scope":
                "SSE_SZSE",
        },

        # ----------------------------------------------------
        # Comparable universe
        # ----------------------------------------------------

        "comparison_principle": {

            "same_date":
                True,

            "same_window":
                True,

            "same_node_universe":
                True,

            "same_common_pair_universe":
                True,

            "same_edge_budget":
                True,

            "industry_used_for_edge_selection":
                False,
        },

        # ----------------------------------------------------
        # Primary comparison
        # ----------------------------------------------------

        "primary_network_comparison": {

            "baseline":
                "B0_RAW_POSITIVE",

            "candidate":
                "M1_RESIDUAL_POSITIVE",

            "graph_type":
                "undirected_simple_weighted",

            "edge_score":
                "positive correlation",

            "primary_density":
                PRIMARY_GLOBAL_DENSITY,

            "primary_density_percent":
                (
                    100
                    *
                    PRIMARY_GLOBAL_DENSITY
                ),

            "edge_budget":
                (
                    "K=floor("
                    "q*M_common_valid_pairs)"
                ),

            "tie_break":
                EDGE_TIE_BREAK_RULE,
        },

        # ----------------------------------------------------
        # Robustness
        # ----------------------------------------------------

        "density_robustness": {

            "q_grid":
                GLOBAL_DENSITY_GRID,

            "q_percent":
                [
                    100 * q
                    for q in
                    GLOBAL_DENSITY_GRID
                ],
        },

        "local_network_robustness": {

            "top_k_grid":
                PER_NODE_TOPK_GRID,

            "symmetrization":
                TOPK_SYMMETRIZATION,
        },

        "signed_network_robustness": {

            "enabled":
                True,

            "selection_score":
                "absolute_correlation",

            "edge_weight":
                "signed_correlation",
        },

        # ----------------------------------------------------
        # Industry
        # ----------------------------------------------------

        "industry_validation": {

            "main_level":
                MAIN_INDUSTRY_LEVEL,

            "robustness_level":
                ROBUSTNESS_INDUSTRY_LEVEL,

            "industry_is_construction_input":
                False,

            "industry_is_validation_target":
                True,
        },

        # ----------------------------------------------------
        # Edge output schema
        # ----------------------------------------------------

        "required_edge_output_columns": [

            "analysis_date",
            "window",
            "density",
            "network_id",

            "security_id_i",
            "security_id_j",

            "correlation",
            "edge_score",
            "edge_weight",
            "edge_sign",

            "rank",
            "same_industry",
        ],

        # ----------------------------------------------------
        # Important interpretation
        # ----------------------------------------------------

        "important_notes": [

            (
                "Raw and Residual networks must be "
                "constructed on the identical common "
                "candidate-pair universe."
            ),

            (
                "The primary network retains strong "
                "positive associations. Negative "
                "residual correlations are analyzed "
                "separately in signed robustness."
            ),

            (
                "Fixed absolute correlation thresholds "
                "are not used as the primary sparsification "
                "rule because correlation distributions "
                "vary across market regimes."
            ),

            (
                "Industry labels are used only for "
                "post-construction validation and never "
                "for selecting network edges."
            ),

            (
                "Market-Residual Pearson is a primary "
                "candidate rather than a final winner; "
                "Day 3 network-level validation can still "
                "challenge this working choice."
            ),
        ],

        "analysis_date_count":
            int(
                len(
                    analysis_dates
                )
            ),
    }

    with open(
        OUTPUT_DIR
        / "day3_network_design_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return config


# ============================================================
# 16. Markdown Design Summary
# ============================================================

def build_markdown_summary(
    validation_df,
    expected_scale_summary,
):

    passed = int(
        validation_df[
            "passed"
        ].sum()
    )

    total = len(
        validation_df
    )

    primary_scale = (
        expected_scale_summary[
            expected_scale_summary[
                "is_primary_density"
            ]
        ]
        .copy()
    )

    scale_lines = []

    for row in (
        primary_scale
        .sort_values(
            "window"
        )
        .itertuples(
            index=False
        )
    ):

        scale_lines.append(
            (
                f"- W={row.window}: "
                f"mean stocks ≈ "
                f"{row.mean_stock_count:.0f}, "
                f"mean edges ≈ "
                f"{row.mean_approx_edge_count:,.0f}, "
                f"mean degree ≈ "
                f"{row.mean_approx_degree:.1f}"
            )
        )

    scale_text = "\n".join(
        scale_lines
    )

    text = f"""# Day 3 Step 1 — Formal Network Construction Design

## 1. Objective

Freeze the formal construction rules before producing full-market
Raw and Market-Residual networks.

The central comparison is:

\[
G^{{Raw}}_{{t,W,q}}
\quad\text{{vs}}\quad
G^{{Residual}}_{{t,W,q}}.
\]

The comparison must use the same date, rolling window, node universe,
candidate-pair universe, and edge budget.

## 2. Primary network

Primary candidate:

**PIT leave-one-out market-residual Pearson positive network**

Baseline:

**Raw Pearson positive network**

Primary sparsification:

\[
q=1\%.
\]

Density robustness:

\[
q\in\{{0.2\%,0.5\%,1\%,2\%\}}.
\]

## 3. Comparable-pair principle

At each \((t,W)\), first construct one common valid pair universe.
Only pairs for which both Raw and Residual correlations are available
are eligible for the matched comparison.

The exact edge count is then

\[
K_{{t,W,q}}
=
\left\lfloor
q M_{{t,W}}^{{common}}
\right\rfloor .
\]

Both Raw and Residual networks retain exactly \(K_{{t,W,q}}\) edges.

## 4. Positive network versus signed network

The primary network ranks pairs by positive correlation.

Signed robustness ranks pairs by absolute correlation and retains the
original sign as the edge weight.

These are treated as distinct economic objects rather than mixed in a
single main network.

## 5. Local-network robustness

Per-node Top-k robustness uses:

\[
k\in\{{10,25,50\}}
\]

with OR symmetrization.

## 6. Primary q=1% approximate network scale

{scale_text}

The exact edge budget will be recomputed in Step 2 from the common
valid-pair universe; the numbers above are only scale diagnostics based
on \(\binom{{p}}{{2}}\).

## 7. Industry information

Industry labels are not used to construct the graph.

They are retained only for post-construction validation such as:

- same-industry edge share;
- industry enrichment;
- industry assortativity;
- community–industry alignment.

This avoids circular validation.

## 8. Design QA

Passed checks:

**{passed}/{total}**

All checks should pass before Step 2 is executed.

## 9. Next step

Day 3 Step 2 will construct matched-density Raw and Market-Residual
networks at each month-end and save complete edge tables for subsequent
structural, industry, and dynamic-stability validation.
"""

    with open(
        OUTPUT_DIR
        / "day3_step1_network_design_summary.md",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            text
        )


# ============================================================
# 17. Main
# ============================================================

def main():

    print("=" * 78)
    print(
        "M1 Day 3 - Step 1"
    )
    print(
        "Freeze Formal Network Construction Design"
    )
    print("=" * 78)

    # ========================================================
    # Load Day 2 design
    # ========================================================

    print()
    print(
        "[1] Load Day 2 baseline design"
    )

    day2_config = (
        load_day2_config()
    )

    (
        day2_windows,
        day2_stock_ratio,
        day2_pair_ratio,
        day2_return_column,
    ) = (
        extract_day2_design(
            day2_config
        )
    )

    analysis_dates = (
        load_analysis_dates()
    )

    step6_summary = (
        load_step6_summary()
    )

    # ========================================================
    # Design consistency checks
    # ========================================================

    print()
    print(
        "[2] Validate design consistency"
    )

    validation_df = (
        validate_design_consistency(
            day2_windows=
                day2_windows,

            day2_stock_valid_ratio=
                day2_stock_ratio,

            day2_pair_valid_ratio=
                day2_pair_ratio,

            day2_return_column=
                day2_return_column,
        )
    )

    validation_df.to_csv(
        OUTPUT_DIR
        / "design_validation_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not (
        validation_df[
            "passed"
        ]
        .all()
    ):

        print()
        print(
            validation_df.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Day 3 network design 与 Day 2 "
            "baseline design 不一致。"
        )

    # ========================================================
    # Formal design tables
    # ========================================================

    print()
    print(
        "[3] Build formal network design tables"
    )

    universe_df = (
        build_universe_definition()
    )

    candidate_df = (
        build_network_candidate_design()
    )

    density_df = (
        build_density_grid()
    )

    topk_df = (
        build_topk_grid()
    )

    edge_rule_df = (
        build_edge_selection_rules()
    )

    metric_plan_df = (
        build_validation_metric_plan()
    )

    input_manifest_df = (
        build_input_manifest()
    )

    # ========================================================
    # Expected scale
    # ========================================================

    print()
    print(
        "[4] Estimate full-market network scale"
    )

    scale_df = (
        build_expected_network_scale(
            step6_summary
        )
    )

    scale_summary_df = (
        summarize_expected_scale(
            scale_df
        )
    )

    # ========================================================
    # Save
    # ========================================================

    print()
    print(
        "[5] Save frozen design"
    )

    universe_df.to_csv(
        OUTPUT_DIR
        / "common_universe_definition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidate_df.to_csv(
        OUTPUT_DIR
        / "network_candidate_design.csv",
        index=False,
        encoding="utf-8-sig",
    )

    density_df.to_csv(
        OUTPUT_DIR
        / "global_density_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topk_df.to_csv(
        OUTPUT_DIR
        / "per_node_topk_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )

    edge_rule_df.to_csv(
        OUTPUT_DIR
        / "formal_edge_selection_rules.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_plan_df.to_csv(
        OUTPUT_DIR
        / "network_validation_metric_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    input_manifest_df.to_csv(
        OUTPUT_DIR
        / "input_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    analysis_dates.to_csv(
        OUTPUT_DIR
        / "day3_analysis_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scale_df.to_csv(
        OUTPUT_DIR
        / "expected_network_scale_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scale_summary_df.to_csv(
        OUTPUT_DIR
        / "expected_network_scale_by_window.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_master_config(
        day2_config=
            day2_config,

        analysis_dates=
            analysis_dates,
    )

    build_markdown_summary(
        validation_df=
            validation_df,

        expected_scale_summary=
            scale_summary_df,
    )

    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 78)
    print(
        "Design Validation"
    )
    print("=" * 78)

    print(
        validation_df[
            [
                "check",
                "passed",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "Network Candidates"
    )
    print("=" * 78)

    print(
        candidate_df[
            [
                "network_id",
                "role",
                "edge_score",
                "sparsification",
                "primary",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "Expected Network Scale"
    )
    print("=" * 78)

    show_scale = (
        scale_summary_df[
            [
                "window",
                "density_percent",
                "mean_stock_count",
                "mean_approx_edge_count",
                "mean_approx_degree",
                "is_primary_density",
            ]
        ]
        .sort_values(
            [
                "window",
                "density_percent",
            ]
        )
    )

    print(
        show_scale.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "Day 4 Step 1 Complete"
    )
    print("=" * 78)

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
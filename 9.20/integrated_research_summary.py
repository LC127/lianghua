from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(
    r"D:\M1_StockNetwork"
)

OUTPUT_ROOT = (
    ROOT
    / "output"
)


# ------------------------------------------------------------
# Day 6
# Research Day 6 -> M1_day6
# ------------------------------------------------------------

DAY6_ROOT = (
    OUTPUT_ROOT
    / "M1_day6"
)

FROZEN_CORE_PATH = (
    DAY6_ROOT
    / "01_step1_freeze_reduced_factor_set"
    / "frozen_core_factor_set.csv"
)

STEP6_DAY6_METADATA_PATH = (
    DAY6_ROOT
    / "06_step6_neutralization_and_classification"
    / "step6_neutralization_metadata.json"
)


# ------------------------------------------------------------
# Day 7
# Research Day 7 -> M1_day7
# ------------------------------------------------------------

DAY7_ROOT = (
    OUTPUT_ROOT
    / "M1_day7"
)


# ------------------------------------------------------------
# Day 7 Step 1
# ------------------------------------------------------------

STEP1_METADATA_PATH = (
    DAY7_ROOT
    / "01_stage1_subsample_regime_robustness"
    / "day7_step1_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 2
# ------------------------------------------------------------

STEP2_METADATA_PATH = (
    DAY7_ROOT
    / "02_stage2_traditional_characteristic_controls"
    / "day7_step2_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 3
# ------------------------------------------------------------

STEP3_METADATA_PATH = (
    DAY7_ROOT
    / "03_stage3_fama_macbeth_regression"
    / "day7_step3_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 4
# ------------------------------------------------------------

STEP4_METADATA_PATH = (
    DAY7_ROOT
    / "04_stage4_turnover_transaction_cost"
    / "day7_step4_metadata.json"
)


# ------------------------------------------------------------
# Day 7 Step 5
# ------------------------------------------------------------

STEP5_DIR = (
    DAY7_ROOT
    / "05_stage5_final_factor_robustness_matrix"
)

STEP5_METADATA_PATH = (
    STEP5_DIR
    / "day7_step5_metadata.json"
)

FACTOR_MATRIX_PATH = (
    STEP5_DIR
    / "final_factor_robustness_matrix.csv"
)

WINDOW_MATRIX_PATH = (
    STEP5_DIR
    / "final_factor_robustness_matrix_by_window.csv"
)

ROLE_SUMMARY_PATH = (
    STEP5_DIR
    / "final_factor_role_summary.csv"
)


# ------------------------------------------------------------
# Day 7 Step 6
# ------------------------------------------------------------

OUTPUT_DIR = (
    DAY7_ROOT
    / "06_stage6_integrated_research_summary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT_PATH = (
    OUTPUT_DIR
    / "M1_Day1_Day7_Integrated_Research_Summary.md"
)

FACTOR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "integrated_factor_summary.csv"
)

TIMELINE_PATH = (
    OUTPUT_DIR
    / "integrated_research_timeline.csv"
)

KEY_FINDINGS_PATH = (
    OUTPUT_DIR
    / "integrated_research_key_findings.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "integrated_research_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day7_step6_metadata.json"
)


# ============================================================
# 1. Frozen design
# ============================================================

EXPECTED_FACTOR_COUNT = 9

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# Feature-family metadata
#
# These are structural labels defined from factor construction.
# They are NOT outcome-based classifications.
# ------------------------------------------------------------

FACTOR_FAMILY = {

    "residual_degree_percentile":
        "LEVEL_STRUCTURAL",

    "delta_degree_percentile":
        "LEVEL_STRUCTURAL",

    "cross_industry_degree_percentile":
        "LEVEL_STRUCTURAL",

    "cross_industry_degree_ratio":
        "LEVEL_STRUCTURAL",

    "outside_community_degree_percentile":
        "LEVEL_STRUCTURAL",

    "delta_residual_degree_percentile_1m":
        "DYNAMIC_1M",

    "delta_cross_industry_degree_percentile_1m":
        "DYNAMIC_1M",

    "neighbor_jaccard_1m":
        "DYNAMIC_1M",

    "neighbor_retention_1m":
        "DYNAMIC_1M",
}


# ============================================================
# 2. Utilities
# ============================================================

def load_json(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing JSON:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json_atomic(
    obj,
    path: Path,
):

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )

    os.replace(
        temp,
        path,
    )


def save_csv_atomic(
    df: pd.DataFrame,
    path: Path,
):

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    df.to_csv(
        temp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temp,
        path,
    )


def save_text_atomic(
    text,
    path,
):

    temp = Path(
        str(path)
        +
        ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            text
        )

    os.replace(
        temp,
        path,
    )


def canonical_hash(obj):

    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


def require_columns(
    df,
    columns,
    name,
):

    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name} missing columns:\n"
            f"{missing}"
        )


def bool_value(x):

    if isinstance(
        x,
        (bool, np.bool_)
    ):

        return bool(
            x
        )

    if pd.isna(
        x
    ):

        return False

    text = str(
        x
    ).strip().lower()

    return (
        text
        in
        {
            "true",
            "1",
            "yes",
        }
    )


def fmt_num(
    x,
    digits=4,
):

    if (
        x is None
        or
        pd.isna(
            x
        )
    ):

        return "NA"

    return (
        f"{float(x):.{digits}f}"
    )


def fmt_bp(
    x,
    digits=2,
):

    if (
        x is None
        or
        pd.isna(
            x
        )
    ):

        return "NA"

    return (
        f"{float(x):.{digits}f} bp"
    )


def direction_label(
    x,
    tolerance=1e-12,
):

    if (
        x is None
        or
        pd.isna(
            x
        )
    ):

        return "NA"

    x = float(
        x
    )

    if x > tolerance:

        return "POSITIVE"

    if x < -tolerance:

        return "NEGATIVE"

    return "NEAR_ZERO"


# ============================================================
# 3. Parent QA validation
# ============================================================

def validate_upstream():

    metadata_paths = {

        "day6_step6":
            STEP6_DAY6_METADATA_PATH,

        "day7_step1":
            STEP1_METADATA_PATH,

        "day7_step2":
            STEP2_METADATA_PATH,

        "day7_step3":
            STEP3_METADATA_PATH,

        "day7_step4":
            STEP4_METADATA_PATH,

        "day7_step5":
            STEP5_METADATA_PATH,
    }

    metadata = {}

    for name, path in (
        metadata_paths.items()
    ):

        meta = load_json(
            path
        )

        formal_qa = meta.get(
            "formal_qa",
            {}
        )

        if (
            formal_qa.get(
                "all_formal_qa_pass"
            )
            is not True
        ):

            raise RuntimeError(
                f"{name} formal QA failed."
            )

        metadata[
            name
        ] = meta

    required = [

        FROZEN_CORE_PATH,

        FACTOR_MATRIX_PATH,

        WINDOW_MATRIX_PATH,

        ROLE_SUMMARY_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    return metadata


# ============================================================
# 4. Load final evidence
# ============================================================

def load_final_evidence():

    factor_matrix = pd.read_csv(
        FACTOR_MATRIX_PATH
    )

    window_matrix = pd.read_csv(
        WINDOW_MATRIX_PATH
    )

    role_summary = pd.read_csv(
        ROLE_SUMMARY_PATH
    )

    require_columns(
        factor_matrix,
        [
            "factor_order",
            "factor_name",
            "evidence_role",
            "cost_profile",

            "full_control_mean_alpha_ic",
            "m3_mean_beta_bps_per_1sd",
            "full_control_mean_spread_bps",
            "full_control_mean_ivol_ic",

            "stable_risk_target_count_out_of_4",

            "regime_alpha_strict_all_windows",
            "regime_ivol_strict_all_windows",

            "cost_magnitude_survival_windows_10bp",

            "mean_roundtrip_traded_notional_ratio",
            "mean_descriptive_break_even_cost_bps",
        ],
        "final_factor_robustness_matrix",
    )

    require_columns(
        window_matrix,
        [
            "factor_order",
            "factor_name",
            "window",

            "full_control_alpha_ic",
            "full_control_alpha_ic_hac_t",

            "full_control_spread_bps",
            "full_control_spread_hac_t",

            "m3_beta_bps_per_1sd",
            "m3_hac_t_stat",

            "ivol_ic",
            "ivol_hac_t",

            "mean_roundtrip_traded_notional_ratio",

            "descriptive_break_even_one_way_cost_bps",
        ],
        "final_factor_robustness_matrix_by_window",
    )

    if (
        len(
            factor_matrix
        )
        !=
        EXPECTED_FACTOR_COUNT
    ):

        raise RuntimeError(
            "Unexpected factor-matrix row count."
        )

    if (
        len(
            window_matrix
        )
        !=
        EXPECTED_FACTOR_COUNT
        *
        len(
            EXPECTED_WINDOWS
        )
    ):

        raise RuntimeError(
            "Unexpected window-matrix row count."
        )

    if (
        factor_matrix[
            "factor_name"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate factor-level rows."
        )

    if (
        window_matrix[
            [
                "factor_name",
                "window",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate factor-window rows."
        )

    return (
        factor_matrix,
        window_matrix,
        role_summary,
    )


# ============================================================
# 5. Research timeline
# ============================================================

def build_research_timeline():

    rows = [

        {
            "research_day":
                1,

            "stage":
                "Full-market data preparation and QA",

            "main_question":
                (
                    "Can a clean point-in-time A-share panel "
                    "and consistent return system be constructed?"
                ),

            "main_work":
                (
                    "Universe definition, daily-market QA, "
                    "network-return validation, holding-return "
                    "validation, suspension/reopening treatment."
                ),

            "key_methodological_rule":
                (
                    "Network construction return and holding-period "
                    "return are kept as two distinct conventions."
                ),
        },

        {
            "research_day":
                2,

            "stage":
                "Initial market association-network construction",

            "main_question":
                (
                    "What structure is visible in broad "
                    "stock-return association networks?"
                ),

            "main_work":
                (
                    "Construct rolling stock association networks "
                    "and establish initial network representations."
                ),

            "key_methodological_rule":
                (
                    "Only contemporaneously available historical "
                    "returns enter network construction."
                ),
        },

        {
            "research_day":
                3,

            "stage":
                "Residual-network construction",

            "main_question":
                (
                    "How does network structure change after "
                    "removing common market-related dependence?"
                ),

            "main_work":
                (
                    "Compare raw and residual association networks "
                    "and examine relative changes in connectivity."
                ),

            "key_methodological_rule":
                (
                    "Residual correlation is interpreted as a "
                    "conditional-association proxy, not as "
                    "conditional independence."
                ),
        },

        {
            "research_day":
                4,

            "stage":
                "Full-market network structural validation",

            "main_question":
                (
                    "Are observed network structures stable "
                    "across construction choices?"
                ),

            "main_work":
                (
                    "Network construction, structural statistics, "
                    "community structure and robustness diagnostics."
                ),

            "key_methodological_rule":
                (
                    "Network-construction robustness is separated "
                    "from subsequent return/risk validation."
                ),
        },

        {
            "research_day":
                5,

            "stage":
                "Network feature engineering",

            "main_question":
                (
                    "Which interpretable node-level characteristics "
                    "summarize network position and dynamics?"
                ),

            "main_work":
                (
                    "Construct 29 candidate characteristics, "
                    "perform pre-outcome QA, and freeze nine "
                    "network variables before future outcomes."
                ),

            "key_methodological_rule":
                (
                    "The nine-factor set is frozen before "
                    "future-return and future-risk labels are used."
                ),
        },

        {
            "research_day":
                6,

            "stage":
                "Initial forward return and risk validation",

            "main_question":
                (
                    "Do frozen network variables contain "
                    "forward return or risk information?"
                ),

            "main_work":
                (
                    "Construct forward labels, Rank IC, quintile "
                    "sorts, risk validation, and industry/size "
                    "neutralization."
                ),

            "key_methodological_rule":
                (
                    "No outcome-based sign flip, factor deletion "
                    "or network-window selection."
                ),
        },

        {
            "research_day":
                7,

            "stage":
                "Robustness and incremental validation",

            "main_question":
                (
                    "Do Day-6 relations survive regimes, "
                    "traditional characteristics, conditional "
                    "regressions and transaction costs?"
                ),

            "main_work":
                (
                    "Subsample/regime tests, traditional-stock "
                    "controls, Fama-MacBeth regressions, turnover "
                    "and transaction-cost analysis, final "
                    "robustness matrix."
                ),

            "key_methodological_rule":
                (
                    "Robustness diagnostics do not re-freeze "
                    "factors or select an optimal window."
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 6. Integrated factor summary
# ============================================================

def build_factor_summary(
    factor_matrix,
):

    result = (
        factor_matrix.copy()
    )

    result[
        "feature_family"
    ] = result[
        "factor_name"
    ].map(
        FACTOR_FAMILY
    )

    result[
        "return_ic_direction"
    ] = result[
        "full_control_mean_alpha_ic"
    ].map(
        direction_label
    )

    result[
        "fmb_beta_direction"
    ] = result[
        "m3_mean_beta_bps_per_1sd"
    ].map(
        direction_label
    )

    result[
        "ivol_direction"
    ] = result[
        "full_control_mean_ivol_ic"
    ].map(
        direction_label
    )

    result[
        "return_direction_consistent"
    ] = (

        result[
            "return_ic_direction"
        ]

        ==

        result[
            "fmb_beta_direction"
        ]
    )

    result[
        "strong_regime_consistency"
    ] = (

        result[
            "regime_alpha_strict_all_windows"
        ]
        .map(
            bool_value
        )

        |

        result[
            "regime_ivol_strict_all_windows"
        ]
        .map(
            bool_value
        )
    )

    keep_columns = [

        "factor_order",

        "factor_name",

        "feature_family",

        "evidence_role",

        "cost_profile",

        "full_control_mean_alpha_ic",

        "return_ic_direction",

        "m3_mean_beta_bps_per_1sd",

        "fmb_beta_direction",

        "return_direction_consistent",

        "full_control_mean_spread_bps",

        "full_control_mean_ivol_ic",

        "ivol_direction",

        "stable_risk_target_count_out_of_4",

        "regime_alpha_strict_all_windows",

        "regime_ivol_strict_all_windows",

        "mean_roundtrip_traded_notional_ratio",

        "mean_descriptive_break_even_cost_bps",

        "cost_magnitude_survival_windows_10bp",

        "strong_regime_consistency",
    ]

    return (
        result[
            keep_columns
        ]
        .sort_values(
            "factor_order"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 7. Cross-factor findings
# ============================================================

def build_key_findings(
    factor_summary,
):

    rows = []

    # --------------------------------------------------------
    # Return direction consistency
    # --------------------------------------------------------

    n_return_consistent = int(
        factor_summary[
            "return_direction_consistent"
        ]
        .sum()
    )

    rows.append(
        {
            "finding_id":
                "KF01",

            "topic":
                "Return consistency",

            "finding":
                (
                    f"{n_return_consistent} of "
                    f"{len(factor_summary)} factors have the "
                    "same sign for full-control mean Rank IC "
                    "and mean M3 Fama-MacBeth beta."
                ),

            "interpretation":
                (
                    "Rank-based and conditional-linear return "
                    "evidence can therefore be compared directly "
                    "for most factors, while disagreements should "
                    "be treated as mixed evidence."
                ),
        }
    )

    # --------------------------------------------------------
    # Risk robustness
    # --------------------------------------------------------

    n_all_risk_stable = int(
        (
            factor_summary[
                "stable_risk_target_count_out_of_4"
            ]
            ==
            4
        )
        .sum()
    )

    rows.append(
        {
            "finding_id":
                "KF02",

            "topic":
                "Risk robustness",

            "finding":
                (
                    f"{n_all_risk_stable} factors preserve the "
                    "same direction across W60/W120/W252 for "
                    "all four future-risk targets."
                ),

            "interpretation":
                (
                    "Risk evidence is assessed separately from "
                    "return evidence and is not reduced to a "
                    "single Alpha criterion."
                ),
        }
    )

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    n_high_cost = int(
        (
            factor_summary[
                "cost_profile"
            ]
            ==
            "HIGH_COST_SENSITIVITY"
        )
        .sum()
    )

    rows.append(
        {
            "finding_id":
                "KF03",

            "topic":
                "Transaction-cost sensitivity",

            "finding":
                (
                    f"{n_high_cost} of "
                    f"{len(factor_summary)} factors are classified "
                    "as HIGH_COST_SENSITIVITY under the fixed "
                    "Step-5 descriptive rule."
                ),

            "interpretation":
                (
                    "Statistical predictability and economic "
                    "implementability are treated as separate "
                    "dimensions."
                ),
        }
    )

    # --------------------------------------------------------
    # Level vs dynamic turnover
    # --------------------------------------------------------

    family_turnover = (

        factor_summary.groupby(
            "feature_family"
        )[
            "mean_roundtrip_traded_notional_ratio"
        ]
        .mean()
    )

    level_turnover = (
        family_turnover.get(
            "LEVEL_STRUCTURAL",
            np.nan,
        )
    )

    dynamic_turnover = (
        family_turnover.get(
            "DYNAMIC_1M",
            np.nan,
        )
    )

    rows.append(
        {
            "finding_id":
                "KF04",

            "topic":
                "Structural vs dynamic features",

            "finding":
                (
                    "Mean round-trip traded-notional ratio is "
                    f"{fmt_num(level_turnover, 3)} for level/"
                    "structural variables and "
                    f"{fmt_num(dynamic_turnover, 3)} for "
                    "one-month dynamic variables."
                ),

            "interpretation":
                (
                    "Short-horizon network dynamics require "
                    "more portfolio rebalancing on average."
                ),
        }
    )

    # --------------------------------------------------------
    # Role distribution
    # --------------------------------------------------------

    role_counts = (
        factor_summary[
            "evidence_role"
        ]
        .value_counts()
        .to_dict()
    )

    role_text = "; ".join(
        [
            f"{k}={v}"
            for k, v in (
                role_counts.items()
            )
        ]
    )

    rows.append(
        {
            "finding_id":
                "KF05",

            "topic":
                "Evidence-role distribution",

            "finding":
                role_text,

            "interpretation":
                (
                    "The final research output is a role-based "
                    "evidence map rather than a best-to-worst "
                    "factor ranking."
                ),
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 8. Markdown helpers
# ============================================================

def dataframe_to_markdown(
    df,
):

    if len(df) == 0:

        return "_No rows._"

    columns = list(
        df.columns
    )

    lines = []

    lines.append(
        "| "
        +
        " | ".join(
            columns
        )
        +
        " |"
    )

    lines.append(
        "|"
        +
        "|".join(
            [
                "---"
                for _ in columns
            ]
        )
        +
        "|"
    )

    for _, row in (
        df.iterrows()
    ):

        values = []

        for column in columns:

            value = row[
                column
            ]

            if pd.isna(
                value
            ):

                text = "NA"

            else:

                text = str(
                    value
                )

            text = (
                text
                .replace(
                    "|",
                    "\\|",
                )
                .replace(
                    "\n",
                    " "
                )
            )

            values.append(
                text
            )

        lines.append(
            "| "
            +
            " | ".join(
                values
            )
            +
            " |"
        )

    return "\n".join(
        lines
    )


# ============================================================
# 9. Final factor table for Markdown
# ============================================================

def build_markdown_factor_table(
    factor_summary,
):

    df = pd.DataFrame(
        {
            "Factor":
                factor_summary[
                    "factor_name"
                ],

            "Family":
                factor_summary[
                    "feature_family"
                ],

            "Evidence role":
                factor_summary[
                    "evidence_role"
                ],

            "Full-control mean IC":
                factor_summary[
                    "full_control_mean_alpha_ic"
                ].map(
                    lambda x:
                    fmt_num(
                        x,
                        4,
                    )
                ),

            "M3 FMB beta":
                factor_summary[
                    "m3_mean_beta_bps_per_1sd"
                ].map(
                    lambda x:
                    fmt_bp(
                        x,
                        2,
                    )
                    +
                    "/SD"
                ),

            "Mean spread":
                factor_summary[
                    "full_control_mean_spread_bps"
                ].map(
                    lambda x:
                    fmt_bp(
                        x,
                        2,
                    )
                ),

            "Mean IVOL IC":
                factor_summary[
                    "full_control_mean_ivol_ic"
                ].map(
                    lambda x:
                    fmt_num(
                        x,
                        4,
                    )
                ),

            "Stable risks":
                (
                    factor_summary[
                        "stable_risk_target_count_out_of_4"
                    ]
                    .astype(int)
                    .astype(str)
                    +
                    "/4"
                ),

            "Cost":
                factor_summary[
                    "cost_profile"
                ],
        }
    )

    return df


# ============================================================
# 10. Detailed factor narratives
# ============================================================

def build_factor_narratives(
    factor_summary,
):

    lines = []

    for row in (
        factor_summary.itertuples(
            index=False
        )
    ):

        alpha_ic = (
            row.full_control_mean_alpha_ic
        )

        fmb_beta = (
            row.m3_mean_beta_bps_per_1sd
        )

        spread = (
            row.full_control_mean_spread_bps
        )

        ivol = (
            row.full_control_mean_ivol_ic
        )

        stable_risk = int(
            row.stable_risk_target_count_out_of_4
        )

        regime_alpha = bool_value(
            row.regime_alpha_strict_all_windows
        )

        regime_ivol = bool_value(
            row.regime_ivol_strict_all_windows
        )

        turnover = (
            row.mean_roundtrip_traded_notional_ratio
        )

        break_even = (
            row.mean_descriptive_break_even_cost_bps
        )

        lines.append(
            f"### `{row.factor_name}`\n"
        )

        lines.append(
            f"- Structural family: "
            f"`{row.feature_family}`.\n"
        )

        lines.append(
            f"- Step-5 evidence role: "
            f"`{row.evidence_role}`; "
            f"cost profile: `{row.cost_profile}`.\n"
        )

        lines.append(
            f"- Full-control mean Alpha Rank IC = "
            f"{fmt_num(alpha_ic, 4)}; "
            f"M3 Fama–MacBeth mean coefficient = "
            f"{fmt_bp(fmb_beta, 2)}/SD; "
            f"mean full-control Q5−Q1 spread = "
            f"{fmt_bp(spread, 2)}.\n"
        )

        lines.append(
            f"- Mean future-IVOL Rank IC = "
            f"{fmt_num(ivol, 4)}; "
            f"{stable_risk}/4 risk targets preserve the "
            f"same sign across W60/W120/W252.\n"
        )

        lines.append(
            f"- Strict regime consistency: "
            f"Alpha={regime_alpha}, "
            f"IVOL={regime_ivol}.\n"
        )

        lines.append(
            f"- Mean round-trip traded-notional ratio = "
            f"{fmt_num(turnover, 3)}; "
            f"mean descriptive one-way break-even cost = "
            f"{fmt_bp(break_even, 2)}.\n\n"
        )

    return "".join(
        lines
    )


# ============================================================
# 11. Build integrated Markdown report
# ============================================================

def build_report(
    timeline,
    factor_summary,
    key_findings,
    metadata,
):

    factor_table = (
        build_markdown_factor_table(
            factor_summary
        )
    )

    timeline_md = dataframe_to_markdown(
        timeline[
            [
                "research_day",
                "stage",
                "main_question",
                "key_methodological_rule",
            ]
        ].rename(
            columns={
                "research_day":
                    "Day",

                "stage":
                    "Stage",

                "main_question":
                    "Research question",

                "key_methodological_rule":
                    "Guardrail",
            }
        )
    )

    factor_table_md = (
        dataframe_to_markdown(
            factor_table
        )
    )

    findings_md = (
        dataframe_to_markdown(
            key_findings[
                [
                    "topic",
                    "finding",
                    "interpretation",
                ]
            ].rename(
                columns={
                    "topic":
                        "Topic",

                    "finding":
                        "Finding",

                    "interpretation":
                        "Interpretation",
                }
            )
        )
    )

    factor_narratives = (
        build_factor_narratives(
            factor_summary
        )
    )

    report = f"""# M1 A-Share Stock Association Network
## Integrated Research Summary: Day 1–Day 7

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Step-6 design hash:** `{metadata["day7_step6_design_hash"]}`

---

# 1. Research Objective

The M1 project studies whether the cross-sectional position of A-share
stocks in rolling association networks contains economically and
statistically meaningful information about subsequent stock returns
and risks.

The research pipeline deliberately separates:

1. network construction;
2. network-feature engineering;
3. future-outcome construction;
4. return validation;
5. risk validation;
6. robustness and incremental-information tests;
7. turnover and transaction-cost diagnostics.

The central empirical question is therefore not merely whether a
network variable is correlated with a future outcome, but whether the
relation survives alternative market states, traditional stock
characteristics, conditional cross-sectional regressions, and economic
implementation constraints.

---

# 2. Core Research Discipline

The empirical pipeline follows four fixed restrictions:

- No future outcome is used to construct the current network or current
  network feature.
- The nine-factor set is frozen before the main future return and risk
  validation.
- Factor signs are not reversed after observing outcomes.
- W60, W120 and W252 are all retained; no optimal window is selected
  ex post.

Accordingly, the final robustness matrix is an **evidence map rather
than a factor-ranking exercise**.

---

# 3. Return Conventions

Two return definitions are intentionally kept separate.

For network construction, the project uses the validated network
return available only when adjacent market dates are both valid
trading observations.

For future holding-period labels, the stock holding return is

\\[
r^{{hold}}_{{i,s}}
=
\\begin{{cases}}
0, & \\text{{stock suspended on day }}s,\\\\
r^{{last-trade}}_{{i,s}}, & \\text{{stock trades with a valid last-trade return}},\\\\
NA, & \\text{{otherwise}}.
\\end{{cases}}
\\]

This allows suspended positions to remain marked at stale value while
the first reopening observation captures the full return accumulated
since the last actual trade.

---

# 4. Day 1–Day 7 Research Pipeline

{timeline_md}

---

# 5. Day-6 Forward Validation Framework

The primary Alpha target is future excess return over the interval

\\[
(analysis\\_date, next\\_analysis\\_date].
\\]

The initial cross-sectional Rank IC is

\\[
IC_{{t,W,f}}
=
Corr_S
\\left(
F_{{i,t,W,f}},
Y_{{i,t,W}}
\\right).
\\]

Portfolio validation uses five equal-weight portfolios, where Q1
contains the lowest factor values and Q5 the highest. The canonical
spread is always

\\[
R_{{Q5}}-R_{{Q1}},
\\]

and negative factors are not flipped after observing outcomes.

Risk validation considers:

- future realized volatility;
- future downside volatility;
- future maximum drawdown;
- future idiosyncratic volatility.

Industry and size neutralization is implemented cross-sectionally
using contemporaneous information.

---

# 6. Day-7 Robustness Chain

Day 7 extends the initial validation through four increasingly strict
tests.

## 6.1 Regime robustness

Relations are evaluated across:

- 2015–2018;
- 2019–2022;
- 2023–2026;
- Bull / Bear states;
- High / Low volatility states;
- their joint regimes.

Regimes are defined only using information available by the current
analysis date.

## 6.2 Traditional stock characteristics

The full specification controls for:

\\[
Industry
+
Size
+
Momentum
+
Reversal
+
PastVolatility
+
Turnover.
\\]

The main comparison is

\\[
Industry+Size
\\rightarrow
FullControls.
\\]

A common stock sample is used so that changes in the estimated signal
cannot be attributed merely to changes in cross-sectional coverage.

## 6.3 Fama–MacBeth regression

For every factor separately,

\\[
R_{{i,t+1}}
=
\\alpha_{{Industry(i,t)}}
+
\\beta_t F_{{i,t}}
+
\\gamma_t^\\top X_{{i,t}}
+
\\varepsilon_{{i,t+1}}.
\\]

The final network coefficient is

\\[
\\bar\\beta
=
\\frac1T
\\sum_{{t=1}}^T
\\hat\\beta_t,
\\]

with HAC inference applied to the monthly coefficient sequence.

## 6.4 Turnover and transaction costs

For target portfolio weights \\(w_{{i,t}}\\), the pre-trade weight is

\\[
w^-_{{i,t}}
=
\\frac{{
w_{{i,t-1}}
(1+R_{{i,t-1\\rightarrow t}})
}}{{
\\sum_j
w_{{j,t-1}}
(1+R_{{j,t-1\\rightarrow t}})
}}.
\\]

The drift-adjusted half-\\(L_1\\) turnover is

\\[
TO_t
=
\\frac12
\\sum_i
|w_{{i,t}}-w^-_{{i,t}}|.
\\]

For one-way transaction cost \\(c\\), the cost drag of the two-leg
Q5−Q1 portfolio is

\\[
Cost_t
=
2c
\\left(
TO_{{Q5,t}}
+
TO_{{Q1,t}}
\\right).
\\]

The project reports 5, 10, 20 and 30 bp scenarios.

---

# 7. Final Factor Robustness Matrix

{factor_table_md}

The table summarizes return information, conditional Fama–MacBeth
coefficients, future-IVOL information, cross-risk stability and
transaction-cost sensitivity. These dimensions should be interpreted
jointly rather than reduced to a single score.

---

# 8. Cross-Factor Findings

{findings_md}

---

# 9. Factor-Level Evidence Profiles

{factor_narratives}

---

# 10. Main Integrated Interpretation

The complete Day 1–7 evidence shows that stock-network characteristics
contain heterogeneous information.

A recurring pattern is that several level or structural network
variables preserve forward-risk information after controlling for
industry, size, momentum, short-term reversal, historical volatility
and turnover.

By contrast, a substantial portion of the initial return predictability
of several network variables is attenuated once traditional stock
characteristics are introduced. This means the empirical results do
not support treating every network characteristic as an independent
Alpha factor.

The distinction between return information and risk information is
therefore central to the final interpretation.

In particular, the project should distinguish:

\\[
\\text{{predictive return information}}
\\neq
\\text{{predictive risk information}}
\\neq
\\text{{tradable strategy performance}}.
\\]

The transaction-cost exercise additionally shows that short-horizon
dynamic network variables can require substantially greater
rebalancing than slower-moving structural characteristics.

---

# 11. Interpretation of Statistical Evidence

HAC \\(t\\)-statistics are reported as exploratory nominal diagnostics.

The project has evaluated multiple:

- factors;
- network windows;
- targets;
- portfolio specifications;
- regimes.

Therefore a nominal threshold such as

\\[
|t|\\ge1.96
\\]

must not be interpreted as a multiple-testing-adjusted discovery rule.

Similarly, a QA PASS means that the computational design and internal
consistency checks succeeded. It does **not** imply that an economic
or statistical hypothesis has been established.

---

# 12. Important Limitations

Several limitations should remain explicit.

First, all current results are descriptive predictive associations.
They do not establish causality between network position and future
stock outcomes.

Second, the association network is based on rolling historical return
dependence. Residual correlation should not be interpreted as
conditional independence.

Third, longer rolling windows mechanically overlap more strongly,
which can contribute to persistence in network features.

Fourth, transaction-cost calculations are stylized equal-weight
diagnostics. They do not yet include all real trading frictions such as
market impact, price limits, liquidity constraints, short-sale
availability or implementation delay.

Fifth, the current traditional controls are intentionally limited to
variables whose point-in-time construction is well defined in the
current database. Additional accounting characteristics require a
separate PIT-data validation stage.

---

# 13. Current Research Conclusion

The Day 1–7 pipeline supports a more nuanced conclusion than a simple
"network Alpha" interpretation.

The empirical evidence suggests that network position and network
reorganization can contain information about subsequent stock
outcomes, but the strongest and most persistent component is often
associated with future risk rather than independent return Alpha.

After controlling for traditional stock characteristics, some return
relations remain modest and directionally stable, while several
future-risk relations remain considerably more persistent.

Accordingly, a natural continuation of M1 is to investigate whether

\\[
\\boxed{{
\\text{{stock-network structure contains incremental forward-looking
idiosyncratic-risk information}}
}}
\\]

and to study the economic mechanism behind that relation.

---

# 14. Suggested Next Research Directions

The next phase should not simply add more network variables. More
informative extensions include:

1. genuine out-of-sample or expanding-window validation;
2. deeper investigation of idiosyncratic-risk mechanisms;
3. interaction between network characteristics and liquidity,
   size or market stress;
4. portfolio applications designed ex ante rather than obtained by
   reversing an observed in-sample factor sign;
5. alternative network estimators as robustness checks;
6. formal multiple-testing or false-discovery control if a larger
   factor search is resumed.

---

# 15. Reproducibility and Governance

The final Day-7 summary inherits only outputs whose formal upstream QA
passed.

No factor was removed in Step 6.

No factor orientation was changed.

No network window was selected.

No additional statistical test was introduced during report
generation.

The report is therefore a deterministic summary of the completed
Day 1–7 research pipeline.
"""

    return report


# ============================================================
# 12. Formal QA
# ============================================================

def formal_qa(
    factor_matrix,
    window_matrix,
    factor_summary,
    timeline,
    upstream_metadata,
):

    qa = {}

    qa[
        "factor_count"
    ] = int(
        len(
            factor_summary
        )
    )

    qa[
        "factor_count_ok"
    ] = bool(
        len(
            factor_summary
        )
        ==
        EXPECTED_FACTOR_COUNT
    )

    qa[
        "factor_window_row_count"
    ] = int(
        len(
            window_matrix
        )
    )

    qa[
        "factor_window_row_count_ok"
    ] = bool(
        len(
            window_matrix
        )
        ==
        EXPECTED_FACTOR_COUNT
        *
        len(
            EXPECTED_WINDOWS
        )
    )

    qa[
        "timeline_day_count"
    ] = int(
        timeline[
            "research_day"
        ]
        .nunique()
    )

    qa[
        "timeline_complete_day1_day7"
    ] = bool(
        sorted(
            timeline[
                "research_day"
            ]
            .unique()
            .tolist()
        )
        ==
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]
    )

    qa[
        "factor_names_unique"
    ] = bool(
        not factor_summary[
            "factor_name"
        ]
        .duplicated()
        .any()
    )

    qa[
        "factor_window_keys_unique"
    ] = bool(
        not window_matrix[
            [
                "factor_name",
                "window",
            ]
        ]
        .duplicated()
        .any()
    )

    qa[
        "feature_family_complete"
    ] = bool(
        factor_summary[
            "feature_family"
        ]
        .notna()
        .all()
    )

    qa[
        "all_parent_qa_pass"
    ] = bool(
        all(
            meta.get(
                "formal_qa",
                {}
            ).get(
                "all_formal_qa_pass"
            )
            is True
            for meta in (
                upstream_metadata.values()
            )
        )
    )

    # --------------------------------------------------------
    # Verify Step-5 governance flags
    # --------------------------------------------------------

    governance_columns = [

        "factor_selected",

        "factor_removed",

        "factor_sign_flipped",

        "window_selected",
    ]

    governance_ok = True

    for column in governance_columns:

        if (
            column
            in
            factor_matrix.columns
        ):

            if (
                factor_matrix[
                    column
                ]
                .map(
                    bool_value
                )
                .any()
            ):

                governance_ok = False

    qa[
        "step5_no_expost_selection_or_sign_flip"
    ] = bool(
        governance_ok
    )

    qa[
        "step6_new_statistical_test_performed"
    ] = False

    qa[
        "step6_factor_selection_performed"
    ] = False

    qa[
        "step6_factor_sign_flip_performed"
    ] = False

    qa[
        "step6_window_selection_performed"
    ] = False

    required = [

        "factor_count_ok",

        "factor_window_row_count_ok",

        "timeline_complete_day1_day7",

        "factor_names_unique",

        "factor_window_keys_unique",

        "feature_family_complete",

        "all_parent_qa_pass",

        "step5_no_expost_selection_or_sign_flip",
    ]

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            qa[
                key
            ]
            for key in required
        )
    )

    return qa


# ============================================================
# 13. Main
# ============================================================

def main():

    print("=" * 80)

    print(
        "M1 A-Share Full-Market Stock Network"
    )

    print(
        "Day 7 - Step 6"
    )

    print(
        "Integrated Research Summary"
    )

    print("=" * 80)

    # ========================================================
    # Parent QA
    # ========================================================

    print()
    print(
        "[1] Validate upstream research chain"
    )

    upstream_metadata = (
        validate_upstream()
    )

    # ========================================================
    # Final evidence
    # ========================================================

    print()
    print(
        "[2] Load final robustness evidence"
    )

    (
        factor_matrix,
        window_matrix,
        role_summary,
    ) = load_final_evidence()

    # ========================================================
    # Timeline
    # ========================================================

    print()
    print(
        "[3] Build Day 1-Day 7 research timeline"
    )

    timeline = (
        build_research_timeline()
    )

    save_csv_atomic(
        timeline,
        TIMELINE_PATH,
    )

    # ========================================================
    # Integrated factor summary
    # ========================================================

    print()
    print(
        "[4] Build integrated factor summary"
    )

    factor_summary = (
        build_factor_summary(
            factor_matrix
        )
    )

    save_csv_atomic(
        factor_summary,
        FACTOR_SUMMARY_PATH,
    )

    # ========================================================
    # Key findings
    # ========================================================

    print()
    print(
        "[5] Build cross-factor findings"
    )

    key_findings = (
        build_key_findings(
            factor_summary
        )
    )

    save_csv_atomic(
        key_findings,
        KEY_FINDINGS_PATH,
    )

    # ========================================================
    # Formal QA before report
    # ========================================================

    print()
    print(
        "[6] Run Step-6 formal QA"
    )

    qa = formal_qa(

        factor_matrix=
            factor_matrix,

        window_matrix=
            window_matrix,

        factor_summary=
            factor_summary,

        timeline=
            timeline,

        upstream_metadata=
            upstream_metadata,
    )

    qa_df = pd.DataFrame(
        [
            {
                "qa_name":
                    key,

                "qa_value":
                    value,
            }
            for key, value
            in qa.items()
        ]
    )

    save_csv_atomic(
        qa_df,
        QA_PATH,
    )

    if (
        not qa[
            "all_formal_qa_pass"
        ]
    ):

        raise RuntimeError(
            "Day-7 Step-6 formal QA failed.\n"
            f"{qa}"
        )

    # ========================================================
    # Freeze summary design
    # ========================================================

    design_payload = {

        "research_day":
            7,

        "step":
            "Step6_Integrated_Research_Summary",

        "parent_design_hashes":
            {

                "day6_step6":
                    upstream_metadata[
                        "day6_step6"
                    ].get(
                        "step6_design_hash"
                    ),

                "day7_step1":
                    upstream_metadata[
                        "day7_step1"
                    ].get(
                        "day7_step1_design_hash"
                    ),

                "day7_step2":
                    upstream_metadata[
                        "day7_step2"
                    ].get(
                        "day7_step2_design_hash"
                    ),

                "day7_step3":
                    upstream_metadata[
                        "day7_step3"
                    ].get(
                        "day7_step3_design_hash"
                    ),

                "day7_step4":
                    upstream_metadata[
                        "day7_step4"
                    ].get(
                        "day7_step4_design_hash"
                    ),

                "day7_step5":
                    upstream_metadata[
                        "day7_step5"
                    ].get(
                        "day7_step5_design_hash"
                    ),
            },

        "factor_count":
            EXPECTED_FACTOR_COUNT,

        "windows":
            EXPECTED_WINDOWS,

        "report_scope":
            "M1 Day1-Day7",

        "new_statistical_test":
            False,

        "factor_ranking":
            False,

        "factor_selection":
            False,

        "factor_removal":
            False,

        "factor_sign_flip":
            False,

        "window_selection":
            False,

        "report_type":
            (
                "Deterministic evidence integration "
                "and research documentation."
            ),
    }

    design_hash = (
        canonical_hash(
            design_payload
        )
    )

    metadata = {

        **design_payload,

        "day7_step6_design_hash":
            design_hash,

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "formal_qa":
            qa,

        "outputs":
            {

                "integrated_report":
                    str(
                        REPORT_PATH
                    ),

                "factor_summary":
                    str(
                        FACTOR_SUMMARY_PATH
                    ),

                "research_timeline":
                    str(
                        TIMELINE_PATH
                    ),

                "key_findings":
                    str(
                        KEY_FINDINGS_PATH
                    ),

                "qa":
                    str(
                        QA_PATH
                    ),
            },

        "interpretation_rules":
            [

                (
                    "Return, risk and tradability are "
                    "reported as distinct dimensions."
                ),

                (
                    "Nominal HAC significance is not treated "
                    "as a multiple-testing-adjusted discovery."
                ),

                (
                    "Negative Q5-Q1 spreads are not reversed "
                    "into ex-post long-short strategies."
                ),

                (
                    "The Step-5 evidence_role field is a "
                    "descriptive classification rather than "
                    "a factor-ranking criterion."
                ),

                (
                    "The Step-6 report introduces no new "
                    "empirical test."
                ),
            ],
    }

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        "[7] Generate integrated Markdown report"
    )

    report = build_report(

        timeline=
            timeline,

        factor_summary=
            factor_summary,

        key_findings=
            key_findings,

        metadata={
            **metadata,
        },
    )

    save_text_atomic(
        report,
        REPORT_PATH,
    )

    save_json_atomic(
        metadata,
        METADATA_PATH,
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Integrated Factor Summary"
    )

    print("=" * 80)

    display_columns = [

        "factor_name",

        "feature_family",

        "evidence_role",

        "cost_profile",

        "full_control_mean_alpha_ic",

        "m3_mean_beta_bps_per_1sd",

        "full_control_mean_ivol_ic",

        "stable_risk_target_count_out_of_4",

        "mean_roundtrip_traded_notional_ratio",
    ]

    print(
        factor_summary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)

    print(
        "Step-6 Formal QA"
    )

    print("=" * 80)

    print(
        f"Factor count: "
        f"{qa['factor_count']}"
    )

    print(
        f"Factor-window rows: "
        f"{qa['factor_window_row_count']}"
    )

    print(
        f"Day1-Day7 timeline complete: "
        f"{qa['timeline_complete_day1_day7']}"
    )

    print(
        f"All parent QA pass: "
        f"{qa['all_parent_qa_pass']}"
    )

    print(
        f"No ex-post selection/sign flip: "
        f"{qa['step5_no_expost_selection_or_sign_flip']}"
    )

    print(
        f"Formal QA pass: "
        f"{qa['all_formal_qa_pass']}"
    )

    print()
    print(
        "Integrated report:"
    )

    print(
        REPORT_PATH
    )

    print()
    print("=" * 80)

    print(
        "Day 7 Step 6 Complete"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
DAY11 = ROOT / "output" / "M1_day11"

STEP1 = (
    DAY11
    / "01_stage1_frozen_prediction_panel"
)

STEP2 = (
    DAY11
    / "02_stage2_factor_redundancy"
)

STEP3 = (
    DAY11
    / "03_stage3_strict_oos_forecasting"
)

STEP4 = (
    DAY11
    / "04_stage4_incremental_oos_metrics"
)

STEP5 = (
    DAY11
    / "05_stage5_oos_temporal_block_validation"
)

OUTDIR = (
    DAY11
    / "06_stage6_joint_oos_validation_summary"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Upstream QA
# ------------------------------------------------------------

STEP1_QA = (
    STEP1
    / "day11_step1_qa.csv"
)

STEP2_QA = (
    STEP2
    / "day11_step2_qa.csv"
)

STEP3_QA = (
    STEP3
    / "day11_step3_qa.csv"
)

STEP4_QA = (
    STEP4
    / "day11_step4_qa.csv"
)

STEP5_QA = (
    STEP5
    / "day11_step5_qa.csv"
)


# ------------------------------------------------------------
# Main result inputs
# ------------------------------------------------------------

REDUNDANCY_PATH = (
    STEP2
    / "multivariate_redundancy_summary.csv"
)

OOS_PATH = (
    STEP4
    / "oos_incremental_risk_summary.csv"
)

TEMPORAL_PATH = (
    STEP5
    / "temporal_oos_summary.csv"
)

BLOCK_PATH = (
    STEP5
    / "block_oos_summary.csv"
)


# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

JOINT_OUT = (
    OUTDIR
    / "day11_joint_oos_validation_summary.csv"
)

BLOCK_OUT = (
    OUTDIR
    / "day11_block_validation_summary.csv"
)

REDUNDANCY_OUT = (
    OUTDIR
    / "day11_redundancy_validation_summary.csv"
)

QA_OUT = (
    OUTDIR
    / "day11_step6_qa.csv"
)

META_OUT = (
    OUTDIR
    / "day11_step6_metadata.json"
)

REPORT_OUT = (
    OUTDIR
    / "day11_joint_oos_validation_report.md"
)


# ============================================================
# 1. Frozen design
# ============================================================

WINDOWS = [60, 120, 252]

SCOPES = [
    "PRIMARY",
    "FULL_RANK",
]

BLOCKS = [
    "LEVEL_POSITION",
    "CROSS_INDUSTRY_COMPOSITION",
    "RECONFIGURATION",
    "NEIGHBOR_STABILITY",
]

FROZEN_FACTOR_COUNT = 9

TOL = 1e-10


# ============================================================
# 2. Helpers
# ============================================================

def save_csv(df, path):

    tmp = Path(
        str(path) + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        tmp,
        path,
    )


def as_bool(x):

    return (
        str(x)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def load_qa(path):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    q = pd.read_csv(
        path
    )

    required = {
        "qa_name",
        "qa_value",
    }

    if not required.issubset(
        q.columns
    ):
        raise RuntimeError(
            f"Invalid QA file: {path}"
        )

    return dict(
        zip(
            q["qa_name"],
            q["qa_value"],
        )
    )


def require_columns(
    df,
    cols,
    name,
):

    missing = [
        c for c in cols
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name} missing columns: "
            f"{missing}"
        )


def pct(x):

    if pd.isna(x):
        return "NA"

    return f"{100 * float(x):.1f}%"


def num(x, digits=4):

    if pd.isna(x):
        return "NA"

    return f"{float(x):.{digits}f}"


# ============================================================
# 3. Upstream QA
# ============================================================

def validate_upstream():

    paths = {
        "step1":
            STEP1_QA,

        "step2":
            STEP2_QA,

        "step3":
            STEP3_QA,

        "step4":
            STEP4_QA,

        "step5":
            STEP5_QA,
    }

    result = {}

    for name, path in paths.items():

        qa = load_qa(
            path
        )

        passed = as_bool(
            qa.get(
                "all_formal_qa_pass",
                False,
            )
        )

        result[
            f"{name}_formal_qa_pass"
        ] = passed

        if not passed:

            raise RuntimeError(
                f"{name} formal QA "
                "has not passed."
            )

    return result


# ============================================================
# 4. Redundancy summary
# ============================================================

def build_redundancy_summary():

    x = pd.read_csv(
        REDUNDANCY_PATH
    )

    required = [
        "sample_scope",
        "window",
        "date_n",
        "median_condition_number",
        "mean_pc1_variance_share",
        "median_pc3_cumulative_share",
        "median_pc_n_80pct",
        "median_pc_n_90pct",
        "median_effective_rank",
    ]

    require_columns(
        x,
        required,
        "Step-2 redundancy summary",
    )

    x["window"] = (
        pd.to_numeric(
            x["window"],
            errors="raise",
        )
        .astype(int)
    )

    expected_scopes = {
        "NATIVE",
        "COMMON_DATE",
    }

    actual_scopes = set(
        x[
            "sample_scope"
        ].unique()
    )

    if actual_scopes != expected_scopes:

        raise RuntimeError(
            "Unexpected redundancy scopes: "
            f"{actual_scopes}"
        )

    if (
        x[
            [
                "sample_scope",
                "window",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Step-2 "
            "scope-window rows."
        )

    native = (
        x[
            x[
                "sample_scope"
            ] == "NATIVE"
        ]
        .copy()
    )

    common = (
        x[
            x[
                "sample_scope"
            ] == "COMMON_DATE"
        ]
        .copy()
    )

    native = native[
        [
            "window",
            "date_n",
            "median_condition_number",
            "mean_pc1_variance_share",
            "median_pc3_cumulative_share",
            "median_pc_n_80pct",
            "median_pc_n_90pct",
            "median_effective_rank",
        ]
    ].rename(
        columns={
            "date_n":
                "redundancy_native_date_n",

            "median_condition_number":
                "redundancy_native_condition_number",

            "mean_pc1_variance_share":
                "redundancy_native_pc1_share",

            "median_pc3_cumulative_share":
                "redundancy_native_pc3_cumulative_share",

            "median_pc_n_80pct":
                "redundancy_native_pc_n_80pct",

            "median_pc_n_90pct":
                "redundancy_native_pc_n_90pct",

            "median_effective_rank":
                "redundancy_native_effective_rank",
        }
    )

    common = common[
        [
            "window",
            "date_n",
            "median_condition_number",
            "mean_pc1_variance_share",
            "median_pc3_cumulative_share",
            "median_pc_n_80pct",
            "median_pc_n_90pct",
            "median_effective_rank",
        ]
    ].rename(
        columns={
            "date_n":
                "redundancy_common_date_n",

            "median_condition_number":
                "redundancy_common_condition_number",

            "mean_pc1_variance_share":
                "redundancy_common_pc1_share",

            "median_pc3_cumulative_share":
                "redundancy_common_pc3_cumulative_share",

            "median_pc_n_80pct":
                "redundancy_common_pc_n_80pct",

            "median_pc_n_90pct":
                "redundancy_common_pc_n_90pct",

            "median_effective_rank":
                "redundancy_common_effective_rank",
        }
    )

    out = native.merge(
        common,
        on="window",
        how="inner",
        validate="one_to_one",
    )

    out = (
        out.sort_values(
            "window"
        )
        .reset_index(
            drop=True
        )
    )

    return out


# ============================================================
# 5. Load Step-4 OOS summary
# ============================================================

def load_oos_summary():

    x = pd.read_csv(
        OOS_PATH
    )

    required = [
        "sample_scope",
        "window",
        "date_n",
        "evaluation_row_n",
        "first_evaluation_date",
        "last_evaluation_date",
        "mean_baseline_ic",
        "mean_network_ic",
        "mean_delta_ic",
        "median_delta_ic",
        "positive_delta_ic_month_share",
        "positive_delta_ic_year_share",
        "year_n",
        "mean_delta_rank_mse",
        "relative_oos_r2_rank",
        "mean_delta_rank_mae",
        "mean_delta_top_bottom_spread",
        "hac_t_delta_ic",
        "hac_p_gt0_delta_ic",
    ]

    require_columns(
        x,
        required,
        "Step-4 OOS summary",
    )

    x["window"] = pd.to_numeric(
        x["window"],
        errors="raise",
    ).astype(int)

    if (
        x[
            [
                "sample_scope",
                "window",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Step-4 "
            "scope-window rows."
        )

    return x


# ============================================================
# 6. Load temporal summary
# ============================================================

def load_temporal_summary():

    x = pd.read_csv(
        TEMPORAL_PATH
    )

    required = [
        "sample_scope",
        "window",
        "year_n",
        "mean_annual_delta_ic",
        "median_annual_delta_ic",
        "positive_year_share",
        "worst_year",
        "worst_year_delta_ic",
        "best_year",
        "best_year_delta_ic",
    ]

    require_columns(
        x,
        required,
        "Step-5 temporal summary",
    )

    x["window"] = pd.to_numeric(
        x["window"],
        errors="raise",
    ).astype(int)

    if (
        x[
            [
                "sample_scope",
                "window",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Step-5 temporal rows."
        )

    return x


# ============================================================
# 7. Load and reshape block summary
# ============================================================

def build_block_wide():

    x = pd.read_csv(
        BLOCK_PATH
    )

    required = [
        "sample_scope",
        "window",
        "block_name",
        "removed_factor_n",
        "date_n",
        "year_n",
        "mean_all_vs_baseline_delta_ic",
        "mean_without_block_vs_baseline_delta_ic",
        "mean_block_marginal_ic",
        "median_block_marginal_ic",
        "positive_block_month_share",
        "positive_block_year_share",
        "hac_t_block_marginal_ic",
        "hac_p_gt0_block_marginal_ic",
        "mean_block_marginal_mse_gain",
        "mean_block_marginal_spread_gain",
        "lobo_full_rank_date_share",
    ]

    require_columns(
        x,
        required,
        "Step-5 block summary",
    )

    x["window"] = pd.to_numeric(
        x["window"],
        errors="raise",
    ).astype(int)

    if (
        x[
            [
                "sample_scope",
                "window",
                "block_name",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Step-5 block rows."
        )

    actual_blocks = set(
        x[
            "block_name"
        ].unique()
    )

    if actual_blocks != set(
        BLOCKS
    ):

        raise RuntimeError(
            "Unexpected block names: "
            f"{actual_blocks}"
        )

    # -----------------------------------------------
    # Keep a clean long-form table
    # -----------------------------------------------

    long_out = (
        x.sort_values(
            [
                "sample_scope",
                "window",
                "block_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------
    # Wide form for the 6-row joint summary
    # -----------------------------------------------

    metrics = [
        "mean_block_marginal_ic",
        "median_block_marginal_ic",
        "positive_block_month_share",
        "positive_block_year_share",
        "mean_block_marginal_mse_gain",
        "mean_block_marginal_spread_gain",
        "hac_t_block_marginal_ic",
        "hac_p_gt0_block_marginal_ic",
    ]

    keys = [
        "sample_scope",
        "window",
    ]

    wide = (
        x[keys]
        .drop_duplicates()
        .copy()
    )

    for block in BLOCKS:

        sub = (
            x[
                x[
                    "block_name"
                ] == block
            ][
                keys + metrics
            ]
            .copy()
        )

        prefix = (
            block
            .lower()
        )

        sub = sub.rename(
            columns={
                m:
                    f"{prefix}_{m}"
                for m in metrics
            }
        )

        wide = wide.merge(
            sub,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    return (
        long_out,
        wide,
    )


# ============================================================
# 8. Build joint 6-row summary
# ============================================================

def build_joint_summary(
    oos,
    temporal,
    redundancy,
    block_wide,
):

    x = oos.merge(
        temporal,
        on=[
            "sample_scope",
            "window",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_temporal",
        ),
    )

    x = x.merge(
        redundancy,
        on="window",
        how="left",
        validate="many_to_one",
    )

    x = x.merge(
        block_wide,
        on=[
            "sample_scope",
            "window",
        ],
        how="left",
        validate="one_to_one",
    )

    scope_order = {
        "PRIMARY": 1,
        "FULL_RANK": 2,
    }

    x["_scope_order"] = (
        x[
            "sample_scope"
        ]
        .map(
            scope_order
        )
    )

    x = (
        x.sort_values(
            [
                "_scope_order",
                "window",
            ]
        )
        .drop(
            columns=[
                "_scope_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# 9. Cross-step consistency checks
# ============================================================

def consistency_checks(
    oos,
    temporal,
    block_long,
):

    # --------------------------------------------------------
    # Step 4 vs Step 5 temporal positive-year share
    # --------------------------------------------------------

    t = oos[
        [
            "sample_scope",
            "window",
            "positive_delta_ic_year_share",
            "year_n",
        ]
    ].merge(
        temporal[
            [
                "sample_scope",
                "window",
                "positive_year_share",
                "year_n",
            ]
        ],
        on=[
            "sample_scope",
            "window",
        ],
        how="inner",
        suffixes=(
            "_step4",
            "_step5",
        ),
        validate="one_to_one",
    )

    temporal_share_error = float(
        np.max(
            np.abs(
                t[
                    "positive_delta_ic_year_share"
                ]
                -
                t[
                    "positive_year_share"
                ]
            )
        )
    )

    temporal_year_n_match = bool(
        (
            t[
                "year_n_step4"
            ]
            ==
            t[
                "year_n_step5"
            ]
        ).all()
    )

    # --------------------------------------------------------
    # Step 4 total delta IC vs Step 5 repeated total delta IC
    # --------------------------------------------------------

    b = block_long.merge(
        oos[
            [
                "sample_scope",
                "window",
                "mean_delta_ic",
            ]
        ],
        on=[
            "sample_scope",
            "window",
        ],
        how="left",
        validate="many_to_one",
    )

    block_total_error = float(
        np.max(
            np.abs(
                b[
                    "mean_all_vs_baseline_delta_ic"
                ]
                -
                b[
                    "mean_delta_ic"
                ]
            )
        )
    )

    return {
        "step4_step5_positive_year_share_max_abs_error":
            temporal_share_error,

        "step4_step5_year_n_match":
            temporal_year_n_match,

        "step4_step5_total_delta_ic_max_abs_error":
            block_total_error,

        "step4_step5_temporal_consistency_pass":
            bool(
                temporal_share_error
                <= TOL
                and
                temporal_year_n_match
            ),

        "step4_step5_block_total_consistency_pass":
            bool(
                block_total_error
                <= TOL
            ),
    }


# ============================================================
# 10. Descriptive result flags
# ============================================================

def descriptive_flags(
    joint,
):

    primary = joint[
        joint[
            "sample_scope"
        ] == "PRIMARY"
    ]

    full_rank = joint[
        joint[
            "sample_scope"
        ] == "FULL_RANK"
    ]

    # IMPORTANT:
    # These are RESULT DESCRIPTORS.
    # They are NOT part of formal QA pass/fail.
    return {
        "primary_delta_ic_positive_all_windows":
            bool(
                (
                    primary[
                        "mean_delta_ic"
                    ] > 0
                ).all()
            ),

        "full_rank_delta_ic_positive_all_windows":
            bool(
                (
                    full_rank[
                        "mean_delta_ic"
                    ] > 0
                ).all()
            ),

        "primary_relative_oos_r2_positive_all_windows":
            bool(
                (
                    primary[
                        "relative_oos_r2_rank"
                    ] > 0
                ).all()
            ),

        "full_rank_relative_oos_r2_positive_all_windows":
            bool(
                (
                    full_rank[
                        "relative_oos_r2_rank"
                    ] > 0
                ).all()
            ),

        "primary_positive_year_share_gt_half_all_windows":
            bool(
                (
                    primary[
                        "positive_year_share"
                    ] > 0.5
                ).all()
            ),

        "full_rank_positive_year_share_gt_half_all_windows":
            bool(
                (
                    full_rank[
                        "positive_year_share"
                    ] > 0.5
                ).all()
            ),
    }


# ============================================================
# 11. Markdown report
# ============================================================

def write_report(
    joint,
    redundancy,
    block_long,
    qa,
    flags,
):

    lines = []

    lines.append(
        "# Day 11 — Joint OOS Validation Summary"
    )

    lines.append("")

    lines.append(
        "## 1. Research objective"
    )

    lines.append("")

    lines.append(
        "Day 11 evaluates whether the nine frozen "
        "network characteristics provide incremental "
        "strict out-of-sample information for future "
        "idiosyncratic volatility beyond the frozen "
        "traditional-characteristic baseline."
    )

    lines.append("")

    lines.append(
        "No model is refitted in Step 6. "
        "All statistics below are assembled from the "
        "frozen outputs of Steps 2, 4, and 5."
    )

    lines.append("")

    # --------------------------------------------------------
    # Redundancy
    # --------------------------------------------------------

    lines.append(
        "## 2. Frozen factor redundancy"
    )

    lines.append("")

    lines.append(
        "| Window | Native effective rank | "
        "Common-date effective rank | "
        "Native PC1 share | Native condition number |"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|"
    )

    for _, r in redundancy.iterrows():

        lines.append(
            f"| {int(r['window'])} "
            f"| {num(r['redundancy_native_effective_rank'], 3)} "
            f"| {num(r['redundancy_common_effective_rank'], 3)} "
            f"| {pct(r['redundancy_native_pc1_share'])} "
            f"| {num(r['redundancy_native_condition_number'], 1)} |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Overall OOS
    # --------------------------------------------------------

    lines.append(
        "## 3. Strict OOS incremental performance"
    )

    lines.append("")

    lines.append(
        "| Scope | W | Baseline IC | Network IC | "
        "Delta IC | Positive months | Positive years | "
        "Relative OOS R² |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, r in joint.iterrows():

        lines.append(
            f"| {r['sample_scope']} "
            f"| {int(r['window'])} "
            f"| {num(r['mean_baseline_ic'])} "
            f"| {num(r['mean_network_ic'])} "
            f"| {num(r['mean_delta_ic'])} "
            f"| {pct(r['positive_delta_ic_month_share'])} "
            f"| {pct(r['positive_year_share'])} "
            f"| {pct(r['relative_oos_r2_rank'])} |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Temporal stability
    # --------------------------------------------------------

    lines.append(
        "## 4. Temporal stability"
    )

    lines.append("")

    lines.append(
        "| Scope | W | Years | Positive-year share | "
        "Worst year | Worst-year Delta IC | "
        "Best year | Best-year Delta IC |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, r in joint.iterrows():

        lines.append(
            f"| {r['sample_scope']} "
            f"| {int(r['window'])} "
            f"| {int(r['year_n'])} "
            f"| {pct(r['positive_year_share'])} "
            f"| {int(r['worst_year'])} "
            f"| {num(r['worst_year_delta_ic'])} "
            f"| {int(r['best_year'])} "
            f"| {num(r['best_year_delta_ic'])} |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Block contributions
    # --------------------------------------------------------

    lines.append(
        "## 5. Conditional leave-one-block-out contributions"
    )

    lines.append("")

    lines.append(
        "Positive marginal IC means that removing the "
        "block lowers OOS IC conditional on the remaining "
        "network blocks. Contributions are not additive."
    )

    lines.append("")

    lines.append(
        "| Scope | W | Block | Marginal IC | "
        "Positive months | Positive years | MSE gain |"
    )

    lines.append(
        "|---|---:|---|---:|---:|---:|---:|"
    )

    block_order = {
        b: i
        for i, b in enumerate(
            BLOCKS
        )
    }

    bshow = block_long.copy()

    bshow["_order"] = (
        bshow[
            "block_name"
        ].map(
            block_order
        )
    )

    scope_order = {
        "PRIMARY": 1,
        "FULL_RANK": 2,
    }

    bshow["_scope"] = (
        bshow[
            "sample_scope"
        ].map(
            scope_order
        )
    )

    bshow = bshow.sort_values(
        [
            "_scope",
            "window",
            "_order",
        ]
    )

    for _, r in bshow.iterrows():

        lines.append(
            f"| {r['sample_scope']} "
            f"| {int(r['window'])} "
            f"| {r['block_name']} "
            f"| {num(r['mean_block_marginal_ic'])} "
            f"| {pct(r['positive_block_month_share'])} "
            f"| {pct(r['positive_block_year_share'])} "
            f"| {num(r['mean_block_marginal_mse_gain'])} |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Frozen interpretation
    # --------------------------------------------------------

    lines.append(
        "## 6. Integrated interpretation"
    )

    lines.append("")

    if flags[
        "primary_delta_ic_positive_all_windows"
    ]:

        lines.append(
            "- The frozen all-network model has positive "
            "mean incremental OOS rank IC at all three "
            "network windows in the PRIMARY evaluation."
        )

    if flags[
        "full_rank_delta_ic_positive_all_windows"
    ]:

        lines.append(
            "- The positive incremental OOS rank IC also "
            "holds at all three windows on the FULL_RANK "
            "numerical-robustness sample."
        )

    if flags[
        "primary_positive_year_share_gt_half_all_windows"
    ]:

        lines.append(
            "- The incremental OOS improvement is positive "
            "in a majority of calendar years at all three "
            "windows."
        )

    lines.append(
        "- Block-level statistics are conditional LOBO "
        "diagnostics and must not be interpreted as an "
        "additive decomposition or as a new feature-selection rule."
    )

    lines.append("")

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    lines.append(
        "## 7. Formal QA"
    )

    lines.append("")

    lines.append(
        f"- all_formal_qa_pass: "
        f"`{qa['all_formal_qa_pass']}`"
    )

    lines.append(
        f"- joint_summary_row_count: "
        f"`{qa['joint_summary_row_count']}`"
    )

    lines.append(
        f"- block_summary_row_count: "
        f"`{qa['block_summary_row_count']}`"
    )

    lines.append(
        "- No new model fitting, factor selection, "
        "window selection, or outcome-driven specification "
        "change is performed in Step 6."
    )

    text = "\n".join(
        lines
    )

    tmp = Path(
        str(REPORT_OUT) + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        tmp,
        REPORT_OUT,
    )


# ============================================================
# 12. Main
# ============================================================

def main():

    print("=" * 80)
    print("M1 - Research Day 11 - Step 6")
    print("Joint OOS Validation Summary")
    print("=" * 80)

    # --------------------------------------------------------
    # 12.1 Upstream QA
    # --------------------------------------------------------

    upstream = (
        validate_upstream()
    )

    # --------------------------------------------------------
    # 12.2 Load frozen results
    # --------------------------------------------------------

    redundancy = (
        build_redundancy_summary()
    )

    oos = (
        load_oos_summary()
    )

    temporal = (
        load_temporal_summary()
    )

    (
        block_long,
        block_wide,
    ) = build_block_wide()

    # --------------------------------------------------------
    # 12.3 Joint summary
    # --------------------------------------------------------

    joint = (
        build_joint_summary(
            oos,
            temporal,
            redundancy,
            block_wide,
        )
    )

    # --------------------------------------------------------
    # 12.4 Cross-step consistency
    # --------------------------------------------------------

    consistency = (
        consistency_checks(
            oos,
            temporal,
            block_long,
        )
    )

    # --------------------------------------------------------
    # 12.5 Result descriptors
    # --------------------------------------------------------

    flags = (
        descriptive_flags(
            joint
        )
    )

    # --------------------------------------------------------
    # 12.6 Save main outputs
    # --------------------------------------------------------

    save_csv(
        joint,
        JOINT_OUT,
    )

    save_csv(
        block_long,
        BLOCK_OUT,
    )

    save_csv(
        redundancy,
        REDUNDANCY_OUT,
    )

    # --------------------------------------------------------
    # 12.7 Structural QA
    #
    # IMPORTANT:
    # QA does NOT depend on whether estimated OOS effects
    # are positive or negative.
    # --------------------------------------------------------

    expected_joint_rows = (
        len(SCOPES)
        * len(WINDOWS)
    )

    expected_block_rows = (
        len(SCOPES)
        * len(WINDOWS)
        * len(BLOCKS)
    )

    joint_dup_n = int(
        joint[
            [
                "sample_scope",
                "window",
            ]
        ]
        .duplicated()
        .sum()
    )

    block_dup_n = int(
        block_long[
            [
                "sample_scope",
                "window",
                "block_name",
            ]
        ]
        .duplicated()
        .sum()
    )

    complete_block_cells = (
        block_long.groupby(
            [
                "sample_scope",
                "window",
            ]
        )[
            "block_name"
        ]
        .nunique()
    )

    all_four_blocks_present = bool(
        (
            complete_block_cells
            == len(BLOCKS)
        ).all()
    )

    redundancy_complete = bool(
        set(
            redundancy[
                "window"
            ]
        )
        ==
        set(
            WINDOWS
        )
    )

    qa = {
        **upstream,

        "joint_summary_row_count":
            int(
                len(joint)
            ),

        "expected_joint_summary_row_count":
            expected_joint_rows,

        "joint_duplicate_scope_window_count":
            joint_dup_n,

        "block_summary_row_count":
            int(
                len(block_long)
            ),

        "expected_block_summary_row_count":
            expected_block_rows,

        "block_duplicate_scope_window_block_count":
            block_dup_n,

        "all_four_blocks_present_each_scope_window":
            all_four_blocks_present,

        "redundancy_window_count":
            int(
                redundancy[
                    "window"
                ].nunique()
            ),

        "redundancy_complete":
            redundancy_complete,

        **consistency,

        "new_model_fitted":
            False,

        "new_factor_constructed":
            False,

        "factor_selected_from_results":
            False,

        "factor_removed_from_primary_model":
            False,

        "window_selected_from_results":
            False,

        "hyperparameter_tuned":
            False,

        "future_target_used_for_model_selection":
            False,

        "new_statistical_test_added":
            False,

        "step6_is_summary_only":
            True,
    }

    # --------------------------------------------------------
    # Formal PASS depends ONLY on integrity / reproducibility,
    # NOT on whether the empirical result is positive.
    # --------------------------------------------------------

    qa[
        "all_formal_qa_pass"
    ] = bool(
        all(
            upstream.values()
        )
        and
        qa[
            "joint_summary_row_count"
        ]
        == expected_joint_rows
        and
        qa[
            "joint_duplicate_scope_window_count"
        ] == 0
        and
        qa[
            "block_summary_row_count"
        ]
        == expected_block_rows
        and
        qa[
            "block_duplicate_scope_window_block_count"
        ] == 0
        and
        qa[
            "all_four_blocks_present_each_scope_window"
        ]
        and
        qa[
            "redundancy_complete"
        ]
        and
        qa[
            "step4_step5_temporal_consistency_pass"
        ]
        and
        qa[
            "step4_step5_block_total_consistency_pass"
        ]
    )

    save_csv(
        pd.DataFrame(
            [
                {
                    "qa_name": k,
                    "qa_value": v,
                }
                for k, v
                in qa.items()
            ]
        ),
        QA_OUT,
    )

    # --------------------------------------------------------
    # 12.8 Metadata
    # --------------------------------------------------------

    metadata = {
        "research_day":
            11,

        "step":
            "Step6_Day11_Joint_OOS_Validation_Summary",

        "purpose":
            (
                "Summary-only integration of frozen "
                "Day-11 redundancy, strict OOS, temporal, "
                "and block-incremental validation results."
            ),

        "windows":
            WINDOWS,

        "sample_scopes":
            SCOPES,

        "frozen_factor_count":
            FROZEN_FACTOR_COUNT,

        "network_blocks":
            BLOCKS,

        "inputs": {
            "step2_redundancy":
                str(
                    REDUNDANCY_PATH
                ),

            "step4_oos":
                str(
                    OOS_PATH
                ),

            "step5_temporal":
                str(
                    TEMPORAL_PATH
                ),

            "step5_block":
                str(
                    BLOCK_PATH
                ),
        },

        "integration_rule": {
            "redundancy":
                (
                    "Step-2 NATIVE and COMMON_DATE "
                    "descriptive redundancy metrics "
                    "are joined by network window."
                ),

            "oos":
                (
                    "Step-4 PRIMARY and FULL_RANK "
                    "OOS metrics are preserved unchanged."
                ),

            "temporal":
                (
                    "Step-5 annual temporal summaries "
                    "are preserved unchanged."
                ),

            "blocks":
                (
                    "All four pre-frozen LOBO blocks "
                    "are retained; no block is selected "
                    "or ranked into the primary model."
                ),
        },

        "result_descriptors":
            flags,

        "important_interpretation_rule":
            (
                "Block marginal contributions are conditional "
                "leave-one-block-out diagnostics. They are not "
                "an additive decomposition and are not used "
                "to redefine the frozen nine-factor primary model."
            ),

        "step6_new_estimation":
            False,

        "formal_qa":
            qa,
    }

    with open(
        META_OUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # 12.9 Markdown report
    # --------------------------------------------------------

    write_report(
        joint,
        redundancy,
        block_long,
        qa,
        flags,
    )

    # --------------------------------------------------------
    # 12.10 Console
    # --------------------------------------------------------

    print()
    print(
        "Joint OOS validation summary:"
    )

    show_cols = [
        "sample_scope",
        "window",
        "mean_delta_ic",
        "positive_delta_ic_month_share",
        "positive_year_share",
        "relative_oos_r2_rank",
        "redundancy_native_effective_rank",

        "level_position_mean_block_marginal_ic",
        "reconfiguration_mean_block_marginal_ic",
        "neighbor_stability_mean_block_marginal_ic",
        "cross_industry_composition_mean_block_marginal_ic",
    ]

    print(
        joint[
            show_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Result descriptors:"
    )

    for k, v in flags.items():

        print(
            f"  {k}: {v}"
        )

    print()
    print(
        "Formal QA:"
    )

    for k, v in qa.items():

        print(
            f"  {k}: {v}"
        )

    if not qa[
        "all_formal_qa_pass"
    ]:

        raise RuntimeError(
            "Day-11 Step-6 formal QA failed."
        )

    print()
    print("=" * 80)
    print("DAY 11 STEP 6 COMPLETE")
    print(f"Output: {OUTDIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
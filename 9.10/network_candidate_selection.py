from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. Paths
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "07_step7_network_candidate_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Step 3
# ------------------------------------------------------------

STEP3_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "03_step3_common_factor"
)

COMMON_FACTOR_SUMMARY_PATH = (
    STEP3_DIR
    / "common_factor_summary.csv"
)


# ------------------------------------------------------------
# Step 6
#
# Step 6 使用修正后的 uniform pair sampler，
# 因此 Step 7 优先使用 Step 6 重新计算的
# Raw + Residual 结果。
# ------------------------------------------------------------

STEP6_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "06_step6_market_residual"
)

MARKET_RESIDUAL_SUMMARY_PATH = (
    STEP6_DIR
    / "market_residual_summary.csv"
)

MARKET_RESIDUAL_WINDOW_PATH = (
    STEP6_DIR
    / "market_residual_window_comparison.csv"
)

MARKET_RESIDUAL_RELATIONSHIP_PATH = (
    STEP6_DIR
    / "market_residual_relationship_summary.csv"
)


# ============================================================
# 1. Main settings
# ============================================================

MAIN_INDUSTRY_LEVEL = (
    "industry_id1"
)

EXPECTED_WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# 这些阈值只用于 diagnostic evidence gates，
# 不是正式统计显著性阈值。
#
# 它们的目的只是自动把已经十分明确的结果
# 分类成 Strong / Weak evidence。
# ------------------------------------------------------------

STRONG_ALIGNMENT_THRESHOLD = 0.90

LARGE_MARKET_REDUCTION_THRESHOLD = 0.80

RESIDUAL_CENTER_THRESHOLD = 0.02

TOPOLOGY_OVERLAP_THRESHOLD = 0.50

HIGH_POSITIVE_MONTH_SHARE = 0.90


# ============================================================
# 2. Helpers
# ============================================================

def require_file(
    path: Path,
    description: str,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{description} 不存在：\n"
            f"{path}"
        )


def safe_corr(
    x,
    y,
):

    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    valid = (
        x.notna()
        &
        y.notna()
    )

    if valid.sum() < 3:

        return np.nan

    xv = (
        x[
            valid
        ]
        .to_numpy(
            dtype=float
        )
    )

    yv = (
        y[
            valid
        ]
        .to_numpy(
            dtype=float
        )
    )

    if (
        np.std(xv) <= 0
        or
        np.std(yv) <= 0
    ):

        return np.nan

    return float(
        np.corrcoef(
            xv,
            yv,
        )[0, 1]
    )


def first_existing_column(
    df,
    candidates,
    required=True,
):

    for col in candidates:

        if col in df.columns:

            return col

    if required:

        raise ValueError(
            "以下候选字段均不存在：\n"
            f"{candidates}\n\n"
            "实际字段：\n"
            f"{df.columns.tolist()}"
        )

    return None


def fmt_float(
    value,
    digits=3,
):

    if not np.isfinite(
        value
    ):

        return "NA"

    return (
        f"{value:.{digits}f}"
    )


def fmt_pct(
    value,
    digits=1,
):

    if not np.isfinite(
        value
    ):

        return "NA"

    return (
        f"{100 * value:.{digits}f}%"
    )


def value_range_text(
    series,
    digits=3,
):

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if x.empty:

        return "NA"

    return (
        f"{x.min():.{digits}f}"
        f"–"
        f"{x.max():.{digits}f}"
    )


# ============================================================
# 3. Load Step 3
# ============================================================

def load_step3():

    require_file(
        COMMON_FACTOR_SUMMARY_PATH,
        "Step 3 common_factor_summary.csv",
    )

    df = pd.read_csv(
        COMMON_FACTOR_SUMMARY_PATH
    )

    df["analysis_date"] = (
        pd.to_datetime(
            df[
                "analysis_date"
            ]
        )
    )

    df["window"] = (
        pd.to_numeric(
            df[
                "window"
            ],
            errors="raise",
        )
        .astype(int)
    )

    pc1_evr_col = (
        first_existing_column(
            df,
            [
                "pca_pc1_evr",
                "pc1_evr",
            ],
        )
    )

    pc1_ew_corr_col = (
        first_existing_column(
            df,
            [
                "pca_pc1_corr_ew_factor",
                "pc1_ew_corr",
            ],
            required=False,
        )
    )

    ew_r2_col = (
        first_existing_column(
            df,
            [
                "ew_r2_median",
            ],
        )
    )

    keep = [
        "analysis_date",
        "window",
        pc1_evr_col,
        ew_r2_col,
    ]

    if (
        pc1_ew_corr_col
        is not None
    ):

        keep.append(
            pc1_ew_corr_col
        )

    result = (
        df[
            keep
        ]
        .copy()
    )

    result = result.rename(
        columns={
            pc1_evr_col:
                "pc1_evr",

            ew_r2_col:
                "ew_market_r2_median",

            pc1_ew_corr_col:
                "pc1_ew_factor_corr",
        }
    )

    return result


# ============================================================
# 4. Load Step 6
# ============================================================

def load_step6():

    require_file(
        MARKET_RESIDUAL_SUMMARY_PATH,
        "Step 6 market_residual_summary.csv",
    )

    require_file(
        MARKET_RESIDUAL_WINDOW_PATH,
        "Step 6 market_residual_window_comparison.csv",
    )

    summary = pd.read_csv(
        MARKET_RESIDUAL_SUMMARY_PATH
    )

    window = pd.read_csv(
        MARKET_RESIDUAL_WINDOW_PATH
    )

    summary[
        "analysis_date"
    ] = pd.to_datetime(
        summary[
            "analysis_date"
        ]
    )

    summary[
        "window"
    ] = pd.to_numeric(
        summary[
            "window"
        ],
        errors="raise",
    ).astype(int)

    window[
        "window"
    ] = pd.to_numeric(
        window[
            "window"
        ],
        errors="raise",
    ).astype(int)

    # --------------------------------------------------------
    # 主分析统一使用 industry_id1
    # --------------------------------------------------------

    summary = (
        summary[
            summary[
                "industry_level"
            ]
            .eq(
                MAIN_INDUSTRY_LEVEL
            )
        ]
        .copy()
    )

    window = (
        window[
            window[
                "industry_level"
            ]
            .eq(
                MAIN_INDUSTRY_LEVEL
            )
        ]
        .copy()
    )

    if summary.empty:

        raise ValueError(
            "Step 6 中没有找到 "
            f"{MAIN_INDUSTRY_LEVEL}。"
        )

    if window.empty:

        raise ValueError(
            "Step 6 Window Summary 中没有找到 "
            f"{MAIN_INDUSTRY_LEVEL}。"
        )

    return (
        summary,
        window,
    )


# ============================================================
# 5. Merge Step 3 + Step 6
#
# 这里重新研究：
#
# corrected Raw Correlation
#        vs
# PC1 / Market R2
#
# 避免 Step 7 继续依赖旧 pair sampler。
# ============================================================

def build_monthly_integrated_evidence(
    step3,
    step6,
):

    required_step6 = [

        "analysis_date",
        "window",

        "raw_all_mean_corr",
        "residual_all_mean_corr",

        "raw_same_cross_gap",
        "residual_same_cross_gap",

        "raw_top1_same_enrichment",
        "residual_top1_same_enrichment",

        "raw_residual_top1_overlap",
        "raw_residual_top5_overlap",

        "raw_residual_pair_corr",

        "raw_residual_sign_change_share",

        "overall_corr_reduction_fraction",
    ]

    missing = [

        x
        for x in required_step6

        if x not in step6.columns
    ]

    if missing:

        raise ValueError(
            "Step 6 Summary 缺少字段：\n"
            f"{missing}"
        )

    merged = (
        step6[
            required_step6
        ]
        .merge(
            step3,
            on=[
                "analysis_date",
                "window",
            ],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
    )

    match_count = int(
        merged[
            "_merge"
        ]
        .eq(
            "both"
        )
        .sum()
    )

    print()
    print(
        "Step 3 × Step 6 matched rows: "
        f"{match_count:,} / {len(merged):,}"
    )

    if match_count == 0:

        raise RuntimeError(
            "Step 3 和 Step 6 "
            "没有任何 Month-End × W 成功匹配。"
        )

    merged = (
        merged.drop(
            columns=[
                "_merge"
            ]
        )
    )

    return merged


# ============================================================
# 6. Window-level Evidence Table
# ============================================================

def build_window_evidence(
    merged,
    step6_window,
):

    rows = []

    for window in sorted(
        merged[
            "window"
        ]
        .unique()
    ):

        g = (
            merged[
                merged[
                    "window"
                ]
                .eq(
                    window
                )
            ]
            .sort_values(
                "analysis_date"
            )
        )

        w = (
            step6_window[
                step6_window[
                    "window"
                ]
                .eq(
                    window
                )
            ]
        )

        if len(w) != 1:

            raise ValueError(
                f"W={window} 的 Step 6 "
                "Window Comparison 不是唯一一行。"
            )

        w = w.iloc[0]

        rows.append(
            {

                "window":
                    int(window),

                "month_count":
                    int(
                        len(g)
                    ),

                # --------------------------------------------
                # Common Market Mode
                # --------------------------------------------

                "mean_pc1_evr":
                    float(
                        g[
                            "pc1_evr"
                        ]
                        .mean()
                    ),

                "median_pc1_ew_factor_corr":
                    float(
                        g[
                            "pc1_ew_factor_corr"
                        ]
                        .median()
                    )
                    if (
                        "pc1_ew_factor_corr"
                        in g.columns
                    )
                    else
                    np.nan,

                "mean_market_r2_median":
                    float(
                        g[
                            "ew_market_r2_median"
                        ]
                        .mean()
                    ),

                "corr_corrected_raw_mean_pc1_evr":
                    safe_corr(
                        g[
                            "raw_all_mean_corr"
                        ],
                        g[
                            "pc1_evr"
                        ],
                    ),

                "corr_corrected_raw_mean_market_r2":
                    safe_corr(
                        g[
                            "raw_all_mean_corr"
                        ],
                        g[
                            "ew_market_r2_median"
                        ],
                    ),

                # --------------------------------------------
                # Raw vs Residual
                # --------------------------------------------

                "raw_mean_corr":
                    float(
                        w[
                            "raw_mean_corr"
                        ]
                    ),

                "residual_mean_corr":
                    float(
                        w[
                            "residual_mean_corr"
                        ]
                    ),

                "corr_reduction_fraction":
                    float(
                        w[
                            "corr_reduction_fraction"
                        ]
                    ),

                "raw_residual_pair_corr":
                    float(
                        w[
                            "raw_residual_pair_corr"
                        ]
                    ),

                "raw_residual_top5_overlap":
                    float(
                        w[
                            "raw_residual_top5_overlap"
                        ]
                    ),

                "raw_residual_top1_overlap":
                    float(
                        w[
                            "raw_residual_top1_overlap"
                        ]
                    ),

                "sign_change_share":
                    float(
                        w[
                            "sign_change_share"
                        ]
                    ),

                # --------------------------------------------
                # Industry structure
                # --------------------------------------------

                "raw_industry_gap":
                    float(
                        w[
                            "raw_same_cross_gap"
                        ]
                    ),

                "residual_industry_gap":
                    float(
                        w[
                            "residual_same_cross_gap"
                        ]
                    ),

                "industry_gap_retention":
                    float(
                        w[
                            "industry_gap_retention"
                        ]
                    ),

                "raw_top1_same_enrichment":
                    float(
                        w[
                            "raw_top1_same_enrichment"
                        ]
                    ),

                "residual_top1_same_enrichment":
                    float(
                        w[
                            "residual_top1_same_enrichment"
                        ]
                    ),

                "raw_top5_same_enrichment":
                    float(
                        w[
                            "raw_top5_same_enrichment"
                        ]
                    ),

                "residual_top5_same_enrichment":
                    float(
                        w[
                            "residual_top5_same_enrichment"
                        ]
                    ),

                # --------------------------------------------
                # Monthly robustness
                # --------------------------------------------

                "share_residual_gap_positive":
                    float(
                        (
                            g[
                                "residual_same_cross_gap"
                            ]
                            >
                            0
                        ).mean()
                    ),

                "share_residual_enrichment_gt_1":
                    float(
                        (
                            g[
                                "residual_top1_same_enrichment"
                            ]
                            >
                            1
                        ).mean()
                    ),

                "share_residual_gap_gt_raw_gap":
                    float(
                        (
                            g[
                                "residual_same_cross_gap"
                            ]
                            >
                            g[
                                "raw_same_cross_gap"
                            ]
                        ).mean()
                    ),

                "corr_market_mode_residual_gap":
                    safe_corr(
                        g[
                            "pc1_evr"
                        ],
                        g[
                            "residual_same_cross_gap"
                        ],
                    ),

                "corr_market_mode_residual_top1_enrichment":
                    safe_corr(
                        g[
                            "pc1_evr"
                        ],
                        g[
                            "residual_top1_same_enrichment"
                        ],
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# 7. Evidence Gates
#
# 不做 0-100 任意加权总分。
#
# 只判断研究证据是否满足若干
# 有明确经济解释的条件。
# ============================================================

def build_decision_flags(
    window_evidence,
):

    strong_market_alignment = bool(
        (
            window_evidence[
                "median_pc1_ew_factor_corr"
            ]
            >=
            STRONG_ALIGNMENT_THRESHOLD
        )
        .all()
    )

    strong_raw_market_link = bool(
        (
            window_evidence[
                "corr_corrected_raw_mean_pc1_evr"
            ]
            >=
            STRONG_ALIGNMENT_THRESHOLD
        )
        .all()
    )

    large_market_component_removed = bool(
        (
            window_evidence[
                "corr_reduction_fraction"
            ]
            >=
            LARGE_MARKET_REDUCTION_THRESHOLD
        )
        .all()
    )

    residual_is_centered = bool(
        (
            window_evidence[
                "residual_mean_corr"
            ]
            .abs()
            <=
            RESIDUAL_CENTER_THRESHOLD
        )
        .all()
    )

    residual_industry_structure_survives = bool(
        (
            window_evidence[
                "residual_industry_gap"
            ]
            >
            0
        )
        .all()
        and
        (
            window_evidence[
                "share_residual_gap_positive"
            ]
            >=
            HIGH_POSITIVE_MONTH_SHARE
        )
        .all()
    )

    residual_strong_edges_industry_enriched = bool(
        (
            window_evidence[
                "residual_top1_same_enrichment"
            ]
            >
            1
        )
        .all()
        and
        (
            window_evidence[
                "share_residual_enrichment_gt_1"
            ]
            >=
            HIGH_POSITIVE_MONTH_SHARE
        )
        .all()
    )

    residual_industry_more_visible_than_raw = bool(
        (
            window_evidence[
                "residual_top1_same_enrichment"
            ]
            >
            window_evidence[
                "raw_top1_same_enrichment"
            ]
        )
        .all()
    )

    topology_materially_changes = bool(
        (
            window_evidence[
                "raw_residual_top1_overlap"
            ]
            <
            TOPOLOGY_OVERLAP_THRESHOLD
        )
        .all()
    )

    flags = pd.DataFrame(
        [
            {
                "evidence_gate":
                    "Strong market-wide first mode",

                "passed":
                    strong_market_alignment,

                "interpretation":
                    (
                        "PC1 is strongly aligned with the "
                        "equal-weighted market factor."
                    ),
            },

            {
                "evidence_gate":
                    "Raw correlation tracks market mode",

                "passed":
                    strong_raw_market_link,

                "interpretation":
                    (
                        "Time variation in corrected Raw Mean "
                        "Correlation closely tracks PC1 strength."
                    ),
            },

            {
                "evidence_gate":
                    "Market removal materially reduces raw baseline",

                "passed":
                    large_market_component_removed,

                "interpretation":
                    (
                        "Removing the market factor eliminates "
                        "a large fraction of the positive average "
                        "cross-sectional correlation baseline."
                    ),
            },

            {
                "evidence_gate":
                    "Residual average correlation is near zero",

                "passed":
                    residual_is_centered,

                "interpretation":
                    (
                        "The broad positive market-wide baseline "
                        "is largely absent in residual correlation."
                    ),
            },

            {
                "evidence_gate":
                    "Industry structure survives market removal",

                "passed":
                    residual_industry_structure_survives,

                "interpretation":
                    (
                        "Same-industry residual correlations "
                        "remain systematically stronger than "
                        "cross-industry residual correlations."
                    ),
            },

            {
                "evidence_gate":
                    "Residual strongest edges remain industry enriched",

                "passed":
                    residual_strong_edges_industry_enriched,

                "interpretation":
                    (
                        "Strong residual associations are "
                        "over-represented within industries."
                    ),
            },

            {
                "evidence_gate":
                    "Industry structure becomes clearer after removal",

                "passed":
                    residual_industry_more_visible_than_raw,

                "interpretation":
                    (
                        "Top-tail industry enrichment is stronger "
                        "in residual space than in raw-return space."
                    ),
            },

            {
                "evidence_gate":
                    "Raw and residual topologies differ materially",

                "passed":
                    topology_materially_changes,

                "interpretation":
                    (
                        "Less than half of Raw Top-1% edges remain "
                        "Top-1% after market-factor removal."
                    ),
            },
        ]
    )

    return flags


# ============================================================
# 8. Candidate Evidence Matrix
# ============================================================

def build_candidate_matrix(
    window_evidence,
    flags,
):

    flag_map = dict(
        zip(
            flags[
                "evidence_gate"
            ],
            flags[
                "passed"
            ],
        )
    )

    raw_market_corr_range = (
        value_range_text(
            window_evidence[
                "corr_corrected_raw_mean_pc1_evr"
            ]
        )
    )

    reduction_range = (
        value_range_text(
            window_evidence[
                "corr_reduction_fraction"
            ]
        )
    )

    residual_mean_range = (
        value_range_text(
            window_evidence[
                "residual_mean_corr"
            ]
        )
    )

    residual_gap_range = (
        value_range_text(
            window_evidence[
                "residual_industry_gap"
            ]
        )
    )

    residual_enrichment_range = (
        value_range_text(
            window_evidence[
                "residual_top1_same_enrichment"
            ]
        )
    )

    top1_overlap_range = (
        value_range_text(
            window_evidence[
                "raw_residual_top1_overlap"
            ]
        )
    )

    all_residual_gates = all(
        [
            flag_map[
                "Market removal materially reduces raw baseline"
            ],

            flag_map[
                "Residual average correlation is near zero"
            ],

            flag_map[
                "Industry structure survives market removal"
            ],

            flag_map[
                "Residual strongest edges remain industry enriched"
            ],

            flag_map[
                "Raw and residual topologies differ materially"
            ],
        ]
    )

    residual_status = (
        "PRIMARY CANDIDATE"
        if all_residual_gates
        else
        "REQUIRES REVIEW"
    )

    rows = [

        # ====================================================
        # Raw Pearson
        # ====================================================

        {
            "candidate_id":
                "C1",

            "network_candidate":
                (
                    "Raw Pearson + "
                    "density-controlled sparsification"
                ),

            "network_object":
                (
                    "Corr(r_i, r_j)"
                ),

            "economic_meaning":
                (
                    "Unconditional stock co-movement."
                ),

            "market_contamination":
                "HIGH",

            "data_evidence":
                (
                    "Corrected Raw Mean Correlation tracks "
                    "PC1 market-mode strength with correlation "
                    f"{raw_market_corr_range} across windows."
                ),

            "industry_structure":
                (
                    "Present, but broad market co-movement "
                    "raises both Same- and Cross-industry "
                    "correlations."
                ),

            "topology_evidence":
                (
                    "Only "
                    f"{top1_overlap_range} "
                    "of Raw Top-1% edges remain Top-1% "
                    "after market removal."
                ),

            "high_dimensional_feasibility":
                (
                    "High. Pairwise correlation is scalable, "
                    "but a sparse edge rule is required because "
                    "the full graph contains millions of pairs."
                ),

            "main_advantage":
                (
                    "Simple, transparent, interpretable baseline."
                ),

            "main_limitation":
                (
                    "Dominant market mode materially affects "
                    "both edge magnitude and edge ranking."
                ),

            "recommended_role":
                "BASELINE",

            "selection_status":
                "KEEP AS BASELINE",
        },

        # ====================================================
        # Market residual correlation
        # ====================================================

        {
            "candidate_id":
                "C2",

            "network_candidate":
                (
                    "PIT EW Market-Residual Pearson + "
                    "density-controlled sparsification"
                ),

            "network_object":
                (
                    "Corr(epsilon_i, epsilon_j), "
                    "after leave-one-out market removal"
                ),

            "economic_meaning":
                (
                    "Pairwise association after removing "
                    "broad contemporaneous market co-movement."
                ),

            "market_contamination":
                "LOWER",

            "data_evidence":
                (
                    "Market removal reduces the positive average "
                    f"correlation by {reduction_range}; "
                    "mean residual correlation is only "
                    f"{residual_mean_range}."
                ),

            "industry_structure":
                (
                    "Strongly preserved. Residual Same-Cross "
                    f"gap is {residual_gap_range}; "
                    "Residual Top-1% industry enrichment is "
                    f"{residual_enrichment_range}."
                ),

            "topology_evidence":
                (
                    "Raw and residual strongest-edge sets differ "
                    "substantially, indicating that residualization "
                    "changes more than the correlation intercept."
                ),

            "high_dimensional_feasibility":
                (
                    "High. Market residualization and pairwise "
                    "correlation scale well to the full universe; "
                    "sparsification is still required."
                ),

            "main_advantage":
                (
                    "Removes the dominant market-wide common "
                    "component while preserving economically "
                    "meaningful industry structure."
                ),

            "main_limitation":
                (
                    "Still a marginal pairwise association measure; "
                    "it does not establish conditional independence "
                    "given all other stocks."
                ),

            "recommended_role":
                "PRIMARY ASSOCIATION NETWORK",

            "selection_status":
                residual_status,
        },

        # ====================================================
        # Market + Industry residual
        # ====================================================

        {
            "candidate_id":
                "C3",

            "network_candidate":
                (
                    "Market + Industry Residual Correlation"
                ),

            "network_object":
                (
                    "Corr(xi_i, xi_j) after removing "
                    "market and industry common components"
                ),

            "economic_meaning":
                (
                    "More stock-specific pairwise association "
                    "beyond broad market and industry structure."
                ),

            "market_contamination":
                "LOWER",

            "data_evidence":
                (
                    "NOT YET ESTIMATED. Step 6 shows that a large "
                    "industry component remains after market removal, "
                    "which motivates this extension."
                ),

            "industry_structure":
                (
                    "Industry component would be intentionally "
                    "removed rather than preserved."
                ),

            "topology_evidence":
                "PENDING",

            "high_dimensional_feasibility":
                (
                    "High to moderate, depending on the "
                    "industry-factor specification."
                ),

            "main_advantage":
                (
                    "Can isolate more idiosyncratic stock-to-stock "
                    "relationships."
                ),

            "main_limitation":
                (
                    "May remove economically meaningful industry "
                    "information that is useful for the M1 network."
                ),

            "recommended_role":
                "EXTENSION / ROBUSTNESS",

            "selection_status":
                "PENDING FUTURE TEST",
        },

        # ====================================================
        # Sparse conditional association
        # ====================================================

        {
            "candidate_id":
                "C4",

            "network_candidate":
                (
                    "Sparse Conditional Association on "
                    "Market Residuals"
                ),

            "network_object":
                (
                    "-Omega_ij / sqrt(Omega_ii Omega_jj), "
                    "estimated with sparse high-dimensional methods"
                ),

            "economic_meaning":
                (
                    "Direct conditional association after "
                    "controlling for the remaining stock system."
                ),

            "market_contamination":
                "LOWER",

            "data_evidence":
                (
                    "NOT YET ESTIMATED at full-market scale. "
                    "Step 6 demonstrates substantial residual "
                    "structure worth testing conditionally."
                ),

            "industry_structure":
                (
                    "Unknown until full-market sparse precision "
                    "estimation is performed."
                ),

            "topology_evidence":
                "PENDING",

            "high_dimensional_feasibility":
                (
                    "Requires sparse regularization because p >> W. "
                    "Nodewise Lasso / CLIME-type estimators are more "
                    "natural full-market candidates than an "
                    "unrestricted dense precision inverse."
                ),

            "main_advantage":
                (
                    "Closer to a conditional-dependence network "
                    "than marginal correlation."
                ),

            "main_limitation":
                (
                    "Tuning, sparsity assumptions, computation, "
                    "and network stability require formal validation."
                ),

            "recommended_role":
                "FORMAL CHALLENGER",

            "selection_status":
                "NEXT-STAGE CANDIDATE",
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 9. Sparsification Rule Matrix
# ============================================================

def build_sparsification_matrix():

    rows = [

        {
            "rule":
                "Fixed absolute correlation threshold",

            "example":
                "Edge if rho_ij > 0.5",

            "density_control":
                "NO",

            "node_coverage_control":
                "NO",

            "cross_regime_comparability":
                "WEAK",

            "interpretation":
                (
                    "Simple absolute-strength criterion."
                ),

            "main_problem":
                (
                    "Network density can change mechanically "
                    "when the overall correlation distribution "
                    "moves across market regimes and window sizes."
                ),

            "recommended_role":
                "ROBUSTNESS ONLY",
        },

        {
            "rule":
                "Global Top-q% / fixed density",

            "example":
                "Keep strongest 1%, 2%, or 5% edges",

            "density_control":
                "YES",

            "node_coverage_control":
                "NO",

            "cross_regime_comparability":
                "STRONG",

            "interpretation":
                (
                    "Retains the strongest relative associations "
                    "under a fixed network density."
                ),

            "main_problem":
                (
                    "Some peripheral stocks may receive few "
                    "or no edges."
                ),

            "recommended_role":
                "PRIMARY SPARSIFICATION BASELINE",
        },

        {
            "rule":
                "Per-node Top-k",

            "example":
                "Each stock retains k strongest neighbors",

            "density_control":
                "APPROXIMATE",

            "node_coverage_control":
                "YES",

            "cross_regime_comparability":
                "STRONG",

            "interpretation":
                (
                    "Maintains local strongest relationships "
                    "for every stock."
                ),

            "main_problem":
                (
                    "Requires a symmetrization rule and can "
                    "retain relatively weak edges for isolated nodes."
                ),

            "recommended_role":
                "ROBUSTNESS / LOCAL NETWORK",
        },

        {
            "rule":
                "MST / filtered backbone",

            "example":
                "Minimum spanning tree or related filtered graph",

            "density_control":
                "YES",

            "node_coverage_control":
                "YES",

            "cross_regime_comparability":
                "STRONG",

            "interpretation":
                (
                    "Produces a connected structural backbone."
                ),

            "main_problem":
                (
                    "Extremely sparse and may discard many "
                    "economically meaningful associations."
                ),

            "recommended_role":
                "VISUALIZATION / BACKBONE",
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. Next-stage Comparison Design
# ============================================================

def build_next_stage_plan():

    return pd.DataFrame(
        [
            {
                "priority":
                    1,

                "method":
                    (
                        "Market-Residual Pearson "
                        "+ Global Top-q%"
                    ),

                "purpose":
                    (
                        "Primary interpretable network candidate."
                    ),

                "required_validation":
                    (
                        "Network stability, turnover, industry/community "
                        "structure, Alpha IC/backtest, risk linkage."
                    ),
            },

            {
                "priority":
                    2,

                "method":
                    (
                        "Market-Residual Pearson "
                        "+ Per-node Top-k"
                    ),

                "purpose":
                    (
                        "Check whether conclusions depend on "
                        "global density control."
                    ),

                "required_validation":
                    (
                        "Node coverage, degree concentration, "
                        "edge stability, Alpha/risk robustness."
                    ),
            },

            {
                "priority":
                    3,

                "method":
                    (
                        "Sparse Conditional Association "
                        "on Market Residuals"
                    ),

                "purpose":
                    (
                        "Formal challenger that targets "
                        "conditional rather than marginal association."
                    ),

                "required_validation":
                    (
                        "Regularization path, sparsity, stability, "
                        "computational scalability, comparison with "
                        "residual Pearson edges."
                    ),
            },

            {
                "priority":
                    4,

                "method":
                    (
                        "Raw Pearson + matched density"
                    ),

                "purpose":
                    (
                        "Unconditional baseline for all downstream "
                        "comparisons."
                    ),

                "required_validation":
                    (
                        "Keep identical density and evaluation periods "
                        "when comparing with residual networks."
                    ),
            },
        ]
    )


# ============================================================
# 11. Figures
# ============================================================

def plot_window_evidence(
    df,
):

    df = (
        df.sort_values(
            "window"
        )
        .copy()
    )

    x = np.arange(
        len(df)
    )

    width = 0.35

    # --------------------------------------------------------
    # Figure 1:
    # Raw vs residual mean correlation
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    ax.bar(
        x - width / 2,
        df[
            "raw_mean_corr"
        ],
        width,
        label=
            "Raw",
    )

    ax.bar(
        x + width / 2,
        df[
            "residual_mean_corr"
        ],
        width,
        label=
            "Market Residual",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df[
            "window"
        ].astype(str)
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Mean Correlation"
    )

    ax.set_title(
        "Raw vs Market-Residual Mean Correlation"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "step7_raw_vs_residual_mean_corr.png",
        dpi=180,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Figure 2:
    # Industry gap
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    ax.bar(
        x - width / 2,
        df[
            "raw_industry_gap"
        ],
        width,
        label=
            "Raw",
    )

    ax.bar(
        x + width / 2,
        df[
            "residual_industry_gap"
        ],
        width,
        label=
            "Market Residual",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df[
            "window"
        ].astype(str)
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Same - Cross Correlation Gap"
    )

    ax.set_title(
        "Industry Structure Before and After Market Removal"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "step7_raw_vs_residual_industry_gap.png",
        dpi=180,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Figure 3:
    # Top 1% enrichment
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    ax.bar(
        x - width / 2,
        df[
            "raw_top1_same_enrichment"
        ],
        width,
        label=
            "Raw",
    )

    ax.bar(
        x + width / 2,
        df[
            "residual_top1_same_enrichment"
        ],
        width,
        label=
            "Market Residual",
    )

    ax.axhline(
        1,
        linewidth=1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df[
            "window"
        ].astype(str)
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Top-1% Same-Industry Enrichment"
    )

    ax.set_title(
        "Industry Concentration among Strongest Associations"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "step7_top1_industry_enrichment.png",
        dpi=180,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Figure 4:
    # Raw vs residual topology overlap
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    ax.bar(
        x,
        df[
            "raw_residual_top1_overlap"
        ],
    )

    ax.axhline(
        0.5,
        linewidth=1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df[
            "window"
        ].astype(str)
    )

    ax.set_xlabel(
        "Window"
    )

    ax.set_ylabel(
        "Top-1% Edge Overlap"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        "Top-1% Edge Overlap: Raw vs Market Residual"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "step7_raw_residual_top1_overlap.png",
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# 12. Markdown Report
# ============================================================

def build_markdown_report(
    window_evidence,
    decision_flags,
    candidate_matrix,
):

    raw_corr_range = (
        value_range_text(
            window_evidence[
                "raw_mean_corr"
            ]
        )
    )

    residual_corr_range = (
        value_range_text(
            window_evidence[
                "residual_mean_corr"
            ],
            digits=4,
        )
    )

    reduction_range = (
        value_range_text(
            window_evidence[
                "corr_reduction_fraction"
            ]
        )
    )

    residual_gap_range = (
        value_range_text(
            window_evidence[
                "residual_industry_gap"
            ]
        )
    )

    residual_enrichment_range = (
        value_range_text(
            window_evidence[
                "residual_top1_same_enrichment"
            ]
        )
    )

    overlap_range = (
        value_range_text(
            window_evidence[
                "raw_residual_top1_overlap"
            ]
        )
    )

    passed = int(
        decision_flags[
            "passed"
        ].sum()
    )

    total = len(
        decision_flags
    )

    residual_status = (
        candidate_matrix.loc[
            candidate_matrix[
                "candidate_id"
            ].eq(
                "C2"
            ),
            "selection_status",
        ]
        .iloc[0]
    )

    text = f"""# Day 2 Step 7 — Network Candidate Evidence Matrix

## 1. Objective

Step 7 integrates the empirical evidence from the raw-correlation,
common-factor, industry, and market-residual analyses to determine the
appropriate role of competing network definitions.

No arbitrary weighted total score is used. The selection is based on
economically interpretable evidence gates.

## 2. Main empirical evidence

Across the 60-, 120-, and 252-day windows:

- Corrected raw mean correlation ranges from **{raw_corr_range}**.
- Market-residual mean correlation falls to **{residual_corr_range}**.
- The fraction of the positive average-correlation baseline removed by
  the market factor ranges from **{reduction_range}**.
- The residual Same-minus-Cross industry correlation gap remains
  **{residual_gap_range}**.
- Same-industry enrichment among the residual Top-1% strongest
  associations ranges from **{residual_enrichment_range}**.
- Raw and market-residual Top-1% edge overlap is only
  **{overlap_range}**.

These results indicate that broad market co-movement materially affects
the raw network, while economically meaningful industry structure
persists after market removal.

## 3. Evidence gates

**{passed}/{total}** diagnostic evidence gates pass.

## 4. Candidate roles

### C1 — Raw Pearson Network

Retain as an **unconditional baseline**. It is simple and transparent,
but its edge magnitudes and strongest-edge rankings are substantially
affected by the market-wide common mode.

### C2 — Market-Residual Pearson Network

Selection status: **{residual_status}**.

This is the preferred interpretable association-network candidate for
the next stage because it removes the broad market component while
preserving strong industry-level economic structure.

### C3 — Market + Industry Residual Network

Retain as a future robustness extension if the objective becomes
stock-specific association beyond both market and industry structure.

### C4 — Sparse Conditional Association

Retain as the main **formal challenger**. Because the full A-share
problem satisfies p >> W, conditional-network estimation should use
high-dimensional sparse methods rather than an unrestricted covariance
inverse.

## 5. Sparsification

A fixed absolute correlation threshold should not be the primary edge
rule because correlation distributions vary substantially across
market regimes and rolling-window lengths.

The preferred first comparison is therefore:

**Market-Residual Pearson + fixed-density Top-q% sparsification**

with:

**Per-node Top-k**

as a local-network robustness check.

## 6. Day 2 conclusion

The evidence supports the following working hierarchy:

Raw Pearson → baseline unconditional co-movement

Market-Residual Pearson → primary interpretable association network

Sparse conditional association on market residuals → next-stage
high-dimensional challenger

The final network should still be selected based on downstream network
stability, turnover, Alpha-factor performance, risk linkage, and
computational scalability rather than Step 7 diagnostics alone.
"""

    output_path = (
        OUTPUT_DIR
        / "day2_step7_network_selection_summary.md"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            text
        )


# ============================================================
# 13. Metadata
# ============================================================

def save_metadata():

    metadata = {

        "step":
            "Day2_Step7_Network_Candidate_Evidence_Matrix",

        "main_industry_level":
            MAIN_INDUSTRY_LEVEL,

        "primary_data_sources": {

            "step3":
                str(
                    COMMON_FACTOR_SUMMARY_PATH
                ),

            "step6_summary":
                str(
                    MARKET_RESIDUAL_SUMMARY_PATH
                ),

            "step6_window":
                str(
                    MARKET_RESIDUAL_WINDOW_PATH
                ),
        },

        "important_design_choice":
            (
                "Step 7 preferentially uses the corrected raw "
                "correlation diagnostics recomputed inside Step 6 "
                "with uniform unordered-pair sampling."
            ),

        "selection_philosophy":
            (
                "Evidence gates and explicit method roles are used "
                "instead of an arbitrary weighted numerical score."
            ),

        "current_primary_candidate":
            (
                "PIT EW market-residual Pearson association "
                "with density-controlled sparsification."
            ),

        "baseline":
            (
                "Raw Pearson correlation network with matched density."
            ),

        "formal_next_stage_challenger":
            (
                "Sparse conditional association estimated on "
                "market-residual returns using a high-dimensional "
                "precision/conditional-regression method."
            ),

        "important_limitations": [

            (
                "Step 7 is a method-selection diagnostic, not "
                "evidence that one network is universally optimal."
            ),

            (
                "Market-residual correlation remains a marginal "
                "pairwise association and does not establish "
                "conditional independence."
            ),

            (
                "Final network selection must incorporate downstream "
                "stability, turnover, Alpha-factor performance, "
                "risk information, and computational scalability."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "step7_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 14. Main
# ============================================================

def main():

    print("=" * 76)
    print("M1 Day 2 - Step 7")
    print("Network Candidate Evidence Matrix")
    print("=" * 76)

    # ========================================================
    # 1. Load
    # ========================================================

    print()
    print("[1] Load Step 3 Common Factor")

    step3 = load_step3()

    print()
    print("[2] Load Step 6 Corrected Raw / Residual Results")

    (
        step6_summary,
        step6_window,
    ) = load_step6()

    # ========================================================
    # 2. Integrated evidence
    # ========================================================

    print()
    print("[3] Integrate Common Factor + Raw/Residual Evidence")

    merged = (
        build_monthly_integrated_evidence(
            step3=
                step3,

            step6=
                step6_summary,
        )
    )

    merged.to_csv(
        OUTPUT_DIR
        / "step7_monthly_integrated_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 3. Window evidence
    # ========================================================

    window_evidence = (
        build_window_evidence(
            merged=
                merged,

            step6_window=
                step6_window,
        )
    )

    window_evidence.to_csv(
        OUTPUT_DIR
        / "step7_window_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 4. Decision gates
    # ========================================================

    print()
    print("[4] Evaluate Evidence Gates")

    decision_flags = (
        build_decision_flags(
            window_evidence
        )
    )

    decision_flags.to_csv(
        OUTPUT_DIR
        / "step7_decision_flags.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 5. Candidate matrix
    # ========================================================

    print()
    print("[5] Build Network Candidate Matrix")

    candidate_matrix = (
        build_candidate_matrix(
            window_evidence=
                window_evidence,

            flags=
                decision_flags,
        )
    )

    candidate_matrix.to_csv(
        OUTPUT_DIR
        / "network_candidate_evidence_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 6. Sparsification
    # ========================================================

    sparsification = (
        build_sparsification_matrix()
    )

    sparsification.to_csv(
        OUTPUT_DIR
        / "sparsification_rule_evidence_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 7. Next-stage plan
    # ========================================================

    next_stage = (
        build_next_stage_plan()
    )

    next_stage.to_csv(
        OUTPUT_DIR
        / "next_stage_network_comparison_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 8. Figures
    # ========================================================

    print()
    print("[6] Figures")

    plot_window_evidence(
        window_evidence
    )

    # ========================================================
    # 9. Markdown summary
    # ========================================================

    build_markdown_report(
        window_evidence=
            window_evidence,

        decision_flags=
            decision_flags,

        candidate_matrix=
            candidate_matrix,
    )

    # ========================================================
    # 10. Metadata
    # ========================================================

    save_metadata()

    # ========================================================
    # 11. Console
    # ========================================================

    print()
    print("=" * 76)
    print("Window Evidence")
    print("=" * 76)

    show_cols = [

        "window",

        "mean_pc1_evr",

        "corr_corrected_raw_mean_pc1_evr",

        "raw_mean_corr",

        "residual_mean_corr",

        "corr_reduction_fraction",

        "raw_industry_gap",

        "residual_industry_gap",

        "raw_top1_same_enrichment",

        "residual_top1_same_enrichment",

        "raw_residual_top1_overlap",

        "sign_change_share",
    ]

    print()
    print(
        window_evidence[
            show_cols
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 76)
    print("Evidence Gates")
    print("=" * 76)

    print()
    print(
        decision_flags[
            [
                "evidence_gate",
                "passed",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 76)
    print("Candidate Selection")
    print("=" * 76)

    print()
    print(
        candidate_matrix[
            [
                "candidate_id",
                "network_candidate",
                "recommended_role",
                "selection_status",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
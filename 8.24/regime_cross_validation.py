from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径
# ============================================================

PROJECT_DIR = Path("stock_network")

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

FIGURE_DIR = (
    PROJECT_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 输入
# ============================================================

REGIME_FILE = (
    PROCESSED_DIR
    / "network_regime_assignment.csv"
)

RAW_PARTIAL_FILE = (
    PROCESSED_DIR
    / "rolling_partial_edge_history.csv"
)

MATCHED_PARTIAL_FILE = (
    PROCESSED_DIR
    / "density_matched_partial_edge_history.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 3. 输出
# ============================================================

WINDOW_METRICS_FILE = (
    PROCESSED_DIR
    / "regime_cross_validation_window_metrics.csv"
)

REGIME_SUMMARY_FILE = (
    PROCESSED_DIR
    / "regime_cross_validation_summary.csv"
)

TRANSITION_SUMMARY_FILE = (
    PROCESSED_DIR
    / "regime_cross_validation_transition_summary.csv"
)

R3_R4_FILE = (
    PROCESSED_DIR
    / "regime_cross_validation_R3_R4.csv"
)


# ============================================================
# 4. 工具函数
# ============================================================

def normalize_code(x):

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    match = re.search(
        r"(\d{6})",
        s
    )

    if match:
        return match.group(1)

    digits = "".join(
        c
        for c in s
        if c.isdigit()
    )

    if digits:
        return digits.zfill(6)

    return s


def canonical_pair(a, b):

    a = normalize_code(a)
    b = normalize_code(b)

    if a <= b:
        return a, b

    return b, a


def convert_bool(series):

    if series.dtype == bool:
        return series

    result = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False
            }
        )
    )

    if result.isna().any():

        bad_values = (
            series[
                result.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"存在无法识别的Boolean值：{bad_values}"
        )

    return result.astype(bool)


# ============================================================
# 5. Metadata
# ============================================================

stock_info = pd.read_csv(
    STOCK_INFO_FILE,
    dtype=str
)


code_candidates = [
    "code",
    "stock_code",
    "ts_code",
    "symbol"
]

name_candidates = [
    "name",
    "stock_name"
]

industry_candidates = [
    "industry",
    "industry_name"
]


code_col = next(
    (
        c
        for c in code_candidates
        if c in stock_info.columns
    ),
    None
)

name_col = next(
    (
        c
        for c in name_candidates
        if c in stock_info.columns
    ),
    None
)

industry_col = next(
    (
        c
        for c in industry_candidates
        if c in stock_info.columns
    ),
    None
)


if (
    code_col is None
    or
    name_col is None
    or
    industry_col is None
):

    raise ValueError(
        "stock_info.csv中无法识别代码、名称或行业字段。"
    )


metadata = (
    stock_info[
        [
            code_col,
            name_col,
            industry_col
        ]
    ]
    .rename(
        columns={
            code_col: "code",
            name_col: "name",
            industry_col: "industry"
        }
    )
    .copy()
)


metadata[
    "code"
] = metadata[
    "code"
].apply(
    normalize_code
)


metadata = (
    metadata
    .drop_duplicates(
        subset="code"
    )
)


name_map = dict(
    zip(
        metadata[
            "code"
        ],
        metadata[
            "name"
        ]
    )
)


industry_map = dict(
    zip(
        metadata[
            "code"
        ],
        metadata[
            "industry"
        ]
    )
)


# ============================================================
# 6. 读取Regime Assignment
# ============================================================

regime_df = pd.read_csv(
    REGIME_FILE
)


date_candidates = [
    "network_date",
    "date",
    "window_end"
]


regime_candidates = [
    "regime",
    "regime_id",
    "Regime"
]


regime_date_col = next(
    (
        c
        for c in date_candidates
        if c in regime_df.columns
    ),
    None
)


regime_col = next(
    (
        c
        for c in regime_candidates
        if c in regime_df.columns
    ),
    None
)


if (
    regime_date_col is None
    or
    regime_col is None
):

    raise ValueError(
        "network_regime_assignment.csv中"
        "无法识别日期或Regime字段。"
    )


regime_assignment = (
    regime_df[
        [
            regime_date_col,
            regime_col
        ]
    ]
    .rename(
        columns={
            regime_date_col:
                "network_date",

            regime_col:
                "regime"
        }
    )
    .copy()
)


regime_assignment[
    "network_date"
] = pd.to_datetime(
    regime_assignment[
        "network_date"
    ]
)


regime_assignment[
    "regime"
] = (
    regime_assignment[
        "regime"
    ]
    .astype(str)
    .str.strip()
)


regime_assignment = (
    regime_assignment
    .drop_duplicates(
        subset="network_date"
    )
    .sort_values(
        "network_date"
    )
)


print(
    "Regime windows：",
    len(
        regime_assignment
    )
)

print(
    "Regimes：",
    regime_assignment[
        "regime"
    ]
    .unique()
)


# ============================================================
# 7. 读取原始Rolling Partial
# ============================================================

raw_partial = pd.read_csv(
    RAW_PARTIAL_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


raw_partial[
    "network_date"
] = pd.to_datetime(
    raw_partial[
        "network_date"
    ]
)


for col in [
    "stock_1",
    "stock_2"
]:

    raw_partial[col] = (
        raw_partial[col]
        .apply(
            normalize_code
        )
    )


pairs = raw_partial.apply(
    lambda row:
        canonical_pair(
            row[
                "stock_1"
            ],
            row[
                "stock_2"
            ]
        ),
    axis=1
)


raw_partial[
    "stock_1"
] = [
    x[0]
    for x in pairs
]


raw_partial[
    "stock_2"
] = [
    x[1]
    for x in pairs
]


raw_partial[
    "partial_correlation"
] = pd.to_numeric(
    raw_partial[
        "partial_correlation"
    ],
    errors="raise"
)


raw_partial[
    "abs_partial"
] = (
    raw_partial[
        "partial_correlation"
    ]
    .abs()
)


# ============================================================
# 8. 重新映射Metadata
# ============================================================

raw_partial[
    "industry_1"
] = (
    raw_partial[
        "stock_1"
    ]
    .map(
        industry_map
    )
)


raw_partial[
    "industry_2"
] = (
    raw_partial[
        "stock_2"
    ]
    .map(
        industry_map
    )
)


if (
    raw_partial[
        [
            "industry_1",
            "industry_2"
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise ValueError(
        "Raw Partial中存在Metadata映射失败。"
    )


raw_partial[
    "same_industry"
] = (
    raw_partial[
        "industry_1"
    ]
    ==
    raw_partial[
        "industry_2"
    ]
)


# ============================================================
# 9. 读取Density-matched Partial
# ============================================================

matched_partial = pd.read_csv(
    MATCHED_PARTIAL_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


matched_partial[
    "network_date"
] = pd.to_datetime(
    matched_partial[
        "network_date"
    ]
)


for col in [
    "stock_1",
    "stock_2"
]:

    matched_partial[col] = (
        matched_partial[col]
        .apply(
            normalize_code
        )
    )


pairs = matched_partial.apply(
    lambda row:
        canonical_pair(
            row[
                "stock_1"
            ],
            row[
                "stock_2"
            ]
        ),
    axis=1
)


matched_partial[
    "stock_1"
] = [
    x[0]
    for x in pairs
]


matched_partial[
    "stock_2"
] = [
    x[1]
    for x in pairs
]


if (
    "selected_density_matched"
    not in
    matched_partial.columns
):

    raise ValueError(
        "density_matched_partial_edge_history.csv"
        "缺少selected_density_matched。"
    )


matched_partial[
    "selected_density_matched"
] = convert_bool(
    matched_partial[
        "selected_density_matched"
    ]
)


matched_partial[
    "partial_correlation"
] = pd.to_numeric(
    matched_partial[
        "partial_correlation"
    ],
    errors="raise"
)


matched_partial[
    "abs_partial"
] = (
    matched_partial[
        "partial_correlation"
    ]
    .abs()
)


matched_partial[
    "industry_1"
] = (
    matched_partial[
        "stock_1"
    ]
    .map(
        industry_map
    )
)


matched_partial[
    "industry_2"
] = (
    matched_partial[
        "stock_2"
    ]
    .map(
        industry_map
    )
)


matched_partial[
    "same_industry"
] = (
    matched_partial[
        "industry_1"
    ]
    ==
    matched_partial[
        "industry_2"
    ]
)


# ============================================================
# 10. 只保留共同日期
# ============================================================

common_dates = sorted(
    set(
        regime_assignment[
            "network_date"
        ]
    )
    &
    set(
        raw_partial[
            "network_date"
        ]
    )
    &
    set(
        matched_partial[
            "network_date"
        ]
    )
)


if len(
    common_dates
) == 0:

    raise ValueError(
        "Regime、Raw Partial和Matched Partial没有共同日期。"
    )


regime_assignment = (
    regime_assignment[
        regime_assignment[
            "network_date"
        ]
        .isin(
            common_dates
        )
    ]
    .copy()
)


raw_partial = (
    raw_partial[
        raw_partial[
            "network_date"
        ]
        .isin(
            common_dates
        )
    ]
    .copy()
)


matched_partial = (
    matched_partial[
        matched_partial[
            "network_date"
        ]
        .isin(
            common_dates
        )
    ]
    .copy()
)


print(
    "用于Cross-validation的共同日期：",
    len(
        common_dates
    )
)


# ============================================================
# 11. Raw Partial：Window-level连续指标
# ============================================================

raw_window_rows = []


for date, group in raw_partial.groupby(
    "network_date"
):

    same = (
        group[
            group[
                "same_industry"
            ]
        ]
    )


    cross = (
        group[
            ~group[
                "same_industry"
            ]
        ]
    )


    same_mean = (
        same[
            "abs_partial"
        ]
        .mean()
    )


    cross_mean = (
        cross[
            "abs_partial"
        ]
        .mean()
    )


    raw_window_rows.append(
        {
            "network_date":
                date,

            # -----------------------------------------------
            # 完全不经过Edge Selection
            # -----------------------------------------------

            "raw_mean_abs_partial_all":
                group[
                    "abs_partial"
                ]
                .mean(),

            "raw_median_abs_partial_all":
                group[
                    "abs_partial"
                ]
                .median(),

            "raw_same_mean_abs_partial":
                same_mean,

            "raw_cross_mean_abs_partial":
                cross_mean,

            "raw_same_cross_strength_gap":
                (
                    same_mean
                    -
                    cross_mean
                ),

            "raw_same_cross_strength_ratio":
                (
                    same_mean
                    /
                    cross_mean
                    if cross_mean > 0
                    else np.nan
                )
        }
    )


raw_window_metrics = pd.DataFrame(
    raw_window_rows
)


# ============================================================
# 12. Density-matched Partial：Window-level结构组成
# ============================================================

matched_window_rows = []


for date, group in matched_partial.groupby(
    "network_date"
):

    selected = (
        group[
            group[
                "selected_density_matched"
            ]
        ]
    )


    edge_count = len(
        selected
    )


    same_selected = (
        selected[
            selected[
                "same_industry"
            ]
        ]
    )


    cross_selected = (
        selected[
            ~selected[
                "same_industry"
            ]
        ]
    )


    same_edges = len(
        same_selected
    )


    cross_edges = len(
        cross_selected
    )


    matched_window_rows.append(
        {
            "network_date":
                date,

            # -----------------------------------------------
            # edge_count本身继承GLasso density target，
            # 不作为独立Regime-validation证据
            # -----------------------------------------------

            "matched_edge_count":
                edge_count,

            "matched_same_edges":
                same_edges,

            "matched_cross_edges":
                cross_edges,

            "matched_same_ratio":
                (
                    same_edges
                    /
                    edge_count
                    if edge_count > 0
                    else np.nan
                ),

            "matched_cross_ratio":
                (
                    cross_edges
                    /
                    edge_count
                    if edge_count > 0
                    else np.nan
                ),

            "matched_mean_abs_partial_selected":
                selected[
                    "abs_partial"
                ]
                .mean(),

            "matched_same_mean_abs_partial_selected":
                (
                    same_selected[
                        "abs_partial"
                    ]
                    .mean()
                    if len(
                        same_selected
                    ) > 0
                    else np.nan
                ),

            "matched_cross_mean_abs_partial_selected":
                (
                    cross_selected[
                        "abs_partial"
                    ]
                    .mean()
                    if len(
                        cross_selected
                    ) > 0
                    else np.nan
                )
        }
    )


matched_window_metrics = pd.DataFrame(
    matched_window_rows
)


# ============================================================
# 13. 合并Regime Labels
# ============================================================

window_metrics = (
    regime_assignment
    .merge(
        raw_window_metrics,
        on="network_date",
        how="inner"
    )
    .merge(
        matched_window_metrics,
        on="network_date",
        how="inner"
    )
    .sort_values(
        "network_date"
    )
)


window_metrics.to_csv(
    WINDOW_METRICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. Regime Summary
# ============================================================

metric_columns = [
    "raw_mean_abs_partial_all",
    "raw_same_mean_abs_partial",
    "raw_cross_mean_abs_partial",
    "raw_same_cross_strength_gap",
    "raw_same_cross_strength_ratio",

    "matched_same_edges",
    "matched_cross_edges",
    "matched_same_ratio",
    "matched_cross_ratio",

    "matched_mean_abs_partial_selected",
    "matched_same_mean_abs_partial_selected",
    "matched_cross_mean_abs_partial_selected"
]


summary_rows = []


regime_order = (
    window_metrics[
        [
            "regime",
            "network_date"
        ]
    ]
    .groupby(
        "regime",
        as_index=False
    )[
        "network_date"
    ]
    .min()
    .sort_values(
        "network_date"
    )[
        "regime"
    ]
    .tolist()
)


for regime in regime_order:

    group = (
        window_metrics[
            window_metrics[
                "regime"
            ]
            ==
            regime
        ]
    )


    row = {
        "regime":
            regime,

        "n_windows":
            len(
                group
            ),

        "start_date":
            group[
                "network_date"
            ]
            .min(),

        "end_date":
            group[
                "network_date"
            ]
            .max()
    }


    for col in metric_columns:

        row[
            f"mean_{col}"
        ] = (
            group[
                col
            ]
            .mean()
        )


        row[
            f"sd_{col}"
        ] = (
            group[
                col
            ]
            .std(
                ddof=1
            )
        )


    summary_rows.append(
        row
    )


regime_summary = pd.DataFrame(
    summary_rows
)


regime_summary.to_csv(
    REGIME_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Adjacent Regime Transitions
# ============================================================

# 标准化时使用所有window的总体标准差；
# 这里只是descriptive standardized difference，
# 不是独立样本效应量检验。
overall_sd = {
    col:
        window_metrics[
            col
        ]
        .std(
            ddof=1
        )
    for col in metric_columns
}


transition_rows = []


for k in range(
    1,
    len(
        regime_order
    )
):

    before_regime = (
        regime_order[
            k - 1
        ]
    )


    after_regime = (
        regime_order[
            k
        ]
    )


    before = (
        window_metrics[
            window_metrics[
                "regime"
            ]
            ==
            before_regime
        ]
    )


    after = (
        window_metrics[
            window_metrics[
                "regime"
            ]
            ==
            after_regime
        ]
    )


    transition = {
        "transition":
            f"{before_regime}->{after_regime}",

        "before_regime":
            before_regime,

        "after_regime":
            after_regime,

        "before_n_windows":
            len(
                before
            ),

        "after_n_windows":
            len(
                after
            ),

        "boundary_before_date":
            before[
                "network_date"
            ]
            .max(),

        "boundary_after_date":
            after[
                "network_date"
            ]
            .min()
    }


    for col in metric_columns:

        before_mean = (
            before[
                col
            ]
            .mean()
        )


        after_mean = (
            after[
                col
            ]
            .mean()
        )


        delta = (
            after_mean
            -
            before_mean
        )


        sd_all = (
            overall_sd[
                col
            ]
        )


        transition[
            f"before_{col}"
        ] = (
            before_mean
        )


        transition[
            f"after_{col}"
        ] = (
            after_mean
        )


        transition[
            f"delta_{col}"
        ] = (
            delta
        )


        transition[
            f"standardized_delta_{col}"
        ] = (
            delta
            /
            sd_all
            if (
                pd.notna(
                    sd_all
                )
                and
                sd_all > 0
            )
            else np.nan
        )


    transition_rows.append(
        transition
    )


transition_summary = pd.DataFrame(
    transition_rows
)


transition_summary.to_csv(
    TRANSITION_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 专门提取R3 -> R4
# ============================================================

# 兼容regime写成R3/R4或者3/4的情况
def locate_regime(
    possible_names
):

    for name in possible_names:

        if name in regime_order:
            return name

    return None


r3_name = locate_regime(
    [
        "R3",
        "3",
        "Regime 3",
        "Regime3"
    ]
)


r4_name = locate_regime(
    [
        "R4",
        "4",
        "Regime 4",
        "Regime4"
    ]
)


if (
    r3_name is not None
    and
    r4_name is not None
):

    transition_name = (
        f"{r3_name}->{r4_name}"
    )


    r3_r4 = (
        transition_summary[
            transition_summary[
                "transition"
            ]
            ==
            transition_name
        ]
        .copy()
    )


    # --------------------------------------------------------
    # 增加几个容易解释的Validation Indicators
    #
    # 注意：这些都是描述性的方向验证，不是p-value。
    # --------------------------------------------------------

    if len(
        r3_r4
    ) == 1:

        r3_r4[
            "V1_raw_industry_strength_gap_increases"
        ] = (
            r3_r4[
                "delta_raw_same_cross_strength_gap"
            ]
            >
            0
        )


        r3_r4[
            "V2_raw_industry_strength_ratio_increases"
        ] = (
            r3_r4[
                "delta_raw_same_cross_strength_ratio"
            ]
            >
            0
        )


        r3_r4[
            "V3_matched_same_ratio_increases"
        ] = (
            r3_r4[
                "delta_matched_same_ratio"
            ]
            >
            0
        )


        r3_r4[
            "V4_raw_overall_partial_strength_increases"
        ] = (
            r3_r4[
                "delta_raw_mean_abs_partial_all"
            ]
            >
            0
        )


        validation_cols = [
            "V1_raw_industry_strength_gap_increases",
            "V2_raw_industry_strength_ratio_increases",
            "V3_matched_same_ratio_increases",
            "V4_raw_overall_partial_strength_increases"
        ]


        r3_r4[
            "n_directional_validation_signals"
        ] = (
            r3_r4[
                validation_cols
            ]
            .sum(
                axis=1
            )
        )


    r3_r4.to_csv(
        R3_R4_FILE,
        index=False,
        encoding="utf-8-sig"
    )


else:

    print(
        "警告：无法自动识别R3和R4，"
        "不会生成R3_R4专门文件。"
    )


# ============================================================
# 17. 屏幕输出
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 5: Regime Cross-validation"
)

print(
    "============================================"
)


print(
    "\n--- Regime Summary ---"
)


display_cols = [
    "regime",
    "n_windows",

    "mean_raw_mean_abs_partial_all",
    "mean_raw_same_mean_abs_partial",
    "mean_raw_cross_mean_abs_partial",
    "mean_raw_same_cross_strength_gap",
    "mean_raw_same_cross_strength_ratio",

    "mean_matched_same_ratio",
    "mean_matched_mean_abs_partial_selected"
]


print(
    regime_summary[
        display_cols
    ]
    .to_string(
        index=False
    )
)


print(
    "\n--- Adjacent Regime Transitions ---"
)


transition_display_cols = [
    "transition",

    "delta_raw_mean_abs_partial_all",

    "delta_raw_same_cross_strength_gap",

    "delta_raw_same_cross_strength_ratio",

    "delta_matched_same_ratio",

    "delta_matched_mean_abs_partial_selected"
]


print(
    transition_summary[
        transition_display_cols
    ]
    .to_string(
        index=False
    )
)


if (
    r3_name is not None
    and
    r4_name is not None
):

    print(
        "\n--- R3 -> R4 Cross-validation ---"
    )


    print(
        r3_r4.to_string(
            index=False
        )
    )


# ============================================================
# 18. 图1：
# Raw Partial Same vs Cross Strength by Regime
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    window_metrics[
        "network_date"
    ],
    window_metrics[
        "raw_same_mean_abs_partial"
    ],
    marker="o",
    label="Same-industry"
)


ax.plot(
    window_metrics[
        "network_date"
    ],
    window_metrics[
        "raw_cross_mean_abs_partial"
    ],
    marker="o",
    label="Cross-industry"
)


# Regime boundary
for i in range(
    1,
    len(
        regime_order
    )
):

    boundary = (
        window_metrics[
            window_metrics[
                "regime"
            ]
            ==
            regime_order[
                i
            ]
        ][
            "network_date"
        ]
        .min()
    )


    ax.axvline(
        boundary,
        linestyle="--",
        linewidth=1
    )


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Mean Absolute Partial Correlation"
)


ax.set_title(
    "Raw Rolling Partial Correlation Across GLasso-defined Regimes"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_validation_raw_partial_strength.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 19. 图2：Raw Industry Strength Ratio
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    window_metrics[
        "network_date"
    ],
    window_metrics[
        "raw_same_cross_strength_ratio"
    ],
    marker="o"
)


ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1
)


for i in range(
    1,
    len(
        regime_order
    )
):

    boundary = (
        window_metrics[
            window_metrics[
                "regime"
            ]
            ==
            regime_order[
                i
            ]
        ][
            "network_date"
        ]
        .min()
    )


    ax.axvline(
        boundary,
        linestyle="--",
        linewidth=1
    )


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Same / Cross Mean |Partial|"
)


ax.set_title(
    "Industry Concentration in Raw Partial Correlations"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_validation_raw_strength_ratio.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. 图3：Density-matched Same-edge Ratio
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    window_metrics[
        "network_date"
    ],
    window_metrics[
        "matched_same_ratio"
    ],
    marker="o"
)


for i in range(
    1,
    len(
        regime_order
    )
):

    boundary = (
        window_metrics[
            window_metrics[
                "regime"
            ]
            ==
            regime_order[
                i
            ]
        ][
            "network_date"
        ]
        .min()
    )


    ax.axvline(
        boundary,
        linestyle="--",
        linewidth=1
    )


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Same-industry Edge Share"
)


ax.set_title(
    "Density-matched Partial Network Composition Across Regimes"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_validation_matched_same_ratio.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. 图4：Regime-level Raw Strength Gap
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


ax.bar(
    regime_summary[
        "regime"
    ],
    regime_summary[
        "mean_raw_same_cross_strength_gap"
    ]
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Regime"
)


ax.set_ylabel(
    "Same - Cross Mean |Partial|"
)


ax.set_title(
    "Raw Partial Industry Strength Gap by Regime"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_validation_regime_strength_gap.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 完成
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 5完成"
)

print(
    "============================================"
)


for path in [
    WINDOW_METRICS_FILE,
    REGIME_SUMMARY_FILE,
    TRANSITION_SUMMARY_FILE,
    R3_R4_FILE
]:

    if path.exists():
        print(
            path
        )
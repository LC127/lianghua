from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径
# ============================================================

PROJECT_DIR = Path(
    "stock_network"
)

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


INPUT_FILE = (
    PROCESSED_DIR
    / "dynamic_network_state_table.csv"
)


HIGH_CHANGE_FILE = (
    PROCESSED_DIR
    / "high_change_periods.csv"
)


FULL_SCORE_FILE = (
    PROCESSED_DIR
    / "network_change_scores.csv"
)


THRESHOLD_FILE = (
    PROCESSED_DIR
    / "high_change_threshold_summary.csv"
)


# ============================================================
# 2. 参数
# ============================================================

MODERATE_Z = 1.0

STRONG_Z = 1.5

VERY_STRONG_Z = 2.0


# ============================================================
# 3. 工具函数
# ============================================================

def standard_zscore(
    series: pd.Series
) -> pd.Series:

    series = pd.to_numeric(
        series,
        errors="coerce"
    )


    mean_value = (
        series.mean()
    )


    sd_value = (
        series.std(
            ddof=1
        )
    )


    if (
        pd.isna(sd_value)
        or
        sd_value <= 1e-12
    ):

        return pd.Series(
            np.nan,
            index=series.index
        )


    return (
        series
        -
        mean_value
    ) / sd_value


def robust_zscore(
    series: pd.Series
) -> pd.Series:

    series = pd.to_numeric(
        series,
        errors="coerce"
    )


    median_value = (
        series.median()
    )


    mad = (
        (
            series
            -
            median_value
        )
        .abs()
        .median()
    )


    if (
        pd.isna(mad)
        or
        mad <= 1e-12
    ):

        return pd.Series(
            np.nan,
            index=series.index
        )


    return (
        0.6745
        *
        (
            series
            -
            median_value
        )
        /
        mad
    )


# ============================================================
# 4. 读取State Table
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


df[
    "network_date"
] = pd.to_datetime(
    df[
        "network_date"
    ]
)


df = (
    df
    .sort_values(
        "network_date"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 5. 检查必要字段
# ============================================================

required_columns = [
    "network_date",
    "edge_count",
    "mean_abs_partial",
    "same_edges",
    "cross_edges",
    "same_industry_ratio",
    "turnover",
    "lost_edges",
    "gained_edges",
    "gross_edge_changes",
    "edge_count_change",
    "cross_edge_changes",
    "cross_change_share"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "缺少必要字段："
        f"{missing_columns}"
    )


# ============================================================
# 6. 转换数值字段
# ============================================================

numeric_columns = [
    "edge_count",
    "mean_abs_partial",
    "same_edges",
    "cross_edges",
    "same_industry_ratio",
    "turnover",
    "lost_edges",
    "gained_edges",
    "gross_edge_changes",
    "edge_count_change",
    "cross_edge_changes",
    "cross_change_share"
]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# 7. 补充基础变化指标
# ============================================================

# ------------------------------------------------------------
# 7.1 绝对净边数变化
# ------------------------------------------------------------

df[
    "abs_edge_count_change"
] = (
    df[
        "edge_count_change"
    ]
    .abs()
)


# ------------------------------------------------------------
# 7.2 Rewiring share
#
# 2 * min(Lost, Gained) / (Lost + Gained)
#
# = 0 : 纯扩张/纯收缩
# = 1 : 完全平衡的边替换
# ------------------------------------------------------------

rewiring_numerator = (
    2
    *
    np.minimum(
        df[
            "lost_edges"
        ],
        df[
            "gained_edges"
        ]
    )
)


df[
    "rewiring_share"
] = np.where(

    df[
        "gross_edge_changes"
    ]
    >
    0,

    rewiring_numerator
    /
    df[
        "gross_edge_changes"
    ],

    0.0
)


# ============================================================
# 8. 网络状态变量的一阶变化
# ============================================================

df[
    "delta_same_edges"
] = (
    df[
        "same_edges"
    ]
    .diff()
)


df[
    "delta_cross_edges"
] = (
    df[
        "cross_edges"
    ]
    .diff()
)


df[
    "delta_mean_abs_partial"
] = (
    df[
        "mean_abs_partial"
    ]
    .diff()
)


df[
    "delta_same_industry_ratio"
] = (
    df[
        "same_industry_ratio"
    ]
    .diff()
)


# ============================================================
# 9. 普通Z-score
# ============================================================

score_columns = [
    "turnover",
    "gross_edge_changes",
    "abs_edge_count_change",
    "cross_edge_changes"
]


for col in score_columns:

    df[
        f"{col}_z"
    ] = standard_zscore(
        df[
            col
        ]
    )


# ============================================================
# 10. Robust Z-score
# ============================================================

for col in [
    "turnover",
    "gross_edge_changes"
]:

    df[
        f"{col}_robust_z"
    ] = robust_zscore(
        df[
            col
        ]
    )


# ============================================================
# 11. 核心异常得分
#
# Turnover与Gross Change各占50%
#
# 注意：
# 这是描述性ranking score，不是统计检验统计量。
# ============================================================

df[
    "core_change_score"
] = (
    0.5
    *
    df[
        "turnover_z"
    ]
    +
    0.5
    *
    df[
        "gross_edge_changes_z"
    ]
)


# ============================================================
# 12. 最大异常程度
# ============================================================

df[
    "max_core_z"
] = (
    df[
        [
            "turnover_z",
            "gross_edge_changes_z"
        ]
    ]
    .max(
        axis=1
    )
)


# ============================================================
# 13. High-change Flags
# ============================================================

df[
    "high_change_moderate"
] = (
    df[
        "max_core_z"
    ]
    >=
    MODERATE_Z
)


df[
    "high_change_strong"
] = (
    df[
        "max_core_z"
    ]
    >=
    STRONG_Z
)


df[
    "high_change_very_strong"
] = (
    df[
        "max_core_z"
    ]
    >=
    VERY_STRONG_Z
)


# ------------------------------------------------------------
# 两个核心指标同时 >= 1
# ------------------------------------------------------------

df[
    "high_change_joint"
] = (
    (
        df[
            "turnover_z"
        ]
        >=
        MODERATE_Z
    )
    &
    (
        df[
            "gross_edge_changes_z"
        ]
        >=
        MODERATE_Z
    )
)


# ============================================================
# 14. Robust辅助Flag
# ============================================================

df[
    "robust_high_turnover"
] = (
    df[
        "turnover_robust_z"
    ]
    >=
    STRONG_Z
)


df[
    "robust_high_gross_change"
] = (
    df[
        "gross_edge_changes_robust_z"
    ]
    >=
    STRONG_Z
)


# ============================================================
# 15. 判断变化类型
# ============================================================

def classify_change_type(
    row
):

    gross = (
        row[
            "gross_edge_changes"
        ]
    )


    net = (
        row[
            "edge_count_change"
        ]
    )


    rewiring_share = (
        row[
            "rewiring_share"
        ]
    )


    if (
        pd.isna(
            gross
        )
    ):

        return (
            "Initial network"
        )


    if gross == 0:

        return (
            "Stable support"
        )


    # 大量Lost和Gained同时存在
    if rewiring_share >= 0.60:

        if net > 0:

            return (
                "Rewiring with expansion"
            )


        if net < 0:

            return (
                "Rewiring with contraction"
            )


        return (
            "Pure rewiring"
        )


    # Rewiring占比较低，主要由规模变化驱动
    if net > 0:

        return (
            "Expansion-dominant"
        )


    if net < 0:

        return (
            "Contraction-dominant"
        )


    return (
        "Balanced structural change"
    )


df[
    "change_type"
] = (
    df
    .apply(
        classify_change_type,
        axis=1
    )
)


# ============================================================
# 16. 判断变化主要来自Same还是Cross
# ============================================================

def classify_change_source(
    row
):

    share = (
        row[
            "cross_change_share"
        ]
    )


    if pd.isna(
        share
    ):

        return (
            "No change"
        )


    if share >= 0.80:

        return (
            "Cross-industry dominated"
        )


    if share <= 0.20:

        return (
            "Same-industry dominated"
        )


    return (
        "Mixed"
    )


df[
    "change_source"
] = (
    df
    .apply(
        classify_change_source,
        axis=1
    )
)


# ============================================================
# 17. Change Rank
#
# 核心得分越高，rank越靠前
# ============================================================

df[
    "change_rank"
] = (
    df[
        "core_change_score"
    ]
    .rank(
        method="min",
        ascending=False
    )
)


# ============================================================
# 18. 保存完整得分表
# ============================================================

df.to_csv(
    FULL_SCORE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. 提取High-change Periods
#
# 主表先保留Moderate及以上，
# 后续可以重点看Strong / Joint。
# ============================================================

high_change_df = (
    df[
        df[
            "high_change_moderate"
        ]
    ]
    .copy()
)


high_change_df = (
    high_change_df
    .sort_values(
        [
            "core_change_score",
            "turnover"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


high_change_df.to_csv(
    HIGH_CHANGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 保存阈值摘要
# ============================================================

threshold_rows = []


for col in [
    "turnover",
    "gross_edge_changes"
]:

    valid = (
        df[
            col
        ]
        .dropna()
    )


    mean_value = (
        valid.mean()
    )


    sd_value = (
        valid.std(
            ddof=1
        )
    )


    median_value = (
        valid.median()
    )


    mad_value = (
        (
            valid
            -
            median_value
        )
        .abs()
        .median()
    )


    threshold_rows.append(
        {
            "metric":
                col,

            "mean":
                mean_value,

            "sd":
                sd_value,

            "median":
                median_value,

            "mad":
                mad_value,

            "z1_raw_threshold":
                mean_value
                +
                MODERATE_Z
                *
                sd_value,

            "z1_5_raw_threshold":
                mean_value
                +
                STRONG_Z
                *
                sd_value,

            "z2_raw_threshold":
                mean_value
                +
                VERY_STRONG_Z
                *
                sd_value
        }
    )


threshold_df = pd.DataFrame(
    threshold_rows
)


threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. 屏幕输出
# ============================================================

print(
    "\n======================================"
)

print(
    "High-change Period Identification"
)

print(
    "======================================"
)


display_columns = [
    "network_date",
    "edge_count",
    "edge_count_change",
    "lost_edges",
    "gained_edges",
    "gross_edge_changes",
    "turnover",
    "rewiring_share",
    "cross_edge_changes",
    "cross_change_share",
    "turnover_z",
    "gross_edge_changes_z",
    "core_change_score",
    "change_rank",
    "change_type",
    "change_source",
    "high_change_strong",
    "high_change_joint"
]


print(
    high_change_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


print(
    "\nModerate High-change数量：",
    int(
        df[
            "high_change_moderate"
        ]
        .sum()
    )
)


print(
    "Strong High-change数量：",
    int(
        df[
            "high_change_strong"
        ]
        .sum()
    )
)


print(
    "Joint High-change数量：",
    int(
        df[
            "high_change_joint"
        ]
        .sum()
    )
)


# ============================================================
# 22. Top 10变化时期
# ============================================================

top10 = (
    df
    .dropna(
        subset=[
            "core_change_score"
        ]
    )
    .sort_values(
        "core_change_score",
        ascending=False
    )
    .head(10)
)


print(
    "\n======================================"
)

print(
    "Top 10 Network Changes"
)

print(
    "======================================"
)


print(
    top10[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 23. 图1：Turnover Z-score
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    df[
        "network_date"
    ],
    df[
        "turnover_z"
    ],
    marker="o"
)


ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=1.5,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Turnover Z-score"
)


ax.set_title(
    "Standardized Network Turnover"
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


TURNOVER_Z_FIGURE = (
    FIGURE_DIR
    / "high_change_turnover_z.png"
)


fig.savefig(
    TURNOVER_Z_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. 图2：Gross Edge Changes Z-score
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    df[
        "network_date"
    ],
    df[
        "gross_edge_changes_z"
    ],
    marker="o"
)


ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=1.5,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Gross Change Z-score"
)


ax.set_title(
    "Standardized Gross Edge Changes"
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


GROSS_Z_FIGURE = (
    FIGURE_DIR
    / "high_change_gross_z.png"
)


fig.savefig(
    GROSS_Z_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. 图3：
# Net Change vs Gross Change
#
# 用于区分收缩、扩张、Rewiring
# ============================================================

plot_df = (
    df
    .dropna(
        subset=[
            "edge_count_change",
            "gross_edge_changes"
        ]
    )
)


fig, ax = plt.subplots(
    figsize=(
        9,
        7
    )
)


ax.scatter(
    plot_df[
        "edge_count_change"
    ],
    plot_df[
        "gross_edge_changes"
    ]
)


for row in plot_df.itertuples():

    if (
        row.high_change_moderate
    ):

        ax.annotate(
            pd.Timestamp(
                row.network_date
            )
            .strftime(
                "%Y-%m-%d"
            ),

            (
                row.edge_count_change,
                row.gross_edge_changes
            ),

            xytext=(
                4,
                4
            ),

            textcoords="offset points",

            fontsize=8
        )


ax.axvline(
    x=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    r"Net Edge Change $\Delta E_t$"
)


ax.set_ylabel(
    "Gross Edge Changes"
)


ax.set_title(
    "Network Expansion, Contraction and Rewiring"
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


CHANGE_TYPE_FIGURE = (
    FIGURE_DIR
    / "high_change_net_vs_gross.png"
)


fig.savefig(
    CHANGE_TYPE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 26. 图4：Cross-industry Change Share
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    df[
        "network_date"
    ],
    df[
        "cross_change_share"
    ],
    marker="o"
)


ax.axhline(
    y=0.8,
    linestyle="--",
    linewidth=1
)


ax.set_ylim(
    0,
    1.05
)


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Cross-industry Share of Edge Changes"
)


ax.set_title(
    "Industry Composition of Dynamic Network Changes"
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


CROSS_SHARE_FIGURE = (
    FIGURE_DIR
    / "high_change_cross_industry_share.png"
)


fig.savefig(
    CROSS_SHARE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 27. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Stage 2完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for path in [
    FULL_SCORE_FILE,
    HIGH_CHANGE_FILE,
    THRESHOLD_FILE,
    TURNOVER_Z_FIGURE,
    GROSS_Z_FIGURE,
    CHANGE_TYPE_FIGURE,
    CROSS_SHARE_FIGURE
]:

    print(
        path
    )
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 0. 中文字体
# ============================================================

def set_chinese_font():

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP"
    ]

    installed = {
        font.name
        for font in font_manager.fontManager.ttflist
    }

    for font_name in candidates:

        if font_name in installed:

            plt.rcParams["font.sans-serif"] = [
                font_name
            ]

            plt.rcParams["axes.unicode_minus"] = False

            print(
                f"使用中文字体：{font_name}"
            )

            return

    print(
        "警告：未找到常见中文字体，"
        "中文标签可能无法正常显示。"
    )


set_chinese_font()


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


STATE_FILE = (
    PROCESSED_DIR
    / "dynamic_network_state_table.csv"
)


# Stage 2结果，用于Change Point验证
CHANGE_SCORE_FILE = (
    PROCESSED_DIR
    / "network_change_scores.csv"
)


# ============================================================
# 2. 输出
# ============================================================

ASSIGNMENT_FILE = (
    PROCESSED_DIR
    / "network_regime_assignment.csv"
)

REGIME_SUMMARY_FILE = (
    PROCESSED_DIR
    / "network_regime_summary.csv"
)

CHANGE_POINT_FILE = (
    PROCESSED_DIR
    / "network_change_point_results.csv"
)

MODEL_SELECTION_FILE = (
    PROCESSED_DIR
    / "network_regime_model_selection.csv"
)

STANDARDIZED_STATE_FILE = (
    PROCESSED_DIR
    / "network_state_standardized.csv"
)


# ============================================================
# 3. 参数
# ============================================================

# 最多考虑5个Regimes
MAX_REGIMES = 5

# 每个Regime至少3个network dates
MIN_REGIME_LENGTH = 3


# ============================================================
# 4. Regime Detection使用的变量
# ============================================================

FEATURES = [
    "same_edges",
    "cross_edges",
    "mean_abs_partial"
]


# ============================================================
# 5. 读取数据
# ============================================================

df = pd.read_csv(
    STATE_FILE
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
# 6. 检查字段
# ============================================================

required_columns = [
    "network_date",
    "edge_count",
    "same_edges",
    "cross_edges",
    "same_industry_ratio",
    "mean_abs_partial",
    "turnover",
    "gross_edge_changes",
    "edge_count_change",
    "lost_edges",
    "gained_edges",
    "cross_change_share"
]


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:

    raise ValueError(
        f"State Table缺少字段：{missing}"
    )


# ============================================================
# 7. 数值转换
# ============================================================

numeric_columns = [
    col
    for col in required_columns
    if col != "network_date"
]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# 8. 检查Regime Features
# ============================================================

if (
    df[
        FEATURES
    ]
    .isna()
    .any()
    .any()
):

    raise ValueError(
        "Regime Detection核心变量存在缺失值。"
    )


N = len(
    df
)

D = len(
    FEATURES
)


print(
    "\nNetwork dates：",
    N
)

print(
    "Regime features：",
    FEATURES
)

print(
    "Maximum regimes：",
    MAX_REGIMES
)

print(
    "Minimum regime length：",
    MIN_REGIME_LENGTH
)


# ============================================================
# 9. 标准化
# ============================================================

means = (
    df[
        FEATURES
    ]
    .mean()
)


sds = (
    df[
        FEATURES
    ]
    .std(
        ddof=1
    )
)


if (
    sds
    <=
    1e-12
).any():

    bad_features = (
        sds[
            sds
            <=
            1e-12
        ]
        .index
        .tolist()
    )

    raise ValueError(
        "以下变量几乎无变化，不能标准化："
        f"{bad_features}"
    )


Z_df = (
    df[
        FEATURES
    ]
    -
    means
) / sds


Z = Z_df.to_numpy(
    dtype=float
)


for feature in FEATURES:

    df[
        f"z_{feature}"
    ] = (
        Z_df[
            feature
        ]
    )


# ============================================================
# 10. Segment Cost的快速计算
#
# C(s,e)
# = sum ||Z_t - mean_segment||^2
#
# 这里使用[start, end)形式
# ============================================================

prefix_sum = np.vstack(
    [
        np.zeros(
            (
                1,
                D
            )
        ),
        np.cumsum(
            Z,
            axis=0
        )
    ]
)


prefix_sq_sum = np.concatenate(
    [
        np.array(
            [
                0.0
            ]
        ),
        np.cumsum(
            np.sum(
                Z ** 2,
                axis=1
            )
        )
    ]
)


def segment_cost(
    start: int,
    end: int
) -> float:

    """
    Cost for observations:
    start, ..., end-1
    """

    length = (
        end
        -
        start
    )


    if length <= 0:

        return np.inf


    segment_sum = (
        prefix_sum[
            end
        ]
        -
        prefix_sum[
            start
        ]
    )


    segment_sq_sum = (
        prefix_sq_sum[
            end
        ]
        -
        prefix_sq_sum[
            start
        ]
    )


    cost = (
        segment_sq_sum
        -
        np.dot(
            segment_sum,
            segment_sum
        )
        /
        length
    )


    # 数值误差可能产生极小负数
    return float(
        max(
            cost,
            0.0
        )
    )


# ============================================================
# 11. Dynamic Programming
#
# 对每个指定Regime数量R，
# 找到最小Within-Regime SSE的连续分段。
# ============================================================

def optimal_segmentation(
    n_regimes: int
):

    inf = np.inf


    dp = np.full(
        (
            n_regimes + 1,
            N + 1
        ),
        inf
    )


    back = np.full(
        (
            n_regimes + 1,
            N + 1
        ),
        -1,
        dtype=int
    )


    dp[
        0,
        0
    ] = 0.0


    for r in range(
        1,
        n_regimes + 1
    ):

        min_end = (
            r
            *
            MIN_REGIME_LENGTH
        )


        for end in range(
            min_end,
            N + 1
        ):

            min_start = (
                (
                    r - 1
                )
                *
                MIN_REGIME_LENGTH
            )


            max_start = (
                end
                -
                MIN_REGIME_LENGTH
            )


            for start in range(
                min_start,
                max_start + 1
            ):

                previous_cost = (
                    dp[
                        r - 1,
                        start
                    ]
                )


                if not np.isfinite(
                    previous_cost
                ):

                    continue


                candidate = (
                    previous_cost
                    +
                    segment_cost(
                        start,
                        end
                    )
                )


                if (
                    candidate
                    <
                    dp[
                        r,
                        end
                    ]
                ):

                    dp[
                        r,
                        end
                    ] = (
                        candidate
                    )


                    back[
                        r,
                        end
                    ] = (
                        start
                    )


    rss = (
        dp[
            n_regimes,
            N
        ]
    )


    if not np.isfinite(
        rss
    ):

        return (
            np.inf,
            []
        )


    # --------------------------------------------------------
    # 回溯Segment
    # --------------------------------------------------------

    segments = []

    end = N


    for r in range(
        n_regimes,
        0,
        -1
    ):

        start = (
            back[
                r,
                end
            ]
        )


        if start < 0:

            raise RuntimeError(
                "Dynamic Programming回溯失败。"
            )


        segments.append(
            (
                start,
                end
            )
        )


        end = start


    segments.reverse()


    return (
        rss,
        segments
    )


# ============================================================
# 12. 比较不同Regime数量
# ============================================================

model_rows = []

segmentation_dict = {}


max_feasible_regimes = min(
    MAX_REGIMES,
    N
    //
    MIN_REGIME_LENGTH
)


for R in range(
    1,
    max_feasible_regimes + 1
):

    rss, segments = (
        optimal_segmentation(
            R
        )
    )


    if not np.isfinite(
        rss
    ):

        continue


    # --------------------------------------------------------
    # BIC-style criterion
    #
    # 每个Regime有D个均值参数；
    # R-1个Change-point位置。
    # --------------------------------------------------------

    n_scalar_observations = (
        N
        *
        D
    )


    k_parameters = (
        R
        *
        D
        +
        (
            R - 1
        )
    )


    rss_safe = max(
        rss,
        1e-12
    )


    bic_style = (
        n_scalar_observations
        *
        np.log(
            rss_safe
            /
            n_scalar_observations
        )
        +
        k_parameters
        *
        np.log(
            N
        )
    )


    segmentation_dict[
        R
    ] = segments


    model_rows.append(
        {
            "n_regimes":
                R,

            "rss":
                rss,

            "n_parameters":
                k_parameters,

            "bic_style":
                bic_style,

            "segments":
                str(
                    segments
                )
        }
    )


model_df = pd.DataFrame(
    model_rows
)


# ============================================================
# 13. 选择最佳Regime数量
# ============================================================

best_row = (
    model_df
    .sort_values(
        "bic_style",
        ascending=True
    )
    .iloc[0]
)


BEST_R = int(
    best_row[
        "n_regimes"
    ]
)


BEST_SEGMENTS = (
    segmentation_dict[
        BEST_R
    ]
)


print(
    "\n======================================"
)

print(
    "Model Selection"
)

print(
    "======================================"
)


print(
    model_df.to_string(
        index=False
    )
)


print(
    "\n选择的Regime数量：",
    BEST_R
)


print(
    "最优Segments：",
    BEST_SEGMENTS
)


# ============================================================
# 14. 给每个Network Date分配Regime
# ============================================================

df[
    "regime"
] = np.nan


for regime_id, (
    start,
    end
) in enumerate(
    BEST_SEGMENTS,
    start=1
):

    df.loc[
        start:end - 1,
        "regime"
    ] = (
        regime_id
    )


df[
    "regime"
] = (
    df[
        "regime"
    ]
    .astype(int)
)


# ============================================================
# 15. 计算Regime Summary
# ============================================================

regime_rows = []


for regime_id, group in df.groupby(
    "regime"
):

    group = (
        group
        .sort_values(
            "network_date"
        )
    )


    # --------------------------------------------------------
    # within-regime turnover
    #
    # 当前日期的Turnover描述上一张网络->当前网络。
    # 所以该Regime第一张网络的Turnover实际上是跨Regime边界，
    # 不应该混入within-regime average。
    # --------------------------------------------------------

    regime_indices = (
        group.index
        .tolist()
    )


    within_turnover_values = []


    for idx in regime_indices:

        if idx == 0:

            continue


        if (
            df.loc[
                idx - 1,
                "regime"
            ]
            ==
            regime_id
        ):

            turnover_value = (
                df.loc[
                    idx,
                    "turnover"
                ]
            )


            if pd.notna(
                turnover_value
            ):

                within_turnover_values.append(
                    turnover_value
                )


    mean_within_turnover = (
        np.mean(
            within_turnover_values
        )
        if len(
            within_turnover_values
        ) > 0
        else np.nan
    )


    regime_rows.append(
        {
            "regime":
                regime_id,

            "start_date":
                group[
                    "network_date"
                ]
                .min(),

            "end_date":
                group[
                    "network_date"
                ]
                .max(),

            "n_networks":
                len(
                    group
                ),

            "mean_edge_count":
                group[
                    "edge_count"
                ]
                .mean(),

            "mean_same_edges":
                group[
                    "same_edges"
                ]
                .mean(),

            "mean_cross_edges":
                group[
                    "cross_edges"
                ]
                .mean(),

            "mean_same_industry_ratio":
                group[
                    "same_industry_ratio"
                ]
                .mean(),

            "mean_abs_partial":
                group[
                    "mean_abs_partial"
                ]
                .mean(),

            "mean_within_regime_turnover":
                mean_within_turnover,

            "sd_edge_count":
                group[
                    "edge_count"
                ]
                .std(
                    ddof=1
                ),

            "sd_cross_edges":
                group[
                    "cross_edges"
                ]
                .std(
                    ddof=1
                ),

            "sd_mean_abs_partial":
                group[
                    "mean_abs_partial"
                ]
                .std(
                    ddof=1
                )
        }
    )


regime_summary_df = pd.DataFrame(
    regime_rows
)


# ============================================================
# 16. Change Point Results
#
# Change Point定义为：
# 新Regime的第一张网络
# ============================================================

change_point_rows = []


for r in range(
    1,
    len(
        BEST_SEGMENTS
    )
):

    previous_segment = (
        BEST_SEGMENTS[
            r - 1
        ]
    )


    current_segment = (
        BEST_SEGMENTS[
            r
        ]
    )


    prev_start, prev_end = (
        previous_segment
    )

    curr_start, curr_end = (
        current_segment
    )


    boundary_idx = (
        curr_start
    )


    boundary_row = (
        df.iloc[
            boundary_idx
        ]
    )


    previous_regime = (
        df.iloc[
            prev_start:prev_end
        ]
    )


    current_regime = (
        df.iloc[
            curr_start:curr_end
        ]
    )


    # --------------------------------------------------------
    # Standardized Regime Mean Jump
    # --------------------------------------------------------

    previous_mean_z = (
        previous_regime[
            [
                f"z_{x}"
                for x in FEATURES
            ]
        ]
        .mean()
        .to_numpy()
    )


    current_mean_z = (
        current_regime[
            [
                f"z_{x}"
                for x in FEATURES
            ]
        ]
        .mean()
        .to_numpy()
    )


    regime_jump_norm = float(
        np.linalg.norm(
            current_mean_z
            -
            previous_mean_z
        )
    )


    change_point_rows.append(
        {
            "change_point_id":
                r,

            "previous_regime":
                r,

            "new_regime":
                r + 1,

            "previous_regime_end_date":
                previous_regime[
                    "network_date"
                ]
                .max(),

            "new_regime_start_date":
                current_regime[
                    "network_date"
                ]
                .min(),

            # -----------------------------------------------
            # Regime-level Mean Difference
            # -----------------------------------------------

            "delta_regime_mean_same_edges":
                (
                    current_regime[
                        "same_edges"
                    ]
                    .mean()
                    -
                    previous_regime[
                        "same_edges"
                    ]
                    .mean()
                ),

            "delta_regime_mean_cross_edges":
                (
                    current_regime[
                        "cross_edges"
                    ]
                    .mean()
                    -
                    previous_regime[
                        "cross_edges"
                    ]
                    .mean()
                ),

            "delta_regime_mean_edge_count":
                (
                    current_regime[
                        "edge_count"
                    ]
                    .mean()
                    -
                    previous_regime[
                        "edge_count"
                    ]
                    .mean()
                ),

            "delta_regime_mean_abs_partial":
                (
                    current_regime[
                        "mean_abs_partial"
                    ]
                    .mean()
                    -
                    previous_regime[
                        "mean_abs_partial"
                    ]
                    .mean()
                ),

            "delta_regime_mean_same_ratio":
                (
                    current_regime[
                        "same_industry_ratio"
                    ]
                    .mean()
                    -
                    previous_regime[
                        "same_industry_ratio"
                    ]
                    .mean()
                ),

            "regime_jump_norm":
                regime_jump_norm,

            # -----------------------------------------------
            # Boundary Transition Information
            # -----------------------------------------------

            "boundary_turnover":
                boundary_row[
                    "turnover"
                ],

            "boundary_gross_edge_changes":
                boundary_row[
                    "gross_edge_changes"
                ],

            "boundary_net_edge_change":
                boundary_row[
                    "edge_count_change"
                ],

            "boundary_lost_edges":
                boundary_row[
                    "lost_edges"
                ],

            "boundary_gained_edges":
                boundary_row[
                    "gained_edges"
                ],

            "boundary_cross_change_share":
                boundary_row[
                    "cross_change_share"
                ]
        }
    )


change_point_df = pd.DataFrame(
    change_point_rows
)


# ============================================================
# 17. 如果Stage 2结果存在，合并High-change信息
# ============================================================

if CHANGE_SCORE_FILE.exists():

    stage2_df = pd.read_csv(
        CHANGE_SCORE_FILE
    )


    stage2_df[
        "network_date"
    ] = pd.to_datetime(
        stage2_df[
            "network_date"
        ]
    )


    stage2_columns = [
        "network_date"
    ]


    optional_cols = [
        "turnover_z",
        "gross_edge_changes_z",
        "core_change_score",
        "change_rank",
        "change_type",
        "change_source",
        "high_change_moderate",
        "high_change_strong",
        "high_change_joint"
    ]


    for col in optional_cols:

        if col in stage2_df.columns:

            stage2_columns.append(
                col
            )


    stage2_small = (
        stage2_df[
            stage2_columns
        ]
        .copy()
    )


    change_point_df = (
        change_point_df
        .merge(
            stage2_small,

            left_on=(
                "new_regime_start_date"
            ),

            right_on=(
                "network_date"
            ),

            how="left"
        )
    )


    if (
        "network_date"
        in change_point_df.columns
    ):

        change_point_df = (
            change_point_df
            .drop(
                columns=[
                    "network_date"
                ]
            )
        )


# ============================================================
# 18. 保存标准化状态数据
# ============================================================

standardized_output_columns = [
    "network_date",
    "regime"
]


for feature in FEATURES:

    standardized_output_columns.extend(
        [
            feature,
            f"z_{feature}"
        ]
    )


df[
    standardized_output_columns
].to_csv(
    STANDARDIZED_STATE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. 保存结果
# ============================================================

df.to_csv(
    ASSIGNMENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


regime_summary_df.to_csv(
    REGIME_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


change_point_df.to_csv(
    CHANGE_POINT_FILE,
    index=False,
    encoding="utf-8-sig"
)


model_df.to_csv(
    MODEL_SELECTION_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 输出结果
# ============================================================

print(
    "\n======================================"
)

print(
    "Regime Summary"
)

print(
    "======================================"
)


print(
    regime_summary_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Change Points"
)

print(
    "======================================"
)


if len(
    change_point_df
) > 0:

    print(
        change_point_df.to_string(
            index=False
        )
    )

else:

    print(
        "未识别到Change Point。"
    )


# ============================================================
# 21. 图1：Model Selection
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        6
    )
)


ax.plot(
    model_df[
        "n_regimes"
    ],
    model_df[
        "bic_style"
    ],
    marker="o"
)


ax.axvline(
    x=BEST_R,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Number of Regimes"
)


ax.set_ylabel(
    "BIC-style Criterion"
)


ax.set_title(
    "Regime Number Selection"
)


ax.set_xticks(
    model_df[
        "n_regimes"
    ]
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


MODEL_SELECTION_FIGURE = (
    FIGURE_DIR
    / "regime_model_selection.png"
)


fig.savefig(
    MODEL_SELECTION_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 图2：Same / Cross Edge Count + Regime Boundaries
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        13,
        7
    )
)


ax.plot(
    df[
        "network_date"
    ],
    df[
        "same_edges"
    ],
    marker="o",
    label="同行业边"
)


ax.plot(
    df[
        "network_date"
    ],
    df[
        "cross_edges"
    ],
    marker="o",
    label="跨行业边"
)


for _, row in change_point_df.iterrows():

    ax.axvline(
        x=row[
            "new_regime_start_date"
        ],
        linestyle="--",
        linewidth=1
    )


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Number of Edges"
)


ax.set_title(
    "Same/Cross Edge Counts and Regime Boundaries"
)


ax.legend()

ax.grid(
    alpha=0.3
)


fig.tight_layout()


EDGE_REGIME_FIGURE = (
    FIGURE_DIR
    / "regime_same_cross_edges.png"
)


fig.savefig(
    EDGE_REGIME_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 图3：Mean Absolute Partial + Regime Boundaries
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        13,
        6
    )
)


ax.plot(
    df[
        "network_date"
    ],
    df[
        "mean_abs_partial"
    ],
    marker="o"
)


for _, row in change_point_df.iterrows():

    ax.axvline(
        x=row[
            "new_regime_start_date"
        ],
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
    "Conditional Association Strength Across Regimes"
)


ax.grid(
    alpha=0.3
)


fig.tight_layout()


STRENGTH_REGIME_FIGURE = (
    FIGURE_DIR
    / "regime_mean_abs_partial.png"
)


fig.savefig(
    STRENGTH_REGIME_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. 图4：标准化Network State
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        13,
        7
    )
)


for feature in FEATURES:

    ax.plot(
        df[
            "network_date"
        ],
        df[
            f"z_{feature}"
        ],
        marker="o",
        label=feature
    )


for _, row in change_point_df.iterrows():

    ax.axvline(
        x=row[
            "new_regime_start_date"
        ],
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
    "Standardized Network State"
)


ax.set_title(
    "Multivariate Dynamic Network State Across Regimes"
)


ax.legend()

ax.grid(
    alpha=0.3
)


fig.tight_layout()


STATE_REGIME_FIGURE = (
    FIGURE_DIR
    / "regime_standardized_state.png"
)


fig.savefig(
    STATE_REGIME_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Stage 3完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for path in [
    ASSIGNMENT_FILE,
    REGIME_SUMMARY_FILE,
    CHANGE_POINT_FILE,
    MODEL_SELECTION_FILE,
    STANDARDIZED_STATE_FILE,
    MODEL_SELECTION_FIGURE,
    EDGE_REGIME_FIGURE,
    STRENGTH_REGIME_FIGURE,
    STATE_REGIME_FIGURE
]:

    print(
        path
    )
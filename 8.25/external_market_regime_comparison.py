from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 项目路径
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
# 2. 输入文件
# ============================================================

EXTERNAL_INDEX_FILE = (
    PROCESSED_DIR
    / "external_index_daily.csv"
)

REGIME_FILE = (
    PROCESSED_DIR
    / "network_regime_assignment.csv"
)


# ============================================================
# 3. 输出文件
# ============================================================

WINDOW_METRICS_FILE = (
    PROCESSED_DIR
    / "external_market_state_window_metrics.csv"
)

REGIME_SUMMARY_FILE = (
    PROCESSED_DIR
    / "external_market_state_regime_summary.csv"
)

TRANSITION_SUMMARY_FILE = (
    PROCESSED_DIR
    / "external_market_state_transition_summary.csv"
)

R3_R4_FILE = (
    PROCESSED_DIR
    / "external_market_state_R3_R4.csv"
)


# ============================================================
# 4. 参数
# ============================================================

MARKET_LOOKBACK = 20

ACTIVITY_LONG_WINDOW = 60

TRADING_DAYS_PER_YEAR = 252


# 主要分析CSI300和CSI500。
# SSE保留为辅助benchmark。
MAIN_INDICES = [
    "CSI300",
    "CSI500"
]

INCLUDE_SSE = True


# ============================================================
# 5. Maximum Drawdown
# ============================================================

def calculate_max_drawdown(
    simple_returns: pd.Series
) -> float:
    """
    根据simple return计算窗口内最大回撤。

    Wealth_t = prod(1 + R_t)
    DD_t = Wealth_t / RunningMax_t - 1
    """

    wealth = (
        1.0
        +
        simple_returns
    ).cumprod()

    running_max = (
        wealth
        .cummax()
    )

    drawdown = (
        wealth
        /
        running_max
        -
        1.0
    )

    return float(
        drawdown.min()
    )


# ============================================================
# 6. 读取外部指数数据
# ============================================================

index_df = pd.read_csv(
    EXTERNAL_INDEX_FILE,
    dtype={
        "index_key": str,
        "index_code": str
    }
)


required_columns = [
    "date",
    "index_key",
    "simple_return",
    "close"
]


missing = [
    col
    for col in required_columns
    if col not in index_df.columns
]


if missing:

    raise ValueError(
        f"external_index_daily.csv缺少必要列：{missing}"
    )


index_df[
    "date"
] = pd.to_datetime(
    index_df[
        "date"
    ]
)


index_df[
    "simple_return"
] = pd.to_numeric(
    index_df[
        "simple_return"
    ],
    errors="coerce"
)


if "amount" in index_df.columns:

    index_df[
        "amount"
    ] = pd.to_numeric(
        index_df[
            "amount"
        ],
        errors="coerce"
    )


index_df = (
    index_df
    .sort_values(
        [
            "index_key",
            "date"
        ]
    )
    .drop_duplicates(
        subset=[
            "index_key",
            "date"
        ],
        keep="last"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 7. 确定分析指数
# ============================================================

available_indices = (
    index_df[
        "index_key"
    ]
    .dropna()
    .unique()
    .tolist()
)


analysis_indices = [
    x
    for x in MAIN_INDICES
    if x in available_indices
]


if (
    INCLUDE_SSE
    and
    "SSE" in available_indices
):

    analysis_indices.append(
        "SSE"
    )


if not all(
    x in analysis_indices
    for x in MAIN_INDICES
):

    raise ValueError(
        "必须至少包含CSI300和CSI500。"
    )


print(
    "分析指数：",
    analysis_indices
)


# ============================================================
# 8. 读取Network Regime
# ============================================================

regime_raw = pd.read_csv(
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


date_col = next(
    (
        x
        for x in date_candidates
        if x in regime_raw.columns
    ),
    None
)


regime_col = next(
    (
        x
        for x in regime_candidates
        if x in regime_raw.columns
    ),
    None
)


if (
    date_col is None
    or
    regime_col is None
):

    raise ValueError(
        "network_regime_assignment.csv中"
        "无法识别日期或Regime字段。"
    )


regime_df = (
    regime_raw[
        [
            date_col,
            regime_col
        ]
    ]
    .rename(
        columns={
            date_col:
                "network_date",

            regime_col:
                "regime"
        }
    )
    .copy()
)


regime_df[
    "network_date"
] = pd.to_datetime(
    regime_df[
        "network_date"
    ]
)


regime_df[
    "regime"
] = (
    regime_df[
        "regime"
    ]
    .astype(str)
    .str.strip()
)


regime_df = (
    regime_df
    .drop_duplicates(
        subset="network_date"
    )
    .sort_values(
        "network_date"
    )
    .reset_index(
        drop=True
    )
)


print(
    "Network dates：",
    len(
        regime_df
    )
)


# ============================================================
# 9. 计算每个Network Date × Index的20日市场状态
#
# 非常重要：
#
# 只使用 date <= network_date 的信息，
# 避免look-ahead bias。
# ============================================================

window_rows = []


for index_key in analysis_indices:

    temp = (
        index_df[
            index_df[
                "index_key"
            ]
            ==
            index_key
        ]
        .copy()
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )


    for _, regime_row in regime_df.iterrows():

        network_date = (
            regime_row[
                "network_date"
            ]
        )

        regime = (
            regime_row[
                "regime"
            ]
        )


        # ----------------------------------------------------
        # 只能找 <= network_date 的最后一个交易日
        # ----------------------------------------------------

        valid_positions = np.flatnonzero(
            (
                temp[
                    "date"
                ]
                <=
                network_date
            )
            .to_numpy()
        )


        if len(
            valid_positions
        ) == 0:

            continue


        end_pos = int(
            valid_positions[
                -1
            ]
        )


        end_date = (
            temp.loc[
                end_pos,
                "date"
            ]
        )


        # ----------------------------------------------------
        # 20日窗口
        # ----------------------------------------------------

        start_pos = (
            end_pos
            -
            MARKET_LOOKBACK
            +
            1
        )


        if start_pos < 0:

            continue


        window_20 = (
            temp
            .iloc[
                start_pos:
                end_pos + 1
            ]
            .copy()
        )


        returns_20 = (
            window_20[
                "simple_return"
            ]
            .dropna()
        )


        if len(
            returns_20
        ) != MARKET_LOOKBACK:

            print(
                f"WARNING: "
                f"{index_key} {network_date.date()} "
                f"20日收益样本数={len(returns_20)}"
            )

            continue


        # ----------------------------------------------------
        # 1. 20日累计收益
        # ----------------------------------------------------

        cumulative_return_20 = (
            (
                1.0
                +
                returns_20
            )
            .prod()
            -
            1.0
        )


        # ----------------------------------------------------
        # 2. 年化波动率
        # ----------------------------------------------------

        annualized_volatility_20 = (
            returns_20
            .std(
                ddof=1
            )
            *
            np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )


        # ----------------------------------------------------
        # 3. Mean Absolute Return
        # ----------------------------------------------------

        mean_abs_return_20 = (
            returns_20
            .abs()
            .mean()
        )


        # ----------------------------------------------------
        # 4. Negative-day Share
        # ----------------------------------------------------

        negative_day_share_20 = (
            (
                returns_20
                <
                0
            )
            .mean()
        )


        # ----------------------------------------------------
        # 5. Maximum Drawdown
        # ----------------------------------------------------

        max_drawdown_20 = (
            calculate_max_drawdown(
                returns_20
            )
        )


        # ----------------------------------------------------
        # 6. Positive-return day share
        #
        # 与NegativeDayShare互补，
        # 留作描述，不作为独立证据。
        # ----------------------------------------------------

        positive_day_share_20 = (
            (
                returns_20
                >
                0
            )
            .mean()
        )


        # ----------------------------------------------------
        # 7. 日内High-Low Range
        #
        # 仅辅助描述市场波动。
        # ----------------------------------------------------

        if (
            "high_low_range"
            in
            window_20.columns
        ):

            mean_high_low_range_20 = (
                pd.to_numeric(
                    window_20[
                        "high_low_range"
                    ],
                    errors="coerce"
                )
                .mean()
            )

        else:

            mean_high_low_range_20 = np.nan


        # ----------------------------------------------------
        # 8. 成交额活跃度
        # Trading Activity：Volume 20d / 60d
        # volume_activity_20_60 > 0
        #   最近20日成交量高于60日平均
        #
        # volume_activity_20_60 < 0
        #   最近20日成交量低于60日平均
        # ----------------------------------------------------

        volume_mean_20 = np.nan
        volume_mean_60 = np.nan
        volume_activity_20_60 = np.nan


        if (
            "volume"
            in temp.columns
            and
            end_pos
            -
            ACTIVITY_LONG_WINDOW
            +
            1
            >=
            0
        ):

            window_60 = (
                temp
                .iloc[
                    end_pos
                    -
                    ACTIVITY_LONG_WINDOW
                    +
                    1:
                    end_pos
                    +
                    1
                ]
                .copy()
            )


            volume_20 = pd.to_numeric(
                window_20[
                    "volume"
                ],
                errors="coerce"
            )


            volume_60 = pd.to_numeric(
                window_60[
                    "volume"
                ],
                errors="coerce"
            )


            volume_mean_20 = (
                volume_20
                .mean()
            )


            volume_mean_60 = (
                volume_60
                .mean()
            )


            if (
                pd.notna(
                    volume_mean_60
                )
                and
                volume_mean_60
                >
                0
            ):

                volume_activity_20_60 = (
                    volume_mean_20
                    /
                    volume_mean_60
                    -
                    1.0
                )


        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        window_rows.append(
            {
                "network_date":
                    network_date,

                "regime":
                    regime,

                "index_key":
                    index_key,

                "market_data_end_date":
                    end_date,

                "market_window_start":
                    window_20[
                        "date"
                    ]
                    .iloc[
                        0
                    ],

                "market_window_end":
                    window_20[
                        "date"
                    ]
                    .iloc[
                        -1
                    ],

                "market_lookback":
                    MARKET_LOOKBACK,

                "cumulative_return_20":
                    cumulative_return_20,

                "annualized_volatility_20":
                    annualized_volatility_20,

                "mean_abs_return_20":
                    mean_abs_return_20,

                "negative_day_share_20":
                    negative_day_share_20,

                "positive_day_share_20":
                    positive_day_share_20,

                "max_drawdown_20":
                    max_drawdown_20,

                "mean_high_low_range_20":
                    mean_high_low_range_20,

                "volume_mean_20":
                    volume_mean_20,

                "volume_mean_60":
                    volume_mean_60,

                "volume_activity_20_60":
                    volume_activity_20_60
            }
        )


window_metrics = pd.DataFrame(
    window_rows
)


window_metrics = (
    window_metrics
    .sort_values(
        [
            "network_date",
            "index_key"
        ]
    )
    .reset_index(
        drop=True
    )
)


window_metrics.to_csv(
    WINDOW_METRICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. Market-state指标
# ============================================================

MARKET_METRICS = [
    "cumulative_return_20",
    "annualized_volatility_20",
    "mean_abs_return_20",
    "negative_day_share_20",
    "max_drawdown_20",
    "mean_high_low_range_20",
    "volume_activity_20_60"
]


# ============================================================
# 11. Regime顺序
# ============================================================

regime_order = (
    regime_df[
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


# ============================================================
# 12. Regime-level Summary
#
# 每个 Regime × 每个 Index 一行
# ============================================================

summary_rows = []


for index_key in analysis_indices:

    temp_index = (
        window_metrics[
            window_metrics[
                "index_key"
            ]
            ==
            index_key
        ]
    )


    for regime in regime_order:

        group = (
            temp_index[
                temp_index[
                    "regime"
                ]
                ==
                regime
            ]
        )


        if group.empty:

            continue


        row = {
            "index_key":
                index_key,

            "regime":
                regime,

            "n_network_dates":
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


        for col in MARKET_METRICS:

            row[
                f"mean_{col}"
            ] = (
                group[
                    col
                ]
                .mean()
            )


            row[
                f"median_{col}"
            ] = (
                group[
                    col
                ]
                .median()
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
# 13. Regime Transition Summary
#
# R1->R2
# R2->R3
# R3->R4
# R4->R5
#
# 每个Index分别计算。
# ============================================================

transition_rows = []


for index_key in analysis_indices:

    temp_index = (
        window_metrics[
            window_metrics[
                "index_key"
            ]
            ==
            index_key
        ]
        .copy()
    )


    # --------------------------------------------------------
    # 用该指数全部network dates的SD进行标准化
    #
    # 仅用于描述，
    # 不是统计显著性检验。
    # --------------------------------------------------------

    overall_sd = {
        col:
            temp_index[
                col
            ]
            .std(
                ddof=1
            )
        for col in MARKET_METRICS
    }


    for i in range(
        1,
        len(
            regime_order
        )
    ):

        before_regime = (
            regime_order[
                i - 1
            ]
        )

        after_regime = (
            regime_order[
                i
            ]
        )


        before = (
            temp_index[
                temp_index[
                    "regime"
                ]
                ==
                before_regime
            ]
        )


        after = (
            temp_index[
                temp_index[
                    "regime"
                ]
                ==
                after_regime
            ]
        )


        if (
            before.empty
            or
            after.empty
        ):

            continue


        row = {
            "index_key":
                index_key,

            "transition":
                f"{before_regime}->{after_regime}",

            "before_regime":
                before_regime,

            "after_regime":
                after_regime,

            "before_n":
                len(
                    before
                ),

            "after_n":
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


        for col in MARKET_METRICS:

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


            sd_value = (
                overall_sd[
                    col
                ]
            )


            row[
                f"before_{col}"
            ] = (
                before_mean
            )


            row[
                f"after_{col}"
            ] = (
                after_mean
            )


            row[
                f"delta_{col}"
            ] = (
                delta
            )


            row[
                f"standardized_delta_{col}"
            ] = (
                delta
                /
                sd_value
                if (
                    pd.notna(
                        sd_value
                    )
                    and
                    sd_value
                    >
                    0
                )
                else np.nan
            )


        transition_rows.append(
            row
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
# 14. 自动识别R3、R4
# ============================================================

def locate_regime(
    candidates
):

    for candidate in candidates:

        if candidate in regime_order:

            return candidate

    return None


r3 = locate_regime(
    [
        "R3",
        "3",
        "Regime3",
        "Regime 3"
    ]
)


r4 = locate_regime(
    [
        "R4",
        "4",
        "Regime4",
        "Regime 4"
    ]
)


# ============================================================
# 15. R3 -> R4 专项分析
# ============================================================

if (
    r3 is not None
    and
    r4 is not None
):

    target_transition = (
        f"{r3}->{r4}"
    )


    r3_r4 = (
        transition_summary[
            transition_summary[
                "transition"
            ]
            ==
            target_transition
        ]
        .copy()
    )


    # --------------------------------------------------------
    # 方向指标
    #
    # 注意：
    # 只是descriptive signals，
    # 不属于显著性检验。
    # --------------------------------------------------------

    r3_r4[
        "V1_return_decreases"
    ] = (
        r3_r4[
            "delta_cumulative_return_20"
        ]
        <
        0
    )


    r3_r4[
        "V2_volatility_increases"
    ] = (
        r3_r4[
            "delta_annualized_volatility_20"
        ]
        >
        0
    )


    r3_r4[
        "V3_drawdown_worsens"
    ] = (
        r3_r4[
            "delta_max_drawdown_20"
        ]
        <
        0
    )


    r3_r4[
        "V4_negative_day_share_increases"
    ] = (
        r3_r4[
            "delta_negative_day_share_20"
        ]
        >
        0
    )


    r3_r4[
        "V5_trading_activity_increases"
    ] = (
        r3_r4[
            "delta_volume_activity_20_60"
        ]
        >
        0
    )


    signal_cols = [
        "V1_return_decreases",
        "V2_volatility_increases",
        "V3_drawdown_worsens",
        "V4_negative_day_share_increases",
        "V5_trading_activity_increases"
    ]


    r3_r4[
        "n_directional_signals"
    ] = (
        r3_r4[
            signal_cols
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
        "WARNING: 无法自动识别R3 / R4。"
    )


# ============================================================
# 16. 屏幕输出
# ============================================================

print(
    "\n=============================================="
)

print(
    "Stage 11 - Stage 2"
)

print(
    "R1-R5 External Market State Comparison"
)

print(
    "=============================================="
)


display_cols = [
    "index_key",
    "regime",
    "n_network_dates",

    "mean_cumulative_return_20",

    "mean_annualized_volatility_20",

    "mean_max_drawdown_20",

    "mean_negative_day_share_20",

    "mean_volume_activity_20_60"
]


print(
    "\n--- Regime Summary ---"
)


print(
    regime_summary[
        display_cols
    ]
    .to_string(
        index=False
    )
)


print(
    "\n--- Transition Summary ---"
)


transition_display_cols = [
    "index_key",
    "transition",

    "delta_cumulative_return_20",

    "delta_annualized_volatility_20",

    "delta_max_drawdown_20",

    "delta_negative_day_share_20",

    "delta_volume_activity_20_60"
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
    r3 is not None
    and
    r4 is not None
):

    print(
        "\n--- R3 -> R4 External Validation ---"
    )

    print(
        r3_r4.to_string(
            index=False
        )
    )


# ============================================================
# 17. 图1：CSI300 / CSI500 20日收益
# ============================================================

plot_df = (
    window_metrics[
        window_metrics[
            "index_key"
        ]
        .isin(
            MAIN_INDICES
        )
    ]
    .copy()
)


fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


for index_key in MAIN_INDICES:

    temp = (
        plot_df[
            plot_df[
                "index_key"
            ]
            ==
            index_key
        ]
    )


    ax.plot(
        temp[
            "network_date"
        ],
        temp[
            "cumulative_return_20"
        ],
        marker="o",
        label=index_key
    )


ax.axhline(
    y=0,
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
        regime_df[
            regime_df[
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
    "20-day Cumulative Return"
)

ax.set_title(
    "External Market Return Across Network Regimes"
)

ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "external_market_return_by_regime.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 18. 图2：20日年化波动率
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


for index_key in MAIN_INDICES:

    temp = (
        plot_df[
            plot_df[
                "index_key"
            ]
            ==
            index_key
        ]
    )


    ax.plot(
        temp[
            "network_date"
        ],
        temp[
            "annualized_volatility_20"
        ],
        marker="o",
        label=index_key
    )


for i in range(
    1,
    len(
        regime_order
    )
):

    boundary = (
        regime_df[
            regime_df[
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
    "20-day Annualized Volatility"
)

ax.set_title(
    "External Market Volatility Across Network Regimes"
)

ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "external_market_volatility_by_regime.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 19. 图3：成交活跃度
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


for index_key in MAIN_INDICES:

    temp = (
        plot_df[
            plot_df[
                "index_key"
            ]
            ==
            index_key
        ]
    )


    ax.plot(
        temp[
            "network_date"
        ],
        temp[
            "volume_activity_20_60"
        ],
        marker="o",
        label=index_key
    )


ax.axhline(
    y=0,
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
        regime_df[
            regime_df[
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
    "20d / 60d Volume Activity - 1"
)

ax.set_title(
    "Trading Activity Across Network Regimes"
)

ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "external_market_activity_by_regime.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. 图4：Regime平均收益
# ============================================================

main_summary = (
    regime_summary[
        regime_summary[
            "index_key"
        ]
        .isin(
            MAIN_INDICES
        )
    ]
    .copy()
)


pivot_return = (
    main_summary
    .pivot(
        index="regime",
        columns="index_key",
        values="mean_cumulative_return_20"
    )
    .reindex(
        regime_order
    )
)


fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


pivot_return.plot(
    kind="bar",
    ax=ax
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Network Regime"
)

ax.set_ylabel(
    "Mean 20-day Cumulative Return"
)

ax.set_title(
    "External Market Return by Network Regime"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "external_market_regime_return_bar.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. 完成
# ============================================================

print(
    "\n=============================================="
)

print(
    "Stage 2完成"
)

print(
    "=============================================="
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
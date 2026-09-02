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


RETURNS_FILE = (
    PROCESSED_DIR
    / "stock_returns.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)

REGIME_FILE = (
    PROCESSED_DIR
    / "network_regime_assignment.csv"
)


# ============================================================
# 2. 输出
# ============================================================

WINDOW_METRICS_FILE = (
    PROCESSED_DIR
    / "market_state_window_metrics.csv"
)

REGIME_SUMMARY_FILE = (
    PROCESSED_DIR
    / "market_state_regime_summary.csv"
)

TRANSITION_SUMMARY_FILE = (
    PROCESSED_DIR
    / "market_state_transition_summary.csv"
)

R3_R4_FILE = (
    PROCESSED_DIR
    / "market_state_R3_R4.csv"
)


# ============================================================
# 3. 参数
# ============================================================

# 与Rolling Network的STEP=20保持一致，
# 用最近20个交易日刻画network date附近的市场状态。
MARKET_LOOKBACK = 20

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# 4. 股票代码标准化
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


# ============================================================
# 5. 读取stock_info
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


code_col = next(
    (
        c
        for c in code_candidates
        if c in stock_info.columns
    ),
    None
)


if code_col is None:

    raise ValueError(
        "stock_info.csv中无法识别股票代码列。"
    )


stock_info[
    "normalized_code"
] = (
    stock_info[
        code_col
    ]
    .apply(
        normalize_code
    )
)


stock_codes = (
    stock_info[
        "normalized_code"
    ]
    .drop_duplicates()
    .tolist()
)


# ============================================================
# 6. 读取收益率
# ============================================================

returns_raw = pd.read_csv(
    RETURNS_FILE
)


date_candidates = [
    "日期",
    "date",
    "trade_date",
    "datetime",
    "Date"
]


date_col = next(
    (
        c
        for c in date_candidates
        if c in returns_raw.columns
    ),
    None
)


if date_col is None:

    raise ValueError(
        "stock_returns.csv中无法找到日期列。"
    )


returns_raw[
    date_col
] = pd.to_datetime(
    returns_raw[
        date_col
    ]
)


returns_raw = (
    returns_raw
    .sort_values(
        date_col
    )
    .set_index(
        date_col
    )
)


# ============================================================
# 7. 标准化收益率列名
# ============================================================

rename_dict = {}


for col in returns_raw.columns:

    normalized = normalize_code(
        col
    )

    if normalized in stock_codes:

        rename_dict[
            col
        ] = normalized


returns_df = (
    returns_raw
    .rename(
        columns=rename_dict
    )
)


available_codes = [
    code
    for code in stock_codes
    if code in returns_df.columns
]


if len(
    available_codes
) < 2:

    raise ValueError(
        "stock_returns与stock_info无法正常匹配。"
    )


returns_df = (
    returns_df[
        available_codes
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .dropna()
)


N_STOCKS = len(
    available_codes
)


print(
    "股票数量：",
    N_STOCKS
)

print(
    "收益率观测数：",
    len(
        returns_df
    )
)


# ============================================================
# 8. 构造每日Equal-weighted Market Return
#
# 当前数据为log return：
#
# r_EW,t = mean_i(r_it)
# ============================================================

daily_market = pd.DataFrame(
    index=returns_df.index
)


daily_market[
    "ew_log_return"
] = (
    returns_df
    .mean(
        axis=1
    )
)


daily_market[
    "ew_abs_return"
] = (
    daily_market[
        "ew_log_return"
    ]
    .abs()
)


# ============================================================
# 9. Daily Market Breadth
#
# 每天上涨股票比例
# ============================================================

daily_market[
    "positive_stock_share"
] = (
    (
        returns_df
        >
        0
    )
    .mean(
        axis=1
    )
)


# ============================================================
# 10. Daily Cross-sectional Dispersion
# ============================================================

daily_market[
    "cross_sectional_dispersion"
] = (
    returns_df
    .std(
        axis=1,
        ddof=1
    )
)


# ============================================================
# 11. 读取Regime Assignment
# ============================================================

regime_raw = pd.read_csv(
    REGIME_FILE
)


regime_date_candidates = [
    "network_date",
    "date",
    "window_end"
]


regime_label_candidates = [
    "regime",
    "regime_id",
    "Regime"
]


regime_date_col = next(
    (
        c
        for c in regime_date_candidates
        if c in regime_raw.columns
    ),
    None
)


regime_col = next(
    (
        c
        for c in regime_label_candidates
        if c in regime_raw.columns
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
        "无法识别日期或Regime列。"
    )


regime_df = (
    regime_raw[
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
)


print(
    "Network dates：",
    len(
        regime_df
    )
)


# ============================================================
# 12. Maximum Drawdown函数
#
# 输入为log return序列。
# ============================================================

def calculate_max_drawdown(
    log_returns: pd.Series
) -> float:

    log_wealth = (
        log_returns
        .cumsum()
    )

    wealth = np.exp(
        log_wealth
    )


    running_max = (
        wealth
        .cummax()
    )


    drawdown = (
        wealth
        /
        running_max
        -
        1
    )


    return float(
        drawdown.min()
    )


# ============================================================
# 13. 对每一个Network Date计算最近20日市场状态
# ============================================================

market_state_rows = []


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


    # --------------------------------------------------------
    # 找到 <= network_date 的最后一个交易日
    # --------------------------------------------------------

    eligible_dates = (
        daily_market.index[
            daily_market.index
            <=
            network_date
        ]
    )


    if len(
        eligible_dates
    ) == 0:

        continue


    end_date = (
        eligible_dates[
            -1
        ]
    )


    end_position = (
        daily_market.index
        .get_loc(
            end_date
        )
    )


    start_position = (
        end_position
        -
        MARKET_LOOKBACK
        +
        1
    )


    if start_position < 0:

        continue


    market_window = (
        daily_market
        .iloc[
            start_position:
            end_position + 1
        ]
        .copy()
    )


    returns_window = (
        returns_df
        .iloc[
            start_position:
            end_position + 1
        ]
        .copy()
    )


    ew_returns = (
        market_window[
            "ew_log_return"
        ]
    )


    # --------------------------------------------------------
    # 累计收益
    # log return -> simple cumulative return
    # --------------------------------------------------------

    cumulative_return = (
        np.exp(
            ew_returns.sum()
        )
        -
        1
    )


    # --------------------------------------------------------
    # 年化波动率
    # --------------------------------------------------------

    annualized_volatility = (
        ew_returns.std(
            ddof=1
        )
        *
        np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )


    # --------------------------------------------------------
    # 平均绝对收益
    # --------------------------------------------------------

    mean_abs_return = (
        ew_returns
        .abs()
        .mean()
    )


    # --------------------------------------------------------
    # 负收益日比例
    # --------------------------------------------------------

    negative_day_share = (
        (
            ew_returns
            <
            0
        )
        .mean()
    )


    # --------------------------------------------------------
    # Market Breadth
    # --------------------------------------------------------

    mean_positive_stock_share = (
        market_window[
            "positive_stock_share"
        ]
        .mean()
    )


    # --------------------------------------------------------
    # 横截面分散度
    # --------------------------------------------------------

    mean_dispersion = (
        market_window[
            "cross_sectional_dispersion"
        ]
        .mean()
    )


    # --------------------------------------------------------
    # Maximum Drawdown
    # --------------------------------------------------------

    max_drawdown = (
        calculate_max_drawdown(
            ew_returns
        )
    )


    # --------------------------------------------------------
    # 个股平均20日累计收益
    # 以及股票之间累计收益分散程度
    # --------------------------------------------------------

    stock_cumulative_returns = (
        np.exp(
            returns_window
            .sum(
                axis=0
            )
        )
        -
        1
    )


    mean_stock_cumulative_return = (
        stock_cumulative_returns
        .mean()
    )


    dispersion_stock_cumulative_return = (
        stock_cumulative_returns
        .std(
            ddof=1
        )
    )


    market_state_rows.append(
        {
            "network_date":
                network_date,

            "regime":
                regime,

            "market_window_start":
                market_window.index[
                    0
                ],

            "market_window_end":
                market_window.index[
                    -1
                ],

            "market_lookback":
                MARKET_LOOKBACK,

            "ew_cumulative_return_20":
                cumulative_return,

            "ew_annualized_volatility_20":
                annualized_volatility,

            "ew_mean_abs_return_20":
                mean_abs_return,

            "negative_day_share_20":
                negative_day_share,

            "mean_positive_stock_share_20":
                mean_positive_stock_share,

            "cross_sectional_dispersion_20":
                mean_dispersion,

            "max_drawdown_20":
                max_drawdown,

            "mean_stock_cumulative_return_20":
                mean_stock_cumulative_return,

            "dispersion_stock_cumulative_return_20":
                dispersion_stock_cumulative_return
        }
    )


market_state_df = pd.DataFrame(
    market_state_rows
)


# ============================================================
# 14. 给主要指标添加Z-score
#
# 仅用于描述性跨指标比较。
# ============================================================

market_metrics = [
    "ew_cumulative_return_20",
    "ew_annualized_volatility_20",
    "ew_mean_abs_return_20",
    "negative_day_share_20",
    "mean_positive_stock_share_20",
    "cross_sectional_dispersion_20",
    "max_drawdown_20",
    "dispersion_stock_cumulative_return_20"
]


for col in market_metrics:

    mean_value = (
        market_state_df[
            col
        ]
        .mean()
    )


    sd_value = (
        market_state_df[
            col
        ]
        .std(
            ddof=1
        )
    )


    market_state_df[
        f"z_{col}"
    ] = (
        (
            market_state_df[
                col
            ]
            -
            mean_value
        )
        /
        sd_value
        if sd_value > 0
        else np.nan
    )


market_state_df.to_csv(
    WINDOW_METRICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Regime顺序
# ============================================================

regime_order = (
    market_state_df[
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
# 16. Regime-level Summary
# ============================================================

summary_rows = []


for regime in regime_order:

    group = (
        market_state_df[
            market_state_df[
                "regime"
            ]
            ==
            regime
        ]
    )


    row = {
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


    for col in market_metrics:

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
# 17. Adjacent Regime Transition Summary
# ============================================================

overall_sd = {
    col:
        market_state_df[
            col
        ]
        .std(
            ddof=1
        )
    for col in market_metrics
}


transition_rows = []


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
        market_state_df[
            market_state_df[
                "regime"
            ]
            ==
            before_regime
        ]
    )


    after = (
        market_state_df[
            market_state_df[
                "regime"
            ]
            ==
            after_regime
        ]
    )


    row = {
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


    for col in market_metrics:

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
                sd_value > 0
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
# 18. 自动寻找R3 / R4
# ============================================================

def locate_regime(
    names
):

    for name in names:

        if name in regime_order:
            return name

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
# 19. R3 -> R4专项分析
# ============================================================

if (
    r3 is not None
    and
    r4 is not None
):

    transition_name = (
        f"{r3}->{r4}"
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


    if len(
        r3_r4
    ) == 1:

        # ----------------------------------------------------
        # 以下全部只是方向性描述指标，
        # 不是统计显著性检验。
        # ----------------------------------------------------

        r3_r4[
            "V1_volatility_increases"
        ] = (
            r3_r4[
                "delta_ew_annualized_volatility_20"
            ]
            >
            0
        )


        r3_r4[
            "V2_cumulative_return_decreases"
        ] = (
            r3_r4[
                "delta_ew_cumulative_return_20"
            ]
            <
            0
        )


        # MDD越负表示回撤更严重
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
            "V4_dispersion_increases"
        ] = (
            r3_r4[
                "delta_cross_sectional_dispersion_20"
            ]
            >
            0
        )


        r3_r4[
            "V5_negative_day_share_increases"
        ] = (
            r3_r4[
                "delta_negative_day_share_20"
            ]
            >
            0
        )


        r3_r4[
            "V6_market_breadth_decreases"
        ] = (
            r3_r4[
                "delta_mean_positive_stock_share_20"
            ]
            <
            0
        )


        validation_cols = [
            "V1_volatility_increases",
            "V2_cumulative_return_decreases",
            "V3_drawdown_worsens",
            "V4_dispersion_increases",
            "V5_negative_day_share_increases",
            "V6_market_breadth_decreases"
        ]


        r3_r4[
            "n_stress_direction_signals"
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
        "警告：未自动识别R3和R4。"
    )


# ============================================================
# 20. 屏幕输出
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 6: Network Regime and Market State"
)

print(
    "============================================"
)


display_cols = [
    "regime",
    "n_network_dates",

    "mean_ew_cumulative_return_20",

    "mean_ew_annualized_volatility_20",

    "mean_max_drawdown_20",

    "mean_negative_day_share_20",

    "mean_mean_positive_stock_share_20",

    "mean_cross_sectional_dispersion_20"
]


print(
    "\n--- Regime Market-state Summary ---"
)


print(
    regime_summary[
        display_cols
    ]
    .to_string(
        index=False
    )
)


transition_display_cols = [
    "transition",

    "delta_ew_cumulative_return_20",

    "delta_ew_annualized_volatility_20",

    "delta_max_drawdown_20",

    "delta_negative_day_share_20",

    "delta_mean_positive_stock_share_20",

    "delta_cross_sectional_dispersion_20"
]


print(
    "\n--- Regime Transition Market-state Changes ---"
)


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
        "\n--- R3 -> R4 ---"
    )

    print(
        r3_r4.to_string(
            index=False
        )
    )


# ============================================================
# 21. 图1：20日累计收益
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    market_state_df[
        "network_date"
    ],
    market_state_df[
        "ew_cumulative_return_20"
    ],
    marker="o"
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
        market_state_df[
            market_state_df[
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
    "20-day Equal-weighted Cumulative Return"
)

ax.set_title(
    "Market Return Across Network Regimes"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_market_return.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 图2：20日年化波动率
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    market_state_df[
        "network_date"
    ],
    market_state_df[
        "ew_annualized_volatility_20"
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
        market_state_df[
            market_state_df[
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
    "Market Volatility Across Network Regimes"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_market_volatility.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 图3：Cross-sectional Dispersion
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    market_state_df[
        "network_date"
    ],
    market_state_df[
        "cross_sectional_dispersion_20"
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
        market_state_df[
            market_state_df[
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
    "Mean Cross-sectional Return Dispersion"
)

ax.set_title(
    "Cross-sectional Dispersion Across Network Regimes"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_market_dispersion.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. 图4：Regime平均波动率
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
        "mean_ew_annualized_volatility_20"
    ]
)


ax.set_xlabel(
    "Regime"
)

ax.set_ylabel(
    "Mean 20-day Annualized Volatility"
)

ax.set_title(
    "Market Volatility by Network Regime"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_average_market_volatility.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. 完成
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 6完成"
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
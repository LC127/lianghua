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

NETWORK_METRICS_FILE = (
    PROCESSED_DIR
    / "regime_cross_validation_window_metrics.csv"
)


# ============================================================
# 3. 输出文件
# ============================================================

EVENT_DAILY_FILE = (
    PROCESSED_DIR
    / "R3_R4_event_window_daily.csv"
)

SEGMENT_SUMMARY_FILE = (
    PROCESSED_DIR
    / "R3_R4_event_window_segment_summary.csv"
)

ALIGNMENT_FILE = (
    PROCESSED_DIR
    / "R3_R4_network_market_alignment.csv"
)


# ============================================================
# 4. 参数
# ============================================================

EVENT_WINDOW_BEFORE = 40
EVENT_WINDOW_AFTER = 40

ROLLING_WINDOW = 20
ACTIVITY_LONG_WINDOW = 60

TRADING_DAYS = 252


# ============================================================
# 5. 读取Regime Assignment
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
        c
        for c in date_candidates
        if c in regime_raw.columns
    ),
    None
)


regime_col = next(
    (
        c
        for c in regime_candidates
        if c in regime_raw.columns
    ),
    None
)


if (
    date_col is None
    or
    regime_col is None
):

    raise ValueError(
        "无法识别Network Date或Regime字段。"
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
    .sort_values(
        "network_date"
    )
    .drop_duplicates(
        subset="network_date"
    )
)


# ============================================================
# 6. 自动识别R3、R4
# ============================================================

def locate_regime(
    candidates
):

    values = set(
        regime_df[
            "regime"
        ]
    )

    for candidate in candidates:

        if candidate in values:
            return candidate

    return None


r3_name = locate_regime(
    [
        "R3",
        "3",
        "Regime3",
        "Regime 3"
    ]
)


r4_name = locate_regime(
    [
        "R4",
        "4",
        "Regime4",
        "Regime 4"
    ]
)


if (
    r3_name is None
    or
    r4_name is None
):

    raise ValueError(
        "无法自动识别R3 / R4。"
    )


last_r3_date = (
    regime_df.loc[
        regime_df[
            "regime"
        ]
        ==
        r3_name,
        "network_date"
    ]
    .max()
)


first_r4_date = (
    regime_df.loc[
        regime_df[
            "regime"
        ]
        ==
        r4_name,
        "network_date"
    ]
    .min()
)


print(
    "Last R3 Network Date:",
    last_r3_date
)


print(
    "First R4 Network Date:",
    first_r4_date
)


# ============================================================
# 7. 读取外部指数
# ============================================================

index_df = pd.read_csv(
    EXTERNAL_INDEX_FILE,
    dtype={
        "index_key": str,
        "index_code": str
    }
)


index_df[
    "date"
] = pd.to_datetime(
    index_df[
        "date"
    ]
)


for col in [
    "simple_return",
    "close",
    "volume"
]:

    if col in index_df.columns:

        index_df[col] = pd.to_numeric(
            index_df[col],
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
        ]
    )
)


# ============================================================
# 8. 只用CSI300和CSI500作为主要比较
# ============================================================

MAIN_INDICES = [
    "CSI300",
    "CSI500"
]


available = set(
    index_df[
        "index_key"
    ]
)


for key in MAIN_INDICES:

    if key not in available:

        raise ValueError(
            f"缺少指数：{key}"
        )


# ============================================================
# 9. 建立共同交易日Daily Panel
# ============================================================

daily_parts = []


for key in MAIN_INDICES:

    temp = (
        index_df[
            index_df[
                "index_key"
            ]
            ==
            key
        ][
            [
                "date",
                "close",
                "simple_return",
                "volume"
            ]
        ]
        .copy()
    )


    temp = temp.rename(
        columns={
            "close":
                f"{key}_close",

            "simple_return":
                f"{key}_return",

            "volume":
                f"{key}_volume"
        }
    )


    daily_parts.append(
        temp.set_index(
            "date"
        )
    )


daily = pd.concat(
    daily_parts,
    axis=1,
    join="inner"
)


daily = (
    daily
    .sort_index()
)


# ============================================================
# 10. 计算每个指数的Rolling 20-day市场指标
# ============================================================

for key in MAIN_INDICES:

    ret_col = (
        f"{key}_return"
    )


    volume_col = (
        f"{key}_volume"
    )


    # --------------------------------------------------------
    # 20-day cumulative return
    # --------------------------------------------------------

    daily[
        f"{key}_return_20"
    ] = (
        (
            1
            +
            daily[
                ret_col
            ]
        )
        .rolling(
            ROLLING_WINDOW
        )
        .apply(
            np.prod,
            raw=True
        )
        -
        1
    )


    # --------------------------------------------------------
    # 20-day annualized volatility
    # --------------------------------------------------------

    daily[
        f"{key}_volatility_20"
    ] = (
        daily[
            ret_col
        ]
        .rolling(
            ROLLING_WINDOW
        )
        .std(
            ddof=1
        )
        *
        np.sqrt(
            TRADING_DAYS
        )
    )


    # --------------------------------------------------------
    # 20-day negative-day share
    # --------------------------------------------------------

    daily[
        f"{key}_negative_share_20"
    ] = (
        (
            daily[
                ret_col
            ]
            <
            0
        )
        .astype(float)
        .rolling(
            ROLLING_WINDOW
        )
        .mean()
    )


    # --------------------------------------------------------
    # 20/60 Volume Activity
    # --------------------------------------------------------

    volume_mean_20 = (
        daily[
            volume_col
        ]
        .rolling(
            20
        )
        .mean()
    )


    volume_mean_60 = (
        daily[
            volume_col
        ]
        .rolling(
            ACTIVITY_LONG_WINDOW
        )
        .mean()
    )


    daily[
        f"{key}_volume_activity_20_60"
    ] = (
        volume_mean_20
        /
        volume_mean_60
        -
        1
    )


# ============================================================
# 11. 构造相对市场指标
# ============================================================

# ------------------------------------------------------------
# CSI500 - CSI300 trailing 20-day return spread
# ------------------------------------------------------------

daily[
    "relative_return_20"
] = (
    daily[
        "CSI500_return_20"
    ]
    -
    daily[
        "CSI300_return_20"
    ]
)


# ------------------------------------------------------------
# Volatility Spread
# ------------------------------------------------------------

daily[
    "relative_volatility_20"
] = (
    daily[
        "CSI500_volatility_20"
    ]
    -
    daily[
        "CSI300_volatility_20"
    ]
)


# ------------------------------------------------------------
# Volume Activity Spread
# ------------------------------------------------------------

daily[
    "relative_volume_activity"
] = (
    daily[
        "CSI500_volume_activity_20_60"
    ]
    -
    daily[
        "CSI300_volume_activity_20_60"
    ]
)


# ============================================================
# 12. 确定First R4对应共同交易日
# ============================================================

eligible_dates = (
    daily.index[
        daily.index
        <=
        first_r4_date
    ]
)


if len(
    eligible_dates
) == 0:

    raise ValueError(
        "First R4 Date之前没有External Market Data。"
    )


anchor_date = (
    eligible_dates[
        -1
    ]
)


anchor_pos = (
    daily.index
    .get_loc(
        anchor_date
    )
)


start_pos = max(
    0,
    anchor_pos
    -
    EVENT_WINDOW_BEFORE
)


end_pos = min(
    len(
        daily
    )
    -
    1,
    anchor_pos
    +
    EVENT_WINDOW_AFTER
)


event_daily = (
    daily
    .iloc[
        start_pos:
        end_pos + 1
    ]
    .copy()
)


# ============================================================
# 13. Event Time
#
# First R4 = 0
# ============================================================

event_daily[
    "event_time"
] = np.arange(
    start_pos
    -
    anchor_pos,
    end_pos
    -
    anchor_pos
    +
    1
)


# ============================================================
# 14. Relative cumulative wealth
#
# W_rel =
# prod[(1+r500)/(1+r300)]
#
# 从Event Window第一天归一化。
# ============================================================

relative_gross_return = (
    (
        1
        +
        event_daily[
            "CSI500_return"
        ]
    )
    /
    (
        1
        +
        event_daily[
            "CSI300_return"
        ]
    )
)


relative_wealth = (
    relative_gross_return
    .cumprod()
)


event_daily[
    "CSI500_vs_CSI300_relative_cumulative_return"
] = (
    relative_wealth
    /
    relative_wealth.iloc[
        0
    ]
    -
    1
)


# ============================================================
# 15. 指数各自从Event Window起点的累计收益
# ============================================================

for key in MAIN_INDICES:

    gross = (
        1
        +
        event_daily[
            f"{key}_return"
        ]
    )


    wealth = (
        gross
        .cumprod()
    )


    event_daily[
        f"{key}_event_cumulative_return"
    ] = (
        wealth
        /
        wealth.iloc[
            0
        ]
        -
        1
    )


# ============================================================
# 16. 标记R3最后日期、R4首日
# ============================================================

event_daily[
    "is_first_R4_anchor"
] = (
    event_daily.index
    ==
    anchor_date
)


# 找<=last_r3_date的最后交易日
r3_eligible = (
    event_daily.index[
        event_daily.index
        <=
        last_r3_date
    ]
)


if len(
    r3_eligible
) > 0:

    last_r3_market_date = (
        r3_eligible[
            -1
        ]
    )

else:

    last_r3_market_date = pd.NaT


event_daily[
    "is_last_R3_market_date"
] = (
    event_daily.index
    ==
    last_r3_market_date
)


# ============================================================
# 17. 定义三个局部阶段
#
# PRE:
# anchor前20交易日
#
# TRANSITION:
# last R3 network date -> first R4 network date
#
# POST:
# first R4后的20交易日
#
# 仅描述，不是因果事件研究。
# ============================================================

event_daily[
    "segment"
] = "Other"


event_daily.loc[
    (
        event_daily.index
        <=
        last_r3_market_date
    )
    &
    (
        event_daily.index
        >
        event_daily.index[
            max(
                0,
                event_daily.index.get_loc(
                    last_r3_market_date
                )
                -
                20
            )
        ]
    ),
    "segment"
] = "Pre-R3"


event_daily.loc[
    (
        event_daily.index
        >
        last_r3_market_date
    )
    &
    (
        event_daily.index
        <=
        anchor_date
    ),
    "segment"
] = "R3-R4 Transition"


post_dates = (
    event_daily.index[
        event_daily.index
        >
        anchor_date
    ][
        :20
    ]
)


event_daily.loc[
    post_dates,
    "segment"
] = "Post-R4"


# ============================================================
# 18. 保存Daily Event Window
# ============================================================

event_daily_out = (
    event_daily
    .reset_index()
    .rename(
        columns={
            "index":
                "date"
        }
    )
)


event_daily_out.to_csv(
    EVENT_DAILY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. Segment Summary
# ============================================================

SEGMENT_METRICS = [
    "CSI300_return_20",
    "CSI500_return_20",

    "relative_return_20",

    "CSI300_volatility_20",
    "CSI500_volatility_20",

    "relative_volatility_20",

    "CSI300_volume_activity_20_60",
    "CSI500_volume_activity_20_60",

    "relative_volume_activity",

    "CSI500_vs_CSI300_relative_cumulative_return"
]


segment_rows = []


segment_order = [
    "Pre-R3",
    "R3-R4 Transition",
    "Post-R4"
]


for segment in segment_order:

    group = (
        event_daily[
            event_daily[
                "segment"
            ]
            ==
            segment
        ]
    )


    if group.empty:

        continue


    row = {
        "segment":
            segment,

        "n_trading_days":
            len(
                group
            ),

        "start_date":
            group.index.min(),

        "end_date":
            group.index.max()
    }


    for col in SEGMENT_METRICS:

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
            f"start_{col}"
        ] = (
            group[
                col
            ]
            .iloc[
                0
            ]
        )


        row[
            f"end_{col}"
        ] = (
            group[
                col
            ]
            .iloc[
                -1
            ]
        )


    segment_rows.append(
        row
    )


segment_summary = pd.DataFrame(
    segment_rows
)


segment_summary.to_csv(
    SEGMENT_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 读取Network Metrics
# ============================================================

network_metrics = pd.read_csv(
    NETWORK_METRICS_FILE
)


network_metrics[
    "network_date"
] = pd.to_datetime(
    network_metrics[
        "network_date"
    ]
)


required_network_cols = [
    "network_date",
    "regime",
    "raw_same_cross_strength_ratio",
    "matched_same_ratio",
    "matched_edge_count"
]


missing_network = [
    col
    for col in required_network_cols
    if col not in network_metrics.columns
]


if missing_network:

    raise ValueError(
        "Network Metrics缺少字段："
        f"{missing_network}"
    )


# ============================================================
# 21. 只保留Event Window附近的Network Dates
# ============================================================

network_local = (
    network_metrics[
        (
            network_metrics[
                "network_date"
            ]
            >=
            event_daily.index.min()
        )
        &
        (
            network_metrics[
                "network_date"
            ]
            <=
            event_daily.index.max()
        )
    ]
    .copy()
    .sort_values(
        "network_date"
    )
)


# ============================================================
# 22. 将External Daily指标匹配到Network Date
#
# 严格使用 <= network_date 的最后可用交易日
# ============================================================

external_for_merge = (
    event_daily
    .rename_axis("market_date")
    .reset_index()
    .sort_values("market_date")
)


network_local = (
    network_local
    .sort_values(
        "network_date"
    )
)


alignment = pd.merge_asof(
    network_local,
    external_for_merge,
    left_on="network_date",
    right_on="market_date",
    direction="backward"
)


# ============================================================
# 23. 网络指标的相邻变化
# ============================================================

alignment[
    "delta_edge_count"
] = (
    alignment[
        "matched_edge_count"
    ]
    .diff()
)


alignment[
    "delta_same_ratio"
] = (
    alignment[
        "matched_same_ratio"
    ]
    .diff()
)


alignment[
    "delta_same_cross_strength_ratio"
] = (
    alignment[
        "raw_same_cross_strength_ratio"
    ]
    .diff()
)


alignment[
    "delta_relative_return_20"
] = (
    alignment[
        "relative_return_20"
    ]
    .diff()
)


alignment.to_csv(
    ALIGNMENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 24. 屏幕输出
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 11 - Stage 3"
)

print(
    "R3 -> R4 Event-window Analysis"
)

print(
    "============================================"
)


print(
    "\nR3 last network date:",
    last_r3_date
)


print(
    "R4 first network date:",
    first_r4_date
)


print(
    "Anchor market date:",
    anchor_date
)


print(
    "\n--- Segment Summary ---"
)


display_cols = [
    "segment",
    "n_trading_days",

    "mean_CSI300_return_20",
    "mean_CSI500_return_20",

    "mean_relative_return_20",

    "mean_CSI300_volatility_20",
    "mean_CSI500_volatility_20",

    "mean_relative_volatility_20",

    "mean_CSI300_volume_activity_20_60",
    "mean_CSI500_volume_activity_20_60"
]


print(
    segment_summary[
        display_cols
    ]
    .to_string(
        index=False
    )
)


print(
    "\n--- Network / Market Alignment ---"
)


alignment_cols = [
    "network_date",
    "regime",

    "matched_edge_count",
    "matched_same_ratio",
    "raw_same_cross_strength_ratio",

    "CSI300_return_20",
    "CSI500_return_20",

    "relative_return_20",

    "CSI300_volatility_20",
    "CSI500_volatility_20"
]


print(
    alignment[
        alignment_cols
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 25. Figure 1：
# CSI300 / CSI500 Event-window cumulative returns
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    event_daily[
        "event_time"
    ],
    event_daily[
        "CSI300_event_cumulative_return"
    ],
    label="CSI300"
)


ax.plot(
    event_daily[
        "event_time"
    ],
    event_daily[
        "CSI500_event_cumulative_return"
    ],
    label="CSI500"
)


ax.axvline(
    x=0,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Trading Days Relative to First R4 Network Date"
)


ax.set_ylabel(
    "Cumulative Return"
)


ax.set_title(
    "CSI300 and CSI500 Around the R3-R4 Network Transition"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "R3_R4_index_cumulative_returns.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 26. Figure 2：
# CSI500 relative to CSI300
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    event_daily[
        "event_time"
    ],
    event_daily[
        "CSI500_vs_CSI300_relative_cumulative_return"
    ]
)


ax.axvline(
    x=0,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Trading Days Relative to First R4 Network Date"
)


ax.set_ylabel(
    "CSI500 Relative Cumulative Return vs CSI300"
)


ax.set_title(
    "Relative Performance Around the R3-R4 Network Transition"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "R3_R4_relative_performance.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 27. Figure 3：
# Trailing 20-day return spread
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    event_daily[
        "event_time"
    ],
    event_daily[
        "relative_return_20"
    ]
)


ax.axvline(
    x=0,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Trading Days Relative to First R4 Network Date"
)


ax.set_ylabel(
    "CSI500 - CSI300 20-day Return"
)


ax.set_title(
    "20-day Relative Return Around the R3-R4 Transition"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "R3_R4_relative_20d_return.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 28. Figure 4：
# Relative Volatility
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    event_daily[
        "event_time"
    ],
    event_daily[
        "relative_volatility_20"
    ]
)


ax.axvline(
    x=0,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Trading Days Relative to First R4 Network Date"
)


ax.set_ylabel(
    "CSI500 - CSI300 Annualized Volatility"
)


ax.set_title(
    "Relative Volatility Around the R3-R4 Transition"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "R3_R4_relative_volatility.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 29. Figure 5：
# Network Same/Cross Strength Ratio
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        6
    )
)


ax.plot(
    alignment[
        "network_date"
    ],
    alignment[
        "raw_same_cross_strength_ratio"
    ],
    marker="o"
)


ax.axvline(
    x=first_r4_date,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Network Date"
)


ax.set_ylabel(
    "Same / Cross Mean |Partial Correlation|"
)


ax.set_title(
    "Network Industry Concentration Around R3-R4"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "R3_R4_network_strength_ratio.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 30. 完成
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 3完成"
)

print(
    "============================================"
)


for path in [
    EVENT_DAILY_FILE,
    SEGMENT_SUMMARY_FILE,
    ALIGNMENT_FILE
]:

    print(
        path
    )
from __future__ import annotations

from pathlib import Path

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

EXTERNAL_INDEX_FILE = (
    PROCESSED_DIR
    / "external_index_daily.csv"
)

NETWORK_METRICS_FILE = (
    PROCESSED_DIR
    / "regime_cross_validation_window_metrics.csv"
)


# ============================================================
# 3. 输出
# ============================================================

DATASET_FILE = (
    PROCESSED_DIR
    / "network_future_market_dataset.csv"
)

SPEARMAN_FILE = (
    PROCESSED_DIR
    / "network_future_spearman_summary.csv"
)

QUANTILE_FILE = (
    PROCESSED_DIR
    / "network_feature_quantile_future_summary.csv"
)

REGIME_FUTURE_FILE = (
    PROCESSED_DIR
    / "network_regime_future_outcome_summary.csv"
)


# ============================================================
# 4. 参数
# ============================================================

FUTURE_HORIZON = 20

TRADING_DAYS_PER_YEAR = 252

MAIN_INDICES = [
    "CSI300",
    "CSI500"
]


# ============================================================
# 5. Maximum Drawdown
# ============================================================

def calculate_max_drawdown(
    simple_returns: pd.Series
) -> float:

    if len(simple_returns) == 0:
        return np.nan


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
# 6. 读取外部指数
# ============================================================

index_df = pd.read_csv(
    EXTERNAL_INDEX_FILE,
    dtype={
        "index_key": str,
        "index_code": str
    }
)


required_index_cols = [
    "date",
    "index_key",
    "simple_return"
]


missing = [
    col
    for col in required_index_cols
    if col not in index_df.columns
]


if missing:

    raise ValueError(
        f"external_index_daily.csv缺少字段：{missing}"
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
# 7. 检查主要指数
# ============================================================

available_indices = set(
    index_df[
        "index_key"
    ]
    .dropna()
)


for key in MAIN_INDICES:

    if key not in available_indices:

        raise ValueError(
            f"缺少外部指数：{key}"
        )


# ============================================================
# 8. 读取Network Features
# ============================================================

network_df = pd.read_csv(
    NETWORK_METRICS_FILE
)


required_network_cols = [
    "network_date",
    "regime",

    "matched_edge_count",
    "matched_same_ratio",

    "raw_mean_abs_partial_all",
    "raw_same_cross_strength_ratio"
]


missing_network = [
    col
    for col in required_network_cols
    if col not in network_df.columns
]


if missing_network:

    raise ValueError(
        "Network Metrics缺少字段："
        f"{missing_network}"
    )


network_df[
    "network_date"
] = pd.to_datetime(
    network_df[
        "network_date"
    ]
)


network_df[
    "regime"
] = (
    network_df[
        "regime"
    ]
    .astype(str)
    .str.strip()
)


network_df = (
    network_df
    .sort_values(
        "network_date"
    )
    .drop_duplicates(
        subset="network_date"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 9. Network Feature命名简化
# ============================================================

network_df = (
    network_df
    .rename(
        columns={
            "matched_edge_count":
                "edge_count",

            "matched_same_ratio":
                "same_edge_ratio",

            "raw_mean_abs_partial_all":
                "mean_abs_partial",

            "raw_same_cross_strength_ratio":
                "same_cross_strength_ratio"
        }
    )
)


NETWORK_FEATURES = [
    "edge_count",
    "same_edge_ratio",
    "mean_abs_partial",
    "same_cross_strength_ratio"
]


# ============================================================
# 10. 为每个Network Date计算Future 20-day Outcome
#
# 严格使用：
#
# date > network_date
#
# 的前20个交易日。
#
# Network Date当天绝不进入Future Outcome。
# ============================================================

future_rows = []


for _, network_row in network_df.iterrows():

    network_date = (
        network_row[
            "network_date"
        ]
    )


    row = {
        "network_date":
            network_date,

        "regime":
            network_row[
                "regime"
            ]
    }


    # 当前Network Features
    for feature in NETWORK_FEATURES:

        row[
            feature
        ] = (
            network_row[
                feature
            ]
        )


    valid_all_indices = True


    for index_key in MAIN_INDICES:

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
        )


        # ----------------------------------------------------
        # 严格未来日期：
        # date > network_date
        # ----------------------------------------------------

        future_data = (
            temp[
                temp[
                    "date"
                ]
                >
                network_date
            ]
            .head(
                FUTURE_HORIZON
            )
            .copy()
        )


        # ----------------------------------------------------
        # 如果不足20日，不使用该Network Date
        # ----------------------------------------------------

        if (
            len(
                future_data
            )
            <
            FUTURE_HORIZON
        ):

            valid_all_indices = False

            break


        returns = (
            future_data[
                "simple_return"
            ]
            .dropna()
        )


        if (
            len(
                returns
            )
            !=
            FUTURE_HORIZON
        ):

            valid_all_indices = False

            break


        # ----------------------------------------------------
        # Future Return
        # ----------------------------------------------------

        future_return = (
            (
                1.0
                +
                returns
            )
            .prod()
            -
            1.0
        )


        # ----------------------------------------------------
        # Future Volatility
        # ----------------------------------------------------

        future_volatility = (
            returns
            .std(
                ddof=1
            )
            *
            np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )


        # ----------------------------------------------------
        # Future Mean Absolute Return
        # ----------------------------------------------------

        future_mean_abs_return = (
            returns
            .abs()
            .mean()
        )


        # ----------------------------------------------------
        # Future Negative-day Share
        # ----------------------------------------------------

        future_negative_share = (
            (
                returns
                <
                0
            )
            .mean()
        )


        # ----------------------------------------------------
        # Future Maximum Drawdown
        # ----------------------------------------------------

        future_mdd = (
            calculate_max_drawdown(
                returns
            )
        )


        row[
            f"{index_key}_future_start_date"
        ] = (
            future_data[
                "date"
            ]
            .iloc[
                0
            ]
        )


        row[
            f"{index_key}_future_end_date"
        ] = (
            future_data[
                "date"
            ]
            .iloc[
                -1
            ]
        )


        row[
            f"{index_key}_future_return_20"
        ] = (
            future_return
        )


        row[
            f"{index_key}_future_volatility_20"
        ] = (
            future_volatility
        )


        row[
            f"{index_key}_future_mean_abs_return_20"
        ] = (
            future_mean_abs_return
        )


        row[
            f"{index_key}_future_negative_share_20"
        ] = (
            future_negative_share
        )


        row[
            f"{index_key}_future_max_drawdown_20"
        ] = (
            future_mdd
        )


    if valid_all_indices:

        # ----------------------------------------------------
        # CSI500 - CSI300 Future Relative Return
        # ----------------------------------------------------

        row[
            "future_relative_return_20"
        ] = (
            row[
                "CSI500_future_return_20"
            ]
            -
            row[
                "CSI300_future_return_20"
            ]
        )


        # ----------------------------------------------------
        # CSI500 - CSI300 Future Volatility
        # ----------------------------------------------------

        row[
            "future_relative_volatility_20"
        ] = (
            row[
                "CSI500_future_volatility_20"
            ]
            -
            row[
                "CSI300_future_volatility_20"
            ]
        )


        future_rows.append(
            row
        )


# ============================================================
# 11. 构造完整Dataset
# ============================================================

dataset = pd.DataFrame(
    future_rows
)


dataset = (
    dataset
    .sort_values(
        "network_date"
    )
    .reset_index(
        drop=True
    )
)


dataset.to_csv(
    DATASET_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "可用于Future分析的Network Dates：",
    len(
        dataset
    )
)


# ============================================================
# 12. Future Outcomes
# ============================================================

FUTURE_OUTCOMES = [
    "CSI300_future_return_20",
    "CSI300_future_volatility_20",
    "CSI300_future_max_drawdown_20",

    "CSI500_future_return_20",
    "CSI500_future_volatility_20",
    "CSI500_future_max_drawdown_20",

    "future_relative_return_20",
    "future_relative_volatility_20"
]


# ============================================================
# 13. Spearman Correlation
#
# 这里只报告rho，
# 不使用naive p-value。
# ============================================================

spearman_rows = []


for feature in NETWORK_FEATURES:

    for outcome in FUTURE_OUTCOMES:

        pair = (
            dataset[
                [
                    feature,
                    outcome
                ]
            ]
            .dropna()
        )


        n = len(
            pair
        )


        if n < 5:

            rho = np.nan

        else:

            rho = (
                pair[
                    feature
                ]
                .corr(
                    pair[
                        outcome
                    ],
                    method="spearman"
                )
            )


        spearman_rows.append(
            {
                "network_feature":
                    feature,

                "future_outcome":
                    outcome,

                "n":
                    n,

                "spearman_rho":
                    rho,

                "abs_spearman_rho":
                    (
                        abs(
                            rho
                        )
                        if pd.notna(
                            rho
                        )
                        else np.nan
                    )
            }
        )


spearman_summary = pd.DataFrame(
    spearman_rows
)


spearman_summary = (
    spearman_summary
    .sort_values(
        "abs_spearman_rho",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


spearman_summary.to_csv(
    SPEARMAN_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. Feature Tercile Analysis
#
# Low / Middle / High
#
# 样本只有约31，所以使用3组而不是4/5组。
#
# 不做统计显著性检验。
# ============================================================

quantile_rows = []


for feature in NETWORK_FEATURES:

    temp = (
        dataset[
            [
                "network_date",
                feature
            ]
            +
            FUTURE_OUTCOMES
        ]
        .dropna(
            subset=[
                feature
            ]
        )
        .copy()
    )


    # --------------------------------------------------------
    # rank(method="first") 避免大量相同值导致qcut失败
    # --------------------------------------------------------

    ranked_feature = (
        temp[
            feature
        ]
        .rank(
            method="first"
        )
    )


    temp[
        "feature_group"
    ] = pd.qcut(
        ranked_feature,
        q=3,
        labels=[
            "Low",
            "Middle",
            "High"
        ]
    )


    for group_name in [
        "Low",
        "Middle",
        "High"
    ]:

        group = (
            temp[
                temp[
                    "feature_group"
                ]
                ==
                group_name
            ]
        )


        row = {
            "network_feature":
                feature,

            "feature_group":
                group_name,

            "n":
                len(
                    group
                ),

            "mean_feature_value":
                group[
                    feature
                ]
                .mean(),

            "median_feature_value":
                group[
                    feature
                ]
                .median()
        }


        for outcome in FUTURE_OUTCOMES:

            row[
                f"mean_{outcome}"
            ] = (
                group[
                    outcome
                ]
                .mean()
            )


            row[
                f"median_{outcome}"
            ] = (
                group[
                    outcome
                ]
                .median()
            )


        quantile_rows.append(
            row
        )


quantile_summary = pd.DataFrame(
    quantile_rows
)


quantile_summary.to_csv(
    QUANTILE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Regime -> Future Outcome
#
# 每个R1-R5对应之后20日发生了什么。
# ============================================================

regime_rows = []


regime_order = (
    dataset[
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
        dataset[
            dataset[
                "regime"
            ]
            ==
            regime
        ]
    )


    row = {
        "regime":
            regime,

        "n":
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


    for outcome in FUTURE_OUTCOMES:

        row[
            f"mean_{outcome}"
        ] = (
            group[
                outcome
            ]
            .mean()
        )


        row[
            f"median_{outcome}"
        ] = (
            group[
                outcome
            ]
            .median()
        )


        row[
            f"sd_{outcome}"
        ] = (
            group[
                outcome
            ]
            .std(
                ddof=1
            )
        )


    regime_rows.append(
        row
    )


regime_future_summary = pd.DataFrame(
    regime_rows
)


regime_future_summary.to_csv(
    REGIME_FUTURE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 屏幕输出
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 11 - Stage 4"
)

print(
    "Network Features vs Future Market Outcomes"
)

print(
    "============================================"
)


print(
    "\n--- Dataset ---"
)


print(
    "N =",
    len(
        dataset
    )
)


print(
    "Date Range:",
    dataset[
        "network_date"
    ]
    .min(),
    "->",
    dataset[
        "network_date"
    ]
    .max()
)


# ============================================================
# 17. 输出Absolute Spearman最大的关系
# ============================================================

print(
    "\n--- Largest Absolute Spearman Relations ---"
)


print(
    spearman_summary
    .head(
        20
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 18. Regime Future Summary
# ============================================================

print(
    "\n--- Future Market Outcomes by Regime ---"
)


regime_display_cols = [
    "regime",
    "n",

    "mean_CSI300_future_return_20",
    "mean_CSI300_future_volatility_20",

    "mean_CSI500_future_return_20",
    "mean_CSI500_future_volatility_20",

    "mean_future_relative_return_20"
]


print(
    regime_future_summary[
        regime_display_cols
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 19. Figure 1：
# Same/Cross Strength Ratio vs Future CSI300 Return
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        6
    )
)


ax.scatter(
    dataset[
        "same_cross_strength_ratio"
    ],
    dataset[
        "CSI300_future_return_20"
    ]
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Current Same / Cross Partial Strength Ratio"
)


ax.set_ylabel(
    "Future 20-day CSI300 Return"
)


ax.set_title(
    "Network Industry Concentration vs Future CSI300 Return"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "network_strength_ratio_vs_future_CSI300_return.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. Figure 2：
# Same/Cross Strength Ratio vs Future Relative Return
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        6
    )
)


ax.scatter(
    dataset[
        "same_cross_strength_ratio"
    ],
    dataset[
        "future_relative_return_20"
    ]
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Current Same / Cross Partial Strength Ratio"
)


ax.set_ylabel(
    "Future CSI500 - CSI300 20-day Return"
)


ax.set_title(
    "Network Industry Concentration vs Future Relative Market Return"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "network_strength_ratio_vs_future_relative_return.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. Figure 3：
# Same Ratio Terciles -> Future CSI300 Return
# ============================================================

same_ratio_groups = (
    quantile_summary[
        quantile_summary[
            "network_feature"
        ]
        ==
        "same_edge_ratio"
    ]
    .copy()
)


group_order = [
    "Low",
    "Middle",
    "High"
]


same_ratio_groups[
    "feature_group"
] = pd.Categorical(
    same_ratio_groups[
        "feature_group"
    ],
    categories=group_order,
    ordered=True
)


same_ratio_groups = (
    same_ratio_groups
    .sort_values(
        "feature_group"
    )
)


fig, ax = plt.subplots(
    figsize=(
        8,
        6
    )
)


ax.bar(
    same_ratio_groups[
        "feature_group"
    ].astype(str),
    same_ratio_groups[
        "mean_CSI300_future_return_20"
    ]
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Current Same-industry Edge Ratio Group"
)


ax.set_ylabel(
    "Mean Future 20-day CSI300 Return"
)


ax.set_title(
    "Future CSI300 Return by Network Industry Concentration"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "same_ratio_tercile_future_CSI300_return.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. Figure 4：
# Regime -> Future CSI300 / CSI500 Return
# ============================================================

plot_regime = (
    regime_future_summary[
        [
            "regime",
            "mean_CSI300_future_return_20",
            "mean_CSI500_future_return_20"
        ]
    ]
    .copy()
    .set_index(
        "regime"
    )
)


fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


plot_regime.plot(
    kind="bar",
    ax=ax
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Current Network Regime"
)


ax.set_ylabel(
    "Mean Future 20-day Return"
)


ax.set_title(
    "Future Market Return by Current Network Regime"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "regime_future_market_return.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 完成
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 4完成"
)

print(
    "============================================"
)


for path in [
    DATASET_FILE,
    SPEARMAN_FILE,
    QUANTILE_FILE,
    REGIME_FUTURE_FILE
]:

    print(
        path
    )
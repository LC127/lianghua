from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

FUTURE_DATA_FILE = (
    PROCESSED_DIR
    / "network_future_market_dataset.csv"
)

CURRENT_MARKET_FILE = (
    PROCESSED_DIR
    / "external_market_state_window_metrics.csv"
)


# ============================================================
# 3. 输出文件
# ============================================================

MODEL_DATA_FILE = (
    PROCESSED_DIR
    / "incremental_prediction_dataset.csv"
)

PREDICTION_FILE = (
    PROCESSED_DIR
    / "incremental_prediction_walk_forward.csv"
)

PERFORMANCE_FILE = (
    PROCESSED_DIR
    / "incremental_prediction_performance.csv"
)

SENSITIVITY_FILE = (
    PROCESSED_DIR
    / "incremental_prediction_train_size_sensitivity.csv"
)

COEFFICIENT_FILE = (
    PROCESSED_DIR
    / "incremental_prediction_coefficient_path.csv"
)


# ============================================================
# 4. 参数
# ============================================================

# 主分析：
# 前15个Network Dates用于第一次训练
INITIAL_TRAIN_SIZE = 15


# 敏感性分析
INITIAL_TRAIN_SIZES = [
    12,
    15,
    18
]


TARGET = (
    "future_relative_volatility_20"
)


BASELINE_FEATURES = [
    "current_relative_volatility_20"
]


NETWORK_FEATURES_MAIN = [
    "current_relative_volatility_20",
    "same_edge_ratio"
]


# Edge Count只作为Sensitivity Check
NETWORK_FEATURES_EDGECOUNT = [
    "current_relative_volatility_20",
    "edge_count"
]


# ============================================================
# 5. 读取Stage 4 Future Dataset
# ============================================================

future_df = pd.read_csv(
    FUTURE_DATA_FILE
)


required_future_cols = [
    "network_date",
    "regime",
    "edge_count",
    "same_edge_ratio",
    "future_relative_volatility_20"
]


missing = [
    col
    for col in required_future_cols
    if col not in future_df.columns
]


if missing:

    raise ValueError(
        "network_future_market_dataset.csv "
        f"缺少字段：{missing}"
    )


future_df[
    "network_date"
] = pd.to_datetime(
    future_df[
        "network_date"
    ]
)


for col in [
    "edge_count",
    "same_edge_ratio",
    "future_relative_volatility_20"
]:

    future_df[col] = pd.to_numeric(
        future_df[col],
        errors="coerce"
    )


future_df = (
    future_df
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
# 6. 读取Stage 2 Current Market State
#
# 每个network_date × index一行。
# ============================================================

current_df = pd.read_csv(
    CURRENT_MARKET_FILE
)


required_current_cols = [
    "network_date",
    "index_key",
    "annualized_volatility_20"
]


missing = [
    col
    for col in required_current_cols
    if col not in current_df.columns
]


if missing:

    raise ValueError(
        "external_market_state_window_metrics.csv "
        f"缺少字段：{missing}"
    )


current_df[
    "network_date"
] = pd.to_datetime(
    current_df[
        "network_date"
    ]
)


current_df[
    "annualized_volatility_20"
] = pd.to_numeric(
    current_df[
        "annualized_volatility_20"
    ],
    errors="coerce"
)


# ============================================================
# 7. 提取CSI300和CSI500当前波动率
# ============================================================

current_vol = (
    current_df[
        current_df[
            "index_key"
        ]
        .isin(
            [
                "CSI300",
                "CSI500"
            ]
        )
    ]
    .pivot(
        index="network_date",
        columns="index_key",
        values="annualized_volatility_20"
    )
    .reset_index()
)


required_indices = [
    "CSI300",
    "CSI500"
]


for col in required_indices:

    if col not in current_vol.columns:

        raise ValueError(
            f"Current Market Data缺少：{col}"
        )


# ============================================================
# 8. 构造Current Relative Volatility
#
# CSI500 - CSI300
# ============================================================

current_vol[
    "current_relative_volatility_20"
] = (
    current_vol[
        "CSI500"
    ]
    -
    current_vol[
        "CSI300"
    ]
)


current_vol = (
    current_vol
    .rename(
        columns={
            "CSI300":
                "current_CSI300_volatility_20",

            "CSI500":
                "current_CSI500_volatility_20"
        }
    )
)


# ============================================================
# 9. 合并Current X和Future Y
# ============================================================

model_df = pd.merge(
    future_df,
    current_vol,
    on="network_date",
    how="inner"
)


required_model_cols = [
    "network_date",
    "regime",

    "edge_count",
    "same_edge_ratio",

    "current_CSI300_volatility_20",
    "current_CSI500_volatility_20",
    "current_relative_volatility_20",

    TARGET
]


model_df = (
    model_df[
        required_model_cols
    ]
    .dropna()
    .sort_values(
        "network_date"
    )
    .reset_index(
        drop=True
    )
)


model_df.to_csv(
    MODEL_DATA_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Prediction Dataset N =",
    len(
        model_df
    )
)


# ============================================================
# 10. Walk-forward函数
# ============================================================

def walk_forward_prediction(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    initial_train_size: int,
    model_name: str
) -> pd.DataFrame:
    """
    Expanding-window one-step-ahead prediction.

    第一次：
        train = [0, ..., initial_train_size - 1]
        test  = initial_train_size

    第二次：
        train = [0, ..., initial_train_size]
        test  = initial_train_size + 1

    依次扩展。

    注意：
    每一步模型只使用预测日期之前的观测。
    """

    rows = []


    for test_pos in range(
        initial_train_size,
        len(data)
    ):

        train = (
            data
            .iloc[
                :test_pos
            ]
            .copy()
        )


        test = (
            data
            .iloc[
                [test_pos]
            ]
            .copy()
        )


        X_train = (
            train[
                feature_cols
            ]
            .to_numpy()
        )


        y_train = (
            train[
                target_col
            ]
            .to_numpy()
        )


        X_test = (
            test[
                feature_cols
            ]
            .to_numpy()
        )


        model = LinearRegression()


        model.fit(
            X_train,
            y_train
        )


        prediction = float(
            model.predict(
                X_test
            )[0]
        )


        actual = float(
            test[
                target_col
            ]
            .iloc[0]
        )


        row = {
            "model":
                model_name,

            "network_date":
                test[
                    "network_date"
                ]
                .iloc[0],

            "regime":
                test[
                    "regime"
                ]
                .iloc[0],

            "train_size":
                len(
                    train
                ),

            "actual":
                actual,

            "prediction":
                prediction,

            "error":
                actual
                -
                prediction,

            "absolute_error":
                abs(
                    actual
                    -
                    prediction
                ),

            "squared_error":
                (
                    actual
                    -
                    prediction
                ) ** 2,

            "intercept":
                float(
                    model.intercept_
                )
        }


        # 保存每一步系数
        for feature, coef in zip(
            feature_cols,
            model.coef_
        ):

            row[
                f"coef_{feature}"
            ] = float(
                coef
            )


        rows.append(
            row
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. 主分析：
# Baseline
# ============================================================

pred_baseline = walk_forward_prediction(
    data=model_df,
    feature_cols=BASELINE_FEATURES,
    target_col=TARGET,
    initial_train_size=INITIAL_TRAIN_SIZE,
    model_name="Baseline_CurrentRelativeVol"
)


# ============================================================
# 12. 主分析：
# Baseline + Same Edge Ratio
# ============================================================

pred_same_ratio = walk_forward_prediction(
    data=model_df,
    feature_cols=NETWORK_FEATURES_MAIN,
    target_col=TARGET,
    initial_train_size=INITIAL_TRAIN_SIZE,
    model_name="Augmented_SameEdgeRatio"
)


# ============================================================
# 13. Sensitivity：
# Baseline + Edge Count
#
# 不与Same Ratio同时放，
# 避免强共线。
# ============================================================

pred_edge_count = walk_forward_prediction(
    data=model_df,
    feature_cols=NETWORK_FEATURES_EDGECOUNT,
    target_col=TARGET,
    initial_train_size=INITIAL_TRAIN_SIZE,
    model_name="Augmented_EdgeCount"
)


# ============================================================
# 14. 合并预测
# ============================================================

predictions = pd.concat(
    [
        pred_baseline,
        pred_same_ratio,
        pred_edge_count
    ],
    ignore_index=True
)


predictions.to_csv(
    PREDICTION_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Performance函数
# ============================================================

def calculate_performance(
    prediction_df: pd.DataFrame
) -> dict:

    actual = (
        prediction_df[
            "actual"
        ]
        .to_numpy()
    )


    pred = (
        prediction_df[
            "prediction"
        ]
        .to_numpy()
    )


    mae = mean_absolute_error(
        actual,
        pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            actual,
            pred
        )
    )


    # 预测值和真实值的描述性相关
    if (
        len(actual) >= 3
        and
        np.std(actual) > 0
        and
        np.std(pred) > 0
    ):

        pearson_corr = np.corrcoef(
            actual,
            pred
        )[0, 1]


        spearman_corr = pd.Series(
            actual
        ).corr(
            pd.Series(
                pred
            ),
            method="spearman"
        )

    else:

        pearson_corr = np.nan
        spearman_corr = np.nan


    return {
        "n_forecasts":
            len(
                prediction_df
            ),

        "MAE":
            mae,

        "RMSE":
            rmse,

        "prediction_actual_pearson":
            pearson_corr,

        "prediction_actual_spearman":
            spearman_corr
    }


# ============================================================
# 16. 计算主模型Performance
# ============================================================

performance_rows = []


for model_name, group in predictions.groupby(
    "model"
):

    perf = calculate_performance(
        group
    )


    perf[
        "model"
    ] = model_name


    performance_rows.append(
        perf
    )


performance = pd.DataFrame(
    performance_rows
)


# ============================================================
# 17. 与Baseline比较
# ============================================================

baseline_perf = (
    performance[
        performance[
            "model"
        ]
        ==
        "Baseline_CurrentRelativeVol"
    ]
    .iloc[0]
)


baseline_rmse = (
    baseline_perf[
        "RMSE"
    ]
)


baseline_mae = (
    baseline_perf[
        "MAE"
    ]
)


# Baseline SSE
baseline_sse = (
    pred_baseline[
        "squared_error"
    ]
    .sum()
)


comparison_rows = []


for _, row in performance.iterrows():

    model_name = row[
        "model"
    ]


    temp_pred = (
        predictions[
            predictions[
                "model"
            ]
            ==
            model_name
        ]
    )


    model_sse = (
        temp_pred[
            "squared_error"
        ]
        .sum()
    )


    rmse_improvement = (
        (
            baseline_rmse
            -
            row[
                "RMSE"
            ]
        )
        /
        baseline_rmse
    )


    mae_improvement = (
        (
            baseline_mae
            -
            row[
                "MAE"
            ]
        )
        /
        baseline_mae
    )


    # --------------------------------------------------------
    # Relative OOS R2 vs Baseline
    #
    # 1 - SSE_model / SSE_baseline
    #
    # >0:
    #   比Baseline好
    #
    # <0:
    #   比Baseline差
    #
    # 注意：
    # 这是相对benchmark的forecast improvement measure，
    # 不是普通样本内R²。
    # --------------------------------------------------------

    relative_oos_r2 = (
        1.0
        -
        model_sse
        /
        baseline_sse
    )


    comparison_rows.append(
        {
            **row.to_dict(),

            "RMSE_improvement_vs_baseline":
                rmse_improvement,

            "MAE_improvement_vs_baseline":
                mae_improvement,

            "relative_OOS_R2_vs_baseline":
                relative_oos_r2
        }
    )


performance_comparison = pd.DataFrame(
    comparison_rows
)


performance_comparison.to_csv(
    PERFORMANCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. 系数路径
#
# 主要查看Same Edge Ratio系数方向是否稳定。
# ============================================================

coefficient_cols = [
    col
    for col in predictions.columns
    if col.startswith(
        "coef_"
    )
]


coefficient_path = (
    predictions[
        [
            "model",
            "network_date",
            "train_size"
        ]
        +
        coefficient_cols
    ]
    .copy()
)


coefficient_path.to_csv(
    COEFFICIENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. Initial Train Size Sensitivity
#
# 12 / 15 / 18
#
# 对Baseline和SameRatio Model分别重复。
# ============================================================

sensitivity_rows = []


for initial_size in INITIAL_TRAIN_SIZES:

    if (
        initial_size
        >=
        len(
            model_df
        )
        -
        2
    ):

        continue


    base = walk_forward_prediction(
        data=model_df,
        feature_cols=BASELINE_FEATURES,
        target_col=TARGET,
        initial_train_size=initial_size,
        model_name="Baseline"
    )


    aug = walk_forward_prediction(
        data=model_df,
        feature_cols=NETWORK_FEATURES_MAIN,
        target_col=TARGET,
        initial_train_size=initial_size,
        model_name="Augmented_SameEdgeRatio"
    )


    base_perf = calculate_performance(
        base
    )


    aug_perf = calculate_performance(
        aug
    )


    base_sse = (
        base[
            "squared_error"
        ]
        .sum()
    )


    aug_sse = (
        aug[
            "squared_error"
        ]
        .sum()
    )


    sensitivity_rows.append(
        {
            "initial_train_size":
                initial_size,

            "n_forecasts":
                len(
                    aug
                ),

            "baseline_MAE":
                base_perf[
                    "MAE"
                ],

            "augmented_MAE":
                aug_perf[
                    "MAE"
                ],

            "MAE_improvement":
                (
                    base_perf[
                        "MAE"
                    ]
                    -
                    aug_perf[
                        "MAE"
                    ]
                )
                /
                base_perf[
                    "MAE"
                ],

            "baseline_RMSE":
                base_perf[
                    "RMSE"
                ],

            "augmented_RMSE":
                aug_perf[
                    "RMSE"
                ],

            "RMSE_improvement":
                (
                    base_perf[
                        "RMSE"
                    ]
                    -
                    aug_perf[
                        "RMSE"
                    ]
                )
                /
                base_perf[
                    "RMSE"
                ],

            "relative_OOS_R2_vs_baseline":
                (
                    1
                    -
                    aug_sse
                    /
                    base_sse
                ),

            "augmented_prediction_actual_spearman":
                aug_perf[
                    "prediction_actual_spearman"
                ]
        }
    )


sensitivity_df = pd.DataFrame(
    sensitivity_rows
)


sensitivity_df.to_csv(
    SENSITIVITY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 屏幕输出
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 11 - Stage 5"
)

print(
    "Incremental Predictive Relevance"
)

print(
    "============================================"
)


print(
    "\n--- Prediction Dataset ---"
)


print(
    "N =",
    len(
        model_df
    )
)


print(
    "Date Range:",
    model_df[
        "network_date"
    ]
    .min(),
    "->",
    model_df[
        "network_date"
    ]
    .max()
)


print(
    "\nInitial Training Size =",
    INITIAL_TRAIN_SIZE
)


print(
    "Number of OOS Forecasts =",
    len(
        pred_baseline
    )
)


# ============================================================
# 21. 输出模型比较
# ============================================================

print(
    "\n--- Walk-forward Performance ---"
)


display_cols = [
    "model",
    "n_forecasts",
    "MAE",
    "RMSE",
    "RMSE_improvement_vs_baseline",
    "MAE_improvement_vs_baseline",
    "relative_OOS_R2_vs_baseline",
    "prediction_actual_spearman"
]


print(
    performance_comparison[
        display_cols
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 22. 输出Initial Train Sensitivity
# ============================================================

print(
    "\n--- Initial Train Size Sensitivity ---"
)


print(
    sensitivity_df.to_string(
        index=False
    )
)


# ============================================================
# 23. Figure 1：
# Actual vs Walk-forward forecasts
# ============================================================

plot_base = (
    pred_baseline[
        [
            "network_date",
            "actual",
            "prediction"
        ]
    ]
    .rename(
        columns={
            "prediction":
                "baseline_prediction"
        }
    )
)


plot_aug = (
    pred_same_ratio[
        [
            "network_date",
            "prediction"
        ]
    ]
    .rename(
        columns={
            "prediction":
                "network_prediction"
        }
    )
)


plot_df = pd.merge(
    plot_base,
    plot_aug,
    on="network_date",
    how="inner"
)


fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    plot_df[
        "network_date"
    ],
    plot_df[
        "actual"
    ],
    marker="o",
    label="Actual"
)


ax.plot(
    plot_df[
        "network_date"
    ],
    plot_df[
        "baseline_prediction"
    ],
    marker="o",
    label="Baseline"
)


ax.plot(
    plot_df[
        "network_date"
    ],
    plot_df[
        "network_prediction"
    ],
    marker="o",
    label="Baseline + Same Edge Ratio"
)


ax.set_xlabel(
    "Forecast Network Date"
)


ax.set_ylabel(
    "Future CSI500 - CSI300 Annualized Volatility"
)


ax.set_title(
    "Walk-forward Forecasts of Future Relative Volatility"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "walk_forward_relative_volatility_forecast.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. Figure 2：
# Cumulative Squared-error Advantage
#
# 定义：
#
# SE_baseline - SE_network
#
# 累积曲线上升：
# Network model累计表现更好。
#
# 累积曲线下降：
# Baseline累计表现更好。
# ============================================================

loss_df = pd.merge(
    pred_baseline[
        [
            "network_date",
            "squared_error"
        ]
    ]
    .rename(
        columns={
            "squared_error":
                "baseline_squared_error"
        }
    ),

    pred_same_ratio[
        [
            "network_date",
            "squared_error"
        ]
    ]
    .rename(
        columns={
            "squared_error":
                "network_squared_error"
        }
    ),

    on="network_date",
    how="inner"
)


loss_df[
    "squared_error_advantage"
] = (
    loss_df[
        "baseline_squared_error"
    ]
    -
    loss_df[
        "network_squared_error"
    ]
)


loss_df[
    "cumulative_squared_error_advantage"
] = (
    loss_df[
        "squared_error_advantage"
    ]
    .cumsum()
)


fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    loss_df[
        "network_date"
    ],
    loss_df[
        "cumulative_squared_error_advantage"
    ],
    marker="o"
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Forecast Network Date"
)


ax.set_ylabel(
    "Cumulative SE Advantage: Baseline - Network Model"
)


ax.set_title(
    "Cumulative Forecast-error Improvement from Same Edge Ratio"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "cumulative_forecast_error_advantage.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. Figure 3：
# Same Edge Ratio coefficient path
# ============================================================

coef_col = (
    "coef_same_edge_ratio"
)


coef_plot = (
    pred_same_ratio[
        [
            "network_date",
            coef_col
        ]
    ]
    .dropna()
    .copy()
)


fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    coef_plot[
        "network_date"
    ],
    coef_plot[
        coef_col
    ],
    marker="o"
)


ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Forecast Network Date"
)


ax.set_ylabel(
    "Estimated Coefficient on Same Edge Ratio"
)


ax.set_title(
    "Walk-forward Coefficient Stability of Same Edge Ratio"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "same_edge_ratio_coefficient_path.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 26. 完成
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
    MODEL_DATA_FILE,
    PREDICTION_FILE,
    PERFORMANCE_FILE,
    SENSITIVITY_FILE,
    COEFFICIENT_FILE
]:

    print(
        path
    )
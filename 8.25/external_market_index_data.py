from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import akshare as ak


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_DIR = Path("stock_network")

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 输出文件
# ============================================================

DAILY_FILE = (
    PROCESSED_DIR
    / "external_index_daily.csv"
)

WIDE_FILE = (
    PROCESSED_DIR
    / "external_index_wide.csv"
)

QUALITY_FILE = (
    PROCESSED_DIR
    / "external_index_data_quality.csv"
)


# ============================================================
# 3. 外部指数
# ============================================================

INDEX_INFO = {
    "CSI300": {
        "code": "000300",
        "name": "沪深300"
    },

    "CSI500": {
        "code": "000905",
        "name": "中证500"
    },

    # 辅助benchmark
    "SSE": {
        "code": "000001",
        "name": "上证综指"
    }
}


# ============================================================
# 4. 数据日期
#
# 从2023年开始，可以覆盖前面股票网络研究区间。
#
# 截止2026-08-25：
# 一方面覆盖原来的网络样本，
# 另一方面为之后Future-20-day分析预留数据。
# ============================================================

START_DATE = "20230101"
END_DATE = "20260825"


# ============================================================
# 5. 是否包含辅助的上证综指
# ============================================================

INCLUDE_SSE = True


# ============================================================
# 6. 下载单个指数
# ============================================================

def download_index(
    key: str,
    code: str,
    name: str
) -> pd.DataFrame:

    print(
        f"\nDownloading {name} ({code}) ..."
    )

    try:

        df = ak.index_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE
        )

    except Exception as error:

        print(
            "Eastmoney download failed; "
            f"falling back to Sina: {error}"
        )


        df = ak.stock_zh_index_daily(
            symbol=f"sh{code}"
        )


        fallback_dates = pd.to_datetime(
            df["date"]
        )


        df = (
            df.loc[
                (
                    fallback_dates
                    >=
                    pd.to_datetime(
                        START_DATE
                    )
                )
                &
                (
                    fallback_dates
                    <=
                    pd.to_datetime(
                        END_DATE
                    )
                )
            ]
            .copy()
        )


    if df.empty:

        raise RuntimeError(
            f"{name} ({code}) 返回空数据。"
        )


    # --------------------------------------------------------
    # 统一列名
    # --------------------------------------------------------

    rename_dict = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude_pct",
        "涨跌幅": "change_pct",
        "涨跌额": "change_amount",
        "换手率": "turnover_pct"
    }


    df = (
        df
        .rename(
            columns=rename_dict
        )
        .copy()
    )


    required_columns = [
        "date",
        "open",
        "close",
        "high",
        "low"
    ]


    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]


    if missing:

        raise ValueError(
            f"{name} 缺少必要列：{missing}"
        )


    # --------------------------------------------------------
    # 日期
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )


    # --------------------------------------------------------
    # 数值列
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude_pct",
        "change_pct",
        "change_amount",
        "turnover_pct"
    ]


    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


    # --------------------------------------------------------
    # 排序、去重
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "date"
        )
        .drop_duplicates(
            subset="date",
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # 添加指数信息
    # --------------------------------------------------------

    df["index_key"] = key
    df["index_code"] = code
    df["index_name"] = name


    # --------------------------------------------------------
    # Simple Return
    #
    # R_t = P_t / P_{t-1} - 1
    # --------------------------------------------------------

    df["simple_return"] = (
        df["close"]
        .pct_change()
    )


    # --------------------------------------------------------
    # Log Return
    #
    # r_t = log(P_t) - log(P_{t-1})
    # --------------------------------------------------------

    df["log_return"] = (
        np.log(
            df["close"]
        )
        .diff()
    )


    # --------------------------------------------------------
    # 验证：
    #
    # simple_return = exp(log_return) - 1
    # --------------------------------------------------------

    df[
        "simple_return_from_log"
    ] = (
        np.exp(
            df["log_return"]
        )
        -
        1
    )


    df[
        "return_consistency_error"
    ] = (
        df["simple_return"]
        -
        df["simple_return_from_log"]
    ).abs()


    # --------------------------------------------------------
    # 当日简单高低价区间
    # 可作为额外市场活跃度信息
    # --------------------------------------------------------

    df[
        "high_low_range"
    ] = (
        df["high"]
        /
        df["low"]
        -
        1
    )


    # --------------------------------------------------------
    # 成交额log变化
    #
    # 注意：
    # 这里只是为以后市场活跃度研究做准备，
    # 不是今天Stage 1的核心指标。
    # --------------------------------------------------------

    if "amount" in df.columns:

        positive_amount = (
            df["amount"]
            >
            0
        )


        df[
            "log_amount"
        ] = np.nan


        df.loc[
            positive_amount,
            "log_amount"
        ] = np.log(
            df.loc[
                positive_amount,
                "amount"
            ]
        )


        df[
            "log_amount_change"
        ] = (
            df[
                "log_amount"
            ]
            .diff()
        )


    print(
        "Rows:",
        len(
            df
        )
    )

    print(
        "Date:",
        df[
            "date"
        ]
        .min(),
        "->",
        df[
            "date"
        ]
        .max()
    )


    return df


# ============================================================
# 7. 下载全部指数
# ============================================================

all_index_data = []


for key, info in INDEX_INFO.items():

    if (
        key == "SSE"
        and
        not INCLUDE_SSE
    ):

        continue


    temp = download_index(
        key=key,
        code=info["code"],
        name=info["name"]
    )


    all_index_data.append(
        temp
    )


    # 对公开接口稍微留一点间隔
    time.sleep(
        0.5
    )


index_daily = pd.concat(
    all_index_data,
    ignore_index=True
)


# ============================================================
# 8. 调整列顺序
# ============================================================

preferred_columns = [
    "date",

    "index_key",
    "index_code",
    "index_name",

    "open",
    "high",
    "low",
    "close",

    "volume",
    "amount",

    "simple_return",
    "log_return",

    "high_low_range",

    "amplitude_pct",
    "change_pct",
    "change_amount",
    "turnover_pct",

    "log_amount",
    "log_amount_change",

    "simple_return_from_log",
    "return_consistency_error"
]


existing_columns = [
    col
    for col in preferred_columns
    if col in index_daily.columns
]


other_columns = [
    col
    for col in index_daily.columns
    if col not in existing_columns
]


index_daily = index_daily[
    existing_columns
    +
    other_columns
]


# ============================================================
# 9. 保存Long-format数据
# ============================================================

index_daily.to_csv(
    DAILY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 构造Wide-format
#
# 每个交易日一行。
#
# 例如：
#
# CSI300_close
# CSI300_simple_return
# CSI500_close
# CSI500_simple_return
# ...
# ============================================================

wide_metrics = [
    "close",
    "simple_return",
    "log_return",
    "volume",
    "amount"
]


wide_parts = []


for metric in wide_metrics:

    if metric not in index_daily.columns:

        continue


    temp = (
        index_daily
        .pivot(
            index="date",
            columns="index_key",
            values=metric
        )
        .copy()
    )


    temp.columns = [
        f"{index_key}_{metric}"
        for index_key in temp.columns
    ]


    wide_parts.append(
        temp
    )


index_wide = pd.concat(
    wide_parts,
    axis=1
)


index_wide = (
    index_wide
    .sort_index()
    .reset_index()
)


# ============================================================
# 11. 标记主要指数共同交易日
#
# 后面做CSI300与CSI500比较时非常重要。
# ============================================================

main_return_cols = [
    "CSI300_simple_return",
    "CSI500_simple_return"
]


available_main_cols = [
    col
    for col in main_return_cols
    if col in index_wide.columns
]


index_wide[
    "main_indices_complete"
] = (
    index_wide[
        available_main_cols
    ]
    .notna()
    .all(
        axis=1
    )
)


index_wide.to_csv(
    WIDE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. Data Quality Summary
# ============================================================

quality_rows = []


for key, group in index_daily.groupby(
    "index_key"
):

    group = (
        group
        .sort_values(
            "date"
        )
        .copy()
    )


    duplicate_dates = int(
        group[
            "date"
        ]
        .duplicated()
        .sum()
    )


    max_consistency_error = (
        group[
            "return_consistency_error"
        ]
        .max()
    )


    quality_rows.append(
        {
            "index_key":
                key,

            "index_code":
                group[
                    "index_code"
                ]
                .iloc[0],

            "index_name":
                group[
                    "index_name"
                ]
                .iloc[0],

            "start_date":
                group[
                    "date"
                ]
                .min(),

            "end_date":
                group[
                    "date"
                ]
                .max(),

            "n_observations":
                len(
                    group
                ),

            "n_duplicate_dates":
                duplicate_dates,

            "n_missing_close":
                int(
                    group[
                        "close"
                    ]
                    .isna()
                    .sum()
                ),

            "n_missing_simple_return":
                int(
                    group[
                        "simple_return"
                    ]
                    .isna()
                    .sum()
                ),

            "min_close":
                group[
                    "close"
                ]
                .min(),

            "max_close":
                group[
                    "close"
                ]
                .max(),

            "mean_daily_return":
                group[
                    "simple_return"
                ]
                .mean(),

            "daily_return_sd":
                group[
                    "simple_return"
                ]
                .std(
                    ddof=1
                ),

            "max_abs_return":
                group[
                    "simple_return"
                ]
                .abs()
                .max(),

            "max_return_consistency_error":
                max_consistency_error
        }
    )


quality_df = pd.DataFrame(
    quality_rows
)


quality_df.to_csv(
    QUALITY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 进一步检查主要指数交易日期是否一致
# ============================================================

main_keys = [
    "CSI300",
    "CSI500"
]


main_date_sets = {}


for key in main_keys:

    dates = set(
        index_daily.loc[
            index_daily[
                "index_key"
            ]
            ==
            key,
            "date"
        ]
    )

    main_date_sets[
        key
    ] = dates


common_dates = (
    main_date_sets[
        "CSI300"
    ]
    &
    main_date_sets[
        "CSI500"
    ]
)


union_dates = (
    main_date_sets[
        "CSI300"
    ]
    |
    main_date_sets[
        "CSI500"
    ]
)


print(
    "\n============================================"
)

print(
    "External Index Data Summary"
)

print(
    "============================================"
)


print(
    quality_df.to_string(
        index=False
    )
)


print(
    "\nCSI300 dates:",
    len(
        main_date_sets[
            "CSI300"
        ]
    )
)


print(
    "CSI500 dates:",
    len(
        main_date_sets[
            "CSI500"
        ]
    )
)


print(
    "Common dates:",
    len(
        common_dates
    )
)


print(
    "Union dates:",
    len(
        union_dates
    )
)


print(
    "Date overlap ratio:",
    len(
        common_dates
    )
    /
    len(
        union_dates
    )
)


# ============================================================
# 14. 检查Return计算精度
# ============================================================

MAX_ALLOWED_ERROR = 1e-12


max_error = (
    index_daily[
        "return_consistency_error"
    ]
    .max()
)


if (
    pd.notna(
        max_error
    )
    and
    max_error
    >
    MAX_ALLOWED_ERROR
):

    print(
        "\nWARNING:"
        " simple return 与 log return 转换误差较大。"
    )

else:

    print(
        "\nReturn consistency check passed."
    )


# ============================================================
# 15. 最终输出
# ============================================================

print(
    "\n============================================"
)

print(
    "Stage 1 完成"
)

print(
    "============================================"
)


print(
    DAILY_FILE
)

print(
    WIDE_FILE
)

print(
    QUALITY_FILE
)

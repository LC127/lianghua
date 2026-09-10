from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt


# ============================================================
# 0. 路径配置
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\M1_StockNetwork\output"
)

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day2"
    / "04_step4_historical_industry_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Day 1 数据
# ------------------------------------------------------------

DAY1_STAGE1_DIR = Path(
    r"D:\M1_StockNetwork\output\M1_day1\01_stage1_final"
)

STOCK_MASTER_PATH = (
    DAY1_STAGE1_DIR
    / "stock_master.parquet"
)

TRADE_CALENDAR_PATH = (
    DAY1_STAGE1_DIR
    / "trade_calendar_used.csv"
)


DAY1_STAGE3_DIR = Path(
    r"D:\M1_StockNetwork\output\M1_day1\03_stage3_return_validation"
)

RETURN_PANEL_PATH = (
    DAY1_STAGE3_DIR
    / "daily_return_panel.parquet"
)


# ------------------------------------------------------------
# 外部历史行业文件
# ------------------------------------------------------------

EXTERNAL_INDUSTRY_PATH = Path(
    r"D:\lowfreq\status\ind.csv"
)


# ============================================================
# 1. 研究参数
# ============================================================

START_DATE = pd.Timestamp(
    "2015-01-05"
)

END_DATE = pd.Timestamp(
    "2026-09-03"
)


# ------------------------------------------------------------
# Panel 中需要验证的行业层级
# ------------------------------------------------------------

PANEL_INDUSTRY_COLUMNS = [
    "industry_id1",
    "industry_id2",
]


# ------------------------------------------------------------
# ind.csv 可能的日期列名称
# ------------------------------------------------------------

DATE_COLUMN_CANDIDATES = [

    "trade_date",
    "tradeDate",
    "date",
    "Date",
    "TRADE_DATE",
]


# ------------------------------------------------------------
# 若 ind.csv 是长表，
# 可能的股票代码列
# ------------------------------------------------------------

CODE_COLUMN_CANDIDATES = [

    "stock_code",
    "ticker",
    "symbol",
    "secCode",
    "code",
]


# ------------------------------------------------------------
# 若 ind.csv 是长表，
# 可能的行业列
# ------------------------------------------------------------

INDUSTRY_COLUMN_CANDIDATES = [

    "industry",
    "industry_id",
    "industryID",
    "industry_name",
    "industryName",
    "ind",
]


# ------------------------------------------------------------
# Parquet 流式读取
# ------------------------------------------------------------

PARQUET_BATCH_SIZE = 300_000


# ------------------------------------------------------------
# 只有至少这么多共同观测，
# 才将某只股票纳入长期一致性判断
# ------------------------------------------------------------

MIN_COMMON_OBS_PER_STOCK = 20


# ------------------------------------------------------------
# 用于判定外部行业 -> panel 行业
# 是否具有较稳定映射
# ------------------------------------------------------------

HIGH_MAPPING_PURITY = 0.90


# ============================================================
# 2. 通用字符串清洗
# ============================================================

def normalize_text(x):

    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    if s == "":
        return pd.NA

    if s.lower() in {
        "nan",
        "none",
        "null",
        "na",
        "n/a",
    }:
        return pd.NA

    return s


def normalize_stock_code(x):

    """
    将股票代码标准化为 6 位数字。

    示例：
        1       -> 000001
        000001  -> 000001
        000001.XSHE -> 000001
        600000.SH   -> 600000
    """

    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    if s == "":
        return pd.NA

    # 提取连续 6 位数字
    match = re.search(
        r"(?<!\d)(\d{6})(?!\d)",
        s,
    )

    if match:

        return match.group(1)

    # 如果 CSV 把 000001 读成 1 或 1.0
    try:

        value = float(s)

        if (
            np.isfinite(value)
            and
            value.is_integer()
            and
            0 <= value <= 999999
        ):

            return f"{int(value):06d}"

    except Exception:
        pass

    return pd.NA


def normalize_industry_value(x):

    """
    行业标签只做轻度标准化。

    不擅自改变行业编码体系。
    """

    x = normalize_text(x)

    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    # 处理 12.0 这种 CSV 数值编码
    try:

        value = float(s)

        if value.is_integer():

            return str(
                int(value)
            )

    except Exception:
        pass

    return s


# ============================================================
# 3. 读取交易日历
# ============================================================

def load_trade_dates():

    calendar = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    if "trade_date" not in calendar.columns:

        raise ValueError(
            "trade_calendar_used.csv "
            "缺少 trade_date。"
        )

    dates = pd.to_datetime(
        calendar["trade_date"],
        errors="coerce",
    )

    dates = (
        dates
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    dates = dates[
        (
            dates >= START_DATE
        )
        &
        (
            dates <= END_DATE
        )
    ]

    return pd.DatetimeIndex(
        dates
    )


# ============================================================
# 4. Stock Master
# ============================================================

def load_stock_master():

    master = pd.read_parquet(
        STOCK_MASTER_PATH
    )

    required = [

        "security_id",
        "stock_code",
        "stock_name",
        "exchange",
        "list_date",
        "delist_date",
    ]

    missing = [

        x
        for x in required

        if x not in master.columns
    ]

    if missing:

        raise ValueError(
            f"stock_master 缺字段：{missing}"
        )

    master = master.copy()

    master["security_id"] = (
        master["security_id"]
        .astype("string")
        .str.strip()
    )

    master[
        "stock_code_normalized"
    ] = (
        master["stock_code"]
        .map(
            normalize_stock_code
        )
        .astype("string")
    )

    master["list_date"] = (
        pd.to_datetime(
            master["list_date"],
            errors="coerce",
        )
    )

    master["delist_date"] = (
        pd.to_datetime(
            master["delist_date"],
            errors="coerce",
        )
    )

    return master


# ============================================================
# 5. 读取 Return Panel 中的行业信息
#
# 只读取行业验证需要的列，
# 不把收益、市值等无关字段加载进内存。
# ============================================================

def load_panel_industry():

    print()
    print("=" * 76)
    print("读取 daily_return_panel 历史行业标签")
    print("=" * 76)

    parquet_file = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    required = [

        "trade_date",
        "security_id",
        "stock_code",
        "stock_name",
        "exchange",
        "industry_id1",
        "industry_id2",
    ]

    missing = [

        x
        for x in required

        if x not in (
            parquet_file
            .schema_arrow
            .names
        )
    ]

    if missing:

        raise ValueError(
            f"daily_return_panel 缺字段：{missing}"
        )

    parts = []

    total = 0

    for batch_no, batch in enumerate(

        parquet_file.iter_batches(

            batch_size=
                PARQUET_BATCH_SIZE,

            columns=
                required,
        ),

        start=1,
    ):

        df = batch.to_pandas()

        df["trade_date"] = pd.to_datetime(
            df["trade_date"],
            errors="coerce",
        )

        df = df[
            (
                df["trade_date"]
                >=
                START_DATE
            )
            &
            (
                df["trade_date"]
                <=
                END_DATE
            )
        ].copy()

        df[
            "security_id"
        ] = (
            df["security_id"]
            .astype("string")
            .str.strip()
        )

        df[
            "stock_code"
        ] = (
            df["stock_code"]
            .map(
                normalize_stock_code
            )
            .astype("string")
        )

        for column in (
            PANEL_INDUSTRY_COLUMNS
        ):

            df[column] = (
                df[column]
                .map(
                    normalize_industry_value
                )
                .astype("string")
            )

        parts.append(
            df
        )

        total += len(
            df
        )

        if (
            batch_no == 1
            or
            batch_no % 10 == 0
        ):

            print(
                f"Batch {batch_no:4d}: "
                f"{total:,} rows"
            )

    panel = pd.concat(
        parts,
        ignore_index=True,
    )

    print()
    print(
        f"Panel industry rows: {len(panel):,}"
    )

    return panel


# ============================================================
# 6. 自动识别 ind.csv 格式
# ============================================================

def detect_external_industry_format():

    sample = pd.read_csv(
        EXTERNAL_INDUSTRY_PATH,
        nrows=5,
        dtype=str,
    )

    columns = (
        sample.columns
        .tolist()
    )

    date_col = next(
        (
            x
            for x in DATE_COLUMN_CANDIDATES
            if x in columns
        ),
        None,
    )

    code_col = next(
        (
            x
            for x in CODE_COLUMN_CANDIDATES
            if x in columns
        ),
        None,
    )

    industry_col = next(
        (
            x
            for x in INDUSTRY_COLUMN_CANDIDATES
            if x in columns
        ),
        None,
    )

    # 长表：
    # date + code + industry
    if (
        date_col is not None
        and
        code_col is not None
        and
        industry_col is not None
    ):

        return {
            "format":
                "long",

            "date_column":
                date_col,

            "code_column":
                code_col,

            "industry_column":
                industry_col,
        }

    # 宽表通常第一列是日期，其余大量列为股票代码
    if len(columns) > 20:

        if date_col is None:

            date_col = (
                columns[0]
            )

        return {
            "format":
                "wide",

            "date_column":
                date_col,

            "code_column":
                None,

            "industry_column":
                None,
        }

    raise ValueError(
        "无法自动判断 ind.csv 是宽表还是长表。\n"
        f"实际列：{columns}"
    )


# ============================================================
# 7. 读取外部历史行业文件
# ============================================================

def load_external_industry():

    info = (
        detect_external_industry_format()
    )

    print()
    print("=" * 76)
    print("读取外部历史行业文件")
    print("=" * 76)

    print(
        f"Detected format: {info['format']}"
    )

    print(
        f"Date column    : {info['date_column']}"
    )

    if info["format"] == "long":

        df = pd.read_csv(
            EXTERNAL_INDUSTRY_PATH,
            dtype=str,
        )

        df = df[
            [
                info["date_column"],
                info["code_column"],
                info["industry_column"],
            ]
        ].copy()

        df.columns = [

            "trade_date",
            "stock_code",
            "external_industry",
        ]

        df["trade_date"] = pd.to_datetime(
            df["trade_date"],
            errors="coerce",
        )

        df["stock_code"] = (
            df["stock_code"]
            .map(
                normalize_stock_code
            )
            .astype("string")
        )

        df["external_industry"] = (
            df["external_industry"]
            .map(
                normalize_industry_value
            )
            .astype("string")
        )

    else:

        # ====================================================
        # 宽表：
        #
        # trade_date | 000001 | 000002 | 600000 | ...
        #
        # 逐 chunk melt，避免一次性产生超大中间表。
        # ====================================================

        date_column = (
            info[
                "date_column"
            ]
        )

        pieces = []

        for chunk_no, chunk in enumerate(

            pd.read_csv(
                EXTERNAL_INDUSTRY_PATH,
                dtype=str,
                chunksize=100,
            ),

            start=1,
        ):

            if (
                date_column
                not in chunk.columns
            ):

                raise ValueError(
                    f"ind.csv 缺少日期列 "
                    f"{date_column}"
                )

            chunk[
                date_column
            ] = pd.to_datetime(
                chunk[
                    date_column
                ],
                errors="coerce",
            )

            chunk = chunk[
                (
                    chunk[
                        date_column
                    ]
                    >=
                    START_DATE
                )
                &
                (
                    chunk[
                        date_column
                    ]
                    <=
                    END_DATE
                )
            ]

            if chunk.empty:

                continue

            long = (
                chunk.melt(
                    id_vars=[
                        date_column
                    ],
                    var_name=
                        "raw_stock_code",
                    value_name=
                        "external_industry",
                )
            )

            long[
                "stock_code"
            ] = (
                long[
                    "raw_stock_code"
                ]
                .map(
                    normalize_stock_code
                )
                .astype("string")
            )

            long[
                "external_industry"
            ] = (
                long[
                    "external_industry"
                ]
                .map(
                    normalize_industry_value
                )
                .astype("string")
            )

            long = long[
                [
                    date_column,
                    "stock_code",
                    "external_industry",
                ]
            ]

            long = long.rename(
                columns={
                    date_column:
                        "trade_date"
                }
            )

            # 没股票代码的列不是证券列
            long = long[
                long[
                    "stock_code"
                ]
                .notna()
            ]

            pieces.append(
                long
            )

            if (
                chunk_no == 1
                or
                chunk_no % 10 == 0
            ):

                print(
                    f"External chunk "
                    f"{chunk_no:4d}"
                )

        if not pieces:

            raise ValueError(
                "ind.csv 在研究区间内没有可用记录。"
            )

        df = pd.concat(
            pieces,
            ignore_index=True,
        )

    # --------------------------------------------------------
    # 日期过滤
    # --------------------------------------------------------

    df = df[
        (
            df["trade_date"]
            >=
            START_DATE
        )
        &
        (
            df["trade_date"]
            <=
            END_DATE
        )
    ].copy()

    # --------------------------------------------------------
    # 去重检查
    # --------------------------------------------------------

    duplicate_count = int(
        df[
            [
                "trade_date",
                "stock_code",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        print(
            "[WARNING] "
            f"外部行业文件存在 "
            f"{duplicate_count:,} 个重复 "
            "(trade_date, stock_code)。"
        )

        # 相同日期/股票只保留最后一条。
        # 同时后面会输出重复统计。
        df = (
            df
            .drop_duplicates(
                [
                    "trade_date",
                    "stock_code",
                ],
                keep="last",
            )
        )

    print(
        f"External industry rows: "
        f"{len(df):,}"
    )

    return (
        df,
        info,
        duplicate_count,
    )


# ============================================================
# 8. 外部行业日期覆盖
# ============================================================

def build_external_date_coverage(
    external,
    trade_dates,
):

    daily = (
        external
        .groupby(
            "trade_date",
            as_index=False,
        )
        .agg(

            external_stock_count=
                (
                    "stock_code",
                    "nunique",
                ),

            external_nonmissing_industry_count=
                (
                    "external_industry",
                    lambda x:
                        x.notna().sum(),
                ),
        )
    )

    calendar = pd.DataFrame(
        {
            "trade_date":
                trade_dates
        }
    )

    result = (
        calendar.merge(
            daily,
            on="trade_date",
            how="left",
        )
    )

    result[
        "external_date_present"
    ] = (
        result[
            "external_stock_count"
        ]
        .notna()
    )

    return result


# ============================================================
# 9. Panel 与 External 合并
#
# 主键使用：
#
# trade_date + 6位 stock_code
#
# 因为 ind.csv 通常没有 security_id。
#
# 同时输出 unmatched 情况，
# 不静默忽略历史代码问题。
# ============================================================

def merge_panel_external(
    panel,
    external,
):

    external_key = (
        external[
            [
                "trade_date",
                "stock_code",
                "external_industry",
            ]
        ]
        .copy()
    )

    merged = (
        panel.merge(

            external_key,

            on=[
                "trade_date",
                "stock_code",
            ],

            how="left",

            validate="many_to_one",

            indicator=True,
        )
    )

    merged[
        "external_record_found"
    ] = (
        merged[
            "_merge"
        ]
        .eq(
            "both"
        )
    )

    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )

    return merged


# ============================================================
# 10. Coverage Summary
# ============================================================

def build_coverage_summary(
    merged,
):

    rows = []

    total = len(
        merged
    )

    for industry_column in (
        PANEL_INDUSTRY_COLUMNS
    ):

        panel_valid = (
            merged[
                industry_column
            ]
            .notna()
        )

        ext_valid = (
            merged[
                "external_industry"
            ]
            .notna()
        )

        common = (
            panel_valid
            &
            ext_valid
        )

        rows.append(
            {

                "panel_industry_level":
                    industry_column,

                "total_panel_stock_days":
                    total,

                "panel_nonmissing_count":
                    int(
                        panel_valid.sum()
                    ),

                "panel_nonmissing_ratio":
                    float(
                        panel_valid.mean()
                    ),

                "external_record_found_count":
                    int(
                        merged[
                            "external_record_found"
                        ]
                        .sum()
                    ),

                "external_record_found_ratio":
                    float(
                        merged[
                            "external_record_found"
                        ]
                        .mean()
                    ),

                "external_nonmissing_count":
                    int(
                        ext_valid.sum()
                    ),

                "external_nonmissing_ratio":
                    float(
                        ext_valid.mean()
                    ),

                "common_nonmissing_count":
                    int(
                        common.sum()
                    ),

                "common_nonmissing_ratio":
                    float(
                        common.mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. Exact Agreement
#
# 只有两个数据源编码恰好相同时，
# exact agreement 才有直接含义。
#
# 若编码体系不同，
# 后面还会计算 dominant mapping purity。
# ============================================================

def build_exact_agreement_summary(
    merged,
):

    rows = []

    for industry_column in (
        PANEL_INDUSTRY_COLUMNS
    ):

        common = (
            merged[
                industry_column
            ]
            .notna()
            &
            merged[
                "external_industry"
            ]
            .notna()
        )

        sub = merged.loc[
            common,
            [
                industry_column,
                "external_industry",
            ]
        ].copy()

        if sub.empty:

            rows.append(
                {
                    "panel_industry_level":
                        industry_column,

                    "common_count":
                        0,

                    "exact_agreement_count":
                        0,

                    "exact_agreement_ratio":
                        np.nan,
                }
            )

            continue

        equal = (
            sub[
                industry_column
            ]
            ==
            sub[
                "external_industry"
            ]
        )

        rows.append(
            {

                "panel_industry_level":
                    industry_column,

                "common_count":
                    int(
                        len(sub)
                    ),

                "exact_agreement_count":
                    int(
                        equal.sum()
                    ),

                "exact_agreement_ratio":
                    float(
                        equal.mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 12. Dominant Mapping / Crosswalk
#
# 即使外部行业代码与 panel 编码不同，
# 仍可以研究：
#
# external industry A
#     -> panel industry X 占 98%
#
# 如果映射很纯，
# 说明两套分类结构高度对应。
# ============================================================

def build_mapping_crosswalk(
    merged,
    panel_column,
):

    common = (
        merged[
            panel_column
        ]
        .notna()
        &
        merged[
            "external_industry"
        ]
        .notna()
    )

    sub = merged.loc[
        common,
        [
            "external_industry",
            panel_column,
        ]
    ].copy()

    if sub.empty:

        return pd.DataFrame()

    counts = (
        sub.groupby(
            [
                "external_industry",
                panel_column,
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "pair_count"
            }
        )
    )

    total = (
        counts.groupby(
            "external_industry",
            as_index=False,
        )[
            "pair_count"
        ]
        .sum()
        .rename(
            columns={
                "pair_count":
                    "external_total_count"
            }
        )
    )

    counts = counts.merge(
        total,
        on="external_industry",
        how="left",
    )

    counts[
        "conditional_share"
    ] = (
        counts[
            "pair_count"
        ]
        /
        counts[
            "external_total_count"
        ]
    )

    counts[
        "rank_within_external"
    ] = (
        counts.groupby(
            "external_industry"
        )[
            "pair_count"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    dominant = (
        counts[
            counts[
                "rank_within_external"
            ]
            .eq(
                1
            )
        ]
        .copy()
    )

    dominant = dominant.rename(
        columns={
            panel_column:
                "dominant_panel_industry",

            "conditional_share":
                "mapping_purity",
        }
    )

    dominant[
        "panel_industry_level"
    ] = panel_column

    dominant[
        "high_purity_mapping"
    ] = (
        dominant[
            "mapping_purity"
        ]
        >=
        HIGH_MAPPING_PURITY
    )

    return dominant[
        [
            "panel_industry_level",
            "external_industry",
            "dominant_panel_industry",
            "external_total_count",
            "pair_count",
            "mapping_purity",
            "high_purity_mapping",
        ]
    ]


# ============================================================
# 13. Mapping Purity Summary
# ============================================================

def summarize_mapping_purity(
    crosswalk,
):

    if crosswalk.empty:

        return {
            "external_industry_count":
                0,

            "weighted_mapping_purity":
                np.nan,

            "median_mapping_purity":
                np.nan,

            "share_external_industry_purity_ge_90":
                np.nan,
        }

    weights = (
        crosswalk[
            "external_total_count"
        ]
        .to_numpy(
            dtype=float
        )
    )

    purity = (
        crosswalk[
            "mapping_purity"
        ]
        .to_numpy(
            dtype=float
        )
    )

    weighted = (
        np.sum(
            weights
            *
            purity
        )
        /
        np.sum(
            weights
        )
    )

    return {

        "external_industry_count":
            int(
                len(
                    crosswalk
                )
            ),

        "weighted_mapping_purity":
            float(
                weighted
            ),

        "median_mapping_purity":
            float(
                np.median(
                    purity
                )
            ),

        "share_external_industry_purity_ge_90":
            float(
                (
                    purity
                    >=
                    HIGH_MAPPING_PURITY
                )
                .mean()
            ),
    }


# ============================================================
# 14. 日度 Coverage / Agreement
# ============================================================

def build_daily_validation_summary(
    merged,
):

    rows = []

    for date, g in (
        merged.groupby(
            "trade_date"
        )
    ):

        row = {

            "trade_date":
                date,

            "panel_stock_count":
                int(
                    g[
                        "security_id"
                    ]
                    .nunique()
                ),

            "external_match_count":
                int(
                    g[
                        "external_record_found"
                    ]
                    .sum()
                ),

            "external_match_ratio":
                float(
                    g[
                        "external_record_found"
                    ]
                    .mean()
                ),
        }

        for column in (
            PANEL_INDUSTRY_COLUMNS
        ):

            common = (
                g[
                    column
                ]
                .notna()
                &
                g[
                    "external_industry"
                ]
                .notna()
            )

            row[
                f"{column}_coverage"
            ] = float(
                g[
                    column
                ]
                .notna()
                .mean()
            )

            row[
                f"{column}_common_count"
            ] = int(
                common.sum()
            )

            if common.any():

                row[
                    f"{column}_exact_agreement"
                ] = float(
                    (
                        g.loc[
                            common,
                            column
                        ]
                        ==
                        g.loc[
                            common,
                            "external_industry"
                        ]
                    ).mean()
                )

            else:

                row[
                    f"{column}_exact_agreement"
                ] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. 行业变更检测
#
# 这是 Step 4 非常重要的一部分。
#
# 如果 panel 对某只股票 2015-2026 始终
# 只有一个行业，但外部历史文件显示它曾变更，
# 则可能存在 current-label backfill 风险。
# ============================================================

def count_label_changes(
    df,
    label_column,
):

    sub = (
        df[
            [
                "trade_date",
                label_column,
            ]
        ]
        .dropna()
        .sort_values(
            "trade_date"
        )
        .copy()
    )

    if sub.empty:

        return 0

    labels = (
        sub[
            label_column
        ]
        .astype("string")
    )

    changes = (
        labels
        .ne(
            labels.shift(1)
        )
    )

    # 第一条不是“变化”
    return max(
        int(
            changes.sum()
        )
        -
        1,
        0,
    )


def build_stock_history_validation(
    merged,
):

    rows = []

    grouped = (
        merged.groupby(
            "security_id",
            sort=False,
        )
    )

    for counter, (
        security_id,
        g,
    ) in enumerate(
        grouped,
        start=1,
    ):

        g = (
            g
            .sort_values(
                "trade_date"
            )
        )

        basic = {

            "security_id":
                security_id,

            "stock_code":
                g[
                    "stock_code"
                ].iloc[-1],

            "stock_name":
                g[
                    "stock_name"
                ].iloc[-1],

            "exchange":
                g[
                    "exchange"
                ].iloc[-1],

            "first_panel_date":
                g[
                    "trade_date"
                ].min(),

            "last_panel_date":
                g[
                    "trade_date"
                ].max(),

            "panel_stock_day_count":
                int(
                    len(g)
                ),

            "external_record_count":
                int(
                    g[
                        "external_record_found"
                    ]
                    .sum()
                ),

            "external_nonmissing_count":
                int(
                    g[
                        "external_industry"
                    ]
                    .notna()
                    .sum()
                ),

            "external_unique_industry_count":
                int(
                    g[
                        "external_industry"
                    ]
                    .dropna()
                    .nunique()
                ),

            "external_change_count":
                count_label_changes(
                    g,
                    "external_industry",
                ),
        }

        for column in (
            PANEL_INDUSTRY_COLUMNS
        ):

            basic[
                f"{column}_nonmissing_count"
            ] = int(
                g[
                    column
                ]
                .notna()
                .sum()
            )

            basic[
                f"{column}_unique_count"
            ] = int(
                g[
                    column
                ]
                .dropna()
                .nunique()
            )

            basic[
                f"{column}_change_count"
            ] = (
                count_label_changes(
                    g,
                    column,
                )
            )

            # -----------------------------------------------
            # 潜在 current label backfill：
            #
            # external 有 >1 个历史行业，
            # 但 panel 整个历史只有一个。
            # -----------------------------------------------

            basic[
                f"{column}_potential_backfill_flag"
            ] = (

                basic[
                    "external_unique_industry_count"
                ]
                >
                1

                and

                basic[
                    f"{column}_unique_count"
                ]
                <=
                1
            )

        rows.append(
            basic
        )

        if (
            counter == 1
            or
            counter % 1000 == 0
        ):

            print(
                "Stock-history validation: "
                f"{counter:,}"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. 具体的行业变更日期
# ============================================================

def extract_change_events(
    df,
    label_column,
    source_name,
):

    rows = []

    for security_id, g in (
        df.groupby(
            "security_id",
            sort=False,
        )
    ):

        g = (
            g[
                [
                    "trade_date",
                    "stock_code",
                    "stock_name",
                    label_column,
                ]
            ]
            .dropna(
                subset=[
                    label_column
                ]
            )
            .sort_values(
                "trade_date"
            )
            .copy()
        )

        if len(g) < 2:

            continue

        g[
            "previous_label"
        ] = (
            g[
                label_column
            ]
            .shift(1)
        )

        change = (
            g[
                label_column
            ]
            .ne(
                g[
                    "previous_label"
                ]
            )
            &
            g[
                "previous_label"
            ]
            .notna()
        )

        events = g.loc[
            change
        ]

        for row in (
            events.itertuples(
                index=False
            )
        ):

            rows.append(
                {

                    "security_id":
                        security_id,

                    "stock_code":
                        row.stock_code,

                    "stock_name":
                        row.stock_name,

                    "source":
                        source_name,

                    "change_date":
                        row.trade_date,

                    "old_industry":
                        row.previous_label,

                    "new_industry":
                        getattr(
                            row,
                            label_column
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. Panel Industry Concentration
#
# 用于发现明显异常：
# 某天行业种类突然从几十个变成 1 个等。
# ============================================================

def build_industry_count_timeseries(
    merged,
):

    daily = (
        merged.groupby(
            "trade_date",
            as_index=False,
        )
        .agg(

            stock_count=
                (
                    "security_id",
                    "nunique",
                ),

            industry_id1_count=
                (
                    "industry_id1",
                    "nunique",
                ),

            industry_id2_count=
                (
                    "industry_id2",
                    "nunique",
                ),

            external_industry_count=
                (
                    "external_industry",
                    "nunique",
                ),

            industry_id1_missing=
                (
                    "industry_id1",
                    lambda x:
                        x.isna().sum(),
                ),

            industry_id2_missing=
                (
                    "industry_id2",
                    lambda x:
                        x.isna().sum(),
                ),

            external_missing=
                (
                    "external_industry",
                    lambda x:
                        x.isna().sum(),
                ),
        )
    )

    return daily


# ============================================================
# 18. 行业层级推荐
#
# 不根据一个指标“自动宣布正确”。
#
# 这里只形成诊断排序：
# 1. coverage
# 2. mapping purity
# 3. backfill flags
# ============================================================

def build_level_recommendation(
    coverage_summary,
    mapping_summaries,
    stock_history,
):

    rows = []

    for column in (
        PANEL_INDUSTRY_COLUMNS
    ):

        cov = (
            coverage_summary[
                coverage_summary[
                    "panel_industry_level"
                ]
                .eq(
                    column
                )
            ]
            .iloc[0]
        )

        mapping = (
            mapping_summaries[
                column
            ]
        )

        backfill_column = (
            f"{column}_potential_backfill_flag"
        )

        rows.append(
            {

                "panel_industry_level":
                    column,

                "panel_nonmissing_ratio":
                    float(
                        cov[
                            "panel_nonmissing_ratio"
                        ]
                    ),

                "external_common_ratio":
                    float(
                        cov[
                            "common_nonmissing_ratio"
                        ]
                    ),

                "weighted_mapping_purity":
                    mapping[
                        "weighted_mapping_purity"
                    ],

                "median_mapping_purity":
                    mapping[
                        "median_mapping_purity"
                    ],

                "high_purity_external_class_share":
                    mapping[
                        "share_external_industry_purity_ge_90"
                    ],

                "potential_backfill_stock_count":
                    int(
                        stock_history[
                            backfill_column
                        ]
                        .sum()
                    ),

                "potential_backfill_stock_share":
                    float(
                        stock_history[
                            backfill_column
                        ]
                        .mean()
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    # 只做诊断排序，不自动当作最终经济分类选择
    result[
        "diagnostic_rank"
    ] = (
        result[
            [
                "panel_nonmissing_ratio",
                "weighted_mapping_purity",
            ]
        ]
        .mean(
            axis=1
        )
        .rank(
            ascending=False,
            method="min",
        )
    )

    result[
        "important_note"
    ] = (
        "Diagnostic rank only. "
        "Final Same/Cross industry level should also consider "
        "economic granularity and vendor classification meaning."
    )

    return result


# ============================================================
# 19. 绘图
# ============================================================

def plot_results(
    daily_summary,
    industry_count_ts,
):

    # --------------------------------------------------------
    # Figure 1: External match coverage
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    ax.plot(
        daily_summary[
            "trade_date"
        ],
        daily_summary[
            "external_match_ratio"
        ],
    )

    ax.set_title(
        "Historical Industry External Match Coverage"
    )

    ax.set_xlabel(
        "Trade Date"
    )

    ax.set_ylabel(
        "External Match Ratio"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "external_industry_match_coverage.png",
        dpi=180,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Figure 2: Number of industry classes
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    ax.plot(
        industry_count_ts[
            "trade_date"
        ],
        industry_count_ts[
            "industry_id1_count"
        ],
        label=
            "industry_id1",
    )

    ax.plot(
        industry_count_ts[
            "trade_date"
        ],
        industry_count_ts[
            "industry_id2_count"
        ],
        label=
            "industry_id2",
    )

    ax.plot(
        industry_count_ts[
            "trade_date"
        ],
        industry_count_ts[
            "external_industry_count"
        ],
        label=
            "external industry",
    )

    ax.set_title(
        "Number of Industry Classes over Time"
    )

    ax.set_xlabel(
        "Trade Date"
    )

    ax.set_ylabel(
        "Number of Unique Industry Labels"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "industry_class_count_timeseries.png",
        dpi=180,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Figure 3: Panel industry coverage
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    ax.plot(
        daily_summary[
            "trade_date"
        ],
        daily_summary[
            "industry_id1_coverage"
        ],
        label=
            "industry_id1",
    )

    ax.plot(
        daily_summary[
            "trade_date"
        ],
        daily_summary[
            "industry_id2_coverage"
        ],
        label=
            "industry_id2",
    )

    ax.set_title(
        "Panel Historical Industry Coverage"
    )

    ax.set_xlabel(
        "Trade Date"
    )

    ax.set_ylabel(
        "Non-Missing Ratio"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "panel_industry_coverage.png",
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# 20. Metadata
# ============================================================

def save_metadata(
    external_format,
    external_duplicate_count,
):

    metadata = {

        "step":
            "Day2_Step4_Historical_Industry_Validation",

        "research_period": {

            "start":
                str(
                    START_DATE.date()
                ),

            "end":
                str(
                    END_DATE.date()
                ),
        },

        "panel_source":
            str(
                RETURN_PANEL_PATH
            ),

        "external_industry_source":
            str(
                EXTERNAL_INDUSTRY_PATH
            ),

        "external_file_detected_format":
            external_format,

        "external_duplicate_key_count":
            external_duplicate_count,

        "panel_industry_levels":
            PANEL_INDUSTRY_COLUMNS,

        "merge_key":
            [
                "trade_date",
                "normalized 6-digit stock_code",
            ],

        "validation_components": [

            "historical coverage",
            "exact code agreement where meaningful",
            "dominant classification mapping purity",
            "industry change counts",
            "potential current-label backfill flags",
            "industry-class count stability",
        ],

        "important_limitations": [

            (
                "High agreement between historically indexed "
                "files does not by itself prove that the vendor "
                "classification was genuinely known in real time. "
                "The test validates historical indexing and "
                "cross-source consistency, not vendor publication "
                "timestamp semantics."
            ),

            (
                "Exact agreement is only directly interpretable "
                "when external and panel files use the same "
                "industry coding system."
            ),

            (
                "When code systems differ, dominant mapping purity "
                "is used as a structural crosswalk diagnostic "
                "rather than treating unequal numeric labels as "
                "classification disagreement."
            ),

            (
                "Historical code changes can create unmatched "
                "records when the external file only contains "
                "ticker rather than a stable security identifier. "
                "Such unmatched observations should be reviewed "
                "rather than automatically mapped using broad "
                "heuristics."
            ),

            (
                "Final choice between industry_id1 and industry_id2 "
                "must consider economic granularity in addition "
                "to statistical consistency."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "step4_historical_industry_metadata.json",
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
# 21. 主程序
# ============================================================

def main():

    print("=" * 76)
    print("M1 Day 2 - Step 4")
    print("Historical Industry Validation")
    print("=" * 76)

    # ========================================================
    # 1. 基础数据
    # ========================================================

    print()
    print("[1] 读取 Trading Calendar")

    trade_dates = (
        load_trade_dates()
    )

    print(
        f"Trading days: {len(trade_dates):,}"
    )

    print()
    print("[2] 读取 Stock Master")

    master = (
        load_stock_master()
    )

    print(
        f"Historical securities: {len(master):,}"
    )

    # ========================================================
    # 2. Panel Industry
    # ========================================================

    print()
    print("[3] 读取 Panel Historical Industry")

    panel = (
        load_panel_industry()
    )

    # ========================================================
    # 3. External Industry
    # ========================================================

    print()
    print("[4] 读取 External Historical Industry")

    (
        external,
        external_info,
        external_duplicate_count,
    ) = (
        load_external_industry()
    )

    # ========================================================
    # 4. External date coverage
    # ========================================================

    external_date_coverage = (
        build_external_date_coverage(
            external=
                external,

            trade_dates=
                trade_dates,
        )
    )

    external_date_coverage.to_csv(
        OUTPUT_DIR
        / "external_industry_date_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 5. Merge
    # ========================================================

    print()
    print("[5] Panel × External Merge")

    merged = (
        merge_panel_external(
            panel=
                panel,

            external=
                external,
        )
    )

    # 不保存 1100 万行 merged 主文件，
    # 避免产生巨大重复结果。
    # 后面的 summary 已足够做 QA。

    # ========================================================
    # 6. Coverage
    # ========================================================

    print()
    print("[6] Coverage Analysis")

    coverage_summary = (
        build_coverage_summary(
            merged
        )
    )

    coverage_summary.to_csv(
        OUTPUT_DIR
        / "industry_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 7. Exact agreement
    # ========================================================

    exact_agreement = (
        build_exact_agreement_summary(
            merged
        )
    )

    exact_agreement.to_csv(
        OUTPUT_DIR
        / "industry_exact_agreement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 8. Crosswalk / Mapping purity
    # ========================================================

    print()
    print("[7] Classification Mapping Diagnostics")

    mapping_summaries = {}

    for column in (
        PANEL_INDUSTRY_COLUMNS
    ):

        crosswalk = (
            build_mapping_crosswalk(
                merged=
                    merged,

                panel_column=
                    column,
            )
        )

        crosswalk.to_csv(
            OUTPUT_DIR
            /
            f"external_to_{column}_crosswalk.csv",
            index=False,
            encoding="utf-8-sig",
        )

        mapping_summaries[
            column
        ] = (
            summarize_mapping_purity(
                crosswalk
            )
        )

    mapping_summary_df = pd.DataFrame(
        [
            {
                "panel_industry_level":
                    level,
                **values,
            }

            for (
                level,
                values,
            ) in (
                mapping_summaries.items()
            )
        ]
    )

    mapping_summary_df.to_csv(
        OUTPUT_DIR
        / "industry_mapping_purity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 9. Daily validation
    # ========================================================

    print()
    print("[8] Daily Historical Validation")

    daily_validation = (
        build_daily_validation_summary(
            merged
        )
    )

    daily_validation.to_csv(
        OUTPUT_DIR
        / "industry_validation_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 10. Stock history
    # ========================================================

    print()
    print("[9] Stock-Level Historical Change Validation")

    stock_history = (
        build_stock_history_validation(
            merged
        )
    )

    stock_history.to_csv(
        OUTPUT_DIR
        / "stock_industry_history_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 潜在 backfill 股票
    # --------------------------------------------------------

    backfill_flags = (
        stock_history[
            stock_history[
                [
                    "industry_id1_potential_backfill_flag",
                    "industry_id2_potential_backfill_flag",
                ]
            ]
            .any(
                axis=1
            )
        ]
        .copy()
    )

    backfill_flags.to_csv(
        OUTPUT_DIR
        / "potential_current_label_backfill_stocks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 11. Change events
    # ========================================================

    print()
    print("[10] Extract Industry Change Events")

    change_parts = []

    for column in (
        PANEL_INDUSTRY_COLUMNS
    ):

        events = (
            extract_change_events(
                df=
                    merged,

                label_column=
                    column,

                source_name=
                    column,
            )
        )

        if not events.empty:

            change_parts.append(
                events
            )

    external_events = (
        extract_change_events(
            df=
                merged,

            label_column=
                "external_industry",

            source_name=
                "external_industry",
        )
    )

    if not external_events.empty:

        change_parts.append(
            external_events
        )

    if change_parts:

        change_events = pd.concat(
            change_parts,
            ignore_index=True,
        )

    else:

        change_events = pd.DataFrame()

    change_events.to_csv(
        OUTPUT_DIR
        / "historical_industry_change_events.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 12. Industry count timeseries
    # ========================================================

    industry_count_ts = (
        build_industry_count_timeseries(
            merged
        )
    )

    industry_count_ts.to_csv(
        OUTPUT_DIR
        / "industry_class_count_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 13. Industry level diagnostic
    # ========================================================

    level_recommendation = (
        build_level_recommendation(
            coverage_summary=
                coverage_summary,

            mapping_summaries=
                mapping_summaries,

            stock_history=
                stock_history,
        )
    )

    level_recommendation.to_csv(
        OUTPUT_DIR
        / "industry_level_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 14. Figures
    # ========================================================

    print()
    print("[11] 绘图")

    plot_results(
        daily_summary=
            daily_validation,

        industry_count_ts=
            industry_count_ts,
    )

    # ========================================================
    # 15. Metadata
    # ========================================================

    save_metadata(
        external_format=
            external_info,

        external_duplicate_count=
            external_duplicate_count,
    )

    # ========================================================
    # 16. Console summary
    # ========================================================

    print()
    print("=" * 76)
    print("Day 2 Step 4 完成")
    print("=" * 76)

    print()
    print("Coverage:")
    print(
        coverage_summary
        .to_string(
            index=False
        )
    )

    print()
    print("Exact agreement:")
    print(
        exact_agreement
        .to_string(
            index=False
        )
    )

    print()
    print("Mapping purity:")
    print(
        mapping_summary_df
        .to_string(
            index=False
        )
    )

    print()
    print("Industry level diagnostics:")
    print(
        level_recommendation
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Potential current-label "
        "backfill stocks: "
        f"{len(backfill_flags):,}"
    )

    print()
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. 路径配置
# ============================================================

# ------------------------------------------------------------
# Day 1 - Step 1
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


# ------------------------------------------------------------
# Day 1 - Step 3
# ------------------------------------------------------------

DAY1_STAGE3_DIR = Path(
    r"D:\M1_StockNetwork\output\M1_day1\03_stage3_return_validation"
)

RETURN_PANEL_PATH = (
    DAY1_STAGE3_DIR
    / "daily_return_panel.parquet"
)

RETURN_DEFINITION_PATH = (
    DAY1_STAGE3_DIR
    / "return_definition.json"
)


# ------------------------------------------------------------
# Day 1 - Step 4
# ------------------------------------------------------------

DAY1_STAGE4_DIR = Path(
    r"D:\M1_StockNetwork\output\M1_day1\04_step4_network_feasibility"
)

FEASIBILITY_SUMMARY_PATH = (
    DAY1_STAGE4_DIR
    / "network_feasibility_summary.csv"
)

FEASIBILITY_LATEST_PATH = (
    DAY1_STAGE4_DIR
    / "network_feasibility_latest_snapshot.csv"
)


# ------------------------------------------------------------
# Day 2 - Step 1 输出
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\M1_StockNetwork\output\M1_day2\01_step1_baseline_config"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. 研究区间
# ============================================================

START_DATE = pd.Timestamp(
    "2015-01-05"
)

END_DATE = pd.Timestamp(
    "2026-09-03"
)

EXPECTED_TRADE_DAY_COUNT = 2837


# ============================================================
# 2. Day 2 基准研究参数
# ============================================================

# ------------------------------------------------------------
# 主收益变量
# ------------------------------------------------------------

RETURN_COLUMN = "return_network"

ROBUST_RETURN_COLUMN = "return_last_trade_log"


# ------------------------------------------------------------
# 滚动窗口
# ------------------------------------------------------------

WINDOWS = [
    60,
    120,
    252,
]


# ------------------------------------------------------------
# 单只股票有效收益率要求
# ------------------------------------------------------------

BASELINE_VALID_RATIO = 0.90

ROBUST_VALID_RATIOS = [
    0.80,
    0.95,
]


# ------------------------------------------------------------
# Pairwise correlation 至少需要的共同样本比例
# ------------------------------------------------------------

PAIRWISE_MIN_COMMON_RATIO = 0.90


# ------------------------------------------------------------
# Day 2 结构诊断频率
#
# 这里使用月末，而不是每天都构造 5000 x 5000 矩阵。
# ------------------------------------------------------------

ANALYSIS_FREQUENCY = "month_end"


# ------------------------------------------------------------
# 主市场
# ------------------------------------------------------------

TARGET_EXCHANGES = [
    "SSE",
    "SZSE",
]


# ------------------------------------------------------------
# 随机抽 pair 时使用统一随机种子
# ------------------------------------------------------------

RANDOM_SEED = 20260910


# ============================================================
# 3. 必须包含的收益面板字段
# ============================================================

REQUIRED_PANEL_COLUMNS = [

    "trade_date",
    "security_id",
    "stock_code",
    "stock_name",
    "exchange",

    "is_open",
    "is_suspended",

    "adjusted_close",

    "return_network",
    "return_last_trade_log",

    "volume",
    "turnover_value",

    "market_value",
    "neg_market_value",
    "turnover_rate",

    "industry_id1",
    "industry_id2",
]


# ============================================================
# 4. 文件存在性检查
# ============================================================

def check_required_files():

    required_files = {

        "stock_master":
            STOCK_MASTER_PATH,

        "trade_calendar":
            TRADE_CALENDAR_PATH,

        "return_panel":
            RETURN_PANEL_PATH,

        "return_definition":
            RETURN_DEFINITION_PATH,

        "feasibility_summary":
            FEASIBILITY_SUMMARY_PATH,

        "feasibility_latest":
            FEASIBILITY_LATEST_PATH,
    }

    rows = []

    missing = []

    for name, path in (
        required_files.items()
    ):

        exists = path.exists()

        rows.append(
            {
                "input_name":
                    name,

                "path":
                    str(path),

                "exists":
                    exists,
            }
        )

        if not exists:

            missing.append(
                str(path)
            )

    manifest = pd.DataFrame(
        rows
    )

    manifest.to_csv(
        OUTPUT_DIR
        / "input_file_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if missing:

        message = (
            "以下必要输入文件不存在：\n"
            +
            "\n".join(
                missing
            )
        )

        raise FileNotFoundError(
            message
        )

    return manifest


# ============================================================
# 5. 读取交易日历
# ============================================================

def load_trade_dates():

    calendar = pd.read_csv(
        TRADE_CALENDAR_PATH
    )

    if (
        "trade_date"
        not in calendar.columns
    ):

        raise ValueError(
            "trade_calendar_used.csv "
            "缺少 trade_date 字段。"
        )

    dates = pd.to_datetime(
        calendar[
            "trade_date"
        ],
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

    if (
        len(dates)
        != EXPECTED_TRADE_DAY_COUNT
    ):

        raise ValueError(
            f"交易日数量={len(dates)}，"
            f"预期={EXPECTED_TRADE_DAY_COUNT}。"
        )

    return pd.DatetimeIndex(
        dates
    )


# ============================================================
# 6. Stock Master 基本检查
# ============================================================

def inspect_stock_master():

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
            f"stock_master 缺少字段：{missing}"
        )

    master = master.copy()

    master[
        "security_id"
    ] = (
        master[
            "security_id"
        ]
        .astype("string")
        .str.strip()
    )

    master[
        "list_date"
    ] = pd.to_datetime(
        master[
            "list_date"
        ],
        errors="coerce",
    )

    master[
        "delist_date"
    ] = pd.to_datetime(
        master[
            "delist_date"
        ],
        errors="coerce",
    )

    duplicate_id_count = int(
        master[
            "security_id"
        ]
        .duplicated()
        .sum()
    )

    summary = {

        "stock_master_rows":
            int(
                len(
                    master
                )
            ),

        "unique_security_count":
            int(
                master[
                    "security_id"
                ]
                .nunique()
            ),

        "duplicate_security_id_count":
            duplicate_id_count,

        "historically_delisted_count":
            int(
                master[
                    "delist_date"
                ]
                .notna()
                .sum()
            ),
    }

    if duplicate_id_count > 0:

        raise ValueError(
            "stock_master 中 security_id "
            "存在重复，请先检查。"
        )

    return (
        master,
        summary,
    )


# ============================================================
# 7. 检查 Return Panel schema
#
# 不加载 1100 多万行数据，只读取 Parquet schema 和 metadata。
# ============================================================

def inspect_return_panel():

    parquet_file = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    schema_names = (
        parquet_file
        .schema_arrow
        .names
    )

    missing_columns = [

        col

        for col in REQUIRED_PANEL_COLUMNS

        if col not in schema_names
    ]

    if missing_columns:

        raise ValueError(
            "daily_return_panel.parquet "
            f"缺少字段：{missing_columns}"
        )

    row_count = int(
        parquet_file
        .metadata
        .num_rows
    )

    row_group_count = int(
        parquet_file
        .metadata
        .num_row_groups
    )

    schema_df = pd.DataFrame(
        {
            "column":
                schema_names,

            "required_for_day2":
                [
                    col
                    in REQUIRED_PANEL_COLUMNS
                    for col
                    in schema_names
                ],
        }
    )

    schema_df.to_csv(
        OUTPUT_DIR
        / "return_panel_schema.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {

        "return_panel_row_count":
            row_count,

        "return_panel_row_group_count":
            row_group_count,

        "return_panel_column_count":
            len(
                schema_names
            ),

        "return_column_found":
            RETURN_COLUMN
            in schema_names,

        "robust_return_column_found":
            ROBUST_RETURN_COLUMN
            in schema_names,
    }

    return summary


# ============================================================
# 8. 检查 Step 3 收益定义
# ============================================================

def load_return_definition():

    with open(
        RETURN_DEFINITION_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        definition = json.load(
            f
        )

    selected_scale = (
        definition.get(
            "selected_chg_pct_scale"
        )
    )

    selected_adj_mode = (
        definition.get(
            "selected_adj_factor_mode"
        )
    )

    if (
        selected_scale
        !=
        "decimal"
    ):

        raise ValueError(
            "Step 3 的 chgPct 定义不是 decimal，"
            "请检查。"
        )

    if (
        selected_adj_mode
        !=
        "multiply"
    ):

        raise ValueError(
            "Step 3 的 AdjFactor 定义不是 multiply，"
            "请检查。"
        )

    return definition


# ============================================================
# 9. 根据真实交易日生成 Month-End 日期
# ============================================================

def build_month_end_dates(
    trade_dates: pd.DatetimeIndex,
):

    calendar = pd.DataFrame(
        {
            "trade_date":
                trade_dates
        }
    )

    calendar[
        "month"
    ] = (
        calendar[
            "trade_date"
        ]
        .dt.to_period(
            "M"
        )
    )

    month_end = (
        calendar
        .groupby(
            "month",
            as_index=False,
        )
        [
            "trade_date"
        ]
        .max()
        .rename(
            columns={
                "trade_date":
                    "analysis_date"
            }
        )
    )

    month_end[
        "year"
    ] = (
        month_end[
            "analysis_date"
        ]
        .dt.year
    )

    month_end[
        "month_number"
    ] = (
        month_end[
            "analysis_date"
        ]
        .dt.month
    )

    return month_end


# ============================================================
# 10. 为不同 W 构造正式分析日期
#
# 对每个 month-end t：
#
# window =
# [t-W+1, ..., t]
#
# 并保存真实的 window_start。
# ============================================================

def build_analysis_dates_by_window(
    trade_dates,
    month_end_dates,
):

    date_to_index = {

        pd.Timestamp(date):
            i

        for i, date
        in enumerate(
            trade_dates
        )
    }

    rows = []

    for window in WINDOWS:

        required_valid_days = int(
            math.ceil(
                BASELINE_VALID_RATIO
                *
                window
            )
        )

        pairwise_required_days = int(
            math.ceil(
                PAIRWISE_MIN_COMMON_RATIO
                *
                window
            )
        )

        for date in (
            month_end_dates[
                "analysis_date"
            ]
        ):

            date = pd.Timestamp(
                date
            )

            if (
                date
                not in
                date_to_index
            ):

                continue

            end_index = (
                date_to_index[
                    date
                ]
            )

            start_index = (
                end_index
                -
                window
                +
                1
            )

            # 没有完整 W 日窗口
            if start_index < 0:

                continue

            window_start = (
                trade_dates[
                    start_index
                ]
            )

            rows.append(
                {

                    "window":
                        window,

                    "analysis_date":
                        date,

                    "window_start":
                        window_start,

                    "window_end":
                        date,

                    "window_trade_day_count":
                        window,

                    "baseline_valid_ratio":
                        BASELINE_VALID_RATIO,

                    "required_valid_days":
                        required_valid_days,

                    "pairwise_min_common_ratio":
                        PAIRWISE_MIN_COMMON_RATIO,

                    "required_pairwise_common_days":
                        pairwise_required_days,

                    "year":
                        date.year,

                    "month":
                        date.month,

                    "trade_day_end_index":
                        end_index,
                }
            )

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# 11. 生成年度代表日期
#
# 用于后续做完整 correlation matrix、
# PCA、Top edge 检查时降低计算量。
#
# 每年取最后一个正式月末分析日期。
# ============================================================

def build_representative_dates(
    analysis_dates,
):

    rows = []

    for window in WINDOWS:

        subset = (
            analysis_dates[
                analysis_dates[
                    "window"
                ]
                .eq(
                    window
                )
            ]
            .copy()
        )

        if subset.empty:

            continue

        annual = (
            subset
            .sort_values(
                "analysis_date"
            )
            .groupby(
                "year",
                as_index=False,
            )
            .tail(
                1
            )
        )

        rows.append(
            annual
        )

    if not rows:

        return pd.DataFrame()

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 12. 检查 Day 1 Network Feasibility
#
# 冻结 c=0.90 的依据。
# ============================================================

def inspect_day1_feasibility():

    summary = pd.read_csv(
        FEASIBILITY_SUMMARY_PATH
    )

    latest = pd.read_csv(
        FEASIBILITY_LATEST_PATH
    )

    required_summary_cols = {
        "window",
        "min_valid_ratio",
        "p_over_w_median",
        "eligible_stock_count_median",
    }

    if not (
        required_summary_cols
        .issubset(
            summary.columns
        )
    ):

        raise ValueError(
            "network_feasibility_summary.csv "
            "字段与预期不一致。"
        )

    baseline_summary = (
        summary[
            np.isclose(
                summary[
                    "min_valid_ratio"
                ],
                BASELINE_VALID_RATIO,
            )
        ]
        .copy()
    )

    baseline_latest = (
        latest[
            np.isclose(
                latest[
                    "min_valid_ratio"
                ],
                BASELINE_VALID_RATIO,
            )
        ]
        .copy()
    )

    missing_windows = (

        set(
            WINDOWS
        )

        -

        set(
            baseline_summary[
                "window"
            ]
            .astype(int)
        )
    )

    if missing_windows:

        raise ValueError(
            "Day 1 feasibility 中缺少窗口："
            f"{sorted(missing_windows)}"
        )

    baseline_summary.to_csv(
        OUTPUT_DIR
        / "day1_feasibility_baseline_c90.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_latest.to_csv(
        OUTPUT_DIR
        / "day1_feasibility_latest_c90.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        baseline_summary,
        baseline_latest,
    )


# ============================================================
# 13. Day 2 Baseline Design Summary
# ============================================================

def build_design_summary(
    analysis_dates,
    baseline_summary,
    baseline_latest,
):

    rows = []

    for window in WINDOWS:

        dates_w = (
            analysis_dates[
                analysis_dates[
                    "window"
                ]
                .eq(
                    window
                )
            ]
        )

        hist = (
            baseline_summary[
                baseline_summary[
                    "window"
                ]
                .eq(
                    window
                )
            ]
        )

        latest = (
            baseline_latest[
                baseline_latest[
                    "window"
                ]
                .eq(
                    window
                )
            ]
        )

        row = {

            "window":
                window,

            "baseline_valid_ratio":
                BASELINE_VALID_RATIO,

            "required_valid_days":
                int(
                    math.ceil(
                        BASELINE_VALID_RATIO
                        *
                        window
                    )
                ),

            "pairwise_min_common_ratio":
                PAIRWISE_MIN_COMMON_RATIO,

            "required_pairwise_common_days":
                int(
                    math.ceil(
                        PAIRWISE_MIN_COMMON_RATIO
                        *
                        window
                    )
                ),

            "analysis_frequency":
                ANALYSIS_FREQUENCY,

            "analysis_date_count":
                int(
                    len(
                        dates_w
                    )
                ),

            "first_analysis_date":
                (
                    dates_w[
                        "analysis_date"
                    ]
                    .min()
                    if not
                    dates_w.empty
                    else pd.NaT
                ),

            "last_analysis_date":
                (
                    dates_w[
                        "analysis_date"
                    ]
                    .max()
                    if not
                    dates_w.empty
                    else pd.NaT
                ),

            "historical_median_p":
                (
                    float(
                        hist[
                            "eligible_stock_count_median"
                        ]
                        .iloc[0]
                    )
                    if not
                    hist.empty
                    else np.nan
                ),

            "historical_median_p_over_w":
                (
                    float(
                        hist[
                            "p_over_w_median"
                        ]
                        .iloc[0]
                    )
                    if not
                    hist.empty
                    else np.nan
                ),

            "latest_eligible_stock_count":
                (
                    int(
                        latest[
                            "eligible_stock_count"
                        ]
                        .iloc[0]
                    )
                    if (
                        not latest.empty
                        and
                        "eligible_stock_count"
                        in latest.columns
                    )
                    else np.nan
                ),

            "latest_p_over_w":
                (
                    float(
                        latest[
                            "p_over_w"
                        ]
                        .iloc[0]
                    )
                    if (
                        not latest.empty
                        and
                        "p_over_w"
                        in latest.columns
                    )
                    else np.nan
                ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 14. 保存正式 Day 2 Config
# ============================================================

def save_config(
    stock_master_summary,
    return_panel_summary,
    return_definition,
    analysis_dates,
):

    config = {

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        "project":
            "M1_A_share_full_market_association_network",

        "research_day":
            "Day_2",

        "step":
            "Step_1_freeze_baseline_design",

        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        # ----------------------------------------------------
        # Research period
        # ----------------------------------------------------

        "research_period": {

            "start":
                str(
                    START_DATE.date()
                ),

            "end":
                str(
                    END_DATE.date()
                ),

            "trade_day_count":
                EXPECTED_TRADE_DAY_COUNT,
        },

        # ----------------------------------------------------
        # Universe
        # ----------------------------------------------------

        "universe": {

            "market":
                "SSE_SZSE_A_share",

            "target_exchanges":
                TARGET_EXCHANGES,

            "security_key":
                "security_id",

            "definition":
                (
                    "Point-in-Time historical A-share universe "
                    "constructed from stock_master; "
                    "historically delisted securities retained."
                ),

            "historical_security_count":
                stock_master_summary[
                    "unique_security_count"
                ],
        },

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        "return": {

            "baseline_column":
                RETURN_COLUMN,

            "robustness_column":
                ROBUST_RETURN_COLUMN,

            "adjusted_price_definition":
                "close_price * adj_factor",

            "network_return_definition":
                (
                    "log(adjusted_close_t / adjusted_close_t-1), "
                    "defined only for consecutive market dates "
                    "with both observations tradable/open."
                ),

            "vendor_chg_pct_scale":
                return_definition.get(
                    "selected_chg_pct_scale"
                ),

            "adj_factor_mode":
                return_definition.get(
                    "selected_adj_factor_mode"
                ),
        },

        # ----------------------------------------------------
        # Rolling window
        # ----------------------------------------------------

        "rolling_design": {

            "windows":
                WINDOWS,

            "baseline_valid_ratio":
                BASELINE_VALID_RATIO,

            "robust_valid_ratios":
                ROBUST_VALID_RATIOS,

            "pairwise_min_common_ratio":
                PAIRWISE_MIN_COMMON_RATIO,

            "stock_eligibility_definition":
                (
                    "At window end t, stock i must belong to "
                    "the PIT research universe and have at least "
                    "ceil(c*W) non-missing return_network "
                    "observations during the W trading-day window."
                ),

            "pairwise_eligibility_definition":
                (
                    "A stock pair must have at least "
                    "ceil(pairwise_min_common_ratio * W) "
                    "jointly non-missing return observations "
                    "when a pairwise association is estimated."
                ),
        },

        # ----------------------------------------------------
        # Analysis schedule
        # ----------------------------------------------------

        "analysis_schedule": {

            "baseline_frequency":
                ANALYSIS_FREQUENCY,

            "definition":
                (
                    "Use the last actual trading day of each "
                    "calendar month as the baseline structural "
                    "diagnostic date."
                ),

            "total_window_date_rows":
                int(
                    len(
                        analysis_dates
                    )
                ),
        },

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        "random_seed":
            RANDOM_SEED,

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        "input_paths": {

            "stock_master":
                str(
                    STOCK_MASTER_PATH
                ),

            "trade_calendar":
                str(
                    TRADE_CALENDAR_PATH
                ),

            "daily_return_panel":
                str(
                    RETURN_PANEL_PATH
                ),

            "day1_feasibility_summary":
                str(
                    FEASIBILITY_SUMMARY_PATH
                ),
        },

        # ----------------------------------------------------
        # Panel summary
        # ----------------------------------------------------

        "return_panel": {

            "row_count":
                return_panel_summary[
                    "return_panel_row_count"
                ],

            "baseline_return_column_found":
                return_panel_summary[
                    "return_column_found"
                ],

            "robust_return_column_found":
                return_panel_summary[
                    "robust_return_column_found"
                ],
        },

        # ----------------------------------------------------
        # Important methodological notes
        # ----------------------------------------------------

        "methodological_notes": [

            (
                "c=0.90 is a stock-level data availability "
                "threshold, not a correlation threshold "
                "and not an edge sparsification threshold."
            ),

            (
                "W=60,120,252 are all retained at this stage. "
                "Step 1 does not select the final network window."
            ),

            (
                "Month-end sampling is used for structural "
                "diagnostics to control computation. "
                "It is not yet the final portfolio "
                "rebalancing frequency."
            ),

            (
                "Raw Pearson, residual correlation, PCA, "
                "Same/Cross Industry and other Day 2 analyses "
                "should use the same baseline universe rules "
                "and analysis dates wherever possible."
            ),

            (
                "ST and external suspension validation are "
                "not part of the current network estimation "
                "universe. They will be incorporated later "
                "when constructing the tradable Alpha universe."
            ),

            (
                "Historical industry validation must be "
                "completed before formal Same/Cross Industry "
                "analysis."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "day2_analysis_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return config


# ============================================================
# 15. 主程序
# ============================================================

def main():

    print("=" * 76)
    print("M1 Day 2 - Step 1")
    print("Freeze Baseline Sample and Analysis Design")
    print("=" * 76)

    # ========================================================
    # A. 输入检查
    # ========================================================

    print()
    print("[1] 检查 Day 1 输入文件")

    check_required_files()

    # ========================================================
    # B. Trading Calendar
    # ========================================================

    print()
    print("[2] 读取真实交易日历")

    trade_dates = (
        load_trade_dates()
    )

    print(
        f"交易日：{len(trade_dates):,}"
    )

    print(
        f"起始日期：{trade_dates.min().date()}"
    )

    print(
        f"结束日期：{trade_dates.max().date()}"
    )

    # ========================================================
    # C. Stock Master
    # ========================================================

    print()
    print("[3] 检查历史证券主表")

    (
        stock_master,
        stock_master_summary,
    ) = (
        inspect_stock_master()
    )

    print(
        "历史证券主体："
        f"{stock_master_summary['unique_security_count']:,}"
    )

    print(
        "历史退市证券："
        f"{stock_master_summary['historically_delisted_count']:,}"
    )

    # ========================================================
    # D. Return Panel
    # ========================================================

    print()
    print("[4] 检查标准收益面板")

    return_panel_summary = (
        inspect_return_panel()
    )

    print(
        "Return Panel 行数："
        f"{return_panel_summary['return_panel_row_count']:,}"
    )

    # ========================================================
    # E. Return Definition
    # ========================================================

    print()
    print("[5] 检查 Step 3 收益定义")

    return_definition = (
        load_return_definition()
    )

    print(
        "chgPct scale："
        f"{return_definition.get('selected_chg_pct_scale')}"
    )

    print(
        "AdjFactor mode："
        f"{return_definition.get('selected_adj_factor_mode')}"
    )

    # ========================================================
    # F. Month-End
    # ========================================================

    print()
    print("[6] 构造 Month-End 分析日期")

    month_end_dates = (
        build_month_end_dates(
            trade_dates
        )
    )

    month_end_dates.to_csv(
        OUTPUT_DIR
        / "all_month_end_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "全部月末日期："
        f"{len(month_end_dates):,}"
    )

    # ========================================================
    # G. W-specific analysis dates
    # ========================================================

    analysis_dates = (
        build_analysis_dates_by_window(
            trade_dates,
            month_end_dates,
        )
    )

    analysis_dates.to_csv(
        OUTPUT_DIR
        / "analysis_dates_by_window.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # H. Representative dates
    # ========================================================

    representative_dates = (
        build_representative_dates(
            analysis_dates
        )
    )

    representative_dates.to_csv(
        OUTPUT_DIR
        / "representative_year_end_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # I. Day 1 feasibility baseline
    # ========================================================

    print()
    print("[7] 固定 c=0.90 基准可行性结果")

    (
        baseline_feasibility,
        baseline_latest,
    ) = (
        inspect_day1_feasibility()
    )

    # ========================================================
    # J. Design summary
    # ========================================================

    design_summary = (
        build_design_summary(
            analysis_dates,
            baseline_feasibility,
            baseline_latest,
        )
    )

    design_summary.to_csv(
        OUTPUT_DIR
        / "day2_baseline_design_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # K. Freeze config
    # ========================================================

    print()
    print("[8] 写入正式 Day 2 Config")

    save_config(
        stock_master_summary,
        return_panel_summary,
        return_definition,
        analysis_dates,
    )

    # ========================================================
    # L. Console Summary
    # ========================================================

    print()
    print("=" * 76)
    print("Day 2 Step 1 完成")
    print("=" * 76)

    print()
    print("基准研究设计：")

    print(
        f"Return          : {RETURN_COLUMN}"
    )

    print(
        f"Windows         : {WINDOWS}"
    )

    print(
        f"Baseline c      : {BASELINE_VALID_RATIO}"
    )

    print(
        f"Robustness c    : {ROBUST_VALID_RATIOS}"
    )

    print(
        "Pairwise common : "
        f"{PAIRWISE_MIN_COMMON_RATIO}"
    )

    print(
        f"Frequency       : {ANALYSIS_FREQUENCY}"
    )

    print()
    print("各窗口正式分析日期：")

    print(
        design_summary[
            [
                "window",
                "required_valid_days",
                "required_pairwise_common_days",
                "analysis_date_count",
                "first_analysis_date",
                "last_analysis_date",
                "historical_median_p",
                "historical_median_p_over_w",
                "latest_eligible_stock_count",
                "latest_p_over_w",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"输出目录：{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
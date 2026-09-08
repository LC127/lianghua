from __future__ import annotations

from pathlib import Path
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================
# 0. 配置
# ============================================================

ROOT = Path(r"D:\lowfreq")

# ------------------------------------------------------------
# Step 1 输出
# ------------------------------------------------------------

STAGE1_DIR = Path(
    r"D:\output\M1_day1\01_stage1_final"
)

STOCK_MASTER_PATH = (
    STAGE1_DIR
    / "stock_master.parquet"
)

# 直接使用 Step 1 已经确认过的真实交易日历
TRADE_CALENDAR_USED_PATH = (
    STAGE1_DIR
    / "trade_calendar_used.csv"
)


# ------------------------------------------------------------
# daily_temp3
# ------------------------------------------------------------

DAILY_TEMP3_DIR = (
    ROOT
    / "daily_temp3"
)


# ------------------------------------------------------------
# Step 2 输出
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\output\M1_day1\02_stage2_daily_market_qa"
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
# 2. 核心字段别名
# ============================================================

ALIASES = {

    "security_id": [
        "secID",
        "SecID",
        "security_id",
        "securityID",
    ],

    "stock_code": [
        "ticker",
        "stock_code",
        "code",
        "secCode",
        "symbol",
    ],

    "stock_name": [
        "secShortName",
        "stock_name",
        "name",
    ],

    "exchange": [
        "exchangeCD",
        "exchange",
        "market",
    ],

    "trade_date": [
        "tradeDate",
        "trade_date",
        "calendarDate",
        "date",
    ],

    "open": [
        "openPrice",
        "open",
        "open_price",
    ],

    "high": [
        "highestPrice",
        "high",
        "highPrice",
        "highest_price",
    ],

    "low": [
        "lowestPrice",
        "low",
        "lowPrice",
        "lowest_price",
    ],

    "close": [
        "closePrice",
        "close",
        "close_price",
    ],

    "volume": [
        "turnoverVol",
        "volume",
        "vol",
    ],

    "turnover_value": [
        "turnoverValue",
        "amount",
        "turnover_value",
    ],

    "deal_amount": [
        "dealAmount",
        "deal_amount",
    ],

    "vwap": [
        "vwap",
        "VWAP",
    ],

    "adj_factor": [
        "accumAdjFactor",
        "adjFactor",
        "adj_factor",
        "accum_adj_factor",
    ],

    "chg_pct": [
        "chgPct",
        "changePct",
        "pct_chg",
    ],

    "is_open": [
        "isOpen",
        "is_open",
        "isTrade",
    ],

    "market_value": [
        "marketValue",
        "market_value",
        "totalMarketValue",
    ],

    "neg_market_value": [
        "negMarketValue",
        "neg_market_value",
        "floatMarketValue",
    ],

    "turnover_rate": [
        "turnoverRate",
        "turnover_rate",
    ],

    "pe": [
        "PE",
        "pe",
    ],

    "pe1": [
        "PE1",
        "pe1",
    ],

    "pb": [
        "PB",
        "pb",
    ],

    "industry1": [
        "industryID1",
        "industry_id1",
    ],

    "industry2": [
        "industryID2",
        "industry_id2",
    ],
}


SUPPORTED_SUFFIX = {
    ".csv",
    ".parquet",
    ".pq",
    ".feather",
}


# ============================================================
# 3. 工具函数
# ============================================================

def normalize_colname(x: str) -> str:

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(x).lower(),
    )


def find_column(
    columns,
    aliases,
):

    columns = list(columns)

    # exact
    for alias in aliases:

        if alias in columns:

            return alias

    normalized = {
        normalize_colname(c): c
        for c in columns
    }

    for alias in aliases:

        key = normalize_colname(
            alias
        )

        if key in normalized:

            return normalized[key]

    return None


def infer_mapping(
    df: pd.DataFrame,
):

    mapping = {}

    for key, aliases in (
        ALIASES.items()
    ):

        mapping[key] = (
            find_column(
                df.columns,
                aliases,
            )
        )

    return mapping


def read_data_file(
    path: Path,
) -> pd.DataFrame:

    suffix = (
        path.suffix.lower()
    )

    if suffix == ".csv":

        try:

            return pd.read_csv(
                path,
                encoding="utf-8-sig",
                low_memory=False,
            )

        except UnicodeDecodeError:

            return pd.read_csv(
                path,
                encoding="gbk",
                low_memory=False,
            )

    if suffix in {
        ".parquet",
        ".pq",
    }:

        return pd.read_parquet(
            path
        )

    if suffix == ".feather":

        return pd.read_feather(
            path
        )

    raise ValueError(
        f"不支持文件格式：{path}"
    )


def numeric(
    series: pd.Series,
):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def normalize_is_open(
    series: pd.Series,
):

    num = pd.to_numeric(
        series,
        errors="coerce",
    )

    text = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    result = (
        num.eq(1)
        |
        text.isin(
            {
                "1",
                "TRUE",
                "T",
                "YES",
                "Y",
                "OPEN",
            }
        )
    )

    return (
        result.fillna(False)
    )


# ============================================================
# 4. 从文件名提取日期
# ============================================================

def extract_date_from_filename(
    path: Path,
) -> pd.Timestamp | None:

    name = path.stem

    patterns = [

        # 2026-09-03
        r"(20\d{2})[-_](\d{2})[-_](\d{2})",

        # 20260903
        r"(20\d{2})(\d{2})(\d{2})",
    ]

    for pattern in patterns:

        m = re.search(
            pattern,
            name,
        )

        if m:

            try:

                return pd.Timestamp(
                    year=int(m.group(1)),
                    month=int(m.group(2)),
                    day=int(m.group(3)),
                )

            except ValueError:

                return None

    return None


# ============================================================
# 5. 建立 daily_temp3 文件地图
# ============================================================

def build_daily_file_map(
    trade_dates: pd.DatetimeIndex,
):

    if not DAILY_TEMP3_DIR.exists():

        raise FileNotFoundError(
            f"目录不存在：{DAILY_TEMP3_DIR}"
        )

    file_map = defaultdict(list)

    ignored_files = []

    for path in DAILY_TEMP3_DIR.rglob("*"):

        if (
            not path.is_file()
            or
            path.suffix.lower()
            not in SUPPORTED_SUFFIX
        ):

            continue

        date = (
            extract_date_from_filename(
                path
            )
        )

        if date is None:

            ignored_files.append(
                str(path)
            )

            continue

        if date in trade_dates:

            file_map[
                date
            ].append(
                path
            )

    # --------------------------------------------------------
    # 检查缺失日期
    # --------------------------------------------------------

    missing_dates = [

        date

        for date in trade_dates

        if date not in file_map
    ]

    # --------------------------------------------------------
    # 检查同一天多个文件
    # --------------------------------------------------------

    duplicated_dates = {

        date: files

        for date, files
        in file_map.items()

        if len(files) > 1
    }

    file_inventory = []

    for date in trade_dates:

        files = file_map.get(
            date,
            [],
        )

        file_inventory.append(
            {
                "trade_date":
                    date,

                "file_count":
                    len(files),

                "file_path":
                    (
                        str(files[0])
                        if len(files) == 1
                        else
                        "|".join(
                            str(x)
                            for x in files
                        )
                    ),
            }
        )

    inventory_df = pd.DataFrame(
        file_inventory
    )

    inventory_df.to_csv(
        OUTPUT_DIR
        / "daily_file_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if ignored_files:

        pd.DataFrame(
            {
                "file_path":
                    ignored_files
            }
        ).to_csv(
            OUTPUT_DIR
            / "daily_files_without_detectable_date.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if missing_dates:

        pd.DataFrame(
            {
                "trade_date":
                    missing_dates
            }
        ).to_csv(
            OUTPUT_DIR
            / "missing_daily_files.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if duplicated_dates:

        rows = []

        for date, files in (
            duplicated_dates.items()
        ):

            for file in files:

                rows.append(
                    {
                        "trade_date":
                            date,

                        "file_path":
                            str(file),
                    }
                )

        pd.DataFrame(
            rows
        ).to_csv(
            OUTPUT_DIR
            / "duplicated_daily_files.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print()
    print("=" * 72)
    print("daily_temp3 文件检查")
    print("=" * 72)

    print(
        f"应有交易日："
        f"{len(trade_dates):,}"
    )

    print(
        f"成功映射日期："
        f"{len(file_map):,}"
    )

    print(
        f"缺失交易日文件："
        f"{len(missing_dates):,}"
    )

    print(
        f"重复日期文件："
        f"{len(duplicated_dates):,}"
    )

    if missing_dates:

        raise RuntimeError(
            "daily_temp3 存在缺失交易日文件，"
            "请先检查 missing_daily_files.csv。"
        )

    if duplicated_dates:

        raise RuntimeError(
            "daily_temp3 存在同一天多个文件，"
            "请先检查 duplicated_daily_files.csv。"
        )

    return {
        date: files[0]
        for date, files in file_map.items()
    }


# ============================================================
# 6. 加载 Step 1 Stock Master
# ============================================================

def load_stock_master():

    if not STOCK_MASTER_PATH.exists():

        raise FileNotFoundError(
            f"找不到：{STOCK_MASTER_PATH}"
        )

    master = pd.read_parquet(
        STOCK_MASTER_PATH
    )

    required = [
        "security_id",
        "list_date",
        "delist_date",
        "exchange",
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

    return master


# ============================================================
# 7. 加载 Step 1 真实交易日历
# ============================================================

def load_trade_dates():

    if not (
        TRADE_CALENDAR_USED_PATH.exists()
    ):

        raise FileNotFoundError(
            f"找不到："
            f"{TRADE_CALENDAR_USED_PATH}"
        )

    calendar = pd.read_csv(
        TRADE_CALENDAR_USED_PATH
    )

    if (
        "trade_date"
        not in calendar.columns
    ):

        raise ValueError(
            "trade_calendar_used.csv "
            "缺少 trade_date。"
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
            f"交易日数量={len(dates)}, "
            f"预期={EXPECTED_TRADE_DAY_COUNT}"
        )

    return pd.DatetimeIndex(
        dates
    )


# ============================================================
# 8. 根据 Stock Master 构造某日理论 Universe
#
# 不需要加载 1100 多万行 raw_universe parquet，
# 直接利用 5553 行 stock_master 即可。
# ============================================================

def expected_universe_on_date(
    stock_master: pd.DataFrame,
    date: pd.Timestamp,
):

    mask = (
        (
            stock_master[
                "list_date"
            ]
            <= date
        )
        &
        (
            stock_master[
                "delist_date"
            ].isna()
            |
            (
                stock_master[
                    "delist_date"
                ]
                > date
            )
        )
    )

    expected = (
        stock_master.loc[
            mask,
            [
                "security_id",
                "stock_code",
                "stock_name",
                "exchange",
            ],
        ]
        .copy()
    )

    return expected


# ============================================================
# 9. daily_temp3 标准化
# ============================================================

def standardize_daily(
    df: pd.DataFrame,
    file_date: pd.Timestamp,
):

    mapping = (
        infer_mapping(
            df
        )
    )

    if (
        mapping[
            "security_id"
        ]
        is None
    ):

        raise ValueError(
            "daily_temp3 中未识别 secID。"
        )

    result = pd.DataFrame(
        index=df.index
    )

    result[
        "security_id"
    ] = (
        df[
            mapping[
                "security_id"
            ]
        ]
        .astype("string")
        .str.strip()
    )

    if (
        mapping[
            "stock_code"
        ]
        is not None
    ):

        result[
            "stock_code"
        ] = (
            df[
                mapping[
                    "stock_code"
                ]
            ]
            .astype("string")
            .str.strip()
        )

    else:

        result[
            "stock_code"
        ] = pd.NA

    if (
        mapping[
            "stock_name"
        ]
        is not None
    ):

        result[
            "stock_name"
        ] = (
            df[
                mapping[
                    "stock_name"
                ]
            ]
            .astype("string")
            .str.strip()
        )

    else:

        result[
            "stock_name"
        ] = pd.NA

    result[
        "trade_date"
    ] = file_date

    # --------------------------------------------------------
    # 核心数值字段
    # --------------------------------------------------------

    numeric_fields = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover_value",
        "deal_amount",
        "vwap",
        "adj_factor",
        "chg_pct",
        "market_value",
        "neg_market_value",
        "turnover_rate",
        "pe",
        "pe1",
        "pb",
    ]

    for field in numeric_fields:

        col = mapping[
            field
        ]

        if col is None:

            result[
                field
            ] = np.nan

        else:

            result[
                field
            ] = numeric(
                df[
                    col
                ]
            )

    # --------------------------------------------------------
    # isOpen
    # --------------------------------------------------------

    if (
        mapping[
            "is_open"
        ]
        is None
    ):

        result[
            "is_open"
        ] = pd.NA

    else:

        result[
            "is_open"
        ] = (
            normalize_is_open(
                df[
                    mapping[
                        "is_open"
                    ]
                ]
            )
        )

    # --------------------------------------------------------
    # Industry
    # --------------------------------------------------------

    for field in [
        "industry1",
        "industry2",
    ]:

        col = mapping[
            field
        ]

        if col is None:

            result[
                field
            ] = pd.NA

        else:

            result[
                field
            ] = (
                df[
                    col
                ]
                .astype("string")
            )

    return (
        result,
        mapping,
    )


# ============================================================
# 10. Universe Coverage QA
# ============================================================

def universe_coverage_check(
    date,
    expected,
    daily,
):

    expected_ids = set(
        expected[
            "security_id"
        ].dropna()
    )

    daily_ids = set(
        daily[
            "security_id"
        ].dropna()
    )

    matched = (
        expected_ids
        &
        daily_ids
    )

    raw_only = (
        expected_ids
        -
        daily_ids
    )

    daily_only = (
        daily_ids
        -
        expected_ids
    )

    expected_count = (
        len(
            expected_ids
        )
    )

    matched_count = (
        len(
            matched
        )
    )

    coverage_ratio = (
        matched_count
        /
        expected_count
        if expected_count > 0
        else np.nan
    )

    summary = {

        "trade_date":
            date,

        "expected_count":
            expected_count,

        "daily_row_count":
            len(daily),

        "daily_unique_security_count":
            len(daily_ids),

        "matched_count":
            matched_count,

        "raw_only_count":
            len(raw_only),

        "daily_only_count":
            len(daily_only),

        "coverage_ratio":
            coverage_ratio,
    }

    mismatch_rows = []

    # --------------------------------------------------------
    # 理论股票池有，但 daily_temp3 没有
    # --------------------------------------------------------

    if raw_only:

        lookup = (
            expected[
                expected[
                    "security_id"
                ].isin(
                    raw_only
                )
            ]
        )

        for row in (
            lookup.itertuples(
                index=False
            )
        ):

            mismatch_rows.append(
                {

                    "trade_date":
                        date,

                    "mismatch_type":
                        "RAW_ONLY",

                    "security_id":
                        row.security_id,

                    "stock_code":
                        row.stock_code,

                    "stock_name":
                        row.stock_name,

                    "exchange":
                        row.exchange,
                }
            )

    # --------------------------------------------------------
    # daily_temp3 有，但理论股票池没有
    # --------------------------------------------------------

    if daily_only:

        lookup = (
            daily[
                daily[
                    "security_id"
                ].isin(
                    daily_only
                )
            ]
            .drop_duplicates(
                subset=[
                    "security_id"
                ]
            )
        )

        for row in (
            lookup.itertuples(
                index=False
            )
        ):

            mismatch_rows.append(
                {

                    "trade_date":
                        date,

                    "mismatch_type":
                        "DAILY_ONLY",

                    "security_id":
                        row.security_id,

                    "stock_code":
                        row.stock_code,

                    "stock_name":
                        row.stock_name,

                    "exchange":
                        pd.NA,
                }
            )

    return (
        summary,
        mismatch_rows,
    )


# ============================================================
# 11. 核心行情质量检查
# ============================================================

def core_market_qa(
    daily: pd.DataFrame,
    date: pd.Timestamp,
):

    anomaly_rows = []

    # --------------------------------------------------------
    # 11.1 Duplicate secID
    # --------------------------------------------------------

    duplicate_mask = (
        daily[
            "security_id"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_count = (
        daily.loc[
            duplicate_mask,
            "security_id",
        ]
        .nunique()
    )

    if duplicate_mask.any():

        dup = (
            daily[
                duplicate_mask
            ]
            .copy()
        )

        for row in (
            dup.itertuples(
                index=False
            )
        ):

            anomaly_rows.append(
                {

                    "trade_date":
                        date,

                    "security_id":
                        row.security_id,

                    "stock_code":
                        row.stock_code,

                    "stock_name":
                        row.stock_name,

                    "issue_type":
                        "DUPLICATE_SECURITY_RECORD",

                    "severity":
                        "HIGH",

                    "note":
                        "同一交易日 secID 出现多条记录。",
                }
            )

    # --------------------------------------------------------
    # 11.2 Missing secID
    # --------------------------------------------------------

    missing_secid = (
        daily[
            "security_id"
        ].isna()
        |
        daily[
            "security_id"
        ].eq("")
    )

    # --------------------------------------------------------
    # 11.3 Price <= 0
    # --------------------------------------------------------

    invalid_price = (

        (
            daily[
                "open"
            ].notna()
            &
            (
                daily[
                    "open"
                ]
                <= 0
            )
        )

        |

        (
            daily[
                "high"
            ].notna()
            &
            (
                daily[
                    "high"
                ]
                <= 0
            )
        )

        |

        (
            daily[
                "low"
            ].notna()
            &
            (
                daily[
                    "low"
                ]
                <= 0
            )
        )

        |

        (
            daily[
                "close"
            ].notna()
            &
            (
                daily[
                    "close"
                ]
                <= 0
            )
        )
    )

    # --------------------------------------------------------
    # 11.4 OHLC 逻辑
    # --------------------------------------------------------

    required_ohlc = (
        daily[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    invalid_ohlc = (

        required_ohlc

        &
        (

            (
                daily[
                    "high"
                ]
                <
                daily[
                    [
                        "open",
                        "close",
                    ]
                ]
                .max(
                    axis=1
                )
            )

            |

            (
                daily[
                    "low"
                ]
                >
                daily[
                    [
                        "open",
                        "close",
                    ]
                ]
                .min(
                    axis=1
                )
            )

            |

            (
                daily[
                    "high"
                ]
                <
                daily[
                    "low"
                ]
            )
        )
    )

    # --------------------------------------------------------
    # 11.5 Volume / Value
    # --------------------------------------------------------

    negative_volume = (
        daily[
            "volume"
        ].notna()
        &
        (
            daily[
                "volume"
            ]
            < 0
        )
    )

    negative_turnover_value = (
        daily[
            "turnover_value"
        ].notna()
        &
        (
            daily[
                "turnover_value"
            ]
            < 0
        )
    )

    negative_deal_amount = (
        daily[
            "deal_amount"
        ].notna()
        &
        (
            daily[
                "deal_amount"
            ]
            < 0
        )
    )

    # --------------------------------------------------------
    # 11.6 AdjFactor
    # --------------------------------------------------------

    invalid_adj_factor = (
        daily[
            "adj_factor"
        ].isna()
        |
        (
            daily[
                "adj_factor"
            ]
            <= 0
        )
    )

    # --------------------------------------------------------
    # 11.7 isOpen consistency
    #
    # 停牌状态但成交量/成交额 > 0：
    # 这是较强的矛盾信号。
    # --------------------------------------------------------

    is_open_available = (
        daily[
            "is_open"
        ].notna()
    )

    closed_but_traded = (

        is_open_available

        &
        (
            daily[
                "is_open"
            ]
            == False
        )

        &
        (

            (
                daily[
                    "volume"
                ].fillna(0)
                > 0
            )

            |

            (
                daily[
                    "turnover_value"
                ].fillna(0)
                > 0
            )
        )
    )

    # --------------------------------------------------------
    # 开市但零成交：
    # 不直接判错误，只做 REVIEW 统计。
    # --------------------------------------------------------

    open_but_zero_trade = (

        is_open_available

        &
        (
            daily[
                "is_open"
            ]
            == True
        )

        &
        (
            daily[
                "volume"
            ].fillna(0)
            == 0
        )

        &
        (
            daily[
                "turnover_value"
            ].fillna(0)
            == 0
        )
    )

    # --------------------------------------------------------
    # 11.8 VWAP
    # --------------------------------------------------------

    invalid_vwap = (

        (
            daily[
                "volume"
            ].fillna(0)
            > 0
        )

        &
        (
            daily[
                "vwap"
            ].isna()
            |
            (
                daily[
                    "vwap"
                ]
                <= 0
            )
        )
    )

    # --------------------------------------------------------
    # 11.9 Market Value
    # --------------------------------------------------------

    invalid_market_value = (
        daily[
            "market_value"
        ].notna()
        &
        (
            daily[
                "market_value"
            ]
            <= 0
        )
    )

    invalid_neg_market_value = (
        daily[
            "neg_market_value"
        ].notna()
        &
        (
            daily[
                "neg_market_value"
            ]
            < 0
        )
    )

    invalid_turnover_rate = (
        daily[
            "turnover_rate"
        ].notna()
        &
        (
            daily[
                "turnover_rate"
            ]
            < 0
        )
    )

    # --------------------------------------------------------
    # 将重要异常写入 row-level anomaly
    # --------------------------------------------------------

    anomaly_definitions = [

        (
            invalid_price,
            "NONPOSITIVE_PRICE",
            "HIGH",
            "OHLC 中存在非正价格。",
        ),

        (
            invalid_ohlc,
            "INVALID_OHLC_RELATION",
            "HIGH",
            "High/Low 与 Open/Close 逻辑不一致。",
        ),

        (
            negative_volume,
            "NEGATIVE_VOLUME",
            "HIGH",
            "turnoverVol < 0。",
        ),

        (
            negative_turnover_value,
            "NEGATIVE_TURNOVER_VALUE",
            "HIGH",
            "turnoverValue < 0。",
        ),

        (
            negative_deal_amount,
            "NEGATIVE_DEAL_AMOUNT",
            "HIGH",
            "dealAmount < 0。",
        ),

        (
            invalid_adj_factor,
            "INVALID_ADJ_FACTOR",
            "HIGH",
            "复权因子缺失或 <= 0。",
        ),

        (
            closed_but_traded,
            "CLOSED_BUT_TRADED",
            "HIGH",
            "isOpen=0 但仍存在成交量或成交额。",
        ),

        (
            invalid_vwap,
            "INVALID_VWAP",
            "REVIEW",
            "有成交但 VWAP 缺失或 <= 0。",
        ),

        (
            invalid_market_value,
            "INVALID_MARKET_VALUE",
            "REVIEW",
            "marketValue <= 0。",
        ),

        (
            invalid_neg_market_value,
            "INVALID_NEG_MARKET_VALUE",
            "REVIEW",
            "negMarketValue < 0。",
        ),

        (
            invalid_turnover_rate,
            "INVALID_TURNOVER_RATE",
            "REVIEW",
            "turnoverRate < 0。",
        ),
    ]

    for (
        mask,
        issue_type,
        severity,
        note,
    ) in anomaly_definitions:

        if not mask.any():

            continue

        subset = (
            daily.loc[
                mask,
                [
                    "security_id",
                    "stock_code",
                    "stock_name",
                ],
            ]
        )

        for row in (
            subset.itertuples(
                index=False
            )
        ):

            anomaly_rows.append(
                {

                    "trade_date":
                        date,

                    "security_id":
                        row.security_id,

                    "stock_code":
                        row.stock_code,

                    "stock_name":
                        row.stock_name,

                    "issue_type":
                        issue_type,

                    "severity":
                        severity,

                    "note":
                        note,
                }
            )

    # --------------------------------------------------------
    # 每日 QA summary
    # --------------------------------------------------------

    summary = {

        "trade_date":
            date,

        "row_count":
            len(daily),

        "unique_security_count":
            daily[
                "security_id"
            ].nunique(),

        "duplicate_security_count":
            int(
                duplicate_count
            ),

        "missing_security_id_count":
            int(
                missing_secid.sum()
            ),

        "missing_open_count":
            int(
                daily[
                    "open"
                ].isna().sum()
            ),

        "missing_high_count":
            int(
                daily[
                    "high"
                ].isna().sum()
            ),

        "missing_low_count":
            int(
                daily[
                    "low"
                ].isna().sum()
            ),

        "missing_close_count":
            int(
                daily[
                    "close"
                ].isna().sum()
            ),

        "missing_volume_count":
            int(
                daily[
                    "volume"
                ].isna().sum()
            ),

        "missing_turnover_value_count":
            int(
                daily[
                    "turnover_value"
                ].isna().sum()
            ),

        "missing_adj_factor_count":
            int(
                daily[
                    "adj_factor"
                ].isna().sum()
            ),

        "invalid_price_count":
            int(
                invalid_price.sum()
            ),

        "invalid_ohlc_count":
            int(
                invalid_ohlc.sum()
            ),

        "negative_volume_count":
            int(
                negative_volume.sum()
            ),

        "negative_turnover_value_count":
            int(
                negative_turnover_value.sum()
            ),

        "negative_deal_amount_count":
            int(
                negative_deal_amount.sum()
            ),

        "invalid_adj_factor_count":
            int(
                invalid_adj_factor.sum()
            ),

        "closed_but_traded_count":
            int(
                closed_but_traded.sum()
            ),

        "open_but_zero_trade_count":
            int(
                open_but_zero_trade.sum()
            ),

        "invalid_vwap_count":
            int(
                invalid_vwap.sum()
            ),

        "invalid_market_value_count":
            int(
                invalid_market_value.sum()
            ),

        "invalid_neg_market_value_count":
            int(
                invalid_neg_market_value.sum()
            ),

        "invalid_turnover_rate_count":
            int(
                invalid_turnover_rate.sum()
            ),

        "open_stock_count":
            int(
                (
                    daily[
                        "is_open"
                    ]
                    == True
                ).sum()
            ),

        "closed_stock_count":
            int(
                (
                    daily[
                        "is_open"
                    ]
                    == False
                ).sum()
            ),
    }

    return (
        summary,
        anomaly_rows,
    )


# ============================================================
# 12. 字段覆盖统计
# ============================================================

def update_field_coverage(
    mapping,
    field_stats,
):

    for field in (
        ALIASES.keys()
    ):

        if (
            mapping[
                field
            ]
            is None
        ):

            field_stats[
                field
            ][
                "absent_day_count"
            ] += 1

        else:

            field_stats[
                field
            ][
                "present_day_count"
            ] += 1


# ============================================================
# 13. 汇总总体质量
# ============================================================

def build_overall_summary(
    coverage_df,
    qa_df,
    anomaly_df,
):

    total_expected = (
        coverage_df[
            "expected_count"
        ].sum()
    )

    total_matched = (
        coverage_df[
            "matched_count"
        ].sum()
    )

    weighted_coverage = (
        total_matched
        /
        total_expected
        if total_expected > 0
        else np.nan
    )

    high_count = (
        anomaly_df[
            "severity"
        ].eq(
            "HIGH"
        ).sum()
        if not anomaly_df.empty
        else 0
    )

    review_count = (
        anomaly_df[
            "severity"
        ].eq(
            "REVIEW"
        ).sum()
        if not anomaly_df.empty
        else 0
    )

    summary = pd.DataFrame(
        {

            "metric": [

                "trade_day_count",

                "total_expected_stock_day_records",

                "total_matched_stock_day_records",

                "weighted_history_coverage_ratio",

                "min_daily_coverage_ratio",

                "mean_daily_coverage_ratio",

                "days_with_raw_only",

                "days_with_daily_only",

                "total_raw_only_records",

                "total_daily_only_records",

                "days_with_duplicate_security",

                "total_duplicate_security_count",

                "total_invalid_price_count",

                "total_invalid_ohlc_count",

                "total_negative_volume_count",

                "total_negative_turnover_value_count",

                "total_invalid_adj_factor_count",

                "total_closed_but_traded_count",

                "total_open_but_zero_trade_count",

                "high_anomaly_rows",

                "review_anomaly_rows",
            ],

            "value": [

                len(
                    coverage_df
                ),

                int(
                    total_expected
                ),

                int(
                    total_matched
                ),

                weighted_coverage,

                coverage_df[
                    "coverage_ratio"
                ].min(),

                coverage_df[
                    "coverage_ratio"
                ].mean(),

                int(
                    (
                        coverage_df[
                            "raw_only_count"
                        ]
                        > 0
                    ).sum()
                ),

                int(
                    (
                        coverage_df[
                            "daily_only_count"
                        ]
                        > 0
                    ).sum()
                ),

                int(
                    coverage_df[
                        "raw_only_count"
                    ].sum()
                ),

                int(
                    coverage_df[
                        "daily_only_count"
                    ].sum()
                ),

                int(
                    (
                        qa_df[
                            "duplicate_security_count"
                        ]
                        > 0
                    ).sum()
                ),

                int(
                    qa_df[
                        "duplicate_security_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "invalid_price_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "invalid_ohlc_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "negative_volume_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "negative_turnover_value_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "invalid_adj_factor_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "closed_but_traded_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "open_but_zero_trade_count"
                    ].sum()
                ),

                int(
                    high_count
                ),

                int(
                    review_count
                ),
            ],
        }
    )

    return summary


# ============================================================
# 14. 主程序
# ============================================================

def main():

    print("=" * 72)
    print("M1 Stage 2")
    print("Daily Market Coverage + Core Market Data QA")
    print("=" * 72)

    # --------------------------------------------------------
    # A. Step 1 stock master
    # --------------------------------------------------------

    stock_master = (
        load_stock_master()
    )

    print(
        f"Stock Master："
        f"{stock_master['security_id'].nunique():,} "
        f"个历史证券主体"
    )

    # --------------------------------------------------------
    # B. 交易日
    # --------------------------------------------------------

    trade_dates = (
        load_trade_dates()
    )

    print(
        f"研究交易日："
        f"{len(trade_dates):,}"
    )

    # --------------------------------------------------------
    # C. daily_temp3 文件映射
    # --------------------------------------------------------

    file_map = (
        build_daily_file_map(
            trade_dates
        )
    )

    # --------------------------------------------------------
    # D. 初始化结果
    # --------------------------------------------------------

    coverage_results = []

    qa_results = []

    mismatch_results = []

    anomaly_results = []

    processing_log = []

    field_stats = defaultdict(
        lambda: {
            "present_day_count": 0,
            "absent_day_count": 0,
        }
    )

    # ========================================================
    # E. 按交易日逐文件处理
    #
    # 不将 2015-2026 所有日行情一次性读入内存。
    # ========================================================

    for i, date in enumerate(
        trade_dates,
        start=1,
    ):

        file = (
            file_map[
                date
            ]
        )

        if (
            i == 1
            or
            i % 100 == 0
            or
            i == len(
                trade_dates
            )
        ):

            print(
                f"[{i:4d}/"
                f"{len(trade_dates)}] "
                f"{date.date()} "
                f"{file.name}"
            )

        try:

            raw_daily = (
                read_data_file(
                    file
                )
            )

            daily, mapping = (
                standardize_daily(
                    raw_daily,
                    date,
                )
            )

            update_field_coverage(
                mapping,
                field_stats,
            )

            # ------------------------------------------------
            # 理论 PIT Universe
            # ------------------------------------------------

            expected = (
                expected_universe_on_date(
                    stock_master,
                    date,
                )
            )

            # ------------------------------------------------
            # Universe coverage
            # ------------------------------------------------

            (
                coverage_summary,
                mismatch_rows,
            ) = (
                universe_coverage_check(
                    date,
                    expected,
                    daily,
                )
            )

            coverage_results.append(
                coverage_summary
            )

            mismatch_results.extend(
                mismatch_rows
            )

            # ------------------------------------------------
            # Core QA
            # ------------------------------------------------

            (
                qa_summary,
                anomaly_rows,
            ) = (
                core_market_qa(
                    daily,
                    date,
                )
            )

            qa_results.append(
                qa_summary
            )

            anomaly_results.extend(
                anomaly_rows
            )

            processing_log.append(
                {
                    "trade_date":
                        date,

                    "file_path":
                        str(file),

                    "status":
                        "SUCCESS",

                    "raw_row_count":
                        len(raw_daily),

                    "error":
                        "",
                }
            )

        except Exception as e:

            processing_log.append(
                {
                    "trade_date":
                        date,

                    "file_path":
                        str(file),

                    "status":
                        "FAILED",

                    "raw_row_count":
                        np.nan,

                    "error":
                        str(e),
                }
            )

            print(
                f"[ERROR] "
                f"{date.date()}: "
                f"{e}"
            )

    # ========================================================
    # 15. 整理结果
    # ========================================================

    coverage_df = pd.DataFrame(
        coverage_results
    )

    qa_df = pd.DataFrame(
        qa_results
    )

    mismatch_df = pd.DataFrame(
        mismatch_results
    )

    anomaly_df = pd.DataFrame(
        anomaly_results
    )

    log_df = pd.DataFrame(
        processing_log
    )

    # --------------------------------------------------------
    # 字段覆盖
    # --------------------------------------------------------

    field_rows = []

    for field, stats in (
        field_stats.items()
    ):

        total = (
            stats[
                "present_day_count"
            ]
            +
            stats[
                "absent_day_count"
            ]
        )

        field_rows.append(
            {

                "field":
                    field,

                "present_day_count":
                    stats[
                        "present_day_count"
                    ],

                "absent_day_count":
                    stats[
                        "absent_day_count"
                    ],

                "present_ratio":
                    (
                        stats[
                            "present_day_count"
                        ]
                        /
                        total
                        if total > 0
                        else np.nan
                    ),
            }
        )

    field_coverage_df = pd.DataFrame(
        field_rows
    )

    # ========================================================
    # 16. 总体 Summary
    # ========================================================

    if (
        not coverage_df.empty
        and
        not qa_df.empty
    ):

        overall_summary = (
            build_overall_summary(
                coverage_df,
                qa_df,
                anomaly_df,
            )
        )

    else:

        overall_summary = (
            pd.DataFrame()
        )

    # ========================================================
    # 17. 输出
    # ========================================================

    coverage_df.to_csv(
        OUTPUT_DIR
        / "daily_market_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qa_df.to_csv(
        OUTPUT_DIR
        / "daily_core_qa_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    field_coverage_df.to_csv(
        OUTPUT_DIR
        / "field_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_summary.to_csv(
        OUTPUT_DIR
        / "core_data_quality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log_df.to_csv(
        OUTPUT_DIR
        / "file_processing_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not mismatch_df.empty:

        mismatch_df.to_parquet(
            OUTPUT_DIR
            / "universe_mismatch_detail.parquet",
            index=False,
        )

    else:

        pd.DataFrame(
            columns=[
                "trade_date",
                "mismatch_type",
                "security_id",
                "stock_code",
                "stock_name",
                "exchange",
            ]
        ).to_parquet(
            OUTPUT_DIR
            / "universe_mismatch_detail.parquet",
            index=False,
        )

    if not anomaly_df.empty:

        anomaly_df.to_parquet(
            OUTPUT_DIR
            / "core_data_anomalies.parquet",
            index=False,
        )

    else:

        pd.DataFrame(
            columns=[
                "trade_date",
                "security_id",
                "stock_code",
                "stock_name",
                "issue_type",
                "severity",
                "note",
            ]
        ).to_parquet(
            OUTPUT_DIR
            / "core_data_anomalies.parquet",
            index=False,
        )

    # ========================================================
    # 18. Metadata
    # ========================================================

    failed_days = (
        log_df[
            "status"
        ]
        .eq(
            "FAILED"
        )
        .sum()
    )

    metadata = {

        "research_start":
            str(
                START_DATE.date()
            ),

        "research_end":
            str(
                END_DATE.date()
            ),

        "trade_day_count":
            int(
                len(
                    trade_dates
                )
            ),

        "stock_master_path":
            str(
                STOCK_MASTER_PATH
            ),

        "daily_temp3_dir":
            str(
                DAILY_TEMP3_DIR
            ),

        "processed_success_days":
            int(
                log_df[
                    "status"
                ]
                .eq(
                    "SUCCESS"
                )
                .sum()
            ),

        "processed_failed_days":
            int(
                failed_days
            ),

        "notes": [

            (
                "PIT expected universe "
                "由 Step 1 stock_master "
                "利用 list_date <= t < delist_date "
                "逐日重新构造。"
            ),

            (
                "Universe coverage "
                "使用稳定 secID 比较，"
                "不依赖 ticker。"
            ),

            (
                "停牌股票即使没有成交，"
                "只要 daily_temp3 中存在该证券记录，"
                "仍计入行情覆盖。"
            ),

            (
                "本阶段仅检查行情数据质量；"
                "收益率重新构造与 chgPct 验证"
                "留到后续 Return Validation Step。"
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "stage2_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # 19. 控制台结果
    # ========================================================

    print()
    print("=" * 72)
    print("Stage 2 完成")
    print("=" * 72)

    print(
        f"成功处理交易日："
        f"{(
            log_df['status'] == 'SUCCESS'
        ).sum():,}"
    )

    print(
        f"失败交易日："
        f"{failed_days:,}"
    )

    if not coverage_df.empty:

        print(
            f"历史加权覆盖率："
            f"{coverage_df['matched_count'].sum()
               / coverage_df['expected_count'].sum():.8f}"
        )

        print(
            f"最低单日覆盖率："
            f"{coverage_df['coverage_ratio'].min():.8f}"
        )

        print(
            f"存在 RAW_ONLY 的交易日："
            f"{(
                coverage_df['raw_only_count'] > 0
            ).sum():,}"
        )

        print(
            f"存在 DAILY_ONLY 的交易日："
            f"{(
                coverage_df['daily_only_count'] > 0
            ).sum():,}"
        )

    if not anomaly_df.empty:

        print()
        print(
            "异常类型统计："
        )

        print(
            anomaly_df[
                "issue_type"
            ]
            .value_counts()
            .to_string()
        )

    print()
    print(
        f"输出目录：{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
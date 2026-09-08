from __future__ import annotations

from pathlib import Path
import json
import re
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# 0. 路径配置
# ============================================================

# ------------------------------------------------------------
# Step 1 最终输出目录
# ------------------------------------------------------------

STAGE1_DIR = Path(
    r"D:\output\M1_day1\01_stage1_final"
)

STOCK_MASTER_PATH = (
    STAGE1_DIR
    / "stock_master.parquet"
)

STOCK_CODE_HISTORY_PATH = (
    STAGE1_DIR
    / "stock_code_history.parquet"
)

TRADE_CALENDAR_USED_PATH = (
    STAGE1_DIR
    / "trade_calendar_used.csv"
)


# ------------------------------------------------------------
# 原始日频行情
# ------------------------------------------------------------

DAILY_TEMP3_DIR = Path(
    r"D:\lowfreq\daily_temp3"
)


# ------------------------------------------------------------
# Step 2 输出
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\output\M1_day1\02_stage2_daily_market_qa_final"
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

TARGET_EXCHANGES = [
    "SSE",
    "SZSE",
]


# ============================================================
# 2. 字段别名
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
# 3. 用于 Value-level Completeness 的字段
# ============================================================

VALUE_COMPLETENESS_FIELDS = [

    "security_id",
    "stock_code",
    "stock_name",
    "exchange",

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

    "is_open",

    "market_value",
    "neg_market_value",
    "turnover_rate",

    "pe",
    "pe1",
    "pb",

    "industry1",
    "industry2",
]


# ============================================================
# 4. 通用工具
# ============================================================

def normalize_colname(
    x: str,
) -> str:

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(x).lower(),
    )


def find_column(
    columns: Iterable[str],
    aliases: list[str],
) -> str | None:

    columns = list(columns)

    for alias in aliases:

        if alias in columns:

            return alias

    normalized = {

        normalize_colname(col): col

        for col in columns
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

    suffix = path.suffix.lower()

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
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# 5. 日期解析
# ============================================================

def parse_date(
    series: pd.Series,
) -> pd.Series:

    x = (
        series
        .astype("string")
        .str.strip()
    )

    result = pd.to_datetime(
        x,
        errors="coerce",
    )

    mask = x.str.fullmatch(
        r"\d{8}",
        na=False,
    )

    if mask.any():

        result.loc[mask] = (
            pd.to_datetime(
                x.loc[mask],
                format="%Y%m%d",
                errors="coerce",
            )
        )

    return result


# ============================================================
# 6. isOpen 标准化
#
# 重要修改：
# 不再把无法识别 / 缺失的 isOpen 自动变成 False。
#
# 返回 nullable boolean：
# True  = 正常交易
# False = 停牌
# <NA>  = 状态缺失
# ============================================================

def normalize_is_open(
    series: pd.Series,
) -> pd.Series:

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    numeric_value = pd.to_numeric(
        series,
        errors="coerce",
    )

    text = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    true_mask = (
        numeric_value.eq(1)
        |
        text.isin(
            {
                "1",
                "TRUE",
                "T",
                "YES",
                "Y",
                "OPEN",
                "TRADE",
            }
        )
    )

    false_mask = (
        numeric_value.eq(0)
        |
        text.isin(
            {
                "0",
                "FALSE",
                "F",
                "NO",
                "N",
                "CLOSE",
                "CLOSED",
                "SUSPEND",
                "SUSPENDED",
            }
        )
    )

    result.loc[
        true_mask
    ] = True

    result.loc[
        false_mask
    ] = False

    return result


# ============================================================
# 7. Exchange 标准化
# ============================================================

def normalize_exchange(
    series: pd.Series,
) -> pd.Series:

    x = (
        series
        .astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
    )

    result = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    result.loc[
        x.str.contains(
            r"SSE|XSHG|SHSE|SHANGHAI|上海|上交",
            regex=True,
            na=False,
        )
    ] = "SSE"

    result.loc[
        x.str.contains(
            r"SZSE|XSHE|SHENZHEN|深圳|深交",
            regex=True,
            na=False,
        )
    ] = "SZSE"

    result.loc[
        x.str.contains(
            r"BSE|XBJ|BEIJING|北京|北交",
            regex=True,
            na=False,
        )
    ] = "BSE"

    return result


def infer_exchange_from_secid(
    security_id: pd.Series,
) -> pd.Series:

    x = (
        security_id
        .astype("string")
        .str.upper()
    )

    result = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    result.loc[
        x.str.contains(
            r"XSHG|\.SH$|SSE",
            regex=True,
            na=False,
        )
    ] = "SSE"

    result.loc[
        x.str.contains(
            r"XSHE|\.SZ$|SZSE",
            regex=True,
            na=False,
        )
    ] = "SZSE"

    result.loc[
        x.str.contains(
            r"XBJ|\.BJ$|BSE",
            regex=True,
            na=False,
        )
    ] = "BSE"

    return result


# ============================================================
# 8. 股票代码标准化
#
# 解决：
# 000991 被 pandas 读成 991 等问题。
#
# 优先从 secID 中提取 6 位股票代码。
# ============================================================

def canonical_stock_code(
    stock_code: pd.Series,
    security_id: pd.Series,
) -> pd.Series:

    sid = (
        security_id
        .astype("string")
        .str.strip()
        .str.upper()
    )

    code_from_sid = (
        sid.str.extract(
            r"^(\d{6})",
            expand=False,
        )
    )

    code = (
        stock_code
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # 提取纯数字
    digit_code = (
        code.str.extract(
            r"^(\d{1,6})$",
            expand=False,
        )
    )

    digit_code = (
        digit_code.str.zfill(6)
    )

    result = (
        code_from_sid
        .fillna(
            digit_code
        )
        .fillna(
            code
        )
    )

    return result


# ============================================================
# 9. 从文件名中提取交易日期
# ============================================================

def extract_date_from_filename(
    path: Path,
) -> pd.Timestamp | None:

    name = path.stem

    patterns = [

        r"(20\d{2})[-_](\d{2})[-_](\d{2})",

        r"(20\d{2})(\d{2})(\d{2})",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            name,
        )

        if match:

            try:

                return pd.Timestamp(
                    year=int(
                        match.group(1)
                    ),
                    month=int(
                        match.group(2)
                    ),
                    day=int(
                        match.group(3)
                    ),
                )

            except ValueError:

                return None

    return None


# ============================================================
# 10. 读取 Step 1 交易日历
# ============================================================

def load_trade_dates():

    if not (
        TRADE_CALENDAR_USED_PATH.exists()
    ):

        raise FileNotFoundError(
            f"找不到：{TRADE_CALENDAR_USED_PATH}"
        )

    df = pd.read_csv(
        TRADE_CALENDAR_USED_PATH
    )

    if (
        "trade_date"
        not in df.columns
    ):

        raise ValueError(
            "trade_calendar_used.csv "
            "缺少 trade_date。"
        )

    dates = pd.to_datetime(
        df[
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
# 11. 建立 daily_temp3 文件地图
# ============================================================

def build_daily_file_map(
    trade_dates: pd.DatetimeIndex,
):

    if not (
        DAILY_TEMP3_DIR.exists()
    ):

        raise FileNotFoundError(
            f"目录不存在：{DAILY_TEMP3_DIR}"
        )

    file_map = defaultdict(
        list
    )

    ignored_files = []

    for path in (
        DAILY_TEMP3_DIR.rglob("*")
    ):

        if not path.is_file():

            continue

        if (
            path.suffix.lower()
            not in
            SUPPORTED_SUFFIX
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

    missing_dates = [

        date

        for date in trade_dates

        if date not in file_map
    ]

    duplicated_dates = {

        date: files

        for date, files
        in file_map.items()

        if len(files) > 1
    }

    inventory_rows = []

    for date in trade_dates:

        files = (
            file_map.get(
                date,
                [],
            )
        )

        inventory_rows.append(
            {
                "trade_date":
                    date,

                "file_count":
                    len(files),

                "file_path":
                    (
                        str(
                            files[0]
                        )
                        if len(files) == 1
                        else
                        "|".join(
                            str(x)
                            for x in files
                        )
                    ),
            }
        )

    pd.DataFrame(
        inventory_rows
    ).to_csv(
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
    print("daily_temp3 文件完整性")
    print("=" * 72)

    print(
        f"应有交易日：{len(trade_dates):,}"
    )

    print(
        f"成功映射：{len(file_map):,}"
    )

    print(
        f"缺失：{len(missing_dates):,}"
    )

    print(
        f"重复：{len(duplicated_dates):,}"
    )

    if missing_dates:

        raise RuntimeError(
            "存在缺失交易日文件。"
        )

    if duplicated_dates:

        raise RuntimeError(
            "存在重复交易日文件。"
        )

    return {

        date: files[0]

        for date, files
        in file_map.items()
    }


# ============================================================
# 12. 读取 Stock Master
# ============================================================

def load_stock_master():

    if not (
        STOCK_MASTER_PATH.exists()
    ):

        raise FileNotFoundError(
            f"找不到：{STOCK_MASTER_PATH}"
        )

    df = pd.read_parquet(
        STOCK_MASTER_PATH
    )

    df = df.copy()

    df[
        "security_id"
    ] = (
        df[
            "security_id"
        ]
        .astype("string")
        .str.strip()
    )

    df[
        "list_date"
    ] = pd.to_datetime(
        df[
            "list_date"
        ],
        errors="coerce",
    )

    df[
        "delist_date"
    ] = pd.to_datetime(
        df[
            "delist_date"
        ],
        errors="coerce",
    )

    return df


# ============================================================
# 13. 读取历史代码表
#
# 用于 Identity Reconciliation。
# ============================================================

def load_stock_code_history():

    if not (
        STOCK_CODE_HISTORY_PATH.exists()
    ):

        print(
            "[WARNING] "
            "未找到 stock_code_history.parquet，"
            "Identity reconciliation 将只使用当前代码和 secID。"
        )

        return pd.DataFrame()

    df = pd.read_parquet(
        STOCK_CODE_HISTORY_PATH
    )

    df = df.copy()

    df[
        "security_id"
    ] = (
        df[
            "security_id"
        ]
        .astype("string")
        .str.strip()
    )

    if (
        "exchange"
        not in df.columns
    ):

        df[
            "exchange"
        ] = (
            infer_exchange_from_secid(
                df[
                    "security_id"
                ]
            )
        )

    else:

        df[
            "exchange"
        ] = (
            normalize_exchange(
                df[
                    "exchange"
                ]
            )
        )

    df[
        "stock_code"
    ] = (
        canonical_stock_code(
            df[
                "stock_code"
            ],
            df[
                "security_id"
            ],
        )
    )

    return df


# ============================================================
# 14. 建立 secID -> 历史代码集合
# ============================================================

def build_code_alias_lookup(
    history: pd.DataFrame,
):

    lookup = defaultdict(
        set
    )

    if history.empty:

        return lookup

    for row in (
        history.itertuples(
            index=False
        )
    ):

        if (
            pd.isna(
                row.security_id
            )
            or
            pd.isna(
                row.stock_code
            )
        ):

            continue

        exchange = (
            row.exchange
            if pd.notna(
                row.exchange
            )
            else ""
        )

        lookup[
            str(
                row.security_id
            )
        ].add(
            (
                str(
                    exchange
                ),
                str(
                    row.stock_code
                ),
            )
        )

    return lookup


# ============================================================
# 15. 构造某日 PIT 理论 Universe
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

    expected[
        "stock_code"
    ] = (
        canonical_stock_code(
            expected[
                "stock_code"
            ],
            expected[
                "security_id"
            ],
        )
    )

    return expected


# ============================================================
# 16. 标准化 daily_temp3
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
            "daily_temp3 中没有识别到 secID。"
        )

    result = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # secID
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ticker 原始值
    # --------------------------------------------------------

    if (
        mapping[
            "stock_code"
        ]
        is not None
    ):

        original_code = (
            df[
                mapping[
                    "stock_code"
                ]
            ]
        )

    else:

        original_code = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

    # --------------------------------------------------------
    # 修复前导零
    # --------------------------------------------------------

    result[
        "stock_code"
    ] = (
        canonical_stock_code(
            original_code,
            result[
                "security_id"
            ],
        )
    )

    # --------------------------------------------------------
    # 股票名称
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Exchange
    # --------------------------------------------------------

    if (
        mapping[
            "exchange"
        ]
        is not None
    ):

        result[
            "exchange"
        ] = (
            normalize_exchange(
                df[
                    mapping[
                        "exchange"
                    ]
                ]
            )
        )

    else:

        result[
            "exchange"
        ] = pd.NA

    # exchange 缺失则由 secID 恢复
    result[
        "exchange"
    ] = (
        result[
            "exchange"
        ]
        .fillna(
            infer_exchange_from_secid(
                result[
                    "security_id"
                ]
            )
        )
    )

    result[
        "trade_date"
    ] = file_date

    # --------------------------------------------------------
    # 数值字段
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

    for field in (
        numeric_fields
    ):

        column = mapping[
            field
        ]

        if column is None:

            result[
                field
            ] = np.nan

        else:

            result[
                field
            ] = numeric(
                df[
                    column
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
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

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

        column = mapping[
            field
        ]

        if column is None:

            result[
                field
            ] = pd.NA

        else:

            result[
                field
            ] = (
                df[
                    column
                ]
                .astype("string")
            )

    # --------------------------------------------------------
    # 只保留当前沪深主研究样本
    # --------------------------------------------------------

    result = (
        result[
            result[
                "exchange"
            ].isin(
                TARGET_EXCHANGES
            )
        ]
        .copy()
    )

    return (
        result,
        mapping,
    )


# ============================================================
# 17. 构造单个证券的候选代码集合
# ============================================================

def candidate_code_keys(
    security_id: str,
    exchange: str,
    stock_code: str,
    alias_lookup,
):

    keys = set()

    if (
        pd.notna(
            stock_code
        )
        and
        pd.notna(
            exchange
        )
    ):

        keys.add(
            (
                str(
                    exchange
                ),
                str(
                    stock_code
                ),
            )
        )

    # 加入历史代码
    if (
        security_id
        in alias_lookup
    ):

        keys.update(
            alias_lookup[
                security_id
            ]
        )

    return keys


# ============================================================
# 18. Identity Reconciliation
#
# 修复问题 1：
#
# 对原来的
#
# RAW_ONLY / DAILY_ONLY
#
# 尝试按照：
#
# exchange + normalized historical ticker
#
# 进行一对一身份匹配。
#
# 只有一对一唯一匹配才自动 reconcile。
# 模糊情况不自动合并。
# ============================================================

def reconcile_identity_mismatches(
    raw_only_df: pd.DataFrame,
    daily_only_df: pd.DataFrame,
    alias_lookup,
    date: pd.Timestamp,
):

    reconciliation_rows = []

    ambiguous_rows = []

    if (
        raw_only_df.empty
        or
        daily_only_df.empty
    ):

        return (
            set(),
            set(),
            reconciliation_rows,
            ambiguous_rows,
        )

    raw_candidates = {}

    for row in (
        raw_only_df.itertuples(
            index=False
        )
    ):

        sid = str(
            row.security_id
        )

        raw_candidates[
            sid
        ] = (
            candidate_code_keys(
                sid,
                row.exchange,
                row.stock_code,
                alias_lookup,
            )
        )

    daily_candidates = {}

    for row in (
        daily_only_df.itertuples(
            index=False
        )
    ):

        sid = str(
            row.security_id
        )

        daily_candidates[
            sid
        ] = (
            candidate_code_keys(
                sid,
                row.exchange,
                row.stock_code,
                alias_lookup,
            )
        )

    # --------------------------------------------------------
    # 构造可能匹配关系
    # --------------------------------------------------------

    raw_to_daily = defaultdict(
        set
    )

    daily_to_raw = defaultdict(
        set
    )

    common_key_lookup = {}

    for raw_sid, raw_keys in (
        raw_candidates.items()
    ):

        for daily_sid, daily_keys in (
            daily_candidates.items()
        ):

            common = (
                raw_keys
                &
                daily_keys
            )

            if common:

                raw_to_daily[
                    raw_sid
                ].add(
                    daily_sid
                )

                daily_to_raw[
                    daily_sid
                ].add(
                    raw_sid
                )

                common_key_lookup[
                    (
                        raw_sid,
                        daily_sid,
                    )
                ] = sorted(
                    common
                )

    reconciled_raw = set()

    reconciled_daily = set()

    # --------------------------------------------------------
    # 只有 mutual one-to-one 才自动 reconcile
    # --------------------------------------------------------

    for raw_sid, daily_set in (
        raw_to_daily.items()
    ):

        if len(
            daily_set
        ) != 1:

            continue

        daily_sid = next(
            iter(
                daily_set
            )
        )

        reverse = (
            daily_to_raw[
                daily_sid
            ]
        )

        if len(
            reverse
        ) != 1:

            continue

        if (
            raw_sid
            not in reverse
        ):

            continue

        reconciled_raw.add(
            raw_sid
        )

        reconciled_daily.add(
            daily_sid
        )

        raw_row = (
            raw_only_df[
                raw_only_df[
                    "security_id"
                ].astype("string")
                .eq(
                    raw_sid
                )
            ]
            .iloc[0]
        )

        daily_row = (
            daily_only_df[
                daily_only_df[
                    "security_id"
                ].astype("string")
                .eq(
                    daily_sid
                )
            ]
            .iloc[0]
        )

        common_keys = (
            common_key_lookup[
                (
                    raw_sid,
                    daily_sid,
                )
            ]
        )

        reconciliation_rows.append(
            {

                "trade_date":
                    date,

                "raw_security_id":
                    raw_sid,

                "daily_security_id":
                    daily_sid,

                "raw_stock_code":
                    raw_row[
                        "stock_code"
                    ],

                "daily_stock_code":
                    daily_row[
                        "stock_code"
                    ],

                "raw_stock_name":
                    raw_row[
                        "stock_name"
                    ],

                "daily_stock_name":
                    daily_row[
                        "stock_name"
                    ],

                "exchange":
                    raw_row[
                        "exchange"
                    ],

                "common_code_keys":
                    "|".join(
                        f"{x[0]}:{x[1]}"
                        for x
                        in common_keys
                    ),

                "reconciliation_type":
                    "ONE_TO_ONE_CODE_ALIAS",

                "status":
                    "RECONCILED",
            }
        )

    # --------------------------------------------------------
    # 模糊匹配保留，不自动合并
    # --------------------------------------------------------

    for raw_sid, candidates in (
        raw_to_daily.items()
    ):

        if (
            raw_sid
            in reconciled_raw
        ):

            continue

        if not candidates:

            continue

        ambiguous_rows.append(
            {

                "trade_date":
                    date,

                "side":
                    "RAW",

                "security_id":
                    raw_sid,

                "candidate_count":
                    len(
                        candidates
                    ),

                "candidate_security_ids":
                    "|".join(
                        sorted(
                            candidates
                        )
                    ),

                "issue_type":
                    "AMBIGUOUS_IDENTITY_RECONCILIATION",

                "severity":
                    "REVIEW",
            }
        )

    return (
        reconciled_raw,
        reconciled_daily,
        reconciliation_rows,
        ambiguous_rows,
    )


# ============================================================
# 19. Universe Coverage
#
# 同时输出：
#
# exact coverage
# reconciled coverage
# unresolved mismatch
# ============================================================

def universe_coverage_check(
    date,
    expected,
    daily,
    alias_lookup,
):

    expected_ids = set(
        expected[
            "security_id"
        ]
        .dropna()
        .astype("string")
    )

    daily_ids = set(
        daily[
            "security_id"
        ]
        .dropna()
        .astype("string")
    )

    exact_matched_ids = (
        expected_ids
        &
        daily_ids
    )

    raw_only_ids = (
        expected_ids
        -
        daily_ids
    )

    daily_only_ids = (
        daily_ids
        -
        expected_ids
    )

    raw_only_df = (
        expected[
            expected[
                "security_id"
            ]
            .astype("string")
            .isin(
                raw_only_ids
            )
        ]
        .copy()
    )

    daily_only_df = (
        daily[
            daily[
                "security_id"
            ]
            .astype("string")
            .isin(
                daily_only_ids
            )
        ]
        .drop_duplicates(
            subset=[
                "security_id"
            ]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Identity reconciliation
    # --------------------------------------------------------

    (
        reconciled_raw_ids,
        reconciled_daily_ids,
        reconciliation_rows,
        ambiguous_rows,
    ) = (
        reconcile_identity_mismatches(
            raw_only_df,
            daily_only_df,
            alias_lookup,
            date,
        )
    )

    unresolved_raw_ids = (
        raw_only_ids
        -
        reconciled_raw_ids
    )

    unresolved_daily_ids = (
        daily_only_ids
        -
        reconciled_daily_ids
    )

    expected_count = len(
        expected_ids
    )

    exact_matched_count = len(
        exact_matched_ids
    )

    reconciled_count = len(
        reconciled_raw_ids
    )

    effective_matched_count = (
        exact_matched_count
        +
        reconciled_count
    )

    coverage_exact = (
        exact_matched_count
        /
        expected_count
        if expected_count > 0
        else np.nan
    )

    coverage_reconciled = (
        effective_matched_count
        /
        expected_count
        if expected_count > 0
        else np.nan
    )

    # --------------------------------------------------------
    # 原始 mismatch detail
    # --------------------------------------------------------

    raw_mismatch_rows = []

    for row in (
        raw_only_df.itertuples(
            index=False
        )
    ):

        raw_mismatch_rows.append(
            {

                "trade_date":
                    date,

                "mismatch_type":
                    "RAW_ONLY_BEFORE_RECONCILIATION",

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

    for row in (
        daily_only_df.itertuples(
            index=False
        )
    ):

        raw_mismatch_rows.append(
            {

                "trade_date":
                    date,

                "mismatch_type":
                    "DAILY_ONLY_BEFORE_RECONCILIATION",

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
    # 未解决 mismatch
    # --------------------------------------------------------

    unresolved_rows = []

    if unresolved_raw_ids:

        subset = (
            raw_only_df[
                raw_only_df[
                    "security_id"
                ]
                .astype("string")
                .isin(
                    unresolved_raw_ids
                )
            ]
        )

        for row in (
            subset.itertuples(
                index=False
            )
        ):

            unresolved_rows.append(
                {

                    "trade_date":
                        date,

                    "mismatch_type":
                        "UNRESOLVED_RAW_ONLY",

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

    if unresolved_daily_ids:

        subset = (
            daily_only_df[
                daily_only_df[
                    "security_id"
                ]
                .astype("string")
                .isin(
                    unresolved_daily_ids
                )
            ]
        )

        for row in (
            subset.itertuples(
                index=False
            )
        ):

            unresolved_rows.append(
                {

                    "trade_date":
                        date,

                    "mismatch_type":
                        "UNRESOLVED_DAILY_ONLY",

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
    # 用于后续 QA 的实际 daily secID
    #
    # exact matched
    # +
    # reconciled daily IDs
    # --------------------------------------------------------

    resolved_daily_ids = (
        exact_matched_ids
        |
        reconciled_daily_ids
    )

    summary = {

        "trade_date":
            date,

        "expected_count":
            expected_count,

        "daily_unique_security_count":
            len(
                daily_ids
            ),

        "exact_matched_count":
            exact_matched_count,

        "raw_only_before_count":
            len(
                raw_only_ids
            ),

        "daily_only_before_count":
            len(
                daily_only_ids
            ),

        "identity_reconciled_count":
            reconciled_count,

        "effective_matched_count":
            effective_matched_count,

        "unresolved_raw_only_count":
            len(
                unresolved_raw_ids
            ),

        "unresolved_daily_only_count":
            len(
                unresolved_daily_ids
            ),

        "coverage_ratio_exact":
            coverage_exact,

        "coverage_ratio_reconciled":
            coverage_reconciled,
    }

    return (

        summary,

        raw_mismatch_rows,

        reconciliation_rows,

        ambiguous_rows,

        unresolved_rows,

        resolved_daily_ids,
    )


# ============================================================
# 20. Missing Value 判定
# ============================================================

def value_missing_mask(
    series: pd.Series,
) -> pd.Series:

    mask = (
        series.isna()
    )

    if (
        pd.api.types.is_string_dtype(
            series.dtype
        )
        or
        series.dtype == object
    ):

        text = (
            series
            .astype("string")
            .str.strip()
        )

        mask = (
            mask
            |
            text.eq("")
        )

    return mask.fillna(
        True
    )


# ============================================================
# 21. Value-level Completeness
#
# 修复问题 3：
#
# 不仅看字段是否存在，
# 还统计每个 stock-day 的实际缺失情况。
#
# 同时区分：
#
# 全部研究记录
# isOpen = 1
# isOpen = 0
# ============================================================

def update_value_completeness(
    daily: pd.DataFrame,
    global_stats,
    daily_rows,
    date,
):

    active = (
        daily[
            "is_open"
        ]
        .eq(
            True
        )
        .fillna(
            False
        )
    )

    suspended = (
        daily[
            "is_open"
        ]
        .eq(
            False
        )
        .fillna(
            False
        )
    )

    unknown_open = (
        daily[
            "is_open"
        ]
        .isna()
    )

    for field in (
        VALUE_COMPLETENESS_FIELDS
    ):

        missing = (
            value_missing_mask(
                daily[
                    field
                ]
            )
        )

        total_count = len(
            daily
        )

        missing_count = int(
            missing.sum()
        )

        nonmissing_count = (
            total_count
            -
            missing_count
        )

        active_count = int(
            active.sum()
        )

        active_missing = int(
            (
                missing
                &
                active
            ).sum()
        )

        suspended_count = int(
            suspended.sum()
        )

        suspended_missing = int(
            (
                missing
                &
                suspended
            ).sum()
        )

        unknown_count = int(
            unknown_open.sum()
        )

        # ----------------------------------------------------
        # 累积全历史
        # ----------------------------------------------------

        stats = (
            global_stats[
                field
            ]
        )

        stats[
            "total_count"
        ] += total_count

        stats[
            "missing_count"
        ] += missing_count

        stats[
            "nonmissing_count"
        ] += nonmissing_count

        stats[
            "active_count"
        ] += active_count

        stats[
            "active_missing_count"
        ] += active_missing

        stats[
            "suspended_count"
        ] += suspended_count

        stats[
            "suspended_missing_count"
        ] += suspended_missing

        stats[
            "unknown_is_open_count"
        ] += unknown_count

        # ----------------------------------------------------
        # 每日长表
        # ----------------------------------------------------

        daily_rows.append(
            {

                "trade_date":
                    date,

                "field":
                    field,

                "total_count":
                    total_count,

                "missing_count":
                    missing_count,

                "missing_rate":
                    (
                        missing_count
                        /
                        total_count
                        if total_count > 0
                        else np.nan
                    ),

                "active_count":
                    active_count,

                "active_missing_count":
                    active_missing,

                "active_missing_rate":
                    (
                        active_missing
                        /
                        active_count
                        if active_count > 0
                        else np.nan
                    ),

                "suspended_count":
                    suspended_count,

                "suspended_missing_count":
                    suspended_missing,

                "suspended_missing_rate":
                    (
                        suspended_missing
                        /
                        suspended_count
                        if suspended_count > 0
                        else np.nan
                    ),
            }
        )


# ============================================================
# 22. Column-level Presence
# ============================================================

def update_column_presence(
    mapping,
    stats,
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

            stats[
                field
            ][
                "absent_day_count"
            ] += 1

        else:

            stats[
                field
            ][
                "present_day_count"
            ] += 1


# ============================================================
# 23. Core Market QA
#
# 修复问题 2：
#
# Price / OHLC 异常只对 isOpen=True 执行。
#
# isOpen=False 的 0 值行情作为停牌占位统计，
# 不再判为 HIGH。
# ============================================================

def core_market_qa(
    daily: pd.DataFrame,
    date: pd.Timestamp,
):

    anomaly_rows = []

    # --------------------------------------------------------
    # 状态
    # --------------------------------------------------------

    active = (
        daily[
            "is_open"
        ]
        .eq(
            True
        )
        .fillna(
            False
        )
    )

    suspended = (
        daily[
            "is_open"
        ]
        .eq(
            False
        )
        .fillna(
            False
        )
    )

    unknown_open = (
        daily[
            "is_open"
        ]
        .isna()
    )

    # --------------------------------------------------------
    # Duplicate
    # --------------------------------------------------------

    duplicate_mask = (
        daily[
            "security_id"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_security_count = (
        daily.loc[
            duplicate_mask,
            "security_id",
        ]
        .nunique()
    )

    # --------------------------------------------------------
    # Missing secID
    # --------------------------------------------------------

    missing_security_id = (
        daily[
            "security_id"
        ].isna()
        |
        daily[
            "security_id"
        ]
        .astype("string")
        .str.strip()
        .eq("")
    )

    # --------------------------------------------------------
    # Active missing close / OHLC
    # --------------------------------------------------------

    active_missing_close = (
        active
        &
        daily[
            "close"
        ].isna()
    )

    active_missing_ohlc = (
        active
        &
        daily[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    # ========================================================
    # 核心修复：
    # 只有 active stock 才检查价格必须 > 0。
    # ========================================================

    invalid_price = (

        active

        &

        (

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
    )

    # --------------------------------------------------------
    # Active OHLC logic only
    # --------------------------------------------------------

    complete_ohlc = (
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

        active

        &

        complete_ohlc

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

    # ========================================================
    # 停牌占位值
    #
    # 不作为 HIGH。
    # ========================================================

    suspended_nonpositive_ohlc = (

        suspended

        &

        (

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
    )

    suspended_all_zero_ohlc = (

        suspended

        &

        daily[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .fillna(
            np.nan
        )
        .eq(0)
        .all(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Volume / Amount
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
    # Adj factor
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
    # 停牌但有成交
    # --------------------------------------------------------

    closed_but_traded = (

        suspended

        &

        (

            (
                daily[
                    "volume"
                ]
                .fillna(0)
                > 0
            )

            |

            (
                daily[
                    "turnover_value"
                ]
                .fillna(0)
                > 0
            )
        )
    )

    # --------------------------------------------------------
    # 开盘但零成交
    # --------------------------------------------------------

    open_but_zero_trade = (

        active

        &

        (
            daily[
                "volume"
            ]
            .fillna(0)
            == 0
        )

        &

        (
            daily[
                "turnover_value"
            ]
            .fillna(0)
            == 0
        )
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    invalid_vwap = (

        active

        &

        (
            daily[
                "volume"
            ]
            .fillna(0)
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
    # Market Value
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

    # ========================================================
    # Row-level anomalies
    # ========================================================

    anomaly_definitions = [

        (
            active_missing_close,
            "ACTIVE_MISSING_CLOSE",
            "HIGH",
            "isOpen=1 但 closePrice 缺失。",
        ),

        (
            active_missing_ohlc,
            "ACTIVE_MISSING_OHLC",
            "REVIEW",
            "isOpen=1 但至少一个 OHLC 字段缺失。",
        ),

        (
            invalid_price,
            "ACTIVE_NONPOSITIVE_PRICE",
            "HIGH",
            "isOpen=1 时 OHLC 存在非正价格。",
        ),

        (
            invalid_ohlc,
            "ACTIVE_INVALID_OHLC_RELATION",
            "HIGH",
            "isOpen=1 时 High/Low 与 Open/Close 逻辑不一致。",
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
            "isOpen=0 但存在成交量或成交额。",
        ),

        (
            invalid_vwap,
            "INVALID_VWAP",
            "REVIEW",
            "正常交易且有成交，但 VWAP 缺失或 <= 0。",
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
    # Daily QA Summary
    # --------------------------------------------------------

    summary = {

        "trade_date":
            date,

        "research_row_count":
            len(
                daily
            ),

        "unique_security_count":
            daily[
                "security_id"
            ].nunique(),

        "active_stock_count":
            int(
                active.sum()
            ),

        "suspended_stock_count":
            int(
                suspended.sum()
            ),

        "unknown_is_open_count":
            int(
                unknown_open.sum()
            ),

        "duplicate_security_count":
            int(
                duplicate_security_count
            ),

        "missing_security_id_count":
            int(
                missing_security_id.sum()
            ),

        "active_missing_close_count":
            int(
                active_missing_close.sum()
            ),

        "active_missing_ohlc_count":
            int(
                active_missing_ohlc.sum()
            ),

        "active_invalid_price_count":
            int(
                invalid_price.sum()
            ),

        "active_invalid_ohlc_count":
            int(
                invalid_ohlc.sum()
            ),

        # 仅诊断，不作为 HIGH
        "suspended_nonpositive_ohlc_count":
            int(
                suspended_nonpositive_ohlc.sum()
            ),

        "suspended_all_zero_ohlc_count":
            int(
                suspended_all_zero_ohlc.sum()
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
    }

    return (
        summary,
        anomaly_rows,
    )


# ============================================================
# 24. Value Completeness Summary
# ============================================================

def build_value_completeness_summary(
    global_stats,
):

    rows = []

    for field in (
        VALUE_COMPLETENESS_FIELDS
    ):

        stats = (
            global_stats[
                field
            ]
        )

        total = (
            stats[
                "total_count"
            ]
        )

        active = (
            stats[
                "active_count"
            ]
        )

        suspended = (
            stats[
                "suspended_count"
            ]
        )

        rows.append(
            {

                "field":
                    field,

                "total_count":
                    total,

                "nonmissing_count":
                    stats[
                        "nonmissing_count"
                    ],

                "missing_count":
                    stats[
                        "missing_count"
                    ],

                "missing_rate":
                    (
                        stats[
                            "missing_count"
                        ]
                        /
                        total
                        if total > 0
                        else np.nan
                    ),

                "active_count":
                    active,

                "active_missing_count":
                    stats[
                        "active_missing_count"
                    ],

                "active_missing_rate":
                    (
                        stats[
                            "active_missing_count"
                        ]
                        /
                        active
                        if active > 0
                        else np.nan
                    ),

                "suspended_count":
                    suspended,

                "suspended_missing_count":
                    stats[
                        "suspended_missing_count"
                    ],

                "suspended_missing_rate":
                    (
                        stats[
                            "suspended_missing_count"
                        ]
                        /
                        suspended
                        if suspended > 0
                        else np.nan
                    ),

                "unknown_is_open_count":
                    stats[
                        "unknown_is_open_count"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 25. Overall Summary
# ============================================================

def build_overall_summary(
    coverage_df,
    qa_df,
    anomaly_df,
    reconciliation_df,
):

    total_expected = (
        coverage_df[
            "expected_count"
        ].sum()
    )

    total_exact = (
        coverage_df[
            "exact_matched_count"
        ].sum()
    )

    total_effective = (
        coverage_df[
            "effective_matched_count"
        ].sum()
    )

    weighted_exact = (
        total_exact
        /
        total_expected
        if total_expected > 0
        else np.nan
    )

    weighted_reconciled = (
        total_effective
        /
        total_expected
        if total_expected > 0
        else np.nan
    )

    high_count = (
        anomaly_df[
            "severity"
        ]
        .eq(
            "HIGH"
        )
        .sum()
        if not
        anomaly_df.empty
        else 0
    )

    review_count = (
        anomaly_df[
            "severity"
        ]
        .eq(
            "REVIEW"
        )
        .sum()
        if not
        anomaly_df.empty
        else 0
    )

    return pd.DataFrame(
        {

            "metric": [

                "trade_day_count",

                "total_expected_stock_day_records",

                "total_exact_matched_records",

                "total_effective_matched_records",

                "weighted_exact_coverage_ratio",

                "weighted_reconciled_coverage_ratio",

                "min_daily_exact_coverage_ratio",

                "min_daily_reconciled_coverage_ratio",

                "total_identity_reconciled_records",

                "days_with_unresolved_raw_only",

                "days_with_unresolved_daily_only",

                "total_unresolved_raw_only_records",

                "total_unresolved_daily_only_records",

                "total_active_stock_day_records",

                "total_suspended_stock_day_records",

                "total_suspended_nonpositive_ohlc_records",

                "total_suspended_all_zero_ohlc_records",

                "total_active_invalid_price_count",

                "total_active_invalid_ohlc_count",

                "total_active_missing_close_count",

                "total_negative_volume_count",

                "total_negative_turnover_value_count",

                "total_invalid_adj_factor_count",

                "total_closed_but_traded_count",

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
                    total_exact
                ),

                int(
                    total_effective
                ),

                weighted_exact,

                weighted_reconciled,

                coverage_df[
                    "coverage_ratio_exact"
                ].min(),

                coverage_df[
                    "coverage_ratio_reconciled"
                ].min(),

                (
                    len(
                        reconciliation_df
                    )
                    if not
                    reconciliation_df.empty
                    else 0
                ),

                int(
                    (
                        coverage_df[
                            "unresolved_raw_only_count"
                        ]
                        > 0
                    ).sum()
                ),

                int(
                    (
                        coverage_df[
                            "unresolved_daily_only_count"
                        ]
                        > 0
                    ).sum()
                ),

                int(
                    coverage_df[
                        "unresolved_raw_only_count"
                    ].sum()
                ),

                int(
                    coverage_df[
                        "unresolved_daily_only_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "active_stock_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "suspended_stock_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "suspended_nonpositive_ohlc_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "suspended_all_zero_ohlc_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "active_invalid_price_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "active_invalid_ohlc_count"
                    ].sum()
                ),

                int(
                    qa_df[
                        "active_missing_close_count"
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
                    high_count
                ),

                int(
                    review_count
                ),
            ],
        }
    )


# ============================================================
# 26. 主程序
# ============================================================

def main():

    print("=" * 72)
    print("M1 Stage 2 - Revised")
    print("Daily Market Coverage + Core QA")
    print("=" * 72)

    # ========================================================
    # A. 基础数据
    # ========================================================

    stock_master = (
        load_stock_master()
    )

    code_history = (
        load_stock_code_history()
    )

    alias_lookup = (
        build_code_alias_lookup(
            code_history
        )
    )

    trade_dates = (
        load_trade_dates()
    )

    file_map = (
        build_daily_file_map(
            trade_dates
        )
    )

    print()
    print(
        f"历史证券主体："
        f"{stock_master['security_id'].nunique():,}"
    )

    print(
        f"研究交易日："
        f"{len(trade_dates):,}"
    )

    # ========================================================
    # B. 初始化
    # ========================================================

    coverage_results = []

    raw_mismatch_results = []

    reconciliation_results = []

    ambiguous_results = []

    unresolved_results = []

    qa_results = []

    anomaly_results = []

    processing_log = []

    # column-level presence
    column_presence_stats = defaultdict(
        lambda: {
            "present_day_count": 0,
            "absent_day_count": 0,
        }
    )

    # value-level completeness
    value_stats = defaultdict(
        lambda: {
            "total_count": 0,
            "nonmissing_count": 0,
            "missing_count": 0,
            "active_count": 0,
            "active_missing_count": 0,
            "suspended_count": 0,
            "suspended_missing_count": 0,
            "unknown_is_open_count": 0,
        }
    )

    daily_value_rows = []

    # ========================================================
    # C. 逐日处理
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

            # ------------------------------------------------
            # 读取 daily
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Column presence
            # ------------------------------------------------

            update_column_presence(
                mapping,
                column_presence_stats,
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
            # Coverage + Reconciliation
            # ------------------------------------------------

            (
                coverage_summary,
                raw_mismatch_rows,
                reconciliation_rows,
                ambiguous_rows,
                unresolved_rows,
                resolved_daily_ids,
            ) = (
                universe_coverage_check(
                    date,
                    expected,
                    daily,
                    alias_lookup,
                )
            )

            coverage_results.append(
                coverage_summary
            )

            raw_mismatch_results.extend(
                raw_mismatch_rows
            )

            reconciliation_results.extend(
                reconciliation_rows
            )

            ambiguous_results.extend(
                ambiguous_rows
            )

            unresolved_results.extend(
                unresolved_rows
            )

            # =================================================
            # 后续 QA 仅对已经属于 PIT Universe
            # 的实际 daily 记录执行。
            #
            # exact match
            # +
            # successfully reconciled alias
            # =================================================

            research_daily = (
                daily[
                    daily[
                        "security_id"
                    ]
                    .astype("string")
                    .isin(
                        resolved_daily_ids
                    )
                ]
                .copy()
            )

            # ------------------------------------------------
            # Core QA
            # ------------------------------------------------

            (
                qa_summary,
                anomaly_rows,
            ) = (
                core_market_qa(
                    research_daily,
                    date,
                )
            )

            qa_results.append(
                qa_summary
            )

            anomaly_results.extend(
                anomaly_rows
            )

            # ------------------------------------------------
            # Value-level completeness
            # ------------------------------------------------

            update_value_completeness(
                research_daily,
                value_stats,
                daily_value_rows,
                date,
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
                        len(
                            raw_daily
                        ),

                    "target_exchange_row_count":
                        len(
                            daily
                        ),

                    "research_row_count":
                        len(
                            research_daily
                        ),

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

                    "target_exchange_row_count":
                        np.nan,

                    "research_row_count":
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
    # D. DataFrames
    # ========================================================

    coverage_df = pd.DataFrame(
        coverage_results
    )

    raw_mismatch_df = pd.DataFrame(
        raw_mismatch_results
    )

    reconciliation_df = pd.DataFrame(
        reconciliation_results
    )

    ambiguous_df = pd.DataFrame(
        ambiguous_results
    )

    unresolved_df = pd.DataFrame(
        unresolved_results
    )

    qa_df = pd.DataFrame(
        qa_results
    )

    anomaly_df = pd.DataFrame(
        anomaly_results
    )

    log_df = pd.DataFrame(
        processing_log
    )

    daily_value_df = pd.DataFrame(
        daily_value_rows
    )

    # ========================================================
    # E. Column-level presence
    # ========================================================

    column_rows = []

    for field, stats in (
        column_presence_stats.items()
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

        column_rows.append(
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
        column_rows
    )

    # ========================================================
    # F. Value-level completeness
    # ========================================================

    value_completeness_df = (
        build_value_completeness_summary(
            value_stats
        )
    )

    # ========================================================
    # G. Overall Summary
    # ========================================================

    overall_summary = (
        build_overall_summary(
            coverage_df,
            qa_df,
            anomaly_df,
            reconciliation_df,
        )
    )

    # ========================================================
    # H. 输出
    # ========================================================

    coverage_df.to_csv(
        OUTPUT_DIR
        / "daily_market_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 原始未 reconciliation 的 mismatch
    raw_mismatch_df.to_parquet(
        OUTPUT_DIR
        / "universe_mismatch_raw_detail.parquet",
        index=False,
    )

    # 成功身份 reconciliation
    reconciliation_df.to_parquet(
        OUTPUT_DIR
        / "identity_reconciliation_detail.parquet",
        index=False,
    )

    # 模糊匹配
    ambiguous_df.to_parquet(
        OUTPUT_DIR
        / "identity_reconciliation_ambiguous.parquet",
        index=False,
    )

    # 最终仍未解决的 mismatch
    unresolved_df.to_parquet(
        OUTPUT_DIR
        / "universe_mismatch_unresolved.parquet",
        index=False,
    )

    # 保留原文件名，但现在表示 reconciliation 后的最终 mismatch
    unresolved_df.to_parquet(
        OUTPUT_DIR
        / "universe_mismatch_detail.parquet",
        index=False,
    )

    qa_df.to_csv(
        OUTPUT_DIR
        / "daily_core_qa_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    anomaly_df.to_parquet(
        OUTPUT_DIR
        / "core_data_anomalies.parquet",
        index=False,
    )

    field_coverage_df.to_csv(
        OUTPUT_DIR
        / "field_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 新增：
    # value-level completeness
    value_completeness_df.to_csv(
        OUTPUT_DIR
        / "field_value_completeness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 新增：
    # 每日字段值完整性
    daily_value_df.to_parquet(
        OUTPUT_DIR
        / "daily_field_value_completeness.parquet",
        index=False,
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

    # ========================================================
    # I. Metadata
    # ========================================================

    failed_days = int(
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

        "target_exchanges":
            TARGET_EXCHANGES,

        "stock_master_path":
            str(
                STOCK_MASTER_PATH
            ),

        "stock_code_history_path":
            str(
                STOCK_CODE_HISTORY_PATH
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
            failed_days,

        "revisions": [

            (
                "RAW_ONLY / DAILY_ONLY "
                "通过 exchange + normalized historical ticker "
                "执行 one-to-one identity reconciliation。"
            ),

            (
                "只有 mutual one-to-one 匹配才自动 reconcile；"
                "ambiguous identity mapping 不自动合并。"
            ),

            (
                "价格与 OHLC 异常只对 isOpen=1 "
                "的正常交易记录执行。"
            ),

            (
                "isOpen=0 的零价格占位仅作为 "
                "suspension diagnostic，不再判 HIGH。"
            ),

            (
                "新增 field_value_completeness_summary，"
                "分别报告全部、正常交易、停牌记录的值缺失率。"
            ),

            (
                "后续 Core QA 与 Value Completeness "
                "只针对 exact matched + reconciled PIT records。"
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
    # J. Console Summary
    # ========================================================

    print()
    print("=" * 72)
    print("Stage 2 Revised 完成")
    print("=" * 72)

    print(
        f"成功交易日："
        f"{(
            log_df['status'] == 'SUCCESS'
        ).sum():,}"
    )

    print(
        f"失败交易日："
        f"{failed_days:,}"
    )

    if not coverage_df.empty:

        exact_coverage = (
            coverage_df[
                "exact_matched_count"
            ].sum()
            /
            coverage_df[
                "expected_count"
            ].sum()
        )

        reconciled_coverage = (
            coverage_df[
                "effective_matched_count"
            ].sum()
            /
            coverage_df[
                "expected_count"
            ].sum()
        )

        print()
        print(
            f"Exact 加权覆盖率："
            f"{exact_coverage:.8f}"
        )

        print(
            f"Reconciled 加权覆盖率："
            f"{reconciled_coverage:.8f}"
        )

        print(
            f"Identity reconciliation："
            f"{coverage_df['identity_reconciled_count'].sum():,}"
        )

        print(
            f"未解决 RAW_ONLY："
            f"{coverage_df['unresolved_raw_only_count'].sum():,}"
        )

        print(
            f"未解决 DAILY_ONLY："
            f"{coverage_df['unresolved_daily_only_count'].sum():,}"
        )

    if not qa_df.empty:

        print()
        print(
            "核心行情 QA："
        )

        print(
            f"正常交易记录："
            f"{qa_df['active_stock_count'].sum():,}"
        )

        print(
            f"停牌记录："
            f"{qa_df['suspended_stock_count'].sum():,}"
        )

        print(
            f"停牌非正 OHLC 占位："
            f"{qa_df['suspended_nonpositive_ohlc_count'].sum():,}"
        )

        print(
            f"正常交易非正价格："
            f"{qa_df['active_invalid_price_count'].sum():,}"
        )

        print(
            f"正常交易 OHLC 逻辑异常："
            f"{qa_df['active_invalid_ohlc_count'].sum():,}"
        )

        print(
            f"正常交易缺 close："
            f"{qa_df['active_missing_close_count'].sum():,}"
        )

        print(
            f"Invalid AdjFactor："
            f"{qa_df['invalid_adj_factor_count'].sum():,}"
        )

        print(
            f"Closed But Traded："
            f"{qa_df['closed_but_traded_count'].sum():,}"
        )

    if not anomaly_df.empty:

        print()
        print(
            "异常类型："
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
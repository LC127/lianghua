from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Iterable

import pandas as pd


# ============================================================
# 0. 配置
# ============================================================

ROOT = Path(r"D:\lowfreq")

SECURITY_PATH = ROOT / "uqer_daily_info"

# 已确认：该文件中存在 isOpen 指标
TRADE_CALENDAR_PATH = ROOT / "trade_calendar" / "trade_date.csv"

OUTPUT_DIR = Path(r"D:\output\M1_day1\01_stage1_fixed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

START_DATE = pd.Timestamp(
    "2015-01-05"
)

END_DATE = pd.Timestamp(
    "2026-09-03"
)

# 已经通过 daily_temp3 确认
EXPECTED_TRADE_DAY_COUNT = 2837

# 2026-09-03 沪深上市股票数量
# 这里只用于 sanity check
EXPECTED_END_UNIVERSE_COUNT = 5215

# 第一版主研究只考虑沪深
INCLUDE_SSE = True
INCLUDE_SZSE = True
INCLUDE_BSE = False

SAVE_FULL_DAILY_UNIVERSE = True


# ============================================================
# 1. 字段别名
# ============================================================

COLUMN_ALIASES = {

    "security_id": [
        "secID",
        "SecID",
        "security_id",
        "securityID",
        "securityid",
        "instrumentID",
    ],

    "stock_code": [
        "ticker",
        "stock_code",
        "code",
        "secCode",
        "symbol",
        "ts_code",
    ],

    "stock_name": [
        "secShortName",
        "stock_name",
        "name",
        "shortName",
        "secName",
    ],

    "exchange": [
        "exchangeCD",
        "exchange",
        "exchange_code",
        "market",
    ],

    "list_date": [
        "listDate",
        "list_date",
        "ipoDate",
        "ipo_date",
        "listed_date",
    ],

    "delist_date": [
        "delistDate",
        "delist_date",
        "delisted_date",
    ],

    "security_type": [
        "secType",
        "secTypeCD",
        "security_type",
        "securityType",
        "type",
    ],

    "list_status": [
        "listStatusCD",
        "listStatus",
        "list_status",
        "status",
    ],
}


TRADE_DATE_ALIASES = [
    "trade_date",
    "tradeDate",
    "calendarDate",
    "date",
    "tradeDay",
    "TRADE_DATE",
]


IS_OPEN_ALIASES = [
    "isOpen",
    "is_open",
    "isTrade",
    "isTradeDay",
    "openFlag",
    "tradeFlag",
]


SUPPORTED_SUFFIX = {
    ".csv",
    ".parquet",
    ".pq",
    ".feather",
    ".pkl",
    ".pickle",
}


# ============================================================
# 2. 通用工具
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

    # 先直接匹配
    for alias in aliases:

        if alias in columns:

            return alias

    # 再忽略大小写和符号
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

    if suffix in {
        ".pkl",
        ".pickle",
    }:

        return pd.read_pickle(
            path
        )

    raise ValueError(
        f"不支持的文件格式：{path}"
    )


def find_data_files(
    path: Path,
) -> list[Path]:

    if path.is_file():

        return [path]

    if not path.exists():

        raise FileNotFoundError(
            f"路径不存在：{path}"
        )

    files = [

        p

        for p in path.rglob("*")

        if (
            p.is_file()
            and p.suffix.lower()
            in SUPPORTED_SUFFIX
        )
    ]

    if not files:

        raise FileNotFoundError(
            f"{path} 下没有找到支持的数据文件。"
        )

    return sorted(files)


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

    # 兼容 20150105
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
# 3. 读取 uqer_daily_info
# ============================================================

def load_uqer_daily_info(
) -> pd.DataFrame:

    files = find_data_files(
        SECURITY_PATH
    )

    print("=" * 72)
    print("读取 uqer_daily_info")
    print("=" * 72)

    dfs = []

    for file in files:

        try:

            df = read_data_file(
                file
            )

        except Exception as e:

            print(
                f"[WARNING] 跳过 {file}: {e}"
            )

            continue

        if df.empty:

            continue

        df = df.copy()

        df["_source_file"] = (
            str(file)
        )

        dfs.append(df)

    if not dfs:

        raise RuntimeError(
            "uqer_daily_info 中没有读取到有效数据。"
        )

    raw = pd.concat(
        dfs,
        ignore_index=True,
        sort=False,
    )

    print(
        f"总行数：{len(raw):,}"
    )

    print(
        f"总列数：{raw.shape[1]}"
    )

    return raw


# ============================================================
# 4. 自动识别证券字段
# ============================================================

def infer_column_mapping(
    df: pd.DataFrame,
) -> dict[str, str | None]:

    mapping = {}

    for (
        standard_name,
        aliases,
    ) in COLUMN_ALIASES.items():

        mapping[
            standard_name
        ] = find_column(
            df.columns,
            aliases,
        )

    print()
    print("=" * 72)
    print("证券字段映射")
    print("=" * 72)

    for key, value in mapping.items():

        print(
            f"{key:15s} <- {value}"
        )

    required = [
        "security_id",
        "stock_code",
        "exchange",
        "list_date",
    ]

    missing = [

        x

        for x in required

        if mapping[x] is None
    ]

    if missing:

        raise ValueError(
            "关键字段未识别："
            f"{missing}"
        )

    return mapping


# ============================================================
# 5. 股票代码标准化
# ============================================================

def clean_stock_code(
    series: pd.Series,
) -> pd.Series:

    """
    对正式股票代码：
        000001.XSHE -> 000001

    对 A19005、DY2xxx 等申报代码：
        保留原始字符串
    """

    x = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    six_digit = x.str.extract(
        r"(?<!\d)(\d{6})(?!\d)",
        expand=False,
    )

    result = x.copy()

    mask = six_digit.notna()

    result.loc[mask] = (
        six_digit.loc[mask]
    )

    return result


# ============================================================
# 6. 交易所标准化
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


def infer_exchange_from_code(
    stock_code: pd.Series,
) -> pd.Series:

    code = (
        stock_code
        .astype("string")
    )

    exchange = pd.Series(
        pd.NA,
        index=code.index,
        dtype="string",
    )

    # 上海 A 股
    exchange.loc[
        code.str.match(
            r"^(600|601|603|605|688)\d{3}$",
            na=False,
        )
    ] = "SSE"

    # 深圳 A 股
    exchange.loc[
        code.str.match(
            r"^(000|001|002|003|300|301)\d{3}$",
            na=False,
        )
    ] = "SZSE"

    # 北交所辅助识别
    exchange.loc[
        code.str.match(
            r"^(43|83|87|88|92)\d{4}$",
            na=False,
        )
    ] = "BSE"

    return exchange


# ============================================================
# 7. 判断是否属于标准 A 股代码
# ============================================================

def is_standard_a_share_code(
    stock_code: pd.Series,
    exchange: pd.Series,
) -> pd.Series:

    code = (
        stock_code
        .astype("string")
    )

    sse = (
        exchange.eq("SSE")
        &
        code.str.match(
            r"^(600|601|603|605|688)\d{3}$",
            na=False,
        )
    )

    szse = (
        exchange.eq("SZSE")
        &
        code.str.match(
            r"^(000|001|002|003|300|301)\d{3}$",
            na=False,
        )
    )

    bse = (
        exchange.eq("BSE")
        &
        code.str.match(
            r"^\d{6}$",
            na=False,
        )
    )

    return (
        sse
        | szse
        | bse
    )


# ============================================================
# 8. 板块识别
# ============================================================

def infer_board(
    exchange: pd.Series,
    code: pd.Series,
) -> pd.Series:

    board = pd.Series(
        "UNKNOWN",
        index=code.index,
        dtype="string",
    )

    code = (
        code
        .astype("string")
    )

    board.loc[
        exchange.eq("SSE")
        &
        code.str.startswith(
            "688",
            na=False,
        )
    ] = "STAR"

    board.loc[
        exchange.eq("SZSE")
        &
        code.str.startswith(
            ("300", "301"),
            na=False,
        )
    ] = "CHINEXT"

    board.loc[
        exchange.eq("BSE")
    ] = "BSE"

    board.loc[
        exchange.eq("SSE")
        &
        board.eq("UNKNOWN")
    ] = "SSE_MAIN"

    board.loc[
        exchange.eq("SZSE")
        &
        board.eq("UNKNOWN")
    ] = "SZSE_MAIN"

    return board


# ============================================================
# 9. 标准化证券记录
# ============================================================

def standardize_security_records(
    raw: pd.DataFrame,
    mapping: dict,
) -> pd.DataFrame:

    result = pd.DataFrame(
        index=raw.index
    )

    result["security_id"] = (
        raw[
            mapping["security_id"]
        ]
        .astype("string")
        .str.strip()
    )

    result["stock_code"] = (
        clean_stock_code(
            raw[
                mapping["stock_code"]
            ]
        )
    )

    if mapping[
        "stock_name"
    ] is not None:

        result["stock_name"] = (
            raw[
                mapping["stock_name"]
            ]
            .astype("string")
            .str.strip()
        )

    else:

        result["stock_name"] = pd.NA

    result["exchange"] = (
        normalize_exchange(
            raw[
                mapping["exchange"]
            ]
        )
    )

    # exchange fallback
    result["exchange"] = (
        result["exchange"]
        .fillna(
            infer_exchange_from_code(
                result["stock_code"]
            )
        )
    )

    result["list_date"] = (
        parse_date(
            raw[
                mapping["list_date"]
            ]
        )
    )

    if mapping[
        "delist_date"
    ] is not None:

        result["delist_date"] = (
            parse_date(
                raw[
                    mapping["delist_date"]
                ]
            )
        )

    else:

        result["delist_date"] = (
            pd.NaT
        )

    if mapping[
        "security_type"
    ] is not None:

        result[
            "security_type"
        ] = (
            raw[
                mapping["security_type"]
            ]
            .astype("string")
            .str.strip()
        )

    else:

        result[
            "security_type"
        ] = pd.NA

    if mapping[
        "list_status"
    ] is not None:

        result[
            "list_status"
        ] = (
            raw[
                mapping["list_status"]
            ]
            .astype("string")
            .str.strip()
        )

    else:

        result[
            "list_status"
        ] = pd.NA

    result["board"] = (
        infer_board(
            result["exchange"],
            result["stock_code"],
        )
    )

    result[
        "is_standard_a_share_code"
    ] = (
        is_standard_a_share_code(
            result["stock_code"],
            result["exchange"],
        )
    )

    result["_source_file"] = (
        raw["_source_file"]
        .astype("string")
    )

    return result


# ============================================================
# 10. 修复问题 2：
#     对缺 listDate 记录重新分类
# ============================================================

def classify_missing_list_date_records(
    records: pd.DataFrame,
):

    missing = records[
        records[
            "list_date"
        ].isna()
    ].copy()

    if missing.empty:

        summary = pd.DataFrame()

        return (
            missing,
            summary,
        )

    # 默认：
    # 非标准代码 + 无 listDate
    # 视为未上市 / 申报阶段记录
    missing[
        "missing_list_date_class"
    ] = (
        "PRELISTING_OR_NEVER_LISTED"
    )

    missing[
        "severity"
    ] = "INFO"

    # 标准 6 位 A 股代码但没有 listDate
    # 不直接判 HIGH
    # 留给 Stage 2 与 daily_temp3 交叉核查
    mask = missing[
        "is_standard_a_share_code"
    ]

    missing.loc[
        mask,
        "missing_list_date_class",
    ] = (
        "REVIEW_STANDARD_CODE_WITHOUT_LISTDATE"
    )

    missing.loc[
        mask,
        "severity",
    ] = "REVIEW"

    missing["note"] = (
        "缺少 listDate；当前不进入正式股票池。"
        "非标准代码通常属于申报/未上市记录；"
        "标准 A 股代码需在 Stage 2 "
        "与 daily_temp3 交叉检查是否曾实际交易。"
    )

    # 不主观解释 listStatusCD
    # 只做客观统计
    summary = (
        missing
        .groupby(
            "list_status",
            dropna=False,
        )
        .agg(
            record_count=(
                "security_id",
                "size",
            ),

            standard_a_share_code_count=(
                "is_standard_a_share_code",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "nonstandard_code_count"
    ] = (
        summary[
            "record_count"
        ]
        -
        summary[
            "standard_a_share_code_count"
        ]
    )

    return (
        missing,
        summary,
    )


# ============================================================
# 11. 正式筛选“曾实际上市”的 A 股记录
# ============================================================

def filter_listed_a_share_records(
    records: pd.DataFrame,
):

    allowed_exchange = []

    if INCLUDE_SSE:

        allowed_exchange.append(
            "SSE"
        )

    if INCLUDE_SZSE:

        allowed_exchange.append(
            "SZSE"
        )

    if INCLUDE_BSE:

        allowed_exchange.append(
            "BSE"
        )

    exchange_mask = (
        records[
            "exchange"
        ].isin(
            allowed_exchange
        )
    )

    valid_code_mask = (
        records[
            "is_standard_a_share_code"
        ]
    )

    valid_list_date_mask = (
        records[
            "list_date"
        ].notna()
    )

    # 正式进入主表：
    # 交易所正确
    # + 标准 A 股代码
    # + 有明确上市日期
    included = records[
        exchange_mask
        &
        valid_code_mask
        &
        valid_list_date_mask
    ].copy()

    # 其余保留用于核查
    excluded = records[
        exchange_mask
        &
        ~(
            valid_code_mask
            &
            valid_list_date_mask
        )
    ].copy()

    excluded[
        "exclude_reason"
    ] = "OTHER"

    excluded.loc[
        ~excluded[
            "is_standard_a_share_code"
        ],
        "exclude_reason",
    ] = (
        "NON_STANDARD_A_SHARE_CODE"
    )

    excluded.loc[
        excluded[
            "is_standard_a_share_code"
        ]
        &
        excluded[
            "list_date"
        ].isna(),
        "exclude_reason",
    ] = (
        "MISSING_LIST_DATE_REVIEW"
    )

    return (
        included,
        excluded,
    )


# ============================================================
# 12. 构造 stock master 和历史代码表
# ============================================================

def build_stock_master(
    listed_records: pd.DataFrame,
):

    stock_code_history = (
        listed_records[
            [
                "security_id",
                "stock_code",
                "stock_name",
                "exchange",
                "board",
                "list_date",
                "delist_date",
                "list_status",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "security_id",
                "list_date",
                "stock_code",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    stock_master = (
        listed_records
        .sort_values(
            [
                "security_id",
                "list_date",
            ]
        )
        .groupby(
            "security_id",
            as_index=False,
        )
        .agg(

            stock_code=(
                "stock_code",
                "last",
            ),

            stock_name=(
                "stock_name",
                "last",
            ),

            exchange=(
                "exchange",
                "last",
            ),

            board=(
                "board",
                "last",
            ),

            list_date=(
                "list_date",
                "min",
            ),

            delist_date=(
                "delist_date",
                "max",
            ),

            list_status=(
                "list_status",
                "last",
            ),

            code_count=(
                "stock_code",
                "nunique",
            ),

            name_count=(
                "stock_name",
                "nunique",
            ),
        )
    )

    return (
        stock_master,
        stock_code_history,
    )


# ============================================================
# 13. 股票主表 QA
# ============================================================

def quality_check_stock_master(
    stock_master,
    listed_records,
    missing_list_date_records,
):

    issues = []

    def add_issue(
        mask,
        issue_type,
        severity,
        note,
    ):

        if not mask.any():

            return

        tmp = stock_master.loc[
            mask,
            [
                "security_id",
                "stock_code",
                "stock_name",
                "exchange",
                "board",
                "list_date",
                "delist_date",
                "code_count",
            ],
        ].copy()

        tmp["issue_type"] = (
            issue_type
        )

        tmp["severity"] = (
            severity
        )

        tmp["note"] = note

        issues.append(tmp)

    # --------------------------------------------------------
    # HIGH：退市日期 <= 上市日期
    # --------------------------------------------------------

    invalid_date = (
        stock_master[
            "delist_date"
        ].notna()
        &
        (
            stock_master[
                "delist_date"
            ]
            <=
            stock_master[
                "list_date"
            ]
        )
    )

    add_issue(
        invalid_date,
        "INVALID_LIST_DELIST_ORDER",
        "HIGH",
        "delist_date <= list_date，需要核查。",
    )

    # --------------------------------------------------------
    # INFO：一个 secID 有多个代码
    # --------------------------------------------------------

    add_issue(
        stock_master[
            "code_count"
        ] > 1,
        "MULTIPLE_HISTORICAL_CODES",
        "INFO",
        "同一 secID 存在多个历史代码，"
        "通常属于代码变更，应保留映射。",
    )

    # --------------------------------------------------------
    # HIGH：同一交易所+代码对应多个 secID
    # --------------------------------------------------------

    id_count = (
        listed_records
        .groupby(
            [
                "exchange",
                "stock_code",
            ]
        )[
            "security_id"
        ]
        .nunique()
    )

    duplicate_keys = set(
        id_count[
            id_count > 1
        ].index
    )

    if duplicate_keys:

        duplicate_mask = (
            listed_records.apply(
                lambda row: (
                    row[
                        "exchange"
                    ],
                    row[
                        "stock_code"
                    ],
                )
                in duplicate_keys,
                axis=1,
            )
        )

        tmp = (
            listed_records.loc[
                duplicate_mask,
                [
                    "security_id",
                    "stock_code",
                    "stock_name",
                    "exchange",
                    "board",
                    "list_date",
                    "delist_date",
                ],
            ]
            .drop_duplicates()
            .copy()
        )

        tmp[
            "code_count"
        ] = pd.NA

        tmp[
            "issue_type"
        ] = (
            "SAME_CODE_MULTIPLE_SECURITY_IDS"
        )

        tmp[
            "severity"
        ] = "HIGH"

        tmp[
            "note"
        ] = (
            "同一交易所同一股票代码对应多个 secID，"
            "需要核查代码复用或数据映射。"
        )

        issues.append(tmp)

    # --------------------------------------------------------
    # REVIEW：
    # 标准股票代码但缺 listDate
    # --------------------------------------------------------

    if (
        not missing_list_date_records.empty
    ):

        review = (
            missing_list_date_records[
                missing_list_date_records[
                    "severity"
                ].eq(
                    "REVIEW"
                )
            ]
            .copy()
        )

        if not review.empty:

            tmp = pd.DataFrame(
                {
                    "security_id":
                        review[
                            "security_id"
                        ],

                    "stock_code":
                        review[
                            "stock_code"
                        ],

                    "stock_name":
                        review[
                            "stock_name"
                        ],

                    "exchange":
                        review[
                            "exchange"
                        ],

                    "board":
                        review[
                            "board"
                        ],

                    "list_date":
                        review[
                            "list_date"
                        ],

                    "delist_date":
                        review[
                            "delist_date"
                        ],

                    "code_count":
                        pd.NA,

                    "issue_type":
                        (
                            "STANDARD_CODE_WITHOUT_LISTDATE"
                        ),

                    "severity":
                        "REVIEW",

                    "note":
                        (
                            "标准 A 股代码但缺 listDate；"
                            "当前不进入 Raw Universe，"
                            "Stage 2 与 daily_temp3 核查。"
                        ),
                }
            )

            issues.append(tmp)

    if issues:

        issues_df = pd.concat(
            issues,
            ignore_index=True,
        )

    else:

        issues_df = pd.DataFrame()

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    high_count = 0
    review_count = 0

    if not issues_df.empty:

        high_count = (
            issues_df[
                "severity"
            ]
            .eq("HIGH")
            .sum()
        )

        review_count = (
            issues_df[
                "severity"
            ]
            .eq("REVIEW")
            .sum()
        )

    summary = pd.DataFrame(
        {

            "metric": [

                "unique_listed_security_count",

                "sse_count",

                "szse_count",

                "bse_count",

                "historically_delisted_count",

                "multiple_code_security_count",

                "missing_list_date_total_records",

                "missing_list_date_standard_code_records",

                "high_issue_rows",

                "review_issue_rows",
            ],

            "value": [

                stock_master[
                    "security_id"
                ].nunique(),

                stock_master.loc[
                    stock_master[
                        "exchange"
                    ].eq("SSE"),
                    "security_id",
                ].nunique(),

                stock_master.loc[
                    stock_master[
                        "exchange"
                    ].eq("SZSE"),
                    "security_id",
                ].nunique(),

                stock_master.loc[
                    stock_master[
                        "exchange"
                    ].eq("BSE"),
                    "security_id",
                ].nunique(),

                stock_master[
                    "delist_date"
                ].notna().sum(),

                (
                    stock_master[
                        "code_count"
                    ] > 1
                ).sum(),

                len(
                    missing_list_date_records
                ),

                (
                    int(
                        missing_list_date_records[
                            "is_standard_a_share_code"
                        ].sum()
                    )
                    if not
                    missing_list_date_records.empty
                    else 0
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

    return (
        summary,
        issues_df,
    )


# ============================================================
# 14. 修复问题 1：
#     正确处理 isOpen
# ============================================================

def normalize_is_open(
    series: pd.Series,
) -> pd.Series:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    # 数值 1
    result = (
        numeric.eq(1)
    )

    # 字符 TRUE/Y 等兼容
    text = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    truthy = {
        "1",
        "TRUE",
        "T",
        "Y",
        "YES",
        "OPEN",
        "TRADE",
    }

    result = (
        result
        |
        text.isin(
            truthy
        )
    )

    return result.fillna(
        False
    )


# ============================================================
# 15. 读取真实交易日
# ============================================================

def load_trade_calendar():

    df = read_data_file(
        TRADE_CALENDAR_PATH
    )

    print()
    print("=" * 72)
    print("交易日历字段")
    print("=" * 72)

    print(
        list(df.columns)
    )

    date_col = find_column(
        df.columns,
        TRADE_DATE_ALIASES,
    )

    is_open_col = find_column(
        df.columns,
        IS_OPEN_ALIASES,
    )

    if date_col is None:

        raise ValueError(
            "没有识别到交易日期字段。"
        )

    if is_open_col is None:

        raise ValueError(
            "没有识别到 isOpen 字段。"
        )

    calendar = pd.DataFrame(
        {

            "trade_date":
                parse_date(
                    df[
                        date_col
                    ]
                ),

            "is_open":
                normalize_is_open(
                    df[
                        is_open_col
                    ]
                ),
        }
    )

    # 删除非法日期
    calendar = calendar[
        calendar[
            "trade_date"
        ].notna()
    ].copy()

    # ========================================================
    # 核心修复：
    # 只保留 isOpen == 1
    # ========================================================

    calendar = calendar[
        calendar[
            "is_open"
        ]
    ].copy()

    # 研究区间
    calendar = calendar[
        (
            calendar[
                "trade_date"
            ] >= START_DATE
        )
        &
        (
            calendar[
                "trade_date"
            ] <= END_DATE
        )
    ].copy()

    calendar = (
        calendar
        .drop_duplicates(
            subset=[
                "trade_date"
            ]
        )
        .sort_values(
            "trade_date"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # QA 1：不允许周末
    # --------------------------------------------------------

    weekend_count = (
        (
            calendar[
                "trade_date"
            ].dt.weekday >= 5
        )
        .sum()
    )

    if weekend_count > 0:

        raise ValueError(
            f"isOpen=1 后仍存在 "
            f"{weekend_count} 个周末日期。"
        )

    # --------------------------------------------------------
    # QA 2：必须等于 2837 个交易日
    # --------------------------------------------------------

    if (
        len(calendar)
        !=
        EXPECTED_TRADE_DAY_COUNT
    ):

        raise ValueError(
            "交易日数量不正确："
            f"当前={len(calendar)}, "
            f"预期={EXPECTED_TRADE_DAY_COUNT}"
        )

    # --------------------------------------------------------
    # QA 3：起止日期
    # --------------------------------------------------------

    if (
        calendar[
            "trade_date"
        ].iloc[0]
        != START_DATE
    ):

        raise ValueError(
            "首个交易日与预期不一致。"
        )

    if (
        calendar[
            "trade_date"
        ].iloc[-1]
        != END_DATE
    ):

        raise ValueError(
            "最后交易日与预期不一致。"
        )

    # 保存实际使用的交易日历
    calendar.to_csv(
        OUTPUT_DIR
        / "trade_calendar_used.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"真实交易日数量："
        f"{len(calendar):,}"
    )

    print(
        f"起点："
        f"{calendar['trade_date'].min().date()}"
    )

    print(
        f"终点："
        f"{calendar['trade_date'].max().date()}"
    )

    return pd.DatetimeIndex(
        calendar[
            "trade_date"
        ]
    )


# ============================================================
# 16. 构造历史 Point-in-Time Raw Universe
# ============================================================

def build_raw_universe(
    stock_master,
    trade_dates,
):

    parts = []

    for row in stock_master.itertuples(
        index=False
    ):

        start = (
            row.list_date
        )

        if pd.isna(
            row.delist_date
        ):

            end = (
                END_DATE
                +
                pd.Timedelta(
                    days=1
                )
            )

        else:

            end = (
                row.delist_date
            )

        left = (
            trade_dates.searchsorted(
                start,
                side="left",
            )
        )

        # 当前定义：
        #
        # list_date <= t < delist_date
        right = (
            trade_dates.searchsorted(
                end,
                side="left",
            )
        )

        if left >= right:

            continue

        valid_dates = (
            trade_dates[
                left:right
            ]
        )

        tmp = pd.DataFrame(
            {

                "trade_date":
                    valid_dates,

                "security_id":
                    row.security_id,

                "stock_code":
                    row.stock_code,

                "stock_name":
                    row.stock_name,

                "exchange":
                    row.exchange,

                "board":
                    row.board,
            }
        )

        parts.append(
            tmp
        )

    if not parts:

        raise RuntimeError(
            "没有生成任何 Raw Universe。"
        )

    raw_universe = pd.concat(
        parts,
        ignore_index=True,
    )

    return raw_universe


# ============================================================
# 17. 每日 Raw Universe 统计
# ============================================================

def summarize_raw_universe(
    raw_universe,
):

    total = (
        raw_universe
        .groupby(
            "trade_date"
        )
        .agg(
            total_stock_count=(
                "security_id",
                "nunique",
            )
        )
        .reset_index()
    )

    by_exchange = (
        raw_universe
        .groupby(
            [
                "trade_date",
                "exchange",
            ]
        )
        .agg(
            stock_count=(
                "security_id",
                "nunique",
            )
        )
        .reset_index()
    )

    exchange_wide = (
        by_exchange
        .pivot(
            index="trade_date",
            columns="exchange",
            values="stock_count",
        )
        .reset_index()
    )

    summary = (
        total
        .merge(
            exchange_wide,
            on="trade_date",
            how="left",
        )
        .sort_values(
            "trade_date"
        )
        .reset_index(
            drop=True
        )
    )

    summary[
        "net_change"
    ] = (
        summary[
            "total_stock_count"
        ]
        .diff()
    )

    return summary


# ============================================================
# 18. 主程序
# ============================================================

def main():

    print("=" * 72)
    print("M1 Step 1 修正版")
    print("历史证券主表 + PIT Raw Universe")
    print("=" * 72)

    # --------------------------------------------------------
    # A. 读取证券数据
    # --------------------------------------------------------

    raw = (
        load_uqer_daily_info()
    )

    # --------------------------------------------------------
    # B. 字段识别
    # --------------------------------------------------------

    mapping = (
        infer_column_mapping(
            raw
        )
    )

    # --------------------------------------------------------
    # C. 标准化
    # --------------------------------------------------------

    records = (
        standardize_security_records(
            raw,
            mapping,
        )
    )

    # --------------------------------------------------------
    # D. 修复 missing listDate 分类
    # --------------------------------------------------------

    (
        missing_list_date_records,
        missing_list_date_status_summary,
    ) = (
        classify_missing_list_date_records(
            records
        )
    )

    # --------------------------------------------------------
    # E. 只保留真正进入股票池的记录
    # --------------------------------------------------------

    (
        listed_records,
        excluded_records,
    ) = (
        filter_listed_a_share_records(
            records
        )
    )

    # --------------------------------------------------------
    # F. 建立股票主表
    # --------------------------------------------------------

    (
        stock_master,
        stock_code_history,
    ) = (
        build_stock_master(
            listed_records
        )
    )

    # --------------------------------------------------------
    # G. QA
    # --------------------------------------------------------

    (
        quality_summary,
        issues,
    ) = (
        quality_check_stock_master(
            stock_master,
            listed_records,
            missing_list_date_records,
        )
    )

    # --------------------------------------------------------
    # H. 正确读取交易日
    # --------------------------------------------------------

    trade_dates = (
        load_trade_calendar()
    )

    # --------------------------------------------------------
    # I. 构造 Raw Universe
    # --------------------------------------------------------

    raw_universe = (
        build_raw_universe(
            stock_master,
            trade_dates,
        )
    )

    daily_summary = (
        summarize_raw_universe(
            raw_universe
        )
    )

    # --------------------------------------------------------
    # J. 最后一个交易日股票数
    # --------------------------------------------------------

    end_count = int(
        daily_summary.loc[
            daily_summary[
                "trade_date"
            ].eq(
                END_DATE
            ),
            "total_stock_count",
        ].iloc[0]
    )

    if (
        EXPECTED_END_UNIVERSE_COUNT
        is not None
        and
        end_count
        !=
        EXPECTED_END_UNIVERSE_COUNT
    ):

        print()
        print(
            "[WARNING] "
            f"{END_DATE.date()} "
            f"股票数={end_count:,}，"
            f"预期={EXPECTED_END_UNIVERSE_COUNT:,}"
        )

    # ========================================================
    # 19. 保存结果
    # ========================================================

    stock_master.to_parquet(
        OUTPUT_DIR
        / "stock_master.parquet",
        index=False,
    )

    stock_code_history.to_parquet(
        OUTPUT_DIR
        / "stock_code_history.parquet",
        index=False,
    )

    quality_summary.to_csv(
        OUTPUT_DIR
        / "stock_master_quality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    issues.to_csv(
        OUTPUT_DIR
        / "stock_master_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_list_date_records.to_csv(
        OUTPUT_DIR
        / "missing_list_date_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_list_date_status_summary.to_csv(
        OUTPUT_DIR
        / "missing_list_date_status_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    excluded_records.to_csv(
        OUTPUT_DIR
        / "excluded_security_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if SAVE_FULL_DAILY_UNIVERSE:

        raw_universe.to_parquet(
            OUTPUT_DIR
            / "raw_universe_by_date.parquet",
            index=False,
        )

    daily_summary.to_csv(
        OUTPUT_DIR
        / "raw_universe_daily_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 20. metadata
    # ========================================================

    metadata = {

        "research_start":
            str(
                START_DATE.date()
            ),

        "research_end":
            str(
                END_DATE.date()
            ),

        "trade_calendar_path":
            str(
                TRADE_CALENDAR_PATH
            ),

        "trade_calendar_rule":
            "isOpen == 1",

        "trade_day_count":
            int(
                len(
                    trade_dates
                )
            ),

        "expected_trade_day_count":
            EXPECTED_TRADE_DAY_COUNT,

        "include_sse":
            INCLUDE_SSE,

        "include_szse":
            INCLUDE_SZSE,

        "include_bse":
            INCLUDE_BSE,

        "security_count":
            int(
                stock_master[
                    "security_id"
                ].nunique()
            ),

        "historically_delisted_count":
            int(
                stock_master[
                    "delist_date"
                ].notna().sum()
            ),

        "missing_list_date_total_records":
            int(
                len(
                    missing_list_date_records
                )
            ),

        "missing_list_date_standard_code_records":
            (
                int(
                    missing_list_date_records[
                        "is_standard_a_share_code"
                    ].sum()
                )
                if not
                missing_list_date_records.empty
                else 0
            ),

        "raw_universe_rows":
            int(
                len(
                    raw_universe
                )
            ),

        "end_date_universe_count":
            end_count,

        "column_mapping":
            mapping,

        "important_notes": [

            "交易日历严格使用 isOpen == 1。",

            "缺少 listDate 的记录不再全部视为 HIGH。",

            (
                "非标准股票代码且缺 listDate 的记录"
                "主要视作 pre-listing / never-listed 记录，"
                "不进入正式 stock master。"
            ),

            (
                "标准 A 股代码但缺 listDate 的记录"
                "标记为 REVIEW，Stage 2 与 daily_temp3 核查。"
            ),

            (
                "正式 stock master 只包含："
                "目标交易所 + 标准 A 股代码 + listDate 非空。"
            ),

            (
                "Raw Universe 当前采用 "
                "list_date <= t < delist_date；"
                "仍需确认供应商 delistDate 的业务定义。"
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "stage1_metadata.json",
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
    # 21. 输出摘要
    # ========================================================

    if issues.empty:

        high_count = 0
        review_count = 0

    else:

        high_count = int(
            issues[
                "severity"
            ]
            .eq(
                "HIGH"
            )
            .sum()
        )

        review_count = int(
            issues[
                "severity"
            ]
            .eq(
                "REVIEW"
            )
            .sum()
        )

    print()
    print("=" * 72)
    print("Step 1 修正版完成")
    print("=" * 72)

    print(
        f"真实交易日数量："
        f"{len(trade_dates):,}"
    )

    print(
        f"正式历史证券主体数："
        f"{stock_master['security_id'].nunique():,}"
    )

    print(
        f"历史退市证券数："
        f"{stock_master['delist_date'].notna().sum():,}"
    )

    print(
        f"缺 listDate 原始记录数："
        f"{len(missing_list_date_records):,}"
    )

    standard_missing_count = (
        int(
            missing_list_date_records[
                "is_standard_a_share_code"
            ].sum()
        )
        if not
        missing_list_date_records.empty
        else 0
    )

    print(
        "其中标准 A 股代码、需 REVIEW："
        f"{standard_missing_count:,}"
    )

    print(
        f"HIGH issue rows："
        f"{high_count:,}"
    )

    print(
        f"REVIEW issue rows："
        f"{review_count:,}"
    )

    print(
        f"{END_DATE.date()} Raw Universe："
        f"{end_count:,}"
    )

    print(
        f"Raw Universe 总行数："
        f"{len(raw_universe):,}"
    )

    print()
    print(
        f"输出目录：{OUTPUT_DIR}"
    )

    print()
    print("主要输出：")

    print(
        "1. stock_master.parquet"
    )

    print(
        "2. stock_code_history.parquet"
    )

    print(
        "3. stock_master_quality_summary.csv"
    )

    print(
        "4. stock_master_issues.csv"
    )

    print(
        "5. missing_list_date_records.csv"
    )

    print(
        "6. missing_list_date_status_summary.csv"
    )

    print(
        "7. excluded_security_records.csv"
    )

    print(
        "8. trade_calendar_used.csv"
    )

    print(
        "9. raw_universe_by_date.parquet"
    )

    print(
        "10. raw_universe_daily_summary.csv"
    )

    print(
        "11. stage1_metadata.json"
    )

    if high_count == 0:

        print()
        print(
            "[PASS CANDIDATE] "
            "没有 HIGH 级证券主表问题。"
        )

    else:

        print()
        print(
            "[REVIEW REQUIRED] "
            "仍存在 HIGH 级问题，"
            "请检查 stock_master_issues.csv。"
        )


if __name__ == "__main__":

    main()
from __future__ import annotations

from pathlib import Path
import json
import re
from itertools import combinations
from typing import Iterable

import pandas as pd


# ============================================================
# 0. 配置
# ============================================================

ROOT = Path(r"D:\lowfreq")

# 股票证券主信息
SECURITY_PATH = ROOT / "uqer_daily_info"

# 用户已确认该文件中存在 isOpen
TRADE_CALENDAR_PATH = ROOT / "trade_calendar" / "trade_date.csv"


# 日频行情文件目录
DAILY_TEMP3_DIR = ROOT / "daily_temp3"

# 如果自动找不到 2026-09-03 文件，
# 可以直接把具体文件路径填在这里，例如：
#
# END_DATE_DAILY_FILE = Path(
#     r"D:\lowfreq\daily_temp3\2026-09-03.csv"
# )
#
END_DATE_DAILY_FILE: Path | None = None

OUTPUT_DIR = Path(r"D:\output\M1_day1\01_stage1_final")

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

# 已经由 daily_temp3 验证
EXPECTED_TRADE_DAY_COUNT = 2837

# 此前数据盘点得到：
# 2026-09-03 沪深上市股票理论数量约为 5215
EXPECTED_END_UNIVERSE_COUNT = 5215


# ============================================================
# 2. 第一版研究范围
# ============================================================

INCLUDE_SSE = True
INCLUDE_SZSE = True
INCLUDE_BSE = False

SAVE_FULL_DAILY_UNIVERSE = True


# ============================================================
# 3. uqer_daily_info 字段别名
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


# ============================================================
# 4. daily_temp3 字段别名
# ============================================================

DAILY_ALIASES = {

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
# 5. 基础函数
# ============================================================

def normalize_colname(x: str) -> str:

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

    # 直接匹配
    for alias in aliases:

        if alias in columns:

            return alias

    # 忽略大小写、下划线等
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
        f"不支持的数据格式：{path}"
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

    # 兼容 YYYYMMDD
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
# 6. 目标交易所
# ============================================================

def get_target_exchanges() -> list[str]:

    target = []

    if INCLUDE_SSE:
        target.append("SSE")

    if INCLUDE_SZSE:
        target.append("SZSE")

    if INCLUDE_BSE:
        target.append("BSE")

    return target


TARGET_EXCHANGES = get_target_exchanges()


# ============================================================
# 7. 读取证券信息
# ============================================================

def load_uqer_daily_info(
) -> pd.DataFrame:

    files = find_data_files(
        SECURITY_PATH
    )

    print("=" * 72)
    print("读取 uqer_daily_info")
    print("=" * 72)

    parts = []

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

        parts.append(df)

    if not parts:

        raise RuntimeError(
            "没有成功读取任何证券主信息。"
        )

    raw = pd.concat(
        parts,
        ignore_index=True,
        sort=False,
    )

    print(
        f"原始记录数：{len(raw):,}"
    )

    print(
        f"原始字段数：{raw.shape[1]}"
    )

    return raw


# ============================================================
# 8. 字段识别
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
            f"以下关键字段未识别：{missing}"
        )

    return mapping


# ============================================================
# 9. 股票代码
# ============================================================

def clean_stock_code(
    series: pd.Series,
) -> pd.Series:

    x = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # 有标准 6 位数字时提取数字
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
# 10. 交易所标准化
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
    code: pd.Series,
) -> pd.Series:

    code = (
        code
        .astype("string")
    )

    result = pd.Series(
        pd.NA,
        index=code.index,
        dtype="string",
    )

    # 上海 A 股：
    # 60xxxx / 68xxxx
    result.loc[
        code.str.fullmatch(
            r"(60|68)\d{4}",
            na=False,
        )
    ] = "SSE"

    # 深圳 A 股：
    # 00xxxx / 30xxxx
    result.loc[
        code.str.fullmatch(
            r"(00|30)\d{4}",
            na=False,
        )
    ] = "SZSE"

    # 北交所辅助规则
    result.loc[
        code.str.fullmatch(
            r"(43|83|87|88|92)\d{4}",
            na=False,
        )
    ] = "BSE"

    return result


# ============================================================
# 11. A股代码判断
#
# 修复：
# 上海不再只列 600/601/603/605/688，
# 而使用 60xxxx / 68xxxx，
# 从而包含 689xxx。
#
# 深圳使用 00xxxx / 30xxxx。
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
        code.str.fullmatch(
            r"(60|68)\d{4}",
            na=False,
        )
    )

    szse = (
        exchange.eq("SZSE")
        &
        code.str.fullmatch(
            r"(00|30)\d{4}",
            na=False,
        )
    )

    bse = (
        exchange.eq("BSE")
        &
        code.str.fullmatch(
            r"\d{6}",
            na=False,
        )
    )

    return (
        sse
        |
        szse
        |
        bse
    )


# ============================================================
# 12. 板块判断
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

    # 688 / 689 归入科创板相关证券
    board.loc[
        exchange.eq("SSE")
        &
        code.str.match(
            r"^(688|689)",
            na=False,
        )
    ] = "STAR"

    # 30xxxx 为创业板
    board.loc[
        exchange.eq("SZSE")
        &
        code.str.startswith(
            "30",
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
# 13. 标准化证券记录
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

    if mapping["stock_name"] is not None:

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

    if mapping["delist_date"] is not None:

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

    if mapping["list_status"] is not None:

        result["list_status"] = (
            raw[
                mapping["list_status"]
            ]
            .astype("string")
            .str.strip()
        )

    else:

        result["list_status"] = pd.NA

    if mapping["security_type"] is not None:

        result["security_type"] = (
            raw[
                mapping["security_type"]
            ]
            .astype("string")
            .str.strip()
        )

    else:

        result["security_type"] = pd.NA

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
# 14. 修复问题 1：
# REVIEW 仅统计当前主研究交易所
# ============================================================

def split_target_records(
    records: pd.DataFrame,
):

    main_records = records[
        records[
            "exchange"
        ].isin(
            TARGET_EXCHANGES
        )
    ].copy()

    non_main_records = records[
        ~records[
            "exchange"
        ].isin(
            TARGET_EXCHANGES
        )
    ].copy()

    return (
        main_records,
        non_main_records,
    )


# ============================================================
# 15. missing listDate 分类
#
# 注意：
# 这里只对 main_records 分析，
# 因此北交所不会进入沪深 REVIEW 统计。
# ============================================================

def classify_missing_list_date_records(
    main_records: pd.DataFrame,
):

    missing = main_records[
        main_records[
            "list_date"
        ].isna()
    ].copy()

    if missing.empty:

        return (
            missing,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    missing[
        "missing_list_date_class"
    ] = (
        "PRELISTING_OR_NEVER_LISTED"
    )

    missing[
        "severity"
    ] = "INFO"

    standard_mask = (
        missing[
            "is_standard_a_share_code"
        ]
    )

    missing.loc[
        standard_mask,
        "missing_list_date_class",
    ] = (
        "STANDARD_A_CODE_WITHOUT_LISTDATE"
    )

    missing.loc[
        standard_mask,
        "severity",
    ] = "REVIEW"

    missing["note"] = (
        "缺少 listDate。"
        "非标准代码记录视作 pre-listing / never-listed；"
        "标准 A 股代码记录进入 REVIEW，"
        "需与 daily_temp3 核查是否真实交易。"
    )

    # --------------------------------------------------------
    # Record-level summary
    # --------------------------------------------------------

    record_summary = (
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

            standard_code_record_count=(
                "is_standard_a_share_code",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Unique security-level summary
    # --------------------------------------------------------

    unique_missing = (
        missing
        .drop_duplicates(
            subset=[
                "security_id"
            ]
        )
        .copy()
    )

    unique_summary = (
        unique_missing
        .groupby(
            "list_status",
            dropna=False,
        )
        .agg(

            unique_security_count=(
                "security_id",
                "nunique",
            ),

            standard_code_security_count=(
                "is_standard_a_share_code",
                "sum",
            ),
        )
        .reset_index()
    )

    return (
        missing,
        record_summary,
        unique_summary,
    )


# ============================================================
# 16. 正式进入 stock master 的记录
# ============================================================

def filter_listed_records(
    main_records: pd.DataFrame,
):

    include_mask = (
        main_records[
            "is_standard_a_share_code"
        ]
        &
        main_records[
            "list_date"
        ].notna()
    )

    listed = (
        main_records[
            include_mask
        ]
        .copy()
    )

    excluded = (
        main_records[
            ~include_mask
        ]
        .copy()
    )

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
        listed,
        excluded,
    )


# ============================================================
# 17. 建立 stock master
# ============================================================

def build_stock_master(
    listed_records: pd.DataFrame,
):

    code_history = (
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
        code_history,
    )


# ============================================================
# 18. 判断两个证券有效区间是否重叠
#
# 使用：
# [list_date, delist_date)
#
# 无退市日期视为持续有效。
# ============================================================

def intervals_overlap(
    list_date_1,
    delist_date_1,
    list_date_2,
    delist_date_2,
) -> bool:

    far_future = pd.Timestamp(
        "2262-01-01"
    )

    end_1 = (
        delist_date_1
        if pd.notna(
            delist_date_1
        )
        else far_future
    )

    end_2 = (
        delist_date_2
        if pd.notna(
            delist_date_2
        )
        else far_future
    )

    overlap_start = max(
        list_date_1,
        list_date_2,
    )

    overlap_end = min(
        end_1,
        end_2,
    )

    return (
        overlap_start
        <
        overlap_end
    )


# ============================================================
# 19. 修复问题 2：
# 同代码多 secID 按时间区间判断
# ============================================================

def analyze_code_reuse(
    stock_master: pd.DataFrame,
):

    rows = []

    grouped = (
        stock_master
        .groupby(
            [
                "exchange",
                "stock_code",
            ],
            dropna=False,
        )
    )

    for (
        exchange,
        stock_code,
    ), group in grouped:

        if (
            group[
                "security_id"
            ].nunique()
            <= 1
        ):

            continue

        group = (
            group
            .drop_duplicates(
                subset=[
                    "security_id"
                ]
            )
        )

        for (
            idx1,
            idx2,
        ) in combinations(
            group.index,
            2,
        ):

            a = (
                group.loc[
                    idx1
                ]
            )

            b = (
                group.loc[
                    idx2
                ]
            )

            overlap = (
                intervals_overlap(

                    a[
                        "list_date"
                    ],

                    a[
                        "delist_date"
                    ],

                    b[
                        "list_date"
                    ],

                    b[
                        "delist_date"
                    ],
                )
            )

            if overlap:

                issue_type = (
                    "OVERLAPPING_CODE_REUSE"
                )

                severity = "HIGH"

                note = (
                    "同一交易所同一代码对应不同 secID，"
                    "且上市有效区间存在重叠，需要人工核查。"
                )

            else:

                issue_type = (
                    "HISTORICAL_CODE_REUSE"
                )

                severity = "INFO"

                note = (
                    "同一代码在不同历史阶段对应不同 secID，"
                    "有效期不重叠，属于历史代码复用。"
                )

            rows.append(
                {

                    "exchange":
                        exchange,

                    "stock_code":
                        stock_code,

                    "security_id_1":
                        a[
                            "security_id"
                        ],

                    "stock_name_1":
                        a[
                            "stock_name"
                        ],

                    "list_date_1":
                        a[
                            "list_date"
                        ],

                    "delist_date_1":
                        a[
                            "delist_date"
                        ],

                    "security_id_2":
                        b[
                            "security_id"
                        ],

                    "stock_name_2":
                        b[
                            "stock_name"
                        ],

                    "list_date_2":
                        b[
                            "list_date"
                        ],

                    "delist_date_2":
                        b[
                            "delist_date"
                        ],

                    "interval_overlap":
                        overlap,

                    "issue_type":
                        issue_type,

                    "severity":
                        severity,

                    "note":
                        note,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 20. 建立统一 issue 表
# ============================================================

def build_issue_table(
    stock_master: pd.DataFrame,
    missing_records: pd.DataFrame,
    code_reuse: pd.DataFrame,
):

    issues = []

    # --------------------------------------------------------
    # A. 日期顺序异常
    # --------------------------------------------------------

    invalid = (
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

    for row in (
        stock_master[
            invalid
        ].itertuples(
            index=False
        )
    ):

        issues.append(
            {

                "issue_type":
                    "INVALID_LIST_DELIST_ORDER",

                "severity":
                    "HIGH",

                "security_id":
                    row.security_id,

                "security_id_2":
                    pd.NA,

                "stock_code":
                    row.stock_code,

                "stock_name":
                    row.stock_name,

                "exchange":
                    row.exchange,

                "list_date":
                    row.list_date,

                "delist_date":
                    row.delist_date,

                "note":
                    "delist_date <= list_date。",
            }
        )

    # --------------------------------------------------------
    # B. 一个 secID 多历史代码
    # --------------------------------------------------------

    multi_code = (
        stock_master[
            "code_count"
        ]
        >
        1
    )

    for row in (
        stock_master[
            multi_code
        ].itertuples(
            index=False
        )
    ):

        issues.append(
            {

                "issue_type":
                    "MULTIPLE_HISTORICAL_CODES",

                "severity":
                    "INFO",

                "security_id":
                    row.security_id,

                "security_id_2":
                    pd.NA,

                "stock_code":
                    row.stock_code,

                "stock_name":
                    row.stock_name,

                "exchange":
                    row.exchange,

                "list_date":
                    row.list_date,

                "delist_date":
                    row.delist_date,

                "note":
                    "同一 secID 存在多个历史代码，保留历史代码映射。",
            }
        )

    # --------------------------------------------------------
    # C. 沪深标准代码但缺 listDate
    # --------------------------------------------------------

    if not missing_records.empty:

        review = (
            missing_records[
                missing_records[
                    "severity"
                ].eq(
                    "REVIEW"
                )
            ]
            .drop_duplicates(
                subset=[
                    "security_id"
                ]
            )
        )

        for row in (
            review.itertuples(
                index=False
            )
        ):

            issues.append(
                {

                    "issue_type":
                        "STANDARD_A_CODE_WITHOUT_LISTDATE",

                    "severity":
                        "REVIEW",

                    "security_id":
                        row.security_id,

                    "security_id_2":
                        pd.NA,

                    "stock_code":
                        row.stock_code,

                    "stock_name":
                        row.stock_name,

                    "exchange":
                        row.exchange,

                    "list_date":
                        row.list_date,

                    "delist_date":
                        row.delist_date,

                    "note":
                        (
                            "标准沪深 A 股代码但缺 listDate；"
                            "需与 daily_temp3 核查。"
                        ),
                }
            )

    # --------------------------------------------------------
    # D. 同代码多 secID
    # --------------------------------------------------------

    if not code_reuse.empty:

        for row in (
            code_reuse.itertuples(
                index=False
            )
        ):

            issues.append(
                {

                    "issue_type":
                        row.issue_type,

                    "severity":
                        row.severity,

                    "security_id":
                        row.security_id_1,

                    "security_id_2":
                        row.security_id_2,

                    "stock_code":
                        row.stock_code,

                    "stock_name":
                        (
                            f"{row.stock_name_1}"
                            f" / "
                            f"{row.stock_name_2}"
                        ),

                    "exchange":
                        row.exchange,

                    "list_date":
                        row.list_date_1,

                    "delist_date":
                        row.delist_date_1,

                    "note":
                        row.note,
                }
            )

    return pd.DataFrame(
        issues
    )


# ============================================================
# 21. 交易日历 isOpen
# ============================================================

def normalize_is_open(
    series: pd.Series,
) -> pd.Series:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = (
        numeric.eq(1)
    )

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
        "YES",
        "Y",
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
# 22. 读取真实交易日
# ============================================================

def load_trade_calendar(
) -> pd.DatetimeIndex:

    df = read_data_file(
        TRADE_CALENDAR_PATH
    )

    print()
    print("=" * 72)
    print("读取交易日历")
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
            "没有找到交易日期列。"
        )

    if is_open_col is None:

        raise ValueError(
            "没有找到 isOpen 列。"
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

    calendar = calendar[
        calendar[
            "trade_date"
        ].notna()
    ].copy()

    # 关键：
    # 只使用真正交易日
    calendar = calendar[
        calendar[
            "is_open"
        ]
    ].copy()

    calendar = calendar[
        (
            calendar[
                "trade_date"
            ]
            >=
            START_DATE
        )
        &
        (
            calendar[
                "trade_date"
            ]
            <=
            END_DATE
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
    # QA
    # --------------------------------------------------------

    weekend_count = (
        (
            calendar[
                "trade_date"
            ].dt.weekday
            >=
            5
        )
        .sum()
    )

    if weekend_count != 0:

        raise ValueError(
            f"isOpen=1 中仍存在 {weekend_count} 个周末日期。"
        )

    if (
        len(calendar)
        !=
        EXPECTED_TRADE_DAY_COUNT
    ):

        raise ValueError(
            "交易日数量异常："
            f"实际={len(calendar)}, "
            f"预期={EXPECTED_TRADE_DAY_COUNT}"
        )

    if (
        calendar[
            "trade_date"
        ].iloc[0]
        !=
        START_DATE
    ):

        raise ValueError(
            "起始交易日不符合预期。"
        )

    if (
        calendar[
            "trade_date"
        ].iloc[-1]
        !=
        END_DATE
    ):

        raise ValueError(
            "结束交易日不符合预期。"
        )

    calendar.to_csv(
        OUTPUT_DIR
        / "trade_calendar_used.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return pd.DatetimeIndex(
        calendar[
            "trade_date"
        ]
    )


# ============================================================
# 23. PIT Raw Universe
# ============================================================

def build_raw_universe(
    stock_master: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
):

    parts = []

    for row in (
        stock_master.itertuples(
            index=False
        )
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

        parts.append(
            pd.DataFrame(
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
        )

    if not parts:

        raise RuntimeError(
            "Raw Universe 为空。"
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ============================================================
# 24. Raw Universe 每日汇总
# ============================================================

def summarize_raw_universe(
    raw_universe: pd.DataFrame,
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

    result = (
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

    result[
        "net_change"
    ] = (
        result[
            "total_stock_count"
        ]
        .diff()
    )

    return result


# ============================================================
# 25. 修复问题 3：
# 自动寻找 END_DATE 对应 daily_temp3 文件
# ============================================================

def find_end_date_daily_file(
) -> Path:

    if (
        END_DATE_DAILY_FILE
        is not None
    ):

        if not (
            END_DATE_DAILY_FILE.exists()
        ):

            raise FileNotFoundError(
                f"指定文件不存在：{END_DATE_DAILY_FILE}"
            )

        return END_DATE_DAILY_FILE

    date_patterns = [

        END_DATE.strftime(
            "%Y-%m-%d"
        ),

        END_DATE.strftime(
            "%Y%m%d"
        ),

        END_DATE.strftime(
            "%Y_%m_%d"
        ),
    ]

    candidates = []

    for file in (
        DAILY_TEMP3_DIR.rglob("*")
    ):

        if not file.is_file():

            continue

        if (
            file.suffix.lower()
            not in
            SUPPORTED_SUFFIX
        ):

            continue

        filename = (
            file.name
        )

        if any(
            token in filename
            for token
            in date_patterns
        ):

            candidates.append(
                file
            )

    if not candidates:

        raise FileNotFoundError(
            "未能自动找到 "
            f"{END_DATE.date()} 对应的 daily_temp3 文件。\n"
            "请直接设置 END_DATE_DAILY_FILE。"
        )

    if len(candidates) > 1:

        print()
        print(
            "[WARNING] 找到多个候选文件："
        )

        for file in candidates:

            print(
                file
            )

        print(
            "默认使用第一个候选文件。"
        )

    return sorted(
        candidates
    )[0]


# ============================================================
# 26. 标准化最新日行情
# ============================================================

def load_end_date_daily(
) -> pd.DataFrame:

    file = (
        find_end_date_daily_file()
    )

    print()
    print("=" * 72)
    print("最新日期 daily_temp3")
    print("=" * 72)

    print(
        f"文件：{file}"
    )

    df = read_data_file(
        file
    )

    mapping = {}

    for key, aliases in (
        DAILY_ALIASES.items()
    ):

        mapping[key] = (
            find_column(
                df.columns,
                aliases,
            )
        )

    if (
        mapping[
            "stock_code"
        ]
        is None
    ):

        raise ValueError(
            "daily_temp3 中没有识别到股票代码。"
        )

    result = pd.DataFrame(
        index=df.index
    )

    if (
        mapping[
            "security_id"
        ]
        is not None
    ):

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

    else:

        result[
            "security_id"
        ] = pd.NA

    result[
        "stock_code"
    ] = (
        clean_stock_code(
            df[
                mapping[
                    "stock_code"
                ]
            ]
        )
    )

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
        ] = (
            infer_exchange_from_code(
                result[
                    "stock_code"
                ]
            )
        )

    # 如果是多日期文件
    if (
        mapping[
            "trade_date"
        ]
        is not None
    ):

        trade_date = (
            parse_date(
                df[
                    mapping[
                        "trade_date"
                    ]
                ]
            )
        )

        result[
            "trade_date"
        ] = (
            trade_date
        )

        result = (
            result[
                result[
                    "trade_date"
                ].eq(
                    END_DATE
                )
            ]
            .copy()
        )

    else:

        result[
            "trade_date"
        ] = END_DATE

    # 只看当前研究交易所
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

    # 记录是否为 6 位数字代码
    result[
        "is_six_digit_numeric"
    ] = (
        result[
            "stock_code"
        ]
        .astype("string")
        .str.fullmatch(
            r"\d{6}",
            na=False,
        )
    )

    result[
        "is_standard_a_share_code"
    ] = (
        is_standard_a_share_code(
            result[
                "stock_code"
            ],
            result[
                "exchange"
            ],
        )
    )

    return (
        result
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 27. END_DATE 横截面差异核验
# ============================================================

def crosscheck_end_date_universe(
    raw_universe: pd.DataFrame,
    all_main_records: pd.DataFrame,
):

    daily = (
        load_end_date_daily()
    )

    raw_end = (
        raw_universe[
            raw_universe[
                "trade_date"
            ].eq(
                END_DATE
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # 先检查 daily_temp3 中
    # 哪些 6 位数字代码未满足当前 A 股规则
    # --------------------------------------------------------

    code_rule_review = (
        daily[
            daily[
                "is_six_digit_numeric"
            ]
            &
            ~daily[
                "is_standard_a_share_code"
            ]
        ]
        .copy()
    )

    # --------------------------------------------------------
    # A. 优先按 secID 比较
    # --------------------------------------------------------

    valid_daily_secids = (
        daily[
            "security_id"
        ]
        .dropna()
        .astype("string")
        .unique()
    )

    raw_secids = (
        raw_end[
            "security_id"
        ]
        .dropna()
        .astype("string")
        .unique()
    )

    use_secid = (
        len(
            valid_daily_secids
        )
        > 0
    )

    if use_secid:

        daily_set = set(
            valid_daily_secids
        )

        raw_set = set(
            raw_secids
        )

        daily_only_ids = (
            daily_set
            -
            raw_set
        )

        raw_only_ids = (
            raw_set
            -
            daily_set
        )

        daily_only = (
            daily[
                daily[
                    "security_id"
                ].isin(
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

        raw_only = (
            raw_end[
                raw_end[
                    "security_id"
                ].isin(
                    raw_only_ids
                )
            ]
            .drop_duplicates(
                subset=[
                    "security_id"
                ]
            )
            .copy()
        )

        # 给 daily_only 补充证券主信息
        master_lookup = (
            all_main_records
            .sort_values(
                [
                    "security_id",
                    "list_date",
                ],
                na_position="last",
            )
            .drop_duplicates(
                subset=[
                    "security_id"
                ],
                keep="last",
            )
            [
                [
                    "security_id",
                    "list_date",
                    "delist_date",
                    "list_status",
                    "is_standard_a_share_code",
                ]
            ]
        )

        daily_only = (
            daily_only
            .merge(
                master_lookup,
                on="security_id",
                how="left",
                suffixes=(
                    "",
                    "_master",
                ),
            )
        )

    else:

        # ----------------------------------------------------
        # B. 若 daily_temp3 没有 secID，
        #    改用 exchange + stock_code
        # ----------------------------------------------------

        daily[
            "compare_key"
        ] = (
            daily[
                "exchange"
            ].astype("string")
            +
            "_"
            +
            daily[
                "stock_code"
            ].astype("string")
        )

        raw_end[
            "compare_key"
        ] = (
            raw_end[
                "exchange"
            ].astype("string")
            +
            "_"
            +
            raw_end[
                "stock_code"
            ].astype("string")
        )

        daily_set = set(
            daily[
                "compare_key"
            ]
        )

        raw_set = set(
            raw_end[
                "compare_key"
            ]
        )

        daily_only_keys = (
            daily_set
            -
            raw_set
        )

        raw_only_keys = (
            raw_set
            -
            daily_set
        )

        daily_only = (
            daily[
                daily[
                    "compare_key"
                ].isin(
                    daily_only_keys
                )
            ]
            .copy()
        )

        raw_only = (
            raw_end[
                raw_end[
                    "compare_key"
                ].isin(
                    raw_only_keys
                )
            ]
            .copy()
        )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    daily_only.to_csv(
        OUTPUT_DIR
        / "end_date_daily_only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_only.to_csv(
        OUTPUT_DIR
        / "end_date_raw_only.csv",
        index=False,
        encoding="utf-8-sig",
    )

    code_rule_review.to_csv(
        OUTPUT_DIR
        / "end_date_code_rule_review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        {

            "metric": [

                "end_date",

                "raw_universe_count",

                "daily_temp3_target_exchange_count",

                "daily_temp3_standard_a_code_count",

                "daily_only_count",

                "raw_only_count",

                "daily_six_digit_but_unrecognized_count",

                "comparison_method",
            ],

            "value": [

                str(
                    END_DATE.date()
                ),

                raw_end[
                    "security_id"
                ].nunique(),

                daily[
                    "security_id"
                ].nunique()
                if daily[
                    "security_id"
                ].notna().any()
                else len(
                    daily
                ),

                daily.loc[
                    daily[
                        "is_standard_a_share_code"
                    ],
                    "security_id",
                ].nunique()
                if daily[
                    "security_id"
                ].notna().any()
                else (
                    daily[
                        "is_standard_a_share_code"
                    ].sum()
                ),

                len(
                    daily_only
                ),

                len(
                    raw_only
                ),

                len(
                    code_rule_review
                ),

                (
                    "security_id"
                    if use_secid
                    else "exchange+stock_code"
                ),
            ],
        }
    )

    summary.to_csv(
        OUTPUT_DIR
        / "end_date_universe_crosscheck_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        summary,
        daily_only,
        raw_only,
        code_rule_review,
    )


# ============================================================
# 28. Quality Summary
# ============================================================

def build_quality_summary(
    stock_master: pd.DataFrame,
    missing_records: pd.DataFrame,
    issues: pd.DataFrame,
    code_reuse: pd.DataFrame,
):

    missing_unique = (
        missing_records[
            "security_id"
        ].nunique()
        if not
        missing_records.empty
        else 0
    )

    missing_standard_unique = (
        missing_records.loc[
            missing_records[
                "is_standard_a_share_code"
            ],
            "security_id",
        ].nunique()
        if not
        missing_records.empty
        else 0
    )

    high_count = (
        issues[
            "severity"
        ].eq("HIGH").sum()
        if not
        issues.empty
        else 0
    )

    review_count = (
        issues[
            "severity"
        ].eq("REVIEW").sum()
        if not
        issues.empty
        else 0
    )

    historical_reuse_count = (
        code_reuse[
            "issue_type"
        ]
        .eq(
            "HISTORICAL_CODE_REUSE"
        )
        .sum()
        if not
        code_reuse.empty
        else 0
    )

    overlapping_reuse_count = (
        code_reuse[
            "issue_type"
        ]
        .eq(
            "OVERLAPPING_CODE_REUSE"
        )
        .sum()
        if not
        code_reuse.empty
        else 0
    )

    return pd.DataFrame(
        {

            "metric": [

                "unique_listed_security_count",

                "sse_count",

                "szse_count",

                "bse_count",

                "historically_delisted_count",

                "multiple_code_security_count",

                "missing_list_date_record_count_main_sample",

                "missing_list_date_unique_security_count_main_sample",

                "missing_list_date_standard_code_unique_security_count",

                "historical_code_reuse_pair_count",

                "overlapping_code_reuse_pair_count",

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
                    ]
                    >
                    1
                ).sum(),

                len(
                    missing_records
                ),

                missing_unique,

                missing_standard_unique,

                historical_reuse_count,

                overlapping_reuse_count,

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
# 29. 主程序
# ============================================================

def main():

    print("=" * 72)
    print("M1 Stage 1 Final")
    print("Stock Master + Point-in-Time Raw Universe")
    print("=" * 72)

    # ========================================================
    # A. Securities
    # ========================================================

    raw = (
        load_uqer_daily_info()
    )

    mapping = (
        infer_column_mapping(
            raw
        )
    )

    records = (
        standardize_security_records(
            raw,
            mapping,
        )
    )

    # ========================================================
    # B. 主研究交易所
    #
    # 修复问题1：
    # REVIEW 从这里开始就只针对沪深。
    # ========================================================

    (
        main_records,
        non_main_records,
    ) = (
        split_target_records(
            records
        )
    )

    # ========================================================
    # C. missing listDate
    # ========================================================

    (
        missing_records,
        missing_record_summary,
        missing_unique_summary,
    ) = (
        classify_missing_list_date_records(
            main_records
        )
    )

    # ========================================================
    # D. 正式上市记录
    # ========================================================

    (
        listed_records,
        excluded_records,
    ) = (
        filter_listed_records(
            main_records
        )
    )

    # ========================================================
    # E. Stock Master
    # ========================================================

    (
        stock_master,
        stock_code_history,
    ) = (
        build_stock_master(
            listed_records
        )
    )

    # ========================================================
    # F. 修复问题2：
    # 同代码多 secID 按时间是否重叠判断
    # ========================================================

    code_reuse = (
        analyze_code_reuse(
            stock_master
        )
    )

    issues = (
        build_issue_table(
            stock_master,
            missing_records,
            code_reuse,
        )
    )

    quality_summary = (
        build_quality_summary(
            stock_master,
            missing_records,
            issues,
            code_reuse,
        )
    )

    # ========================================================
    # G. 交易日历
    # ========================================================

    trade_dates = (
        load_trade_calendar()
    )

    # ========================================================
    # H. Raw Universe
    # ========================================================

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

    # ========================================================
    # I. 修复问题3：
    # 与 END_DATE daily_temp3 做真实集合比较
    # ========================================================

    (
        end_crosscheck,
        daily_only,
        raw_only,
        code_rule_review,
    ) = (
        crosscheck_end_date_universe(
            raw_universe,
            main_records,
        )
    )

    # ========================================================
    # J. 保存证券主表
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

    # ========================================================
    # K. 保存 QA
    # ========================================================

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

    code_reuse.to_csv(
        OUTPUT_DIR
        / "code_reuse_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # L. missing listDate
    # ========================================================

    missing_records.to_csv(
        OUTPUT_DIR
        / "missing_list_date_records_main_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_record_summary.to_csv(
        OUTPUT_DIR
        / "missing_list_date_record_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_unique_summary.to_csv(
        OUTPUT_DIR
        / "missing_list_date_unique_security_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    excluded_records.to_csv(
        OUTPUT_DIR
        / "excluded_security_records_main_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 非主样本单独保存，避免混入主研究 REVIEW
    non_main_records.to_csv(
        OUTPUT_DIR
        / "non_main_exchange_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # M. Raw Universe
    # ========================================================

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
    # N. 汇总信息
    # ========================================================

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

    high_count = (
        int(
            issues[
                "severity"
            ]
            .eq(
                "HIGH"
            )
            .sum()
        )
        if not
        issues.empty
        else 0
    )

    review_count = (
        int(
            issues[
                "severity"
            ]
            .eq(
                "REVIEW"
            )
            .sum()
        )
        if not
        issues.empty
        else 0
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

        "target_exchanges":
            TARGET_EXCHANGES,

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

        "end_date_raw_universe_count":
            end_count,

        "expected_end_universe_count":
            EXPECTED_END_UNIVERSE_COUNT,

        "high_issue_rows":
            high_count,

        "review_issue_rows":
            review_count,

        "raw_universe_rows":
            int(
                len(
                    raw_universe
                )
            ),

        "column_mapping":
            mapping,

        "important_notes": [

            (
                "missing listDate 和 REVIEW "
                "只针对当前 TARGET_EXCHANGES 统计。"
            ),

            (
                "同一代码多 secID 不再自动判 HIGH；"
                "只有有效期重叠才判 OVERLAPPING_CODE_REUSE / HIGH。"
            ),

            (
                "有效期不重叠的代码复用记为 "
                "HISTORICAL_CODE_REUSE / INFO。"
            ),

            (
                "A 股代码规则采用："
                "SSE=60xxxx/68xxxx，"
                "SZSE=00xxxx/30xxxx。"
            ),

            (
                "程序会自动读取 END_DATE 的 daily_temp3，"
                "生成 daily_only 和 raw_only 差异表。"
            ),

            (
                "Raw Universe 当前仍采用 "
                "list_date <= t < delist_date。"
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
    # O. 控制台输出
    # ========================================================

    print()
    print("=" * 72)
    print("Stage 1 Final 完成")
    print("=" * 72)

    print(
        f"交易日数量："
        f"{len(trade_dates):,}"
    )

    print(
        f"历史上市证券主体："
        f"{stock_master['security_id'].nunique():,}"
    )

    print(
        f"历史退市证券："
        f"{stock_master['delist_date'].notna().sum():,}"
    )

    print(
        f"{END_DATE.date()} Raw Universe："
        f"{end_count:,}"
    )

    print(
        f"HIGH issues："
        f"{high_count:,}"
    )

    print(
        f"REVIEW issues："
        f"{review_count:,}"
    )

    print()
    print(
        "END_DATE 与 daily_temp3 差异："
    )

    print(
        end_crosscheck.to_string(
            index=False
        )
    )

    print()
    print(
        f"输出目录：{OUTPUT_DIR}"
    )

    # ========================================================
    # P. 最终 Pass Candidate
    # ========================================================

    if (
        high_count == 0
        and
        len(
            daily_only
        ) == 0
        and
        len(
            raw_only
        ) == 0
    ):

        print()
        print(
            "[PASS] "
            "Stage 1 未发现 HIGH 问题，"
            "且 END_DATE Raw Universe "
            "与 daily_temp3 完全一致。"
        )

    else:

        print()
        print(
            "[REVIEW] "
            "Stage 1 主体已完成，"
            "但仍存在需要查看的差异。"
        )

        if (
            len(
                daily_only
            )
            > 0
        ):

            print(
                "请查看："
                "end_date_daily_only.csv"
            )

        if (
            len(
                raw_only
            )
            > 0
        ):

            print(
                "请查看："
                "end_date_raw_only.csv"
            )

        if (
            len(
                code_rule_review
            )
            > 0
        ):

            print(
                "请查看："
                "end_date_code_rule_review.csv"
            )


if __name__ == "__main__":

    main()
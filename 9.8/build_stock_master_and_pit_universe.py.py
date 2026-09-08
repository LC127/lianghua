from __future__ import annotations

from pathlib import Path
import re
import json

import pandas as pd


# ============================================================
# 0. 路径和研究参数
# ============================================================

ROOT = Path(r"D:\lowfreq")

# 股票基础信息目录
SECURITY_PATH = ROOT / "uqer_daily_info"

# 沪深交易日历
TRADE_CALENDAR_PATH = ROOT / "trade_calendar" / "trade_date.csv"

# 输出目录
OUTPUT_DIR = Path(r"D:\output\M1_day1\01_stage1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# M1 第一版主样本
START_DATE = pd.Timestamp("2015-01-05")
END_DATE = pd.Timestamp("2026-09-03")

# 第一版只做沪深 A 股
INCLUDE_SSE = True
INCLUDE_SZSE = True
INCLUDE_BSE = False


# ============================================================
# 1. 可能出现的字段名称
#
# 由于你本地 uqer_daily_info 的实际列名目前还没贴出来，
# 这里先兼容 UQER 常见命名和一般数据库命名。
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
# 2. 一些通用工具
# ============================================================

SUPPORTED_SUFFIX = {
    ".csv",
    ".parquet",
    ".pq",
    ".feather",
    ".pkl",
    ".pickle",
}


def normalize_colname(x: str) -> str:
    """
    字段名标准化：
    sec_ID、SecID、secid -> secid
    """
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def find_column(columns, aliases):
    """
    在真实列名中寻找某个标准字段。
    """
    columns = list(columns)

    # 先完全匹配
    for alias in aliases:
        if alias in columns:
            return alias

    # 再忽略大小写、下划线等
    normalized = {
        normalize_colname(col): col
        for col in columns
    }

    for alias in aliases:
        key = normalize_colname(alias)
        if key in normalized:
            return normalized[key]

    return None


def read_data_file(path: Path) -> pd.DataFrame:
    """
    读取 csv / parquet / feather / pickle。
    """
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

    elif suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    elif suffix == ".feather":
        return pd.read_feather(path)

    elif suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    else:
        raise ValueError(f"暂不支持文件格式：{path}")


def find_data_files(path: Path):
    """
    uqer_daily_info 如果是目录，则递归读取其中的数据文件；
    如果本身就是一个文件，则直接读取。
    """
    if path.is_file():
        return [path]

    if not path.exists():
        raise FileNotFoundError(
            f"没有找到路径：{path}"
        )

    files = [
        p for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIX
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            f"{path} 下没有找到支持的数据文件。"
        )

    return sorted(files)


# ============================================================
# 3. 读取 uqer_daily_info
# ============================================================

def load_uqer_daily_info():
    files = find_data_files(SECURITY_PATH)

    print("=" * 70)
    print("找到 uqer_daily_info 数据文件：")
    print("=" * 70)

    for f in files:
        print(f)

    dfs = []

    for file in files:

        try:
            df = read_data_file(file)

        except Exception as e:
            print(
                f"[WARNING] 无法读取 {file}: {e}"
            )
            continue

        if len(df) == 0:
            continue

        df = df.copy()

        # 记录数据来自哪个文件
        df["_source_file"] = str(file)

        dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError(
            "uqer_daily_info 中没有成功读取任何数据。"
        )

    raw = pd.concat(
        dfs,
        ignore_index=True,
        sort=False,
    )

    print()
    print("=" * 70)
    print("uqer_daily_info 基本情况")
    print("=" * 70)

    print(f"总行数：{len(raw):,}")
    print(f"总列数：{raw.shape[1]}")

    print("\n全部字段：")

    for col in raw.columns:
        print("  ", col)

    return raw


# ============================================================
# 4. 自动识别真实字段
# ============================================================

def infer_column_mapping(df):

    mapping = {}

    for standard_name, aliases in COLUMN_ALIASES.items():

        mapping[standard_name] = find_column(
            df.columns,
            aliases,
        )

    print()
    print("=" * 70)
    print("自动识别字段结果")
    print("=" * 70)

    for key, value in mapping.items():
        print(
            f"{key:15s} <- {value}"
        )

    # Stage 1 必须至少有代码和上市日期
    required = [
        "stock_code",
        "list_date",
    ]

    missing = [
        x
        for x in required
        if mapping[x] is None
    ]

    if len(missing) > 0:

        raise ValueError(
            "\n无法自动识别以下必需字段："
            f"{missing}\n\n"
            "请把 uqer_daily_info 的真实 columns 发给我，"
            "或者把真实字段名补入 COLUMN_ALIASES。"
        )

    return mapping


# ============================================================
# 5. 股票代码、交易所、日期标准化
# ============================================================

def clean_stock_code(series):
    """
    统一得到 6 位股票代码。

    例如：
        000001.XSHE -> 000001
        600000.SH   -> 600000
        000001      -> 000001
    """
    x = (
        series
        .astype("string")
        .str.strip()
    )

    code = x.str.extract(
        r"(\d{6})",
        expand=False,
    )

    return code.fillna(x)


def normalize_exchange(series):
    """
    将不同交易所命名统一为：
        SSE
        SZSE
        BSE
    """
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

    # 上海
    mask = x.str.contains(
        r"SSE|XSHG|SHSE|SHANGHAI|上海|上交",
        regex=True,
        na=False,
    )

    result.loc[mask] = "SSE"

    # 深圳
    mask = x.str.contains(
        r"SZSE|XSHE|SHENZHEN|深圳|深交",
        regex=True,
        na=False,
    )

    result.loc[mask] = "SZSE"

    # 北京
    mask = x.str.contains(
        r"BSE|XBJ|BEIJING|北京|北交",
        regex=True,
        na=False,
    )

    result.loc[mask] = "BSE"

    return result


def infer_exchange_from_code(stock_code):
    """
    如果 exchange 字段缺失，
    根据股票代码做辅助推断。

    注意：
    这是 fallback，
    正式研究应优先使用供应商提供的 exchange 字段。
    """
    code = stock_code.astype("string")

    exchange = pd.Series(
        pd.NA,
        index=code.index,
        dtype="string",
    )

    # 上海
    exchange.loc[
        code.str.match(
            r"^(600|601|603|605|688|689)",
            na=False,
        )
    ] = "SSE"

    # 深圳
    exchange.loc[
        code.str.match(
            r"^(000|001|002|003|300|301)",
            na=False,
        )
    ] = "SZSE"

    # 北交所，仅作辅助判断
    exchange.loc[
        code.str.match(
            r"^(43|83|87|88|92)",
            na=False,
        )
    ] = "BSE"

    return exchange


def infer_board(exchange, code):

    board = pd.Series(
        "UNKNOWN",
        index=code.index,
        dtype="string",
    )

    code = code.astype("string")

    # 科创板
    board.loc[
        (exchange == "SSE")
        & code.str.startswith("688", na=False)
    ] = "STAR"

    # 创业板
    board.loc[
        (exchange == "SZSE")
        & code.str.startswith(
            ("300", "301"),
            na=False,
        )
    ] = "CHINEXT"

    # 北交所
    board.loc[
        exchange == "BSE"
    ] = "BSE"

    # 沪市主板
    board.loc[
        (exchange == "SSE")
        & (board == "UNKNOWN")
    ] = "SSE_MAIN"

    # 深市主板
    board.loc[
        (exchange == "SZSE")
        & (board == "UNKNOWN")
    ] = "SZSE_MAIN"

    return board


# ============================================================
# 6. 构造标准证券记录
# ============================================================

def standardize_security_records(raw, mapping):

    result = pd.DataFrame(
        index=raw.index
    )

    # --------------------------------------------------------
    # security_id
    # --------------------------------------------------------

    if mapping["security_id"] is not None:

        result["security_id"] = (
            raw[mapping["security_id"]]
            .astype("string")
            .str.strip()
        )

        result["security_id_source"] = "vendor"

    else:

        result["security_id"] = pd.NA

        result["security_id_source"] = "fallback"

    # --------------------------------------------------------
    # 股票代码
    # --------------------------------------------------------

    result["stock_code"] = clean_stock_code(
        raw[mapping["stock_code"]]
    )

    # --------------------------------------------------------
    # 股票名称
    # --------------------------------------------------------

    if mapping["stock_name"] is not None:

        result["stock_name"] = (
            raw[mapping["stock_name"]]
            .astype("string")
            .str.strip()
        )

    else:

        result["stock_name"] = pd.NA

    # --------------------------------------------------------
    # 交易所
    # --------------------------------------------------------

    if mapping["exchange"] is not None:

        result["exchange"] = normalize_exchange(
            raw[mapping["exchange"]]
        )

    else:

        result["exchange"] = pd.NA

    # 如果 exchange 没识别出来，再用股票代码辅助
    inferred_exchange = infer_exchange_from_code(
        result["stock_code"]
    )

    result["exchange"] = (
        result["exchange"]
        .fillna(inferred_exchange)
    )

    # --------------------------------------------------------
    # 板块
    # --------------------------------------------------------

    result["board"] = infer_board(
        result["exchange"],
        result["stock_code"],
    )

    # --------------------------------------------------------
    # 上市日期
    # --------------------------------------------------------

    result["list_date"] = pd.to_datetime(
        raw[mapping["list_date"]],
        errors="coerce",
    )

    # --------------------------------------------------------
    # 退市日期
    # --------------------------------------------------------

    if mapping["delist_date"] is not None:

        result["delist_date"] = pd.to_datetime(
            raw[mapping["delist_date"]],
            errors="coerce",
        )

    else:

        result["delist_date"] = pd.NaT

    # --------------------------------------------------------
    # security type
    # --------------------------------------------------------

    if mapping["security_type"] is not None:

        result["security_type"] = (
            raw[mapping["security_type"]]
            .astype("string")
        )

    else:

        result["security_type"] = pd.NA

    # --------------------------------------------------------
    # list status
    # --------------------------------------------------------

    if mapping["list_status"] is not None:

        result["list_status"] = (
            raw[mapping["list_status"]]
            .astype("string")
        )

    else:

        result["list_status"] = pd.NA

    # --------------------------------------------------------
    # 如果没有稳定 security_id
    #
    # 暂时使用 exchange_stockcode。
    #
    # 注意：
    # 这种 fallback 无法完美解决历史代码变更，
    # 所以必须在 QA 中单独标记。
    # --------------------------------------------------------

    fallback_id = (
        result["exchange"]
        .fillna("UNKNOWN")
        + "_"
        + result["stock_code"]
        .fillna("UNKNOWN")
    )

    result["security_id"] = (
        result["security_id"]
        .fillna(fallback_id)
    )

    # 来源文件
    result["_source_file"] = (
        raw["_source_file"]
        .astype("string")
    )

    return result


# ============================================================
# 7. 排除明显非普通 A 股
# ============================================================

def probable_non_a_share(df):

    security_type = (
        df["security_type"]
        .astype("string")
        .fillna("")
        .str.upper()
    )

    stock_name = (
        df["stock_name"]
        .astype("string")
        .fillna("")
        .str.upper()
    )

    excluded_words = [
        "ETF",
        "LOF",
        "FUND",
        "基金",
        "BOND",
        "债",
        "可转债",
        "INDEX",
        "指数",
        "WARRANT",
        "权证",
        "B股",
        "B SHARE",
    ]

    mask = pd.Series(
        False,
        index=df.index,
    )

    for word in excluded_words:

        mask |= security_type.str.contains(
            word,
            regex=False,
            na=False,
        )

        mask |= stock_name.str.contains(
            word,
            regex=False,
            na=False,
        )

    return mask


# ============================================================
# 8. 构造 entity-level stock master
#
# 为什么要再聚合一次？
#
# uqer_daily_info 中可能同一个 security_id 有多条记录，
# 例如代码变更、简称变更等。
#
# 网络节点应该尽量以稳定 security_id 为单位，
# 而不是把同一证券主体重复计算。
# ============================================================

def build_stock_master(records):

    # 先筛主研究交易所
    allowed_exchange = []

    if INCLUDE_SSE:
        allowed_exchange.append("SSE")

    if INCLUDE_SZSE:
        allowed_exchange.append("SZSE")

    if INCLUDE_BSE:
        allowed_exchange.append("BSE")

    records = records[
        records["exchange"].isin(
            allowed_exchange
        )
    ].copy()

    # 排除明显非 A 股
    non_a_mask = probable_non_a_share(
        records
    )

    records["probable_non_a_share"] = (
        non_a_mask
    )

    records = records[
        ~non_a_mask
    ].copy()

    # --------------------------------------------------------
    # 建立代码历史表
    # --------------------------------------------------------

    code_history = (
        records[
            [
                "security_id",
                "stock_code",
                "stock_name",
                "exchange",
                "board",
                "list_date",
                "delist_date",
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
    )

    # --------------------------------------------------------
    # entity-level 主表
    #
    # 同一 security_id：
    # list_date 取最早
    # delist_date 取最晚
    #
    # latest_code 这里只是展示字段，
    # 后面与 daily_temp3 merge 时仍应尽量使用 security_id。
    # --------------------------------------------------------

    master = (
        records
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
            security_type=(
                "security_type",
                "last",
            ),
            list_status=(
                "list_status",
                "last",
            ),
            security_id_source=(
                "security_id_source",
                "first",
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

    return master, code_history


# ============================================================
# 9. Stage 1 证券主表 QA
# ============================================================

def quality_check_stock_master(master):

    issues = []

    def add_issue(mask, issue_type, severity, note):

        if not mask.any():
            return

        tmp = master.loc[
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

        tmp["issue_type"] = issue_type
        tmp["severity"] = severity
        tmp["note"] = note

        issues.append(tmp)

    # --------------------------------------------------------
    # 上市日期缺失
    # --------------------------------------------------------

    add_issue(
        master["list_date"].isna(),
        "missing_list_date",
        "HIGH",
        "缺少上市日期，无法构造历史股票池。",
    )

    # --------------------------------------------------------
    # 未识别交易所
    # --------------------------------------------------------

    add_issue(
        master["exchange"].isna(),
        "unknown_exchange",
        "HIGH",
        "无法识别交易所。",
    )

    # --------------------------------------------------------
    # 退市日期 <= 上市日期
    # --------------------------------------------------------

    invalid_date = (
        master["delist_date"].notna()
        & master["list_date"].notna()
        & (
            master["delist_date"]
            <= master["list_date"]
        )
    )

    add_issue(
        invalid_date,
        "invalid_list_delist_order",
        "HIGH",
        "退市日期早于或等于上市日期。",
    )

    # --------------------------------------------------------
    # 同一 security_id 多个历史代码
    #
    # 这不一定错误，可能只是代码变更。
    # --------------------------------------------------------

    add_issue(
        master["code_count"] > 1,
        "multiple_historical_codes",
        "INFO",
        "该证券主体存在多个历史代码，需要保留代码映射。",
    )

    if len(issues) > 0:

        issues_df = pd.concat(
            issues,
            ignore_index=True,
        )

    else:

        issues_df = pd.DataFrame()

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "unique_security_count",
                "sse_count",
                "szse_count",
                "bse_count",
                "historically_delisted_count",
                "missing_list_date_count",
                "multiple_code_security_count",
                "fallback_security_id_count",
            ],

            "value": [
                master["security_id"].nunique(),

                master.loc[
                    master["exchange"] == "SSE",
                    "security_id",
                ].nunique(),

                master.loc[
                    master["exchange"] == "SZSE",
                    "security_id",
                ].nunique(),

                master.loc[
                    master["exchange"] == "BSE",
                    "security_id",
                ].nunique(),

                master[
                    "delist_date"
                ].notna().sum(),

                master[
                    "list_date"
                ].isna().sum(),

                (
                    master["code_count"]
                    > 1
                ).sum(),

                (
                    master[
                        "security_id_source"
                    ]
                    == "fallback"
                ).sum(),
            ],
        }
    )

    return summary, issues_df


# ============================================================
# 10. 读取沪深交易日历
# ============================================================

def load_trade_calendar():

    df = read_data_file(
        TRADE_CALENDAR_PATH
    )

    print()
    print("=" * 70)
    print("交易日历字段")
    print("=" * 70)

    print(list(df.columns))

    # 尝试自动找到日期列
    candidate_columns = [
        "trade_date",
        "tradeDate",
        "calendarDate",
        "date",
        "tradeDay",
    ]

    date_col = find_column(
        df.columns,
        candidate_columns,
    )

    if date_col is None:

        # 如果只有一列，则直接把这一列当日期
        if df.shape[1] == 1:

            date_col = df.columns[0]

        else:

            raise ValueError(
                "无法识别 trade_calendar 日期列。"
            )

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    dates = (
        dates
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    dates = dates[
        (dates >= START_DATE)
        & (dates <= END_DATE)
    ]

    print()
    print(
        f"研究期交易日："
        f"{dates.min().date()} ~ "
        f"{dates.max().date()}"
    )

    print(
        f"交易日数量：{len(dates):,}"
    )

    return pd.DatetimeIndex(
        dates
    )


# ============================================================
# 11. 构造 Point-in-Time Raw Universe
# ============================================================

def build_raw_universe(
    stock_master,
    trade_dates,
):

    """
    对每个证券主体 i：

        list_date_i <= t < delist_date_i

    时认为证券属于当天 Raw Universe。

    注意：
    这里暂定 delist_date 当天不属于 Universe。

    如果供应商文档说明：
    delistDate 是“最后交易日”，
    则这里未来要改成 <=。
    """

    parts = []

    for row in stock_master.itertuples(
        index=False
    ):

        if pd.isna(row.list_date):
            continue

        start = row.list_date

        if pd.isna(row.delist_date):

            end = (
                END_DATE
                + pd.Timedelta(days=1)
            )

        else:

            end = row.delist_date

        # 在交易日历中找到对应区间
        left = trade_dates.searchsorted(
            start,
            side="left",
        )

        right = trade_dates.searchsorted(
            end,
            side="left",
        )

        if left >= right:
            continue

        valid_dates = trade_dates[
            left:right
        ]

        tmp = pd.DataFrame(
            {
                "trade_date": valid_dates,
                "security_id": row.security_id,
                "stock_code": row.stock_code,
                "stock_name": row.stock_name,
                "exchange": row.exchange,
                "board": row.board,
            }
        )

        parts.append(tmp)

    if len(parts) == 0:

        raise RuntimeError(
            "没有生成任何历史股票池记录。"
        )

    raw_universe = pd.concat(
        parts,
        ignore_index=True,
    )

    return raw_universe


# ============================================================
# 12. 每日股票池规模统计
# ============================================================

def summarize_raw_universe(
    raw_universe
):

    # 全市场
    total = (
        raw_universe
        .groupby("trade_date")
        .agg(
            total_stock_count=(
                "security_id",
                "nunique",
            )
        )
        .reset_index()
    )

    # 分交易所
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

    # 变成宽表更方便查看
    exchange_wide = (
        by_exchange
        .pivot(
            index="trade_date",
            columns="exchange",
            values="stock_count",
        )
        .reset_index()
    )

    summary = total.merge(
        exchange_wide,
        on="trade_date",
        how="left",
    )

    return summary


# ============================================================
# 13. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("M1 Step 1")
    print("历史证券主表 + Point-in-Time Raw Universe")
    print("=" * 70)

    # --------------------------------------------------------
    # A. 读取 uqer_daily_info
    # --------------------------------------------------------

    raw = load_uqer_daily_info()

    # --------------------------------------------------------
    # B. 字段识别
    # --------------------------------------------------------

    mapping = infer_column_mapping(
        raw
    )

    # --------------------------------------------------------
    # C. 标准化证券记录
    # --------------------------------------------------------

    records = standardize_security_records(
        raw,
        mapping,
    )

    # --------------------------------------------------------
    # D. 建立 entity-level stock master
    # --------------------------------------------------------

    stock_master, code_history = (
        build_stock_master(
            records
        )
    )

    # --------------------------------------------------------
    # E. QA
    # --------------------------------------------------------

    quality_summary, issues = (
        quality_check_stock_master(
            stock_master
        )
    )

    # --------------------------------------------------------
    # F. 交易日历
    # --------------------------------------------------------

    trade_dates = (
        load_trade_calendar()
    )

    # --------------------------------------------------------
    # G. 构造历史 Raw Universe
    # --------------------------------------------------------

    raw_universe = (
        build_raw_universe(
            stock_master,
            trade_dates,
        )
    )

    # --------------------------------------------------------
    # H. 每日股票数量
    # --------------------------------------------------------

    daily_summary = (
        summarize_raw_universe(
            raw_universe
        )
    )

    # ========================================================
    # 14. 输出结果
    # ========================================================

    stock_master.to_parquet(
        OUTPUT_DIR
        / "stock_master.parquet",
        index=False,
    )

    code_history.to_parquet(
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

    # --------------------------------------------------------
    # metadata
    # --------------------------------------------------------

    metadata = {

        "research_start": str(
            START_DATE.date()
        ),

        "research_end": str(
            END_DATE.date()
        ),

        "include_sse": INCLUDE_SSE,

        "include_szse": INCLUDE_SZSE,

        "include_bse": INCLUDE_BSE,

        "trade_day_count": int(
            len(trade_dates)
        ),

        "security_count": int(
            stock_master[
                "security_id"
            ].nunique()
        ),

        "raw_universe_rows": int(
            len(raw_universe)
        ),

        "column_mapping": mapping,

        "important_note": (
            "当前采用 "
            "list_date <= t < delist_date。"
            "请进一步确认供应商 delistDate "
            "究竟表示最后交易日还是正式退市生效日。"
        ),
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
    # 15. 控制台结果
    # ========================================================

    print()
    print("=" * 70)
    print("Step 1 完成")
    print("=" * 70)

    print(
        f"证券主体数量："
        f"{stock_master['security_id'].nunique():,}"
    )

    print(
        f"历史退市证券数量："
        f"{stock_master['delist_date'].notna().sum():,}"
    )

    print(
        f"交易日数量："
        f"{len(trade_dates):,}"
    )

    print(
        f"Raw Universe 总行数："
        f"{len(raw_universe):,}"
    )

    print()
    print("输出目录：")
    print(OUTPUT_DIR)

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
        "5. raw_universe_by_date.parquet"
    )
    print(
        "6. raw_universe_daily_summary.csv"
    )
    print(
        "7. stage1_metadata.json"
    )


if __name__ == "__main__":
    main()
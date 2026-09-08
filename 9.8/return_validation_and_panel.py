from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json
import math
import re

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    raise ImportError(
        "Step 3 需要 pyarrow 来流式写入全市场 Parquet。"
        "请先安装：pip install pyarrow"
    ) from e


# ============================================================
# 0. 路径
# ============================================================

# ------------------------------------------------------------
# Step 1
# ------------------------------------------------------------

STAGE1_DIR = Path(
    r"D:\output\M1_day1\01_stage1_final"
)

STOCK_MASTER_PATH = (
    STAGE1_DIR
    / "stock_master.parquet"
)

TRADE_CALENDAR_PATH = (
    STAGE1_DIR
    / "trade_calendar_used.csv"
)


# ------------------------------------------------------------
# Step 2
# ------------------------------------------------------------

STAGE2_DIR = Path(
    r"D:\output\M1_day1\02_stage2_daily_market_qa"
)

IDENTITY_RECONCILIATION_PATH = (
    STAGE2_DIR
    / "identity_reconciliation_detail.parquet"
)


# ------------------------------------------------------------
# 原始行情
# ------------------------------------------------------------

DAILY_TEMP3_DIR = Path(
    r"D:\lowfreq\daily_temp3"
)


# ------------------------------------------------------------
# Step 3 输出
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\output\M1_day1\03_stage3_return_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

EXPECTED_TRADE_DAY_COUNT = 2837

TARGET_EXCHANGES = [
    "SSE",
    "SZSE",
]


# ============================================================
# 2. 收益定义参数
# ============================================================

# "auto"：
# 自动判断 adjFactor 应该乘还是除
#
# 也可人工指定：
# "multiply"
# "divide"

ADJ_FACTOR_MODE = "auto"


# chgPct 口径：
#
# "auto"：
# 自动判断是 decimal 还是 percentage point
#
# 例如：
# 0.01 = 1%
# 或
# 1.00 = 1%
#
# 也可以指定：
# "decimal"
# "percentage"

CHG_PCT_SCALE_MODE = "auto"


# 用于判断 adjFactor 是否变化
FACTOR_CHANGE_LOG_TOL = 1e-10


# 若因子变化样本少于该数量，
# adjFactor multiply/divide 的识别退回到全部有效样本
MIN_FACTOR_CHANGE_PAIRS = 20


# chgPct 与自己计算收益之间
# 超过 5bp 时记录为 REVIEW
RETURN_DIFF_TOL = 5e-4


SUPPORTED_SUFFIX = {
    ".csv",
    ".parquet",
    ".pq",
    ".feather",
}


# ============================================================
# 3. 字段别名
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

    "open": [
        "openPrice",
        "open",
        "open_price",
    ],

    "high": [
        "highestPrice",
        "high",
        "highPrice",
    ],

    "low": [
        "lowestPrice",
        "low",
        "lowPrice",
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
        "turnover_value",
        "amount",
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
    ],

    "neg_market_value": [
        "negMarketValue",
        "neg_market_value",
    ],

    "turnover_rate": [
        "turnoverRate",
        "turnover_rate",
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


# ============================================================
# 4. 通用工具
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
    x: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        x,
        errors="coerce",
    )


# ============================================================
# 5. isOpen
# ============================================================

def normalize_is_open(
    series: pd.Series,
) -> pd.Series:

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

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

    true_mask = (
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

    false_mask = (
        num.eq(0)
        |
        text.isin(
            {
                "0",
                "FALSE",
                "F",
                "NO",
                "N",
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
# 6. 交易所
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
            "XSHG",
            na=False,
        )
    ] = "SSE"

    result.loc[
        x.str.contains(
            "XSHE",
            na=False,
        )
    ] = "SZSE"

    result.loc[
        x.str.contains(
            "XBJ",
            na=False,
        )
    ] = "BSE"

    return result


# ============================================================
# 7. 股票代码
# ============================================================

def canonical_stock_code(
    stock_code: pd.Series,
    security_id: pd.Series,
) -> pd.Series:

    sid = (
        security_id
        .astype("string")
        .str.strip()
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
    )

    numeric_code = (
        code.str.extract(
            r"^(\d{1,6})$",
            expand=False,
        )
    )

    numeric_code = (
        numeric_code.str.zfill(6)
    )

    return (
        code_from_sid
        .fillna(
            numeric_code
        )
        .fillna(
            code
        )
    )


# ============================================================
# 8. 文件名日期
# ============================================================

def extract_date_from_filename(
    path: Path,
):

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
# 9. 交易日
# ============================================================

def load_trade_dates():

    df = pd.read_csv(
        TRADE_CALENDAR_PATH
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
# 10. Daily 文件映射
# ============================================================

def build_daily_file_map(
    trade_dates,
):

    file_map = defaultdict(
        list
    )

    for file in (
        DAILY_TEMP3_DIR.rglob("*")
    ):

        if not file.is_file():
            continue

        if (
            file.suffix.lower()
            not in SUPPORTED_SUFFIX
        ):
            continue

        date = (
            extract_date_from_filename(
                file
            )
        )

        if (
            date is not None
            and
            date in trade_dates
        ):

            file_map[
                date
            ].append(
                file
            )

    missing = []

    duplicated = []

    result = {}

    for date in trade_dates:

        files = (
            file_map.get(
                date,
                [],
            )
        )

        if len(files) == 0:

            missing.append(
                date
            )

        elif len(files) > 1:

            duplicated.append(
                date
            )

        else:

            result[
                date
            ] = files[0]

    if missing:

        raise RuntimeError(
            f"缺失 {len(missing)} 个 daily 文件。"
        )

    if duplicated:

        raise RuntimeError(
            f"存在 {len(duplicated)} 个重复日期文件。"
        )

    return result


# ============================================================
# 11. Stock Master
# ============================================================

def load_stock_master():

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
# 12. Step 2 Identity Reconciliation
# ============================================================

def load_identity_reconciliation():

    result = {}

    if not (
        IDENTITY_RECONCILIATION_PATH.exists()
    ):

        return result

    try:

        df = pd.read_parquet(
            IDENTITY_RECONCILIATION_PATH
        )

    except Exception:

        return result

    required = {
        "trade_date",
        "raw_security_id",
        "daily_security_id",
    }

    if (
        df.empty
        or
        not required.issubset(
            set(
                df.columns
            )
        )
    ):

        return result

    df = df.copy()

    df[
        "trade_date"
    ] = pd.to_datetime(
        df[
            "trade_date"
        ],
        errors="coerce",
    )

    for date, group in (
        df.groupby(
            "trade_date"
        )
    ):

        result[
            date
        ] = dict(
            zip(
                group[
                    "daily_security_id"
                ].astype("string"),
                group[
                    "raw_security_id"
                ].astype("string"),
            )
        )

    return result


# ============================================================
# 13. 某日 Expected PIT Universe
# ============================================================

def expected_security_ids(
    stock_master,
    date,
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

    return set(
        stock_master.loc[
            mask,
            "security_id",
        ].astype("string")
    )


# ============================================================
# 14. 标准化 Daily
# ============================================================

def standardize_daily(
    raw,
    date,
):

    mapping = (
        infer_mapping(
            raw
        )
    )

    required = [
        "security_id",
        "close",
        "adj_factor",
        "chg_pct",
        "is_open",
    ]

    missing = [

        field

        for field in required

        if mapping[
            field
        ] is None
    ]

    if missing:

        raise ValueError(
            f"{date.date()} 缺关键字段：{missing}"
        )

    result = pd.DataFrame(
        index=raw.index
    )

    # secID
    result[
        "source_security_id"
    ] = (
        raw[
            mapping[
                "security_id"
            ]
        ]
        .astype("string")
        .str.strip()
    )

    # code
    if (
        mapping[
            "stock_code"
        ]
        is not None
    ):

        original_code = (
            raw[
                mapping[
                    "stock_code"
                ]
            ]
        )

    else:

        original_code = pd.Series(
            pd.NA,
            index=raw.index,
        )

    result[
        "stock_code"
    ] = (
        canonical_stock_code(
            original_code,
            result[
                "source_security_id"
            ],
        )
    )

    # name
    if (
        mapping[
            "stock_name"
        ]
        is not None
    ):

        result[
            "stock_name"
        ] = (
            raw[
                mapping[
                    "stock_name"
                ]
            ]
            .astype("string")
        )

    else:

        result[
            "stock_name"
        ] = pd.NA

    # exchange
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
                raw[
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

    result[
        "exchange"
    ] = (
        result[
            "exchange"
        ]
        .fillna(
            infer_exchange_from_secid(
                result[
                    "source_security_id"
                ]
            )
        )
    )

    result[
        "trade_date"
    ] = date

    # 数值字段
    numeric_fields = [

        "open",
        "high",
        "low",
        "close",

        "volume",
        "turnover_value",

        "adj_factor",
        "chg_pct",

        "market_value",
        "neg_market_value",
        "turnover_rate",
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
                raw[
                    col
                ]
            )

    # isOpen
    result[
        "is_open"
    ] = (
        normalize_is_open(
            raw[
                mapping[
                    "is_open"
                ]
            ]    
        )
    )

    # industry
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
                raw[
                    col
                ]
                .astype("string")
            )

    # 主研究只保留沪深
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

    return result


# ============================================================
# 15. 将 Step 2 reconciliation 应用于 Daily
# ============================================================

def build_research_daily(
    daily,
    date,
    stock_master,
    reconciliation_map,
):

    daily = daily.copy()

    date_map = (
        reconciliation_map.get(
            date,
            {},
        )
    )

    mapped_id = (
        daily[
            "source_security_id"
        ]
        .map(
            date_map
        )
    )

    daily[
        "identity_reconciled"
    ] = (
        mapped_id.notna()
    )

    daily[
        "security_id"
    ] = (
        mapped_id
        .fillna(
            daily[
                "source_security_id"
            ]
        )
        .astype("string")
    )

    expected_ids = (
        expected_security_ids(
            stock_master,
            date,
        )
    )

    research = (
        daily[
            daily[
                "security_id"
            ].isin(
                expected_ids
            )
        ]
        .copy()
    )

    duplicated = (
        research[
            "security_id"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicated.any():

        ids = (
            research.loc[
                duplicated,
                "security_id",
            ]
            .unique()
        )

        raise ValueError(
            f"{date.date()} canonical security_id 重复："
            f"{ids[:10]}"
        )

    return research


# ============================================================
# 16. Pair statistics
#
# 用于比较：
# 自己计算的收益 vs vendor chgPct
# ============================================================

def new_pair_stats():

    return {
        "count": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
        "max_abs": 0.0,
    }


def update_pair_stats(
    stats,
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if not valid.any():
        return

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    diff = (
        x
        -
        y
    )

    stats[
        "count"
    ] += len(
        diff
    )

    stats[
        "sum_abs"
    ] += float(
        np.abs(
            diff
        ).sum()
    )

    stats[
        "sum_sq"
    ] += float(
        np.square(
            diff
        ).sum()
    )

    stats[
        "sum_x"
    ] += float(
        x.sum()
    )

    stats[
        "sum_y"
    ] += float(
        y.sum()
    )

    stats[
        "sum_x2"
    ] += float(
        np.square(
            x
        ).sum()
    )

    stats[
        "sum_y2"
    ] += float(
        np.square(
            y
        ).sum()
    )

    stats[
        "sum_xy"
    ] += float(
        (
            x
            *
            y
        ).sum()
    )

    stats[
        "max_abs"
    ] = max(
        stats[
            "max_abs"
        ],
        float(
            np.abs(
                diff
            ).max()
        ),
    )


def finalize_pair_stats(
    stats,
):

    n = stats[
        "count"
    ]

    if n == 0:

        return {
            "count": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "corr": np.nan,
            "max_abs": np.nan,
        }

    mae = (
        stats[
            "sum_abs"
        ]
        /
        n
    )

    rmse = math.sqrt(
        stats[
            "sum_sq"
        ]
        /
        n
    )

    sx = stats[
        "sum_x"
    ]

    sy = stats[
        "sum_y"
    ]

    sxx = stats[
        "sum_x2"
    ]

    syy = stats[
        "sum_y2"
    ]

    sxy = stats[
        "sum_xy"
    ]

    numerator = (
        n
        *
        sxy
        -
        sx
        *
        sy
    )

    denom_x = (
        n
        *
        sxx
        -
        sx ** 2
    )

    denom_y = (
        n
        *
        syy
        -
        sy ** 2
    )

    if (
        denom_x > 0
        and
        denom_y > 0
    ):

        corr = (
            numerator
            /
            math.sqrt(
                denom_x
                *
                denom_y
            )
        )

    else:

        corr = np.nan

    return {
        "count": n,
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "max_abs": stats[
            "max_abs"
        ],
    }


# ============================================================
# 17. 第一遍扫描：
#
# 自动判断：
#
# 1. chgPct scale
# 2. adjFactor multiply / divide
# ============================================================

def infer_return_definition(
    trade_dates,
    file_map,
    stock_master,
    reconciliation_map,
):

    print()
    print("=" * 72)
    print("Pass 1：推断 chgPct 与 adjFactor 口径")
    print("=" * 72)

    # --------------------------------------------------------
    # chgPct scale
    #
    # 仅在 adjFactor 未变化时，
    # raw close return 应与 adjusted return 一致。
    # --------------------------------------------------------

    scale_stats = {
        "decimal": new_pair_stats(),
        "percentage": new_pair_stats(),
    }

    # --------------------------------------------------------
    # adjustment mode
    # --------------------------------------------------------

    mode_stats = {}

    for scale in [
        "decimal",
        "percentage",
    ]:

        for mode in [
            "multiply",
            "divide",
        ]:

            mode_stats[
                (
                    scale,
                    mode,
                    "all",
                )
            ] = new_pair_stats()

            mode_stats[
                (
                    scale,
                    mode,
                    "factor_change",
                )
            ] = new_pair_stats()

    # --------------------------------------------------------
    # 上一个 market date 状态
    # --------------------------------------------------------

    prev_close = {}
    prev_factor = {}
    prev_open = {}
    prev_day_index = {}

    for day_index, date in enumerate(
        trade_dates
    ):

        file = file_map[
            date
        ]

        raw = read_data_file(
            file
        )

        daily = standardize_daily(
            raw,
            date,
        )

        daily = build_research_daily(
            daily,
            date,
            stock_master,
            reconciliation_map,
        )

        sid = (
            daily[
                "security_id"
            ]
            .astype("string")
        )

        p_close = (
            sid
            .map(
                prev_close
            )
            .astype(float)
        )

        p_factor = (
            sid
            .map(
                prev_factor
            )
            .astype(float)
        )

        p_open = (
            sid
            .map(
                prev_open
            )
        )

        p_day = (
            sid
            .map(
                prev_day_index
            )
        )

        consecutive = (
            p_day
            .eq(
                day_index
                -
                1
            )
            .fillna(
                False
            )
        )

        active_pair = (

            daily[
                "is_open"
            ]
            .eq(
                True
            )
            .fillna(
                False
            )

            &

            p_open
            .eq(
                True
            )
            .fillna(
                False
            )

            &

            consecutive
        )

        valid_price = (

            active_pair

            &

            daily[
                "close"
            ].gt(0)

            &

            p_close.gt(0)

            &

            daily[
                "adj_factor"
            ].gt(0)

            &

            p_factor.gt(0)

            &

            daily[
                "chg_pct"
            ].notna()
        )

        # ----------------------------------------------------
        # factor change
        # ----------------------------------------------------

        log_factor_change = pd.Series(
            np.nan,
            index=daily.index,
            dtype=float,
        )

        factor_valid = (
            daily[
                "adj_factor"
            ].gt(0)
            &
            p_factor.gt(0)
        )

        log_factor_change.loc[
            factor_valid
        ] = np.log(
            daily.loc[
                factor_valid,
                "adj_factor",
            ]
            /
            p_factor.loc[
                factor_valid
            ]
        )

        factor_changed = (
            valid_price
            &
            log_factor_change.abs().gt(
                FACTOR_CHANGE_LOG_TOL
            )
        )

        factor_same = (
            valid_price
            &
            ~factor_changed
        )

        # ----------------------------------------------------
        # raw simple return
        # ----------------------------------------------------

        raw_simple_return = (
            daily[
                "close"
            ]
            /
            p_close
            -
            1.0
        )

        vendor_decimal = (
            daily[
                "chg_pct"
            ]
        )

        vendor_percentage = (
            daily[
                "chg_pct"
            ]
            /
            100.0
        )

        # ----------------------------------------------------
        # Step A:
        # 推断 chgPct scale
        #
        # 只使用 factor 未变化样本
        # ----------------------------------------------------

        update_pair_stats(
            scale_stats[
                "decimal"
            ],
            raw_simple_return.loc[
                factor_same
            ],
            vendor_decimal.loc[
                factor_same
            ],
        )

        update_pair_stats(
            scale_stats[
                "percentage"
            ],
            raw_simple_return.loc[
                factor_same
            ],
            vendor_percentage.loc[
                factor_same
            ],
        )

        # ----------------------------------------------------
        # adjusted price candidates
        # ----------------------------------------------------

        curr_mul = (
            daily[
                "close"
            ]
            *
            daily[
                "adj_factor"
            ]
        )

        prev_mul = (
            p_close
            *
            p_factor
        )

        ret_mul = (
            curr_mul
            /
            prev_mul
            -
            1.0
        )

        curr_div = (
            daily[
                "close"
            ]
            /
            daily[
                "adj_factor"
            ]
        )

        prev_div = (
            p_close
            /
            p_factor
        )

        ret_div = (
            curr_div
            /
            prev_div
            -
            1.0
        )

        vendor_dict = {
            "decimal":
                vendor_decimal,

            "percentage":
                vendor_percentage,
        }

        mode_return_dict = {
            "multiply":
                ret_mul,

            "divide":
                ret_div,
        }

        # ----------------------------------------------------
        # 同时为两种 scale 都积累 mode error
        # ----------------------------------------------------

        for scale_name, vendor_ret in (
            vendor_dict.items()
        ):

            for mode_name, model_ret in (
                mode_return_dict.items()
            ):

                update_pair_stats(
                    mode_stats[
                        (
                            scale_name,
                            mode_name,
                            "all",
                        )
                    ],
                    model_ret.loc[
                        valid_price
                    ],
                    vendor_ret.loc[
                        valid_price
                    ],
                )

                update_pair_stats(
                    mode_stats[
                        (
                            scale_name,
                            mode_name,
                            "factor_change",
                        )
                    ],
                    model_ret.loc[
                        factor_changed
                    ],
                    vendor_ret.loc[
                        factor_changed
                    ],
                )

        # ----------------------------------------------------
        # 更新 previous state
        # ----------------------------------------------------

        prev_close.update(
            zip(
                sid,
                daily[
                    "close"
                ],
            )
        )

        prev_factor.update(
            zip(
                sid,
                daily[
                    "adj_factor"
                ],
            )
        )

        prev_open.update(
            zip(
                sid,
                daily[
                    "is_open"
                ],
            )
        )

        prev_day_index.update(
            zip(
                sid,
                [day_index]
                *
                len(
                    daily
                ),
            )
        )

        if (
            day_index == 0
            or
            (
                day_index
                +
                1
            )
            % 250
            == 0
        ):

            print(
                f"[Pass1 "
                f"{day_index + 1:4d}/"
                f"{len(trade_dates)}] "
                f"{date.date()}"
            )

    # ========================================================
    # chgPct scale 选择
    # ========================================================

    scale_results = {

        key:
            finalize_pair_stats(
                value
            )

        for key, value
        in scale_stats.items()
    }

    if (
        CHG_PCT_SCALE_MODE
        ==
        "auto"
    ):

        candidates = [

            (
                name,
                metrics[
                    "mae"
                ],
            )

            for name, metrics
            in scale_results.items()

            if np.isfinite(
                metrics[
                    "mae"
                ]
            )
        ]

        if not candidates:

            raise RuntimeError(
                "无法根据数据判断 chgPct scale。"
            )

        selected_scale = min(
            candidates,
            key=lambda x:
                x[1],
        )[0]

    else:

        selected_scale = (
            CHG_PCT_SCALE_MODE
        )

    # ========================================================
    # adj factor mode
    # ========================================================

    factor_change_mul = (
        finalize_pair_stats(
            mode_stats[
                (
                    selected_scale,
                    "multiply",
                    "factor_change",
                )
            ]
        )
    )

    factor_change_div = (
        finalize_pair_stats(
            mode_stats[
                (
                    selected_scale,
                    "divide",
                    "factor_change",
                )
            ]
        )
    )

    use_factor_change = (
        min(
            factor_change_mul[
                "count"
            ],
            factor_change_div[
                "count"
            ],
        )
        >=
        MIN_FACTOR_CHANGE_PAIRS
    )

    if use_factor_change:

        mode_sample = (
            "factor_change"
        )

    else:

        mode_sample = (
            "all"
        )

    multiply_metrics = (
        finalize_pair_stats(
            mode_stats[
                (
                    selected_scale,
                    "multiply",
                    mode_sample,
                )
            ]
        )
    )

    divide_metrics = (
        finalize_pair_stats(
            mode_stats[
                (
                    selected_scale,
                    "divide",
                    mode_sample,
                )
            ]
        )
    )

    if (
        ADJ_FACTOR_MODE
        ==
        "auto"
    ):

        candidates = [

            (
                "multiply",
                multiply_metrics[
                    "mae"
                ],
            ),

            (
                "divide",
                divide_metrics[
                    "mae"
                ],
            ),
        ]

        candidates = [
            x
            for x in candidates
            if np.isfinite(
                x[1]
            )
        ]

        if not candidates:

            raise RuntimeError(
                "无法判断 adjFactor 的 multiply/divide 口径。"
            )

        selected_adj_mode = min(
            candidates,
            key=lambda x:
                x[1],
        )[0]

    else:

        selected_adj_mode = (
            ADJ_FACTOR_MODE
        )

    # ========================================================
    # Diagnostics table
    # ========================================================

    diagnostic_rows = []

    for scale_name, metrics in (
        scale_results.items()
    ):

        diagnostic_rows.append(
            {
                "stage":
                    "chgPct_scale",

                "candidate":
                    scale_name,

                "sample":
                    "factor_unchanged",

                **metrics,

                "selected":
                    (
                        scale_name
                        ==
                        selected_scale
                    ),
            }
        )

    for scale_name in [
        "decimal",
        "percentage",
    ]:

        for mode_name in [
            "multiply",
            "divide",
        ]:

            for sample_name in [
                "all",
                "factor_change",
            ]:

                metrics = (
                    finalize_pair_stats(
                        mode_stats[
                            (
                                scale_name,
                                mode_name,
                                sample_name,
                            )
                        ]
                    )
                )

                diagnostic_rows.append(
                    {
                        "stage":
                            "adjFactor_mode",

                        "candidate":
                            (
                                f"{scale_name}"
                                f"+"
                                f"{mode_name}"
                            ),

                        "sample":
                            sample_name,

                        **metrics,

                        "selected":
                            (
                                scale_name
                                ==
                                selected_scale
                                and
                                mode_name
                                ==
                                selected_adj_mode
                                and
                                sample_name
                                ==
                                mode_sample
                            ),
                    }
                )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    diagnostics.to_csv(
        OUTPUT_DIR
        / "return_definition_inference.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # scale factor
    # ========================================================

    if (
        selected_scale
        ==
        "decimal"
    ):

        vendor_scale_factor = (
            1.0
        )

    else:

        vendor_scale_factor = (
            0.01
        )

    definition = {

        "selected_chg_pct_scale":
            selected_scale,

        "vendor_scale_factor":
            vendor_scale_factor,

        "selected_adj_factor_mode":
            selected_adj_mode,

        "adj_mode_selection_sample":
            mode_sample,

        "factor_change_pair_count":
            int(
                factor_change_mul[
                    "count"
                ]
            ),
    }

    with open(
        OUTPUT_DIR
        / "return_definition.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            definition,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "识别结果："
    )

    print(
        f"chgPct scale："
        f"{selected_scale}"
    )

    print(
        f"adjFactor mode："
        f"{selected_adj_mode}"
    )

    print(
        f"mode selection sample："
        f"{mode_sample}"
    )

    return (
        selected_scale,
        vendor_scale_factor,
        selected_adj_mode,
        diagnostics,
    )


# ============================================================
# 18. 调整后价格
# ============================================================

def adjusted_price(
    close,
    factor,
    mode,
):

    close = np.asarray(
        close,
        dtype=float,
    )

    factor = np.asarray(
        factor,
        dtype=float,
    )

    result = np.full(
        len(close),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            close
        )
        &
        np.isfinite(
            factor
        )
        &
        (
            close > 0
        )
        &
        (
            factor > 0
        )
    )

    if mode == "multiply":

        result[
            valid
        ] = (
            close[
                valid
            ]
            *
            factor[
                valid
            ]
        )

    elif mode == "divide":

        result[
            valid
        ] = (
            close[
                valid
            ]
            /
            factor[
                valid
            ]
        )

    else:

        raise ValueError(
            f"未知 adjFactor mode：{mode}"
        )

    return result


# ============================================================
# 19. Panel Schema
# ============================================================

PANEL_SCHEMA = pa.schema(
    [

        pa.field(
            "trade_date",
            pa.timestamp("ns"),
        ),

        pa.field(
            "security_id",
            pa.string(),
        ),

        pa.field(
            "source_security_id",
            pa.string(),
        ),

        pa.field(
            "identity_reconciled",
            pa.bool_(),
        ),

        pa.field(
            "stock_code",
            pa.string(),
        ),

        pa.field(
            "stock_name",
            pa.string(),
        ),

        pa.field(
            "exchange",
            pa.string(),
        ),

        pa.field(
            "is_open",
            pa.bool_(),
        ),

        pa.field(
            "is_suspended",
            pa.bool_(),
        ),

        pa.field(
            "close_price",
            pa.float64(),
        ),

        pa.field(
            "adj_factor",
            pa.float64(),
        ),

        pa.field(
            "adjusted_close",
            pa.float64(),
        ),

        pa.field(
            "adj_factor_changed",
            pa.bool_(),
        ),

        pa.field(
            "chg_pct_raw",
            pa.float64(),
        ),

        pa.field(
            "vendor_return_decimal",
            pa.float64(),
        ),

        pa.field(
            "vendor_return_log",
            pa.float64(),
        ),

        # 主网络收益
        pa.field(
            "return_network",
            pa.float64(),
        ),

        pa.field(
            "return_network_simple",
            pa.float64(),
        ),

        # 上一个真实交易日收益
        pa.field(
            "return_last_trade_log",
            pa.float64(),
        ),

        pa.field(
            "return_last_trade_simple",
            pa.float64(),
        ),

        pa.field(
            "days_since_last_trade",
            pa.int32(),
        ),

        pa.field(
            "is_consecutive_open_pair",
            pa.bool_(),
        ),

        pa.field(
            "is_reopen_day",
            pa.bool_(),
        ),

        pa.field(
            "return_validation_diff",
            pa.float64(),
        ),

        pa.field(
            "volume",
            pa.float64(),
        ),

        pa.field(
            "turnover_value",
            pa.float64(),
        ),

        pa.field(
            "market_value",
            pa.float64(),
        ),

        pa.field(
            "neg_market_value",
            pa.float64(),
        ),

        pa.field(
            "turnover_rate",
            pa.float64(),
        ),

        pa.field(
            "industry_id1",
            pa.string(),
        ),

        pa.field(
            "industry_id2",
            pa.string(),
        ),
    ]
)


PANEL_COLUMNS = [
    field.name
    for field in PANEL_SCHEMA
]


# ============================================================
# 20. 强制 Panel 类型
# ============================================================

def prepare_panel_for_arrow(
    df,
):

    result = df.copy()

    string_cols = [
        "security_id",
        "source_security_id",
        "stock_code",
        "stock_name",
        "exchange",
        "industry_id1",
        "industry_id2",
    ]

    bool_cols = [
        "identity_reconciled",
        "is_open",
        "is_suspended",
        "adj_factor_changed",
        "is_consecutive_open_pair",
        "is_reopen_day",
    ]

    float_cols = [
        "close_price",
        "adj_factor",
        "adjusted_close",
        "chg_pct_raw",
        "vendor_return_decimal",
        "vendor_return_log",
        "return_network",
        "return_network_simple",
        "return_last_trade_log",
        "return_last_trade_simple",
        "return_validation_diff",
        "volume",
        "turnover_value",
        "market_value",
        "neg_market_value",
        "turnover_rate",
    ]

    for col in string_cols:

        result[
            col
        ] = (
            result[
                col
            ]
            .astype("string")
        )

    for col in bool_cols:

        result[
            col
        ] = (
            result[
                col
            ]
            .astype("boolean")
        )

    for col in float_cols:

        result[
            col
        ] = pd.to_numeric(
            result[
                col
            ],
            errors="coerce",
        ).astype(float)

    result[
        "days_since_last_trade"
    ] = (
        pd.to_numeric(
            result[
                "days_since_last_trade"
            ],
            errors="coerce",
        )
        .astype("Int32")
    )

    result[
        "trade_date"
    ] = pd.to_datetime(
        result[
            "trade_date"
        ]
    )

    return result[
        PANEL_COLUMNS
    ]


# ============================================================
# 21. 第二遍：
#
# 构造正式收益 Panel
# ============================================================

def build_return_panel(
    trade_dates,
    file_map,
    stock_master,
    reconciliation_map,
    vendor_scale_factor,
    adj_mode,
):

    print()
    print("=" * 72)
    print("Pass 2：构造标准 Daily Return Panel")
    print("=" * 72)

    panel_path = (
        OUTPUT_DIR
        / "daily_return_panel.parquet"
    )

    if panel_path.exists():

        panel_path.unlink()

    writer = pq.ParquetWriter(
        panel_path,
        PANEL_SCHEMA,
        compression="snappy",
    )

    # --------------------------------------------------------
    # 上一个 market date 状态
    # --------------------------------------------------------

    prev_adjusted_close = {}
    prev_factor = {}
    prev_open = {}
    prev_day_index = {}

    # --------------------------------------------------------
    # 上一个“实际开盘交易”状态
    #
    # 用于计算复牌 / 停牌跨期收益
    # --------------------------------------------------------

    last_trade_adjusted_close = {}
    last_trade_day_index = {}

    # --------------------------------------------------------
    # 全局 validation
    # --------------------------------------------------------

    global_validation_stats = (
        new_pair_stats()
    )

    validation_daily_rows = []

    anomaly_parts = []

    daily_summary_rows = []

    total_panel_rows = 0
    total_network_return = 0
    total_last_trade_return = 0
    total_reopen = 0

    try:

        for day_index, date in enumerate(
            trade_dates
        ):

            file = file_map[
                date
            ]

            raw = read_data_file(
                file
            )

            daily = standardize_daily(
                raw,
                date,
            )

            daily = build_research_daily(
                daily,
                date,
                stock_master,
                reconciliation_map,
            )

            sid = (
                daily[
                    "security_id"
                ]
                .astype("string")
            )

            # =================================================
            # Adjusted Close
            # =================================================

            curr_adj = adjusted_price(
                daily[
                    "close"
                ],
                daily[
                    "adj_factor"
                ],
                adj_mode,
            )

            curr_adj = pd.Series(
                curr_adj,
                index=daily.index,
                dtype=float,
            )

            # ------------------------------------------------
            # Previous market date
            # ------------------------------------------------

            prev_adj = (
                sid
                .map(
                    prev_adjusted_close
                )
                .astype(float)
            )

            p_factor = (
                sid
                .map(
                    prev_factor
                )
                .astype(float)
            )

            p_open = (
                sid
                .map(
                    prev_open
                )
            )

            p_day = (
                sid
                .map(
                    prev_day_index
                )
            )

            consecutive_day = (
                p_day
                .eq(
                    day_index
                    -
                    1
                )
                .fillna(
                    False
                )
            )

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

            consecutive_open_pair = (

                active

                &

                p_open
                .eq(
                    True
                )
                .fillna(
                    False
                )

                &

                consecutive_day

                &

                curr_adj.gt(0)

                &

                prev_adj.gt(0)
            )

            # =================================================
            # 正式 Network Return
            #
            # 只有 t 和 t-1 两个市场日均正常交易，
            # 才定义每日网络收益。
            # =================================================

            network_simple = pd.Series(
                np.nan,
                index=daily.index,
                dtype=float,
            )

            network_log = pd.Series(
                np.nan,
                index=daily.index,
                dtype=float,
            )

            network_simple.loc[
                consecutive_open_pair
            ] = (
                curr_adj.loc[
                    consecutive_open_pair
                ]
                /
                prev_adj.loc[
                    consecutive_open_pair
                ]
                -
                1.0
            )

            network_log.loc[
                consecutive_open_pair
            ] = np.log(
                curr_adj.loc[
                    consecutive_open_pair
                ]
                /
                prev_adj.loc[
                    consecutive_open_pair
                ]
            )

            # =================================================
            # Last-trade Return
            #
            # 如果中间经历停牌，
            # 使用上一个真正交易日计算累计价格变化。
            # =================================================

            last_adj = (
                sid
                .map(
                    last_trade_adjusted_close
                )
                .astype(float)
            )

            last_idx = (
                sid
                .map(
                    last_trade_day_index
                )
            )

            valid_last_trade = (

                active

                &

                curr_adj.gt(0)

                &

                last_adj.gt(0)

                &

                last_idx.notna()
            )

            last_trade_simple = pd.Series(
                np.nan,
                index=daily.index,
                dtype=float,
            )

            last_trade_log = pd.Series(
                np.nan,
                index=daily.index,
                dtype=float,
            )

            last_trade_simple.loc[
                valid_last_trade
            ] = (
                curr_adj.loc[
                    valid_last_trade
                ]
                /
                last_adj.loc[
                    valid_last_trade
                ]
                -
                1.0
            )

            last_trade_log.loc[
                valid_last_trade
            ] = np.log(
                curr_adj.loc[
                    valid_last_trade
                ]
                /
                last_adj.loc[
                    valid_last_trade
                ]
            )

            days_since_last_trade = (
                pd.Series(
                    day_index,
                    index=daily.index,
                    dtype=float,
                )
                -
                pd.to_numeric(
                    last_idx,
                    errors="coerce",
                )
            )

            days_since_last_trade = (
                days_since_last_trade
                .where(
                    valid_last_trade
                )
            )

            # ------------------------------------------------
            # Reopen
            #
            # 上一个市场日明确停牌，
            # 当前恢复交易。
            # ------------------------------------------------

            is_reopen_day = (

                active

                &

                consecutive_day

                &

                p_open
                .eq(
                    False
                )
                .fillna(
                    False
                )
            )

            # ------------------------------------------------
            # factor change
            # ------------------------------------------------

            factor_changed = (

                daily[
                    "adj_factor"
                ].gt(0)

                &

                p_factor.gt(0)

                &

                (
                    np.log(
                        daily[
                            "adj_factor"
                        ]
                        /
                        p_factor
                    )
                    .abs()
                    >
                    FACTOR_CHANGE_LOG_TOL
                )
            )

            factor_changed = (
                factor_changed
                .fillna(
                    False
                )
            )

            # =================================================
            # Vendor Return
            # =================================================

            vendor_return = (
                daily[
                    "chg_pct"
                ]
                *
                vendor_scale_factor
            )

            vendor_log = pd.Series(
                np.nan,
                index=daily.index,
                dtype=float,
            )

            valid_vendor_log = (
                vendor_return
                .notna()
                &
                (
                    vendor_return
                    >
                    -1
                )
            )

            vendor_log.loc[
                valid_vendor_log
            ] = np.log1p(
                vendor_return.loc[
                    valid_vendor_log
                ]
            )

            # =================================================
            # Validation
            #
            # 用 last-trade simple return 验证 chgPct。
            #
            # 原因：
            # 在复牌日，vendor chgPct 通常相对于上一个
            # 实际交易价格，而不是停牌日的 0 占位。
            # =================================================

            validation_return = (
                last_trade_simple
            )

            validation_mask = (

                validation_return
                .notna()

                &

                vendor_return
                .notna()

                &

                np.isfinite(
                    validation_return
                )

                &

                np.isfinite(
                    vendor_return
                )
            )

            validation_diff = pd.Series(
                np.nan,
                index=daily.index,
                dtype=float,
            )

            validation_diff.loc[
                validation_mask
            ] = (
                validation_return.loc[
                    validation_mask
                ]
                -
                vendor_return.loc[
                    validation_mask
                ]
            )

            update_pair_stats(
                global_validation_stats,
                validation_return.loc[
                    validation_mask
                ],
                vendor_return.loc[
                    validation_mask
                ],
            )

            # ------------------------------------------------
            # Daily validation
            # ------------------------------------------------

            if validation_mask.any():

                diff = (
                    validation_diff.loc[
                        validation_mask
                    ]
                    .to_numpy()
                )

                daily_validation = {

                    "trade_date":
                        date,

                    "validation_count":
                        int(
                            validation_mask.sum()
                        ),

                    "mae":
                        float(
                            np.abs(
                                diff
                            ).mean()
                        ),

                    "rmse":
                        float(
                            np.sqrt(
                                np.square(
                                    diff
                                ).mean()
                            )
                        ),

                    "max_abs_diff":
                        float(
                            np.abs(
                                diff
                            ).max()
                        ),

                    "within_1bp_ratio":
                        float(
                            (
                                np.abs(
                                    diff
                                )
                                <=
                                1e-4
                            ).mean()
                        ),

                    "within_5bp_ratio":
                        float(
                            (
                                np.abs(
                                    diff
                                )
                                <=
                                5e-4
                            ).mean()
                        ),
                }

            else:

                daily_validation = {

                    "trade_date":
                        date,

                    "validation_count":
                        0,

                    "mae":
                        np.nan,

                    "rmse":
                        np.nan,

                    "max_abs_diff":
                        np.nan,

                    "within_1bp_ratio":
                        np.nan,

                    "within_5bp_ratio":
                        np.nan,
                }

            validation_daily_rows.append(
                daily_validation
            )

            # =================================================
            # Return anomalies
            # =================================================

            anomaly_mask = (

                validation_mask

                &

                validation_diff
                .abs()
                .gt(
                    RETURN_DIFF_TOL
                )
            )

            if anomaly_mask.any():

                anomaly = (
                    daily.loc[
                        anomaly_mask,
                        [
                            "trade_date",
                            "security_id",
                            "source_security_id",
                            "stock_code",
                            "stock_name",
                            "exchange",
                            "close",
                            "adj_factor",
                            "chg_pct",
                            "is_open",
                        ],
                    ]
                    .copy()
                )

                anomaly[
                    "adjusted_close"
                ] = (
                    curr_adj.loc[
                        anomaly_mask
                    ]
                    .values
                )

                anomaly[
                    "computed_return"
                ] = (
                    validation_return.loc[
                        anomaly_mask
                    ]
                    .values
                )

                anomaly[
                    "vendor_return_decimal"
                ] = (
                    vendor_return.loc[
                        anomaly_mask
                    ]
                    .values
                )

                anomaly[
                    "abs_diff"
                ] = (
                    validation_diff.loc[
                        anomaly_mask
                    ]
                    .abs()
                    .values
                )

                anomaly[
                    "adj_factor_changed"
                ] = (
                    factor_changed.loc[
                        anomaly_mask
                    ]
                    .values
                )

                anomaly[
                    "is_reopen_day"
                ] = (
                    is_reopen_day.loc[
                        anomaly_mask
                    ]
                    .values
                )

                anomaly[
                    "issue_type"
                ] = (
                    "RETURN_VENDOR_MISMATCH"
                )

                anomaly[
                    "severity"
                ] = "REVIEW"

                anomaly_parts.append(
                    anomaly
                )

            # =================================================
            # 构建 Panel
            # =================================================

            panel = pd.DataFrame(
                {

                    "trade_date":
                        date,

                    "security_id":
                        daily[
                            "security_id"
                        ],

                    "source_security_id":
                        daily[
                            "source_security_id"
                        ],

                    "identity_reconciled":
                        daily[
                            "identity_reconciled"
                        ],

                    "stock_code":
                        daily[
                            "stock_code"
                        ],

                    "stock_name":
                        daily[
                            "stock_name"
                        ],

                    "exchange":
                        daily[
                            "exchange"
                        ],

                    "is_open":
                        daily[
                            "is_open"
                        ],

                    "is_suspended":
                        daily[
                            "is_open"
                        ]
                        .eq(
                            False
                        ),

                    "close_price":
                        daily[
                            "close"
                        ],

                    "adj_factor":
                        daily[
                            "adj_factor"
                        ],

                    "adjusted_close":
                        curr_adj,

                    "adj_factor_changed":
                        factor_changed,

                    "chg_pct_raw":
                        daily[
                            "chg_pct"
                        ],

                    "vendor_return_decimal":
                        vendor_return,

                    "vendor_return_log":
                        vendor_log,

                    # 这里是后续网络默认收益列
                    "return_network":
                        network_log,

                    "return_network_simple":
                        network_simple,

                    # 复牌 / gap robustness
                    "return_last_trade_log":
                        last_trade_log,

                    "return_last_trade_simple":
                        last_trade_simple,

                    "days_since_last_trade":
                        days_since_last_trade,

                    "is_consecutive_open_pair":
                        consecutive_open_pair,

                    "is_reopen_day":
                        is_reopen_day,

                    "return_validation_diff":
                        validation_diff,

                    "volume":
                        daily[
                            "volume"
                        ],

                    "turnover_value":
                        daily[
                            "turnover_value"
                        ],

                    "market_value":
                        daily[
                            "market_value"
                        ],

                    "neg_market_value":
                        daily[
                            "neg_market_value"
                        ],

                    "turnover_rate":
                        daily[
                            "turnover_rate"
                        ],

                    "industry_id1":
                        daily[
                            "industry1"
                        ],

                    "industry_id2":
                        daily[
                            "industry2"
                        ],
                }
            )

            panel = (
                prepare_panel_for_arrow(
                    panel
                )
            )

            table = pa.Table.from_pandas(
                panel,
                schema=PANEL_SCHEMA,
                preserve_index=False,
                safe=False,
            )

            writer.write_table(
                table
            )

            # =================================================
            # Daily panel summary
            # =================================================

            daily_summary_rows.append(
                {

                    "trade_date":
                        date,

                    "panel_count":
                        len(
                            panel
                        ),

                    "active_count":
                        int(
                            active.sum()
                        ),

                    "suspended_count":
                        int(
                            (
                                daily[
                                    "is_open"
                                ]
                                .eq(
                                    False
                                )
                                .fillna(
                                    False
                                )
                            ).sum()
                        ),

                    "network_return_count":
                        int(
                            network_log
                            .notna()
                            .sum()
                        ),

                    "last_trade_return_count":
                        int(
                            last_trade_log
                            .notna()
                            .sum()
                        ),

                    "reopen_count":
                        int(
                            is_reopen_day.sum()
                        ),

                    "factor_change_count":
                        int(
                            factor_changed.sum()
                        ),

                    "validation_count":
                        int(
                            validation_mask.sum()
                        ),

                    "validation_anomaly_count":
                        int(
                            anomaly_mask.sum()
                        ),
                }
            )

            total_panel_rows += (
                len(
                    panel
                )
            )

            total_network_return += (
                int(
                    network_log
                    .notna()
                    .sum()
                )
            )

            total_last_trade_return += (
                int(
                    last_trade_log
                    .notna()
                    .sum()
                )
            )

            total_reopen += (
                int(
                    is_reopen_day.sum()
                )
            )

            # =================================================
            # 更新 previous market-day state
            # =================================================

            prev_adjusted_close.update(
                zip(
                    sid,
                    curr_adj,
                )
            )

            prev_factor.update(
                zip(
                    sid,
                    daily[
                        "adj_factor"
                    ],
                )
            )

            prev_open.update(
                zip(
                    sid,
                    daily[
                        "is_open"
                    ],
                )
            )

            prev_day_index.update(
                zip(
                    sid,
                    [day_index]
                    *
                    len(
                        daily
                    ),
                )
            )

            # =================================================
            # 更新 last actual trading state
            # =================================================

            valid_current_trade = (

                active

                &

                curr_adj.gt(0)
            )

            if valid_current_trade.any():

                active_sid = (
                    sid.loc[
                        valid_current_trade
                    ]
                )

                active_adj = (
                    curr_adj.loc[
                        valid_current_trade
                    ]
                )

                last_trade_adjusted_close.update(
                    zip(
                        active_sid,
                        active_adj,
                    )
                )

                last_trade_day_index.update(
                    zip(
                        active_sid,
                        [day_index]
                        *
                        len(
                            active_sid
                        ),
                    )
                )

            if (
                day_index == 0
                or
                (
                    day_index
                    +
                    1
                )
                % 250
                == 0
            ):

                print(
                    f"[Pass2 "
                    f"{day_index + 1:4d}/"
                    f"{len(trade_dates)}] "
                    f"{date.date()}"
                )

    finally:

        writer.close()

    # ========================================================
    # Validation summary
    # ========================================================

    validation_metrics = (
        finalize_pair_stats(
            global_validation_stats
        )
    )

    validation_summary = pd.DataFrame(
        [
            {
                "metric":
                    "validation_count",

                "value":
                    validation_metrics[
                        "count"
                    ],
            },

            {
                "metric":
                    "mae",

                "value":
                    validation_metrics[
                        "mae"
                    ],
            },

            {
                "metric":
                    "rmse",

                "value":
                    validation_metrics[
                        "rmse"
                    ],
            },

            {
                "metric":
                    "correlation",

                "value":
                    validation_metrics[
                        "corr"
                    ],
            },

            {
                "metric":
                    "max_abs_diff",

                "value":
                    validation_metrics[
                        "max_abs"
                    ],
            },

            {
                "metric":
                    "return_diff_tolerance",

                "value":
                    RETURN_DIFF_TOL,
            },
        ]
    )

    validation_summary.to_csv(
        OUTPUT_DIR
        / "return_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_daily = pd.DataFrame(
        validation_daily_rows
    )

    validation_daily.to_csv(
        OUTPUT_DIR
        / "return_validation_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Return anomalies
    # ========================================================

    if anomaly_parts:

        anomalies = pd.concat(
            anomaly_parts,
            ignore_index=True,
        )

    else:

        anomalies = pd.DataFrame()

    anomalies.to_parquet(
        OUTPUT_DIR
        / "return_anomalies.parquet",
        index=False,
    )

    # ========================================================
    # Panel daily summary
    # ========================================================

    panel_daily_summary = pd.DataFrame(
        daily_summary_rows
    )

    panel_daily_summary.to_csv(
        OUTPUT_DIR
        / "return_panel_daily_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Overall panel summary
    # ========================================================

    panel_summary = pd.DataFrame(
        {

            "metric": [

                "total_panel_stock_day_records",

                "network_return_nonmissing_count",

                "last_trade_return_nonmissing_count",

                "reopen_stock_day_count",

                "return_validation_anomaly_count",

            ],

            "value": [

                total_panel_rows,

                total_network_return,

                total_last_trade_return,

                total_reopen,

                len(
                    anomalies
                ),
            ],
        }
    )

    panel_summary.to_csv(
        OUTPUT_DIR
        / "return_panel_quality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "panel_path":
            str(
                panel_path
            ),

        "total_panel_rows":
            total_panel_rows,

        "total_network_return":
            total_network_return,

        "total_last_trade_return":
            total_last_trade_return,

        "total_reopen":
            total_reopen,

        "validation_metrics":
            validation_metrics,

        "anomaly_count":
            len(
                anomalies
            ),
    }


# ============================================================
# 22. 主程序
# ============================================================

def main():

    print("=" * 72)
    print("M1 Stage 3")
    print("Return Validation + Daily Return Panel")
    print("=" * 72)

    # --------------------------------------------------------
    # 基础数据
    # --------------------------------------------------------

    stock_master = (
        load_stock_master()
    )

    trade_dates = (
        load_trade_dates()
    )

    file_map = (
        build_daily_file_map(
            trade_dates
        )
    )

    reconciliation_map = (
        load_identity_reconciliation()
    )

    print(
        f"历史证券主体："
        f"{stock_master['security_id'].nunique():,}"
    )

    print(
        f"交易日："
        f"{len(trade_dates):,}"
    )

    # ========================================================
    # Pass 1:
    # 自动识别 Return Definition
    # ========================================================

    (
        selected_scale,
        vendor_scale_factor,
        selected_adj_mode,
        diagnostics,
    ) = (
        infer_return_definition(
            trade_dates,
            file_map,
            stock_master,
            reconciliation_map,
        )
    )

    # ========================================================
    # Pass 2:
    # 构造 Panel
    # ========================================================

    panel_result = (
        build_return_panel(
            trade_dates,
            file_map,
            stock_master,
            reconciliation_map,
            vendor_scale_factor,
            selected_adj_mode,
        )
    )

    # ========================================================
    # Metadata
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

        "trade_day_count":
            int(
                len(
                    trade_dates
                )
            ),

        "selected_chg_pct_scale":
            selected_scale,

        "vendor_scale_factor":
            vendor_scale_factor,

        "selected_adj_factor_mode":
            selected_adj_mode,

        "network_return_definition":
            (
                "log(adjusted_close_t / "
                "adjusted_close_t-1), "
                "only when both current and previous "
                "market-day observations have isOpen=1"
            ),

        "last_trade_return_definition":
            (
                "log(adjusted_close_t / "
                "adjusted_close_previous_actual_trade)"
            ),

        "suspension_rule":
            (
                "isOpen=0 observations remain in panel, "
                "but return_network is NA rather than zero."
            ),

        "reopen_rule":
            (
                "first reopen-day return_network remains NA "
                "under the conservative network definition; "
                "return_last_trade_log preserves the cumulative "
                "price change since the previous actual trade."
            ),

        "return_diff_tolerance":
            RETURN_DIFF_TOL,

        **panel_result,
    }

    with open(
        OUTPUT_DIR
        / "stage3_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 72)
    print("Stage 3 完成")
    print("=" * 72)

    print(
        f"chgPct scale："
        f"{selected_scale}"
    )

    print(
        f"AdjFactor mode："
        f"{selected_adj_mode}"
    )

    print(
        f"Panel stock-day："
        f"{panel_result['total_panel_rows']:,}"
    )

    print(
        f"Network Return 非缺失："
        f"{panel_result['total_network_return']:,}"
    )

    print(
        f"Last-trade Return 非缺失："
        f"{panel_result['total_last_trade_return']:,}"
    )

    print(
        f"Reopen stock-day："
        f"{panel_result['total_reopen']:,}"
    )

    metrics = (
        panel_result[
            "validation_metrics"
        ]
    )

    print()
    print(
        "Return Validation："
    )

    print(
        f"样本数："
        f"{metrics['count']:,}"
    )

    print(
        f"MAE："
        f"{metrics['mae']:.10f}"
    )

    print(
        f"RMSE："
        f"{metrics['rmse']:.10f}"
    )

    print(
        f"Correlation："
        f"{metrics['corr']:.10f}"
    )

    print(
        f"超过 {RETURN_DIFF_TOL:.6f} 的记录："
        f"{panel_result['anomaly_count']:,}"
    )

    print()
    print(
        f"输出目录：{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
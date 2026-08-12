from pathlib import Path
import time

import akshare as ak
import pandas as pd


# ============================================================
# 1. 参数
# ============================================================

START_DATE = "20230101"
END_DATE = "20260731"

MIN_COVERAGE = 0.95


# ============================================================
# 2. 股票池
# ============================================================

stock_pool = {

    # 银行
    "000001": {"name": "平安银行", "industry": "银行"},
    "600036": {"name": "招商银行", "industry": "银行"},
    "601398": {"name": "工商银行", "industry": "银行"},
    "601288": {"name": "农业银行", "industry": "银行"},

    # 食品饮料
    "600519": {"name": "贵州茅台", "industry": "食品饮料"},
    "000858": {"name": "五粮液", "industry": "食品饮料"},
    "000568": {"name": "泸州老窖", "industry": "食品饮料"},
    "600887": {"name": "伊利股份", "industry": "食品饮料"},

    # 医药
    "600276": {"name": "恒瑞医药", "industry": "医药"},
    "000538": {"name": "云南白药", "industry": "医药"},
    "300760": {"name": "迈瑞医疗", "industry": "医药"},

    # 新能源 / 汽车
    "300750": {"name": "宁德时代", "industry": "新能源"},
    "002594": {"name": "比亚迪", "industry": "汽车"},
    "601012": {"name": "隆基绿能", "industry": "新能源"},
    "002460": {"name": "赣锋锂业", "industry": "新能源"},

    # 电子
    "002415": {"name": "海康威视", "industry": "电子"},
    "000725": {"name": "京东方A", "industry": "电子"},
    "603501": {"name": "韦尔股份", "industry": "电子"},
    "002475": {"name": "立讯精密", "industry": "电子"},

    # 非银金融
    "601318": {"name": "中国平安", "industry": "非银金融"},
    "600030": {"name": "中信证券", "industry": "非银金融"},
}


# ============================================================
# 3. 项目目录
# ============================================================

PROJECT_DIR = Path("stock_network")

RAW_DIR = (
    PROJECT_DIR /
    "data" /
    "raw"
)

PROCESSED_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. 单股票下载函数
# ============================================================

def download_one_stock(
    code,
    start_date,
    end_date,
    max_retries=3
):

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df.empty:
                raise ValueError(
                    "返回数据为空"
                )

            required_columns = {
                "日期",
                "收盘"
            }

            if not required_columns.issubset(
                df.columns
            ):
                raise ValueError(
                    "缺少日期或收盘字段"
                )

            df["日期"] = pd.to_datetime(
                df["日期"]
            )

            df = (
                df
                .sort_values("日期")
                .drop_duplicates(
                    subset="日期",
                    keep="last"
                )
            )

            print(
                f"{code}: "
                f"{len(df)} 个交易日"
            )

            return df

        except Exception as e:

            print(
                f"{code} "
                f"第 {attempt} 次失败："
                f"{e}"
            )

            if attempt < max_retries:
                time.sleep(
                    2 * attempt
                )

    return None


# ============================================================
# 5. 下载股票数据
# ============================================================

price_series = {}

for code, info in stock_pool.items():

    print(
        f"\n下载 "
        f"{code} "
        f"{info['name']}"
    )

    df = download_one_stock(
        code=code,
        start_date=START_DATE,
        end_date=END_DATE
    )

    if df is None:
        continue

    # 保存单只股票原始数据
    df.to_csv(
        RAW_DIR / f"{code}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 提取前复权收盘价
    close = (
        df[["日期", "收盘"]]
        .set_index("日期")["收盘"]
        .rename(code)
    )

    price_series[code] = close

    time.sleep(0.5)


# ============================================================
# 6. 构建价格矩阵
# ============================================================

if not price_series:
    raise RuntimeError(
        "没有成功下载任何股票数据"
    )

prices = pd.concat(
    price_series.values(),
    axis=1,
    join="outer"
)

prices = prices.sort_index()


# ============================================================
# 7. 数据质量检查
# ============================================================

quality_rows = []

for code in prices.columns:

    s = prices[code]

    valid = s.dropna()

    quality_rows.append({

        "code":
            code,

        "name":
            stock_pool[code]["name"],

        "industry":
            stock_pool[code]["industry"],

        "n_obs":
            valid.shape[0],

        "first_date":
            valid.index.min()
            if not valid.empty
            else pd.NaT,

        "last_date":
            valid.index.max()
            if not valid.empty
            else pd.NaT,

        "n_missing":
            s.isna().sum(),

        "coverage_rate":
            s.notna().mean()
    })


quality = pd.DataFrame(
    quality_rows
)


# ============================================================
# 8. 根据覆盖率过滤
# ============================================================

valid_codes = quality.loc[
    quality["coverage_rate"]
    >= MIN_COVERAGE,
    "code"
].tolist()

prices_clean = (
    prices[valid_codes]
    .copy()
)


# ============================================================
# 9. 股票信息
# ============================================================

stock_info = pd.DataFrame(
    [
        {
            "code": code,
            "name": stock_pool[code]["name"],
            "industry":
                stock_pool[code]["industry"]
        }
        for code in valid_codes
    ]
)


# ============================================================
# 10. 保存
# ============================================================

prices_clean.to_csv(
    PROCESSED_DIR /
    "prices_qfq.csv",
    encoding="utf-8-sig"
)

stock_info.to_csv(
    PROCESSED_DIR /
    "stock_info.csv",
    index=False,
    encoding="utf-8-sig"
)

quality.to_csv(
    PROCESSED_DIR /
    "data_quality.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 输出摘要
# ============================================================

print("\n==============================")
print("数据准备完成")
print("==============================")

print(
    "日期范围：",
    prices_clean.index.min(),
    "至",
    prices_clean.index.max()
)

print(
    "股票数量：",
    prices_clean.shape[1]
)

print(
    "交易日期数量：",
    prices_clean.shape[0]
)

print("\n数据覆盖率：")

print(
    quality[
        [
            "code",
            "name",
            "industry",
            "coverage_rate"
        ]
    ]
    .sort_values(
        "coverage_rate"
    )
    .to_string(
        index=False
    )
)
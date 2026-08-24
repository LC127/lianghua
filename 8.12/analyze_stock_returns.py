from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_DIR = Path("stock_network")

PROCESSED_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

FIGURE_DIR = PROJECT_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PRICE_FILE = (
    PROCESSED_DIR /
    "prices_qfq.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR /
    "stock_info.csv"
)


# ============================================================
# 2. 参数设置
# ============================================================

# 一对股票至少需要多少个共同有效收益率观测，
# 才计算二者之间的相关系数。
MIN_PAIR_OBS = 120

# 一年交易日约为 252 天，
# 用于计算年化波动率。
TRADING_DAYS = 252

# 输出相关性最高的前多少对股票。
TOP_N_PAIRS = 30


# ============================================================
# 3. 读取价格数据
# ============================================================

prices = pd.read_csv(
    PRICE_FILE,
    index_col=0,
    parse_dates=True
)

stock_info = pd.read_csv(
    STOCK_INFO_FILE,
    dtype={"code": str}
)


# 股票代码统一补齐为六位字符串
stock_info["code"] = (
    stock_info["code"]
    .str.zfill(6)
)

prices.columns = (
    prices.columns
    .astype(str)
    .str.zfill(6)
)


# ============================================================
# 4. 基础数据检查
# ============================================================

# 日期排序
prices = prices.sort_index()

# 删除重复交易日
prices = prices.loc[
    ~prices.index.duplicated(
        keep="last"
    )
]

# 转换为数值
prices = prices.apply(
    pd.to_numeric,
    errors="coerce"
)


# 检查是否存在非正价格
non_positive_count = (
    prices <= 0
).sum().sum()

if non_positive_count > 0:

    print(
        f"警告：发现 "
        f"{non_positive_count} "
        f"个非正价格。"
    )

    # 对数收益率要求价格必须 > 0
    prices = prices.mask(
        prices <= 0
    )


print(
    "价格矩阵维度：",
    prices.shape
)

print(
    "价格日期范围：",
    prices.index.min(),
    "至",
    prices.index.max()
)

print(
    "股票数量：",
    prices.shape[1]
)


# ============================================================
# 5. 计算日对数收益率
# ============================================================

log_prices = np.log(prices)

returns = log_prices.diff()


# 注意：
# 不使用 fillna(0)
# 也不使用前向填充
# 缺失价格对应的收益继续保留为 NaN


# ============================================================
# 6. 删除完全没有收益率的日期
# ============================================================

returns = returns.dropna(
    how="all"
)


# ============================================================
# 7. 保存收益率矩阵
# ============================================================

RETURN_FILE = (
    PROCESSED_DIR /
    "stock_returns.csv"
)

returns.to_csv(
    RETURN_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 收益率描述统计
# ============================================================

return_summary = pd.DataFrame(
    index=returns.columns
)

return_summary.index.name = "code"


return_summary["n_obs"] = (
    returns.notna().sum()
)

return_summary["mean_daily_return"] = (
    returns.mean()
)

return_summary["std_daily_return"] = (
    returns.std()
)

return_summary["annualized_volatility"] = (
    returns.std()
    * np.sqrt(TRADING_DAYS)
)

return_summary["min_return"] = (
    returns.min()
)

return_summary["max_return"] = (
    returns.max()
)


# 合并股票名称和行业
stock_info_indexed = (
    stock_info
    .set_index("code")
)

return_summary = (
    return_summary
    .join(
        stock_info_indexed[
            ["name", "industry"]
        ],
        how="left"
    )
)


return_summary.to_csv(
    PROCESSED_DIR /
    "return_summary.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 9. 计算股票之间共同有效观测数量
# ============================================================

valid_matrix = (
    returns
    .notna()
    .astype(int)
)

pairwise_nobs = (
    valid_matrix.T
    @ valid_matrix
)


pairwise_nobs.to_csv(
    PROCESSED_DIR /
    "pairwise_observations.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 10. 计算 Pearson 相关矩阵
# ============================================================

corr = returns.corr(
    method="pearson",
    min_periods=MIN_PAIR_OBS
)


corr.to_csv(
    PROCESSED_DIR /
    "stock_correlation.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 11. 找出相关性最高的股票对
# ============================================================

corr_values = corr.values

upper_triangle = np.triu(
    np.ones(
        corr_values.shape,
        dtype=bool
    ),
    k=1
)

rows, cols = np.where(
    upper_triangle
)


pair_rows = []

for i, j in zip(rows, cols):

    rho = corr.iloc[i, j]

    if pd.isna(rho):
        continue

    code_i = corr.index[i]
    code_j = corr.columns[j]

    pair_rows.append(
        {
            "stock_1":
                code_i,

            "stock_1_name":
                stock_info_indexed
                .get(
                    "name",
                    pd.Series()
                )
                .get(
                    code_i,
                    np.nan
                ),

            "industry_1":
                stock_info_indexed
                .get(
                    "industry",
                    pd.Series()
                )
                .get(
                    code_i,
                    np.nan
                ),

            "stock_2":
                code_j,

            "stock_2_name":
                stock_info_indexed
                .get(
                    "name",
                    pd.Series()
                )
                .get(
                    code_j,
                    np.nan
                ),

            "industry_2":
                stock_info_indexed
                .get(
                    "industry",
                    pd.Series()
                )
                .get(
                    code_j,
                    np.nan
                ),

            "correlation":
                rho,

            "n_common":
                pairwise_nobs
                .loc[
                    code_i,
                    code_j
                ]
        }
    )


correlation_pairs = pd.DataFrame(
    pair_rows
)

correlation_pairs = (
    correlation_pairs
    .sort_values(
        "correlation",
        ascending=False
    )
)


top_pairs = (
    correlation_pairs
    .head(TOP_N_PAIRS)
)


top_pairs.to_csv(
    PROCESSED_DIR /
    "top_correlation_pairs.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 按行业对股票排序
# ============================================================

plot_info = (
    stock_info[
        stock_info["code"]
        .isin(corr.columns)
    ]
    .copy()
)


plot_info = (
    plot_info
    .sort_values(
        ["industry", "code"]
    )
)


ordered_codes = (
    plot_info["code"]
    .tolist()
)


# 避免 stock_info 中少量股票信息缺失
remaining_codes = [
    code
    for code in corr.columns
    if code not in ordered_codes
]

ordered_codes += remaining_codes


corr_plot = corr.loc[
    ordered_codes,
    ordered_codes
]


# ============================================================
# 13. 绘制相关矩阵热力图
# ============================================================

fig_size = max(
    10,
    len(ordered_codes) * 0.45
)

plt.figure(
    figsize=(
        fig_size,
        fig_size
    )
)

image = plt.imshow(
    corr_plot.values,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(
    image,
    label="Pearson correlation"
)

plt.xticks(
    range(
        len(ordered_codes)
    ),
    ordered_codes,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(
        len(ordered_codes)
    ),
    ordered_codes,
    fontsize=7
)

plt.xlabel("Stock")
plt.ylabel("Stock")

plt.title(
    "Stock Return Correlation Matrix"
)

plt.tight_layout()


HEATMAP_FILE = (
    FIGURE_DIR /
    "correlation_heatmap.png"
)

plt.savefig(
    HEATMAP_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. 输出结果摘要
# ============================================================

print(
    "\n================================"
)

print(
    "第四阶段处理完成"
)

print(
    "================================"
)

print(
    "收益率矩阵：",
    returns.shape
)

print(
    "相关矩阵：",
    corr.shape
)

print(
    "\n前 10 对最高相关股票："
)

print(
    top_pairs[
        [
            "stock_1",
            "stock_1_name",
            "industry_1",
            "stock_2",
            "stock_2_name",
            "industry_2",
            "correlation",
            "n_common"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)

print(
    "\n输出文件："
)

print(
    RETURN_FILE
)

print(
    PROCESSED_DIR /
    "stock_correlation.csv"
)

print(
    HEATMAP_FILE
)
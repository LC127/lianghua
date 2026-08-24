from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_DIR = Path("stock_network")

PROCESSED_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

RETURN_FILE = (
    PROCESSED_DIR /
    "stock_returns.csv"
)


COV_FILE = (
    PROCESSED_DIR /
    "stock_covariance.csv"
)

CORR_FILE = (
    PROCESSED_DIR /
    "stock_correlation_stage2.csv"
)

CORR_FROM_COV_FILE = (
    PROCESSED_DIR /
    "stock_correlation_from_covariance.csv"
)


# ============================================================
# 2. 读取得到的对数收益率
# ============================================================

returns = pd.read_csv(
    RETURN_FILE,
    index_col=0,
    parse_dates=True
)

# 股票代码统一为6位字符串
returns.columns = (
    returns.columns
    .astype(str)
    .str.zfill(6)
)

returns = returns.sort_index()


print(
    "原始收益率矩阵维度：",
    returns.shape
)

print(
    "\n各股票缺失值数量："
)

print(
    returns.isna().sum()
)


# ============================================================
# 3. 为了严格验证矩阵关系，使用共同完整样本
# ============================================================

returns_complete = (
    returns
    .dropna(
        axis=0,
        how="any"
    )
)


print(
    "\n完整样本收益率矩阵维度：",
    returns_complete.shape
)


# ============================================================
# 4. 计算样本均值
# ============================================================

mean_returns = (
    returns_complete.mean()
)


print(
    "\n各股票日平均对数收益率："
)

print(
    mean_returns
)


# ============================================================
# 5. 计算样本协方差矩阵
# ============================================================

cov_matrix = (
    returns_complete
    .cov()
)


print(
    "\n样本协方差矩阵维度：",
    cov_matrix.shape
)

print(
    "\n样本协方差矩阵："
)

print(
    cov_matrix.round(8)
)


# 保存协方差矩阵
cov_matrix.to_csv(
    COV_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 6. 从协方差矩阵提取样本方差和样本标准差
# ============================================================

# 协方差矩阵对角线 = 各股票样本方差
sample_variance = pd.Series(
    np.diag(
        cov_matrix.values
    ),
    index=cov_matrix.index,
    name="variance"
)


# 标准差 = 方差开平方
sample_std = np.sqrt(
    sample_variance
)

sample_std.name = "std"


variance_std_table = pd.concat(
    [
        sample_variance,
        sample_std
    ],
    axis=1
)


print(
    "\n样本方差与标准差："
)

print(
    variance_std_table
)


# ============================================================
# 7. 直接计算 Pearson 相关矩阵
# ============================================================

corr_direct = (
    returns_complete
    .corr(
        method="pearson"
    )
)


print(
    "\n直接计算得到的 Pearson 相关矩阵："
)

print(
    corr_direct.round(4)
)


corr_direct.to_csv(
    CORR_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 根据协方差矩阵手动构造 Pearson 相关矩阵
#
#    R = D^(-1/2) S D^(-1/2)
# ============================================================

std_array = (
    sample_std
    .values
)


# 外积：
#
# std_outer[i,j]
# =
# std_i * std_j
#
std_outer = np.outer(
    std_array,
    std_array
)


# Pearson相关系数：
#
# rho_ij
# =
# covariance_ij
# /
# (std_i * std_j)
#
corr_from_cov_values = (
    cov_matrix.values
    /
    std_outer
)


corr_from_cov = pd.DataFrame(
    corr_from_cov_values,
    index=cov_matrix.index,
    columns=cov_matrix.columns
)


print(
    "\n由协方差矩阵标准化得到的 Pearson 相关矩阵："
)

print(
    corr_from_cov.round(4)
)


corr_from_cov.to_csv(
    CORR_FROM_COV_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 验证：
#
#    corr_direct
#    是否等于
#    corr_from_cov
# ============================================================

difference = (
    corr_direct
    -
    corr_from_cov
)


max_abs_difference = (
    np.abs(
        difference.values
    )
    .max()
)


mean_abs_difference = (
    np.abs(
        difference.values
    )
    .mean()
)


print(
    "\n================================"
)

print(
    "Pearson相关矩阵验证"
)

print(
    "================================"
)


print(
    "\n最大绝对误差：",
    max_abs_difference
)

print(
    "平均绝对误差：",
    mean_abs_difference
)


if np.allclose(
    corr_direct.values,
    corr_from_cov.values,
    atol=1e-12
):

    print(
        "\n验证成功："
        "Pearson相关矩阵等于"
        "样本协方差矩阵标准化后的结果。"
    )

else:

    print(
        "\n两种结果存在非忽略差异，"
        "需要检查缺失值处理或样本口径。"
    )


# ============================================================
# 10. 进一步用矩阵公式验证
#
# R = D^(-1/2) S D^(-1/2)
# ============================================================

D_inverse_sqrt = np.diag(
    1.0
    /
    std_array
)


corr_matrix_formula = (
    D_inverse_sqrt
    @ cov_matrix.values
    @ D_inverse_sqrt
)


matrix_formula_difference = (
    np.abs(
        corr_matrix_formula
        -
        corr_direct.values
    )
    .max()
)


print(
    "\n矩阵公式验证的最大误差：",
    matrix_formula_difference
)


# ============================================================
# 11. 手工验证一个股票对
# ============================================================

stock_i = returns_complete.columns[0]
stock_j = returns_complete.columns[1]


cov_ij = (
    cov_matrix.loc[
        stock_i,
        stock_j
    ]
)


std_i = (
    sample_std.loc[
        stock_i
    ]
)


std_j = (
    sample_std.loc[
        stock_j
    ]
)


rho_manual = (
    cov_ij
    /
    (
        std_i
        *
        std_j
    )
)


rho_direct = (
    corr_direct.loc[
        stock_i,
        stock_j
    ]
)


print(
    "\n================================"
)

print(
    "单个股票对验证"
)

print(
    "================================"
)


print(
    f"\n股票："
    f"{stock_i} 与 {stock_j}"
)

print(
    f"样本协方差 = "
    f"{cov_ij:.10f}"
)

print(
    f"{stock_i} 标准差 = "
    f"{std_i:.10f}"
)

print(
    f"{stock_j} 标准差 = "
    f"{std_j:.10f}"
)

print(
    f"\n手工计算 Pearson："
    f"{rho_manual:.10f}"
)

print(
    f"pandas 直接计算 Pearson："
    f"{rho_direct:.10f}"
)


# ============================================================
# 12. 检查协方差矩阵能否稳定求逆, 为计算精度矩阵做准备
# ============================================================

eigenvalues = np.linalg.eigvalsh(
    cov_matrix.values
)


condition_number = np.linalg.cond(
    cov_matrix.values
)


rank = np.linalg.matrix_rank(
    cov_matrix.values
)


print(
    "\n================================"
)

print(
    "协方差矩阵数值检查"
)

print(
    "================================"
)


print(
    "\n矩阵维度：",
    cov_matrix.shape
)

print(
    "矩阵秩：",
    rank
)

print(
    "最小特征值：",
    eigenvalues.min()
)

print(
    "最大特征值：",
    eigenvalues.max()
)

print(
    "条件数：",
    condition_number
)


if rank == cov_matrix.shape[0]:

    print(
        "\n协方差矩阵满秩，"
        "从线性代数角度可以求逆。"
    )

else:

    print(
        "\n协方差矩阵不满秩，"
        "不能直接求普通逆矩阵。"
    )


# ============================================================
# 13. 完成
# ============================================================

print(
    "\n================================"
)

print(
    "阶段二完成"
)

print(
    "================================"
)


print(
    "\n输出文件："
)

print(
    COV_FILE
)

print(
    CORR_FILE
)

print(
    CORR_FROM_COV_FILE
)
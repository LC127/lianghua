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


COV_FILE = (
    PROCESSED_DIR /
    "stock_covariance.csv"
)

PRECISION_FILE = (
    PROCESSED_DIR /
    "precision_matrix.csv"
)

PARTIAL_CORR_FILE = (
    PROCESSED_DIR /
    "partial_correlation.csv"
)


# ============================================================
# 2. 读取得到的样本协方差矩阵
# ============================================================

cov_matrix = pd.read_csv(
    COV_FILE,
    index_col=0
)


# 股票代码统一为6位字符串
cov_matrix.index = (
    cov_matrix.index
    .astype(str)
    .str.zfill(6)
)

cov_matrix.columns = (
    cov_matrix.columns
    .astype(str)
    .str.zfill(6)
)


print(
    "样本协方差矩阵维度：",
    cov_matrix.shape
)


print(
    "\n样本协方差矩阵："
)

print(
    cov_matrix.round(8)
)


# ============================================================
# 3. 检查协方差矩阵是否对称
# ============================================================

is_symmetric = np.allclose(
    cov_matrix.values,
    cov_matrix.values.T,
    atol=1e-12
)


print(
    "\n协方差矩阵是否对称：",
    is_symmetric
)


if not is_symmetric:
    raise ValueError(
        "协方差矩阵不是对称矩阵，请检查阶段二的数据处理。"
    )


# ============================================================
# 4. 检查协方差矩阵是否可以稳定求逆
# ============================================================

cov_values = cov_matrix.values


rank = np.linalg.matrix_rank(
    cov_values
)


dimension = (
    cov_values.shape[0]
)


eigenvalues = np.linalg.eigvalsh(
    cov_values
)


condition_number = np.linalg.cond(
    cov_values
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
    "矩阵维度：",
    dimension
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


if rank < dimension:

    raise ValueError(
        "样本协方差矩阵不满秩，"
        "无法直接计算普通逆矩阵。"
    )


# ============================================================
# 5. 计算精度矩阵
#
# Omega_hat = Sigma_hat^{-1}
# ============================================================

precision_values = np.linalg.inv(
    cov_values
)


precision_matrix = pd.DataFrame(
    precision_values,
    index=cov_matrix.index,
    columns=cov_matrix.columns
)


print(
    "\n================================"
)

print(
    "精度矩阵"
)

print(
    "================================"
)


print(
    precision_matrix.round(4)
)


# ============================================================
# 6. 验证精度矩阵
#
# Sigma_hat @ Omega_hat ≈ I
# ============================================================

identity_check = (
    cov_values
    @ precision_values
)


identity_matrix = np.eye(
    dimension
)


max_inverse_error = np.max(
    np.abs(
        identity_check
        -
        identity_matrix
    )
)


print(
    "\n协方差矩阵 × 精度矩阵"
    " 与单位矩阵之间的最大误差："
)

print(
    max_inverse_error
)


# ============================================================
# 7. 保存精度矩阵
# ============================================================

precision_matrix.to_csv(
    PRECISION_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 从精度矩阵计算偏相关矩阵
#
# rho_ij|rest
# =
# - Omega_ij /
# sqrt(Omega_ii * Omega_jj)
# ============================================================

precision_diag = np.diag(
    precision_values
)


# 精度矩阵正定时对角线应为正
if np.any(
    precision_diag <= 0
):

    raise ValueError(
        "精度矩阵存在非正对角元素，"
        "请检查协方差矩阵的数值性质。"
    )


sqrt_precision_diag = np.sqrt(
    precision_diag
)


normalization = np.outer(
    sqrt_precision_diag,
    sqrt_precision_diag
)


partial_corr_values = (
    -precision_values
    /
    normalization
)


# ============================================================
# 9. 将偏相关矩阵对角线设为1
# ============================================================

np.fill_diagonal(
    partial_corr_values,
    1.0
)


partial_corr_matrix = pd.DataFrame(
    partial_corr_values,
    index=cov_matrix.index,
    columns=cov_matrix.columns
)


print(
    "\n================================"
)

print(
    "偏相关矩阵"
)

print(
    "================================"
)


print(
    partial_corr_matrix.round(4)
)


# ============================================================
# 10. 检查偏相关矩阵
# ============================================================

partial_is_symmetric = np.allclose(
    partial_corr_values,
    partial_corr_values.T,
    atol=1e-12
)


partial_min = np.min(
    partial_corr_values
)

partial_max = np.max(
    partial_corr_values
)


print(
    "\n偏相关矩阵是否对称：",
    partial_is_symmetric
)

print(
    "偏相关系数最小值：",
    partial_min
)

print(
    "偏相关系数最大值：",
    partial_max
)


# ============================================================
# 11. 保存偏相关矩阵
# ============================================================

partial_corr_matrix.to_csv(
    PARTIAL_CORR_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 找出绝对偏相关最大的股票对
# ============================================================

upper_mask = np.triu(
    np.ones(
        partial_corr_matrix.shape,
        dtype=bool
    ),
    k=1
)


partial_pairs = (
    partial_corr_matrix
    .where(
        upper_mask
    )
    .stack()
    .reset_index()
)


partial_pairs.columns = [
    "stock_1",
    "stock_2",
    "partial_correlation"
]


partial_pairs[
    "abs_partial_correlation"
] = (
    partial_pairs[
        "partial_correlation"
    ]
    .abs()
)


partial_pairs = (
    partial_pairs
    .sort_values(
        "abs_partial_correlation",
        ascending=False
    )
)


print(
    "\n绝对偏相关最高的10组股票："
)

print(
    partial_pairs
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 13. 输出完成信息
# ============================================================

print(
    "\n================================"
)

print(
    "阶段三完成"
)

print(
    "================================"
)


print(
    "\n精度矩阵已保存至："
)

print(
    PRECISION_FILE
)


print(
    "\n偏相关矩阵已保存至："
)

print(
    PARTIAL_CORR_FILE
)
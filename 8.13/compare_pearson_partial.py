from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 中文字体
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_DIR = Path("stock_network")

PROCESSED_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

FIGURE_DIR = (
    PROJECT_DIR /
    "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PEARSON_FILE = (
    PROCESSED_DIR /
    "stock_correlation.csv"
)

PARTIAL_FILE = (
    PROCESSED_DIR /
    "partial_correlation.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR /
    "stock_info.csv"
)


COMPARISON_FILE = (
    PROCESSED_DIR /
    "pearson_partial_comparison.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR /
    "pearson_partial_summary.csv"
)


# ============================================================
# 2. 描述性分类参数
# ============================================================

# 与昨天的 Pearson 阈值保持一致
PEARSON_STRONG = 0.50

# 偏相关达到该水平时，
# 暂时称为“仍有较明显条件关联”
PARTIAL_STRONG = 0.20

# 小于该值时暂时视为“接近0”
NEAR_ZERO = 0.05


# ============================================================
# 3. 读取方阵函数
# ============================================================

def read_square_matrix(file_path):

    matrix = pd.read_csv(
        file_path,
        index_col=0
    )

    # 防止 000001 被 pandas 读成 1
    matrix.index = (
        matrix.index
        .astype(str)
        .str.zfill(6)
    )

    matrix.columns = (
        matrix.columns
        .astype(str)
        .str.zfill(6)
    )

    matrix = matrix.astype(float)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{file_path.name} 不是方阵。"
        )

    return matrix


# ============================================================
# 4. 读取 Pearson 和偏相关矩阵
# ============================================================

pearson = read_square_matrix(
    PEARSON_FILE
)

partial = read_square_matrix(
    PARTIAL_FILE
)


print(
    "Pearson相关矩阵维度：",
    pearson.shape
)

print(
    "偏相关矩阵维度：",
    partial.shape
)


# ============================================================
# 5. 检查两个矩阵使用相同股票
# ============================================================

pearson_codes = set(
    pearson.index
)

partial_codes = set(
    partial.index
)


if pearson_codes != partial_codes:

    raise ValueError(
        "Pearson矩阵与偏相关矩阵中的股票代码不完全一致。"
    )


# 按 Pearson 顺序统一
codes = pearson.index.tolist()

partial = partial.loc[
    codes,
    codes
]


# ============================================================
# 6. 基本矩阵检查
# ============================================================

print(
    "\nPearson矩阵是否对称：",
    np.allclose(
        pearson.values,
        pearson.values.T,
        atol=1e-10
    )
)

print(
    "偏相关矩阵是否对称：",
    np.allclose(
        partial.values,
        partial.values.T,
        atol=1e-10
    )
)


print(
    "\nPearson对角线是否为1：",
    np.allclose(
        np.diag(
            pearson.values
        ),
        1.0
    )
)

print(
    "偏相关对角线是否为1：",
    np.allclose(
        np.diag(
            partial.values
        ),
        1.0
    )
)


# ============================================================
# 7. 读取股票名称和行业（可选）
# ============================================================

if STOCK_INFO_FILE.exists():

    stock_info = pd.read_csv(
        STOCK_INFO_FILE,
        dtype={"code": str}
    )

    stock_info["code"] = (
        stock_info["code"]
        .str.zfill(6)
    )

    stock_info = (
        stock_info
        .set_index("code")
    )

else:

    stock_info = pd.DataFrame(
        index=codes
    )


def get_name(code):

    if (
        code in stock_info.index
        and
        "name" in stock_info.columns
    ):
        return stock_info.loc[
            code,
            "name"
        ]

    return code


def get_industry(code):

    if (
        code in stock_info.index
        and
        "industry" in stock_info.columns
    ):
        return stock_info.loc[
            code,
            "industry"
        ]

    return "未知行业"


# ============================================================
# 8. 提取105个不同股票对
# ============================================================

comparison_rows = []


for i in range(
    len(codes)
):

    for j in range(
        i + 1,
        len(codes)
    ):

        stock_i = codes[i]
        stock_j = codes[j]

        rho_pearson = (
            pearson.loc[
                stock_i,
                stock_j
            ]
        )

        rho_partial = (
            partial.loc[
                stock_i,
                stock_j
            ]
        )


        # ----------------------------------------
        # 有符号变化
        #
        # partial - Pearson
        # ----------------------------------------

        signed_change = (
            rho_partial
            -
            rho_pearson
        )


        # ----------------------------------------
        # 相关强度变化
        #
        # |Pearson| - |Partial|
        #
        # > 0 : 控制以后减弱
        # < 0 : 控制以后增强
        # ----------------------------------------

        attenuation = (
            abs(
                rho_pearson
            )
            -
            abs(
                rho_partial
            )
        )


        # ----------------------------------------
        # 强度保留比例
        #
        # |Partial| / |Pearson|
        # ----------------------------------------

        if abs(rho_pearson) > 1e-12:

            retention_ratio = (
                abs(rho_partial)
                /
                abs(rho_pearson)
            )

        else:

            retention_ratio = np.nan


        # ----------------------------------------
        # 符号是否反转
        # ----------------------------------------

        sign_flip = (
            rho_pearson
            *
            rho_partial
            <
            0
        )


        # 排除接近0造成的“伪反转”
        meaningful_sign_flip = (
            sign_flip
            and
            abs(rho_pearson)
            >= NEAR_ZERO
            and
            abs(rho_partial)
            >= NEAR_ZERO
        )


        # ----------------------------------------
        # 对关系进行描述性分类
        # ----------------------------------------

        if (
            abs(rho_pearson)
            >= PEARSON_STRONG
        ):

            if (
                abs(rho_partial)
                <= NEAR_ZERO
            ):

                relation_type = (
                    "Pearson强相关基本消失"
                )

            elif (
                meaningful_sign_flip
            ):

                relation_type = (
                    "条件关系符号反转"
                )

            elif (
                abs(rho_partial)
                >= PARTIAL_STRONG
            ):

                relation_type = (
                    "强相关仍部分保留"
                )

            else:

                relation_type = (
                    "强相关明显减弱"
                )

        else:

            if (
                meaningful_sign_flip
            ):

                relation_type = (
                    "条件关系符号反转"
                )

            elif (
                abs(rho_partial)
                >
                abs(rho_pearson)
            ):

                relation_type = (
                    "控制后关联增强"
                )

            else:

                relation_type = (
                    "其他"
                )


        comparison_rows.append(
            {
                "stock_1":
                    stock_i,

                "name_1":
                    get_name(
                        stock_i
                    ),

                "industry_1":
                    get_industry(
                        stock_i
                    ),

                "stock_2":
                    stock_j,

                "name_2":
                    get_name(
                        stock_j
                    ),

                "industry_2":
                    get_industry(
                        stock_j
                    ),

                "pearson":
                    rho_pearson,

                "partial":
                    rho_partial,

                "abs_pearson":
                    abs(
                        rho_pearson
                    ),

                "abs_partial":
                    abs(
                        rho_partial
                    ),

                "signed_change":
                    signed_change,

                "attenuation":
                    attenuation,

                "retention_ratio":
                    retention_ratio,

                "sign_flip":
                    sign_flip,

                "meaningful_sign_flip":
                    meaningful_sign_flip,

                "relation_type":
                    relation_type
            }
        )


comparison = pd.DataFrame(
    comparison_rows
)


# ============================================================
# 9. 保存完整股票对比较表
# ============================================================

comparison = (
    comparison
    .sort_values(
        "attenuation",
        ascending=False
    )
)


comparison.to_csv(
    COMPARISON_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 汇总统计
# ============================================================

n_pairs = len(
    comparison
)


n_pearson_strong = (
    comparison[
        "abs_pearson"
    ]
    >=
    PEARSON_STRONG
).sum()


n_partial_strong = (
    comparison[
        "abs_partial"
    ]
    >=
    PARTIAL_STRONG
).sum()


n_all_weakened = (
    comparison[
        "attenuation"
    ]
    >
    0
).sum()


n_sign_flip = (
    comparison[
        "sign_flip"
    ]
).sum()


n_meaningful_sign_flip = (
    comparison[
        "meaningful_sign_flip"
    ]
).sum()


# Pearson 与 Partial 非对角元素之间的相关性
matrix_relationship = (
    comparison[
        "pearson"
    ]
    .corr(
        comparison[
            "partial"
        ]
    )
)


summary = pd.DataFrame(
    [
        {
            "n_stock_pairs":
                n_pairs,

            "mean_pearson":
                comparison[
                    "pearson"
                ].mean(),

            "mean_partial":
                comparison[
                    "partial"
                ].mean(),

            "mean_abs_pearson":
                comparison[
                    "abs_pearson"
                ].mean(),

            "mean_abs_partial":
                comparison[
                    "abs_partial"
                ].mean(),

            "median_abs_pearson":
                comparison[
                    "abs_pearson"
                ].median(),

            "median_abs_partial":
                comparison[
                    "abs_partial"
                ].median(),

            "mean_attenuation":
                comparison[
                    "attenuation"
                ].mean(),

            "n_pairs_weakened":
                int(
                    n_all_weakened
                ),

            "n_pearson_strong":
                int(
                    n_pearson_strong
                ),

            "n_partial_strong":
                int(
                    n_partial_strong
                ),

            "n_sign_flip":
                int(
                    n_sign_flip
                ),

            "n_meaningful_sign_flip":
                int(
                    n_meaningful_sign_flip
                ),

            "pearson_partial_pair_correlation":
                matrix_relationship
        }
    ]
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n================================"
)

print(
    "Pearson vs Partial 汇总"
)

print(
    "================================"
)

print(
    summary.T
)


# ============================================================
# 11. 输出衰减最大的股票对
# ============================================================

print(
    "\n================================"
)

print(
    "控制其他股票后，"
    "相关强度衰减最大的10组股票"
)

print(
    "================================"
)


print(
    comparison[
        [
            "stock_1",
            "name_1",
            "stock_2",
            "name_2",
            "pearson",
            "partial",
            "attenuation"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 12. Pearson强相关股票对
# ============================================================

pearson_strong_pairs = (
    comparison[
        comparison[
            "abs_pearson"
        ]
        >=
        PEARSON_STRONG
    ]
    .sort_values(
        "abs_partial",
        ascending=False
    )
)


print(
    "\n================================"
)

print(
    "Pearson强相关股票对及其偏相关"
)

print(
    "================================"
)


print(
    pearson_strong_pairs[
        [
            "stock_1",
            "name_1",
            "stock_2",
            "name_2",
            "pearson",
            "partial",
            "retention_ratio",
            "relation_type"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 13. 有意义的符号反转
# ============================================================

sign_flip_pairs = (
    comparison[
        comparison[
            "meaningful_sign_flip"
        ]
    ]
)


print(
    "\n================================"
)

print(
    "较明显的符号反转股票对"
)

print(
    "================================"
)


if len(sign_flip_pairs) > 0:

    print(
        sign_flip_pairs[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "pearson",
                "partial"
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有达到当前标准的明显符号反转。"
    )


# ============================================================
# 14. 热力图标签
# ============================================================

labels = [
    f"{code}\n{get_name(code)}"
    for code in codes
]


# ============================================================
# 15. Pearson相关矩阵 Heatmap
# ============================================================

plt.figure(
    figsize=(12, 10)
)

image = plt.imshow(
    pearson.values,
    vmin=-1,
    vmax=1
)

plt.colorbar(
    image,
    label="Pearson相关系数"
)

plt.xticks(
    range(
        len(codes)
    ),
    labels,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(
        len(codes)
    ),
    labels,
    fontsize=7
)

plt.title(
    "Pearson相关矩阵"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "pearson_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. 偏相关矩阵 Heatmap
# ============================================================

plt.figure(
    figsize=(12, 10)
)

image = plt.imshow(
    partial.values,
    vmin=-1,
    vmax=1
)

plt.colorbar(
    image,
    label="偏相关系数"
)

plt.xticks(
    range(
        len(codes)
    ),
    labels,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(
        len(codes)
    ),
    labels,
    fontsize=7
)

plt.title(
    "偏相关矩阵"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "partial_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. 相关强度衰减矩阵
#
# A_ij =
# |Pearson_ij| - |Partial_ij|
#
# 正值越大：
# 控制其他股票后衰减越明显
# ============================================================

attenuation_matrix = (
    np.abs(
        pearson.values
    )
    -
    np.abs(
        partial.values
    )
)


# 对角线不分析
np.fill_diagonal(
    attenuation_matrix,
    0.0
)


plt.figure(
    figsize=(12, 10)
)

image = plt.imshow(
    attenuation_matrix
)

plt.colorbar(
    image,
    label="|Pearson| - |Partial|"
)

plt.xticks(
    range(
        len(codes)
    ),
    labels,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(
        len(codes)
    ),
    labels,
    fontsize=7
)

plt.title(
    "控制其他股票后的相关强度衰减"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "correlation_attenuation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. Pearson vs Partial 散点图
# ============================================================

plt.figure(
    figsize=(8, 8)
)


plt.scatter(
    comparison[
        "pearson"
    ],
    comparison[
        "partial"
    ],
    alpha=0.7
)


# y = x 参考线
lower = min(
    comparison[
        "pearson"
    ].min(),
    comparison[
        "partial"
    ].min()
)

upper = max(
    comparison[
        "pearson"
    ].max(),
    comparison[
        "partial"
    ].max()
)


plt.plot(
    [
        lower,
        upper
    ],
    [
        lower,
        upper
    ]
)


plt.axhline(
    0,
    linewidth=0.8
)

plt.axvline(
    0,
    linewidth=0.8
)


plt.xlabel(
    "Pearson相关系数"
)

plt.ylabel(
    "偏相关系数"
)

plt.title(
    "Pearson相关与偏相关的股票对比较"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR /
    "pearson_vs_partial_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 19. 完成
# ============================================================

print(
    "\n================================"
)

print(
    "阶段四完成"
)

print(
    "================================"
)


print(
    "\n主要输出文件："
)

print(
    COMPARISON_FILE
)

print(
    SUMMARY_FILE
)

print(
    FIGURE_DIR /
    "pearson_correlation_heatmap.png"
)

print(
    FIGURE_DIR /
    "partial_correlation_heatmap.png"
)

print(
    FIGURE_DIR /
    "correlation_attenuation_heatmap.png"
)

print(
    FIGURE_DIR /
    "pearson_vs_partial_scatter.png"
)
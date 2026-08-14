from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso


# ============================================================
# 0. 基本绘图设置
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC"
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_DIR = Path("stock_network")

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

FIGURE_DIR = (
    PROJECT_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RETURN_FILE = (
    PROCESSED_DIR
    / "stock_returns.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)

PATH_OUTPUT_FILE = (
    PROCESSED_DIR
    / "glasso_regularization_path.csv"
)


# ============================================================
# 2. 参数
# ============================================================

# alpha路径上的点数
N_ALPHA = 30

# 判断精度矩阵元素是否为0时的数值容忍度
ZERO_TOL = 1e-8

# 最大迭代次数
MAX_ITER = 5000


# ============================================================
# 3. 读取收益率数据
# ============================================================

returns = pd.read_csv(
    RETURN_FILE,
    index_col=0,
    parse_dates=True
)


returns.columns = (
    returns.columns
    .astype(str)
    .str.zfill(6)
)


returns = (
    returns
    .sort_index()
    .dropna(
        axis=0,
        how="any"
    )
)


print(
    "收益率矩阵维度：",
    returns.shape
)


codes = (
    returns.columns
    .tolist()
)


n_samples = returns.shape[0]
n_stocks = returns.shape[1]


# ============================================================
# 4. 标准化股票收益率
#
# Z_it = (r_it - mean_i) / std_i
# ============================================================

Z = (
    returns
    -
    returns.mean()
) / returns.std(
    ddof=0
)


print(
    "\n标准化后均值（应接近0）："
)

print(
    Z.mean()
)


print(
    "\n标准化后标准差（应接近1）："
)

print(
    Z.std(
        ddof=0
    )
)


# ============================================================
# 5. 计算标准化收益率的经验协方差矩阵
#
# sklearn内部使用经验协方差，
# 这里使用 1/n，与MLE形式保持一致
# ============================================================

Z_values = Z.values


S = (
    Z_values.T
    @ Z_values
    /
    n_samples
)


# ============================================================
# 6. 数据驱动确定 alpha 范围
#
# alpha_max =
# max_{i != j} |S_ij|
#
# alpha足够大时，
# 精度矩阵将趋向于对角结构。
# ============================================================

S_offdiag = (
    S.copy()
)

np.fill_diagonal(
    S_offdiag,
    0.0
)


alpha_max = (
    np.max(
        np.abs(
            S_offdiag
        )
    )
)


# 从 alpha_max 的 1% 开始
alpha_min = (
    alpha_max
    *
    0.01
)


# 稍微超过 alpha_max，
# 便于观察网络接近完全无边的情形
alpha_upper = (
    alpha_max
    *
    1.05
)


alphas = np.geomspace(
    alpha_min,
    alpha_upper,
    N_ALPHA
)


print(
    "\nalpha_min =",
    alpha_min
)

print(
    "alpha_max approximately =",
    alpha_max
)

print(
    "alpha_upper =",
    alpha_upper
)


# ============================================================
# 7. 工具函数：
#    精度矩阵 -> 偏相关矩阵
# ============================================================

def precision_to_partial(
    precision
):

    diag = np.sqrt(
        np.diag(
            precision
        )
    )


    partial = (
        -precision
        /
        np.outer(
            diag,
            diag
        )
    )


    np.fill_diagonal(
        partial,
        1.0
    )


    return partial


# ============================================================
# 8. 工具函数：
#    精度矩阵 -> NetworkX网络
# ============================================================

def build_graph(
    precision,
    partial,
    alpha
):

    G = nx.Graph()


    # 添加所有节点
    for code in codes:

        G.add_node(
            code
        )


    # 根据 precision 的非零结构决定是否有边
    for i in range(
        n_stocks
    ):

        for j in range(
            i + 1,
            n_stocks
        ):

            omega_ij = (
                precision[
                    i,
                    j
                ]
            )


            if (
                abs(
                    omega_ij
                )
                >
                ZERO_TOL
            ):

                rho_partial = (
                    partial[
                        i,
                        j
                    ]
                )


                G.add_edge(
                    codes[i],
                    codes[j],

                    precision=
                        omega_ij,

                    partial_correlation=
                        rho_partial,

                    strength=
                        abs(
                            rho_partial
                        ),

                    alpha=
                        alpha
                )


    return G


# ============================================================
# 9. 对整个 alpha 路径重复估计
# ============================================================

path_rows = []

models = {}

graphs = {}

partial_matrices = {}


total_possible_edges = (
    n_stocks
    *
    (
        n_stocks - 1
    )
    /
    2
)


for alpha in alphas:

    model = GraphicalLasso(
    alpha=float(alpha),

    mode="cd",

    # 外层 Graphical Lasso 收敛要求
    tol=1e-4,

    # 内层 coordinate-descent
    # 搜索方向算得更精确
    enet_tol=1e-6,

    max_iter=5000,

    # Z 已经标准化、中心化
    assume_centered=True
)

    model.fit(
        Z_values
    )


    precision = (
        model.precision_
    )


    partial = (
        precision_to_partial(
            precision
        )
    )


    G = build_graph(
        precision,
        partial,
        alpha
    )


    # ----------------------------------------
    # 基本网络统计
    # ----------------------------------------

    n_edges = (
        G.number_of_edges()
    )


    density = (
        n_edges
        /
        total_possible_edges
    )


    n_components = (
        nx.number_connected_components(
            G
        )
    )


    n_isolated = len(
        list(
            nx.isolates(
                G
            )
        )
    )


    degrees = [
        degree
        for _, degree
        in G.degree()
    ]


    mean_degree = (
        np.mean(
            degrees
        )
    )


    max_degree = (
        np.max(
            degrees
        )
    )


    # ----------------------------------------
    # 非零边对应的平均绝对偏相关
    # ----------------------------------------

    edge_strengths = [
        data[
            "strength"
        ]

        for _, _, data
        in G.edges(
            data=True
        )
    ]


    if len(
        edge_strengths
    ) > 0:

        mean_edge_strength = (
            np.mean(
                edge_strengths
            )
        )

        max_edge_strength = (
            np.max(
                edge_strengths
            )
        )

    else:

        mean_edge_strength = 0.0
        max_edge_strength = 0.0


    path_rows.append(
        {
            "alpha":
                alpha,

            "n_edges":
                n_edges,

            "density":
                density,

            "n_components":
                n_components,

            "n_isolated":
                n_isolated,

            "mean_degree":
                mean_degree,

            "max_degree":
                max_degree,

            "mean_abs_partial":
                mean_edge_strength,

            "max_abs_partial":
                max_edge_strength,

            "n_iter":
                model.n_iter_
        }
    )


    # 保存后面绘图需要的对象
    models[
        float(alpha)
    ] = model

    graphs[
        float(alpha)
    ] = G

    partial_matrices[
        float(alpha)
    ] = partial


# ============================================================
# 10. 整理 regularization path
# ============================================================

path_df = pd.DataFrame(
    path_rows
)


path_df.to_csv(
    PATH_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n================================"
)

print(
    "Graphical Lasso Regularization Path"
)

print(
    "================================"
)


print(
    path_df[
        [
            "alpha",
            "n_edges",
            "density",
            "n_components",
            "n_isolated"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 11. 图1：
#     alpha -> edge count
# ============================================================

plt.figure(
    figsize=(
        8,
        6
    )
)


plt.plot(
    path_df[
        "alpha"
    ],

    path_df[
        "n_edges"
    ],

    marker="o"
)


plt.xscale(
    "log"
)


plt.xlabel(
    "Graphical Lasso alpha"
)

plt.ylabel(
    "Number of edges"
)

plt.title(
    "Regularization parameter vs network edge count"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "glasso_alpha_edges.png",

    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 12. 图2：
#     alpha -> density
# ============================================================

plt.figure(
    figsize=(
        8,
        6
    )
)


plt.plot(
    path_df[
        "alpha"
    ],

    path_df[
        "density"
    ],

    marker="o"
)


plt.xscale(
    "log"
)


plt.xlabel(
    "Graphical Lasso alpha"
)

plt.ylabel(
    "Network density"
)

plt.title(
    "Regularization parameter vs network density"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "glasso_alpha_density.png",

    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 13. 图3：
#     alpha -> components / isolated nodes
# ============================================================

plt.figure(
    figsize=(
        8,
        6
    )
)


plt.plot(
    path_df[
        "alpha"
    ],

    path_df[
        "n_components"
    ],

    marker="o",

    label=
        "Connected components"
)


plt.plot(
    path_df[
        "alpha"
    ],

    path_df[
        "n_isolated"
    ],

    marker="s",

    label=
        "Isolated nodes"
)


plt.xscale(
    "log"
)


plt.xlabel(
    "Graphical Lasso alpha"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Network fragmentation along the regularization path"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "glasso_alpha_components.png",

    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 14. 选择3个代表性 alpha
#
# 低、中、高正则化
# ============================================================

snapshot_indices = [
    2,
    len(alphas) // 2,
    len(alphas) - 3
]


snapshot_alphas = [
    float(
        alphas[i]
    )
    for i
    in snapshot_indices
]


print(
    "\n用于网络快照的 alpha："
)

for alpha in snapshot_alphas:

    G = graphs[
        alpha
    ]

    print(
        f"alpha={alpha:.6f}, "
        f"edges={G.number_of_edges()}"
    )


# ============================================================
# 15. 股票行业信息（可选）
# ============================================================

industry_map = {}


if STOCK_INFO_FILE.exists():

    stock_info = pd.read_csv(
        STOCK_INFO_FILE,
        dtype={
            "code": str
        }
    )


    stock_info[
        "code"
    ] = (
        stock_info[
            "code"
        ]
        .str
        .zfill(6)
    )


    if (
        "name"
        in stock_info.columns
    ):

        name_map = dict(
            zip(
                stock_info[
                    "code"
                ],

                stock_info[
                    "name"
                ]
            )
        )

    if (
        "industry"
        in stock_info.columns
    ):

        industry_map = dict(
            zip(
                stock_info[
                    "code"
                ],

                stock_info[
                    "industry"
                ]
            )
        )

print(
    "\n股票名称映射："
)

for code in codes:

    print(
        code,
        "->",
        name_map.get(
            code,
            code
        )
    )


# ============================================================
# 16. 行业颜色
# ============================================================

if industry_map:

    industries = sorted(
        {
            industry_map.get(
                code,
                "未知行业"
            )

            for code in codes
        }
    )


    cmap = plt.get_cmap(
        "tab20"
    )


    industry_color = {
        industry:
            cmap(
                i % 20
            )

        for i, industry
        in enumerate(
            industries
        )
    }


    node_colors = [
        industry_color[
            industry_map.get(
                code,
                "未知行业"
            )
        ]

        for code in codes
    ]

else:

    node_colors = None


# ============================================================
# 17. 图4：
#     低 / 中 / 高 alpha 下的网络快照
#
# 三张图使用完全相同的节点位置
# ============================================================


# ------------------------------------------------------------
# 17.1 节点标签
#
# 第一行：股票代码
# 第二行：股票名称
# ------------------------------------------------------------

node_labels = {

    code:
        f"{code}\n{name_map.get(code, code)}"

    for code in codes
}


# ------------------------------------------------------------
# 17.2 固定节点位置
# ------------------------------------------------------------

fixed_pos = (
    nx.circular_layout(
        codes
    )
)


# ------------------------------------------------------------
# 17.3 创建三个网络子图
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        24,
        8
    )
)


for ax, alpha in zip(
    axes,
    snapshot_alphas
):

    G = graphs[
        alpha
    ]


    # --------------------------------------------------------
    # 节点
    # --------------------------------------------------------

    nx.draw_networkx_nodes(
        G,
        fixed_pos,

        node_size=
            1500,

        node_color=
            node_colors,

        edgecolors=
            "black",

        linewidths=
            0.8,

        ax=
            ax
    )


    # --------------------------------------------------------
    # 边宽：
    # 根据绝对偏相关大小设置
    # --------------------------------------------------------

    edge_widths = [

        0.8
        +
        8
        *
        data[
            "strength"
        ]

        for _, _, data
        in G.edges(
            data=True
        )
    ]


    nx.draw_networkx_edges(
        G,
        fixed_pos,

        width=
            edge_widths,

        alpha=
            0.65,

        ax=
            ax
    )


    # --------------------------------------------------------
    # 节点标签
    #
    # 第一行股票代码
    # 第二行股票名称
    # --------------------------------------------------------

    nx.draw_networkx_labels(
        G,
        fixed_pos,

        labels=
            node_labels,

        font_size=
            7,

        horizontalalignment=
            "center",

        verticalalignment=
            "center",

        ax=
            ax
    )


    # --------------------------------------------------------
    # 当前 alpha 的网络统计
    # --------------------------------------------------------

    row = path_df[
        np.isclose(
            path_df[
                "alpha"
            ],
            alpha
        )
    ].iloc[0]


    # --------------------------------------------------------
    # 子图标题
    # --------------------------------------------------------

    ax.set_title(
        f"alpha = {alpha:.4f}\n"
        f"Edges = {int(row['n_edges'])}, "
        f"Density = {row['density']:.3f}\n"
        f"Components = {int(row['n_components'])}",
        fontsize=11
    )


    ax.axis(
        "off"
    )


# ============================================================
# 总标题
# ============================================================

fig.suptitle(
    "Graphical Lasso网络随正则化参数变化",
    fontsize=16
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "glasso_network_path.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 18. 完成
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
    "\n输出文件："
)

print(
    PATH_OUTPUT_FILE
)

print(
    FIGURE_DIR
    / "glasso_alpha_edges.png"
)

print(
    FIGURE_DIR
    / "glasso_alpha_density.png"
)

print(
    FIGURE_DIR
    / "glasso_alpha_components.png"
)

print(
    FIGURE_DIR
    / "glasso_network_path.png"
)
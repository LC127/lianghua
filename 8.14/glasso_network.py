from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.lines import Line2D

from sklearn.covariance import GraphicalLassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. 中文字体
# ============================================================

def set_chinese_font():

    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP"
    ]

    installed_fonts = {
        f.name
        for f in font_manager.fontManager.ttflist
    }

    for font_name in candidate_fonts:

        if font_name in installed_fonts:

            plt.rcParams["font.sans-serif"] = [
                font_name
            ]

            plt.rcParams[
                "axes.unicode_minus"
            ] = False

            print(
                f"使用中文字体：{font_name}"
            )

            return

    print(
        "警告：未找到常见中文字体。"
    )


set_chinese_font()


# ============================================================
# 1. 路径
# ============================================================

PROJECT_DIR = Path(
    "stock_network"
)

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

PATH_FILE = (
    PROCESSED_DIR
    / "glasso_regularization_path.csv"
)


CV_RESULT_FILE = (
    PROCESSED_DIR
    / "glasso_cv_results.csv"
)

PRECISION_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_precision.csv"
)

PARTIAL_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_partial_correlation.csv"
)

EDGE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_edges.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_network_summary.csv"
)


# ============================================================
# 2. 参数
# ============================================================

N_SPLITS = 5

ZERO_TOL = 1e-8

MAX_ITER = 5000

# 结合前面收敛问题，
# 内层 coordinate descent 使用更严格容差
ENET_TOL = 1e-6

OUTER_TOL = 1e-4


# ============================================================
# 3. 读取股票收益率
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


codes = (
    returns.columns
    .tolist()
)


print(
    "收益率数据维度：",
    returns.shape
)


# ============================================================
# 4. 标准化收益率
#
# 每只股票：
# mean -> 0
# variance -> 1
# ============================================================

scaler = StandardScaler(
    with_mean=True,
    with_std=True
)


Z_values = scaler.fit_transform(
    returns.values
)


Z = pd.DataFrame(
    Z_values,
    index=returns.index,
    columns=codes
)


print(
    "\n标准化后最大绝对均值：",
    np.abs(
        Z.mean()
    ).max()
)


print(
    "标准化后标准差："
)

print(
    Z.std(
        ddof=0
    )
)


# ============================================================
# 5. 读取阶段三 regularization path
#
# 直接把阶段三已经研究过的 alpha
# 作为阶段四的候选集合
# ============================================================

path_df = pd.read_csv(
    PATH_FILE
)


candidate_alphas = (
    path_df[
        "alpha"
    ]
    .dropna()
    .astype(float)
    .unique()
)


candidate_alphas = np.sort(
    candidate_alphas
)


candidate_alphas = candidate_alphas[
    candidate_alphas > 0
]


print(
    "\n候选 alpha 数量：",
    len(
        candidate_alphas
    )
)


print(
    "alpha范围：",
    candidate_alphas.min(),
    "到",
    candidate_alphas.max()
)


# ============================================================
# 6. 时间序列交叉验证
# ============================================================

time_cv = TimeSeriesSplit(
    n_splits=N_SPLITS
)


# ============================================================
# 7. GraphicalLassoCV
#
# alpha由交叉验证选择
# ============================================================

cv_model = GraphicalLassoCV(

    # 使用阶段三的alpha路径
    alphas=
        candidate_alphas,

    cv=
        time_cv,

    mode=
        "cd",

    tol=
        OUTER_TOL,

    enet_tol=
        ENET_TOL,

    max_iter=
        MAX_ITER,

    n_jobs=
        -1,

    # 数据已经中心化
    assume_centered=
        True,

    verbose=
        False
)


print(
    "\n开始 GraphicalLassoCV ..."
)


cv_model.fit(
    Z_values
)


selected_alpha = float(
    cv_model.alpha_
)


print(
    "\n================================"
)

print(
    "Graphical Lasso参数选择"
)

print(
    "================================"
)


print(
    "CV选择的 alpha =",
    selected_alpha
)


print(
    "最终模型迭代次数 =",
    cv_model.n_iter_
)


# ============================================================
# 8. 保存CV结果
# ============================================================

cv_results = cv_model.cv_results_


cv_table = pd.DataFrame(
    {
        "alpha":
            cv_results[
                "alphas"
            ],

        "mean_test_score":
            cv_results[
                "mean_test_score"
            ],

        "std_test_score":
            cv_results[
                "std_test_score"
            ]
    }
)


# alpha从小到大排序
cv_table = (
    cv_table
    .sort_values(
        "alpha"
    )
    .reset_index(
        drop=True
    )
)


cv_table[
    "selected"
] = np.isclose(
    cv_table[
        "alpha"
    ],
    selected_alpha
)


cv_table.to_csv(
    CV_RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nCV结果："
)

print(
    cv_table
    .to_string(
        index=False
    )
)


# ============================================================
# 9. 提取最终精度矩阵
#
# GraphicalLassoCV找到alpha后，
# 会在全样本上重新拟合
# ============================================================

precision = (
    cv_model.precision_
)


precision_df = pd.DataFrame(
    precision,
    index=codes,
    columns=codes
)


precision_df.to_csv(
    PRECISION_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 10. Precision -> Partial Correlation
# ============================================================

precision_diag = np.sqrt(
    np.diag(
        precision
    )
)


partial = (
    -precision
    /
    np.outer(
        precision_diag,
        precision_diag
    )
)


np.fill_diagonal(
    partial,
    1.0
)


partial_df = pd.DataFrame(
    partial,
    index=codes,
    columns=codes
)


partial_df.to_csv(
    PARTIAL_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 读取股票信息
# ============================================================

name_map = {}
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


def get_name(code):

    return name_map.get(
        code,
        code
    )


def get_industry(code):

    return industry_map.get(
        code,
        "未知行业"
    )


# ============================================================
# 12. 根据 precision 的非零结构构建网络
#
# 非零 -> 有边
# 0    -> 无边
# ============================================================

G = nx.Graph()


for code in codes:

    G.add_node(
        code,
        name=get_name(
            code
        ),
        industry=get_industry(
            code
        )
    )


edge_rows = []


for i in range(
    len(codes)
):

    for j in range(
        i + 1,
        len(codes)
    ):

        omega_ij = (
            precision[
                i,
                j
            ]
        )


        # ZERO_TOL仅用于浮点数容忍，
        # 不是新的统计阈值
        if (
            abs(
                omega_ij
            )
            >
            ZERO_TOL
        ):

            rho_ij = (
                partial[
                    i,
                    j
                ]
            )


            code_i = codes[i]
            code_j = codes[j]


            G.add_edge(

                code_i,
                code_j,

                precision=
                    float(
                        omega_ij
                    ),

                partial_correlation=
                    float(
                        rho_ij
                    ),

                strength=
                    float(
                        abs(
                            rho_ij
                        )
                    )
            )


            edge_rows.append(
                {
                    "stock_1":
                        code_i,

                    "name_1":
                        get_name(
                            code_i
                        ),

                    "industry_1":
                        get_industry(
                            code_i
                        ),

                    "stock_2":
                        code_j,

                    "name_2":
                        get_name(
                            code_j
                        ),

                    "industry_2":
                        get_industry(
                            code_j
                        ),

                    "precision":
                        omega_ij,

                    "partial_correlation":
                        rho_ij,

                    "abs_partial_correlation":
                        abs(
                            rho_ij
                        ),

                    "sign":
                        (
                            "positive"
                            if rho_ij > 0
                            else "negative"
                        )
                }
            )


# ============================================================
# 13. 保存边表
# ============================================================

edge_df = pd.DataFrame(
    edge_rows
)


if not edge_df.empty:

    edge_df = (
        edge_df
        .sort_values(
            "abs_partial_correlation",
            ascending=False
        )
    )


edge_df.to_csv(
    EDGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 网络基本统计
# ============================================================

n_nodes = (
    G.number_of_nodes()
)

n_edges = (
    G.number_of_edges()
)

density = nx.density(
    G
)

n_components = (
    nx.number_connected_components(
        G
    )
)

isolated_nodes = list(
    nx.isolates(
        G
    )
)


n_isolated = len(
    isolated_nodes
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


summary_df = pd.DataFrame(
    [
        {
            "selected_alpha":
                selected_alpha,

            "n_nodes":
                n_nodes,

            "n_edges":
                n_edges,

            "density":
                density,

            "mean_degree":
                mean_degree,

            "max_degree":
                max_degree,

            "n_components":
                n_components,

            "n_isolated":
                n_isolated
        }
    ]
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n================================"
)

print(
    "最终 Graphical Lasso 网络"
)

print(
    "================================"
)


print(
    "alpha：",
    selected_alpha
)

print(
    "节点数：",
    n_nodes
)

print(
    "边数：",
    n_edges
)

print(
    "网络密度：",
    round(
        density,
        4
    )
)

print(
    "平均Degree：",
    round(
        mean_degree,
        4
    )
)

print(
    "最大Degree：",
    max_degree
)

print(
    "连通分量：",
    n_components
)

print(
    "孤立节点：",
    n_isolated
)


# ============================================================
# 15. 图1：
#     CV score vs alpha
# ============================================================

valid_cv = (
    cv_table
    .dropna(
        subset=[
            "mean_test_score"
        ]
    )
)


plt.figure(
    figsize=(
        8,
        6
    )
)


plt.plot(
    valid_cv[
        "alpha"
    ],

    valid_cv[
        "mean_test_score"
    ],

    marker="o"
)


plt.fill_between(

    valid_cv[
        "alpha"
    ],

    valid_cv[
        "mean_test_score"
    ]
    -
    valid_cv[
        "std_test_score"
    ],

    valid_cv[
        "mean_test_score"
    ]
    +
    valid_cv[
        "std_test_score"
    ],

    alpha=
        0.2
)


plt.axvline(
    selected_alpha,

    linestyle=
        "--",

    label=
        f"Selected alpha = {selected_alpha:.4f}"
)


plt.xscale(
    "log"
)


plt.xlabel(
    "Graphical Lasso alpha"
)

plt.ylabel(
    "Mean CV log-likelihood"
)

plt.title(
    "Graphical Lasso参数选择"
)


plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "glasso_cv_alpha_selection.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 16. 行业颜色
# ============================================================

industries = sorted(
    {
        get_industry(
            code
        )

        for code in codes
    }
)


cmap = plt.get_cmap(
    "tab20"
)


industry_color_map = {

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

    industry_color_map[
        get_industry(
            code
        )
    ]

    for code in G.nodes
]


# ============================================================
# 17. 节点标签
#
# 第一行：股票代码
# 第二行：股票名称
# ============================================================

node_labels = {

    code:
        f"{code}\n{get_name(code)}"

    for code in G.nodes
}


# ============================================================
# 18. Graphical Lasso网络布局
# ============================================================

pos = nx.spring_layout(
    G,

    seed=
        42,

    weight=
        "strength"
)


# ============================================================
# 19. 边宽
# ============================================================

edge_strengths = np.array(
    [
        data[
            "strength"
        ]

        for _, _, data
        in G.edges(
            data=True
        )
    ]
)


if len(
    edge_strengths
) > 0:

    if (
        edge_strengths.max()
        >
        edge_strengths.min()
    ):

        edge_widths = (

            1.0
            +
            7.0
            *
            (
                edge_strengths
                -
                edge_strengths.min()
            )
            /
            (
                edge_strengths.max()
                -
                edge_strengths.min()
            )
        )

    else:

        edge_widths = np.repeat(
            4.0,
            len(
                edge_strengths
            )
        )

else:

    edge_widths = []


# ============================================================
# 20. 绘制最终Graphical Lasso网络
# ============================================================

plt.figure(
    figsize=(
        15,
        12
    )
)


nx.draw_networkx_nodes(

    G,
    pos,

    node_color=
        node_colors,

    node_size=
        1900,

    edgecolors=
        "black",

    linewidths=
        0.8
)


edge_list = list(
    G.edges(
        data=True
    )
)


for k, (
    u,
    v,
    data
) in enumerate(
    edge_list
):

    rho = (
        data[
            "partial_correlation"
        ]
    )


    style = (
        "solid"
        if rho > 0
        else
        "dashed"
    )


    nx.draw_networkx_edges(

        G,
        pos,

        edgelist=[
            (
                u,
                v
            )
        ],

        width=
            edge_widths[k],

        style=
            style,

        alpha=
            0.75
    )


nx.draw_networkx_labels(

    G,
    pos,

    labels=
        node_labels,

    font_size=
        7
)


# 边标签显示GLasso偏相关
edge_labels = {

    (
        u,
        v
    ):
        f"{data['partial_correlation']:.2f}"

    for u, v, data
    in G.edges(
        data=True
    )
}


nx.draw_networkx_edge_labels(

    G,
    pos,

    edge_labels=
        edge_labels,

    font_size=
        7
)


# ============================================================
# 21. 图例
# ============================================================

legend_handles = [

    Line2D(
        [0],
        [0],

        marker="o",
        linestyle="",

        label=
            industry,

        markerfacecolor=
            color,

        markeredgecolor=
            "black",

        markersize=
            9
    )

    for industry, color
    in industry_color_map.items()
]


legend_handles.extend(
    [
        Line2D(
            [0],
            [0],

            linestyle=
                "solid",

            label=
                "正偏相关"
        ),

        Line2D(
            [0],
            [0],

            linestyle=
                "dashed",

            label=
                "负偏相关"
        )
    ]
)


plt.legend(
    handles=
        legend_handles,

    loc=
        "best",

    title=
        "行业 / 条件关联方向"
)


plt.title(
    "Graphical Lasso 股票条件关联网络\n"
    f"CV selected alpha = {selected_alpha:.4f}"
)


plt.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "graphical_lasso_network.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 22. 输出最强的GLasso边
# ============================================================

print(
    "\n绝对偏相关最大的 Graphical Lasso 网络边："
)


if not edge_df.empty:

    print(
        edge_df[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "partial_correlation",
                "abs_partial_correlation"
            ]
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 23. 完成
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
    "\n输出文件："
)

print(
    CV_RESULT_FILE
)

print(
    PRECISION_FILE
)

print(
    PARTIAL_FILE
)

print(
    EDGE_FILE
)

print(
    SUMMARY_FILE
)

print(
    FIGURE_DIR
    / "glasso_cv_alpha_selection.png"
)

print(
    FIGURE_DIR
    / "graphical_lasso_network.png"
)
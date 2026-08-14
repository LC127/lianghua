from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.lines import Line2D

from sklearn.covariance import GraphicalLasso
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

            plt.rcParams["axes.unicode_minus"] = False

            print(
                f"使用中文字体：{font_name}"
            )

            return


set_chinese_font()


# ============================================================
# 1. 文件路径
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

CV_RESULT_FILE = (
    PROCESSED_DIR
    / "glasso_cv_results.csv"
)

# 阶段四已经生成的CV网络边表
CV_EDGE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_edges.csv"
)


# ============================================================
# 2. 输出文件
# ============================================================

SELECTION_FILE = (
    PROCESSED_DIR
    / "glasso_1se_selection.csv"
)

PRECISION_1SE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_precision.csv"
)

PARTIAL_1SE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_partial_correlation.csv"
)

EDGE_1SE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_edges.csv"
)

SUMMARY_1SE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_network_summary.csv"
)

EDGE_COMPARE_FILE = (
    PROCESSED_DIR
    / "glasso_cv_vs_1se_edge_comparison.csv"
)

NETWORK_COMPARE_FILE = (
    PROCESSED_DIR
    / "glasso_cv_vs_1se_network_summary.csv"
)


# ============================================================
# 3. 参数
# ============================================================

N_SPLITS = 5

ZERO_TOL = 1e-8

MAX_ITER = 5000

OUTER_TOL = 1e-4

ENET_TOL = 1e-6


# ============================================================
# 4. 读取CV结果
# ============================================================

cv_df = pd.read_csv(
    CV_RESULT_FILE
)


cv_df = (
    cv_df
    .sort_values(
        "alpha"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 5. 找到CV最优alpha
# ============================================================

best_idx = (
    cv_df[
        "mean_test_score"
    ]
    .idxmax()
)


best_row = (
    cv_df
    .loc[
        best_idx
    ]
)


alpha_cv = float(
    best_row[
        "alpha"
    ]
)

best_mean = float(
    best_row[
        "mean_test_score"
    ]
)

best_std = float(
    best_row[
        "std_test_score"
    ]
)


# ============================================================
# 6. 计算1-SE阈值
# ============================================================

best_se = (
    best_std
    /
    np.sqrt(
        N_SPLITS
    )
)


one_se_threshold = (
    best_mean
    -
    best_se
)


# ============================================================
# 7. 找到满足1-SE条件的最大alpha
#
# score越大越好：
#
# mean_test_score >= best_mean - SE
#
# 在满足条件的候选值中选最大的alpha
# ============================================================

eligible_df = cv_df[
    (
        cv_df[
            "alpha"
        ]
        >
        0
    )
    &
    (
        cv_df[
            "mean_test_score"
        ]
        >=
        one_se_threshold
    )
].copy()


alpha_1se = float(
    eligible_df[
        "alpha"
    ].max()
)


row_1se = (
    eligible_df[
        np.isclose(
            eligible_df[
                "alpha"
            ],
            alpha_1se
        )
    ]
    .iloc[
        0
    ]
)


score_1se = float(
    row_1se[
        "mean_test_score"
    ]
)


print(
    "\n======================================"
)

print(
    "1-SE Graphical Lasso参数选择"
)

print(
    "======================================"
)


print(
    f"CV最优 alpha       = {alpha_cv:.6f}"
)

print(
    f"CV最优平均score    = {best_mean:.6f}"
)

print(
    f"CV score标准差     = {best_std:.6f}"
)

print(
    f"CV score标准误     = {best_se:.6f}"
)

print(
    f"1-SE score下界     = {one_se_threshold:.6f}"
)

print(
    f"1-SE选择 alpha     = {alpha_1se:.6f}"
)

print(
    f"1-SE对应CV score   = {score_1se:.6f}"
)


# ============================================================
# 8. 保存参数选择结果
# ============================================================

selection_df = pd.DataFrame(
    [
        {
            "alpha_cv":
                alpha_cv,

            "cv_best_mean_score":
                best_mean,

            "cv_best_std_score":
                best_std,

            "n_splits":
                N_SPLITS,

            "cv_best_se":
                best_se,

            "one_se_score_threshold":
                one_se_threshold,

            "alpha_1se":
                alpha_1se,

            "alpha_1se_mean_score":
                score_1se
        }
    ]
)


selection_df.to_csv(
    SELECTION_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 读取股票收益率
# ============================================================

returns = pd.read_csv(
    RETURN_FILE,
    index_col=0,
    parse_dates=True
)


returns.columns = (
    returns.columns
    .astype(str)
    .str
    .zfill(6)
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
    "\n收益率维度：",
    returns.shape
)


# ============================================================
# 10. 全样本标准化
#
# 这里保持和你阶段四最终模型完全相同的定义，
# 以保证CV网络和1-SE网络可直接比较
# ============================================================

scaler = StandardScaler(
    with_mean=True,
    with_std=True
)


Z_values = scaler.fit_transform(
    returns.values
)


# ============================================================
# 11. 使用alpha_1SE重新拟合Graphical Lasso
# ============================================================

model_1se = GraphicalLasso(

    alpha=
        alpha_1se,

    mode=
        "cd",

    tol=
        OUTER_TOL,

    enet_tol=
        ENET_TOL,

    max_iter=
        MAX_ITER,

    assume_centered=
        True
)


model_1se.fit(
    Z_values
)


precision_1se = (
    model_1se.precision_
)


print(
    "\n1-SE模型迭代次数：",
    model_1se.n_iter_
)


if hasattr(
    model_1se,
    "costs_"
):

    final_gap = (
        model_1se
        .costs_[-1][1]
    )

    print(
        "最终dual gap：",
        final_gap
    )


# ============================================================
# 12. 保存1-SE精度矩阵
# ============================================================

precision_1se_df = pd.DataFrame(
    precision_1se,
    index=codes,
    columns=codes
)


precision_1se_df.to_csv(
    PRECISION_1SE_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 13. Precision -> Partial Correlation
# ============================================================

diag = np.sqrt(
    np.diag(
        precision_1se
    )
)


partial_1se = (
    -precision_1se
    /
    np.outer(
        diag,
        diag
    )
)


np.fill_diagonal(
    partial_1se,
    1.0
)


partial_1se_df = pd.DataFrame(
    partial_1se,
    index=codes,
    columns=codes
)


partial_1se_df.to_csv(
    PARTIAL_1SE_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 读取股票基本信息
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

    value = name_map.get(
        code,
        code
    )

    if pd.isna(
        value
    ):

        return code

    return str(
        value
    )


def get_industry(code):

    value = industry_map.get(
        code,
        "未知行业"
    )

    if pd.isna(
        value
    ):

        return "未知行业"

    return str(
        value
    )


# ============================================================
# 15. 构建1-SE网络
# ============================================================

G_1se = nx.Graph()


for code in codes:

    G_1se.add_node(

        code,

        name=
            get_name(
                code
            ),

        industry=
            get_industry(
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
            precision_1se[
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

            rho_ij = (
                partial_1se[
                    i,
                    j
                ]
            )


            code_i = (
                codes[i]
            )

            code_j = (
                codes[j]
            )


            G_1se.add_edge(

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


edge_1se_df = pd.DataFrame(
    edge_rows
)


if not edge_1se_df.empty:

    edge_1se_df = (
        edge_1se_df
        .sort_values(
            "abs_partial_correlation",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


edge_1se_df.to_csv(
    EDGE_1SE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 计算1-SE网络统计
# ============================================================

n_nodes = (
    G_1se.number_of_nodes()
)

n_edges = (
    G_1se.number_of_edges()
)

density = (
    nx.density(
        G_1se
    )
)

n_components = (
    nx.number_connected_components(
        G_1se
    )
)

n_isolated = len(
    list(
        nx.isolates(
            G_1se
        )
    )
)


degrees = [
    degree

    for _, degree
    in G_1se.degree()
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


summary_1se_df = pd.DataFrame(
    [
        {
            "alpha_1se":
                alpha_1se,

            "cv_mean_score":
                score_1se,

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


summary_1se_df.to_csv(
    SUMMARY_1SE_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n======================================"
)

print(
    "1-SE Graphical Lasso网络"
)

print(
    "======================================"
)


print(
    "alpha：",
    alpha_1se
)

print(
    "边数：",
    n_edges
)

print(
    "Density：",
    round(
        density,
        4
    )
)

print(
    "Mean degree：",
    round(
        mean_degree,
        4
    )
)

print(
    "Max degree：",
    max_degree
)

print(
    "Components：",
    n_components
)

print(
    "Isolated：",
    n_isolated
)


# ============================================================
# 17. CV网络 vs 1-SE网络
# ============================================================

if CV_EDGE_FILE.exists():

    cv_edges_df = pd.read_csv(
        CV_EDGE_FILE,
        dtype={
            "stock_1": str,
            "stock_2": str
        }
    )


    cv_edges_df[
        "stock_1"
    ] = (
        cv_edges_df[
            "stock_1"
        ]
        .str
        .zfill(6)
    )


    cv_edges_df[
        "stock_2"
    ] = (
        cv_edges_df[
            "stock_2"
        ]
        .str
        .zfill(6)
    )


    # --------------------------------------------------------
    # 将边表示成排序后的tuple
    # --------------------------------------------------------

    def edge_key(
        a,
        b
    ):

        return tuple(
            sorted(
                [
                    str(a).zfill(6),
                    str(b).zfill(6)
                ]
            )
        )


    E_cv = {
        edge_key(
            row.stock_1,
            row.stock_2
        )

        for row in cv_edges_df.itertuples()
    }


    E_1se = {
        edge_key(
            row.stock_1,
            row.stock_2
        )

        for row in edge_1se_df.itertuples()
    }


    E_common = (
        E_cv
        &
        E_1se
    )

    E_cv_only = (
        E_cv
        -
        E_1se
    )

    E_1se_only = (
        E_1se
        -
        E_cv
    )

    E_union = (
        E_cv
        |
        E_1se
    )


    jaccard = (
        len(
            E_common
        )
        /
        len(
            E_union
        )

        if len(
            E_union
        ) > 0
        else np.nan
    )


    cv_retention = (
        len(
            E_common
        )
        /
        len(
            E_cv
        )

        if len(
            E_cv
        ) > 0
        else np.nan
    )


    # --------------------------------------------------------
    # 网络比较摘要
    # --------------------------------------------------------

    network_compare_df = pd.DataFrame(
        [
            {
                "alpha_cv":
                    alpha_cv,

                "alpha_1se":
                    alpha_1se,

                "cv_edges":
                    len(
                        E_cv
                    ),

                "one_se_edges":
                    len(
                        E_1se
                    ),

                "common_edges":
                    len(
                        E_common
                    ),

                "cv_only_edges":
                    len(
                        E_cv_only
                    ),

                "one_se_only_edges":
                    len(
                        E_1se_only
                    ),

                "jaccard_similarity":
                    jaccard,

                "cv_edge_retention":
                    cv_retention
            }
        ]
    )


    network_compare_df.to_csv(
        NETWORK_COMPARE_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # 边级比较表
    # --------------------------------------------------------

    compare_rows = []


    for edge in sorted(
        E_union
    ):

        stock_1, stock_2 = edge


        if (
            edge in E_common
        ):

            edge_type = (
                "Common"
            )

        elif (
            edge in E_cv_only
        ):

            edge_type = (
                "CV-only"
            )

        else:

            edge_type = (
                "1SE-only"
            )


        compare_rows.append(
            {
                "stock_1":
                    stock_1,

                "name_1":
                    get_name(
                        stock_1
                    ),

                "stock_2":
                    stock_2,

                "name_2":
                    get_name(
                        stock_2
                    ),

                "edge_type":
                    edge_type
            }
        )


    edge_compare_df = pd.DataFrame(
        compare_rows
    )


    edge_compare_df.to_csv(
        EDGE_COMPARE_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "\n======================================"
    )

    print(
        "CV网络 vs 1-SE网络"
    )

    print(
        "======================================"
    )


    print(
        "CV边数：",
        len(
            E_cv
        )
    )

    print(
        "1-SE边数：",
        len(
            E_1se
        )
    )

    print(
        "共同边：",
        len(
            E_common
        )
    )

    print(
        "CV-only：",
        len(
            E_cv_only
        )
    )

    print(
        "1SE-only：",
        len(
            E_1se_only
        )
    )

    print(
        "Jaccard：",
        round(
            jaccard,
            4
        )
    )

    print(
        "CV边保留率：",
        round(
            cv_retention,
            4
        )
    )


# ============================================================
# 18. 绘制1-SE网络
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

    for code in G_1se.nodes()
]


node_labels = {

    code:
        f"{code}\n{get_name(code)}"

    for code in G_1se.nodes()
}


# Graphical Lasso中的strength只是相似性强度；
# spring layout用于可视化，
# 节点距离不解释为金融距离
pos = nx.spring_layout(

    G_1se,

    seed=42,

    weight=
        "strength"
)


edge_list = list(
    G_1se.edges(
        data=True
    )
)


edge_strengths = np.array(
    [
        data[
            "strength"
        ]

        for _, _, data
        in edge_list
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

            1
            +
            6
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
            3.0,
            len(
                edge_strengths
            )
        )

else:

    edge_widths = []


plt.figure(
    figsize=(
        15,
        12
    )
)


nx.draw_networkx_nodes(

    G_1se,
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


    nx.draw_networkx_edges(

        G_1se,
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
            (
                "solid"
                if rho > 0
                else "dashed"
            ),

        alpha=
            0.75
    )


nx.draw_networkx_labels(

    G_1se,
    pos,

    labels=
        node_labels,

    font_size=
        7
)


edge_labels = {

    (
        u,
        v
    ):
        f"{data['partial_correlation']:.2f}"

    for u, v, data
    in edge_list
}


nx.draw_networkx_edge_labels(

    G_1se,
    pos,

    edge_labels=
        edge_labels,

    font_size=
        7
)


plt.title(
    "1-SE Graphical Lasso股票条件关联网络\n"
    f"alpha = {alpha_1se:.4f}, "
    f"Edges = {n_edges}"
)


plt.axis(
    "off"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "graphical_lasso_1se_network.png",

    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 19. CV score图：同时标出CV和1-SE参数
# ============================================================

plt.figure(
    figsize=(
        8,
        6
    )
)


plt.plot(

    cv_df[
        "alpha"
    ],

    cv_df[
        "mean_test_score"
    ],

    marker=
        "o"
)


plt.axvline(

    alpha_cv,

    linestyle=
        "--",

    label=
        f"CV alpha={alpha_cv:.4f}"
)


plt.axvline(

    alpha_1se,

    linestyle=
        ":",

    label=
        f"1-SE alpha={alpha_1se:.4f}"
)


plt.axhline(

    one_se_threshold,

    linestyle=
        "-.",

    label=
        f"1-SE score threshold={one_se_threshold:.3f}"
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
    "CV-optimal 与 1-SE Graphical Lasso 参数"
)


plt.legend()

plt.grid(
    alpha=
        0.3
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "glasso_cv_vs_1se_selection.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 20. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "阶段四补充完成"
)

print(
    "======================================"
)


print(
    "\n输出文件："
)


for path in [

    SELECTION_FILE,

    PRECISION_1SE_FILE,

    PARTIAL_1SE_FILE,

    EDGE_1SE_FILE,

    SUMMARY_1SE_FILE,

    EDGE_COMPARE_FILE,

    NETWORK_COMPARE_FILE,

    FIGURE_DIR
    / "graphical_lasso_1se_network.png",

    FIGURE_DIR
    / "glasso_cv_vs_1se_selection.png"
]:

    print(
        path
    )
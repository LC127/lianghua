from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.lines import Line2D


# ============================================================
# 0. 中文字体
# ============================================================

def set_chinese_font():

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC"
    ]

    installed = {
        f.name
        for f in font_manager.fontManager.ttflist
    }

    for font_name in candidates:

        if font_name in installed:

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


# ------------------------------------------------------------
# 股票信息
# ------------------------------------------------------------

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ------------------------------------------------------------
# 昨日：非正则化偏相关网络
# ------------------------------------------------------------

PARTIAL_MATRIX_FILE = (
    PROCESSED_DIR
    / "partial_correlation.csv"
)

PARTIAL_EDGE_FILE = (
    PROCESSED_DIR
    / "partial_network_edges.csv"
)


# ------------------------------------------------------------
# 今日：CV Graphical Lasso
# ------------------------------------------------------------

GLASSO_CV_MATRIX_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_partial_correlation.csv"
)

GLASSO_CV_EDGE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_edges.csv"
)


# ------------------------------------------------------------
# 今日：1-SE Graphical Lasso
# ------------------------------------------------------------

GLASSO_1SE_MATRIX_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_partial_correlation.csv"
)

GLASSO_1SE_EDGE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_edges.csv"
)


# ============================================================
# 2. 输出文件
# ============================================================

PAIR_OUTPUT_FILE = (
    PROCESSED_DIR
    / "partial_vs_glasso_all_pairs.csv"
)

OVERLAP_OUTPUT_FILE = (
    PROCESSED_DIR
    / "partial_vs_glasso_overlap_summary.csv"
)

NETWORK_OUTPUT_FILE = (
    PROCESSED_DIR
    / "partial_vs_glasso_network_summary.csv"
)

NODE_OUTPUT_FILE = (
    PROCESSED_DIR
    / "partial_vs_glasso_node_comparison.csv"
)

COMMON_EDGE_OUTPUT_FILE = (
    PROCESSED_DIR
    / "partial_vs_glasso_common_edges.csv"
)


# ============================================================
# 3. 工具函数：股票代码标准化
# ============================================================

def normalize_code(x):

    return str(
        x
    ).strip().zfill(6)


# ============================================================
# 4. 读取完整矩阵
# ============================================================

def read_matrix(
    path
):

    df = pd.read_csv(
        path,
        index_col=0
    )

    df.index = [
        normalize_code(
            x
        )
        for x in df.index
    ]

    df.columns = [
        normalize_code(
            x
        )
        for x in df.columns
    ]

    return df.astype(float)


partial_matrix = read_matrix(
    PARTIAL_MATRIX_FILE
)

glasso_cv_matrix = read_matrix(
    GLASSO_CV_MATRIX_FILE
)

glasso_1se_matrix = read_matrix(
    GLASSO_1SE_MATRIX_FILE
)


# ============================================================
# 5. 检查股票集合
# ============================================================

codes = (
    partial_matrix
    .columns
    .tolist()
)


if set(
    glasso_cv_matrix.columns
) != set(
    codes
):

    raise ValueError(
        "CV GLasso矩阵和偏相关矩阵股票代码不一致。"
    )


if set(
    glasso_1se_matrix.columns
) != set(
    codes
):

    raise ValueError(
        "1-SE GLasso矩阵和偏相关矩阵股票代码不一致。"
    )


# 按完全相同的顺序排列
glasso_cv_matrix = (
    glasso_cv_matrix
    .loc[
        codes,
        codes
    ]
)

glasso_1se_matrix = (
    glasso_1se_matrix
    .loc[
        codes,
        codes
    ]
)


print(
    "股票数量：",
    len(
        codes
    )
)


# ============================================================
# 6. 读取股票名称和行业
# ============================================================

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
    .apply(
        normalize_code
    )
)


name_map = {}

industry_map = {}


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


def get_name(
    code
):

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


def get_industry(
    code
):

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
# 7. 读取网络边表
# ============================================================

def read_edge_file(
    path
):

    df = pd.read_csv(
        path,
        dtype={
            "stock_1": str,
            "stock_2": str
        }
    )


    df[
        "stock_1"
    ] = (
        df[
            "stock_1"
        ]
        .apply(
            normalize_code
        )
    )


    df[
        "stock_2"
    ] = (
        df[
            "stock_2"
        ]
        .apply(
            normalize_code
        )
    )


    return df


partial_edges = read_edge_file(
    PARTIAL_EDGE_FILE
)

glasso_cv_edges = read_edge_file(
    GLASSO_CV_EDGE_FILE
)

glasso_1se_edges = read_edge_file(
    GLASSO_1SE_EDGE_FILE
)


# ============================================================
# 8. 边统一表示
# ============================================================

def edge_key(
    a,
    b
):

    return tuple(
        sorted(
            [
                normalize_code(
                    a
                ),

                normalize_code(
                    b
                )
            ]
        )
    )


def edge_set(
    edge_df
):

    return {
        edge_key(
            row.stock_1,
            row.stock_2
        )

        for row
        in edge_df.itertuples()
    }


E_partial = edge_set(
    partial_edges
)

E_cv = edge_set(
    glasso_cv_edges
)

E_1se = edge_set(
    glasso_1se_edges
)


print(
    "\n边数："
)

print(
    "Partial threshold =",
    len(
        E_partial
    )
)

print(
    "GLasso CV        =",
    len(
        E_cv
    )
)

print(
    "GLasso 1-SE      =",
    len(
        E_1se
    )
)


# ============================================================
# 9. 两个网络的边集比较函数
# ============================================================

def compare_edge_sets(
    E_A,
    E_B,
    name_A,
    name_B
):

    common = (
        E_A
        &
        E_B
    )

    A_only = (
        E_A
        -
        E_B
    )

    B_only = (
        E_B
        -
        E_A
    )

    union = (
        E_A
        |
        E_B
    )


    jaccard = (
        len(
            common
        )
        /
        len(
            union
        )

        if len(
            union
        ) > 0

        else np.nan
    )


    A_retention = (
        len(
            common
        )
        /
        len(
            E_A
        )

        if len(
            E_A
        ) > 0

        else np.nan
    )


    B_supported_by_A = (
        len(
            common
        )
        /
        len(
            E_B
        )

        if len(
            E_B
        ) > 0

        else np.nan
    )


    return {
        "comparison":
            f"{name_A} vs {name_B}",

        "edges_A":
            len(
                E_A
            ),

        "edges_B":
            len(
                E_B
            ),

        "common_edges":
            len(
                common
            ),

        "A_only":
            len(
                A_only
            ),

        "B_only":
            len(
                B_only
            ),

        "union_edges":
            len(
                union
            ),

        "jaccard":
            jaccard,

        "A_edge_retention":
            A_retention,

        "B_edge_supported_by_A":
            B_supported_by_A
    }


# ============================================================
# 10. 三组比较
#
# 主要：
# Partial vs 1-SE
#
# 敏感性：
# Partial vs CV
#
# 同时保留已知：
# CV vs 1-SE
# ============================================================

overlap_rows = [

    compare_edge_sets(
        E_partial,
        E_1se,
        "Partial",
        "GLasso_1SE"
    ),

    compare_edge_sets(
        E_partial,
        E_cv,
        "Partial",
        "GLasso_CV"
    ),

    compare_edge_sets(
        E_cv,
        E_1se,
        "GLasso_CV",
        "GLasso_1SE"
    )
]


overlap_df = pd.DataFrame(
    overlap_rows
)


overlap_df.to_csv(
    OVERLAP_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========================================"
)

print(
    "边结构比较"
)

print(
    "========================================"
)


print(
    overlap_df
    .to_string(
        index=False
    )
)


# ============================================================
# 11. 构建所有105个股票对的比较表
# ============================================================

pair_rows = []


for i in range(
    len(
        codes
    )
):

    for j in range(
        i + 1,
        len(
            codes
        )
    ):

        stock_1 = (
            codes[i]
        )

        stock_2 = (
            codes[j]
        )


        edge = edge_key(
            stock_1,
            stock_2
        )


        partial_value = float(
            partial_matrix.loc[
                stock_1,
                stock_2
            ]
        )


        cv_value = float(
            glasso_cv_matrix.loc[
                stock_1,
                stock_2
            ]
        )


        one_se_value = float(
            glasso_1se_matrix.loc[
                stock_1,
                stock_2
            ]
        )


        in_partial = (
            edge in E_partial
        )

        in_cv = (
            edge in E_cv
        )

        in_1se = (
            edge in E_1se
        )


        # ----------------------------------------------------
        # 主要比较：Partial vs 1-SE
        # ----------------------------------------------------

        if (
            in_partial
            and
            in_1se
        ):

            main_edge_type = (
                "Common"
            )

        elif in_partial:

            main_edge_type = (
                "Partial-only"
            )

        elif in_1se:

            main_edge_type = (
                "GLasso-1SE-only"
            )

        else:

            main_edge_type = (
                "Neither"
            )


        # ----------------------------------------------------
        # 符号一致性
        # ----------------------------------------------------

        if (
            in_partial
            and
            in_1se
        ):

            sign_same = (
                np.sign(
                    partial_value
                )
                ==
                np.sign(
                    one_se_value
                )
            )

        else:

            sign_same = (
                np.nan
            )


        # ----------------------------------------------------
        # GLasso相对于普通Partial的绝对值变化
        # ----------------------------------------------------

        abs_shrinkage_1se = (
            abs(
                partial_value
            )
            -
            abs(
                one_se_value
            )
        )


        pair_rows.append(
            {
                "stock_1":
                    stock_1,

                "name_1":
                    get_name(
                        stock_1
                    ),

                "industry_1":
                    get_industry(
                        stock_1
                    ),

                "stock_2":
                    stock_2,

                "name_2":
                    get_name(
                        stock_2
                    ),

                "industry_2":
                    get_industry(
                        stock_2
                    ),

                "partial":
                    partial_value,

                "glasso_cv_partial":
                    cv_value,

                "glasso_1se_partial":
                    one_se_value,

                "abs_partial":
                    abs(
                        partial_value
                    ),

                "abs_glasso_cv":
                    abs(
                        cv_value
                    ),

                "abs_glasso_1se":
                    abs(
                        one_se_value
                    ),

                "in_partial_network":
                    in_partial,

                "in_glasso_cv":
                    in_cv,

                "in_glasso_1se":
                    in_1se,

                "main_edge_type":
                    main_edge_type,

                "sign_same_partial_1se":
                    sign_same,

                "abs_shrinkage_partial_to_1se":
                    abs_shrinkage_1se,

                "same_industry":
                    (
                        get_industry(
                            stock_1
                        )
                        ==
                        get_industry(
                            stock_2
                        )
                    )
            }
        )


pair_df = pd.DataFrame(
    pair_rows
)


pair_df.to_csv(
    PAIR_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 主比较中的共同边
# ============================================================

common_df = pair_df[
    (
        pair_df[
            "in_partial_network"
        ]
    )
    &
    (
        pair_df[
            "in_glasso_1se"
        ]
    )
].copy()


common_df = (
    common_df
    .sort_values(
        "abs_partial",
        ascending=False
    )
)


common_df.to_csv(
    COMMON_EDGE_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 共同边符号一致率
# ============================================================

if len(
    common_df
) > 0:

    sign_agreement = (
        common_df[
            "sign_same_partial_1se"
        ]
        .mean()
    )

else:

    sign_agreement = (
        np.nan
    )


print(
    "\nPartial vs 1-SE共同边符号一致率：",
    sign_agreement
)


# ============================================================
# 14. 根据边表构建Graph
# ============================================================

def build_graph(
    edge_df,
    all_codes
):

    G = nx.Graph()


    for code in all_codes:

        G.add_node(

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


    for row in edge_df.itertuples():

        u = normalize_code(
            row.stock_1
        )

        v = normalize_code(
            row.stock_2
        )


        # ----------------------------------------------------
        # 找 partial correlation 列
        # ----------------------------------------------------

        if hasattr(
            row,
            "partial_correlation"
        ):

            rho = float(
                row.partial_correlation
            )

        else:

            rho = (
                np.nan
            )


        strength = (
            abs(
                rho
            )

            if not np.isnan(
                rho
            )

            else 1.0
        )


        G.add_edge(

            u,
            v,

            partial_correlation=
                rho,

            strength=
                strength
        )


    return G


G_partial = build_graph(
    partial_edges,
    codes
)

G_cv = build_graph(
    glasso_cv_edges,
    codes
)

G_1se = build_graph(
    glasso_1se_edges,
    codes
)


# ============================================================
# 15. 网络层统计
# ============================================================

def network_summary(
    G,
    network_name
):

    n_nodes = (
        G.number_of_nodes()
    )

    n_edges = (
        G.number_of_edges()
    )


    degrees = [
        degree
        for _, degree
        in G.degree()
    ]


    # --------------------------------------------------------
    # 同行业边比例
    # --------------------------------------------------------

    same_industry_edges = 0


    for u, v in G.edges():

        if (
            get_industry(
                u
            )
            ==
            get_industry(
                v
            )
        ):

            same_industry_edges += 1


    same_industry_ratio = (

        same_industry_edges
        /
        n_edges

        if n_edges > 0

        else np.nan
    )


    # --------------------------------------------------------
    # 行业 assortativity
    # --------------------------------------------------------

    for node in G.nodes():

        G.nodes[
            node
        ][
            "industry"
        ] = get_industry(
            node
        )


    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            assortativity = (
                nx.attribute_assortativity_coefficient(
                    G,
                    "industry"
                )
            )

    except Exception:

        assortativity = (
            np.nan
        )


    return {
        "network":
            network_name,

        "n_nodes":
            n_nodes,

        "n_edges":
            n_edges,

        "density":
            nx.density(
                G
            ),

        "mean_degree":
            np.mean(
                degrees
            ),

        "max_degree":
            np.max(
                degrees
            ),

        "n_components":
            nx.number_connected_components(
                G
            ),

        "n_isolated":
            len(
                list(
                    nx.isolates(
                        G
                    )
                )
            ),

        "same_industry_edges":
            same_industry_edges,

        "same_industry_edge_ratio":
            same_industry_ratio,

        "industry_assortativity":
            assortativity
    }


network_df = pd.DataFrame(
    [
        network_summary(
            G_partial,
            "Partial_threshold"
        ),

        network_summary(
            G_cv,
            "GLasso_CV"
        ),

        network_summary(
            G_1se,
            "GLasso_1SE"
        )
    ]
)


network_df.to_csv(
    NETWORK_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========================================"
)

print(
    "网络整体统计"
)

print(
    "========================================"
)


print(
    network_df
    .to_string(
        index=False
    )
)


# ============================================================
# 16. 节点层指标
#
# Degree
# Strength
# Betweenness
# Closeness
# ============================================================

def node_metrics(
    G,
    suffix
):

    degree = dict(
        G.degree()
    )


    strength = {

        node:
            sum(
                abs(
                    data.get(
                        "partial_correlation",
                        0.0
                    )
                )

                for _, _, data
                in G.edges(
                    node,
                    data=True
                )

                if not np.isnan(
                    data.get(
                        "partial_correlation",
                        np.nan
                    )
                )
            )

        for node in G.nodes()
    }


    # 使用无权重中心性：
    # 避免把相似性误当成距离
    betweenness = (
        nx.betweenness_centrality(
            G,
            weight=None
        )
    )


    closeness = (
        nx.closeness_centrality(
            G
        )
    )


    return pd.DataFrame(
        {
            "code":
                list(
                    G.nodes()
                ),

            f"degree_{suffix}":
                [
                    degree[
                        code
                    ]
                    for code in G.nodes()
                ],

            f"strength_{suffix}":
                [
                    strength[
                        code
                    ]
                    for code in G.nodes()
                ],

            f"betweenness_{suffix}":
                [
                    betweenness[
                        code
                    ]
                    for code in G.nodes()
                ],

            f"closeness_{suffix}":
                [
                    closeness[
                        code
                    ]
                    for code in G.nodes()
                ]
        }
    )


node_partial = node_metrics(
    G_partial,
    "partial"
)

node_cv = node_metrics(
    G_cv,
    "cv"
)

node_1se = node_metrics(
    G_1se,
    "1se"
)


node_df = (
    node_partial
    .merge(
        node_cv,
        on="code"
    )
    .merge(
        node_1se,
        on="code"
    )
)


node_df[
    "name"
] = (
    node_df[
        "code"
    ]
    .map(
        get_name
    )
)


node_df[
    "industry"
] = (
    node_df[
        "code"
    ]
    .map(
        get_industry
    )
)


# ------------------------------------------------------------
# Degree变化
# ------------------------------------------------------------

node_df[
    "degree_change_1se_minus_partial"
] = (
    node_df[
        "degree_1se"
    ]
    -
    node_df[
        "degree_partial"
    ]
)


node_df[
    "strength_change_1se_minus_partial"
] = (
    node_df[
        "strength_1se"
    ]
    -
    node_df[
        "strength_partial"
    ]
)


node_df = (
    node_df
    .sort_values(
        "degree_1se",
        ascending=False
    )
)


node_df.to_csv(
    NODE_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========================================"
)

print(
    "节点Degree比较"
)

print(
    "========================================"
)


print(
    node_df[
        [
            "code",
            "name",
            "degree_partial",
            "degree_cv",
            "degree_1se",
            "degree_change_1se_minus_partial"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 17. 绘制主比较图：
#
# Partial threshold vs GLasso 1-SE
#
# Common        -> 实线
# Partial-only  -> 虚线
# GLasso-only   -> 点线
# ============================================================

E_common = (
    E_partial
    &
    E_1se
)

E_partial_only = (
    E_partial
    -
    E_1se
)

E_1se_only = (
    E_1se
    -
    E_partial
)


# ------------------------------------------------------------
# 用两个网络的并集确定布局
# ------------------------------------------------------------

G_union = nx.Graph()


G_union.add_nodes_from(
    codes
)


G_union.add_edges_from(
    E_partial
    |
    E_1se
)


# 为了突出“边变化”，
# 使用固定圆形布局，
# 不赋予节点间几何距离金融含义
pos = nx.circular_layout(
    codes
)


# ============================================================
# 18. 行业颜色
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
        get_industry(
            code
        )
    ]

    for code in codes
]


node_labels = {

    code:
        f"{code}\n{get_name(code)}"

    for code in codes
}


# ============================================================
# 19. 绘图
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        15,
        13
    )
)


nx.draw_networkx_nodes(

    G_union,
    pos,

    node_color=
        node_colors,

    node_size=
        1900,

    edgecolors=
        "black",

    linewidths=
        0.8,

    ax=
        ax
)


# ------------------------------------------------------------
# Common edges
# ------------------------------------------------------------

nx.draw_networkx_edges(

    G_union,
    pos,

    edgelist=
        list(
            E_common
        ),

    width=
        3.0,

    style=
        "solid",

    edge_color=
        "#333333",

    alpha=
        0.8,

    ax=
        ax
)


# ------------------------------------------------------------
# Partial-only
# ------------------------------------------------------------

nx.draw_networkx_edges(

    G_union,
    pos,

    edgelist=
        list(
            E_partial_only
        ),

    width=
        2.0,

    style=
        "dashed",

    edge_color=
        "#E69F00",

    alpha=
        0.8,

    ax=
        ax
)


# ------------------------------------------------------------
# GLasso 1-SE only
# ------------------------------------------------------------

nx.draw_networkx_edges(

    G_union,
    pos,

    edgelist=
        list(
            E_1se_only
        ),

    width=
        2.0,

    style=
        "dotted",

    edge_color=
        "#0072B2",

    alpha=
        0.8,

    ax=
        ax
)


nx.draw_networkx_labels(

    G_union,
    pos,

    labels=
        node_labels,

    font_size=
        7,

    ax=
        ax
)


# ============================================================
# 20. 图例
# ============================================================

industry_handles = [

    Line2D(
        [0],
        [0],

        marker=
            "o",

        linestyle=
            "",

        markerfacecolor=
            color,

        markeredgecolor=
            "black",

        markersize=
            9,

        label=
            industry
    )

    for industry, color
    in industry_color.items()
]


edge_handles = [

    Line2D(
        [0],
        [0],

        color=
            "#333333",

        linewidth=
            3,

        linestyle=
            "solid",

        label=
            "两个网络共同边"
    ),

    Line2D(
        [0],
        [0],

        color=
            "#E69F00",

        linewidth=
            2,

        linestyle=
            "dashed",

        label=
            "仅偏相关阈值网络"
    ),

    Line2D(
        [0],
        [0],

        color=
            "#0072B2",

        linewidth=
            2,

        linestyle=
            "dotted",

        label=
            "仅GLasso 1-SE网络"
    )
]


legend_1 = ax.legend(

    handles=
        industry_handles,

    title=
        "行业",

    loc=
        "upper left",

    bbox_to_anchor=
        (
            1.01,
            1.00
        )
)


ax.add_artist(
    legend_1
)


ax.legend(

    handles=
        edge_handles,

    title=
        "边类型",

    loc=
        "lower left",

    bbox_to_anchor=
        (
            1.01,
            0.00
        )
)


ax.set_title(
    "偏相关阈值网络 vs Graphical Lasso 1-SE网络\n"
    "Common / Partial-only / GLasso-only",
    fontsize=15
)


ax.axis(
    "off"
)


plt.tight_layout()


plt.savefig(

    FIGURE_DIR
    / "partial_vs_glasso_1se_edge_change_network.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 21. 完成
# ============================================================

print(
    "\n========================================"
)

print(
    "阶段五完成"
)

print(
    "========================================"
)


print(
    "\n输出文件："
)


for path in [

    PAIR_OUTPUT_FILE,

    OVERLAP_OUTPUT_FILE,

    NETWORK_OUTPUT_FILE,

    NODE_OUTPUT_FILE,

    COMMON_EDGE_OUTPUT_FILE,

    FIGURE_DIR
    / "partial_vs_glasso_1se_edge_change_network.png"

]:

    print(
        path
    )
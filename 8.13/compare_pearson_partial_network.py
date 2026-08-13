from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager


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

            plt.rcParams[
                "font.sans-serif"
            ] = [font_name]

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


# 昨日 Pearson 阈值网络
PEARSON_EDGE_FILE = (
    PROCESSED_DIR
    / "threshold_edges.csv"
)

# 今日偏相关网络
PARTIAL_EDGE_FILE = (
    PROCESSED_DIR
    / "partial_network_edges.csv"
)

# 股票名称、行业
STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 2. 读取股票信息
# ============================================================

stock_info = pd.read_csv(
    STOCK_INFO_FILE,
    dtype={
        "code": str
    }
)

stock_info["code"] = (
    stock_info[
        "code"
    ]
    .str
    .zfill(6)
)

stock_info = (
    stock_info
    .set_index(
        "code"
    )
)


all_codes = (
    stock_info
    .index
    .tolist()
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
# 3. 读取 Pearson 阈值网络边表
# ============================================================

pearson_edges = pd.read_csv(
    PEARSON_EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


pearson_edges[
    "stock_1"
] = (
    pearson_edges[
        "stock_1"
    ]
    .str
    .zfill(6)
)

pearson_edges[
    "stock_2"
] = (
    pearson_edges[
        "stock_2"
    ]
    .str
    .zfill(6)
)


# 兼容昨日不同版本的列名
if (
    "correlation"
    in pearson_edges.columns
):

    pearson_edges[
        "pearson"
    ] = (
        pearson_edges[
            "correlation"
        ]
    )

elif (
    "correlation_weight"
    in pearson_edges.columns
):

    pearson_edges[
        "pearson"
    ] = (
        pearson_edges[
            "correlation_weight"
        ]
    )

else:

    raise ValueError(
        "threshold_edges.csv 中"
        "没有 correlation 或 correlation_weight 列。"
    )


# ============================================================
# 4. 读取偏相关网络边表
# ============================================================

partial_edges = pd.read_csv(
    PARTIAL_EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


partial_edges[
    "stock_1"
] = (
    partial_edges[
        "stock_1"
    ]
    .str
    .zfill(6)
)

partial_edges[
    "stock_2"
] = (
    partial_edges[
        "stock_2"
    ]
    .str
    .zfill(6)
)


if (
    "partial_correlation"
    not in partial_edges.columns
):

    raise ValueError(
        "partial_network_edges.csv "
        "缺少 partial_correlation 列。"
    )


partial_edges[
    "abs_partial"
] = (
    partial_edges[
        "partial_correlation"
    ]
    .abs()
)


print(
    "Pearson网络边数：",
    len(
        pearson_edges
    )
)

print(
    "偏相关网络边数：",
    len(
        partial_edges
    )
)


# ============================================================
# 5. 构建 Pearson 网络
# ============================================================

G_pearson = nx.Graph()


# 添加所有股票，保证孤立节点也存在
for code in all_codes:

    G_pearson.add_node(
        code,

        name=get_name(
            code
        ),

        industry=get_industry(
            code
        )
    )


for _, row in (
    pearson_edges
    .iterrows()
):

    G_pearson.add_edge(

        row[
            "stock_1"
        ],

        row[
            "stock_2"
        ],

        correlation=float(
            row[
                "pearson"
            ]
        ),

        strength=float(
            abs(
                row[
                    "pearson"
                ]
            )
        )
    )


# ============================================================
# 6. 构建偏相关网络
# ============================================================

G_partial = nx.Graph()


for code in all_codes:

    G_partial.add_node(
        code,

        name=get_name(
            code
        ),

        industry=get_industry(
            code
        )
    )


for _, row in (
    partial_edges
    .iterrows()
):

    G_partial.add_edge(

        row[
            "stock_1"
        ],

        row[
            "stock_2"
        ],

        partial_correlation=float(
            row[
                "partial_correlation"
            ]
        ),

        strength=float(
            abs(
                row[
                    "partial_correlation"
                ]
            )
        )
    )


# ============================================================
# 7. 将边标准化为无方向集合
# ============================================================

def edge_set(G):

    return {

        tuple(
            sorted(
                [
                    u,
                    v
                ]
            )
        )

        for u, v
        in G.edges()
    }


E_P = edge_set(
    G_pearson
)

E_PC = edge_set(
    G_partial
)


# ============================================================
# 8. 边重叠分析
# ============================================================

common_edges = (
    E_P
    &
    E_PC
)


pearson_only = (
    E_P
    -
    E_PC
)


partial_only = (
    E_PC
    -
    E_P
)


union_edges = (
    E_P
    |
    E_PC
)


# Jaccard similarity
if len(
    union_edges
) > 0:

    jaccard = (
        len(
            common_edges
        )
        /
        len(
            union_edges
        )
    )

else:

    jaccard = np.nan


# Pearson边保留比例
if len(
    E_P
) > 0:

    pearson_retention = (
        len(
            common_edges
        )
        /
        len(
            E_P
        )
    )

else:

    pearson_retention = np.nan


# Partial网络中有多少来自Pearson网络
if len(
    E_PC
) > 0:

    partial_overlap_ratio = (
        len(
            common_edges
        )
        /
        len(
            E_PC
        )
    )

else:

    partial_overlap_ratio = np.nan


print(
    "\n================================"
)

print(
    "边重叠分析"
)

print(
    "================================"
)


print(
    "Pearson网络边数：",
    len(
        E_P
    )
)

print(
    "偏相关网络边数：",
    len(
        E_PC
    )
)

print(
    "共同边数：",
    len(
        common_edges
    )
)

print(
    "Pearson only：",
    len(
        pearson_only
    )
)

print(
    "Partial only：",
    len(
        partial_only
    )
)

print(
    "Jaccard similarity：",
    round(
        jaccard,
        4
    )
)

print(
    "Pearson边保留比例：",
    round(
        pearson_retention,
        4
    )
)


# ============================================================
# 9. 为每条边建立比较表
# ============================================================

pearson_lookup = {

    tuple(
        sorted(
            (
                row[
                    "stock_1"
                ],
                row[
                    "stock_2"
                ]
            )
        )
    ):
        row[
            "pearson"
        ]

    for _, row
    in pearson_edges.iterrows()
}


partial_lookup = {

    tuple(
        sorted(
            (
                row[
                    "stock_1"
                ],
                row[
                    "stock_2"
                ]
            )
        )
    ):
        row[
            "partial_correlation"
        ]

    for _, row
    in partial_edges.iterrows()
}


edge_comparison_rows = []


for edge in sorted(
    union_edges
):

    u, v = edge


    in_pearson = (
        edge
        in E_P
    )

    in_partial = (
        edge
        in E_PC
    )


    if (
        in_pearson
        and
        in_partial
    ):

        status = (
            "Both"
        )

    elif in_pearson:

        status = (
            "Pearson only"
        )

    else:

        status = (
            "Partial only"
        )


    edge_comparison_rows.append(
        {
            "stock_1":
                u,

            "name_1":
                get_name(
                    u
                ),

            "industry_1":
                get_industry(
                    u
                ),

            "stock_2":
                v,

            "name_2":
                get_name(
                    v
                ),

            "industry_2":
                get_industry(
                    v
                ),

            "pearson":
                pearson_lookup.get(
                    edge,
                    np.nan
                ),

            "partial":
                partial_lookup.get(
                    edge,
                    np.nan
                ),

            "edge_status":
                status,

            "same_industry":
                (
                    get_industry(u)
                    ==
                    get_industry(v)
                )
        }
    )


edge_comparison = pd.DataFrame(
    edge_comparison_rows
)


edge_comparison.to_csv(
    PROCESSED_DIR
    /
    "pearson_partial_edge_comparison.csv",

    index=False,

    encoding=
        "utf-8-sig"
)


# ============================================================
# 10. 网络整体统计函数
# ============================================================

def same_industry_edge_ratio(
    G
):

    if (
        G.number_of_edges()
        ==
        0
    ):

        return np.nan


    same_count = 0


    for u, v in (
        G.edges()
    ):

        if (
            get_industry(
                u
            )
            ==
            get_industry(
                v
            )
        ):

            same_count += 1


    return (
        same_count
        /
        G.number_of_edges()
    )


def network_statistics(
    G,
    network_name
):

    degrees = [
        degree

        for _, degree
        in G.degree()
    ]


    isolated = list(
        nx.isolates(
            G
        )
    )


    try:

        assortativity = (
            nx.attribute_assortativity_coefficient(
                G,
                "industry"
            )
        )

    except Exception:

        assortativity = np.nan


    return {

        "network":
            network_name,

        "n_nodes":
            G.number_of_nodes(),

        "n_edges":
            G.number_of_edges(),

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
                isolated
            ),

        "same_industry_edge_ratio":
            same_industry_edge_ratio(
                G
            ),

        "industry_assortativity":
            assortativity
    }


network_summary = pd.DataFrame(
    [
        network_statistics(
            G_pearson,
            "Pearson threshold"
        ),

        network_statistics(
            G_partial,
            "Partial correlation"
        )
    ]
)


network_summary.to_csv(
    PROCESSED_DIR
    /
    "pearson_partial_network_summary.csv",

    index=False,

    encoding=
        "utf-8-sig"
)


print(
    "\n================================"
)

print(
    "网络整体结构比较"
)

print(
    "================================"
)


print(
    network_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 11. 节点指标
# ============================================================

def node_metrics(
    G,
    network_name
):

    # ----------------------------
    # Degree
    # ----------------------------

    degree = dict(
        G.degree()
    )


    # ----------------------------
    # Weighted Degree / Strength
    # ----------------------------

    strength = {}


    for node in G.nodes:

        strength[node] = sum(
            data[
                "strength"
            ]

            for _, _, data
            in G.edges(
                node,
                data=True
            )
        )


    # ----------------------------
    # Degree Centrality
    # ----------------------------

    degree_centrality = (
        nx.degree_centrality(
            G
        )
    )


    # ----------------------------
    # Betweenness
    #
    # 此处使用无权最短路径，
    # 便于两种网络直接比较
    # ----------------------------

    betweenness = (
        nx.betweenness_centrality(
            G,
            normalized=True
        )
    )


    # ----------------------------
    # Closeness
    #
    # 同样使用无权距离
    # ----------------------------

    closeness = (
        nx.closeness_centrality(
            G
        )
    )


    rows = []


    for node in G.nodes:

        rows.append(
            {
                "network":
                    network_name,

                "code":
                    node,

                "name":
                    get_name(
                        node
                    ),

                "industry":
                    get_industry(
                        node
                    ),

                "degree":
                    degree[
                        node
                    ],

                "degree_centrality":
                    degree_centrality[
                        node
                    ],

                "strength":
                    strength[
                        node
                    ],

                "betweenness":
                    betweenness[
                        node
                    ],

                "closeness":
                    closeness[
                        node
                    ]
            }
        )


    return pd.DataFrame(
        rows
    )


pearson_node = node_metrics(
    G_pearson,
    "Pearson"
)


partial_node = node_metrics(
    G_partial,
    "Partial"
)


# ============================================================
# 12. 将节点指标并列比较
# ============================================================

pearson_node_compare = (
    pearson_node[
        [
            "code",
            "name",
            "industry",
            "degree",
            "strength",
            "betweenness",
            "closeness"
        ]
    ]
    .rename(
        columns={
            "degree":
                "pearson_degree",

            "strength":
                "pearson_strength",

            "betweenness":
                "pearson_betweenness",

            "closeness":
                "pearson_closeness"
        }
    )
)


partial_node_compare = (
    partial_node[
        [
            "code",
            "degree",
            "strength",
            "betweenness",
            "closeness"
        ]
    ]
    .rename(
        columns={
            "degree":
                "partial_degree",

            "strength":
                "partial_strength",

            "betweenness":
                "partial_betweenness",

            "closeness":
                "partial_closeness"
        }
    )
)


node_comparison = pd.merge(
    pearson_node_compare,
    partial_node_compare,
    on="code",
    how="outer"
)


# ============================================================
# 13. 计算节点指标变化
# ============================================================

node_comparison[
    "degree_change"
] = (
    node_comparison[
        "partial_degree"
    ]
    -
    node_comparison[
        "pearson_degree"
    ]
)


node_comparison[
    "strength_change"
] = (
    node_comparison[
        "partial_strength"
    ]
    -
    node_comparison[
        "pearson_strength"
    ]
)


node_comparison[
    "betweenness_change"
] = (
    node_comparison[
        "partial_betweenness"
    ]
    -
    node_comparison[
        "pearson_betweenness"
    ]
)


node_comparison.to_csv(
    PROCESSED_DIR
    /
    "pearson_partial_node_comparison.csv",

    index=False,

    encoding=
        "utf-8-sig"
)


# ============================================================
# 14. 输出节点变化最大的股票
# ============================================================

print(
    "\n================================"
)

print(
    "Degree下降最多的股票"
)

print(
    "================================"
)


print(
    node_comparison[
        [
            "code",
            "name",
            "industry",
            "pearson_degree",
            "partial_degree",
            "degree_change"
        ]
    ]
    .sort_values(
        "degree_change"
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. 输出共同边
# ============================================================

print(
    "\n================================"
)

print(
    "两个网络共同保留的边"
)

print(
    "================================"
)


common_table = (
    edge_comparison[
        edge_comparison[
            "edge_status"
        ]
        ==
        "Both"
    ]
)


if not common_table.empty:

    print(
        common_table[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "pearson",
                "partial",
                "same_industry"
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有共同边。"
    )


# ============================================================
# 16. 输出 Pearson-only 边
# ============================================================

print(
    "\n================================"
)

print(
    "Pearson有、Partial没有的边"
)

print(
    "================================"
)


pearson_only_table = (
    edge_comparison[
        edge_comparison[
            "edge_status"
        ]
        ==
        "Pearson only"
    ]
)


if not pearson_only_table.empty:

    print(
        pearson_only_table[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "pearson",
                "same_industry"
            ]
        ]
        .sort_values(
            "pearson",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 17. 输出 Partial-only 边
# ============================================================

print(
    "\n================================"
)

print(
    "Partial有、Pearson没有的边"
)

print(
    "================================"
)


partial_only_table = (
    edge_comparison[
        edge_comparison[
            "edge_status"
        ]
        ==
        "Partial only"
    ]
)


if not partial_only_table.empty:

    print(
        partial_only_table[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "partial",
                "same_industry"
            ]
        ]
        .assign(
            abs_partial=lambda x:
                x[
                    "partial"
                ]
                .abs()
        )
        .sort_values(
            "abs_partial",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 18. 行业颜色
# ============================================================

industries = sorted(
    stock_info[
        "industry"
    ]
    .fillna(
        "未知行业"
    )
    .unique()
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


def node_colors(
    G
):

    return [

        industry_color_map.get(
            get_industry(
                node
            )
        )

        for node
        in G.nodes
    ]


# ============================================================
# 19. 使用统一节点布局
# ============================================================

G_union = nx.compose(
    G_pearson,
    G_partial
)


pos = nx.spring_layout(
    G_union,
    seed=42
)


labels = {

    code:
        f"{get_name(code)}\n{code}"

    for code
    in all_codes
}


# ============================================================
# 20. 边变化图
#
# 共同边、Pearson-only、Partial-only
# ============================================================

plt.figure(
    figsize=(
        14,
        11
    )
)


nx.draw_networkx_nodes(
    G_union,
    pos,

    node_color=
        node_colors(
            G_union
        ),

    node_size=
        1800,

    edgecolors=
        "black"
)


# Common edges
nx.draw_networkx_edges(

    G_union,
    pos,

    edgelist=
        list(
            common_edges
        ),

    width=
        4,

    style=
        "solid",

    alpha=
        0.8
)


# Pearson only
nx.draw_networkx_edges(

    G_union,
    pos,

    edgelist=
        list(
            pearson_only
        ),

    width=
        2,

    style=
        "dashed",

    alpha=
        0.6
)


# Partial only
nx.draw_networkx_edges(

    G_union,
    pos,

    edgelist=
        list(
            partial_only
        ),

    width=
        2,

    style=
        "dotted",

    alpha=
        0.7
)


nx.draw_networkx_labels(
    G_union,
    pos,

    labels=
        labels,

    font_size=
        8
)


edge_legend = [

    Line2D(
        [0],
        [0],
        linestyle=
            "solid",
        label=
            "两个网络均保留"
    ),

    Line2D(
        [0],
        [0],
        linestyle=
            "dashed",
        label=
            "仅Pearson网络"
    ),

    Line2D(
        [0],
        [0],
        linestyle=
            "dotted",
        label=
            "仅偏相关网络"
    )
]


plt.legend(
    handles=
        edge_legend,

    loc=
        "best"
)


plt.title(
    "Pearson阈值网络与偏相关网络的边结构变化"
)


plt.axis(
    "off"
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    /
    "pearson_partial_edge_change_network.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 21. 保存总体比较指标
# ============================================================

overlap_summary = pd.DataFrame(
    [
        {
            "pearson_edges":
                len(
                    E_P
                ),

            "partial_edges":
                len(
                    E_PC
                ),

            "common_edges":
                len(
                    common_edges
                ),

            "pearson_only_edges":
                len(
                    pearson_only
                ),

            "partial_only_edges":
                len(
                    partial_only
                ),

            "jaccard_similarity":
                jaccard,

            "pearson_edge_retention":
                pearson_retention,

            "partial_overlap_ratio":
                partial_overlap_ratio
        }
    ]
)


overlap_summary.to_csv(
    PROCESSED_DIR
    /
    "pearson_partial_overlap_summary.csv",

    index=False,

    encoding=
        "utf-8-sig"
)


# ============================================================
# 22. 完成
# ============================================================

print(
    "\n================================"
)

print(
    "阶段六完成"
)

print(
    "================================"
)


print(
    "\n输出文件："
)

print(
    "pearson_partial_edge_comparison.csv"
)

print(
    "pearson_partial_network_summary.csv"
)

print(
    "pearson_partial_node_comparison.csv"
)

print(
    "pearson_partial_overlap_summary.csv"
)

print(
    "pearson_partial_edge_change_network.png"
)
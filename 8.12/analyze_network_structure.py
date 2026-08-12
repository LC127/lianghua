from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 路径
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


STOCK_INFO_FILE = (
    PROCESSED_DIR /
    "stock_info.csv"
)

THRESHOLD_EDGE_FILE = (
    PROCESSED_DIR /
    "threshold_edges.csv"
)

MST_EDGE_FILE = (
    PROCESSED_DIR /
    "mst_edges.csv"
)


# ============================================================
# 2. 读取股票信息
# ============================================================

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


all_codes = (
    stock_info.index
    .tolist()
)


# ============================================================
# 3. 读取边表
# ============================================================

threshold_edges = pd.read_csv(
    THRESHOLD_EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)

mst_edges = pd.read_csv(
    MST_EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


for df in [
    threshold_edges,
    mst_edges
]:

    df["stock_1"] = (
        df["stock_1"]
        .str.zfill(6)
    )

    df["stock_2"] = (
        df["stock_2"]
        .str.zfill(6)
    )


# ============================================================
# 4. 建立网络函数
# ============================================================

def build_graph(
    edge_df,
    all_nodes,
    stock_info_df
):

    G = nx.Graph()

    # ----------------------------
    # 添加所有股票
    # 包括阈值网络中的孤立股票
    # ----------------------------

    for code in all_nodes:

        name = (
            stock_info_df.loc[
                code,
                "name"
            ]
            if "name"
            in stock_info_df.columns
            else code
        )

        industry = (
            stock_info_df.loc[
                code,
                "industry"
            ]
            if "industry"
            in stock_info_df.columns
            else "未知行业"
        )

        G.add_node(
            code,
            name=name,
            industry=industry
        )


    # ----------------------------
    # 添加边
    # ----------------------------

    for _, row in edge_df.iterrows():

        u = row["stock_1"]
        v = row["stock_2"]

        # 兼容阶段五的列名称
        if "correlation" in row.index:
            correlation = float(
                row["correlation"]
            )

        elif "correlation_weight" in row.index:
            correlation = float(
                row["correlation_weight"]
            )

        else:
            correlation = np.nan


        if "distance" in row.index:
            distance = float(
                row["distance"]
            )

        elif "distance_weight" in row.index:
            distance = float(
                row["distance_weight"]
            )

        else:

            distance = np.sqrt(
                2 *
                (
                    1 -
                    correlation
                )
            )


        G.add_edge(
            u,
            v,
            correlation=correlation,
            distance=distance
        )

    return G


G_threshold = build_graph(
    threshold_edges,
    all_codes,
    stock_info
)

G_mst = build_graph(
    mst_edges,
    all_codes,
    stock_info
)


# ============================================================
# 5. 节点中心性函数
# ============================================================

def calculate_node_metrics(
    G,
    network_name,
    is_mst=False
):

    # ----------------------------
    # Degree
    # ----------------------------

    degree = dict(
        G.degree()
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
    # Weighted Degree
    #
    # 使用相关系数之和
    # ----------------------------

    weighted_degree = {}

    average_neighbor_corr = {}


    for node in G.nodes:

        correlations = [
            data["correlation"]

            for _, _, data
            in G.edges(
                node,
                data=True
            )

            if not pd.isna(
                data["correlation"]
            )
        ]


        weighted_degree[node] = (
            np.sum(
                correlations
            )
            if correlations
            else 0.0
        )


        average_neighbor_corr[node] = (
            np.mean(
                correlations
            )
            if correlations
            else np.nan
        )


    # ----------------------------
    # Betweenness
    #
    # 注意：
    # NetworkX 的 weight 表示“距离”
    # 所以这里必须使用 distance
    # 不能直接使用 correlation
    # ----------------------------

    betweenness = (
        nx.betweenness_centrality(
            G,
            weight="distance",
            normalized=True
        )
    )


    # ----------------------------
    # Closeness
    #
    # 同样使用相关距离
    # ----------------------------

    closeness = (
        nx.closeness_centrality(
            G,
            distance="distance"
        )
    )


    # ----------------------------
    # Connected component
    # ----------------------------

    component_id = {}
    component_size = {}


    for cid, component in enumerate(
        nx.connected_components(G),
        start=1
    ):

        size = len(component)

        for node in component:

            component_id[node] = cid
            component_size[node] = size


    # ----------------------------
    # 输出表
    # ----------------------------

    rows = []

    for node in G.nodes:

        rows.append(
            {
                "network":
                    network_name,

                "code":
                    node,

                "name":
                    G.nodes[node]
                    .get(
                        "name",
                        node
                    ),

                "industry":
                    G.nodes[node]
                    .get(
                        "industry",
                        "未知行业"
                    ),

                "degree":
                    degree[node],

                "degree_centrality":
                    degree_centrality[
                        node
                    ],

                "weighted_degree":
                    weighted_degree[
                        node
                    ],

                "average_neighbor_corr":
                    average_neighbor_corr[
                        node
                    ],

                "betweenness":
                    betweenness[
                        node
                    ],

                "closeness":
                    closeness[
                        node
                    ],

                "component_id":
                    component_id[
                        node
                    ],

                "component_size":
                    component_size[
                        node
                    ],

                "is_leaf":
                    (
                        degree[node] == 1
                        if is_mst
                        else False
                    )
            }
        )


    result = pd.DataFrame(
        rows
    )


    return result


# ============================================================
# 6. 计算阈值网络指标
# ============================================================

threshold_metrics = (
    calculate_node_metrics(
        G_threshold,
        network_name="threshold",
        is_mst=False
    )
)


# ============================================================
# 7. 计算 MST 指标
# ============================================================

mst_metrics = (
    calculate_node_metrics(
        G_mst,
        network_name="mst",
        is_mst=True
    )
)


# ============================================================
# 8. 保存节点指标
# ============================================================

threshold_metrics.to_csv(
    PROCESSED_DIR /
    "threshold_node_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)


mst_metrics.to_csv(
    PROCESSED_DIR /
    "mst_node_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 行业边分析
# ============================================================

def analyze_industry_edges(
    G,
    network_name
):

    rows = []


    for u, v, data in G.edges(
        data=True
    ):

        industry_u = (
            G.nodes[u]
            .get(
                "industry",
                "未知行业"
            )
        )

        industry_v = (
            G.nodes[v]
            .get(
                "industry",
                "未知行业"
            )
        )


        same_industry = (
            industry_u
            ==
            industry_v
        )


        rows.append(
            {
                "network":
                    network_name,

                "stock_1":
                    u,

                "industry_1":
                    industry_u,

                "stock_2":
                    v,

                "industry_2":
                    industry_v,

                "same_industry":
                    same_industry,

                "correlation":
                    data["correlation"],

                "distance":
                    data["distance"]
            }
        )


    return pd.DataFrame(
        rows
    )


threshold_industry_edges = (
    analyze_industry_edges(
        G_threshold,
        "threshold"
    )
)


mst_industry_edges = (
    analyze_industry_edges(
        G_mst,
        "mst"
    )
)


threshold_industry_edges.to_csv(
    PROCESSED_DIR /
    "threshold_industry_edges.csv",
    index=False,
    encoding="utf-8-sig"
)


mst_industry_edges.to_csv(
    PROCESSED_DIR /
    "mst_industry_edges.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 行业聚集指标
# ============================================================

def calculate_industry_summary(
    G,
    edge_df,
    network_name
):

    if len(edge_df) > 0:

        same_industry_ratio = (
            edge_df[
                "same_industry"
            ]
            .mean()
        )

    else:

        same_industry_ratio = np.nan


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
            nx.density(G),

        "n_components":
            nx.number_connected_components(
                G
            ),

        "same_industry_edge_ratio":
            same_industry_ratio,

        "industry_assortativity":
            assortativity
    }


industry_summary = pd.DataFrame(
    [
        calculate_industry_summary(
            G_threshold,
            threshold_industry_edges,
            "threshold"
        ),

        calculate_industry_summary(
            G_mst,
            mst_industry_edges,
            "mst"
        )
    ]
)


industry_summary.to_csv(
    PROCESSED_DIR /
    "industry_structure_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 查看核心股票
# ============================================================

def print_top_stocks(
    metrics_df,
    network_name
):

    print(
        "\n================================"
    )

    print(
        f"{network_name} 网络"
    )

    print(
        "================================"
    )


    for metric in [
        "degree",
        "weighted_degree",
        "betweenness",
        "closeness"
    ]:

        print(
            f"\n{metric} 前5名："
        )

        temp = (
            metrics_df
            .sort_values(
                metric,
                ascending=False
            )
            [
                [
                    "code",
                    "name",
                    "industry",
                    metric
                ]
            ]
            .head(5)
        )

        print(
            temp.to_string(
                index=False
            )
        )


print_top_stocks(
    threshold_metrics,
    "Threshold"
)

print_top_stocks(
    mst_metrics,
    "MST"
)


# ============================================================
# 12. 行业聚集结果
# ============================================================

print(
    "\n================================"
)

print(
    "行业结构分析"
)

print(
    "================================"
)

print(
    industry_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 13. MST 叶节点
# ============================================================

mst_leaf_nodes = (
    mst_metrics[
        mst_metrics[
            "is_leaf"
        ]
    ]
    [
        [
            "code",
            "name",
            "industry",
            "degree",
            "weighted_degree",
            "betweenness"
        ]
    ]
)


print(
    "\nMST 叶节点："
)

print(
    mst_leaf_nodes
    .to_string(
        index=False
    )
)


# ============================================================
# 14. 行业颜色
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


def get_node_colors(G):

    return [
        industry_color_map.get(
            G.nodes[node]
            .get(
                "industry",
                "未知行业"
            ),
            "gray"
        )

        for node in G.nodes
    ]


def build_legend():

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=industry,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=9
        )

        for industry, color
        in industry_color_map.items()
    ]


# ============================================================
# 15. 绘制阈值网络中心性图
# ============================================================

threshold_metric_index = (
    threshold_metrics
    .set_index("code")
)


# 节点大小：
# Weighted Degree 越大，节点越大
weighted_values = np.array(
    [
        threshold_metric_index.loc[
            node,
            "weighted_degree"
        ]

        for node
        in G_threshold.nodes
    ]
)


if weighted_values.max() > weighted_values.min():

    threshold_node_sizes = (
        900
        +
        3500
        *
        (
            weighted_values
            -
            weighted_values.min()
        )
        /
        (
            weighted_values.max()
            -
            weighted_values.min()
        )
    )

else:

    threshold_node_sizes = (
        np.repeat(
            1500,
            len(weighted_values)
        )
    )


pos_threshold = (
    nx.spring_layout(
        G_threshold,
        seed=42,
        weight="correlation"
    )
)


plt.figure(
    figsize=(14, 11)
)


edge_widths = [
    0.8
    +
    8
    *
    data["correlation"]

    for _, _, data
    in G_threshold.edges(
        data=True
    )
]


nx.draw_networkx_nodes(
    G_threshold,
    pos_threshold,
    node_color=get_node_colors(
        G_threshold
    ),
    node_size=threshold_node_sizes,
    edgecolors="black"
)


nx.draw_networkx_edges(
    G_threshold,
    pos_threshold,
    width=edge_widths,
    alpha=0.7
)


labels = {
    node:
        f"{G_threshold.nodes[node]['name']}\n{node}"

    for node
    in G_threshold.nodes
}


nx.draw_networkx_labels(
    G_threshold,
    pos_threshold,
    labels=labels,
    font_size=8
)


plt.legend(
    handles=build_legend(),
    title="行业",
    loc="best"
)


plt.title(
    "阈值相关网络："
    "节点大小 = Weighted Degree"
)

plt.axis("off")

plt.tight_layout()


plt.savefig(
    FIGURE_DIR /
    "threshold_centrality_network.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. MST 距离保持布局
# ============================================================

def weighted_tree_layout(
    G,
    root=None,
    distance_attr="distance",
    scale=5.0
):

    if root is None:

        root = max(
            G.degree,
            key=lambda x: x[1]
        )[0]


    pos = {
        root:
            np.array(
                [0.0, 0.0]
            )
    }


    parent = {
        root:
            None
    }


    children = {
        node:
            []

        for node
        in G.nodes
    }


    queue = [root]


    while queue:

        node = queue.pop(0)


        for nbr in G.neighbors(
            node
        ):

            if nbr == parent[node]:
                continue

            if nbr in parent:
                continue


            parent[nbr] = node

            children[node].append(
                nbr
            )

            queue.append(
                nbr
            )


    def subtree_size(node):

        if not children[node]:

            return 1


        return sum(
            subtree_size(child)

            for child
            in children[node]
        )


    sizes = {
        node:
            subtree_size(
                node
            )

        for node
        in G.nodes
    }


    def place_children(
        node,
        angle_start,
        angle_end
    ):

        child_list = (
            children[node]
        )

        if not child_list:
            return


        total_size = sum(
            sizes[child]

            for child
            in child_list
        )


        current_angle = (
            angle_start
        )


        for child in child_list:

            fraction = (
                sizes[child]
                /
                total_size
            )


            child_angle_end = (
                current_angle
                +
                fraction
                *
                (
                    angle_end
                    -
                    angle_start
                )
            )


            theta = (
                current_angle
                +
                child_angle_end
            ) / 2


            distance = (
                G[node][child][
                    distance_attr
                ]
            )


            length = (
                scale
                *
                distance
            )


            pos[child] = (
                pos[node]
                +
                length
                *
                np.array(
                    [
                        np.cos(theta),
                        np.sin(theta)
                    ]
                )
            )


            place_children(
                child,
                current_angle,
                child_angle_end
            )


            current_angle = (
                child_angle_end
            )


    place_children(
        root,
        0,
        2 * np.pi
    )


    return pos


# ============================================================
# 17. 绘制 MST 中心性图
# ============================================================

mst_metric_index = (
    mst_metrics
    .set_index("code")
)


# 节点大小：
# Betweenness 越大，节点越大
between_values = np.array(
    [
        mst_metric_index.loc[
            node,
            "betweenness"
        ]

        for node
        in G_mst.nodes
    ]
)


if between_values.max() > between_values.min():

    mst_node_sizes = (
        900
        +
        4000
        *
        (
            between_values
            -
            between_values.min()
        )
        /
        (
            between_values.max()
            -
            between_values.min()
        )
    )

else:

    mst_node_sizes = (
        np.repeat(
            1500,
            len(between_values)
        )
    )


pos_mst = (
    weighted_tree_layout(
        G_mst,
        distance_attr="distance",
        scale=5
    )
)


mst_edge_widths = [
    0.8
    +
    8
    *
    data["correlation"]

    for _, _, data
    in G_mst.edges(
        data=True
    )
]


plt.figure(
    figsize=(14, 11)
)


nx.draw_networkx_nodes(
    G_mst,
    pos_mst,
    node_color=get_node_colors(
        G_mst
    ),
    node_size=mst_node_sizes,
    edgecolors="black"
)


nx.draw_networkx_edges(
    G_mst,
    pos_mst,
    width=mst_edge_widths,
    alpha=0.75
)


mst_labels = {
    node:
        f"{G_mst.nodes[node]['name']}\n{node}"

    for node
    in G_mst.nodes
}


nx.draw_networkx_labels(
    G_mst,
    pos_mst,
    labels=mst_labels,
    font_size=8
)


plt.legend(
    handles=build_legend(),
    title="行业",
    loc="best"
)


plt.title(
    "MST："
    "节点大小 = Betweenness Centrality"
)


plt.axis("off")

plt.tight_layout()


plt.savefig(
    FIGURE_DIR /
    "mst_centrality_network.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. 完成
# ============================================================

print(
    "\n阶段六分析完成。"
)

print(
    "\n主要输出文件："
)

print(
    "threshold_node_metrics.csv"
)

print(
    "mst_node_metrics.csv"
)

print(
    "threshold_industry_edges.csv"
)

print(
    "mst_industry_edges.csv"
)

print(
    "industry_structure_summary.csv"
)

print(
    "threshold_centrality_network.png"
)

print(
    "mst_centrality_network.png"
)
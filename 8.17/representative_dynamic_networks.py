from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# 0. 中文字体
# ============================================================

def set_chinese_font():

    candidates = [
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

    for font_name in candidates:

        if font_name in installed_fonts:

            plt.rcParams["font.sans-serif"] = [
                font_name
            ]

            plt.rcParams["axes.unicode_minus"] = False

            print(
                f"使用中文字体：{font_name}"
            )

            return

    print(
        "未找到常见中文字体，中文可能显示异常。"
    )


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


EDGE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

NODE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_node_metrics.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_network_summary.csv"
)


# ============================================================
# 2. 选择代表性窗口
# ============================================================

# 22：结构变化前
# 23：最大单步边数下降后的网络
# 26：后续较稀疏状态

TARGET_WINDOWS = [
    22,
    23,
    26
]


# ============================================================
# 3. 读取数据
# ============================================================

edge_df = pd.read_csv(
    EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


node_df = pd.read_csv(
    NODE_FILE,
    dtype={
        "code": str
    }
)


summary_df = pd.read_csv(
    SUMMARY_FILE
)


# ------------------------------------------------------------
# 股票代码补齐为6位
# ------------------------------------------------------------

edge_df["stock_1"] = (
    edge_df["stock_1"]
    .astype(str)
    .str
    .zfill(6)
)

edge_df["stock_2"] = (
    edge_df["stock_2"]
    .astype(str)
    .str
    .zfill(6)
)

node_df["code"] = (
    node_df["code"]
    .astype(str)
    .str
    .zfill(6)
)


# ------------------------------------------------------------
# 日期
# ------------------------------------------------------------

for col in [
    "window_start",
    "window_end",
    "network_date"
]:

    if col in edge_df.columns:

        edge_df[col] = pd.to_datetime(
            edge_df[col]
        )

    if col in node_df.columns:

        node_df[col] = pd.to_datetime(
            node_df[col]
        )

    if col in summary_df.columns:

        summary_df[col] = pd.to_datetime(
            summary_df[col]
        )


# ============================================================
# 4. 股票名称和行业
# ============================================================

node_metadata = (
    node_df[
        [
            "code",
            "name",
            "industry"
        ]
    ]
    .drop_duplicates(
        subset=[
            "code"
        ]
    )
)


codes = (
    node_metadata[
        "code"
    ]
    .tolist()
)


name_map = dict(
    zip(
        node_metadata[
            "code"
        ],
        node_metadata[
            "name"
        ]
    )
)


industry_map = dict(
    zip(
        node_metadata[
            "code"
        ],
        node_metadata[
            "industry"
        ]
    )
)


# ============================================================
# 5. 行业颜色
# ============================================================

industries = sorted(
    node_metadata[
        "industry"
    ]
    .dropna()
    .unique()
)


cmap = plt.cm.tab20


industry_color_map = {

    industry:
        cmap(
            i
            /
            max(
                len(industries),
                1
            )
        )

    for i, industry
    in enumerate(
        industries
    )
}


# ============================================================
# 6. 构建每个窗口的 Graph
# ============================================================

def build_graph(
    window_id
):

    G = nx.Graph()


    # --------------------------------------------------------
    # 节点
    # --------------------------------------------------------

    nodes_current = (
        node_df[
            node_df[
                "window_id"
            ]
            ==
            window_id
        ]
    )


    for row in nodes_current.itertuples():

        G.add_node(

            row.code,

            name=
                row.name,

            industry=
                row.industry,

            degree_value=
                row.degree,

            strength_value=
                row.strength
        )


    # --------------------------------------------------------
    # 当前窗口被选择的边
    # --------------------------------------------------------

    edges_current = (
        edge_df[
            (
                edge_df[
                    "window_id"
                ]
                ==
                window_id
            )
            &
            (
                edge_df[
                    "selected"
                ]
                ==
                True
            )
        ]
    )


    for row in edges_current.itertuples():

        G.add_edge(

            row.stock_1,
            row.stock_2,

            partial_correlation=
                row.partial_correlation,

            abs_partial=
                row.abs_partial_correlation,

            same_industry=
                row.same_industry
        )


    return G


graphs = {

    window_id:
        build_graph(
            window_id
        )

    for window_id
    in TARGET_WINDOWS
}


# ============================================================
# 7. 构建三个网络的边并集
#
# 目的：
# 用同一张 union graph 生成一次 layout，
# 然后三个网络共享同一组节点位置。
# ============================================================

G_union = nx.Graph()


for code in codes:

    G_union.add_node(
        code
    )


for window_id in TARGET_WINDOWS:

    G = graphs[
        window_id
    ]


    for u, v, data in G.edges(
        data=True
    ):

        # 用平均绝对partial作为layout weight的基础
        weight = data[
            "abs_partial"
        ]


        if G_union.has_edge(
            u,
            v
        ):

            G_union[
                u
            ][
                v
            ][
                "weight"
            ] += weight

            G_union[
                u
            ][
                v
            ][
                "count"
            ] += 1

        else:

            G_union.add_edge(

                u,
                v,

                weight=
                    weight,

                count=
                    1
            )


# ------------------------------------------------------------
# 对重复出现的边使用平均权重
# ------------------------------------------------------------

for u, v, data in G_union.edges(
    data=True
):

    data[
        "weight"
    ] = (
        data[
            "weight"
        ]
        /
        data[
            "count"
        ]
    )


# ============================================================
# 8. 固定Layout
# ============================================================

pos = nx.spring_layout(

    G_union,

    weight=
        "weight",

    seed=
        20260817,

    k=
        1.0,

    iterations=
        500
)


# ============================================================
# 9. 找三个窗口都存在的Persistent Edges
# ============================================================

def edge_set(
    G
):

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


edge_sets = {

    window_id:
        edge_set(
            graphs[
                window_id
            ]
        )

    for window_id
    in TARGET_WINDOWS
}


persistent_edges = set.intersection(
    *[
        edge_sets[
            window_id
        ]

        for window_id
        in TARGET_WINDOWS
    ]
)


print(
    "三个代表窗口均存在的边数：",
    len(
        persistent_edges
    )
)


# ============================================================
# 10. 统一Edge Width尺度
# ============================================================

all_abs_partial = []


for window_id in TARGET_WINDOWS:

    G = graphs[
        window_id
    ]


    for _, _, data in G.edges(
        data=True
    ):

        all_abs_partial.append(
            data[
                "abs_partial"
            ]
        )


GLOBAL_MAX_PARTIAL = max(
    all_abs_partial
)


def edge_width(
    abs_partial
):

    # 所有窗口统一映射
    return (
        0.6
        +
        5.0
        *
        abs_partial
        /
        GLOBAL_MAX_PARTIAL
    )


# ============================================================
# 11. 绘制单个网络
# ============================================================

def draw_one_network(
    ax,
    window_id
):

    G = graphs[
        window_id
    ]


    summary_row = (
        summary_df[
            summary_df[
                "window_id"
            ]
            ==
            window_id
        ]
        .iloc[0]
    )


    # --------------------------------------------------------
    # 节点颜色：行业
    # --------------------------------------------------------

    node_colors = [

        industry_color_map.get(
            G.nodes[
                code
            ][
                "industry"
            ],
            "lightgray"
        )

        for code
        in G.nodes()
    ]


    # --------------------------------------------------------
    # 节点大小：Degree
    #
    # 所有窗口采用同一尺度
    # --------------------------------------------------------

    node_sizes = [

        650
        +
        90
        *
        G.nodes[
            code
        ][
            "degree_value"
        ]

        for code
        in G.nodes()
    ]


    # --------------------------------------------------------
    # Persistent / non-persistent edges
    # --------------------------------------------------------

    persistent_positive = []

    persistent_negative = []

    dynamic_positive = []

    dynamic_negative = []


    for u, v, data in G.edges(
        data=True
    ):

        key = tuple(
            sorted(
                [
                    u,
                    v
                ]
            )
        )


        rho = data[
            "partial_correlation"
        ]


        if key in persistent_edges:

            if rho >= 0:

                persistent_positive.append(
                    (
                        u,
                        v
                    )
                )

            else:

                persistent_negative.append(
                    (
                        u,
                        v
                    )
                )

        else:

            if rho >= 0:

                dynamic_positive.append(
                    (
                        u,
                        v
                    )
                )

            else:

                dynamic_negative.append(
                    (
                        u,
                        v
                    )
                )


    # --------------------------------------------------------
    # 获取宽度
    # --------------------------------------------------------

    def widths_for(
        edges
    ):

        return [

            edge_width(
                G[
                    u
                ][
                    v
                ][
                    "abs_partial"
                ]
            )

            for u, v
            in edges
        ]


    # --------------------------------------------------------
    # 先画非持续边，较浅
    # --------------------------------------------------------

    nx.draw_networkx_edges(

        G,
        pos,

        edgelist=
            dynamic_positive,

        width=
            widths_for(
                dynamic_positive
            ),

        edge_color=
            "lightgray",

        alpha=
            0.65,

        style=
            "solid",

        ax=
            ax
    )


    nx.draw_networkx_edges(

        G,
        pos,

        edgelist=
            dynamic_negative,

        width=
            widths_for(
                dynamic_negative
            ),

        edge_color=
            "lightgray",

        alpha=
            0.65,

        style=
            "dashed",

        ax=
            ax
    )


    # --------------------------------------------------------
    # 再画三个窗口都存在的Persistent edges
    # --------------------------------------------------------

    nx.draw_networkx_edges(

        G,
        pos,

        edgelist=
            persistent_positive,

        width=
            widths_for(
                persistent_positive
            ),

        edge_color=
            "dimgray",

        alpha=
            0.90,

        style=
            "solid",

        ax=
            ax
    )


    nx.draw_networkx_edges(

        G,
        pos,

        edgelist=
            persistent_negative,

        width=
            widths_for(
                persistent_negative
            ),

        edge_color=
            "dimgray",

        alpha=
            0.90,

        style=
            "dashed",

        ax=
            ax
    )


    # --------------------------------------------------------
    # 节点
    # --------------------------------------------------------

    nx.draw_networkx_nodes(

        G,
        pos,

        node_color=
            node_colors,

        node_size=
            node_sizes,

        edgecolors=
            "black",

        linewidths=
            0.8,

        alpha=
            0.95,

        ax=
            ax
    )


    # --------------------------------------------------------
    # 标签
    # --------------------------------------------------------

    labels = {

        code:
            f"{code}\n{name_map[code]}"

        for code
        in G.nodes()
    }


    nx.draw_networkx_labels(

        G,
        pos,

        labels=
            labels,

        font_size=
            7,

        ax=
            ax
    )


    # --------------------------------------------------------
    # 标题
    # --------------------------------------------------------

    network_date = pd.to_datetime(
        summary_row[
            "network_date"
        ]
    ).strftime(
        "%Y-%m-%d"
    )


    title = (

        f"Window {window_id}: {network_date}\n"

        f"Edges={int(summary_row['n_edges'])}, "
        f"Density={summary_row['density']:.3f}, "
        f"Mean |partial|={summary_row['mean_abs_partial']:.3f}\n"

        f"Same-industry={summary_row['same_industry_edge_ratio']:.1%}"
    )


    ax.set_title(
        title,
        fontsize=
            11
    )


    ax.set_axis_off()


# ============================================================
# 12. 三个代表性动态网络放在同一张图
# ============================================================

fig, axes = plt.subplots(

    1,
    3,

    figsize=(
        24,
        8
    )
)


for ax, window_id in zip(
    axes,
    TARGET_WINDOWS
):

    draw_one_network(
        ax,
        window_id
    )


# ============================================================
# 13. 图例
# ============================================================

industry_handles = [

    Patch(

        facecolor=
            industry_color_map[
                industry
            ],

        edgecolor=
            "black",

        label=
            industry
    )

    for industry
    in industries
]


edge_handles = [

    Line2D(
        [0],
        [0],

        color=
            "dimgray",

        lw=
            2.5,

        linestyle=
            "solid",

        label=
            "三个窗口均存在的正向边"
    ),

    Line2D(
        [0],
        [0],

        color=
            "lightgray",

        lw=
            2.5,

        linestyle=
            "solid",

        label=
            "非持续正向边"
    ),

    Line2D(
        [0],
        [0],

        color=
            "dimgray",

        lw=
            2.5,

        linestyle=
            "dashed",

        label=
            "负向条件关联"
    )
]


fig.legend(

    handles=
        industry_handles
        +
        edge_handles,

    loc=
        "lower center",

    ncol=
        min(
            8,
            len(
                industry_handles
                +
                edge_handles
            )
        ),

    frameon=
        False,

    fontsize=
        9
)


fig.suptitle(

    "Rolling Graphical Lasso代表性动态网络比较",

    fontsize=
        16,

    y=
        0.98
)


plt.tight_layout(

    rect=[
        0,
        0.10,
        1,
        0.94
    ]
)


OUTPUT_FIGURE = (
    FIGURE_DIR
    / "rolling_glasso_representative_network_comparison.png"
)


plt.savefig(

    OUTPUT_FIGURE,

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


print(
    "\n网络比较图已保存："
)

print(
    OUTPUT_FIGURE
)


# ============================================================
# 14. 计算相邻代表窗口Edge Changes
# ============================================================

comparison_rows = []


comparison_pairs = [
    (
        22,
        23
    ),
    (
        23,
        26
    )
]


for window_from, window_to in comparison_pairs:

    E_from = edge_sets[
        window_from
    ]

    E_to = edge_sets[
        window_to
    ]


    common = (
        E_from
        &
        E_to
    )


    lost = (
        E_from
        -
        E_to
    )


    gained = (
        E_to
        -
        E_from
    )


    union = (
        E_from
        |
        E_to
    )


    jaccard = (
        len(
            common
        )
        /
        len(
            union
        )
    )


    comparison_rows.append(
        {
            "window_from":
                window_from,

            "window_to":
                window_to,

            "edges_from":
                len(
                    E_from
                ),

            "edges_to":
                len(
                    E_to
                ),

            "common_edges":
                len(
                    common
                ),

            "lost_edges":
                len(
                    lost
                ),

            "gained_edges":
                len(
                    gained
                ),

            "jaccard":
                jaccard
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


COMPARISON_FILE = (
    PROCESSED_DIR
    / "representative_window_network_comparison.csv"
)


comparison_df.to_csv(

    COMPARISON_FILE,

    index=False,

    encoding=
        "utf-8-sig"
)


print(
    "\n代表窗口网络比较："
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 15. 输出具体Lost / Gained Edges
# ============================================================

change_rows = []


for window_from, window_to in comparison_pairs:

    E_from = edge_sets[
        window_from
    ]

    E_to = edge_sets[
        window_to
    ]


    lost = (
        E_from
        -
        E_to
    )


    gained = (
        E_to
        -
        E_from
    )


    for u, v in sorted(
        lost
    ):

        change_rows.append(
            {
                "window_from":
                    window_from,

                "window_to":
                    window_to,

                "stock_1":
                    u,

                "name_1":
                    name_map[
                        u
                    ],

                "stock_2":
                    v,

                "name_2":
                    name_map[
                        v
                    ],

                "change":
                    "lost"
            }
        )


    for u, v in sorted(
        gained
    ):

        change_rows.append(
            {
                "window_from":
                    window_from,

                "window_to":
                    window_to,

                "stock_1":
                    u,

                "name_1":
                    name_map[
                        u
                    ],

                "stock_2":
                    v,

                "name_2":
                    name_map[
                        v
                    ],

                "change":
                    "gained"
            }
        )


change_df = pd.DataFrame(
    change_rows
)


CHANGE_FILE = (
    PROCESSED_DIR
    / "representative_window_edge_changes.csv"
)


change_df.to_csv(

    CHANGE_FILE,

    index=False,

    encoding=
        "utf-8-sig"
)


print(
    "\n具体Edge Changes："
)

print(
    change_df.to_string(
        index=False
    )
)
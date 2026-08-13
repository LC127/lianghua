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
        font.name
        for font in font_manager.fontManager.ttflist
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
        "警告：没有找到常见中文字体。"
    )


set_chinese_font()


# ============================================================
# 1. 路径设置
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


PARTIAL_FILE = (
    PROCESSED_DIR
    / "partial_correlation.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 2. 参数设置
# ============================================================

# 初步工作阈值
PARTIAL_THRESHOLD = 0.20

# 节点大小
NODE_SIZE = 1800

# 标签字体
NODE_LABEL_SIZE = 8
EDGE_LABEL_SIZE = 7


# ============================================================
# 3. 读取偏相关矩阵
# ============================================================

partial = pd.read_csv(
    PARTIAL_FILE,
    index_col=0
)


partial.index = (
    partial.index
    .astype(str)
    .str.zfill(6)
)

partial.columns = (
    partial.columns
    .astype(str)
    .str.zfill(6)
)


partial = partial.astype(
    float
)


print(
    "偏相关矩阵维度：",
    partial.shape
)


# ============================================================
# 4. 基本检查
# ============================================================

if (
    partial.shape[0]
    !=
    partial.shape[1]
):

    raise ValueError(
        "偏相关矩阵不是方阵。"
    )


if not np.allclose(
    partial.values,
    partial.values.T,
    atol=1e-10
):

    raise ValueError(
        "偏相关矩阵不是对称矩阵。"
    )


print(
    "偏相关矩阵对称性检查：通过"
)


# ============================================================
# 5. 读取股票信息
# ============================================================

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

    stock_info = (
        stock_info
        .set_index(
            "code"
        )
    )

else:

    stock_info = pd.DataFrame(
        index=partial.index
    )


codes = (
    partial.index
    .tolist()
)


def get_name(code):

    if (
        code
        in stock_info.index
        and
        "name"
        in stock_info.columns
    ):

        return stock_info.loc[
            code,
            "name"
        ]

    return code


def get_industry(code):

    if (
        code
        in stock_info.index
        and
        "industry"
        in stock_info.columns
    ):

        return stock_info.loc[
            code,
            "industry"
        ]

    return "未知行业"


# ============================================================
# 6. 提取全部不同股票对
# ============================================================

pair_rows = []


for i in range(
    len(codes)
):

    for j in range(
        i + 1,
        len(codes)
    ):

        code_i = codes[i]
        code_j = codes[j]

        rho = partial.loc[
            code_i,
            code_j
        ]

        pair_rows.append(
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

                "partial_correlation":
                    rho,

                "abs_partial_correlation":
                    abs(rho)
            }
        )


partial_pairs = pd.DataFrame(
    pair_rows
)


print(
    "\n股票对数量：",
    len(
        partial_pairs
    )
)


# ============================================================
# 7. 偏相关描述统计
# ============================================================

print(
    "\n偏相关系数描述统计："
)

print(
    partial_pairs[
        "partial_correlation"
    ]
    .describe()
)


print(
    "\n绝对偏相关描述统计："
)

print(
    partial_pairs[
        "abs_partial_correlation"
    ]
    .describe()
)


# ============================================================
# 8. 阈值敏感性分析
# ============================================================

threshold_list = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]


sensitivity_rows = []


for threshold in threshold_list:

    n_edges = (
        partial_pairs[
            "abs_partial_correlation"
        ]
        >= threshold
    ).sum()


    sensitivity_rows.append(
        {
            "threshold":
                threshold,

            "n_edges":
                int(
                    n_edges
                ),

            "density":
                n_edges
                /
                (
                    len(codes)
                    *
                    (
                        len(codes)
                        -
                        1
                    )
                    /
                    2
                )
        }
    )


threshold_sensitivity = (
    pd.DataFrame(
        sensitivity_rows
    )
)


threshold_sensitivity.to_csv(
    PROCESSED_DIR
    / "partial_threshold_sensitivity.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n不同偏相关阈值下的网络："
)

print(
    threshold_sensitivity
    .to_string(
        index=False
    )
)


# ============================================================
# 9. 构建偏相关网络
# ============================================================

G_partial = nx.Graph()


# ------------------------------------------------------------
# 9.1 添加所有股票节点
# ------------------------------------------------------------

for code in codes:

    G_partial.add_node(
        code,

        name=get_name(
            code
        ),

        industry=get_industry(
            code
        )
    )


# ------------------------------------------------------------
# 9.2 根据绝对偏相关添加边
# ------------------------------------------------------------

for _, row in (
    partial_pairs
    .iterrows()
):

    rho = row[
        "partial_correlation"
    ]


    if (
        abs(rho)
        >=
        PARTIAL_THRESHOLD
    ):

        G_partial.add_edge(

            row[
                "stock_1"
            ],

            row[
                "stock_2"
            ],

            # 带符号的偏相关
            partial_correlation=float(
                rho
            ),

            # 关联强度
            abs_partial=float(
                abs(rho)
            )
        )


# ============================================================
# 10. 网络基本统计
# ============================================================

n_nodes = (
    G_partial
    .number_of_nodes()
)

n_edges = (
    G_partial
    .number_of_edges()
)

density = nx.density(
    G_partial
)

n_components = (
    nx.number_connected_components(
        G_partial
    )
)

n_isolated = len(
    list(
        nx.isolates(
            G_partial
        )
    )
)


print(
    "\n================================"
)

print(
    "偏相关网络基本统计"
)

print(
    "================================"
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
    "连通分量：",
    n_components
)

print(
    "孤立节点：",
    n_isolated
)


# ============================================================
# 11. 保存偏相关网络边表
# ============================================================

edge_rows = []


for u, v, data in (
    G_partial.edges(
        data=True
    )
):

    edge_rows.append(
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

            "partial_correlation":
                data[
                    "partial_correlation"
                ],

            "abs_partial_correlation":
                data[
                    "abs_partial"
                ],

            "sign":
                (
                    "positive"
                    if
                    data[
                        "partial_correlation"
                    ]
                    > 0

                    else
                    "negative"
                )
        }
    )


partial_edges = pd.DataFrame(
    edge_rows
)


if not partial_edges.empty:

    partial_edges = (
        partial_edges
        .sort_values(
            "abs_partial_correlation",
            ascending=False
        )
    )


partial_edges.to_csv(
    PROCESSED_DIR
    / "partial_network_edges.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 保存网络汇总
# ============================================================

network_summary = pd.DataFrame(
    [
        {
            "network":
                "partial_correlation",

            "threshold":
                PARTIAL_THRESHOLD,

            "n_nodes":
                n_nodes,

            "n_edges":
                n_edges,

            "density":
                density,

            "n_components":
                n_components,

            "n_isolated":
                n_isolated
        }
    ]
)


network_summary.to_csv(
    PROCESSED_DIR
    / "partial_network_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 行业颜色设置
# ============================================================

industries = sorted(
    {
        get_industry(
            code
        )
        for code
        in codes
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

    for code
    in G_partial.nodes
]


# ============================================================
# 14. 节点标签
# ============================================================

labels = {

    code:
        f"{get_name(code)}\n{code}"

    for code
    in G_partial.nodes
}


# ============================================================
# 15. 网络布局
# ============================================================

pos = nx.spring_layout(
    G_partial,
    seed=42,

    # 这里使用绝对偏相关作为布局权重
    weight="abs_partial"
)


# ============================================================
# 16. 边宽设置
#
# 使用 min-max scaling 增强差异
# ============================================================

edge_strengths = np.array(
    [
        data[
            "abs_partial"
        ]

        for _, _, data
        in G_partial.edges(
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
# 17. 绘制偏相关网络
# ============================================================

plt.figure(
    figsize=(
        14,
        11
    )
)


nx.draw_networkx_nodes(
    G_partial,
    pos,

    node_color=
        node_colors,

    node_size=
        NODE_SIZE,

    edgecolors=
        "black",

    linewidths=
        0.8
)


# ------------------------------------------------------------
# 正、负偏相关分别绘制
# ------------------------------------------------------------

edge_list = list(
    G_partial.edges(
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
        else "dashed"
    )


    nx.draw_networkx_edges(

        G_partial,

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


# ------------------------------------------------------------
# 节点标签
# ------------------------------------------------------------

nx.draw_networkx_labels(
    G_partial,
    pos,

    labels=
        labels,

    font_size=
        NODE_LABEL_SIZE
)


# ------------------------------------------------------------
# 边标签：
# 直接显示带符号偏相关系数
# ------------------------------------------------------------

edge_labels = {

    (
        u,
        v
    ):
        f"{data['partial_correlation']:.2f}"

    for u, v, data
    in G_partial.edges(
        data=True
    )
}


nx.draw_networkx_edge_labels(
    G_partial,
    pos,

    edge_labels=
        edge_labels,

    font_size=
        EDGE_LABEL_SIZE
)


# ============================================================
# 18. 行业图例
# ============================================================

legend_handles = [

    Line2D(
        [0],
        [0],

        marker=
            "o",

        linestyle=
            "",

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


plt.legend(
    handles=
        legend_handles,

    title=
        "行业",

    loc=
        "best"
)


# ============================================================
# 19. 标题和保存
# ============================================================

plt.title(
    f"偏相关网络 "
    f"(|partial correlation| >= {PARTIAL_THRESHOLD})\n"
    "边越粗表示条件关联越强；"
    "实线为正偏相关，虚线为负偏相关"
)


plt.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "partial_correlation_network.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 20. 输出最强偏相关边
# ============================================================

print(
    "\n绝对偏相关最高的股票关系："
)


if not partial_edges.empty:

    print(
        partial_edges[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "partial_correlation",
                "abs_partial_correlation"
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# 21. 完成
# ============================================================

print(
    "\n================================"
)

print(
    "阶段五完成"
)

print(
    "================================"
)


print(
    "\n输出文件："
)

print(
    PROCESSED_DIR
    / "partial_threshold_sensitivity.csv"
)

print(
    PROCESSED_DIR
    / "partial_network_edges.csv"
)

print(
    PROCESSED_DIR
    / "partial_network_summary.csv"
)

print(
    FIGURE_DIR
    / "partial_correlation_network.png"
)
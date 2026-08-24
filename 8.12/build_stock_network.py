from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


# ============================================================
# 0. 中文字体设置
# ============================================================

def set_chinese_font():
    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Arial Unicode MS"
    ]

    installed_fonts = {
        font.name
        for font in font_manager.fontManager.ttflist
    }

    for font_name in candidate_fonts:
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"使用中文字体：{font_name}")
            return

    print("警告：未找到常见中文字体，中文可能显示异常。")


set_chinese_font()


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_DIR = Path("stock_network")

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR = PROJECT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RETURN_FILE = PROCESSED_DIR / "stock_returns.csv"
STOCK_INFO_FILE = PROCESSED_DIR / "stock_info.csv"


# ============================================================
# 2. 参数设置
# ============================================================

CORR_THRESHOLD = 0.50
USE_ABSOLUTE_THRESHOLD = False

# MST 图中长度缩放倍数
MST_LENGTH_SCALE = 5.0

# 节点大小
NODE_SIZE = 1800

# 标签字号
LABEL_SIZE = 8
EDGE_LABEL_SIZE = 7


# ============================================================
# 3. 读取收益率
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

returns = returns.sort_index()

print("收益率矩阵维度：", returns.shape)


# ============================================================
# 4. 读取股票信息
# ============================================================

if STOCK_INFO_FILE.exists():

    stock_info = pd.read_csv(
        STOCK_INFO_FILE,
        dtype={"code": str}
    )

    stock_info["code"] = stock_info["code"].str.zfill(6)
    stock_info = stock_info.set_index("code")

else:
    stock_info = pd.DataFrame(index=returns.columns)

print("股票信息表维度：", stock_info.shape)


# ============================================================
# 5. 计算相关矩阵
# ============================================================

corr = returns.corr(method="pearson")

print("相关矩阵维度：", corr.shape)


# ============================================================
# 6. 相关系数描述统计
# ============================================================

upper_mask = np.triu(
    np.ones(corr.shape, dtype=bool),
    k=1
)

pair_corr = corr.where(upper_mask).stack()

print("\n股票对相关系数描述统计：")
print(pair_corr.describe())


# ============================================================
# 7. 阈值敏感性分析
# ============================================================

threshold_list = [0.30, 0.40, 0.50, 0.60]
sensitivity_rows = []

for threshold in threshold_list:

    if USE_ABSOLUTE_THRESHOLD:
        n_edges = (pair_corr.abs() >= threshold).sum()
    else:
        n_edges = (pair_corr >= threshold).sum()

    sensitivity_rows.append({
        "threshold": threshold,
        "n_edges": int(n_edges)
    })

threshold_sensitivity = pd.DataFrame(sensitivity_rows)

threshold_sensitivity.to_csv(
    PROCESSED_DIR / "threshold_sensitivity.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n不同阈值对应边数：")
print(threshold_sensitivity.to_string(index=False))


# ============================================================
# 8. 行业颜色映射
# ============================================================

def build_industry_color_map(stock_info_df, node_list):
    """
    为行业生成颜色映射，并返回：
    1. industry_color_map
    2. node_colors
    """

    # 找出节点中实际存在的行业
    industries = []
    for code in node_list:
        if (
            code in stock_info_df.index
            and "industry" in stock_info_df.columns
        ):
            industries.append(stock_info_df.loc[code, "industry"])
        else:
            industries.append("未知行业")

    unique_industries = sorted(set(industries))

    cmap = plt.get_cmap("tab20")
    industry_color_map = {
        industry: cmap(i % 20)
        for i, industry in enumerate(unique_industries)
    }

    node_colors = [
        industry_color_map[industry]
        for industry in industries
    ]

    return industry_color_map, node_colors


def build_legend_handles(industry_color_map):
    handles = []
    for industry, color in industry_color_map.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=industry,
                markerfacecolor=color,
                markersize=10
            )
        )
    return handles


# ============================================================
# 9. 标签函数
# ============================================================

def build_node_labels(G, stock_info_df):
    labels = {}

    for code in G.nodes:
        if (
            code in stock_info_df.index
            and "name" in stock_info_df.columns
        ):
            name = stock_info_df.loc[code, "name"]
            labels[code] = f"{name}\n{code}"
        else:
            labels[code] = code

    return labels


# ============================================================
# 10. 构建阈值相关网络
# ============================================================

G_threshold = nx.Graph()
codes = corr.columns.tolist()

# 添加节点
for code in codes:
    attrs = {}

    if code in stock_info.index:
        if "name" in stock_info.columns:
            attrs["name"] = stock_info.loc[code, "name"]
        if "industry" in stock_info.columns:
            attrs["industry"] = stock_info.loc[code, "industry"]

    G_threshold.add_node(code, **attrs)

# 添加边
for i in range(len(codes)):
    for j in range(i + 1, len(codes)):

        code_i = codes[i]
        code_j = codes[j]

        rho = corr.loc[code_i, code_j]

        if pd.isna(rho):
            continue

        if USE_ABSOLUTE_THRESHOLD:
            selected = abs(rho) >= CORR_THRESHOLD
        else:
            selected = rho >= CORR_THRESHOLD

        if selected:
            distance = np.sqrt(2 * (1 - rho))

            G_threshold.add_edge(
                code_i,
                code_j,
                correlation=float(rho),      # 阈值网络边权：相关系数
                distance=float(distance)
            )


# ============================================================
# 11. 保存阈值网络边表
# ============================================================

threshold_edge_rows = []

for u, v, data in G_threshold.edges(data=True):
    threshold_edge_rows.append({
        "stock_1": u,
        "stock_2": v,
        "correlation_weight": data["correlation"],
        "distance": data["distance"]
    })

threshold_edges = pd.DataFrame(threshold_edge_rows)

if not threshold_edges.empty:
    threshold_edges = threshold_edges.sort_values(
        "correlation_weight",
        ascending=False
    )

threshold_edges.to_csv(
    PROCESSED_DIR / "threshold_edges.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 绘制阈值相关网络
# ============================================================

plt.figure(figsize=(14, 11))

pos_threshold = nx.spring_layout(
    G_threshold,
    seed=42,
    weight="correlation"
)

threshold_labels = build_node_labels(
    G_threshold,
    stock_info
)

industry_color_map_t, node_colors_t = build_industry_color_map(
    stock_info,
    list(G_threshold.nodes)
)

# 边宽按“相关系数”增强差异
threshold_edge_widths = [
    0.8 + 9 * data["correlation"]
    for _, _, data in G_threshold.edges(data=True)
]

# 边标签：显示边权（相关系数）
threshold_edge_labels = {
    (u, v): f"{data['correlation']:.2f}"
    for u, v, data in G_threshold.edges(data=True)
}

nx.draw_networkx_nodes(
    G_threshold,
    pos_threshold,
    node_color=node_colors_t,
    node_size=NODE_SIZE,
    edgecolors="black",
    linewidths=0.8
)

nx.draw_networkx_edges(
    G_threshold,
    pos_threshold,
    width=threshold_edge_widths,
    alpha=0.75
)

nx.draw_networkx_labels(
    G_threshold,
    pos_threshold,
    labels=threshold_labels,
    font_size=LABEL_SIZE
)

nx.draw_networkx_edge_labels(
    G_threshold,
    pos_threshold,
    edge_labels=threshold_edge_labels,
    font_size=EDGE_LABEL_SIZE,
    label_pos=0.5
)

legend_handles_t = build_legend_handles(industry_color_map_t)

plt.legend(
    handles=legend_handles_t,
    title="行业",
    loc="best",
    fontsize=8
)

plt.title(
    f"阈值相关网络（边权=相关系数，阈值 rho >= {CORR_THRESHOLD}）"
)

plt.axis("off")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "threshold_network.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. 构建完整相关距离网络
# ============================================================

G_full = nx.Graph()

for code in codes:
    G_full.add_node(code)

for i in range(len(codes)):
    for j in range(i + 1, len(codes)):

        code_i = codes[i]
        code_j = codes[j]

        rho = corr.loc[code_i, code_j]

        if pd.isna(rho):
            continue

        distance = np.sqrt(2 * (1 - rho))

        G_full.add_edge(
            code_i,
            code_j,
            correlation=float(rho),
            distance=float(distance)   # MST 边权候选：相关距离
        )


# ============================================================
# 14. 计算 MST
# ============================================================

G_mst = nx.minimum_spanning_tree(
    G_full,
    weight="distance"
)

print("\nMST 节点数量：", G_mst.number_of_nodes())
print("MST 边数量：", G_mst.number_of_edges())


# ============================================================
# 15. 保存 MST 边表
# ============================================================

mst_edge_rows = []

for u, v, data in G_mst.edges(data=True):
    mst_edge_rows.append({
        "stock_1": u,
        "stock_2": v,
        "distance_weight": data["distance"],    # MST 边权：相关距离
        "correlation": data["correlation"]
    })

mst_edges = pd.DataFrame(mst_edge_rows)

mst_edges = mst_edges.sort_values(
    "distance_weight",
    ascending=True
)

mst_edges.to_csv(
    PROCESSED_DIR / "mst_edges.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. MST 距离保持型布局
#     让每条 MST 边的几何长度与 distance 成比例
# ============================================================

def weighted_tree_layout(
    G,
    root=None,
    distance_attr="distance",
    scale=5.0
):
    """
    对树进行二维布局，使每条边的几何长度
    与指定 distance 属性严格成比例。
    """

    if root is None:
        root = max(G.degree, key=lambda x: x[1])[0]

    pos = {
        root: np.array([0.0, 0.0])
    }

    parent = {root: None}
    children = {node: [] for node in G.nodes}

    queue = [root]

    while queue:
        node = queue.pop(0)

        for nbr in G.neighbors(node):
            if nbr == parent[node]:
                continue
            if nbr in parent:
                continue

            parent[nbr] = node
            children[node].append(nbr)
            queue.append(nbr)

    def subtree_size(node):
        if not children[node]:
            return 1
        return sum(subtree_size(child) for child in children[node])

    sizes = {node: subtree_size(node) for node in G.nodes}

    def place_children(node, angle_start, angle_end):
        child_list = children[node]

        if not child_list:
            return

        total_size = sum(sizes[child] for child in child_list)
        current_angle = angle_start

        for child in child_list:
            fraction = sizes[child] / total_size
            child_angle_end = current_angle + fraction * (angle_end - angle_start)
            theta = (current_angle + child_angle_end) / 2

            d = G[node][child][distance_attr]
            length = scale * d

            pos[child] = (
                pos[node]
                + length * np.array([np.cos(theta), np.sin(theta)])
            )

            place_children(child, current_angle, child_angle_end)
            current_angle = child_angle_end

    place_children(root, 0, 2 * np.pi)

    return pos


# ============================================================
# 17. 绘制 MST
# ============================================================

plt.figure(figsize=(14, 11))

# 用“相关距离”控制边长
pos_mst = weighted_tree_layout(
    G_mst,
    distance_attr="distance",
    scale=MST_LENGTH_SCALE
)

mst_labels = build_node_labels(
    G_mst,
    stock_info
)

industry_color_map_m, node_colors_m = build_industry_color_map(
    stock_info,
    list(G_mst.nodes)
)

# 边宽按“相关系数”变化：相关越强，边越粗
mst_widths = [
    0.8 + 9 * data["correlation"]
    for _, _, data in G_mst.edges(data=True)
]

# 边标签：显示边权（相关距离）
mst_edge_labels = {
    (u, v): f"{data['distance']:.2f}"
    for u, v, data in G_mst.edges(data=True)
}

nx.draw_networkx_nodes(
    G_mst,
    pos_mst,
    node_color=node_colors_m,
    node_size=NODE_SIZE,
    edgecolors="black",
    linewidths=0.8
)

nx.draw_networkx_edges(
    G_mst,
    pos_mst,
    width=mst_widths,
    alpha=0.78
)

nx.draw_networkx_labels(
    G_mst,
    pos_mst,
    labels=mst_labels,
    font_size=LABEL_SIZE
)

nx.draw_networkx_edge_labels(
    G_mst,
    pos_mst,
    edge_labels=mst_edge_labels,
    font_size=EDGE_LABEL_SIZE,
    label_pos=0.5
)

legend_handles_m = build_legend_handles(industry_color_map_m)

plt.legend(
    handles=legend_handles_m,
    title="行业",
    loc="best",
    fontsize=8
)

plt.title(
    "MST（边权=相关距离；边长 ∝ 相关距离；边越短表示相关性越强）"
)

plt.axis("off")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "mst_network.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. 网络基本统计
# ============================================================

threshold_components = nx.number_connected_components(G_threshold)
threshold_density = nx.density(G_threshold)

network_summary = pd.DataFrame(
    [
        {
            "network": "threshold",
            "n_nodes": G_threshold.number_of_nodes(),
            "n_edges": G_threshold.number_of_edges(),
            "density": threshold_density,
            "n_components": threshold_components
        },
        {
            "network": "mst",
            "n_nodes": G_mst.number_of_nodes(),
            "n_edges": G_mst.number_of_edges(),
            "density": nx.density(G_mst),
            "n_components": nx.number_connected_components(G_mst)
        }
    ]
)

network_summary.to_csv(
    PROCESSED_DIR / "network_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. 输出摘要
# ============================================================

print("\n================================")
print("阶段五（修改版）完成")
print("================================")

print("\n阈值相关网络：")
print(f"节点数 = {G_threshold.number_of_nodes()}")
print(f"边数 = {G_threshold.number_of_edges()}")
print(f"连通分量 = {threshold_components}")
print(f"网络密度 = {threshold_density:.4f}")

print("\nMST：")
print(f"节点数 = {G_mst.number_of_nodes()}")
print(f"边数 = {G_mst.number_of_edges()}")

print("\n阈值网络中相关性最高的边：")
if not threshold_edges.empty:
    print(threshold_edges.head(10).to_string(index=False))
else:
    print("阈值网络没有边。")

print("\nMST 中最短的边（相关性通常更强）：")
print(mst_edges.head(10).to_string(index=False))
from __future__ import annotations

from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# 0. Project directory
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. Input / output paths
# ============================================================

NODE_FILE = (
    BASE_DIR
    / "stage6_2_node_structural_driver"
    / "node_structural_driver_summary.csv"
)


EDGE_FILE = (
    BASE_DIR
    / "stage5_differential_network"
    / "R3_R4_differential_edges_full.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "stage6_4_structural_driver_subnetwork"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Main settings
# ============================================================

# Number of major structural-driver nodes shown.
#
# Stage 6.3 used Top 6 as Major Drivers.
TOP_K_DRIVERS = 6


# For every Top Driver, retain its largest L differential edges.
TOP_EDGES_PER_DRIVER = 3


# Also retain non-zero differential edges directly connecting
# two Top Driver nodes.
INCLUDE_EDGES_AMONG_TOP_DRIVERS = True


# Ignore extremely small numerical differences.
DELTA_TOL = 1e-10


# Layout seed for reproducibility.
LAYOUT_SEED = 42


# Figure size
FIGSIZE = (
    13,
    10,
)


# Node-size range
MIN_NODE_SIZE = 700
MAX_NODE_SIZE = 2600


# Edge-width range
MIN_EDGE_WIDTH = 1.0
MAX_EDGE_WIDTH = 5.5


# ============================================================
# 3. Chinese font setup
# ============================================================

def setup_chinese_font() -> tuple[FontProperties, str]:

    preferred_fonts = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "DengXian",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]


    installed_fonts = {
        f.name
        for f in font_manager.fontManager.ttflist
    }


    # --------------------------------------------------------
    # Matplotlib registered fonts
    # --------------------------------------------------------

    for font_name in preferred_fonts:

        if font_name in installed_fonts:

            plt.rcParams["font.family"] = "sans-serif"

            plt.rcParams["font.sans-serif"] = [
                font_name,
                "DejaVu Sans",
            ]

            plt.rcParams["axes.unicode_minus"] = False


            return (
                FontProperties(
                    family=font_name
                ),
                font_name,
            )


    # --------------------------------------------------------
    # Windows direct path fallback
    # --------------------------------------------------------

    windows_fonts = [
        (
            "Microsoft YaHei",
            r"C:\Windows\Fonts\msyh.ttc",
        ),
        (
            "SimHei",
            r"C:\Windows\Fonts\simhei.ttf",
        ),
        (
            "DengXian",
            r"C:\Windows\Fonts\Deng.ttf",
        ),
        (
            "SimSun",
            r"C:\Windows\Fonts\simsun.ttc",
        ),
    ]


    for font_name, font_path in windows_fonts:

        if os.path.exists(font_path):

            try:
                font_manager.fontManager.addfont(
                    font_path
                )

            except Exception:
                pass


            plt.rcParams["axes.unicode_minus"] = False


            return (
                FontProperties(
                    fname=font_path
                ),
                font_name,
            )


    print(
        "WARNING: No Chinese-capable font detected."
    )


    return (
        FontProperties(
            family="DejaVu Sans"
        ),
        "DejaVu Sans (fallback)",
    )


CHINESE_FONT_PROP, CHINESE_FONT_NAME = (
    setup_chinese_font()
)


# ============================================================
# 4. CSV reader
# ============================================================

def read_csv_with_encoding_fallback(
    path: Path,
) -> pd.DataFrame:

    encodings = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
    ]


    last_error = None


    for encoding in encodings:

        try:

            df = pd.read_csv(
                path,
                encoding=encoding,
            )


            print(
                f"Loaded {path.name} "
                f"with encoding={encoding}"
            )


            return df


        except UnicodeDecodeError as exc:

            last_error = exc


    raise RuntimeError(
        f"Unable to read:\n"
        f"{path}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# 5. Check files
# ============================================================

if not NODE_FILE.exists():

    raise FileNotFoundError(
        f"Cannot find node summary:\n"
        f"{NODE_FILE}"
    )


if not EDGE_FILE.exists():

    raise FileNotFoundError(
        f"Cannot find differential-edge file:\n"
        f"{EDGE_FILE}"
    )


node_df = read_csv_with_encoding_fallback(
    NODE_FILE
)


edge_df = read_csv_with_encoding_fallback(
    EDGE_FILE
)


# ============================================================
# 6. Required columns
# ============================================================

required_node_columns = [
    "stock",
    "name",
    "industry",
    "driver_score",
    "driver_share",
    "driver_rank",
]


required_edge_columns = [
    "stock_i",
    "stock_j",
    "industry_i",
    "industry_j",
    "partial_R3",
    "partial_R4",
    "delta_partial",
    "abs_delta_partial",
    "edge_R3",
    "edge_R4",
    "support_change",
]


missing_node = [
    col
    for col in required_node_columns
    if col not in node_df.columns
]


missing_edge = [
    col
    for col in required_edge_columns
    if col not in edge_df.columns
]


if missing_node:

    raise ValueError(
        f"Missing node columns:\n"
        f"{missing_node}"
    )


if missing_edge:

    raise ValueError(
        f"Missing edge columns:\n"
        f"{missing_edge}"
    )


# ============================================================
# 7. Clean stock codes
# ============================================================

def clean_stock_code(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(6)
    )


node_df["stock"] = clean_stock_code(
    node_df["stock"]
)


edge_df["stock_i"] = clean_stock_code(
    edge_df["stock_i"]
)


edge_df["stock_j"] = clean_stock_code(
    edge_df["stock_j"]
)


# Numeric columns
for col in [
    "driver_score",
    "driver_share",
    "driver_rank",
]:

    node_df[col] = pd.to_numeric(
        node_df[col],
        errors="raise",
    )


for col in [
    "partial_R3",
    "partial_R4",
    "delta_partial",
    "abs_delta_partial",
    "edge_R3",
    "edge_R4",
]:

    edge_df[col] = pd.to_numeric(
        edge_df[col],
        errors="raise",
    )


# ============================================================
# 8. Select Top structural drivers
# ============================================================

node_df = (
    node_df
    .sort_values(
        "driver_score",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


top_driver_df = (
    node_df
    .head(
        TOP_K_DRIVERS
    )
    .copy()
)


top_driver_stocks = set(
    top_driver_df["stock"]
)


print(
    "\n"
    + "=" * 78
)


print(
    "Stage 6.4: Structural Driver Subnetwork"
)


print(
    "=" * 78
)


print(
    "\nTop structural drivers:"
)


print(
    top_driver_df[
        [
            "driver_rank",
            "stock",
            "name",
            "industry",
            "driver_score",
            "driver_share",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 9. Construct unordered edge ID
# ============================================================

def make_edge_key(
    stock_i: str,
    stock_j: str,
) -> tuple[str, str]:

    return tuple(
        sorted(
            [
                stock_i,
                stock_j,
            ]
        )
    )


edge_df["edge_key"] = edge_df.apply(
    lambda row: make_edge_key(
        row["stock_i"],
        row["stock_j"],
    ),
    axis=1,
)


# ============================================================
# 10. Candidate differential edges
#
# Remove inactive / numerically zero differences.
# ============================================================

candidate_edges = edge_df.loc[
    (
        edge_df["abs_delta_partial"]
        >
        DELTA_TOL
    )
    &
    (
        edge_df["support_change"]
        !=
        "Absent"
    )
].copy()


# ============================================================
# 11. Select Top-L incident edges for each driver
# ============================================================

selected_edge_keys: set[
    tuple[str, str]
] = set()


for driver in top_driver_stocks:

    incident = candidate_edges.loc[
        (
            candidate_edges["stock_i"]
            ==
            driver
        )
        |
        (
            candidate_edges["stock_j"]
            ==
            driver
        )
    ].copy()


    incident = (
        incident
        .sort_values(
            "abs_delta_partial",
            ascending=False,
        )
        .head(
            TOP_EDGES_PER_DRIVER
        )
    )


    selected_edge_keys.update(
        incident["edge_key"]
    )


# ============================================================
# 12. Optionally include all non-zero edges among major drivers
# ============================================================

if INCLUDE_EDGES_AMONG_TOP_DRIVERS:

    top_to_top = candidate_edges.loc[
        candidate_edges["stock_i"].isin(
            top_driver_stocks
        )
        &
        candidate_edges["stock_j"].isin(
            top_driver_stocks
        )
    ]


    selected_edge_keys.update(
        top_to_top["edge_key"]
    )


# ============================================================
# 13. Final subnetwork edge table
# ============================================================

sub_edges = edge_df.loc[
    edge_df["edge_key"].isin(
        selected_edge_keys
    )
].copy()


sub_edges = (
    sub_edges
    .sort_values(
        "abs_delta_partial",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


if len(sub_edges) == 0:

    raise RuntimeError(
        "No differential edges were selected."
    )


# ============================================================
# 14. Identify all nodes appearing in subnetwork
# ============================================================

subnetwork_stocks = sorted(
    set(
        sub_edges["stock_i"]
    )
    |
    set(
        sub_edges["stock_j"]
    )
)


sub_nodes = node_df.loc[
    node_df["stock"].isin(
        subnetwork_stocks
    )
].copy()


sub_nodes[
    "is_top_driver"
] = (
    sub_nodes["stock"]
    .isin(
        top_driver_stocks
    )
)


sub_nodes = (
    sub_nodes
    .sort_values(
        [
            "is_top_driver",
            "driver_score",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 15. Save selected node / edge tables
# ============================================================

sub_nodes.to_csv(
    OUTPUT_DIR
    / "R3_R4_structural_driver_subnetwork_nodes.csv",

    index=False,
    encoding="utf-8-sig",
)


sub_edges.drop(
    columns=[
        "edge_key"
    ],
    errors="ignore",
).to_csv(
    OUTPUT_DIR
    / "R3_R4_structural_driver_subnetwork_edges.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. Simple summary
# ============================================================

summary = pd.DataFrame(
    [
        {
            "top_k_drivers":
                TOP_K_DRIVERS,

            "top_edges_per_driver":
                TOP_EDGES_PER_DRIVER,

            "n_subnetwork_nodes":
                len(sub_nodes),

            "n_subnetwork_edges":
                len(sub_edges),

            "n_persistent":
                int(
                    (
                        sub_edges[
                            "support_change"
                        ]
                        ==
                        "Persistent"
                    )
                    .sum()
                ),

            "n_lost":
                int(
                    (
                        sub_edges[
                            "support_change"
                        ]
                        ==
                        "Lost"
                    )
                    .sum()
                ),

            "n_gained":
                int(
                    (
                        sub_edges[
                            "support_change"
                        ]
                        ==
                        "Gained"
                    )
                    .sum()
                ),

            "mean_abs_delta":
                float(
                    sub_edges[
                        "abs_delta_partial"
                    ]
                    .mean()
                ),

            "max_abs_delta":
                float(
                    sub_edges[
                        "abs_delta_partial"
                    ]
                    .max()
                ),
        }
    ]
)


summary.to_csv(
    OUTPUT_DIR
    / "R3_R4_structural_driver_subnetwork_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. Build NetworkX graph
# ============================================================

G = nx.Graph()


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

for row in sub_nodes.itertuples():

    G.add_node(
        row.stock,

        name=row.name,

        industry=row.industry,

        driver_score=row.driver_score,

        driver_share=row.driver_share,

        driver_rank=row.driver_rank,

        is_top_driver=row.is_top_driver,
    )


# ------------------------------------------------------------
# Edges
# ------------------------------------------------------------

for row in sub_edges.itertuples():

    G.add_edge(
        row.stock_i,
        row.stock_j,

        partial_R3=row.partial_R3,

        partial_R4=row.partial_R4,

        delta_partial=row.delta_partial,

        abs_delta_partial=row.abs_delta_partial,

        support_change=row.support_change,
    )


# ============================================================
# 18. Layout
# ============================================================

pos = nx.spring_layout(
    G,
    seed=LAYOUT_SEED,
    weight="abs_delta_partial",
    k=None,
)


# ============================================================
# 19. Node sizes
# ============================================================

driver_scores = np.array(
    [
        G.nodes[node][
            "driver_score"
        ]
        for node in G.nodes
    ],
    dtype=float,
)


min_score = float(
    driver_scores.min()
)


max_score = float(
    driver_scores.max()
)


def scale_value(
    value: float,
    vmin: float,
    vmax: float,
    out_min: float,
    out_max: float,
) -> float:

    if np.isclose(
        vmax,
        vmin,
    ):

        return (
            out_min
            +
            out_max
        ) / 2


    return (
        out_min
        +
        (
            value
            -
            vmin
        )
        /
        (
            vmax
            -
            vmin
        )
        *
        (
            out_max
            -
            out_min
        )
    )


node_sizes = [
    scale_value(
        G.nodes[node][
            "driver_score"
        ],
        min_score,
        max_score,
        MIN_NODE_SIZE,
        MAX_NODE_SIZE,
    )
    for node in G.nodes
]


# ============================================================
# 20. Industry colors
# ============================================================

industries = sorted(
    {
        G.nodes[node][
            "industry"
        ]
        for node in G.nodes
    }
)


cmap = plt.get_cmap(
    "tab10"
)


industry_color = {
    industry:
        cmap(
            i
            %
            cmap.N
        )
    for i, industry in enumerate(
        industries
    )
}


node_colors = [
    industry_color[
        G.nodes[node][
            "industry"
        ]
    ]
    for node in G.nodes
]


# ============================================================
# 21. Edge widths
# ============================================================

edge_abs_values = np.array(
    [
        G.edges[
            u,
            v,
        ][
            "abs_delta_partial"
        ]
        for u, v in G.edges
    ],
    dtype=float,
)


edge_min = float(
    edge_abs_values.min()
)


edge_max = float(
    edge_abs_values.max()
)


def edge_width(
    value: float,
) -> float:

    return scale_value(
        value,
        edge_min,
        edge_max,
        MIN_EDGE_WIDTH,
        MAX_EDGE_WIDTH,
    )


# ============================================================
# 22. Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE
)


# ============================================================
# 23. Draw nodes
# ============================================================

nx.draw_networkx_nodes(
    G,
    pos,

    node_size=node_sizes,

    node_color=node_colors,

    edgecolors="black",

    linewidths=[
        2.5
        if G.nodes[node][
            "is_top_driver"
        ]
        else 1.0

        for node in G.nodes
    ],

    alpha=0.9,

    ax=ax,
)


# ============================================================
# 24. Draw edges by support-change category
#
# Persistent = solid
# Lost       = dashed
# Gained     = dotted
# ============================================================

support_styles = {
    "Persistent":
        "solid",

    "Lost":
        "dashed",

    "Gained":
        "dotted",
}


# Use Matplotlib default color cycle.
support_colors = {
    "Persistent":
        "C0",

    "Lost":
        "C1",

    "Gained":
        "C2",
}


for support_type in [
    "Persistent",
    "Lost",
    "Gained",
]:

    edges_this_type = [
        (
            u,
            v,
        )
        for u, v, data
        in G.edges(
            data=True
        )
        if data[
            "support_change"
        ]
        ==
        support_type
    ]


    if not edges_this_type:

        continue


    widths = [
        edge_width(
            G.edges[
                u,
                v,
            ][
                "abs_delta_partial"
            ]
        )
        for u, v
        in edges_this_type
    ]


    nx.draw_networkx_edges(
        G,
        pos,

        edgelist=edges_this_type,

        width=widths,

        style=support_styles[
            support_type
        ],

        edge_color=support_colors[
            support_type
        ],

        alpha=0.8,

        ax=ax,
    )


# ============================================================
# 25. Chinese node labels
# ============================================================

for node, (
    x,
    y,
) in pos.items():

    node_data = G.nodes[
        node
    ]


    label = (
        f"{node_data['name']}\n"
        f"{node}"
    )


    # Major drivers are bold.
    weight = (
        "bold"
        if node_data[
            "is_top_driver"
        ]
        else "normal"
    )


    ax.annotate(
        label,

        xy=(
            x,
            y,
        ),

        xytext=(
            0,
            0,
        ),

        textcoords="offset points",

        ha="center",
        va="center",

        fontproperties=CHINESE_FONT_PROP,

        fontsize=9,

        fontweight=weight,
    )


# ============================================================
# 26. Edge labels
#
# Display Delta partial correlation.
# ============================================================

edge_labels = {
    (
        u,
        v,
    ):
        f"{data['delta_partial']:+.3f}"

    for u, v, data
    in G.edges(
        data=True
    )
}


nx.draw_networkx_edge_labels(
    G,
    pos,

    edge_labels=edge_labels,

    font_size=7,

    rotate=False,

    label_pos=0.5,

    ax=ax,
)


# ============================================================
# 27. Industry legend
# ============================================================

industry_handles = []


for industry in industries:

    handle = plt.Line2D(
        [
            0
        ],
        [
            0
        ],

        marker="o",

        linestyle="",

        markerfacecolor=industry_color[
            industry
        ],

        markeredgecolor="black",

        markersize=9,

        label=industry,
    )


    industry_handles.append(
        handle
    )


legend_industry = ax.legend(
    handles=industry_handles,

    title="行业",

    loc="upper left",

    bbox_to_anchor=(
        1.01,
        1.0,
    ),

    prop=CHINESE_FONT_PROP,
)


legend_industry.get_title().set_fontproperties(
    CHINESE_FONT_PROP
)


ax.add_artist(
    legend_industry
)


# ============================================================
# 28. Edge-type legend
# ============================================================

edge_handles = []


for support_type in [
    "Persistent",
    "Lost",
    "Gained",
]:

    handle = plt.Line2D(
        [
            0,
            1,
        ],
        [
            0,
            0,
        ],

        linestyle=support_styles[
            support_type
        ],

        color=support_colors[
            support_type
        ],

        linewidth=2.5,

        label=support_type,
    )


    edge_handles.append(
        handle
    )


ax.legend(
    handles=edge_handles,

    title="边类型",

    loc="upper left",

    bbox_to_anchor=(
        1.01,
        0.55,
    ),

    prop=CHINESE_FONT_PROP,
)


# ============================================================
# 29. Figure title / note
# ============================================================

ax.set_title(
    (
        "R3 → R4 Structural Driver Subnetwork\n"
        "节点大小表示 Driver Score，"
        "边宽表示 |ΔPartial Correlation|"
    ),

    fontproperties=CHINESE_FONT_PROP,

    fontsize=15,

    pad=20,
)


ax.text(
    0.5,
    -0.04,

    (
        "实线=Persistent，虚线=Lost，点线=Gained；"
        "粗边框节点为 Top Structural Drivers。"
    ),

    transform=ax.transAxes,

    ha="center",

    fontproperties=CHINESE_FONT_PROP,

    fontsize=9,
)


ax.axis(
    "off"
)


plt.tight_layout()


OUTPUT_FIGURE = (
    OUTPUT_DIR
    / "R3_R4_structural_driver_subnetwork.png"
)


plt.savefig(
    OUTPUT_FIGURE,

    dpi=300,

    bbox_inches="tight",
)


plt.close()


# ============================================================
# 30. Console output
# ============================================================

print(
    "\n"
    + "=" * 78
)


print(
    "Selected subnetwork"
)


print(
    "=" * 78
)


print(
    f"Number of nodes: "
    f"{len(sub_nodes)}"
)


print(
    f"Number of edges: "
    f"{len(sub_edges)}"
)


print(
    "\nSelected differential edges:"
)


print(
    sub_edges[
        [
            "stock_i",
            "stock_j",
            "industry_i",
            "industry_j",
            "partial_R3",
            "partial_R4",
            "delta_partial",
            "abs_delta_partial",
            "support_change",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 78
)


print(
    "Stage 6.4 completed successfully"
)


print(
    "=" * 78
)


print(
    f"\nChinese font used: "
    f"{CHINESE_FONT_NAME}"
)


print(
    "\nOutput directory:"
)


print(
    OUTPUT_DIR.resolve()
)


print(
    "\nMain output files:"
)


for filename in [
    "R3_R4_structural_driver_subnetwork_nodes.csv",
    "R3_R4_structural_driver_subnetwork_edges.csv",
    "R3_R4_structural_driver_subnetwork_summary.csv",
    "R3_R4_structural_driver_subnetwork.png",
]:

    print(
        f"  - {filename}"
    )
from __future__ import annotations

from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

import numpy as np
import pandas as pd


# ============================================================
# 0. Project directory
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. Input / output paths
# ============================================================

INPUT_FILE = (
    BASE_DIR
    / "stage5_differential_network"
    / "R3_R4_differential_edges_full.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "stage6_2_node_structural_driver"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Settings
# ============================================================

# Number of top structural-driver stocks saved separately.
TOP_K_NODES = 10


# Number of largest incident differential edges saved
# for each top driver stock.
TOP_EDGES_PER_NODE = 5


# Figure size
RANKING_FIGSIZE = (
    10,
    7,
)


# ============================================================
# 3. Chinese font configuration
# ============================================================

def setup_chinese_font() -> tuple[FontProperties, str]:
    """
    Automatically detect a Chinese-capable font.

    Returns
    -------
    font_prop:
        FontProperties used for plotting Chinese labels.

    font_name:
        Description of selected font.
    """

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
    # 1. Search fonts registered by Matplotlib
    # --------------------------------------------------------

    for font_name in preferred_fonts:

        if font_name in installed_fonts:

            plt.rcParams[
                "font.family"
            ] = "sans-serif"


            plt.rcParams[
                "font.sans-serif"
            ] = [
                font_name,
                "DejaVu Sans",
            ]


            plt.rcParams[
                "axes.unicode_minus"
            ] = False


            font_prop = FontProperties(
                family=font_name
            )


            print(
                f"Chinese font selected: "
                f"{font_name}"
            )


            return (
                font_prop,
                font_name,
            )


    # --------------------------------------------------------
    # 2. Direct Windows font-file fallback
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

        if os.path.exists(
            font_path
        ):

            try:

                font_manager.fontManager.addfont(
                    font_path
                )

            except Exception:

                pass


            font_prop = FontProperties(
                fname=font_path
            )


            plt.rcParams[
                "axes.unicode_minus"
            ] = False


            print(
                f"Chinese font selected directly: "
                f"{font_name}"
            )


            print(
                f"Font file: "
                f"{font_path}"
            )


            return (
                font_prop,
                font_name,
            )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    print(
        "WARNING: No Chinese-capable font was detected."
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
# 4. CSV reading helper
# ============================================================

def read_csv_with_encoding_fallback(
    path: Path,
    **kwargs,
) -> pd.DataFrame:
    """
    Read CSV using common Chinese/Windows encodings.
    """

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
                **kwargs,
            )


            print(
                f"Loaded {path.name} "
                f"with encoding={encoding}"
            )


            return df


        except UnicodeDecodeError as exc:

            last_error = exc


    raise RuntimeError(
        f"Unable to decode file:\n"
        f"{path}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# 5. Load Stage-5 differential-edge table
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "Cannot find Stage-5 differential edge table:\n"
        f"{INPUT_FILE}"
    )


edge_df = read_csv_with_encoding_fallback(
    INPUT_FILE
)


# ============================================================
# 6. Required columns
# ============================================================

required_columns = [
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


missing_columns = [
    col
    for col in required_columns
    if col not in edge_df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        f"{missing_columns}\n\n"
        f"Available columns:\n"
        f"{list(edge_df.columns)}"
    )


# ============================================================
# 7. Basic cleaning
# ============================================================

numeric_columns = [
    "partial_R3",
    "partial_R4",
    "delta_partial",
    "abs_delta_partial",
    "edge_R3",
    "edge_R4",
]


for col in numeric_columns:

    edge_df[
        col
    ] = pd.to_numeric(
        edge_df[col],
        errors="raise",
    )


for col in [
    "stock_i",
    "stock_j",
]:

    edge_df[
        col
    ] = (
        edge_df[
            col
        ]
        .astype(str)
        .str.strip()
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(6)
    )


for col in [
    "industry_i",
    "industry_j",
]:

    edge_df[
        col
    ] = (
        edge_df[
            col
        ]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 8. Basic diagnostics
# ============================================================

stocks = sorted(
    set(
        edge_df[
            "stock_i"
        ]
    )
    |
    set(
        edge_df[
            "stock_j"
        ]
    )
)


p = len(
    stocks
)


expected_pairs = (
    p
    *
    (p - 1)
    //
    2
)


print(
    "\n"
    + "=" * 78
)


print(
    "Stage 6.2: Node-level Structural Driver Analysis"
)


print(
    "=" * 78
)


print(
    f"Number of stocks: "
    f"{p}"
)


print(
    f"Observed stock pairs: "
    f"{len(edge_df)}"
)


print(
    f"Expected stock pairs: "
    f"{expected_pairs}"
)


if len(
    edge_df
) != expected_pairs:

    print(
        "WARNING:"
    )

    print(
        "The edge table does not contain every "
        "possible stock pair."
    )


# ============================================================
# 9. Construct stock metadata from edge table
# ============================================================

metadata_rows = []


for row in edge_df.itertuples():

    item_i = {
        "stock":
            row.stock_i,

        "industry":
            row.industry_i,
    }


    item_j = {
        "stock":
            row.stock_j,

        "industry":
            row.industry_j,
    }


    if (
        "name_i"
        in
        edge_df.columns
    ):

        item_i[
            "name"
        ] = getattr(
            row,
            "name_i"
        )


    if (
        "name_j"
        in
        edge_df.columns
    ):

        item_j[
            "name"
        ] = getattr(
            row,
            "name_j"
        )


    metadata_rows.append(
        item_i
    )


    metadata_rows.append(
        item_j
    )


metadata = pd.DataFrame(
    metadata_rows
)


metadata = (
    metadata
    .drop_duplicates(
        subset=[
            "stock"
        ]
    )
    .set_index(
        "stock"
    )
)


HAS_NAME = (
    "name"
    in
    metadata.columns
)


# ============================================================
# 10. Edge-level structural-change energy
#
# e_ij =
# (rho_R4 - rho_R3)^2
# ============================================================

edge_df[
    "transition_energy"
] = (
    edge_df[
        "delta_partial"
    ]
    ** 2
)


# ------------------------------------------------------------
# Same vs Cross
# ------------------------------------------------------------

edge_df[
    "relation"
] = np.where(
    edge_df[
        "industry_i"
    ]
    ==
    edge_df[
        "industry_j"
    ],
    "Same",
    "Cross",
)


# ------------------------------------------------------------
# Support-change energy
#
# Lost + Gained
# ------------------------------------------------------------

edge_df[
    "support_change_energy"
] = np.where(
    edge_df[
        "support_change"
    ]
    .isin(
        [
            "Lost",
            "Gained",
        ]
    ),
    edge_df[
        "transition_energy"
    ],
    0.0,
)


# ------------------------------------------------------------
# Persistent-edge strength-change energy
# ------------------------------------------------------------

edge_df[
    "persistent_change_energy"
] = np.where(
    edge_df[
        "support_change"
    ]
    ==
    "Persistent",
    edge_df[
        "transition_energy"
    ],
    0.0,
)


# ============================================================
# 11. Whole-network transition energy
# ============================================================

total_transition_energy = float(
    edge_df[
        "transition_energy"
    ]
    .sum()
)


if total_transition_energy <= 0:

    raise ValueError(
        "Total transition energy is zero."
    )


total_transition_intensity = float(
    np.sqrt(
        total_transition_energy
    )
)


print(
    f"Total transition energy: "
    f"{total_transition_energy:.8f}"
)


print(
    f"Total transition intensity: "
    f"{total_transition_intensity:.8f}"
)


# ============================================================
# 12. Compute node-level structural driver scores
# ============================================================

node_rows = []


for stock in stocks:

    # --------------------------------------------------------
    # Select all edges incident to stock i
    # --------------------------------------------------------

    incident = edge_df.loc[
        (
            edge_df[
                "stock_i"
            ]
            ==
            stock
        )
        |
        (
            edge_df[
                "stock_j"
            ]
            ==
            stock
        )
    ].copy()


    n_possible_incident_edges = len(
        incident
    )


    # --------------------------------------------------------
    # Structural Driver Score
    #
    # D_i =
    # sum_j (Delta rho_ij)^2
    # --------------------------------------------------------

    driver_score = float(
        incident[
            "transition_energy"
        ]
        .sum()
    )


    # --------------------------------------------------------
    # Same / Cross decomposition
    # --------------------------------------------------------

    same_driver_score = float(
        incident.loc[
            incident[
                "relation"
            ]
            ==
            "Same",

            "transition_energy",
        ]
        .sum()
    )


    cross_driver_score = float(
        incident.loc[
            incident[
                "relation"
            ]
            ==
            "Cross",

            "transition_energy",
        ]
        .sum()
    )


    # Check decomposition
    if not np.isclose(
        driver_score,
        (
            same_driver_score
            +
            cross_driver_score
        ),
        atol=1e-12,
    ):

        raise RuntimeError(
            f"Same/Cross decomposition failed "
            f"for stock {stock}."
        )


    cross_driver_share = (
        cross_driver_score
        /
        driver_score

        if driver_score > 0
        else np.nan
    )


    same_driver_share = (
        same_driver_score
        /
        driver_score

        if driver_score > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Support-change vs Persistent-strength decomposition
    # --------------------------------------------------------

    support_change_score = float(
        incident[
            "support_change_energy"
        ]
        .sum()
    )


    persistent_change_score = float(
        incident[
            "persistent_change_energy"
        ]
        .sum()
    )


    support_change_share = (
        support_change_score
        /
        driver_score

        if driver_score > 0
        else np.nan
    )


    persistent_change_share = (
        persistent_change_score
        /
        driver_score

        if driver_score > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Support-set counts
    # --------------------------------------------------------

    lost_degree = int(
        (
            incident[
                "support_change"
            ]
            ==
            "Lost"
        )
        .sum()
    )


    gained_degree = int(
        (
            incident[
                "support_change"
            ]
            ==
            "Gained"
        )
        .sum()
    )


    persistent_degree = int(
        (
            incident[
                "support_change"
            ]
            ==
            "Persistent"
        )
        .sum()
    )


    absent_degree = int(
        (
            incident[
                "support_change"
            ]
            ==
            "Absent"
        )
        .sum()
    )


    support_turnover = (
        lost_degree
        +
        gained_degree
    )


    support_turnover_rate = (
        support_turnover
        /
        n_possible_incident_edges

        if n_possible_incident_edges > 0
        else np.nan
    )


    net_degree_change = (
        gained_degree
        -
        lost_degree
    )


    # --------------------------------------------------------
    # R3 / R4 selected degree
    # --------------------------------------------------------

    degree_R3 = int(
        incident[
            "edge_R3"
        ]
        .sum()
    )


    degree_R4 = int(
        incident[
            "edge_R4"
        ]
        .sum()
    )


    # --------------------------------------------------------
    # Persistent-edge retention
    # --------------------------------------------------------

    retention_from_R3 = (
        persistent_degree
        /
        degree_R3

        if degree_R3 > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Mean absolute change among all possible incident pairs
    # --------------------------------------------------------

    mean_abs_delta = float(
        incident[
            "abs_delta_partial"
        ]
        .mean()
    )


    max_abs_delta = float(
        incident[
            "abs_delta_partial"
        ]
        .max()
    )


    industry = (
        metadata.at[
            stock,
            "industry",
        ]
    )


    if HAS_NAME:

        stock_name = (
            metadata.at[
                stock,
                "name",
            ]
        )

    else:

        stock_name = stock


    node_rows.append(
        {
            "stock":
                stock,

            "name":
                stock_name,

            "industry":
                industry,

            "possible_incident_pairs":
                n_possible_incident_edges,

            "degree_R3":
                degree_R3,

            "degree_R4":
                degree_R4,

            "persistent_degree":
                persistent_degree,

            "lost_degree":
                lost_degree,

            "gained_degree":
                gained_degree,

            "absent_degree":
                absent_degree,

            "support_turnover":
                support_turnover,

            "support_turnover_rate":
                support_turnover_rate,

            "net_degree_change":
                net_degree_change,

            "retention_from_R3":
                retention_from_R3,

            "driver_score":
                driver_score,

            "same_driver_score":
                same_driver_score,

            "cross_driver_score":
                cross_driver_score,

            "same_driver_share":
                same_driver_share,

            "cross_driver_share":
                cross_driver_share,

            "support_change_score":
                support_change_score,

            "persistent_change_score":
                persistent_change_score,

            "support_change_share":
                support_change_share,

            "persistent_change_share":
                persistent_change_share,

            "mean_abs_delta":
                mean_abs_delta,

            "max_abs_delta":
                max_abs_delta,
        }
    )


node_summary = pd.DataFrame(
    node_rows
)


# ============================================================
# 13. Node-attributed Driver Share
#
# Each edge contributes its transition energy to BOTH end
# nodes.
#
# Therefore:
#
# sum_i D_i = 2 * total_transition_energy.
#
# We normalize by sum_i D_i, so:
#
# sum_i DriverShare_i = 1.
# ============================================================

total_node_driver_score = float(
    node_summary[
        "driver_score"
    ]
    .sum()
)


expected_total_node_score = (
    2.0
    *
    total_transition_energy
)


if not np.isclose(
    total_node_driver_score,
    expected_total_node_score,
    atol=1e-10,
):

    print(
        "WARNING:"
    )

    print(
        "Node driver scores do not sum to "
        "2 × total transition energy."
    )


node_summary[
    "driver_share"
] = (
    node_summary[
        "driver_score"
    ]
    /
    total_node_driver_score
)


# ============================================================
# 14. Ranking
# ============================================================

node_summary[
    "driver_rank"
] = (
    node_summary[
        "driver_score"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)


node_summary[
    "turnover_rank"
] = (
    node_summary[
        "support_turnover"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)


node_summary[
    "cross_driver_rank"
] = (
    node_summary[
        "cross_driver_score"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)


node_summary = (
    node_summary
    .sort_values(
        [
            "driver_score",
            "support_turnover",
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
# 15. Save main summary
# ============================================================

node_summary.to_csv(
    OUTPUT_DIR
    / "node_structural_driver_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. Save Top-K drivers
# ============================================================

top_nodes = (
    node_summary
    .head(
        TOP_K_NODES
    )
    .copy()
)


top_nodes.to_csv(
    OUTPUT_DIR
    / "top_node_structural_drivers.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. Save top incident differential edges
#     for top structural drivers
# ============================================================

top_driver_edge_rows = []


for stock in (
    top_nodes[
        "stock"
    ]
):

    incident = edge_df.loc[
        (
            edge_df[
                "stock_i"
            ]
            ==
            stock
        )
        |
        (
            edge_df[
                "stock_j"
            ]
            ==
            stock
        )
    ].copy()


    incident = (
        incident
        .sort_values(
            "abs_delta_partial",
            ascending=False,
        )
        .head(
            TOP_EDGES_PER_NODE
        )
    )


    for row in incident.itertuples():

        if row.stock_i == stock:

            counterpart = (
                row.stock_j
            )

            counterpart_industry = (
                row.industry_j
            )

            if (
                "name_j"
                in
                edge_df.columns
            ):

                counterpart_name = (
                    getattr(
                        row,
                        "name_j"
                    )
                )

            else:

                counterpart_name = (
                    counterpart
                )


        else:

            counterpart = (
                row.stock_i
            )

            counterpart_industry = (
                row.industry_i
            )

            if (
                "name_i"
                in
                edge_df.columns
            ):

                counterpart_name = (
                    getattr(
                        row,
                        "name_i"
                    )
                )

            else:

                counterpart_name = (
                    counterpart
                )


        top_driver_edge_rows.append(
            {
                "driver_stock":
                    stock,

                "driver_name":
                    metadata.at[
                        stock,
                        "name",
                    ]
                    if HAS_NAME
                    else stock,

                "driver_industry":
                    metadata.at[
                        stock,
                        "industry",
                    ],

                "counterpart_stock":
                    counterpart,

                "counterpart_name":
                    counterpart_name,

                "counterpart_industry":
                    counterpart_industry,

                "relation":
                    row.relation,

                "partial_R3":
                    row.partial_R3,

                "partial_R4":
                    row.partial_R4,

                "delta_partial":
                    row.delta_partial,

                "abs_delta_partial":
                    row.abs_delta_partial,

                "transition_energy":
                    row.transition_energy,

                "support_change":
                    row.support_change,
            }
        )


top_driver_edges = pd.DataFrame(
    top_driver_edge_rows
)


top_driver_edges.to_csv(
    OUTPUT_DIR
    / "top_driver_incident_edges.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 18. Figure 1:
#     Driver-score ranking
# ============================================================

plot_df = (
    node_summary
    .head(
        TOP_K_NODES
    )
    .sort_values(
        "driver_score",
        ascending=True,
    )
    .copy()
)


plot_df[
    "plot_label"
] = (
    plot_df[
        "name"
    ]
    .astype(str)
    +
    "\n"
    +
    plot_df[
        "stock"
    ]
    .astype(str)
)


fig, ax = plt.subplots(
    figsize=RANKING_FIGSIZE
)


ax.barh(
    plot_df[
        "plot_label"
    ],
    plot_df[
        "driver_score"
    ],
)


ax.set_xlabel(
    "Structural Driver Score",
)


ax.set_ylabel(
    "股票",
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_title(
    "R3 → R4 主要股票结构变化驱动者",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


for tick in (
    ax.get_yticklabels()
):

    tick.set_fontproperties(
        CHINESE_FONT_PROP
    )


for i, value in enumerate(
    plot_df[
        "driver_score"
    ]
):

    ax.text(
        value,
        i,
        f" {value:.4f}",
        va="center",
        fontsize=8,
    )


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "node_driver_ranking.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 19. Figure 2:
#     Driver Share ranking
# ============================================================

plot_df_share = (
    node_summary
    .head(
        TOP_K_NODES
    )
    .sort_values(
        "driver_share",
        ascending=True,
    )
    .copy()
)


plot_df_share[
    "plot_label"
] = (
    plot_df_share[
        "name"
    ]
    .astype(str)
    +
    "\n"
    +
    plot_df_share[
        "stock"
    ]
    .astype(str)
)


fig, ax = plt.subplots(
    figsize=RANKING_FIGSIZE
)


ax.barh(
    plot_df_share[
        "plot_label"
    ],
    plot_df_share[
        "driver_share"
    ],
)


ax.set_xlabel(
    "Node-attributed Driver Share",
)


ax.set_ylabel(
    "股票",
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_title(
    "R3 → R4 股票结构变化贡献比例",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


for tick in (
    ax.get_yticklabels()
):

    tick.set_fontproperties(
        CHINESE_FONT_PROP
    )


for i, value in enumerate(
    plot_df_share[
        "driver_share"
    ]
):

    ax.text(
        value,
        i,
        f" {value:.1%}",
        va="center",
        fontsize=8,
    )


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "node_driver_share_ranking.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 20. Figure 3:
#     Same / Cross Driver Score
# ============================================================

same_cross_plot = (
    node_summary
    .head(
        TOP_K_NODES
    )
    .sort_values(
        "driver_score",
        ascending=True,
    )
    .copy()
)


same_cross_plot[
    "plot_label"
] = (
    same_cross_plot[
        "name"
    ]
    .astype(str)
    +
    "\n"
    +
    same_cross_plot[
        "stock"
    ]
    .astype(str)
)


y_positions = np.arange(
    len(
        same_cross_plot
    )
)


fig, ax = plt.subplots(
    figsize=RANKING_FIGSIZE
)


ax.barh(
    y_positions,
    same_cross_plot[
        "same_driver_score"
    ],
    label="同行业",
)


ax.barh(
    y_positions,
    same_cross_plot[
        "cross_driver_score"
    ],
    left=same_cross_plot[
        "same_driver_score"
    ],
    label="跨行业",
)


ax.set_yticks(
    y_positions
)


ax.set_yticklabels(
    same_cross_plot[
        "plot_label"
    ],
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_xlabel(
    "Structural Driver Score",
)


ax.set_ylabel(
    "股票",
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_title(
    "主要结构变化驱动股票的同行业 / 跨行业贡献",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


legend = ax.legend()


for text in (
    legend.get_texts()
):

    text.set_fontproperties(
        CHINESE_FONT_PROP
    )


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "node_same_cross_driver_decomposition.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 21. Figure 4:
#     Driver score vs support turnover
#
# This plot is useful for later Stage 6.3.
# No formal role classification is made here.
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        7,
    )
)


ax.scatter(
    node_summary[
        "support_turnover"
    ],
    node_summary[
        "driver_score"
    ],
    s=70,
)


for row in (
    node_summary.itertuples()
):

    label = (
        str(
            row.name
        )
        if HAS_NAME
        else row.stock
    )


    ax.annotate(
        label,
        (
            row.support_turnover,
            row.driver_score,
        ),
        xytext=(
            4,
            4,
        ),
        textcoords="offset points",
        fontproperties=CHINESE_FONT_PROP,
        fontsize=8,
    )


ax.set_xlabel(
    "Support Turnover = Lost + Gained"
)


ax.set_ylabel(
    "Structural Driver Score"
)


ax.set_title(
    "股票结构变化强度与支持集换边程度",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "driver_score_vs_support_turnover.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 22. Sanity checks
# ============================================================

print(
    "\n"
    + "=" * 78
)


print(
    "Sanity checks"
)


print(
    "=" * 78
)


print(
    f"Total network transition energy: "
    f"{total_transition_energy:.8f}"
)


print(
    f"2 × network transition energy: "
    f"{expected_total_node_score:.8f}"
)


print(
    f"Sum of node driver scores: "
    f"{total_node_driver_score:.8f}"
)


print(
    f"Sum of node driver shares: "
    f"{node_summary['driver_share'].sum():.8f}"
)


if not np.isclose(
    node_summary[
        "driver_share"
    ]
    .sum(),
    1.0,
    atol=1e-10,
):

    print(
        "WARNING:"
    )

    print(
        "Driver shares do not sum to one."
    )


# ============================================================
# 23. Console: Top structural drivers
# ============================================================

print(
    "\n"
    + "=" * 78
)


print(
    "Top structural-driver stocks"
)


print(
    "=" * 78
)


display_columns = [
    "driver_rank",
    "stock",
    "name",
    "industry",
    "driver_score",
    "driver_share",
    "same_driver_score",
    "cross_driver_score",
    "cross_driver_share",
    "support_turnover",
    "support_turnover_rate",
    "lost_degree",
    "gained_degree",
    "persistent_degree",
    "support_change_share",
    "persistent_change_share",
]


print(
    node_summary[
        display_columns
    ]
    .head(
        TOP_K_NODES
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 24. Console: top Cross-driven nodes
# ============================================================

print(
    "\n"
    + "=" * 78
)


print(
    "Top stocks by Cross-industry structural-change score"
)


print(
    "=" * 78
)


print(
    node_summary
    .sort_values(
        "cross_driver_score",
        ascending=False,
    )[
        [
            "stock",
            "name",
            "industry",
            "driver_score",
            "cross_driver_score",
            "cross_driver_share",
            "support_turnover",
        ]
    ]
    .head(
        TOP_K_NODES
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 25. Console: top support-turnover nodes
# ============================================================

print(
    "\n"
    + "=" * 78
)


print(
    "Top stocks by support turnover"
)


print(
    "=" * 78
)


print(
    node_summary
    .sort_values(
        [
            "support_turnover",
            "driver_score",
        ],
        ascending=[
            False,
            False,
        ],
    )[
        [
            "stock",
            "name",
            "industry",
            "support_turnover",
            "lost_degree",
            "gained_degree",
            "net_degree_change",
            "driver_score",
            "driver_share",
        ]
    ]
    .head(
        TOP_K_NODES
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 26. Finish
# ============================================================

print(
    "\n"
    + "=" * 78
)


print(
    "Stage 6.2 completed successfully"
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


main_outputs = [
    "node_structural_driver_summary.csv",
    "top_node_structural_drivers.csv",
    "top_driver_incident_edges.csv",
    "node_driver_ranking.png",
    "node_driver_share_ranking.png",
    "node_same_cross_driver_decomposition.png",
    "driver_score_vs_support_turnover.png",
]


for filename in main_outputs:

    print(
        f"  - {filename}"
    )
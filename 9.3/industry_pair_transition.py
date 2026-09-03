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
    / "stage6_1_industry_pair_transition"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Settings
# ============================================================

# Number of top industry pairs saved separately
TOP_K_INDUSTRY_PAIRS = 10


# Whether heatmap cells should display values
ANNOTATE_HEATMAP = True


# Heatmap figure size
HEATMAP_FIGSIZE = (
    10,
    8,
)


# ============================================================
# 3. Chinese font configuration
# ============================================================

def setup_chinese_font() -> tuple[FontProperties, str]:
    """
    Automatically find a Chinese-capable font.

    Priority:
        Microsoft YaHei
        SimHei
        DengXian
        SimSun
        Noto Sans CJK SC
        Source Han Sans SC

    Returns
    -------
    font_prop
        Matplotlib FontProperties object.

    font_name
        Name/description of the selected font.
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
    # First try Matplotlib-registered fonts
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
    # Windows direct font-file fallback
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


    print(
        "Chinese labels may not render correctly."
    )


    plt.rcParams[
        "axes.unicode_minus"
    ] = False


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
    Read CSV with several common encodings.
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
# 5. Check input file
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "Cannot find Stage-5 edge table:\n"
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
# 7. Basic data checks
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "Stage 6.1: Industry-pair Transition Decomposition"
)

print(
    "=" * 78
)


print(
    f"Input rows: "
    f"{len(edge_df)}"
)


# For 15 stocks we expect 105 possible pairs.
if len(edge_df) != 105:

    print(
        "WARNING:"
    )

    print(
        f"Expected 105 stock pairs for 15 stocks, "
        f"but found {len(edge_df)}."
    )


# ------------------------------------------------------------
# Convert numeric columns explicitly
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Clean industry labels
# ------------------------------------------------------------

edge_df[
    "industry_i"
] = (
    edge_df[
        "industry_i"
    ]
    .astype(str)
    .str.strip()
)


edge_df[
    "industry_j"
] = (
    edge_df[
        "industry_j"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# 8. Construct unordered industry pair
#
# Example:
#
#   电子 - 银行
#   银行 - 电子
#
# are treated as the same pair.
# ============================================================

def canonical_industry_pair(
    industry_i: str,
    industry_j: str,
) -> tuple[str, str]:
    """
    Return an unordered/canonical industry pair.
    """

    industries = sorted(
        [
            industry_i,
            industry_j,
        ]
    )


    return (
        industries[0],
        industries[1],
    )


industry_pairs = edge_df.apply(
    lambda row: canonical_industry_pair(
        row[
            "industry_i"
        ],
        row[
            "industry_j"
        ],
    ),
    axis=1,
)


edge_df[
    "industry_a"
] = [
    pair[0]
    for pair in industry_pairs
]


edge_df[
    "industry_b"
] = [
    pair[1]
    for pair in industry_pairs
]


edge_df[
    "industry_pair"
] = (
    edge_df[
        "industry_a"
    ]
    +
    " - "
    +
    edge_df[
        "industry_b"
    ]
)


# Same/Cross indicator recomputed directly from industry pair
edge_df[
    "industry_relation"
] = np.where(
    edge_df[
        "industry_a"
    ]
    ==
    edge_df[
        "industry_b"
    ],
    "Same",
    "Cross",
)


# ============================================================
# 9. Construct transition-energy quantities
#
# Total edge transition energy:
#
#     (Delta rho_ij)^2
#
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
# Support-change energy
#
# Lost / Gained edges only
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
# 10. Whole-network totals
# ============================================================

total_transition_energy = float(
    edge_df[
        "transition_energy"
    ]
    .sum()
)


total_support_change_energy = float(
    edge_df[
        "support_change_energy"
    ]
    .sum()
)


total_persistent_change_energy = float(
    edge_df[
        "persistent_change_energy"
    ]
    .sum()
)


if total_transition_energy <= 0:

    raise ValueError(
        "Total transition energy is zero. "
        "R3 and R4 appear identical."
    )


print(
    f"Total transition energy: "
    f"{total_transition_energy:.6f}"
)


print(
    f"Total transition intensity: "
    f"{np.sqrt(total_transition_energy):.6f}"
)


# ============================================================
# 11. Industry-pair aggregation
# ============================================================

summary_rows = []


grouped = edge_df.groupby(
    [
        "industry_a",
        "industry_b",
    ],
    sort=True,
)


for (
    industry_a,
    industry_b,
), group in grouped:

    possible_pairs = len(
        group
    )


    R3_edges = int(
        group[
            "edge_R3"
        ]
        .sum()
    )


    R4_edges = int(
        group[
            "edge_R4"
        ]
        .sum()
    )


    persistent = int(
        (
            group[
                "support_change"
            ]
            ==
            "Persistent"
        )
        .sum()
    )


    lost = int(
        (
            group[
                "support_change"
            ]
            ==
            "Lost"
        )
        .sum()
    )


    gained = int(
        (
            group[
                "support_change"
            ]
            ==
            "Gained"
        )
        .sum()
    )


    absent = int(
        (
            group[
                "support_change"
            ]
            ==
            "Absent"
        )
        .sum()
    )


    gross_change = (
        lost
        +
        gained
    )


    net_edge_change = (
        gained
        -
        lost
    )


    support_change_rate = (
        gross_change
        /
        possible_pairs

        if possible_pairs > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Jaccard support similarity
    # --------------------------------------------------------

    union_count = (
        persistent
        +
        lost
        +
        gained
    )


    jaccard = (
        persistent
        /
        union_count

        if union_count > 0
        else np.nan
    )


    # --------------------------------------------------------
    # R3 -> R4 edge retention
    # --------------------------------------------------------

    retention_from_R3 = (
        persistent
        /
        R3_edges

        if R3_edges > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Continuous transition energy
    # --------------------------------------------------------

    transition_energy = float(
        group[
            "transition_energy"
        ]
        .sum()
    )


    transition_intensity = float(
        np.sqrt(
            transition_energy
        )
    )


    energy_share = (
        transition_energy
        /
        total_transition_energy
    )


    # --------------------------------------------------------
    # Normalize by all possible pairs in this industry pair
    # --------------------------------------------------------

    mean_energy_per_pair = (
        transition_energy
        /
        possible_pairs

        if possible_pairs > 0
        else np.nan
    )


    mean_abs_delta_per_pair = float(
        group[
            "abs_delta_partial"
        ]
        .mean()
    )


    # --------------------------------------------------------
    # Support vs persistent-strength decomposition
    # --------------------------------------------------------

    support_energy = float(
        group[
            "support_change_energy"
        ]
        .sum()
    )


    persistent_energy = float(
        group[
            "persistent_change_energy"
        ]
        .sum()
    )


    support_energy_share_within_pair = (
        support_energy
        /
        transition_energy

        if transition_energy > 0
        else np.nan
    )


    persistent_energy_share_within_pair = (
        persistent_energy
        /
        transition_energy

        if transition_energy > 0
        else np.nan
    )


    # Contribution to WHOLE-network support-change energy
    support_energy_global_share = (
        support_energy
        /
        total_support_change_energy

        if total_support_change_energy > 0
        else np.nan
    )


    # Contribution to WHOLE-network persistent-change energy
    persistent_energy_global_share = (
        persistent_energy
        /
        total_persistent_change_energy

        if total_persistent_change_energy > 0
        else np.nan
    )


    relation = (
        "Same"
        if industry_a == industry_b
        else "Cross"
    )


    summary_rows.append(
        {
            "industry_a":
                industry_a,

            "industry_b":
                industry_b,

            "industry_pair":
                f"{industry_a} - {industry_b}",

            "relation":
                relation,

            "possible_pairs":
                possible_pairs,

            "R3_edges":
                R3_edges,

            "R4_edges":
                R4_edges,

            "persistent":
                persistent,

            "lost":
                lost,

            "gained":
                gained,

            "absent":
                absent,

            "gross_support_change":
                gross_change,

            "net_edge_change":
                net_edge_change,

            "support_change_rate":
                support_change_rate,

            "jaccard":
                jaccard,

            "retention_from_R3":
                retention_from_R3,

            "transition_energy":
                transition_energy,

            "transition_intensity":
                transition_intensity,

            "energy_share":
                energy_share,

            "mean_energy_per_pair":
                mean_energy_per_pair,

            "mean_abs_delta_per_pair":
                mean_abs_delta_per_pair,

            "support_change_energy":
                support_energy,

            "persistent_change_energy":
                persistent_energy,

            "support_energy_share_within_pair":
                support_energy_share_within_pair,

            "persistent_energy_share_within_pair":
                persistent_energy_share_within_pair,

            "support_energy_global_share":
                support_energy_global_share,

            "persistent_energy_global_share":
                persistent_energy_global_share,
        }
    )


industry_summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 12. Rank industry pairs
# ============================================================

industry_summary[
    "energy_rank"
] = (
    industry_summary[
        "transition_energy"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)


industry_summary[
    "mean_energy_rank"
] = (
    industry_summary[
        "mean_energy_per_pair"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)


industry_summary[
    "support_change_rank"
] = (
    industry_summary[
        "gross_support_change"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)


industry_summary = (
    industry_summary
    .sort_values(
        [
            "transition_energy",
            "gross_support_change",
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
# 13. Save main industry-pair summary
# ============================================================

MAIN_SUMMARY_FILE = (
    OUTPUT_DIR
    / "industry_pair_transition_summary.csv"
)


industry_summary.to_csv(
    MAIN_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. Save top industry pairs
# ============================================================

top_industry_pairs = (
    industry_summary
    .head(
        TOP_K_INDUSTRY_PAIRS
    )
    .copy()
)


top_industry_pairs.to_csv(
    OUTPUT_DIR
    / "top_industry_pair_transitions.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. Industry-level overall summary
#
# Aggregate each industry across ALL relationships involving
# that industry.
#
# Important:
# An A-B pair contributes to both industry A and industry B.
# Therefore this table is node-like attribution at the
# INDUSTRY level and shares need not sum to 1.
# ============================================================

industries = sorted(
    set(
        edge_df[
            "industry_i"
        ]
    )
    |
    set(
        edge_df[
            "industry_j"
        ]
    )
)


industry_driver_rows = []


for industry in industries:

    mask = (
        (
            edge_df[
                "industry_i"
            ]
            ==
            industry
        )
        |
        (
            edge_df[
                "industry_j"
            ]
            ==
            industry
        )
    )


    group = edge_df.loc[
        mask
    ]


    driver_energy = float(
        group[
            "transition_energy"
        ]
        .sum()
    )


    gross_change = int(
        group[
            "support_change"
        ]
        .isin(
            [
                "Lost",
                "Gained",
            ]
        )
        .sum()
    )


    industry_driver_rows.append(
        {
            "industry":
                industry,

            "incident_possible_pairs":
                len(group),

            "transition_energy":
                driver_energy,

            "transition_intensity":
                np.sqrt(
                    driver_energy
                ),

            "incident_support_changes":
                gross_change,
        }
    )


industry_driver_summary = pd.DataFrame(
    industry_driver_rows
)


industry_driver_summary = (
    industry_driver_summary
    .sort_values(
        "transition_energy",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


industry_driver_summary.to_csv(
    OUTPUT_DIR
    / "industry_level_transition_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. Matrix construction helper
# ============================================================

def build_symmetric_matrix(
    summary_df: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Build a symmetric industry-by-industry matrix from the
    unordered industry-pair summary.
    """

    matrix = pd.DataFrame(
        0.0,
        index=industries,
        columns=industries,
    )


    for row in (
        summary_df.itertuples()
    ):

        a = row.industry_a
        b = row.industry_b

        value = float(
            getattr(
                row,
                value_column,
            )
        )


        matrix.at[
            a,
            b,
        ] = value


        matrix.at[
            b,
            a,
        ] = value


    return matrix


# ============================================================
# 17. Build and save matrices
# ============================================================

energy_matrix = build_symmetric_matrix(
    industry_summary,
    "transition_energy",
)


energy_share_matrix = build_symmetric_matrix(
    industry_summary,
    "energy_share",
)


mean_energy_matrix = build_symmetric_matrix(
    industry_summary,
    "mean_energy_per_pair",
)


support_change_rate_matrix = build_symmetric_matrix(
    industry_summary,
    "support_change_rate",
)


gross_change_matrix = build_symmetric_matrix(
    industry_summary,
    "gross_support_change",
)


energy_matrix.to_csv(
    OUTPUT_DIR
    / "industry_pair_transition_energy_matrix.csv",

    encoding="utf-8-sig",
)


energy_share_matrix.to_csv(
    OUTPUT_DIR
    / "industry_pair_energy_share_matrix.csv",

    encoding="utf-8-sig",
)


mean_energy_matrix.to_csv(
    OUTPUT_DIR
    / "industry_pair_mean_energy_matrix.csv",

    encoding="utf-8-sig",
)


support_change_rate_matrix.to_csv(
    OUTPUT_DIR
    / "industry_pair_support_change_rate_matrix.csv",

    encoding="utf-8-sig",
)


gross_change_matrix.to_csv(
    OUTPUT_DIR
    / "industry_pair_gross_support_change_matrix.csv",

    encoding="utf-8-sig",
)


# ============================================================
# 18. General heatmap helper
# ============================================================

def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_file: Path,
    value_format: str = ".3f",
) -> None:
    """
    Draw a heatmap using matplotlib only.
    """

    values = matrix.to_numpy(
        dtype=float,
        copy=True,
    )


    fig, ax = plt.subplots(
        figsize=HEATMAP_FIGSIZE
    )


    image = ax.imshow(
        values,
        aspect="auto",
    )


    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            len(
                matrix.columns
            )
        )
    )


    ax.set_yticks(
        np.arange(
            len(
                matrix.index
            )
        )
    )


    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
        fontproperties=CHINESE_FONT_PROP,
        fontsize=9,
    )


    ax.set_yticklabels(
        matrix.index,
        fontproperties=CHINESE_FONT_PROP,
        fontsize=9,
    )


    ax.set_title(
        title,
        fontproperties=CHINESE_FONT_PROP,
        fontsize=14,
        pad=15,
    )


    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )


    colorbar.set_label(
        colorbar_label,
        fontproperties=CHINESE_FONT_PROP,
    )


    # --------------------------------------------------------
    # Cell annotations
    # --------------------------------------------------------

    if ANNOTATE_HEATMAP:

        max_value = (
            np.nanmax(
                values
            )
            if values.size > 0
            else 0
        )


        threshold = (
            max_value
            *
            0.5
        )


        for i in range(
            values.shape[0]
        ):

            for j in range(
                values.shape[1]
            ):

                value = values[
                    i,
                    j,
                ]


                if np.isnan(
                    value
                ):

                    continue


                text = format(
                    value,
                    value_format,
                )


                # No custom colors are required for analysis,
                # use default matplotlib text appearance.
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                )


    plt.tight_layout()


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


# ============================================================
# 19. Heatmap 1:
#     Total transition energy
# ============================================================

plot_heatmap(
    matrix=energy_matrix,
    title=(
        "R3 → R4 行业组合结构转换能量"
    ),
    colorbar_label=(
        "Transition Energy"
    ),
    output_file=(
        OUTPUT_DIR
        / "industry_pair_transition_energy_heatmap.png"
    ),
    value_format=".3f",
)


# ============================================================
# 20. Heatmap 2:
#     Energy share
# ============================================================

plot_heatmap(
    matrix=energy_share_matrix,
    title=(
        "R3 → R4 行业组合对全网结构变化的贡献比例"
    ),
    colorbar_label=(
        "Energy Share"
    ),
    output_file=(
        OUTPUT_DIR
        / "industry_pair_energy_share_heatmap.png"
    ),
    value_format=".1%",
)


# ============================================================
# 21. Heatmap 3:
#     Mean transition energy per possible pair
# ============================================================

plot_heatmap(
    matrix=mean_energy_matrix,
    title=(
        "R3 → R4 行业组合平均每股票对结构变化强度"
    ),
    colorbar_label=(
        "Mean Energy per Pair"
    ),
    output_file=(
        OUTPUT_DIR
        / "industry_pair_mean_energy_heatmap.png"
    ),
    value_format=".3f",
)


# ============================================================
# 22. Heatmap 4:
#     Support-change rate
# ============================================================

plot_heatmap(
    matrix=support_change_rate_matrix,
    title=(
        "R3 → R4 行业组合边支持集变化率"
    ),
    colorbar_label=(
        "Support Change Rate"
    ),
    output_file=(
        OUTPUT_DIR
        / "industry_pair_support_change_rate_heatmap.png"
    ),
    value_format=".1%",
)


# ============================================================
# 23. Bar chart:
#     Top industry-pair energy shares
# ============================================================

top_plot = (
    industry_summary
    .head(
        TOP_K_INDUSTRY_PAIRS
    )
    .sort_values(
        "energy_share",
        ascending=True,
    )
    .copy()
)


fig, ax = plt.subplots(
    figsize=(
        9,
        6,
    )
)


ax.barh(
    top_plot[
        "industry_pair"
    ],
    top_plot[
        "energy_share"
    ],
)


ax.set_xlabel(
    "全网结构变化贡献比例",
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_ylabel(
    "行业组合",
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_title(
    (
        f"R3 → R4 结构变化贡献最高的 "
        f"{TOP_K_INDUSTRY_PAIRS} 个行业组合"
    ),
    fontproperties=CHINESE_FONT_PROP,
    fontsize=13,
)


for tick in ax.get_yticklabels():

    tick.set_fontproperties(
        CHINESE_FONT_PROP
    )


# Percentage labels
for i, value in enumerate(
    top_plot[
        "energy_share"
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
    / "top_industry_pair_energy_share.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 24. Sanity checks
# ============================================================

energy_share_sum = float(
    industry_summary[
        "energy_share"
    ]
    .sum()
)


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
    f"Sum of industry-pair energy shares: "
    f"{energy_share_sum:.8f}"
)


if not np.isclose(
    energy_share_sum,
    1.0,
    atol=1e-8,
):

    print(
        "WARNING: energy shares do not sum to 1."
    )


reconstructed_energy = float(
    industry_summary[
        "transition_energy"
    ]
    .sum()
)


print(
    f"Original total transition energy: "
    f"{total_transition_energy:.8f}"
)


print(
    f"Industry-pair reconstructed energy: "
    f"{reconstructed_energy:.8f}"
)


if not np.isclose(
    reconstructed_energy,
    total_transition_energy,
    atol=1e-10,
):

    print(
        "WARNING: industry-pair aggregation "
        "does not reconstruct total energy."
    )


# ============================================================
# 25. Console: Top results
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "Top industry pairs by transition energy"
)

print(
    "=" * 78
)


display_columns = [
    "industry_pair",
    "relation",
    "possible_pairs",
    "R3_edges",
    "R4_edges",
    "lost",
    "gained",
    "persistent",
    "gross_support_change",
    "support_change_rate",
    "transition_energy",
    "energy_share",
    "mean_energy_per_pair",
]


print(
    industry_summary[
        display_columns
    ]
    .head(
        TOP_K_INDUSTRY_PAIRS
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 26. Console: Top pairs by per-pair intensity
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "Top industry pairs by mean energy per possible pair"
)

print(
    "=" * 78
)


print(
    industry_summary
    .sort_values(
        "mean_energy_per_pair",
        ascending=False,
    )[
        [
            "industry_pair",
            "possible_pairs",
            "transition_energy",
            "energy_share",
            "mean_energy_per_pair",
            "gross_support_change",
        ]
    ]
    .head(
        TOP_K_INDUSTRY_PAIRS
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 27. Console: Top support-turnover industry pairs
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "Top industry pairs by support-change rate"
)

print(
    "=" * 78
)


print(
    industry_summary
    .sort_values(
        [
            "support_change_rate",
            "gross_support_change",
        ],
        ascending=[
            False,
            False,
        ],
    )[
        [
            "industry_pair",
            "possible_pairs",
            "lost",
            "gained",
            "gross_support_change",
            "support_change_rate",
            "transition_energy",
            "energy_share",
        ]
    ]
    .head(
        TOP_K_INDUSTRY_PAIRS
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 28. Finish
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "Stage 6.1 completed successfully"
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
    "industry_pair_transition_summary.csv",
    "top_industry_pair_transitions.csv",
    "industry_level_transition_summary.csv",
    "industry_pair_transition_energy_matrix.csv",
    "industry_pair_energy_share_matrix.csv",
    "industry_pair_mean_energy_matrix.csv",
    "industry_pair_support_change_rate_matrix.csv",
    "industry_pair_gross_support_change_matrix.csv",
    "industry_pair_transition_energy_heatmap.png",
    "industry_pair_energy_share_heatmap.png",
    "industry_pair_mean_energy_heatmap.png",
    "industry_pair_support_change_rate_heatmap.png",
    "top_industry_pair_energy_share.png",
]


for filename in main_outputs:

    print(
        f"  - {filename}"
    )
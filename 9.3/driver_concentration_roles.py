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
    / "stage6_2_node_structural_driver"
    / "node_structural_driver_summary.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "stage6_3_driver_concentration_roles"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Settings
# ============================================================

# Top-K stocks treated as "Major Drivers".
#
# This is a descriptive reporting rule rather than
# a statistical significance threshold.
MAJOR_DRIVER_TOP_K = 5


# Mechanism threshold:
#
# support_change_share >= 0.60
#     -> Support-rewiring
#
# support_change_share <= 0.40
#     -> Persistent-strength repricing
#
# 0.40 < share < 0.60
#     -> Mixed
MECHANISM_HIGH_THRESHOLD = 0.60
MECHANISM_LOW_THRESHOLD = 0.40


# Same/Cross orientation threshold:
#
# cross_driver_share >= 0.60
#     -> Cross-driven
#
# cross_driver_share <= 0.40
#     -> Same-driven
#
# otherwise:
#     -> Mixed Same/Cross
ORIENTATION_HIGH_THRESHOLD = 0.60
ORIENTATION_LOW_THRESHOLD = 0.40


# High-turnover threshold based on empirical upper third.
TURNOVER_QUANTILE = 2 / 3


# Number of stocks shown in ranking plots.
TOP_K_PLOT = 10


# ============================================================
# 3. Chinese font configuration
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
                f"Chinese font selected: {font_name}"
            )


            return (
                font_prop,
                font_name,
            )


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


            return (
                font_prop,
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
    **kwargs,
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
# 5. Load Stage-6.2 results
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "Cannot find Stage-6.2 node summary:\n"
        f"{INPUT_FILE}"
    )


df = read_csv_with_encoding_fallback(
    INPUT_FILE
)


required_columns = [
    "stock",
    "name",
    "industry",
    "driver_score",
    "driver_share",
    "driver_rank",
    "same_driver_score",
    "cross_driver_score",
    "same_driver_share",
    "cross_driver_share",
    "support_turnover",
    "support_turnover_rate",
    "support_change_score",
    "persistent_change_score",
    "support_change_share",
    "persistent_change_share",
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        f"{missing_columns}\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


# ============================================================
# 6. Clean / sort
# ============================================================

numeric_columns = [
    "driver_score",
    "driver_share",
    "driver_rank",
    "same_driver_score",
    "cross_driver_score",
    "same_driver_share",
    "cross_driver_share",
    "support_turnover",
    "support_turnover_rate",
    "support_change_score",
    "persistent_change_score",
    "support_change_share",
    "persistent_change_share",
]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="raise",
    )


df = (
    df
    .sort_values(
        "driver_score",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


n_nodes = len(
    df
)


print(
    "\n"
    + "=" * 78
)

print(
    "Stage 6.3: Driver Concentration & Structural Roles"
)

print(
    "=" * 78
)


print(
    f"Number of stocks: {n_nodes}"
)


# ============================================================
# 7. Sanity check
# ============================================================

driver_share_sum = float(
    df[
        "driver_share"
    ]
    .sum()
)


print(
    f"Sum of Driver Shares: "
    f"{driver_share_sum:.8f}"
)


if not np.isclose(
    driver_share_sum,
    1.0,
    atol=1e-8,
):

    raise ValueError(
        "Driver shares do not sum to 1."
    )


# ============================================================
# 8. Concentration ratios
#
# CR_k = sum of Top-k Driver Shares
# ============================================================

def concentration_ratio(
    shares: pd.Series,
    k: int,
) -> float:

    k = min(
        k,
        len(shares),
    )


    return float(
        shares
        .iloc[:k]
        .sum()
    )


CR1 = concentration_ratio(
    df[
        "driver_share"
    ],
    1,
)


CR2 = concentration_ratio(
    df[
        "driver_share"
    ],
    2,
)


CR3 = concentration_ratio(
    df[
        "driver_share"
    ],
    3,
)


CR5 = concentration_ratio(
    df[
        "driver_share"
    ],
    5,
)


CR10 = concentration_ratio(
    df[
        "driver_share"
    ],
    10,
)


# ============================================================
# 9. HHI
#
# HHI = sum_i s_i^2
#
# Equal-share benchmark:
#
#     HHI_equal = 1 / n
#
# Effective number:
#
#     N_eff = 1 / HHI
# ============================================================

shares = (
    df[
        "driver_share"
    ]
    .to_numpy(
        dtype=float,
        copy=True,
    )
)


HHI = float(
    np.sum(
        shares
        ** 2
    )
)


HHI_equal = (
    1.0
    /
    n_nodes
)


effective_nodes_HHI = (
    1.0
    /
    HHI
)


# ============================================================
# 10. Entropy concentration
#
# H = -sum s_i log(s_i)
#
# normalized entropy:
#
# H / log(n)
#
# = 1 under equal contribution.
# ============================================================

positive_shares = shares[
    shares > 0
]


entropy = float(
    -np.sum(
        positive_shares
        *
        np.log(
            positive_shares
        )
    )
)


normalized_entropy = (
    entropy
    /
    np.log(
        n_nodes
    )
)


effective_nodes_entropy = float(
    np.exp(
        entropy
    )
)


# ============================================================
# 11. Gini coefficient
# ============================================================

def gini_coefficient(
    values: np.ndarray,
) -> float:

    x = np.asarray(
        values,
        dtype=float,
    )


    if np.any(
        x < 0
    ):

        raise ValueError(
            "Gini requires nonnegative values."
        )


    if np.allclose(
        x.sum(),
        0,
    ):

        return 0.0


    x = np.sort(
        x
    )


    n = len(
        x
    )


    index = np.arange(
        1,
        n + 1,
    )


    gini = (
        2
        *
        np.sum(
            index
            *
            x
        )
        /
        (
            n
            *
            np.sum(
                x
            )
        )
        -
        (
            n + 1
        )
        /
        n
    )


    return float(
        gini
    )


GINI = gini_coefficient(
    shares
)


# ============================================================
# 12. Concentration summary
# ============================================================

concentration_summary = pd.DataFrame(
    [
        {
            "n_nodes":
                n_nodes,

            "CR1":
                CR1,

            "CR2":
                CR2,

            "CR3":
                CR3,

            "CR5":
                CR5,

            "CR10":
                CR10,

            "HHI":
                HHI,

            "HHI_equal_benchmark":
                HHI_equal,

            "effective_nodes_HHI":
                effective_nodes_HHI,

            "entropy":
                entropy,

            "normalized_entropy":
                normalized_entropy,

            "effective_nodes_entropy":
                effective_nodes_entropy,

            "gini":
                GINI,
        }
    ]
)


concentration_summary.to_csv(
    OUTPUT_DIR
    / "driver_concentration_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 13. Cumulative concentration curve
# ============================================================

concentration_curve = (
    df[
        [
            "driver_rank",
            "stock",
            "name",
            "industry",
            "driver_share",
        ]
    ]
    .copy()
)


concentration_curve[
    "cumulative_driver_share"
] = (
    concentration_curve[
        "driver_share"
    ]
    .cumsum()
)


concentration_curve[
    "equal_share_cumulative"
] = (
    np.arange(
        1,
        n_nodes + 1,
    )
    /
    n_nodes
)


concentration_curve.to_csv(
    OUTPUT_DIR
    / "driver_concentration_curve.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. Role classification
#
# IMPORTANT:
#
# Structural role should NOT be classified only by
# Driver Score and Turnover.
#
# Stage 6.2 already showed:
#
# high Driver Score + low Turnover
#
# can still be support-rewiring driven if only a few
# very large edges enter/exit.
#
# Therefore:
#
# 1. Contribution tier:
#    Major / Secondary
#
# 2. Same/Cross orientation
#
# 3. Mechanism:
#    Support-rewiring / Persistent-repricing / Mixed
#
# 4. Turnover intensity:
#    High / Lower
# ============================================================


# ------------------------------------------------------------
# Contribution tier
# ------------------------------------------------------------

df[
    "contribution_tier"
] = np.where(
    df[
        "driver_rank"
    ]
    <=
    MAJOR_DRIVER_TOP_K,
    "Major Driver",
    "Secondary Driver",
)


# ------------------------------------------------------------
# Same / Cross orientation
# ------------------------------------------------------------

def classify_orientation(
    cross_share: float,
) -> str:

    if cross_share >= ORIENTATION_HIGH_THRESHOLD:

        return "Cross-driven"


    if cross_share <= ORIENTATION_LOW_THRESHOLD:

        return "Same-driven"


    return "Mixed Same/Cross"


df[
    "orientation_role"
] = (
    df[
        "cross_driver_share"
    ]
    .apply(
        classify_orientation
    )
)


# ------------------------------------------------------------
# Structural-change mechanism
# ------------------------------------------------------------

def classify_mechanism(
    support_share: float,
) -> str:

    if support_share >= MECHANISM_HIGH_THRESHOLD:

        return "Support-rewiring"


    if support_share <= MECHANISM_LOW_THRESHOLD:

        return "Persistent-strength repricing"


    return "Mixed support/strength"


df[
    "mechanism_role"
] = (
    df[
        "support_change_share"
    ]
    .apply(
        classify_mechanism
    )
)


# ------------------------------------------------------------
# Turnover tier
#
# Empirical upper-third threshold.
# ------------------------------------------------------------

turnover_threshold = float(
    df[
        "support_turnover"
    ]
    .quantile(
        TURNOVER_QUANTILE
    )
)


df[
    "turnover_tier"
] = np.where(
    df[
        "support_turnover"
    ]
    >=
    turnover_threshold,
    "High-turnover",
    "Lower-turnover",
)


# ------------------------------------------------------------
# Composite descriptive role
# ------------------------------------------------------------

df[
    "structural_role"
] = (
    df[
        "contribution_tier"
    ]
    +
    " | "
    +
    df[
        "orientation_role"
    ]
    +
    " | "
    +
    df[
        "mechanism_role"
    ]
    +
    " | "
    +
    df[
        "turnover_tier"
    ]
)


# ============================================================
# 15. Save structural-role table
# ============================================================

role_columns = [
    "driver_rank",
    "stock",
    "name",
    "industry",
    "driver_score",
    "driver_share",
    "support_turnover",
    "support_turnover_rate",
    "same_driver_share",
    "cross_driver_share",
    "support_change_share",
    "persistent_change_share",
    "contribution_tier",
    "orientation_role",
    "mechanism_role",
    "turnover_tier",
    "structural_role",
]


node_roles = (
    df[
        role_columns
    ]
    .copy()
)


node_roles.to_csv(
    OUTPUT_DIR
    / "node_structural_roles.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. Major-driver summary
# ============================================================

major_drivers = (
    node_roles.loc[
        node_roles[
            "contribution_tier"
        ]
        ==
        "Major Driver"
    ]
    .copy()
)


major_drivers.to_csv(
    OUTPUT_DIR
    / "major_structural_drivers.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. Industry-level node-attributed driver contribution
#
# Sum of node driver shares by industry.
#
# Because driver_share already sums to one,
# industry shares also sum to one.
# ============================================================

industry_driver_summary = (
    df
    .groupby(
        "industry",
        observed=False,
    )
    .agg(
        n_stocks=(
            "stock",
            "size",
        ),

        driver_score=(
            "driver_score",
            "sum",
        ),

        driver_share=(
            "driver_share",
            "sum",
        ),

        mean_driver_share=(
            "driver_share",
            "mean",
        ),

        mean_cross_driver_share=(
            "cross_driver_share",
            "mean",
        ),

        total_support_turnover=(
            "support_turnover",
            "sum",
        ),
    )
    .reset_index()
    .sort_values(
        "driver_share",
        ascending=False,
    )
)


industry_driver_summary.to_csv(
    OUTPUT_DIR
    / "industry_node_driver_concentration.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 18. Plot 1:
#     Cumulative driver concentration
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        6,
    )
)


ax.plot(
    concentration_curve[
        "driver_rank"
    ],
    concentration_curve[
        "cumulative_driver_share"
    ],
    marker="o",
    label="Observed",
)


ax.plot(
    concentration_curve[
        "driver_rank"
    ],
    concentration_curve[
        "equal_share_cumulative"
    ],
    linestyle="--",
    label="Equal-share benchmark",
)


ax.set_xlabel(
    "Driver Rank"
)


ax.set_ylabel(
    "Cumulative Driver Share"
)


ax.set_title(
    "R3 → R4 节点结构变化贡献集中度",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


ax.set_ylim(
    0,
    1.05,
)


ax.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "driver_concentration_curve.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 19. Plot 2:
#     Driver score vs support turnover
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        7,
    )
)


ax.scatter(
    df[
        "support_turnover"
    ],
    df[
        "driver_score"
    ],
    s=80,
)


for row in (
    df.itertuples()
):

    ax.annotate(
        str(
            row.name
        ),
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


ax.axvline(
    turnover_threshold,
    linestyle="--",
)


driver_major_threshold = float(
    df.loc[
        df[
            "driver_rank"
        ]
        ==
        MAJOR_DRIVER_TOP_K,
        "driver_score",
    ]
    .iloc[0]
)


ax.axhline(
    driver_major_threshold,
    linestyle="--",
)


ax.set_xlabel(
    "Support Turnover = Lost + Gained"
)


ax.set_ylabel(
    "Structural Driver Score"
)


ax.set_title(
    "结构变化强度与边支持集换边程度",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "driver_score_vs_turnover_roles.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 20. Plot 3:
#     Mechanism decomposition for Top-K drivers
# ============================================================

plot_df = (
    df
    .head(
        TOP_K_PLOT
    )
    .sort_values(
        "driver_score",
        ascending=True,
    )
    .copy()
)


plot_df[
    "label"
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


y = np.arange(
    len(
        plot_df
    )
)


fig, ax = plt.subplots(
    figsize=(
        10,
        7,
    )
)


ax.barh(
    y,
    plot_df[
        "support_change_score"
    ],
    label="Support change",
)


ax.barh(
    y,
    plot_df[
        "persistent_change_score"
    ],
    left=plot_df[
        "support_change_score"
    ],
    label="Persistent strength change",
)


ax.set_yticks(
    y
)


ax.set_yticklabels(
    plot_df[
        "label"
    ],
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_xlabel(
    "Driver Score"
)


ax.set_title(
    "主要结构变化节点的变化机制分解",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


ax.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "top_driver_mechanism_decomposition.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 21. Plot 4:
#     Same / Cross orientation
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7,
    )
)


ax.barh(
    y,
    plot_df[
        "same_driver_score"
    ],
    label="同行业",
)


ax.barh(
    y,
    plot_df[
        "cross_driver_score"
    ],
    left=plot_df[
        "same_driver_score"
    ],
    label="跨行业",
)


ax.set_yticks(
    y
)


ax.set_yticklabels(
    plot_df[
        "label"
    ],
    fontproperties=CHINESE_FONT_PROP,
)


ax.set_xlabel(
    "Driver Score"
)


ax.set_title(
    "主要结构变化节点的同行业 / 跨行业贡献",
    fontproperties=CHINESE_FONT_PROP,
    fontsize=14,
)


legend = ax.legend()


for text in legend.get_texts():

    text.set_fontproperties(
        CHINESE_FONT_PROP
    )


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "top_driver_same_cross_decomposition.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 22. Console results
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "Driver concentration"
)

print(
    "=" * 78
)


print(
    f"CR1  : {CR1:.2%}"
)


print(
    f"CR2  : {CR2:.2%}"
)


print(
    f"CR3  : {CR3:.2%}"
)


print(
    f"CR5  : {CR5:.2%}"
)


print(
    f"CR10 : {CR10:.2%}"
)


print(
    f"\nHHI                   : "
    f"{HHI:.6f}"
)


print(
    f"Equal-share HHI       : "
    f"{HHI_equal:.6f}"
)


print(
    f"Effective nodes (HHI) : "
    f"{effective_nodes_HHI:.2f}"
)


print(
    f"Normalized entropy    : "
    f"{normalized_entropy:.4f}"
)


print(
    f"Effective nodes (Entropy): "
    f"{effective_nodes_entropy:.2f}"
)


print(
    f"Gini coefficient      : "
    f"{GINI:.4f}"
)


print(
    f"\nHigh-turnover threshold: "
    f"{turnover_threshold:.2f}"
)


print(
    "\n"
    + "=" * 78
)

print(
    "Major structural drivers"
)

print(
    "=" * 78
)


print(
    major_drivers[
        [
            "driver_rank",
            "stock",
            "name",
            "industry",
            "driver_share",
            "support_turnover",
            "orientation_role",
            "mechanism_role",
            "turnover_tier",
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
    "Industry-level node-attributed contribution"
)

print(
    "=" * 78
)


print(
    industry_driver_summary.to_string(
        index=False
    )
)


# ============================================================
# 23. Finish
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "Stage 6.3 completed successfully"
)

print(
    "=" * 78
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
    "driver_concentration_summary.csv",
    "driver_concentration_curve.csv",
    "node_structural_roles.csv",
    "major_structural_drivers.csv",
    "industry_node_driver_concentration.csv",
    "driver_concentration_curve.png",
    "driver_score_vs_turnover_roles.png",
    "top_driver_mechanism_decomposition.png",
    "top_driver_same_cross_decomposition.png",
]


for filename in main_outputs:

    print(
        f"  - {filename}"
    )
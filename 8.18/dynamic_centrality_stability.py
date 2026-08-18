from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


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
        font.name
        for font in font_manager.fontManager.ttflist
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
        "警告：未找到常见中文字体，"
        "中文股票名称可能显示异常。"
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


# ------------------------------------------------------------
# 输入
# ------------------------------------------------------------

NODE_METRICS_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_node_metrics.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ------------------------------------------------------------
# 输出
# ------------------------------------------------------------

WINDOW_LEVEL_FILE = (
    PROCESSED_DIR
    / "dynamic_centrality_window_metrics_with_ranks.csv"
)

STABILITY_FILE = (
    PROCESSED_DIR
    / "dynamic_centrality_stability.csv"
)

ROLE_SUMMARY_FILE = (
    PROCESSED_DIR
    / "dynamic_centrality_role_summary.csv"
)

STABLE_CENTRAL_FILE = (
    PROCESSED_DIR
    / "stable_central_nodes.csv"
)

DYNAMIC_CENTRAL_FILE = (
    PROCESSED_DIR
    / "dynamic_central_nodes.csv"
)


# ============================================================
# 2. 工具函数
# ============================================================

def normalize_code(x):

    s = str(
        x
    ).strip()

    if s.endswith(".0"):
        s = s[:-2]

    match = re.search(
        r"(\d{6})",
        s
    )

    if match:
        return match.group(1)

    digits = "".join(
        ch
        for ch in s
        if ch.isdigit()
    )

    if digits:
        return digits.zfill(6)

    return s


def find_column(
    columns,
    candidates,
    required=True
):

    for col in candidates:

        if col in columns:
            return col

    if required:

        raise ValueError(
            "无法找到字段。候选字段："
            f"{candidates}\n"
            f"当前字段：{list(columns)}"
        )

    return None


def safe_cv(
    mean_value,
    sd_value
):

    if (
        pd.isna(mean_value)
        or
        abs(mean_value) < 1e-12
    ):

        return np.nan

    return (
        sd_value
        /
        abs(mean_value)
    )


# ============================================================
# 3. 读取Node Metrics
# ============================================================

raw = pd.read_csv(
    NODE_METRICS_FILE,
    dtype=str
)


# ============================================================
# 4. 自动识别字段
# ============================================================

code_col = find_column(
    raw.columns,
    [
        "stock_code",
        "code",
        "stock",
        "symbol"
    ]
)


name_col = find_column(
    raw.columns,
    [
        "stock_name",
        "name"
    ],
    required=False
)


industry_col = find_column(
    raw.columns,
    [
        "industry",
        "industry_name"
    ],
    required=False
)


window_col = find_column(
    raw.columns,
    [
        "window_id",
        "rolling_window_id"
    ]
)


date_col = find_column(
    raw.columns,
    [
        "network_date",
        "window_end",
        "date"
    ]
)


degree_col = find_column(
    raw.columns,
    [
        "degree",
        "Degree"
    ]
)


strength_col = find_column(
    raw.columns,
    [
        "strength",
        "Strength"
    ]
)


# ============================================================
# 5. 统一字段名称
# ============================================================

rename_dict = {
    code_col:
        "code",

    window_col:
        "window_id",

    date_col:
        "network_date",

    degree_col:
        "degree",

    strength_col:
        "strength"
}


if name_col is not None:

    rename_dict[
        name_col
    ] = "name"


if industry_col is not None:

    rename_dict[
        industry_col
    ] = "industry"


df = raw.rename(
    columns=rename_dict
).copy()


# ============================================================
# 6. 类型转换
# ============================================================

df["code"] = (
    df["code"]
    .apply(normalize_code)
)


df["window_id"] = (
    pd.to_numeric(
        df["window_id"],
        errors="raise"
    )
    .astype(int)
)


df["network_date"] = (
    pd.to_datetime(
        df["network_date"]
    )
)


df["degree"] = (
    pd.to_numeric(
        df["degree"],
        errors="raise"
    )
)


df["strength"] = (
    pd.to_numeric(
        df["strength"],
        errors="raise"
    )
)


# ============================================================
# 7. 使用stock_info重新确认名称和行业
# ============================================================

if STOCK_INFO_FILE.exists():

    stock_info = pd.read_csv(
        STOCK_INFO_FILE,
        dtype=str
    )


    info_code_col = find_column(
        stock_info.columns,
        [
            "code",
            "stock_code",
            "symbol",
            "ts_code"
        ]
    )


    info_name_col = find_column(
        stock_info.columns,
        [
            "name",
            "stock_name"
        ]
    )


    info_industry_col = find_column(
        stock_info.columns,
        [
            "industry",
            "industry_name"
        ]
    )


    stock_info = (
        stock_info[
            [
                info_code_col,
                info_name_col,
                info_industry_col
            ]
        ]
        .rename(
            columns={
                info_code_col:
                    "code",

                info_name_col:
                    "name_meta",

                info_industry_col:
                    "industry_meta"
            }
        )
        .copy()
    )


    stock_info["code"] = (
        stock_info["code"]
        .apply(normalize_code)
    )


    stock_info = (
        stock_info
        .drop_duplicates(
            subset="code"
        )
    )


    df = df.merge(
        stock_info,

        on="code",

        how="left"
    )


    # 始终优先使用权威metadata
    df["name"] = (
        df["name_meta"]
    )


    df["industry"] = (
        df["industry_meta"]
    )


    df = df.drop(
        columns=[
            "name_meta",
            "industry_meta"
        ]
    )


else:

    if (
        "name" not in df.columns
        or
        "industry" not in df.columns
    ):

        raise ValueError(
            "Node Metrics中缺少名称/行业，"
            "同时stock_info.csv不存在。"
        )


# ============================================================
# 8. 基础检查
# ============================================================

if (
    df["name"]
    .isna()
    .any()
):

    raise ValueError(
        "存在股票名称缺失。"
    )


if (
    df["industry"]
    .isna()
    .any()
):

    raise ValueError(
        "存在股票行业缺失。"
    )


# 检查每个window-stock是否唯一
duplicate_check = (
    df
    .duplicated(
        subset=[
            "window_id",
            "code"
        ]
    )
)


if duplicate_check.any():

    bad = df[
        duplicate_check
    ]

    raise ValueError(
        "存在同一window中同一股票重复记录：\n"
        f"{bad}"
    )


# ============================================================
# 9. 排序
# ============================================================

df = (
    df
    .sort_values(
        [
            "window_id",
            "code"
        ]
    )
    .reset_index(
        drop=True
    )
)


TOTAL_WINDOWS = (
    df[
        "window_id"
    ]
    .nunique()
)


N_STOCKS = (
    df[
        "code"
    ]
    .nunique()
)


# 取前1/3作为Top-central
TOP_K = int(
    np.ceil(
        N_STOCKS
        /
        3
    )
)


print(
    "\n总Rolling Windows：",
    TOTAL_WINDOWS
)


print(
    "股票数量：",
    N_STOCKS
)


print(
    "Top-central定义：每个窗口排名前",
    TOP_K
)


# ============================================================
# 10. 检查每只股票窗口数量是否一致
# ============================================================

window_counts = (
    df
    .groupby("code")[
        "window_id"
    ]
    .nunique()
)


bad_counts = window_counts[
    window_counts
    !=
    TOTAL_WINDOWS
]


if len(
    bad_counts
) > 0:

    print(
        bad_counts
    )

    raise ValueError(
        "部分股票缺少Rolling Window记录。"
    )


# ============================================================
# 11. 每个窗口内计算Degree / Strength排名
#
# rank = 1代表当前窗口最中心
# ============================================================

df[
    "degree_rank"
] = (
    df
    .groupby(
        "window_id"
    )[
        "degree"
    ]
    .rank(
        method="average",
        ascending=False
    )
)


df[
    "strength_rank"
] = (
    df
    .groupby(
        "window_id"
    )[
        "strength"
    ]
    .rank(
        method="average",
        ascending=False
    )
)


# ============================================================
# 12. 每个窗口是否属于Top-central
# ============================================================

df[
    "top_degree"
] = (
    df[
        "degree_rank"
    ]
    <=
    TOP_K
)


df[
    "top_strength"
] = (
    df[
        "strength_rank"
    ]
    <=
    TOP_K
)


# ============================================================
# 13. 保存Window-level结果
# ============================================================

df.to_csv(
    WINDOW_LEVEL_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. Node-level汇总
# ============================================================

node_rows = []


for code, group in df.groupby(
    "code"
):

    group = (
        group
        .sort_values(
            "window_id"
        )
    )


    # --------------------------------------------------------
    # Degree
    # --------------------------------------------------------

    mean_degree = (
        group[
            "degree"
        ]
        .mean()
    )


    sd_degree = (
        group[
            "degree"
        ]
        .std(
            ddof=1
        )
    )


    cv_degree = safe_cv(
        mean_degree,
        sd_degree
    )


    min_degree = (
        group[
            "degree"
        ]
        .min()
    )


    max_degree = (
        group[
            "degree"
        ]
        .max()
    )


    range_degree = (
        max_degree
        -
        min_degree
    )


    # --------------------------------------------------------
    # Strength
    # --------------------------------------------------------

    mean_strength = (
        group[
            "strength"
        ]
        .mean()
    )


    sd_strength = (
        group[
            "strength"
        ]
        .std(
            ddof=1
        )
    )


    cv_strength = safe_cv(
        mean_strength,
        sd_strength
    )


    min_strength = (
        group[
            "strength"
        ]
        .min()
    )


    max_strength = (
        group[
            "strength"
        ]
        .max()
    )


    range_strength = (
        max_strength
        -
        min_strength
    )


    # --------------------------------------------------------
    # Degree Rank Stability
    # --------------------------------------------------------

    mean_degree_rank = (
        group[
            "degree_rank"
        ]
        .mean()
    )


    sd_degree_rank = (
        group[
            "degree_rank"
        ]
        .std(
            ddof=1
        )
    )


    min_degree_rank = (
        group[
            "degree_rank"
        ]
        .min()
    )


    max_degree_rank = (
        group[
            "degree_rank"
        ]
        .max()
    )


    range_degree_rank = (
        max_degree_rank
        -
        min_degree_rank
    )


    # --------------------------------------------------------
    # Strength Rank Stability
    # --------------------------------------------------------

    mean_strength_rank = (
        group[
            "strength_rank"
        ]
        .mean()
    )


    sd_strength_rank = (
        group[
            "strength_rank"
        ]
        .std(
            ddof=1
        )
    )


    min_strength_rank = (
        group[
            "strength_rank"
        ]
        .min()
    )


    max_strength_rank = (
        group[
            "strength_rank"
        ]
        .max()
    )


    range_strength_rank = (
        max_strength_rank
        -
        min_strength_rank
    )


    # --------------------------------------------------------
    # Top-central Share
    # --------------------------------------------------------

    top_degree_share = (
        group[
            "top_degree"
        ]
        .mean()
    )


    top_strength_share = (
        group[
            "top_strength"
        ]
        .mean()
    )


    # --------------------------------------------------------
    # Degree与Strength在时间上的相关
    # --------------------------------------------------------

    degree_strength_corr = (
        group[
            "degree"
        ]
        .corr(
            group[
                "strength"
            ]
        )
    )


    node_rows.append(
        {
            "code":
                code,

            "name":
                group[
                    "name"
                ]
                .iloc[0],

            "industry":
                group[
                    "industry"
                ]
                .iloc[0],

            "n_windows":
                len(
                    group
                ),

            # Degree
            "mean_degree":
                mean_degree,

            "sd_degree":
                sd_degree,

            "cv_degree":
                cv_degree,

            "min_degree":
                min_degree,

            "max_degree":
                max_degree,

            "range_degree":
                range_degree,

            # Strength
            "mean_strength":
                mean_strength,

            "sd_strength":
                sd_strength,

            "cv_strength":
                cv_strength,

            "min_strength":
                min_strength,

            "max_strength":
                max_strength,

            "range_strength":
                range_strength,

            # Degree Rank
            "mean_degree_rank":
                mean_degree_rank,

            "sd_degree_rank":
                sd_degree_rank,

            "min_degree_rank":
                min_degree_rank,

            "max_degree_rank":
                max_degree_rank,

            "range_degree_rank":
                range_degree_rank,

            # Strength Rank
            "mean_strength_rank":
                mean_strength_rank,

            "sd_strength_rank":
                sd_strength_rank,

            "min_strength_rank":
                min_strength_rank,

            "max_strength_rank":
                max_strength_rank,

            "range_strength_rank":
                range_strength_rank,

            # Top Share
            "top_degree_share":
                top_degree_share,

            "top_strength_share":
                top_strength_share,

            # Correlation
            "degree_strength_corr":
                degree_strength_corr
        }
    )


stability_df = pd.DataFrame(
    node_rows
)


# ============================================================
# 15. 定义Rank Stability阈值
#
# 使用15只股票自身的rank SD中位数作为描述性阈值
# ============================================================

degree_rank_sd_threshold = (
    stability_df[
        "sd_degree_rank"
    ]
    .median()
)


strength_rank_sd_threshold = (
    stability_df[
        "sd_strength_rank"
    ]
    .median()
)


print(
    "\nDegree Rank SD中位数：",
    degree_rank_sd_threshold
)


print(
    "Strength Rank SD中位数：",
    strength_rank_sd_threshold
)


# ============================================================
# 16. Degree角色分类
# ============================================================

def classify_degree_role(
    row
):

    central = (
        row[
            "mean_degree_rank"
        ]
        <=
        TOP_K
    )


    stable = (
        row[
            "sd_degree_rank"
        ]
        <=
        degree_rank_sd_threshold
    )


    if (
        central
        and
        stable
    ):

        return (
            "Stable central"
        )


    if (
        central
        and
        not stable
    ):

        return (
            "Dynamic central"
        )


    if (
        not central
        and
        stable
    ):

        return (
            "Stable peripheral"
        )


    return (
        "Dynamic peripheral"
    )


stability_df[
    "degree_role"
] = (
    stability_df
    .apply(
        classify_degree_role,
        axis=1
    )
)


# ============================================================
# 17. Strength角色分类
# ============================================================

def classify_strength_role(
    row
):

    central = (
        row[
            "mean_strength_rank"
        ]
        <=
        TOP_K
    )


    stable = (
        row[
            "sd_strength_rank"
        ]
        <=
        strength_rank_sd_threshold
    )


    if (
        central
        and
        stable
    ):

        return (
            "Stable central"
        )


    if (
        central
        and
        not stable
    ):

        return (
            "Dynamic central"
        )


    if (
        not central
        and
        stable
    ):

        return (
            "Stable peripheral"
        )


    return (
        "Dynamic peripheral"
    )


stability_df[
    "strength_role"
] = (
    stability_df
    .apply(
        classify_strength_role,
        axis=1
    )
)


# ============================================================
# 18. 综合角色
#
# 这里保持简单：
#
# Degree和Strength都Stable Central
# -> Strong stable central
#
# 至少一个是Dynamic Central
# -> Dynamic central candidate
#
# 其余作为Mixed / Peripheral
# ============================================================

def classify_combined_role(
    row
):

    degree_role = (
        row[
            "degree_role"
        ]
    )

    strength_role = (
        row[
            "strength_role"
        ]
    )


    if (
        degree_role
        ==
        "Stable central"
        and
        strength_role
        ==
        "Stable central"
    ):

        return (
            "Strong stable central"
        )


    if (
        degree_role
        ==
        "Dynamic central"
        or
        strength_role
        ==
        "Dynamic central"
    ):

        return (
            "Dynamic central candidate"
        )


    if (
        degree_role
        ==
        "Stable peripheral"
        and
        strength_role
        ==
        "Stable peripheral"
    ):

        return (
            "Stable peripheral"
        )


    if (
        degree_role
        ==
        "Dynamic peripheral"
        and
        strength_role
        ==
        "Dynamic peripheral"
    ):

        return (
            "Dynamic peripheral"
        )


    return (
        "Mixed role"
    )


stability_df[
    "combined_role"
] = (
    stability_df
    .apply(
        classify_combined_role,
        axis=1
    )
)


# ============================================================
# 19. 排序
#
# 优先显示长期中心股票
# ============================================================

stability_df = (
    stability_df
    .sort_values(
        [
            "mean_degree_rank",
            "mean_strength_rank"
        ],
        ascending=[
            True,
            True
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 20. Role Summary
# ============================================================

role_summary_df = (
    stability_df[
        "combined_role"
    ]
    .value_counts()
    .rename_axis(
        "combined_role"
    )
    .reset_index(
        name="n_nodes"
    )
)


# ============================================================
# 21. Stable Central / Dynamic Central子集
# ============================================================

stable_central_df = stability_df[
    stability_df[
        "combined_role"
    ]
    ==
    "Strong stable central"
].copy()


dynamic_central_df = stability_df[
    stability_df[
        "combined_role"
    ]
    ==
    "Dynamic central candidate"
].copy()


# ============================================================
# 22. 保存CSV
# ============================================================

stability_df.to_csv(
    STABILITY_FILE,
    index=False,
    encoding="utf-8-sig"
)


role_summary_df.to_csv(
    ROLE_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


stable_central_df.to_csv(
    STABLE_CENTRAL_FILE,
    index=False,
    encoding="utf-8-sig"
)


dynamic_central_df.to_csv(
    DYNAMIC_CENTRAL_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. 屏幕输出
# ============================================================

print(
    "\n======================================"
)

print(
    "Dynamic Centrality Stability"
)

print(
    "======================================"
)


display_columns = [
    "code",
    "name",
    "industry",

    "mean_degree",
    "cv_degree",

    "mean_strength",
    "cv_strength",

    "mean_degree_rank",
    "sd_degree_rank",

    "mean_strength_rank",
    "sd_strength_rank",

    "top_degree_share",
    "top_strength_share",

    "degree_role",
    "strength_role",
    "combined_role"
]


print(
    stability_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Combined Role Summary"
)

print(
    "======================================"
)


print(
    role_summary_df.to_string(
        index=False
    )
)


# ============================================================
# 24. 图1：Mean Degree vs CV Degree
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


ax.scatter(
    stability_df[
        "mean_degree"
    ],
    stability_df[
        "cv_degree"
    ]
)


for row in stability_df.itertuples():

    ax.annotate(
        f"{row.name}({row.code})",

        (
            row.mean_degree,
            row.cv_degree
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points",

        fontsize=8
    )


ax.set_xlabel(
    "Mean Degree"
)


ax.set_ylabel(
    "CV of Degree"
)


ax.set_title(
    "节点平均Degree与Degree相对波动"
)


fig.tight_layout()


DEGREE_CV_FIGURE = (
    FIGURE_DIR
    / "dynamic_centrality_mean_degree_vs_cv.png"
)


fig.savefig(
    DEGREE_CV_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. 图2：Mean Strength vs CV Strength
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


ax.scatter(
    stability_df[
        "mean_strength"
    ],
    stability_df[
        "cv_strength"
    ]
)


for row in stability_df.itertuples():

    ax.annotate(
        f"{row.name}({row.code})",

        (
            row.mean_strength,
            row.cv_strength
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points",

        fontsize=8
    )


ax.set_xlabel(
    "Mean Strength"
)


ax.set_ylabel(
    "CV of Strength"
)


ax.set_title(
    "节点平均Strength与Strength相对波动"
)


fig.tight_layout()


STRENGTH_CV_FIGURE = (
    FIGURE_DIR
    / "dynamic_centrality_mean_strength_vs_cv.png"
)


fig.savefig(
    STRENGTH_CV_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 26. 图3：
# Mean Degree Rank vs Rank SD
#
# 越靠左 -> 平均排名越靠前
# 越靠下 -> 排名越稳定
# 左下角最接近稳定核心节点
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


ax.scatter(
    stability_df[
        "mean_degree_rank"
    ],
    stability_df[
        "sd_degree_rank"
    ]
)


ax.axvline(
    x=TOP_K,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=degree_rank_sd_threshold,
    linestyle="--",
    linewidth=1
)


for row in stability_df.itertuples():

    ax.annotate(
        f"{row.name}({row.code})",

        (
            row.mean_degree_rank,
            row.sd_degree_rank
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points",

        fontsize=8
    )


ax.set_xlabel(
    "Mean Degree Rank (smaller = more central)"
)


ax.set_ylabel(
    "SD of Degree Rank"
)


ax.set_title(
    "Degree-based Dynamic Node Role"
)


# 排名1应该视觉上更靠“核心”
ax.invert_xaxis()


fig.tight_layout()


DEGREE_RANK_FIGURE = (
    FIGURE_DIR
    / "dynamic_centrality_degree_rank_stability.png"
)


fig.savefig(
    DEGREE_RANK_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 27. 图4：
# Mean Strength Rank vs Rank SD
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


ax.scatter(
    stability_df[
        "mean_strength_rank"
    ],
    stability_df[
        "sd_strength_rank"
    ]
)


ax.axvline(
    x=TOP_K,
    linestyle="--",
    linewidth=1
)


ax.axhline(
    y=strength_rank_sd_threshold,
    linestyle="--",
    linewidth=1
)


for row in stability_df.itertuples():

    ax.annotate(
        f"{row.name}({row.code})",

        (
            row.mean_strength_rank,
            row.sd_strength_rank
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points",

        fontsize=8
    )


ax.set_xlabel(
    "Mean Strength Rank (smaller = more central)"
)


ax.set_ylabel(
    "SD of Strength Rank"
)


ax.set_title(
    "Strength-based Dynamic Node Role"
)


ax.invert_xaxis()


fig.tight_layout()


STRENGTH_RANK_FIGURE = (
    FIGURE_DIR
    / "dynamic_centrality_strength_rank_stability.png"
)


fig.savefig(
    STRENGTH_RANK_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 28. 图5：Top-central Share
# ============================================================

plot_df = (
    stability_df
    .sort_values(
        "top_degree_share",
        ascending=True
    )
    .copy()
)


labels = [
    f"{row.name}({row.code})"
    for row
    in plot_df.itertuples()
]


y = np.arange(
    len(
        plot_df
    )
)


fig, ax = plt.subplots(
    figsize=(
        11,
        8
    )
)


ax.scatter(
    plot_df[
        "top_degree_share"
    ],
    y,
    label="Degree Top-share"
)


ax.scatter(
    plot_df[
        "top_strength_share"
    ],
    y,
    label="Strength Top-share"
)


ax.set_yticks(
    y
)


ax.set_yticklabels(
    labels,
    fontsize=9
)


ax.set_xlim(
    0,
    1.05
)


ax.set_xlabel(
    f"Share of windows ranked in Top {TOP_K}"
)


ax.set_ylabel(
    "Stock"
)


ax.set_title(
    "节点长期处于网络核心位置的比例"
)


ax.legend()


fig.tight_layout()


TOP_SHARE_FIGURE = (
    FIGURE_DIR
    / "dynamic_centrality_top_share.png"
)


fig.savefig(
    TOP_SHARE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 29. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Dynamic Centrality Stability分析完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for path in [
    WINDOW_LEVEL_FILE,
    STABILITY_FILE,
    ROLE_SUMMARY_FILE,
    STABLE_CENTRAL_FILE,
    DYNAMIC_CENTRAL_FILE,
    DEGREE_CV_FIGURE,
    STRENGTH_CV_FIGURE,
    DEGREE_RANK_FIGURE,
    STRENGTH_RANK_FIGURE,
    TOP_SHARE_FIGURE
]:

    print(
        path
    )
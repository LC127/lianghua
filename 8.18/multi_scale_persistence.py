from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 中文字体设置
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
        "股票中文名称可能无法正常显示。"
    )


set_chinese_font()


# ============================================================
# 0. 路径
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


# ============================================================
# 1. 输入文件
# ============================================================

INPUT_FILE = (
    PROCESSED_DIR
    / "window_size_edge_persistence.csv"
)


# ============================================================
# 2. 输出文件
# ============================================================

ALL_EDGES_FILE = (
    PROCESSED_DIR
    / "multi_scale_edge_persistence.csv"
)

MULTI_SCALE_CORE_FILE = (
    PROCESSED_DIR
    / "multi_scale_persistent_edges.csv"
)

ALWAYS_CORE_FILE = (
    PROCESSED_DIR
    / "multi_scale_always_persistent_edges.csv"
)

SCALE_DEPENDENT_FILE = (
    PROCESSED_DIR
    / "scale_dependent_persistent_edges.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "multi_scale_persistence_summary.csv"
)

INDUSTRY_SUMMARY_FILE = (
    PROCESSED_DIR
    / "multi_scale_persistence_industry_summary.csv"
)

STRONG_NINE_FILE = (
    PROCESSED_DIR
    / "strong_nine_multi_scale_check.csv"
)


# ============================================================
# 3. 参数
# ============================================================

WINDOW_SIZES = [
    126,
    252,
    504
]


PERSISTENCE_THRESHOLD = 0.80


# 用于判断数值上是否等于1
EPS = 1e-10


# ============================================================
# 4. 工具函数
# ============================================================

def normalize_code(x) -> str:

    s = str(
        x
    ).strip()


    if s.endswith(
        ".0"
    ):

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


def canonical_pair(
    stock_1,
    stock_2
):

    a = normalize_code(
        stock_1
    )

    b = normalize_code(
        stock_2
    )


    if a <= b:

        return a, b


    return b, a


# ============================================================
# 5. 读取Stage 2结果
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


df[
    "stock_1"
] = (
    df[
        "stock_1"
    ]
    .apply(
        normalize_code
    )
)


df[
    "stock_2"
] = (
    df[
        "stock_2"
    ]
    .apply(
        normalize_code
    )
)


df[
    "window_size"
] = (
    pd.to_numeric(
        df[
            "window_size"
        ],
        errors="raise"
    )
    .astype(int)
)


# ============================================================
# 6. 股票对canonicalize
# ============================================================

pairs = df.apply(

    lambda row:
        canonical_pair(
            row[
                "stock_1"
            ],
            row[
                "stock_2"
            ]
        ),

    axis=1
)


df[
    "stock_1"
] = [
    pair[0]
    for pair in pairs
]


df[
    "stock_2"
] = [
    pair[1]
    for pair in pairs
]


# ============================================================
# 7. 检查三个Window Size是否完整
# ============================================================

available_windows = sorted(
    df[
        "window_size"
    ]
    .unique()
    .tolist()
)


print(
    "文件中的Window Sizes：",
    available_windows
)


missing_windows = (
    set(
        WINDOW_SIZES
    )
    -
    set(
        available_windows
    )
)


if missing_windows:

    raise ValueError(
        "缺少以下Window Size结果："
        f"{sorted(missing_windows)}"
    )


# 只保留需要比较的三个尺度
df = df[
    df[
        "window_size"
    ]
    .isin(
        WINDOW_SIZES
    )
].copy()


# ============================================================
# 8. 检查每条边是否都有三个尺度结果
# ============================================================

scale_count = (
    df
    .groupby(
        [
            "stock_1",
            "stock_2"
        ]
    )[
        "window_size"
    ]
    .nunique()
)


bad_pairs = scale_count[
    scale_count
    !=
    len(
        WINDOW_SIZES
    )
]


if len(
    bad_pairs
) > 0:

    print(
        bad_pairs
    )

    raise ValueError(
        "存在股票对缺少一个或多个Window Size结果。"
    )


# ============================================================
# 9. 检查名称和行业映射是否一致
# ============================================================

for col in [
    "name_1",
    "name_2",
    "industry_1",
    "industry_2"
]:

    if col not in df.columns:

        raise ValueError(
            f"输入文件缺少字段：{col}"
        )


metadata_conflict = (
    df
    .groupby(
        [
            "stock_1",
            "stock_2"
        ]
    )
    .agg(

        name_1_n=(
            "name_1",
            "nunique"
        ),

        name_2_n=(
            "name_2",
            "nunique"
        ),

        industry_1_n=(
            "industry_1",
            "nunique"
        ),

        industry_2_n=(
            "industry_2",
            "nunique"
        )
    )
)


conflict_rows = metadata_conflict[
    (
        metadata_conflict[
            "name_1_n"
        ]
        >
        1
    )
    |
    (
        metadata_conflict[
            "name_2_n"
        ]
        >
        1
    )
    |
    (
        metadata_conflict[
            "industry_1_n"
        ]
        >
        1
    )
    |
    (
        metadata_conflict[
            "industry_2_n"
        ]
        >
        1
    )
]


if len(
    conflict_rows
) > 0:

    print(
        conflict_rows
    )

    raise ValueError(
        "股票名称或行业映射在不同Window Size之间不一致。"
    )


# ============================================================
# 10. 提取每条边的Metadata
# ============================================================

metadata = (
    df[
        [
            "stock_1",
            "name_1",
            "industry_1",
            "stock_2",
            "name_2",
            "industry_2",
            "same_industry"
        ]
    ]
    .drop_duplicates(
        subset=[
            "stock_1",
            "stock_2"
        ]
    )
    .copy()
)


# ============================================================
# 11. Persistence转为宽表
# ============================================================

persistence_wide = (
    df
    .pivot(
        index=[
            "stock_1",
            "stock_2"
        ],

        columns=
            "window_size",

        values=
            "persistence"
    )
    .reset_index()
)


persistence_wide.columns.name = None


persistence_wide = (
    persistence_wide
    .rename(
        columns={
            126:
                "persistence_W126",

            252:
                "persistence_W252",

            504:
                "persistence_W504"
        }
    )
)


# ============================================================
# 12. 被选择时平均Absolute Partial Correlation
# ============================================================

strength_wide = (
    df
    .pivot(
        index=[
            "stock_1",
            "stock_2"
        ],

        columns=
            "window_size",

        values=
            "mean_abs_partial_when_selected"
    )
    .reset_index()
)


strength_wide.columns.name = None


strength_wide = (
    strength_wide
    .rename(
        columns={
            126:
                "mean_abs_partial_W126",

            252:
                "mean_abs_partial_W252",

            504:
                "mean_abs_partial_W504"
        }
    )
)


# ============================================================
# 13. Windows Selected宽表
# ============================================================

selected_wide = (
    df
    .pivot(
        index=[
            "stock_1",
            "stock_2"
        ],

        columns=
            "window_size",

        values=
            "windows_selected"
    )
    .reset_index()
)


selected_wide.columns.name = None


selected_wide = (
    selected_wide
    .rename(
        columns={
            126:
                "selected_windows_W126",

            252:
                "selected_windows_W252",

            504:
                "selected_windows_W504"
        }
    )
)


# ============================================================
# 14. 合并
# ============================================================

result = (
    metadata
    .merge(
        persistence_wide,

        on=[
            "stock_1",
            "stock_2"
        ],

        how="inner"
    )
    .merge(
        strength_wide,

        on=[
            "stock_1",
            "stock_2"
        ],

        how="left"
    )
    .merge(
        selected_wide,

        on=[
            "stock_1",
            "stock_2"
        ],

        how="left"
    )
)


# ============================================================
# 15. Multi-scale Persistence指标
# ============================================================

persistence_columns = [
    "persistence_W126",
    "persistence_W252",
    "persistence_W504"
]


result[
    "min_persistence"
] = (
    result[
        persistence_columns
    ]
    .min(
        axis=1
    )
)


result[
    "max_persistence"
] = (
    result[
        persistence_columns
    ]
    .max(
        axis=1
    )
)


result[
    "mean_persistence"
] = (
    result[
        persistence_columns
    ]
    .mean(
        axis=1
    )
)


result[
    "sd_persistence"
] = (
    result[
        persistence_columns
    ]
    .std(
        axis=1,
        ddof=0
    )
)


result[
    "persistence_range"
] = (
    result[
        "max_persistence"
    ]
    -
    result[
        "min_persistence"
    ]
)


# ============================================================
# 16. 每条边有几个尺度Persistence >= 0.8
# ============================================================

result[
    "n_scales_persistent"
] = (
    result[
        persistence_columns
    ]
    .ge(
        PERSISTENCE_THRESHOLD
    )
    .sum(
        axis=1
    )
)


# ============================================================
# 17. 有几个尺度Persistence = 1
# ============================================================

result[
    "n_scales_always_persistent"
] = (
    np.isclose(
        result[
            persistence_columns
        ],
        1.0,
        atol=EPS
    )
    .sum(
        axis=1
    )
)


# ============================================================
# 18. 核心Flag
# ============================================================

result[
    "multi_scale_persistent"
] = (
    result[
        "min_persistence"
    ]
    >=
    PERSISTENCE_THRESHOLD
)


result[
    "multi_scale_always_persistent"
] = np.isclose(
    result[
        "min_persistence"
    ],
    1.0,
    atol=EPS
)


# ============================================================
# 19. Scale-dependent Persistent Edge
#
# 至少一个尺度>=0.8，
# 但不是三个尺度都>=0.8
# ============================================================

result[
    "scale_dependent_persistent"
] = (
    (
        result[
            "n_scales_persistent"
        ]
        >=
        1
    )
    &
    (
        result[
            "n_scales_persistent"
        ]
        <
        3
    )
)


# ============================================================
# 20. Persistence Pattern
# ============================================================

def classify_pattern(
    row
):

    p126 = row[
        "persistence_W126"
    ]

    p252 = row[
        "persistence_W252"
    ]

    p504 = row[
        "persistence_W504"
    ]


    tol = 1e-12


    if (
        abs(
            p126 - p252
        )
        <=
        tol
        and
        abs(
            p252 - p504
        )
        <=
        tol
    ):

        return (
            "Constant"
        )


    if (
        p126
        <=
        p252 + tol
        and
        p252
        <=
        p504 + tol
    ):

        return (
            "Non-decreasing with W"
        )


    if (
        p126
        >=
        p252 - tol
        and
        p252
        >=
        p504 - tol
    ):

        return (
            "Non-increasing with W"
        )


    return (
        "Non-monotonic"
    )


result[
    "persistence_pattern"
] = result.apply(
    classify_pattern,
    axis=1
)


# ============================================================
# 21. 分类
# ============================================================

def classify_core(
    row
):

    if row[
        "multi_scale_always_persistent"
    ]:

        return (
            "Always persistent at all scales"
        )


    if row[
        "multi_scale_persistent"
    ]:

        return (
            "Multi-scale persistent"
        )


    if row[
        "n_scales_persistent"
    ] == 2:

        return (
            "Persistent at 2 scales"
        )


    if row[
        "n_scales_persistent"
    ] == 1:

        return (
            "Persistent at 1 scale"
        )


    return (
        "Not persistent at any scale"
    )


result[
    "multi_scale_class"
] = result.apply(
    classify_core,
    axis=1
)


# ============================================================
# 22. 平均Edge Strength
# ============================================================

strength_columns = [
    "mean_abs_partial_W126",
    "mean_abs_partial_W252",
    "mean_abs_partial_W504"
]


result[
    "mean_abs_partial_across_scales"
] = (
    result[
        strength_columns
    ]
    .mean(
        axis=1
    )
)


# ============================================================
# 23. 排序
#
# 最差尺度Persistence优先，
# 然后平均Persistence，
# 再按Edge Strength。
# ============================================================

result = (
    result
    .sort_values(
        [
            "min_persistence",
            "mean_persistence",
            "mean_abs_partial_across_scales"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 24. 提取核心边
# ============================================================

core_df = result[
    result[
        "multi_scale_persistent"
    ]
].copy()


always_df = result[
    result[
        "multi_scale_always_persistent"
    ]
].copy()


scale_dependent_df = result[
    result[
        "scale_dependent_persistent"
    ]
].copy()


# ============================================================
# 25. 行业层面的Multi-scale Core统计
# ============================================================

result[
    "industry_relation"
] = np.where(
    result[
        "same_industry"
    ],
    "Same industry",
    "Cross industry"
)


industry_summary = (
    result
    .groupby(
        "industry_relation",
        as_index=False
    )
    .agg(

        n_possible_pairs=(
            "stock_1",
            "size"
        ),

        n_multi_scale_persistent=(
            "multi_scale_persistent",
            "sum"
        ),

        n_multi_scale_always=(
            "multi_scale_always_persistent",
            "sum"
        ),

        mean_min_persistence=(
            "min_persistence",
            "mean"
        ),

        median_min_persistence=(
            "min_persistence",
            "median"
        ),

        mean_persistence_range=(
            "persistence_range",
            "mean"
        )
    )
)


industry_summary[
    "multi_scale_persistent_rate"
] = (
    industry_summary[
        "n_multi_scale_persistent"
    ]
    /
    industry_summary[
        "n_possible_pairs"
    ]
)


industry_summary[
    "multi_scale_always_rate"
] = (
    industry_summary[
        "n_multi_scale_always"
    ]
    /
    industry_summary[
        "n_possible_pairs"
    ]
)


# ============================================================
# 26. 总体Summary
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "total_pairs":
                len(
                    result
                ),

            "multi_scale_persistent_edges":
                len(
                    core_df
                ),

            "multi_scale_always_persistent_edges":
                len(
                    always_df
                ),

            "scale_dependent_persistent_edges":
                len(
                    scale_dependent_df
                ),

            "same_industry_multi_scale_edges":
                int(
                    core_df[
                        "same_industry"
                    ]
                    .sum()
                ),

            "cross_industry_multi_scale_edges":
                int(
                    (
                        ~core_df[
                            "same_industry"
                        ]
                    )
                    .sum()
                ),

            "mean_min_persistence_all_pairs":
                result[
                    "min_persistence"
                ]
                .mean(),

            "median_min_persistence_all_pairs":
                result[
                    "min_persistence"
                ]
                .median(),

            "mean_persistence_range":
                result[
                    "persistence_range"
                ]
                .mean(),

            "max_persistence_range":
                result[
                    "persistence_range"
                ]
                .max()
        }
    ]
)


# ============================================================
# 27. 检查前期9条强核心边
# ============================================================

STRONG_NINE = [

    ("601398", "601288"),  # 工商银行 - 农业银行

    ("000858", "000568"),  # 五粮液 - 泸州老窖

    ("000001", "600036"),  # 平安银行 - 招商银行

    ("600519", "000858"),  # 贵州茅台 - 五粮液

    ("601318", "600030"),  # 中国平安 - 中信证券

    ("000001", "601318"),  # 平安银行 - 中国平安

    ("603501", "002475"),  # 韦尔股份 - 立讯精密

    ("600519", "000568"),  # 贵州茅台 - 泸州老窖

    ("600519", "600887")   # 贵州茅台 - 伊利股份
]


strong_nine_keys = {
    canonical_pair(
        a,
        b
    )
    for a, b
    in STRONG_NINE
}


strong_nine_df = result[
    result.apply(

        lambda row:
            canonical_pair(
                row[
                    "stock_1"
                ],
                row[
                    "stock_2"
                ]
            )
            in
            strong_nine_keys,

        axis=1
    )
].copy()


# ============================================================
# 28. 保存结果
# ============================================================

result.to_csv(
    ALL_EDGES_FILE,
    index=False,
    encoding="utf-8-sig"
)


core_df.to_csv(
    MULTI_SCALE_CORE_FILE,
    index=False,
    encoding="utf-8-sig"
)


always_df.to_csv(
    ALWAYS_CORE_FILE,
    index=False,
    encoding="utf-8-sig"
)


scale_dependent_df.to_csv(
    SCALE_DEPENDENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


industry_summary.to_csv(
    INDUSTRY_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


strong_nine_df.to_csv(
    STRONG_NINE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 29. 屏幕输出
# ============================================================

print(
    "\n======================================"
)

print(
    "Multi-scale Persistence Summary"
)

print(
    "======================================"
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Industry Summary"
)

print(
    "======================================"
)


print(
    industry_summary.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Multi-scale Persistent Edges"
)

print(
    "======================================"
)


print(
    core_df[
        [
            "stock_1",
            "name_1",
            "industry_1",
            "stock_2",
            "name_2",
            "industry_2",
            "persistence_W126",
            "persistence_W252",
            "persistence_W504",
            "min_persistence",
            "mean_persistence",
            "persistence_range",
            "multi_scale_always_persistent",
            "mean_abs_partial_across_scales"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Scale-dependent Persistent Edges"
)

print(
    "======================================"
)


if len(
    scale_dependent_df
) > 0:

    print(
        scale_dependent_df[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "persistence_W126",
                "persistence_W252",
                "persistence_W504",
                "min_persistence",
                "max_persistence",
                "persistence_range",
                "n_scales_persistent",
                "persistence_pattern"
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有Scale-dependent Persistent Edges。"
    )


print(
    "\n======================================"
)

print(
    "前期9条强核心边检查"
)

print(
    "======================================"
)


print(
    strong_nine_df[
        [
            "stock_1",
            "name_1",
            "stock_2",
            "name_2",
            "persistence_W126",
            "persistence_W252",
            "persistence_W504",
            "min_persistence",
            "multi_scale_persistent",
            "multi_scale_always_persistent"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 30. 图1：三尺度Persistence分布
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


ax.boxplot(
    [
        result[
            "persistence_W126"
        ],

        result[
            "persistence_W252"
        ],

        result[
            "persistence_W504"
        ]
    ],

    tick_labels=[
        "W=126",
        "W=252",
        "W=504"
    ]
)


ax.axhline(
    y=PERSISTENCE_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.set_ylabel(
    "Edge persistence"
)


ax.set_title(
    "Edge Persistence across Window Sizes"
)


fig.tight_layout()


PERSISTENCE_FIGURE = (
    FIGURE_DIR
    / "multi_scale_persistence_distribution.png"
)


fig.savefig(
    PERSISTENCE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 31. 图2：Multi-scale Class数量
# ============================================================

class_order = [
    "Always persistent at all scales",
    "Multi-scale persistent",
    "Persistent at 2 scales",
    "Persistent at 1 scale",
    "Not persistent at any scale"
]


class_counts = (
    result[
        "multi_scale_class"
    ]
    .value_counts()
    .reindex(
        class_order,
        fill_value=0
    )
)


fig, ax = plt.subplots(
    figsize=(
        11,
        6
    )
)


ax.bar(
    class_counts.index,
    class_counts.values
)


ax.set_ylabel(
    "Number of stock pairs"
)


ax.set_title(
    "Multi-scale Persistence Classification"
)


ax.tick_params(
    axis="x",
    rotation=20
)


for i, value in enumerate(
    class_counts.values
):

    ax.text(
        i,
        value + 0.5,
        str(
            value
        ),
        ha="center"
    )


fig.tight_layout()


CLASS_FIGURE = (
    FIGURE_DIR
    / "multi_scale_persistence_classes.png"
)


fig.savefig(
    CLASS_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 32. 图3：Top Multi-scale Core Heatmap
# ============================================================

TOP_N = min(
    30,
    len(
        core_df
    )
)


if TOP_N > 0:

    top_core = (
        core_df
        .head(
            TOP_N
        )
        .copy()
    )


    matrix = (
        top_core[
            [
                "persistence_W126",
                "persistence_W252",
                "persistence_W504"
            ]
        ]
        .to_numpy()
    )


    labels = [

        f"{row.name_1}-{row.name_2}"

        for row
        in top_core.itertuples()
    ]


    fig_height = max(
        7,
        TOP_N * 0.35
    )


    fig, ax = plt.subplots(
        figsize=(
            8,
            fig_height
        )
    )


    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1
    )


    ax.set_xticks(
        [
            0,
            1,
            2
        ]
    )


    ax.set_xticklabels(
        [
            "W=126",
            "W=252",
            "W=504"
        ]
    )


    ax.set_yticks(
        np.arange(
            TOP_N
        )
    )


    ax.set_yticklabels(
        labels
    )


    ax.set_title(
        "Top Multi-scale Persistent Edges"
    )


    fig.colorbar(
        image,
        ax=ax,
        label="Persistence"
    )


    fig.tight_layout()


    HEATMAP_FILE = (
        FIGURE_DIR
        / "multi_scale_persistent_edge_heatmap.png"
    )


    fig.savefig(
        HEATMAP_FILE,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# 33. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Multi-scale Persistence分析完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for path in [

    ALL_EDGES_FILE,
    MULTI_SCALE_CORE_FILE,
    ALWAYS_CORE_FILE,
    SCALE_DEPENDENT_FILE,
    SUMMARY_FILE,
    INDUSTRY_SUMMARY_FILE,
    STRONG_NINE_FILE,
    PERSISTENCE_FIGURE,
    CLASS_FIGURE

]:

    print(
        path
    )
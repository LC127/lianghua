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
        "中文标签可能无法正常显示。"
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


# ============================================================
# 2. 输入文件
# ============================================================

EDGE_HISTORY_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 3. 输出文件
# ============================================================

STATE_TABLE_FILE = (
    PROCESSED_DIR
    / "dynamic_network_state_table.csv"
)

TRANSITION_DETAIL_FILE = (
    PROCESSED_DIR
    / "dynamic_network_transition_detail.csv"
)


# ============================================================
# 4. 工具函数：股票代码
# ============================================================

def normalize_code(x) -> str:

    s = str(
        x
    ).strip()


    # Excel有时会将000001读成1.0
    if s.endswith(".0"):

        s = s[:-2]


    # 优先寻找连续6位数字
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
# 5. 工具函数：Boolean
# ============================================================

def convert_bool(
    series,
    column_name
):

    if series.dtype == bool:

        return series


    converted = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False
            }
        )
    )


    if converted.isna().any():

        bad_values = (
            series[
                converted.isna()
            ]
            .unique()
        )


        raise ValueError(
            f"{column_name}中存在无法识别的值："
            f"{bad_values}"
        )


    return converted.astype(bool)


# ============================================================
# 6. 安全Z-score
# ============================================================

def zscore(
    series
):

    mean_value = (
        series.mean()
    )

    sd_value = (
        series.std(
            ddof=1
        )
    )


    if (
        pd.isna(sd_value)
        or
        sd_value <= 1e-12
    ):

        return pd.Series(
            np.nan,
            index=series.index
        )


    return (
        series
        -
        mean_value
    ) / sd_value


# ============================================================
# 7. 读取股票信息
# ============================================================

stock_info = pd.read_csv(
    STOCK_INFO_FILE,
    dtype=str
)


code_candidates = [
    "code",
    "stock_code",
    "ts_code",
    "symbol"
]

name_candidates = [
    "name",
    "stock_name"
]

industry_candidates = [
    "industry",
    "industry_name"
]


code_col = next(
    (
        col
        for col in code_candidates
        if col in stock_info.columns
    ),
    None
)


name_col = next(
    (
        col
        for col in name_candidates
        if col in stock_info.columns
    ),
    None
)


industry_col = next(
    (
        col
        for col in industry_candidates
        if col in stock_info.columns
    ),
    None
)


if (
    code_col is None
    or
    name_col is None
    or
    industry_col is None
):

    raise ValueError(
        "stock_info.csv必须包含"
        "股票代码、股票名称和行业字段。\n"
        f"当前字段：{stock_info.columns.tolist()}"
    )


metadata = (
    stock_info[
        [
            code_col,
            name_col,
            industry_col
        ]
    ]
    .rename(
        columns={
            code_col: "code",
            name_col: "name",
            industry_col: "industry"
        }
    )
    .copy()
)


metadata["code"] = (
    metadata[
        "code"
    ]
    .apply(
        normalize_code
    )
)


metadata = (
    metadata
    .drop_duplicates(
        subset="code"
    )
    .reset_index(
        drop=True
    )
)


name_map = dict(
    zip(
        metadata[
            "code"
        ],
        metadata[
            "name"
        ]
    )
)


industry_map = dict(
    zip(
        metadata[
            "code"
        ],
        metadata[
            "industry"
        ]
    )
)


N_STOCKS = len(
    metadata
)


MAX_EDGES = (
    N_STOCKS
    *
    (
        N_STOCKS - 1
    )
    //
    2
)


print(
    "股票数量：",
    N_STOCKS
)

print(
    "最大可能边数：",
    MAX_EDGES
)


# ============================================================
# 8. 读取Rolling Edge History
# ============================================================

df = pd.read_csv(
    EDGE_HISTORY_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


required_columns = [
    "stock_1",
    "stock_2",
    "selected",
    "partial_correlation"
]


for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"缺少必要字段：{col}"
        )


# ============================================================
# 9. 股票代码标准化
# ============================================================

df["stock_1"] = (
    df[
        "stock_1"
    ]
    .apply(
        normalize_code
    )
)


df["stock_2"] = (
    df[
        "stock_2"
    ]
    .apply(
        normalize_code
    )
)


# ============================================================
# 10. Canonicalize Pair
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


df["stock_1"] = [
    pair[0]
    for pair in pairs
]


df["stock_2"] = [
    pair[1]
    for pair in pairs
]


# ============================================================
# 11. canonicalize后重新映射名称/行业
# ============================================================

df["name_1"] = (
    df[
        "stock_1"
    ]
    .map(
        name_map
    )
)


df["name_2"] = (
    df[
        "stock_2"
    ]
    .map(
        name_map
    )
)


df["industry_1"] = (
    df[
        "stock_1"
    ]
    .map(
        industry_map
    )
)


df["industry_2"] = (
    df[
        "stock_2"
    ]
    .map(
        industry_map
    )
)


if (
    df[
        [
            "name_1",
            "name_2",
            "industry_1",
            "industry_2"
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise ValueError(
        "部分股票无法从stock_info.csv中找到名称或行业。"
    )


df[
    "same_industry"
] = (
    df[
        "industry_1"
    ]
    ==
    df[
        "industry_2"
    ]
)


# ============================================================
# 12. selected / partial转换
# ============================================================

df[
    "selected"
] = convert_bool(
    df[
        "selected"
    ],
    "selected"
)


df[
    "partial_correlation"
] = pd.to_numeric(
    df[
        "partial_correlation"
    ],
    errors="raise"
)


if (
    "abs_partial_correlation"
    not in
    df.columns
):

    df[
        "abs_partial_correlation"
    ] = (
        df[
            "partial_correlation"
        ]
        .abs()
    )

else:

    df[
        "abs_partial_correlation"
    ] = pd.to_numeric(
        df[
            "abs_partial_correlation"
        ],
        errors="raise"
    )


# ============================================================
# 13. 日期
# ============================================================

date_candidates = [
    "network_date",
    "window_end"
]


DATE_COL = next(
    (
        col
        for col in date_candidates
        if col in df.columns
    ),
    None
)


if DATE_COL is None:

    raise ValueError(
        "找不到network_date或window_end字段。"
    )


df[
    "network_date"
] = pd.to_datetime(
    df[
        DATE_COL
    ]
)


# ============================================================
# 14. Window ID
# ============================================================

if (
    "window_id"
    not in
    df.columns
):

    date_order = (
        df[
            "network_date"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )


    date_to_id = {
        date: i + 1
        for i, date
        in enumerate(
            date_order
        )
    }


    df[
        "window_id"
    ] = (
        df[
            "network_date"
        ]
        .map(
            date_to_id
        )
    )


df[
    "window_id"
] = (
    pd.to_numeric(
        df[
            "window_id"
        ],
        errors="raise"
    )
    .astype(int)
)


# ============================================================
# 15. 基础完整性检查
# ============================================================

pair_count_by_window = (
    df
    .groupby(
        "window_id"
    )
    .size()
)


bad_windows = (
    pair_count_by_window[
        pair_count_by_window
        !=
        MAX_EDGES
    ]
)


if len(
    bad_windows
) > 0:

    print(
        bad_windows
    )

    raise ValueError(
        "部分Rolling Window不是完整的股票对集合。"
    )


duplicate_check = (
    df
    .duplicated(
        subset=[
            "window_id",
            "stock_1",
            "stock_2"
        ]
    )
)


if duplicate_check.any():

    raise ValueError(
        "同一Window中存在重复股票对。"
    )


# ============================================================
# 16. 排序
# ============================================================

df = (
    df
    .sort_values(
        [
            "window_id",
            "stock_1",
            "stock_2"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 17. 构造每个Network Date的状态变量
# ============================================================

state_rows = []


for window_id, group in df.groupby(
    "window_id"
):

    group = (
        group
        .sort_values(
            [
                "stock_1",
                "stock_2"
            ]
        )
    )


    network_date = (
        group[
            "network_date"
        ]
        .iloc[0]
    )


    selected_df = (
        group[
            group[
                "selected"
            ]
        ]
    )


    # --------------------------------------------------------
    # Edge Count
    # --------------------------------------------------------

    edge_count = len(
        selected_df
    )


    density = (
        edge_count
        /
        MAX_EDGES
    )


    # --------------------------------------------------------
    # Edge Strength
    # --------------------------------------------------------

    if edge_count > 0:

        mean_abs_partial = (
            selected_df[
                "abs_partial_correlation"
            ]
            .mean()
        )


        median_abs_partial = (
            selected_df[
                "abs_partial_correlation"
            ]
            .median()
        )


        total_abs_partial = (
            selected_df[
                "abs_partial_correlation"
            ]
            .sum()
        )


    else:

        mean_abs_partial = np.nan
        median_abs_partial = np.nan
        total_abs_partial = 0.0


    # --------------------------------------------------------
    # Same / Cross
    # --------------------------------------------------------

    same_df = (
        selected_df[
            selected_df[
                "same_industry"
            ]
        ]
    )


    cross_df = (
        selected_df[
            ~selected_df[
                "same_industry"
            ]
        ]
    )


    same_edges = len(
        same_df
    )


    cross_edges = len(
        cross_df
    )


    if edge_count > 0:

        same_edge_ratio = (
            same_edges
            /
            edge_count
        )


        cross_edge_ratio = (
            cross_edges
            /
            edge_count
        )


    else:

        same_edge_ratio = np.nan
        cross_edge_ratio = np.nan


    # --------------------------------------------------------
    # Same / Cross Strength
    # --------------------------------------------------------

    mean_abs_partial_same = (
        same_df[
            "abs_partial_correlation"
        ]
        .mean()
        if len(
            same_df
        ) > 0
        else np.nan
    )


    mean_abs_partial_cross = (
        cross_df[
            "abs_partial_correlation"
        ]
        .mean()
        if len(
            cross_df
        ) > 0
        else np.nan
    )


    # --------------------------------------------------------
    # 正负边
    # --------------------------------------------------------

    positive_edges = int(
        (
            selected_df[
                "partial_correlation"
            ]
            >
            0
        )
        .sum()
    )


    negative_edges = int(
        (
            selected_df[
                "partial_correlation"
            ]
            <
            0
        )
        .sum()
    )


    # --------------------------------------------------------
    # Window Start
    # --------------------------------------------------------

    if (
        "window_start"
        in
        group.columns
    ):

        window_start = pd.to_datetime(
            group[
                "window_start"
            ]
            .iloc[0]
        )

    else:

        window_start = pd.NaT


    state_rows.append(
        {
            "window_id":
                int(
                    window_id
                ),

            "window_start":
                window_start,

            "network_date":
                network_date,

            "edge_count":
                edge_count,

            "density":
                density,

            "mean_abs_partial":
                mean_abs_partial,

            "median_abs_partial":
                median_abs_partial,

            "total_abs_partial":
                total_abs_partial,

            "same_edges":
                same_edges,

            "cross_edges":
                cross_edges,

            "same_industry_ratio":
                same_edge_ratio,

            "cross_industry_ratio":
                cross_edge_ratio,

            "mean_abs_partial_same":
                mean_abs_partial_same,

            "mean_abs_partial_cross":
                mean_abs_partial_cross,

            "positive_edges":
                positive_edges,

            "negative_edges":
                negative_edges
        }
    )


state_df = pd.DataFrame(
    state_rows
)


state_df = (
    state_df
    .sort_values(
        "window_id"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 18. 构造相邻网络变化指标
# ============================================================

transition_rows = []


window_ids = (
    state_df[
        "window_id"
    ]
    .tolist()
)


for k in range(
    1,
    len(
        window_ids
    )
):

    previous_id = (
        window_ids[
            k - 1
        ]
    )


    current_id = (
        window_ids[
            k
        ]
    )


    previous_group = df[
        (
            df[
                "window_id"
            ]
            ==
            previous_id
        )
        &
        (
            df[
                "selected"
            ]
        )
    ]


    current_group = df[
        (
            df[
                "window_id"
            ]
            ==
            current_id
        )
        &
        (
            df[
                "selected"
            ]
        )
    ]


    E_previous = {

        canonical_pair(
            row.stock_1,
            row.stock_2
        )

        for row
        in previous_group.itertuples()
    }


    E_current = {

        canonical_pair(
            row.stock_1,
            row.stock_2
        )

        for row
        in current_group.itertuples()
    }


    # --------------------------------------------------------
    # Set operations
    # --------------------------------------------------------

    common = (
        E_previous
        &
        E_current
    )


    lost = (
        E_previous
        -
        E_current
    )


    gained = (
        E_current
        -
        E_previous
    )


    union = (
        E_previous
        |
        E_current
    )


    # --------------------------------------------------------
    # Jaccard / Turnover
    # --------------------------------------------------------

    if len(
        union
    ) > 0:

        jaccard = (
            len(
                common
            )
            /
            len(
                union
            )
        )

    else:

        jaccard = 1.0


    turnover = (
        1.0
        -
        jaccard
    )


    # --------------------------------------------------------
    # 判断Same / Cross
    # --------------------------------------------------------

    def is_same_industry(
        edge
    ):

        stock_1, stock_2 = edge

        return (
            industry_map[
                stock_1
            ]
            ==
            industry_map[
                stock_2
            ]
        )


    lost_same = sum(

        is_same_industry(
            edge
        )

        for edge
        in lost
    )


    gained_same = sum(

        is_same_industry(
            edge
        )

        for edge
        in gained
    )


    lost_cross = (
        len(
            lost
        )
        -
        lost_same
    )


    gained_cross = (
        len(
            gained
        )
        -
        gained_same
    )


    gross_edge_changes = (
        len(
            lost
        )
        +
        len(
            gained
        )
    )


    same_edge_changes = (
        lost_same
        +
        gained_same
    )


    cross_edge_changes = (
        lost_cross
        +
        gained_cross
    )


    if gross_edge_changes > 0:

        cross_change_share = (
            cross_edge_changes
            /
            gross_edge_changes
        )

    else:

        cross_change_share = np.nan


    previous_edge_count = len(
        E_previous
    )


    current_edge_count = len(
        E_current
    )


    transition_rows.append(
        {
            "window_from":
                previous_id,

            "window_to":
                current_id,

            "network_date":
                state_df.loc[
                    state_df[
                        "window_id"
                    ]
                    ==
                    current_id,

                    "network_date"
                ]
                .iloc[0],

            "previous_edge_count":
                previous_edge_count,

            "current_edge_count":
                current_edge_count,

            "edge_count_change":
                (
                    current_edge_count
                    -
                    previous_edge_count
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

            "gross_edge_changes":
                gross_edge_changes,

            "jaccard":
                jaccard,

            "turnover":
                turnover,

            "lost_same":
                lost_same,

            "gained_same":
                gained_same,

            "lost_cross":
                lost_cross,

            "gained_cross":
                gained_cross,

            "same_edge_changes":
                same_edge_changes,

            "cross_edge_changes":
                cross_edge_changes,

            "cross_change_share":
                cross_change_share
        }
    )


transition_df = pd.DataFrame(
    transition_rows
)


# ============================================================
# 19. 合并到Dynamic Network State Table
# ============================================================

state_df = (
    state_df
    .merge(
        transition_df[
            [
                "network_date",
                "edge_count_change",
                "common_edges",
                "lost_edges",
                "gained_edges",
                "gross_edge_changes",
                "jaccard",
                "turnover",
                "lost_same",
                "gained_same",
                "lost_cross",
                "gained_cross",
                "same_edge_changes",
                "cross_edge_changes",
                "cross_change_share"
            ]
        ],

        on="network_date",

        how="left"
    )
)


# ============================================================
# 20. 为Stage 2预先计算Z-score
#
# 第一窗口没有transition，因此Turnover等为NaN。
# pandas会自动忽略NaN计算均值/SD。
# ============================================================

zscore_columns = [
    "edge_count",
    "density",
    "mean_abs_partial",
    "same_edges",
    "cross_edges",
    "same_industry_ratio",
    "turnover",
    "gross_edge_changes"
]


for col in zscore_columns:

    state_df[
        f"{col}_z"
    ] = zscore(
        state_df[
            col
        ]
    )


# ============================================================
# 21. 简单标记High-turnover / High-change
#
# 这里只作为Stage 2的预备字段。
# 暂不把它当正式Change Point。
# ============================================================

state_df[
    "high_turnover_z1"
] = (
    state_df[
        "turnover_z"
    ]
    >
    1.0
)


state_df[
    "high_turnover_z1_5"
] = (
    state_df[
        "turnover_z"
    ]
    >
    1.5
)


state_df[
    "high_gross_change_z1"
] = (
    state_df[
        "gross_edge_changes_z"
    ]
    >
    1.0
)


# ============================================================
# 22. 保存
# ============================================================

state_df.to_csv(
    STATE_TABLE_FILE,
    index=False,
    encoding="utf-8-sig"
)


transition_df.to_csv(
    TRANSITION_DETAIL_FILE,
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
    "Dynamic Network State Table"
)

print(
    "======================================"
)


display_columns = [
    "window_id",
    "network_date",
    "edge_count",
    "density",
    "mean_abs_partial",
    "same_edges",
    "cross_edges",
    "same_industry_ratio",
    "turnover",
    "lost_edges",
    "gained_edges",
    "gross_edge_changes",
    "cross_change_share"
]


print(
    state_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


print(
    "\n网络数量：",
    len(
        state_df
    )
)


print(
    "日期范围：",
    state_df[
        "network_date"
    ]
    .min(),
    "至",
    state_df[
        "network_date"
    ]
    .max()
)


# ============================================================
# 24. 找出Turnover最高的5个时期
# ============================================================

top_turnover = (
    state_df
    .dropna(
        subset=[
            "turnover"
        ]
    )
    .sort_values(
        "turnover",
        ascending=False
    )
    .head(5)
)


print(
    "\n======================================"
)

print(
    "Turnover最高的5个Network Dates"
)

print(
    "======================================"
)


print(
    top_turnover[
        [
            "network_date",
            "edge_count",
            "edge_count_change",
            "lost_edges",
            "gained_edges",
            "gross_edge_changes",
            "turnover",
            "cross_edge_changes",
            "cross_change_share"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 25. 图1：Edge Count
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    state_df[
        "network_date"
    ],
    state_df[
        "edge_count"
    ],
    marker="o"
)


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Number of Edges"
)

ax.set_title(
    "Dynamic Network Edge Count"
)

ax.grid(
    alpha=0.3
)


fig.tight_layout()


EDGE_COUNT_FIGURE = (
    FIGURE_DIR
    / "network_state_edge_count.png"
)


fig.savefig(
    EDGE_COUNT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 26. 图2：Turnover
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    state_df[
        "network_date"
    ],
    state_df[
        "turnover"
    ],
    marker="o"
)


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Network Turnover"
)

ax.set_title(
    "Dynamic Network Turnover"
)

ax.grid(
    alpha=0.3
)


fig.tight_layout()


TURNOVER_FIGURE = (
    FIGURE_DIR
    / "network_state_turnover.png"
)


fig.savefig(
    TURNOVER_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 27. 图3：Same vs Cross Edge Count
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    state_df[
        "network_date"
    ],
    state_df[
        "same_edges"
    ],
    marker="o",
    label="同行业边"
)


ax.plot(
    state_df[
        "network_date"
    ],
    state_df[
        "cross_edges"
    ],
    marker="o",
    label="跨行业边"
)


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Number of Edges"
)

ax.set_title(
    "Same-industry vs Cross-industry Edge Count"
)

ax.legend()

ax.grid(
    alpha=0.3
)


fig.tight_layout()


SAME_CROSS_FIGURE = (
    FIGURE_DIR
    / "network_state_same_cross_edges.png"
)


fig.savefig(
    SAME_CROSS_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 28. 图4：Gross Edge Changes
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        6
    )
)


ax.plot(
    state_df[
        "network_date"
    ],
    state_df[
        "gross_edge_changes"
    ],
    marker="o"
)


ax.set_xlabel(
    "Network Date"
)

ax.set_ylabel(
    "Lost + Gained Edges"
)

ax.set_title(
    "Dynamic Network Gross Edge Changes"
)

ax.grid(
    alpha=0.3
)


fig.tight_layout()


GROSS_CHANGE_FIGURE = (
    FIGURE_DIR
    / "network_state_gross_changes.png"
)


fig.savefig(
    GROSS_CHANGE_FIGURE,
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
    "Stage 1完成"
)

print(
    "======================================"
)


print(
    "\n主结果文件："
)

print(
    STATE_TABLE_FILE
)


print(
    "\n相邻网络变化明细："
)

print(
    TRANSITION_DETAIL_FILE
)
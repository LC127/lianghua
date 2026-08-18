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

    installed = {
        f.name
        for f in font_manager.fontManager.ttflist
    }

    for font_name in candidates:

        if font_name in installed:

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

# Stage 2：
# 每条边 × Window Size 的Persistence
WINDOW_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "window_size_edge_persistence.csv"
)


# Stage 2：
# 每条边 × Window Size × Network Date的selected历史
WINDOW_EDGE_HISTORY_FILE = (
    PROCESSED_DIR
    / "window_size_edge_history.csv"
)


# Stage 3：
# Multi-scale Persistence
MULTISCALE_FILE = (
    PROCESSED_DIR
    / "multi_scale_edge_persistence.csv"
)


# Stage 4：
# W=252完整时期的Lifecycle
LIFECYCLE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_lifecycle.csv"
)


# 权威股票名称和行业信息
STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 3. 输出文件
# ============================================================

# 最重要：
# 每个W下Same/Cross综合比较
SCALE_VALIDATION_FILE = (
    PROCESSED_DIR
    / "stage6_scale_industry_validation_summary.csv"
)


# 一个W一行，直接给出Same-Cross差异和比率
SCALE_CONTRAST_FILE = (
    PROCESSED_DIR
    / "stage6_scale_core_periphery_contrast.csv"
)


# Stage 3 Multi-scale行业总结
MULTISCALE_SUMMARY_FILE = (
    PROCESSED_DIR
    / "stage6_multiscale_core_industry_summary.csv"
)


# Stage 4 Lifecycle行业总结
LIFECYCLE_SUMMARY_FILE = (
    PROCESSED_DIR
    / "stage6_lifecycle_industry_summary.csv"
)


# 只研究非Persistent且至少出现3次的Active Non-core
ACTIVE_NONCORE_FILE = (
    PROCESSED_DIR
    / "stage6_active_noncore_lifecycle_summary.csv"
)


# 综合证据矩阵
EVIDENCE_FILE = (
    PROCESSED_DIR
    / "stage6_industry_core_periphery_evidence_matrix.csv"
)


# ============================================================
# 4. 参数
# ============================================================

WINDOW_SIZES = [
    126,
    252,
    504
]

PERSISTENCE_THRESHOLD = 0.80

EPS = 1e-12


# ============================================================
# 5. 工具函数
# ============================================================

def normalize_code(x) -> str:

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


def to_bool(
    series,
    column_name
):

    if series.dtype == bool:
        return series

    result = (
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

    if result.isna().any():

        bad_values = (
            series[
                result.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"{column_name}包含无法识别的布尔值："
            f"{bad_values}"
        )

    return result.astype(bool)


def safe_ratio(
    numerator,
    denominator
):

    if (
        pd.isna(denominator)
        or
        abs(
            denominator
        )
        <
        EPS
    ):

        return np.nan

    return (
        numerator
        /
        denominator
    )


# ============================================================
# 6. 读取权威股票Metadata
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
        x
        for x in code_candidates
        if x in stock_info.columns
    ),
    None
)

name_col = next(
    (
        x
        for x in name_candidates
        if x in stock_info.columns
    ),
    None
)

industry_col = next(
    (
        x
        for x in industry_candidates
        if x in stock_info.columns
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
        "stock_info.csv缺少代码、名称或行业字段。"
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
    metadata["code"]
    .apply(normalize_code)
)


metadata = (
    metadata
    .drop_duplicates(
        subset="code"
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


# ============================================================
# 7. 统一修正Pair-level文件中的代码/名称/行业
#
# 之前曾出现canonicalize代码后name未同步的问题，
# 所以Stage 6始终从stock_info重新映射。
# ============================================================

def prepare_pair_df(
    df
):

    df = df.copy()


    df["stock_1"] = (
        df["stock_1"]
        .apply(normalize_code)
    )

    df["stock_2"] = (
        df["stock_2"]
        .apply(normalize_code)
    )


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


    # --------------------------------------------------------
    # canonicalize以后重新映射
    # --------------------------------------------------------

    df["name_1"] = (
        df["stock_1"]
        .map(name_map)
    )

    df["name_2"] = (
        df["stock_2"]
        .map(name_map)
    )


    df["industry_1"] = (
        df["stock_1"]
        .map(industry_map)
    )

    df["industry_2"] = (
        df["stock_2"]
        .map(industry_map)
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
            "部分股票无法从stock_info.csv映射名称/行业。"
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


    df[
        "industry_relation"
    ] = np.where(
        df[
            "same_industry"
        ],
        "Same industry",
        "Cross industry"
    )


    return df


# ============================================================
# PART A
# Stage 2：Persistence的多尺度Same/Cross比较
# ============================================================

persistence_df = pd.read_csv(
    WINDOW_PERSISTENCE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


persistence_df = prepare_pair_df(
    persistence_df
)


persistence_df[
    "window_size"
] = (
    pd.to_numeric(
        persistence_df[
            "window_size"
        ],
        errors="raise"
    )
    .astype(int)
)


persistence_df[
    "persistence"
] = (
    pd.to_numeric(
        persistence_df[
            "persistence"
        ],
        errors="raise"
    )
)


if (
    "mean_abs_partial_when_selected"
    in
    persistence_df.columns
):

    persistence_df[
        "mean_abs_partial_when_selected"
    ] = pd.to_numeric(
        persistence_df[
            "mean_abs_partial_when_selected"
        ],
        errors="coerce"
    )

else:

    persistence_df[
        "mean_abs_partial_when_selected"
    ] = np.nan


# ------------------------------------------------------------
# 检查：
# 每个W每个股票对应该只有一行
# ------------------------------------------------------------

duplicates = persistence_df.duplicated(
    subset=[
        "window_size",
        "stock_1",
        "stock_2"
    ]
)


if duplicates.any():

    raise ValueError(
        "window_size_edge_persistence.csv"
        "中存在重复股票对。"
    )


# ------------------------------------------------------------
# 汇总Persistence
# ------------------------------------------------------------

persistence_summary = (
    persistence_df
    .groupby(
        [
            "window_size",
            "industry_relation"
        ],
        as_index=False
    )
    .agg(

        n_possible_pairs=(
            "persistence",
            "size"
        ),

        mean_persistence=(
            "persistence",
            "mean"
        ),

        median_persistence=(
            "persistence",
            "median"
        ),

        persistent_edges=(
            "persistence",
            lambda x:
                int(
                    (
                        x
                        >=
                        PERSISTENCE_THRESHOLD
                    )
                    .sum()
                )
        ),

        always_edges=(
            "persistence",
            lambda x:
                int(
                    np.isclose(
                        x,
                        1.0
                    )
                    .sum()
                )
        ),

        mean_abs_partial_when_selected=(
            "mean_abs_partial_when_selected",
            "mean"
        )
    )
)


persistence_summary[
    "persistent_rate"
] = (
    persistence_summary[
        "persistent_edges"
    ]
    /
    persistence_summary[
        "n_possible_pairs"
    ]
)


persistence_summary[
    "always_rate"
] = (
    persistence_summary[
        "always_edges"
    ]
    /
    persistence_summary[
        "n_possible_pairs"
    ]
)


# ============================================================
# PART B
# Stage 2 Edge History：
# 计算标准化的State Change / Entry / Exit Rate
# ============================================================

history_df = pd.read_csv(
    WINDOW_EDGE_HISTORY_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


history_df = prepare_pair_df(
    history_df
)


history_df[
    "window_size"
] = (
    pd.to_numeric(
        history_df[
            "window_size"
        ],
        errors="raise"
    )
    .astype(int)
)


history_df[
    "selected"
] = to_bool(
    history_df[
        "selected"
    ],
    "selected"
)


# ------------------------------------------------------------
# 日期
# ------------------------------------------------------------

if (
    "network_date"
    not in
    history_df.columns
):

    if (
        "window_end"
        in
        history_df.columns
    ):

        history_df[
            "network_date"
        ] = history_df[
            "window_end"
        ]

    else:

        raise ValueError(
            "window_size_edge_history.csv"
            "缺少network_date/window_end。"
        )


history_df[
    "network_date"
] = pd.to_datetime(
    history_df[
        "network_date"
    ]
)


# ------------------------------------------------------------
# 唯一性检查
# ------------------------------------------------------------

duplicates = history_df.duplicated(
    subset=[
        "window_size",
        "network_date",
        "stock_1",
        "stock_2"
    ]
)


if duplicates.any():

    raise ValueError(
        "Edge History中存在重复的"
        "W-Date-StockPair记录。"
    )


history_df = (
    history_df
    .sort_values(
        [
            "window_size",
            "stock_1",
            "stock_2",
            "network_date"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# Previous selected
# ------------------------------------------------------------

history_df[
    "prev_selected"
] = (
    history_df
    .groupby(
        [
            "window_size",
            "stock_1",
            "stock_2"
        ]
    )[
        "selected"
    ]
    .shift(1)
)


transition_df = history_df[
    history_df[
        "prev_selected"
    ]
    .notna()
].copy()


transition_df[
    "prev_selected"
] = (
    transition_df[
        "prev_selected"
    ]
    .astype(bool)
)


transition_df[
    "state_changed"
] = (
    transition_df[
        "selected"
    ]
    !=
    transition_df[
        "prev_selected"
    ]
)


transition_df[
    "entry"
] = (
    (~transition_df["prev_selected"])
    &
    transition_df["selected"]
)


transition_df[
    "exit"
] = (
    transition_df["prev_selected"]
    &
    (~transition_df["selected"])
)


# ============================================================
# 8. 每个W / Same-Cross计算动态率
# ============================================================

transition_rows = []


for (
    window_size,
    relation
), group in transition_df.groupby(
    [
        "window_size",
        "industry_relation"
    ]
):

    n_pairs = (
        history_df[
            (
                history_df[
                    "window_size"
                ]
                ==
                window_size
            )
            &
            (
                history_df[
                    "industry_relation"
                ]
                ==
                relation
            )
        ][
            [
                "stock_1",
                "stock_2"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


    n_windows = (
        history_df[
            history_df[
                "window_size"
            ]
            ==
            window_size
        ][
            "network_date"
        ]
        .nunique()
    )


    transition_opportunities = (
        n_pairs
        *
        (
            n_windows - 1
        )
    )


    state_changes = int(
        group[
            "state_changed"
        ]
        .sum()
    )


    entries = int(
        group[
            "entry"
        ]
        .sum()
    )


    exits = int(
        group[
            "exit"
        ]
        .sum()
    )


    previous_selected_exposures = int(
        group[
            "prev_selected"
        ]
        .sum()
    )


    previous_absent_exposures = int(
        (
            ~group[
                "prev_selected"
            ]
        )
        .sum()
    )


    state_change_rate = (
        state_changes
        /
        transition_opportunities
        if transition_opportunities > 0
        else np.nan
    )


    exit_rate = (
        exits
        /
        previous_selected_exposures
        if previous_selected_exposures > 0
        else np.nan
    )


    entry_rate = (
        entries
        /
        previous_absent_exposures
        if previous_absent_exposures > 0
        else np.nan
    )


    selected_exposures = (
        history_df[
            (
                history_df[
                    "window_size"
                ]
                ==
                window_size
            )
            &
            (
                history_df[
                    "industry_relation"
                ]
                ==
                relation
            )
        ][
            "selected"
        ]
        .sum()
    )


    average_selected_edges = (
        selected_exposures
        /
        n_windows
    )


    transition_rows.append(
        {
            "window_size":
                window_size,

            "industry_relation":
                relation,

            "n_possible_pairs":
                n_pairs,

            "n_windows":
                n_windows,

            "transition_opportunities":
                transition_opportunities,

            "state_changes":
                state_changes,

            "entries":
                entries,

            "exits":
                exits,

            "previous_selected_exposures":
                previous_selected_exposures,

            "previous_absent_exposures":
                previous_absent_exposures,

            "state_change_rate":
                state_change_rate,

            "entry_rate":
                entry_rate,

            "exit_rate":
                exit_rate,

            "average_selected_edges":
                average_selected_edges
        }
    )


transition_summary = pd.DataFrame(
    transition_rows
)


# ============================================================
# 9. 计算Cross在全部Edge Changes中的占比
# ============================================================

change_total_by_w = (
    transition_summary
    .groupby(
        "window_size"
    )[
        "state_changes"
    ]
    .transform(
        "sum"
    )
)


transition_summary[
    "share_of_all_state_changes"
] = (
    transition_summary[
        "state_changes"
    ]
    /
    change_total_by_w
)


# ============================================================
# 10. 合并Persistence + Dynamics
# ============================================================

scale_validation_df = (
    persistence_summary
    .merge(
        transition_summary,

        on=[
            "window_size",
            "industry_relation"
        ],

        how="inner",

        suffixes=(
            "_persistence",
            "_transition"
        )
    )
)


scale_validation_df = (
    scale_validation_df
    .sort_values(
        [
            "window_size",
            "industry_relation"
        ]
    )
    .reset_index(
        drop=True
    )
)


scale_validation_df.to_csv(
    SCALE_VALIDATION_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 一个Window一行：
# Same vs Cross的直接对比
# ============================================================

contrast_rows = []


for window_size in WINDOW_SIZES:

    temp = scale_validation_df[
        scale_validation_df[
            "window_size"
        ]
        ==
        window_size
    ]


    same = temp[
        temp[
            "industry_relation"
        ]
        ==
        "Same industry"
    ]


    cross = temp[
        temp[
            "industry_relation"
        ]
        ==
        "Cross industry"
    ]


    if (
        len(same) != 1
        or
        len(cross) != 1
    ):

        raise ValueError(
            f"W={window_size}缺少Same/Cross结果。"
        )


    same = same.iloc[0]
    cross = cross.iloc[0]


    contrast_rows.append(
        {
            "window_size":
                window_size,

            # -----------------------------------------------
            # Persistence
            # -----------------------------------------------

            "same_mean_persistence":
                same[
                    "mean_persistence"
                ],

            "cross_mean_persistence":
                cross[
                    "mean_persistence"
                ],

            "persistence_difference_same_minus_cross":
                same[
                    "mean_persistence"
                ]
                -
                cross[
                    "mean_persistence"
                ],

            "persistence_ratio_same_over_cross":
                safe_ratio(
                    same[
                        "mean_persistence"
                    ],
                    cross[
                        "mean_persistence"
                    ]
                ),

            # -----------------------------------------------
            # Persistent >= .8
            # -----------------------------------------------

            "same_persistent_rate":
                same[
                    "persistent_rate"
                ],

            "cross_persistent_rate":
                cross[
                    "persistent_rate"
                ],

            "persistent_rate_ratio_same_over_cross":
                safe_ratio(
                    same[
                        "persistent_rate"
                    ],
                    cross[
                        "persistent_rate"
                    ]
                ),

            # -----------------------------------------------
            # Always
            # -----------------------------------------------

            "same_always_rate":
                same[
                    "always_rate"
                ],

            "cross_always_rate":
                cross[
                    "always_rate"
                ],

            "always_rate_ratio_same_over_cross":
                safe_ratio(
                    same[
                        "always_rate"
                    ],
                    cross[
                        "always_rate"
                    ]
                ),

            # -----------------------------------------------
            # State changes
            # -----------------------------------------------

            "same_state_change_rate":
                same[
                    "state_change_rate"
                ],

            "cross_state_change_rate":
                cross[
                    "state_change_rate"
                ],

            "state_change_rate_ratio_cross_over_same":
                safe_ratio(
                    cross[
                        "state_change_rate"
                    ],
                    same[
                        "state_change_rate"
                    ]
                ),

            # -----------------------------------------------
            # Exit
            # -----------------------------------------------

            "same_exit_rate":
                same[
                    "exit_rate"
                ],

            "cross_exit_rate":
                cross[
                    "exit_rate"
                ],

            "exit_rate_ratio_cross_over_same":
                safe_ratio(
                    cross[
                        "exit_rate"
                    ],
                    same[
                        "exit_rate"
                    ]
                ),

            # -----------------------------------------------
            # Entry
            # -----------------------------------------------

            "same_entry_rate":
                same[
                    "entry_rate"
                ],

            "cross_entry_rate":
                cross[
                    "entry_rate"
                ],

            # -----------------------------------------------
            # Raw share of changes
            # -----------------------------------------------

            "cross_share_of_all_changes":
                cross[
                    "share_of_all_state_changes"
                ]
        }
    )


contrast_df = pd.DataFrame(
    contrast_rows
)


contrast_df.to_csv(
    SCALE_CONTRAST_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PART C
# Stage 3：Multi-scale Core的行业组成
# ============================================================

multi_df = pd.read_csv(
    MULTISCALE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


multi_df = prepare_pair_df(
    multi_df
)


# ------------------------------------------------------------
# 如果flag不存在，就根据min_persistence重建
# ------------------------------------------------------------

if (
    "multi_scale_persistent"
    in
    multi_df.columns
):

    multi_df[
        "multi_scale_persistent"
    ] = to_bool(
        multi_df[
            "multi_scale_persistent"
        ],
        "multi_scale_persistent"
    )

else:

    multi_df[
        "multi_scale_persistent"
    ] = (
        pd.to_numeric(
            multi_df[
                "min_persistence"
            ],
            errors="raise"
        )
        >=
        PERSISTENCE_THRESHOLD
    )


if (
    "multi_scale_always_persistent"
    in
    multi_df.columns
):

    multi_df[
        "multi_scale_always_persistent"
    ] = to_bool(
        multi_df[
            "multi_scale_always_persistent"
        ],
        "multi_scale_always_persistent"
    )

else:

    multi_df[
        "multi_scale_always_persistent"
    ] = np.isclose(
        pd.to_numeric(
            multi_df[
                "min_persistence"
            ],
            errors="raise"
        ),
        1.0
    )


multi_df[
    "min_persistence"
] = pd.to_numeric(
    multi_df[
        "min_persistence"
    ],
    errors="raise"
)


multiscale_summary_df = (
    multi_df
    .groupby(
        "industry_relation",
        as_index=False
    )
    .agg(

        n_possible_pairs=(
            "stock_1",
            "size"
        ),

        n_multiscale_persistent=(
            "multi_scale_persistent",
            "sum"
        ),

        n_multiscale_always=(
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
        )
    )
)


multiscale_summary_df[
    "multiscale_persistent_rate"
] = (
    multiscale_summary_df[
        "n_multiscale_persistent"
    ]
    /
    multiscale_summary_df[
        "n_possible_pairs"
    ]
)


multiscale_summary_df[
    "multiscale_always_rate"
] = (
    multiscale_summary_df[
        "n_multiscale_always"
    ]
    /
    multiscale_summary_df[
        "n_possible_pairs"
    ]
)


multiscale_summary_df.to_csv(
    MULTISCALE_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PART D
# Stage 4：Lifecycle的同行业/跨行业比较
#
# 注意：
# 这是W=252完整31-window历史，
# 与Stage 2三个W共同的18个窗口不是同一时间覆盖。
# 因此它是补充证据，不直接与上述数字混算。
# ============================================================

lifecycle_df = pd.read_csv(
    LIFECYCLE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


lifecycle_df = prepare_pair_df(
    lifecycle_df
)


numeric_lifecycle_cols = [
    "windows_selected",
    "persistence",
    "n_episodes",
    "longest_consecutive_run",
    "longest_run_share",
    "transition_rate",
    "mean_abs_partial_when_selected"
]


for col in numeric_lifecycle_cols:

    if col in lifecycle_df.columns:

        lifecycle_df[
            col
        ] = pd.to_numeric(
            lifecycle_df[
                col
            ],
            errors="coerce"
        )


# ============================================================
# 12. Lifecycle总体行业总结
# ============================================================

def class_count(
    x,
    class_name
):

    return int(
        (
            x
            ==
            class_name
        )
        .sum()
    )


lifecycle_summary_df = (
    lifecycle_df
    .groupby(
        "industry_relation",
        as_index=False
    )
    .agg(

        n_possible_pairs=(
            "stock_1",
            "size"
        ),

        mean_persistence=(
            "persistence",
            "mean"
        ),

        mean_transition_rate=(
            "transition_rate",
            "mean"
        ),

        mean_n_episodes=(
            "n_episodes",
            "mean"
        ),

        mean_longest_run=(
            "longest_consecutive_run",
            "mean"
        ),

        persistent_core_edges=(
            "lifecycle_class",
            lambda x:
                class_count(
                    x,
                    "Persistent core"
                )
        ),

        regime_dependent_edges=(
            "lifecycle_class",
            lambda x:
                class_count(
                    x,
                    "Regime-dependent"
                )
        ),

        intermittent_edges=(
            "lifecycle_class",
            lambda x:
                class_count(
                    x,
                    "Intermittent"
                )
        ),

        two_episode_edges=(
            "lifecycle_class",
            lambda x:
                class_count(
                    x,
                    "Two-episode transitional"
                )
        ),

        rare_edges=(
            "lifecycle_class",
            lambda x:
                class_count(
                    x,
                    "Rare"
                )
        )
    )
)


for col in [
    "persistent_core_edges",
    "regime_dependent_edges",
    "intermittent_edges",
    "two_episode_edges",
    "rare_edges"
]:

    lifecycle_summary_df[
        col.replace(
            "_edges",
            "_rate"
        )
    ] = (
        lifecycle_summary_df[
            col
        ]
        /
        lifecycle_summary_df[
            "n_possible_pairs"
        ]
    )


lifecycle_summary_df.to_csv(
    LIFECYCLE_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. Active Non-core Lifecycle
#
# 目的：
# 避免大量从未出现/只出现1次的边把平均transition rate压低。
#
# 条件：
# persistence < .8
# windows_selected >= 3
# ============================================================

active_noncore_df = lifecycle_df[
    (
        lifecycle_df[
            "persistence"
        ]
        <
        PERSISTENCE_THRESHOLD
    )
    &
    (
        lifecycle_df[
            "windows_selected"
        ]
        >=
        3
    )
].copy()


active_noncore_summary_df = (
    active_noncore_df
    .groupby(
        "industry_relation",
        as_index=False
    )
    .agg(

        n_active_noncore_edges=(
            "stock_1",
            "size"
        ),

        mean_persistence=(
            "persistence",
            "mean"
        ),

        mean_n_episodes=(
            "n_episodes",
            "mean"
        ),

        mean_transition_rate=(
            "transition_rate",
            "mean"
        ),

        mean_longest_run_share=(
            "longest_run_share",
            "mean"
        ),

        regime_dependent_edges=(
            "lifecycle_class",
            lambda x:
                class_count(
                    x,
                    "Regime-dependent"
                )
        ),

        intermittent_edges=(
            "lifecycle_class",
            lambda x:
                class_count(
                    x,
                    "Intermittent"
                )
        ),

        two_episode_edges=(
            "lifecycle_class",
            lambda x:
                class_count(
                    x,
                    "Two-episode transitional"
                )
        )
    )
)


for col in [
    "regime_dependent_edges",
    "intermittent_edges",
    "two_episode_edges"
]:

    active_noncore_summary_df[
        col.replace(
            "_edges",
            "_rate"
        )
    ] = (
        active_noncore_summary_df[
            col
        ]
        /
        active_noncore_summary_df[
            "n_active_noncore_edges"
        ]
    )


active_noncore_summary_df.to_csv(
    ACTIVE_NONCORE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PART E
# 综合Evidence Matrix
#
# 这是描述性判据，不是统计显著性检验。
# ============================================================

evidence_rows = []


# ------------------------------------------------------------
# 14. 每个Window Size的Evidence
# ------------------------------------------------------------

for row in contrast_df.itertuples():

    W = row.window_size


    evidence_rows.append(
        {
            "evidence_group":
                "Multi-scale persistence",

            "window_size":
                W,

            "criterion":
                "Same mean persistence > Cross mean persistence",

            "support":
                (
                    row.same_mean_persistence
                    >
                    row.cross_mean_persistence
                ),

            "same_value":
                row.same_mean_persistence,

            "cross_value":
                row.cross_mean_persistence
        }
    )


    evidence_rows.append(
        {
            "evidence_group":
                "Persistent-edge rate",

            "window_size":
                W,

            "criterion":
                "Same persistent rate > Cross persistent rate",

            "support":
                (
                    row.same_persistent_rate
                    >
                    row.cross_persistent_rate
                ),

            "same_value":
                row.same_persistent_rate,

            "cross_value":
                row.cross_persistent_rate
        }
    )


    evidence_rows.append(
        {
            "evidence_group":
                "State-change dynamics",

            "window_size":
                W,

            "criterion":
                "Cross normalized change rate > Same normalized change rate",

            "support":
                (
                    row.cross_state_change_rate
                    >
                    row.same_state_change_rate
                ),

            "same_value":
                row.same_state_change_rate,

            "cross_value":
                row.cross_state_change_rate
        }
    )


    evidence_rows.append(
        {
            "evidence_group":
                "Exit dynamics",

            "window_size":
                W,

            "criterion":
                "Cross exit rate > Same exit rate",

            "support":
                (
                    row.cross_exit_rate
                    >
                    row.same_exit_rate
                ),

            "same_value":
                row.same_exit_rate,

            "cross_value":
                row.cross_exit_rate
        }
    )


# ------------------------------------------------------------
# 15. Multi-scale Core Evidence
# ------------------------------------------------------------

same_multi = (
    multiscale_summary_df[
        multiscale_summary_df[
            "industry_relation"
        ]
        ==
        "Same industry"
    ]
    .iloc[0]
)


cross_multi = (
    multiscale_summary_df[
        multiscale_summary_df[
            "industry_relation"
        ]
        ==
        "Cross industry"
    ]
    .iloc[0]
)


evidence_rows.append(
    {
        "evidence_group":
            "True multi-scale core",

        "window_size":
            "All",

        "criterion":
            "Same multi-scale core rate > Cross multi-scale core rate",

        "support":
            (
                same_multi[
                    "multiscale_persistent_rate"
                ]
                >
                cross_multi[
                    "multiscale_persistent_rate"
                ]
            ),

        "same_value":
            same_multi[
                "multiscale_persistent_rate"
            ],

        "cross_value":
            cross_multi[
                "multiscale_persistent_rate"
            ]
    }
)


evidence_rows.append(
    {
        "evidence_group":
            "Always multi-scale core",

        "window_size":
            "All",

        "criterion":
            "Same always-core rate > Cross always-core rate",

        "support":
            (
                same_multi[
                    "multiscale_always_rate"
                ]
                >
                cross_multi[
                    "multiscale_always_rate"
                ]
            ),

        "same_value":
            same_multi[
                "multiscale_always_rate"
            ],

        "cross_value":
            cross_multi[
                "multiscale_always_rate"
            ]
    }
)


evidence_df = pd.DataFrame(
    evidence_rows
)


evidence_df.to_csv(
    EVIDENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 屏幕输出
# ============================================================

print(
    "\n======================================"
)

print(
    "Stage 6：Scale-level Same vs Cross"
)

print(
    "======================================"
)


print(
    contrast_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Multi-scale Core Industry Summary"
)

print(
    "======================================"
)


print(
    multiscale_summary_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Lifecycle Industry Summary"
)

print(
    "======================================"
)


print(
    lifecycle_summary_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Active Non-core Lifecycle Summary"
)

print(
    "======================================"
)


print(
    active_noncore_summary_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Integrated Evidence Matrix"
)

print(
    "======================================"
)


print(
    evidence_df.to_string(
        index=False
    )
)


print(
    "\n支持的描述性判据数：",
    int(
        evidence_df[
            "support"
        ]
        .sum()
    ),
    "/",
    len(
        evidence_df
    )
)


# ============================================================
# 17. 图1：
# Same vs Cross Mean Persistence
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


for relation in [
    "Same industry",
    "Cross industry"
]:

    temp = scale_validation_df[
        scale_validation_df[
            "industry_relation"
        ]
        ==
        relation
    ]


    label = (
        "同行业"
        if relation
        ==
        "Same industry"
        else
        "跨行业"
    )


    ax.plot(
        temp[
            "window_size"
        ],
        temp[
            "mean_persistence"
        ],
        marker="o",
        label=label
    )


ax.set_xlabel(
    "Rolling Window Size"
)

ax.set_ylabel(
    "Mean Edge Persistence"
)

ax.set_title(
    "不同时间尺度下同行业与跨行业边的Persistence"
)

ax.set_xticks(
    WINDOW_SIZES
)

ax.set_ylim(
    0,
    1.05
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


PERSISTENCE_FIGURE = (
    FIGURE_DIR
    / "stage6_same_cross_mean_persistence.png"
)


fig.savefig(
    PERSISTENCE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 18. 图2：
# Persistent Rate >= .8
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


for relation in [
    "Same industry",
    "Cross industry"
]:

    temp = scale_validation_df[
        scale_validation_df[
            "industry_relation"
        ]
        ==
        relation
    ]


    label = (
        "同行业"
        if relation
        ==
        "Same industry"
        else
        "跨行业"
    )


    ax.plot(
        temp[
            "window_size"
        ],
        temp[
            "persistent_rate"
        ],
        marker="o",
        label=label
    )


ax.set_xlabel(
    "Rolling Window Size"
)

ax.set_ylabel(
    "Share of edges with Persistence >= 0.8"
)

ax.set_title(
    "不同时间尺度下的高持续边比例"
)

ax.set_xticks(
    WINDOW_SIZES
)

ax.set_ylim(
    0,
    1.05
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


PERSISTENT_RATE_FIGURE = (
    FIGURE_DIR
    / "stage6_same_cross_persistent_rate.png"
)


fig.savefig(
    PERSISTENT_RATE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 19. 图3：
# Normalized State Change Rate
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


for relation in [
    "Same industry",
    "Cross industry"
]:

    temp = scale_validation_df[
        scale_validation_df[
            "industry_relation"
        ]
        ==
        relation
    ]


    label = (
        "同行业"
        if relation
        ==
        "Same industry"
        else
        "跨行业"
    )


    ax.plot(
        temp[
            "window_size"
        ],
        temp[
            "state_change_rate"
        ],
        marker="o",
        label=label
    )


ax.set_xlabel(
    "Rolling Window Size"
)

ax.set_ylabel(
    "Normalized State-change Rate"
)

ax.set_title(
    "同行业与跨行业边的支持集变化率"
)

ax.set_xticks(
    WINDOW_SIZES
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


CHANGE_FIGURE = (
    FIGURE_DIR
    / "stage6_same_cross_state_change_rate.png"
)


fig.savefig(
    CHANGE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 20. 图4：
# Multi-scale Core Rate
# ============================================================

plot_multi = (
    multiscale_summary_df
    .copy()
)


labels = [
    (
        "同行业"
        if x == "Same industry"
        else "跨行业"
    )
    for x in plot_multi[
        "industry_relation"
    ]
]


x = np.arange(
    len(
        plot_multi
    )
)


width = 0.35


fig, ax = plt.subplots(
    figsize=(
        8,
        6
    )
)


ax.bar(
    x - width / 2,
    plot_multi[
        "multiscale_persistent_rate"
    ],
    width,
    label="Multi-scale Persistent"
)


ax.bar(
    x + width / 2,
    plot_multi[
        "multiscale_always_rate"
    ],
    width,
    label="Always Persistent"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "Share of Candidate Edges"
)

ax.set_title(
    "真正跨尺度核心边的行业结构"
)

ax.legend()


fig.tight_layout()


MULTISCALE_FIGURE = (
    FIGURE_DIR
    / "stage6_multiscale_core_rate.png"
)


fig.savefig(
    MULTISCALE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 21. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Stage 6综合验证完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for path in [

    SCALE_VALIDATION_FILE,
    SCALE_CONTRAST_FILE,
    MULTISCALE_SUMMARY_FILE,
    LIFECYCLE_SUMMARY_FILE,
    ACTIVE_NONCORE_FILE,
    EVIDENCE_FILE,
    PERSISTENCE_FIGURE,
    PERSISTENT_RATE_FIGURE,
    CHANGE_FIGURE,
    MULTISCALE_FIGURE

]:

    print(
        path
    )
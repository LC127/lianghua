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
        "中文股票名称可能无法正常显示。"
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

ROLLING_EDGE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_history.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)

# 可选：
# 用于研究昨天Static/Dynamic分类不一致的7条边
STATIC_DYNAMIC_FILE = (
    PROCESSED_DIR
    / "static_stable_vs_dynamic_persistent_all_pairs.csv"
)


# ------------------------------------------------------------
# 输出
# ------------------------------------------------------------

LIFECYCLE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_lifecycle.csv"
)

EPISODE_FILE = (
    PROCESSED_DIR
    / "rolling_glasso_edge_lifecycle_episodes.csv"
)

PERSISTENT_CORE_FILE = (
    PROCESSED_DIR
    / "persistent_core_edges_lifecycle.csv"
)

REGIME_FILE = (
    PROCESSED_DIR
    / "regime_dependent_edges.csv"
)

INTERMITTENT_FILE = (
    PROCESSED_DIR
    / "intermittent_edges.csv"
)

TRANSITIONAL_FILE = (
    PROCESSED_DIR
    / "two_episode_transitional_edges.csv"
)

RARE_FILE = (
    PROCESSED_DIR
    / "rare_edges.csv"
)

CLASS_SUMMARY_FILE = (
    PROCESSED_DIR
    / "edge_lifecycle_class_summary.csv"
)

DISAGREEMENT_FILE = (
    PROCESSED_DIR
    / "static_dynamic_disagreement_edge_lifecycle.csv"
)


# ============================================================
# 2. 参数
# ============================================================

PERSISTENCE_THRESHOLD = 0.80

MIN_NONRARE_SELECTED = 3

INTERMITTENT_MIN_EPISODES = 3


# ============================================================
# 3. 工具函数
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


def convert_boolean_column(
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

        bad = (
            series[
                converted.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"{column_name}中存在无法识别的布尔值："
            f"{bad}"
        )

    return converted.astype(bool)


# ============================================================
# 4. 股票代码 -> 名称/行业映射
#
# 优先使用stock_info.csv，
# 避免之前canonicalize导致的name/industry错位问题
# ============================================================

if STOCK_INFO_FILE.exists():

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
            c
            for c in code_candidates
            if c in stock_info.columns
        ),
        None
    )

    name_col = next(
        (
            c
            for c in name_candidates
            if c in stock_info.columns
        ),
        None
    )

    industry_col = next(
        (
            c
            for c in industry_candidates
            if c in stock_info.columns
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
            "stock_info.csv缺少股票代码、名称或行业字段。"
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
        .reset_index(
            drop=True
        )
    )

    print(
        "使用stock_info.csv建立股票名称/行业映射。"
    )


else:

    # --------------------------------------------------------
    # fallback：
    # 从尚未canonicalize的原始rolling文件建立映射
    # --------------------------------------------------------

    raw_meta = pd.read_csv(
        ROLLING_EDGE_FILE,
        dtype={
            "stock_1": str,
            "stock_2": str
        }
    )

    raw_meta["stock_1"] = (
        raw_meta["stock_1"]
        .apply(normalize_code)
    )

    raw_meta["stock_2"] = (
        raw_meta["stock_2"]
        .apply(normalize_code)
    )

    meta_1 = (
        raw_meta[
            [
                "stock_1",
                "name_1",
                "industry_1"
            ]
        ]
        .rename(
            columns={
                "stock_1": "code",
                "name_1": "name",
                "industry_1": "industry"
            }
        )
    )

    meta_2 = (
        raw_meta[
            [
                "stock_2",
                "name_2",
                "industry_2"
            ]
        ]
        .rename(
            columns={
                "stock_2": "code",
                "name_2": "name",
                "industry_2": "industry"
            }
        )
    )

    metadata = pd.concat(
        [
            meta_1,
            meta_2
        ],
        ignore_index=True
    )

    # 名称冲突检查
    conflict_name = (
        metadata
        .groupby("code")["name"]
        .nunique(
            dropna=True
        )
    )

    if (
        conflict_name > 1
    ).any():

        raise ValueError(
            "原始rolling文件中存在股票代码名称映射冲突。"
        )

    conflict_industry = (
        metadata
        .groupby("code")["industry"]
        .nunique(
            dropna=True
        )
    )

    if (
        conflict_industry > 1
    ).any():

        raise ValueError(
            "原始rolling文件中存在股票行业映射冲突。"
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

    print(
        "未找到stock_info.csv，"
        "从原始rolling文件建立股票映射。"
    )


name_map = dict(
    zip(
        metadata["code"],
        metadata["name"]
    )
)

industry_map = dict(
    zip(
        metadata["code"],
        metadata["industry"]
    )
)


# ============================================================
# 5. 读取Rolling Edge History
# ============================================================

df = pd.read_csv(
    ROLLING_EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


df["stock_1"] = (
    df["stock_1"]
    .apply(normalize_code)
)

df["stock_2"] = (
    df["stock_2"]
    .apply(normalize_code)
)


# ============================================================
# 6. 股票对canonicalize
# ============================================================

pairs = df.apply(
    lambda row:
        canonical_pair(
            row["stock_1"],
            row["stock_2"]
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
# 7. canonicalize后重新映射名称/行业
# ============================================================

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

df["same_industry"] = (
    df["industry_1"]
    ==
    df["industry_2"]
)


# ============================================================
# 8. selected转换
# ============================================================

df["selected"] = (
    convert_boolean_column(
        df["selected"],
        "selected"
    )
)


# ============================================================
# 9. 日期
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
        "文件中缺少network_date/window_end。"
    )


df[DATE_COL] = pd.to_datetime(
    df[DATE_COL]
)


# ============================================================
# 10. window_id检查
# ============================================================

if "window_id" not in df.columns:

    # 根据日期自动生成window_id
    date_order = (
        df[DATE_COL]
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

    df["window_id"] = (
        df[DATE_COL]
        .map(date_to_id)
    )


df["window_id"] = (
    pd.to_numeric(
        df["window_id"]
    )
    .astype(int)
)


# ============================================================
# 11. 排序
# ============================================================

df = (
    df
    .sort_values(
        [
            "stock_1",
            "stock_2",
            "window_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 12. 检查窗口数量
# ============================================================

all_window_ids = sorted(
    df["window_id"]
    .unique()
)


TOTAL_WINDOWS = len(
    all_window_ids
)


print(
    "\n总Rolling Windows：",
    TOTAL_WINDOWS
)


# 每个股票对应有TOTAL_WINDOWS条记录
pair_counts = (
    df
    .groupby(
        [
            "stock_1",
            "stock_2"
        ]
    )["window_id"]
    .nunique()
)


bad_pairs = pair_counts[
    pair_counts
    !=
    TOTAL_WINDOWS
]


if len(
    bad_pairs
) > 0:

    print(
        bad_pairs
    )

    raise ValueError(
        "部分股票对缺少Rolling Window记录。"
    )


# ============================================================
# 13. Episode提取函数
# ============================================================

def extract_episodes(
    selected_array,
    window_ids,
    dates
):

    """
    输入：
        selected_array:
            True/False序列

    返回：
        episodes:
            每个连续True区间的信息
    """

    episodes = []

    start_idx = None


    for i, selected in enumerate(
        selected_array
    ):

        # episode开始
        if (
            selected
            and
            start_idx is None
        ):

            start_idx = i


        # episode结束
        is_last = (
            i
            ==
            len(selected_array) - 1
        )


        if (
            start_idx is not None
            and
            (
                (
                    not selected
                )
                or
                (
                    selected
                    and
                    is_last
                )
            )
        ):

            if (
                not selected
            ):

                end_idx = i - 1

            else:

                end_idx = i


            length = (
                end_idx
                -
                start_idx
                +
                1
            )


            episodes.append(
                {
                    "start_position":
                        start_idx,

                    "end_position":
                        end_idx,

                    "start_window_id":
                        int(
                            window_ids[
                                start_idx
                            ]
                        ),

                    "end_window_id":
                        int(
                            window_ids[
                                end_idx
                            ]
                        ),

                    "start_date":
                        dates[
                            start_idx
                        ],

                    "end_date":
                        dates[
                            end_idx
                        ],

                    "length":
                        int(
                            length
                        )
                }
            )


            start_idx = None


    return episodes


# ============================================================
# 14. 对每条边计算Lifecycle
# ============================================================

lifecycle_rows = []

episode_rows = []


grouped = df.groupby(
    [
        "stock_1",
        "stock_2"
    ],
    sort=False
)


for (
    stock_1,
    stock_2
), group in grouped:

    group = (
        group
        .sort_values(
            "window_id"
        )
        .reset_index(
            drop=True
        )
    )


    selected = (
        group["selected"]
        .to_numpy(
            dtype=bool
        )
    )


    window_ids = (
        group["window_id"]
        .to_numpy()
    )


    dates = (
        group[DATE_COL]
        .to_numpy()
    )


    n_selected = int(
        selected.sum()
    )


    persistence = (
        n_selected
        /
        TOTAL_WINDOWS
    )


    # --------------------------------------------------------
    # Episode
    # --------------------------------------------------------

    episodes = extract_episodes(
        selected,
        window_ids,
        dates
    )


    n_episodes = len(
        episodes
    )


    if n_episodes > 0:

        episode_lengths = [
            episode["length"]
            for episode in episodes
        ]


        longest_run = max(
            episode_lengths
        )


        mean_episode_length = float(
            np.mean(
                episode_lengths
            )
        )


        longest_episode_index = int(
            np.argmax(
                episode_lengths
            )
        )


        longest_episode = episodes[
            longest_episode_index
        ]


        first_selected_date = (
            episodes[0][
                "start_date"
            ]
        )


        last_selected_date = (
            episodes[-1][
                "end_date"
            ]
        )


        longest_run_start_date = (
            longest_episode[
                "start_date"
            ]
        )


        longest_run_end_date = (
            longest_episode[
                "end_date"
            ]
        )


        longest_run_share = (
            longest_run
            /
            n_selected
        )


    else:

        longest_run = 0

        mean_episode_length = 0.0

        first_selected_date = pd.NaT

        last_selected_date = pd.NaT

        longest_run_start_date = pd.NaT

        longest_run_end_date = pd.NaT

        longest_run_share = np.nan


    # --------------------------------------------------------
    # Switches
    # --------------------------------------------------------

    if TOTAL_WINDOWS > 1:

        switches = (
            selected[1:]
            !=
            selected[:-1]
        )


        switch_count = int(
            switches.sum()
        )


        transition_rate = (
            switch_count
            /
            (
                TOTAL_WINDOWS - 1
            )
        )


        entry_count = int(
            (
                (~selected[:-1])
                &
                selected[1:]
            )
            .sum()
        )


        exit_count = int(
            (
                selected[:-1]
                &
                (~selected[1:])
            )
            .sum()
        )


    else:

        switch_count = 0
        transition_rate = 0.0
        entry_count = 0
        exit_count = 0


    # --------------------------------------------------------
    # 被选择时的Partial
    # --------------------------------------------------------

    selected_group = group[
        group["selected"]
    ]


    if len(
        selected_group
    ) > 0:

        mean_partial_when_selected = (
            selected_group[
                "partial_correlation"
            ]
            .mean()
        )


        mean_abs_partial_when_selected = (
            selected_group[
                "abs_partial_correlation"
            ]
            .mean()
        )


        max_abs_partial_when_selected = (
            selected_group[
                "abs_partial_correlation"
            ]
            .max()
        )


        signs = np.sign(
            selected_group[
                "partial_correlation"
            ]
            .to_numpy()
        )


        positive_count = int(
            (
                signs > 0
            )
            .sum()
        )


        negative_count = int(
            (
                signs < 0
            )
            .sum()
        )


        sign_consistency = (
            max(
                positive_count,
                negative_count
            )
            /
            len(
                selected_group
            )
        )


    else:

        mean_partial_when_selected = np.nan
        mean_abs_partial_when_selected = np.nan
        max_abs_partial_when_selected = np.nan
        sign_consistency = np.nan


    # --------------------------------------------------------
    # Lifecycle分类
    # --------------------------------------------------------

    if (
        persistence
        >=
        PERSISTENCE_THRESHOLD
    ):

        lifecycle_class = (
            "Persistent core"
        )


    elif (
        n_selected
        <=
        2
    ):

        lifecycle_class = (
            "Rare"
        )


    elif (
        n_episodes
        ==
        1
        and
        longest_run
        >=
        MIN_NONRARE_SELECTED
    ):

        lifecycle_class = (
            "Regime-dependent"
        )


    elif (
        n_episodes
        >=
        INTERMITTENT_MIN_EPISODES
    ):

        lifecycle_class = (
            "Intermittent"
        )


    elif (
        n_episodes
        ==
        2
    ):

        lifecycle_class = (
            "Two-episode transitional"
        )


    else:

        lifecycle_class = (
            "Other"
        )


    # --------------------------------------------------------
    # 选择序列字符串
    # --------------------------------------------------------

    selection_sequence = "".join(
        "1"
        if x
        else
        "0"

        for x in selected
    )


    lifecycle_rows.append(
        {
            "stock_1":
                stock_1,

            "name_1":
                name_map[
                    stock_1
                ],

            "industry_1":
                industry_map[
                    stock_1
                ],

            "stock_2":
                stock_2,

            "name_2":
                name_map[
                    stock_2
                ],

            "industry_2":
                industry_map[
                    stock_2
                ],

            "same_industry":
                (
                    industry_map[
                        stock_1
                    ]
                    ==
                    industry_map[
                        stock_2
                    ]
                ),

            "total_windows":
                TOTAL_WINDOWS,

            "windows_selected":
                n_selected,

            "persistence":
                persistence,

            "first_selected_date":
                first_selected_date,

            "last_selected_date":
                last_selected_date,

            "n_episodes":
                n_episodes,

            "longest_consecutive_run":
                longest_run,

            "longest_run_start_date":
                longest_run_start_date,

            "longest_run_end_date":
                longest_run_end_date,

            "longest_run_share":
                longest_run_share,

            "mean_episode_length":
                mean_episode_length,

            "switch_count":
                switch_count,

            "entry_count":
                entry_count,

            "exit_count":
                exit_count,

            "transition_rate":
                transition_rate,

            "mean_partial_when_selected":
                mean_partial_when_selected,

            "mean_abs_partial_when_selected":
                mean_abs_partial_when_selected,

            "max_abs_partial_when_selected":
                max_abs_partial_when_selected,

            "sign_consistency":
                sign_consistency,

            "selection_sequence":
                selection_sequence,

            "lifecycle_class":
                lifecycle_class
        }
    )


    # --------------------------------------------------------
    # Episode-level文件
    # --------------------------------------------------------

    for episode_number, episode in enumerate(
        episodes,
        start=1
    ):

        episode_rows.append(
            {
                "stock_1":
                    stock_1,

                "name_1":
                    name_map[
                        stock_1
                    ],

                "stock_2":
                    stock_2,

                "name_2":
                    name_map[
                        stock_2
                    ],

                "same_industry":
                    (
                        industry_map[
                            stock_1
                        ]
                        ==
                        industry_map[
                            stock_2
                        ]
                    ),

                "episode_number":
                    episode_number,

                "episode_start_window":
                    episode[
                        "start_window_id"
                    ],

                "episode_end_window":
                    episode[
                        "end_window_id"
                    ],

                "episode_start_date":
                    episode[
                        "start_date"
                    ],

                "episode_end_date":
                    episode[
                        "end_date"
                    ],

                "episode_length":
                    episode[
                        "length"
                    ]
            }
        )


# ============================================================
# 15. DataFrame
# ============================================================

lifecycle_df = pd.DataFrame(
    lifecycle_rows
)


episode_df = pd.DataFrame(
    episode_rows
)


# ============================================================
# 16. 排序
# ============================================================

lifecycle_df = (
    lifecycle_df
    .sort_values(
        [
            "persistence",
            "longest_consecutive_run",
            "mean_abs_partial_when_selected"
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
# 17. 分类结果
# ============================================================

persistent_core_df = lifecycle_df[
    lifecycle_df[
        "lifecycle_class"
    ]
    ==
    "Persistent core"
].copy()


regime_df = lifecycle_df[
    lifecycle_df[
        "lifecycle_class"
    ]
    ==
    "Regime-dependent"
].copy()


intermittent_df = lifecycle_df[
    lifecycle_df[
        "lifecycle_class"
    ]
    ==
    "Intermittent"
].copy()


transitional_df = lifecycle_df[
    lifecycle_df[
        "lifecycle_class"
    ]
    ==
    "Two-episode transitional"
].copy()


rare_df = lifecycle_df[
    lifecycle_df[
        "lifecycle_class"
    ]
    ==
    "Rare"
].copy()


# ============================================================
# 18. Lifecycle Class Summary
# ============================================================

class_summary_df = (
    lifecycle_df
    .groupby(
        "lifecycle_class",
        as_index=False
    )
    .agg(

        n_edges=(
            "stock_1",
            "size"
        ),

        mean_persistence=(
            "persistence",
            "mean"
        ),

        mean_episodes=(
            "n_episodes",
            "mean"
        ),

        mean_longest_run=(
            "longest_consecutive_run",
            "mean"
        ),

        mean_transition_rate=(
            "transition_rate",
            "mean"
        ),

        mean_abs_partial=(
            "mean_abs_partial_when_selected",
            "mean"
        ),

        same_industry_edges=(
            "same_industry",
            "sum"
        )
    )
)


class_summary_df[
    "same_industry_ratio"
] = (
    class_summary_df[
        "same_industry_edges"
    ]
    /
    class_summary_df[
        "n_edges"
    ]
)


# ============================================================
# 19. 保存
# ============================================================

lifecycle_df.to_csv(
    LIFECYCLE_FILE,
    index=False,
    encoding="utf-8-sig"
)


episode_df.to_csv(
    EPISODE_FILE,
    index=False,
    encoding="utf-8-sig"
)


persistent_core_df.to_csv(
    PERSISTENT_CORE_FILE,
    index=False,
    encoding="utf-8-sig"
)


regime_df.to_csv(
    REGIME_FILE,
    index=False,
    encoding="utf-8-sig"
)


intermittent_df.to_csv(
    INTERMITTENT_FILE,
    index=False,
    encoding="utf-8-sig"
)


transitional_df.to_csv(
    TRANSITIONAL_FILE,
    index=False,
    encoding="utf-8-sig"
)


rare_df.to_csv(
    RARE_FILE,
    index=False,
    encoding="utf-8-sig"
)


class_summary_df.to_csv(
    CLASS_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 可选：
# Static Stable vs Dynamic Persistent分类不一致的边
# ============================================================

if STATIC_DYNAMIC_FILE.exists():

    compare = pd.read_csv(
        STATIC_DYNAMIC_FILE,
        dtype={
            "stock_1": str,
            "stock_2": str
        }
    )


    compare[
        "stock_1"
    ] = (
        compare[
            "stock_1"
        ]
        .apply(
            normalize_code
        )
    )


    compare[
        "stock_2"
    ] = (
        compare[
            "stock_2"
        ]
        .apply(
            normalize_code
        )
    )


    comparison_pairs = compare.apply(

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


    compare[
        "stock_1"
    ] = [
        x[0]
        for x in comparison_pairs
    ]


    compare[
        "stock_2"
    ] = [
        x[1]
        for x in comparison_pairs
    ]


    disagreement = compare[
        compare[
            "stability_class"
        ]
        .isin(
            [
                "Static stable only",
                "Dynamic persistent only"
            ]
        )
    ][
        [
            "stock_1",
            "stock_2",
            "stability_class"
        ]
    ].copy()


    disagreement_lifecycle = (
        disagreement
        .merge(
            lifecycle_df,

            on=[
                "stock_1",
                "stock_2"
            ],

            how="left"
        )
    )


    disagreement_lifecycle.to_csv(
        DISAGREEMENT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 21. 打印主要结果
# ============================================================

print(
    "\n======================================"
)

print(
    "Edge Lifecycle Class Summary"
)

print(
    "======================================"
)


print(
    class_summary_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "Regime-dependent Edges"
)

print(
    "======================================"
)


if len(
    regime_df
) > 0:

    print(
        regime_df[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "persistence",
                "n_episodes",
                "longest_consecutive_run",
                "longest_run_start_date",
                "longest_run_end_date",
                "transition_rate",
                "selection_sequence"
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有Regime-dependent Edges。"
    )


print(
    "\n======================================"
)

print(
    "Intermittent Edges"
)

print(
    "======================================"
)


if len(
    intermittent_df
) > 0:

    print(
        intermittent_df[
            [
                "stock_1",
                "name_1",
                "stock_2",
                "name_2",
                "persistence",
                "n_episodes",
                "longest_consecutive_run",
                "switch_count",
                "transition_rate",
                "selection_sequence"
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "没有Intermittent Edges。"
    )


# ============================================================
# 22. 图1：Lifecycle类别数量
# ============================================================

class_order = [
    "Persistent core",
    "Regime-dependent",
    "Two-episode transitional",
    "Intermittent",
    "Rare",
    "Other"
]


class_counts = (
    lifecycle_df[
        "lifecycle_class"
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
    "Edge Lifecycle Classification"
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
        value + 0.3,
        str(
            value
        ),
        ha="center"
    )


fig.tight_layout()


CLASS_FIGURE = (
    FIGURE_DIR
    / "edge_lifecycle_class_counts.png"
)


fig.savefig(
    CLASS_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 图2：
# Persistence vs Number of Episodes
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


ax.scatter(
    lifecycle_df[
        "persistence"
    ],
    lifecycle_df[
        "n_episodes"
    ],
    alpha=0.7
)


ax.axvline(
    x=PERSISTENCE_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Persistence"
)


ax.set_ylabel(
    "Number of episodes"
)


ax.set_title(
    "Persistence与Edge Episodes"
)


# 标注episodes最多的10条边
label_df = (
    lifecycle_df
    .sort_values(
        [
            "n_episodes",
            "transition_rate"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(10)
)


for row in label_df.itertuples():

    label = (
        f"{row.name_1}-{row.name_2}"
    )


    ax.annotate(
        label,

        (
            row.persistence,
            row.n_episodes
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points",

        fontsize=8
    )


fig.tight_layout()


EPISODE_FIGURE = (
    FIGURE_DIR
    / "edge_lifecycle_persistence_vs_episodes.png"
)


fig.savefig(
    EPISODE_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. 图3：
# Persistence vs Longest Run Share
# ============================================================

plot_df = lifecycle_df[
    lifecycle_df[
        "windows_selected"
    ]
    >
    0
].copy()


fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


ax.scatter(
    plot_df[
        "persistence"
    ],
    plot_df[
        "longest_run_share"
    ],
    alpha=0.7
)


ax.axvline(
    x=PERSISTENCE_THRESHOLD,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Persistence"
)


ax.set_ylabel(
    "Longest Run / Selected Windows"
)


ax.set_title(
    "边的持续程度与时间集中度"
)


# 标注最长run share较高但Persistence<0.8的边
label_df = (
    plot_df[
        plot_df[
            "persistence"
        ]
        <
        PERSISTENCE_THRESHOLD
    ]
    .sort_values(
        [
            "longest_run_share",
            "longest_consecutive_run"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(10)
)


for row in label_df.itertuples():

    label = (
        f"{row.name_1}-{row.name_2}"
    )


    ax.annotate(
        label,

        (
            row.persistence,
            row.longest_run_share
        ),

        xytext=(
            5,
            5
        ),

        textcoords="offset points",

        fontsize=8
    )


fig.tight_layout()


CONCENTRATION_FIGURE = (
    FIGURE_DIR
    / "edge_lifecycle_persistence_vs_run_share.png"
)


fig.savefig(
    CONCENTRATION_FIGURE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. 图4：
# 关键Lifecycle Edges时间序列Heatmap
# ============================================================

# 优先：
# Regime-dependent + Intermittent
key_edges = pd.concat(
    [
        regime_df,
        intermittent_df
    ],
    ignore_index=True
)


# 最多画25条，避免过密
if len(
    key_edges
) > 0:

    key_edges = (
        key_edges
        .sort_values(
            [
                "persistence",
                "longest_consecutive_run"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(25)
        .copy()
    )


    heatmap_rows = []

    heatmap_labels = []


    for row in key_edges.itertuples():

        group = df[
            (
                df[
                    "stock_1"
                ]
                ==
                row.stock_1
            )
            &
            (
                df[
                    "stock_2"
                ]
                ==
                row.stock_2
            )
        ].sort_values(
            "window_id"
        )


        heatmap_rows.append(
            group[
                "selected"
            ]
            .astype(int)
            .to_numpy()
        )


        heatmap_labels.append(
            (
                f"{row.name_1}({row.stock_1})"
                f" — "
                f"{row.name_2}({row.stock_2})"
            )
        )


    matrix = np.array(
        heatmap_rows
    )


    fig_height = max(
        6,
        len(
            key_edges
        )
        *
        0.4
    )


    fig, ax = plt.subplots(
        figsize=(
            14,
            fig_height
        )
    )


    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1
    )


    ax.set_yticks(
        np.arange(
            len(
                heatmap_labels
            )
        )
    )


    ax.set_yticklabels(
        heatmap_labels,
        fontsize=8
    )


    # X轴使用Network Date
    date_table = (
        df[
            [
                "window_id",
                DATE_COL
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "window_id"
        )
    )


    date_labels = (
        date_table[
            DATE_COL
        ]
        .dt.strftime(
            "%Y-%m"
        )
        .tolist()
    )


    tick_step = max(
        1,
        len(
            date_labels
        )
        //
        10
    )


    x_ticks = np.arange(
        0,
        len(
            date_labels
        ),
        tick_step
    )


    ax.set_xticks(
        x_ticks
    )


    ax.set_xticklabels(
        [
            date_labels[i]
            for i in x_ticks
        ],
        rotation=45,
        ha="right"
    )


    ax.set_xlabel(
        "Network Date"
    )


    ax.set_ylabel(
        "股票条件关联边"
    )


    ax.set_title(
        "Regime-dependent与Intermittent Edge Lifecycle"
    )


    colorbar = fig.colorbar(
        image,
        ax=ax
    )


    colorbar.set_label(
        "Selected (0/1)"
    )


    fig.tight_layout()


    TIMELINE_FIGURE = (
        FIGURE_DIR
        / "edge_lifecycle_selected_timeline.png"
    )


    fig.savefig(
        TIMELINE_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


# ============================================================
# 26. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Edge Lifecycle分析完成"
)

print(
    "======================================"
)


print(
    "\n主要输出文件："
)


for path in [

    LIFECYCLE_FILE,
    EPISODE_FILE,
    PERSISTENT_CORE_FILE,
    REGIME_FILE,
    INTERMITTENT_FILE,
    TRANSITIONAL_FILE,
    RARE_FILE,
    CLASS_SUMMARY_FILE,
    CLASS_FIGURE,
    EPISODE_FIGURE,
    CONCENTRATION_FIGURE

]:

    print(
        path
    )
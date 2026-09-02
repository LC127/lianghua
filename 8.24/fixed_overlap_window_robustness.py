from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# 1. 路径
# ============================================================

PROJECT_DIR = Path("stock_network")

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


RETURNS_FILE = (
    PROCESSED_DIR
    / "stock_returns.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)


# ============================================================
# 2. 输出
# ============================================================

NETWORK_SUMMARY_FILE = (
    PROCESSED_DIR
    / "overlap_control_network_summary.csv"
)

EDGE_HISTORY_FILE = (
    PROCESSED_DIR
    / "overlap_control_edge_history.csv"
)

TURNOVER_FILE = (
    PROCESSED_DIR
    / "overlap_control_turnover.csv"
)

EDGE_PERSISTENCE_FILE = (
    PROCESSED_DIR
    / "overlap_control_edge_persistence.csv"
)

INDUSTRY_SUMMARY_FILE = (
    PROCESSED_DIR
    / "overlap_control_industry_persistence_summary.csv"
)

WINDOW_SUMMARY_FILE = (
    PROCESSED_DIR
    / "overlap_control_window_summary.csv"
)

DESIGN_COMPARISON_FILE = (
    PROCESSED_DIR
    / "overlap_control_design_comparison.csv"
)

DIAGNOSTICS_FILE = (
    PROCESSED_DIR
    / "overlap_control_glasso_diagnostics.csv"
)


# ============================================================
# 3. Graphical Lasso参数
# ============================================================

# 保持前面研究使用的固定1-SE alpha
ALPHA = 0.216910

MAX_ITER = 5000

TOL = 1e-4

EDGE_TOL = 1e-8

PERSISTENCE_THRESHOLD = 0.80


# ============================================================
# 4. 两个实验设计
# ============================================================

DESIGNS = {

    # --------------------------------------------
    # 原来的设计：
    # STEP固定，所以Overlap Ratio随W改变
    # --------------------------------------------

    "same_step": {
        126: 20,
        252: 20,
        504: 20,
    },

    # --------------------------------------------
    # 新的控制实验：
    # STEP/W固定，因此Overlap Ratio保持约92.06%
    # --------------------------------------------

    "fixed_overlap": {
        126: 10,
        252: 20,
        504: 40,
    }
}


# ============================================================
# 5. 股票代码标准化
# ============================================================

def normalize_code(x):

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    match = re.search(
        r"(\d{6})",
        s
    )

    if match:
        return match.group(1)

    digits = "".join(
        c
        for c in s
        if c.isdigit()
    )

    if digits:
        return digits.zfill(6)

    return s


# ============================================================
# 6. 读取股票Metadata
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
        "stock_info.csv中无法识别"
        "股票代码、股票名称或行业字段。"
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
            code_col: "stock_code",
            name_col: "stock_name",
            industry_col: "industry"
        }
    )
    .copy()
)


metadata[
    "stock_code"
] = metadata[
    "stock_code"
].apply(
    normalize_code
)


metadata = (
    metadata
    .drop_duplicates(
        subset="stock_code"
    )
    .reset_index(
        drop=True
    )
)


name_map = dict(
    zip(
        metadata[
            "stock_code"
        ],
        metadata[
            "stock_name"
        ]
    )
)


industry_map = dict(
    zip(
        metadata[
            "stock_code"
        ],
        metadata[
            "industry"
        ]
    )
)


# ============================================================
# 7. 读取收益率数据
# ============================================================

returns_raw = pd.read_csv(
    RETURNS_FILE
)


date_candidates = [
    "日期",
    "date",
    "trade_date",
    "datetime",
    "Date"
]


date_col = next(
    (
        c
        for c in date_candidates
        if c in returns_raw.columns
    ),
    None
)


if date_col is None:

    raise ValueError(
        "stock_returns.csv中无法找到日期列。"
    )


returns_raw[
    date_col
] = pd.to_datetime(
    returns_raw[
        date_col
    ]
)


returns_raw = (
    returns_raw
    .sort_values(
        date_col
    )
    .set_index(
        date_col
    )
)


# ============================================================
# 8. 标准化收益率列名
# ============================================================

rename_dict = {}

for col in returns_raw.columns:

    normalized = normalize_code(
        col
    )

    if normalized in name_map:

        rename_dict[
            col
        ] = normalized


returns_df = (
    returns_raw
    .rename(
        columns=rename_dict
    )
)


stock_codes = [
    code
    for code in metadata[
        "stock_code"
    ]
    if code in returns_df.columns
]


if len(
    stock_codes
) < 2:

    raise ValueError(
        "收益率文件与stock_info无法匹配股票代码。"
    )


returns_df = (
    returns_df[
        stock_codes
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .dropna()
)


print(
    "股票数量：",
    len(
        stock_codes
    )
)

print(
    "有效收益率观测数：",
    len(
        returns_df
    )
)


# ============================================================
# 9. 基本检查
# ============================================================

MAX_WINDOW = max(
    max(
        design.keys()
    )
    for design in DESIGNS.values()
)


if len(
    returns_df
) < MAX_WINDOW:

    raise ValueError(
        f"有效观测数不足{MAX_WINDOW}，"
        "无法运行W=504。"
    )


N_STOCKS = len(
    stock_codes
)

N_PAIRS = (
    N_STOCKS
    *
    (
        N_STOCKS - 1
    )
    //
    2
)


# ============================================================
# 10. 为了公平比较，所有实验从同一个Anchor Date开始
#
# Anchor = 第504个有效收益率观测
# ============================================================

ANCHOR_INDEX = (
    MAX_WINDOW
    -
    1
)

ANCHOR_DATE = (
    returns_df
    .index[
        ANCHOR_INDEX
    ]
)


print(
    "统一比较起点：",
    ANCHOR_DATE
)


# ============================================================
# 11. 根据Precision计算Partial Correlation
# ============================================================

def precision_to_partial(
    precision
):

    diagonal = np.diag(
        precision
    )


    if np.any(
        diagonal
        <=
        0
    ):

        raise ValueError(
            "Precision Matrix存在非正对角元素。"
        )


    denominator = np.sqrt(
        np.outer(
            diagonal,
            diagonal
        )
    )


    partial = (
        -
        precision
        /
        denominator
    )


    np.fill_diagonal(
        partial,
        1.0
    )


    return partial


# ============================================================
# 12. 单个Rolling GLasso实验
# ============================================================

def run_rolling_glasso(
    design_name,
    window_size,
    step
):

    network_rows = []

    edge_rows = []

    diagnostic_rows = []


    overlap_ratio = (
        1
        -
        step
        /
        window_size
    )


    # --------------------------------------------------------
    # 所有Window Size均从统一Anchor endpoint开始
    # --------------------------------------------------------

    endpoint_indices = list(
        range(
            ANCHOR_INDEX,
            len(
                returns_df
            ),
            step
        )
    )


    for window_id, end_idx in enumerate(
        endpoint_indices,
        start=1
    ):

        start_idx = (
            end_idx
            -
            window_size
            +
            1
        )


        if start_idx < 0:
            continue


        window = (
            returns_df
            .iloc[
                start_idx:
                end_idx + 1
            ]
            .copy()
        )


        # ----------------------------------------------------
        # Window内部标准化
        # ----------------------------------------------------

        means = (
            window
            .mean(
                axis=0
            )
        )


        sds = (
            window
            .std(
                axis=0,
                ddof=0
            )
        )


        if (
            sds
            <=
            1e-12
        ).any():

            raise ValueError(
                f"{design_name}, W={window_size}: "
                "某只股票窗口内标准差接近0。"
            )


        X = (
            (
                window
                -
                means
            )
            /
            sds
        ).to_numpy(
            dtype=float
        )


        # ----------------------------------------------------
        # Graphical Lasso
        # ----------------------------------------------------

        model = GraphicalLasso(
            alpha=ALPHA,
            max_iter=MAX_ITER,
            tol=TOL,
            assume_centered=True
        )


        with warnings.catch_warnings(
            record=True
        ) as caught:

            warnings.simplefilter(
                "always"
            )

            model.fit(
                X
            )


        convergence_warning = any(
            issubclass(
                w.category,
                ConvergenceWarning
            )
            for w in caught
        )


        precision = (
            model.precision_
        )


        partial = (
            precision_to_partial(
                precision
            )
        )


        # ----------------------------------------------------
        # 构造边
        # ----------------------------------------------------

        edge_count = 0

        same_edges = 0

        cross_edges = 0

        abs_partial_values = []


        for i in range(
            N_STOCKS
        ):

            for j in range(
                i + 1,
                N_STOCKS
            ):

                stock_1 = (
                    stock_codes[
                        i
                    ]
                )

                stock_2 = (
                    stock_codes[
                        j
                    ]
                )


                selected = (
                    abs(
                        precision[
                            i,
                            j
                        ]
                    )
                    >
                    EDGE_TOL
                )


                partial_ij = (
                    partial[
                        i,
                        j
                    ]
                )


                same_industry = (
                    industry_map[
                        stock_1
                    ]
                    ==
                    industry_map[
                        stock_2
                    ]
                )


                if selected:

                    edge_count += 1

                    abs_partial_values.append(
                        abs(
                            partial_ij
                        )
                    )


                    if same_industry:

                        same_edges += 1

                    else:

                        cross_edges += 1


                edge_rows.append(
                    {
                        "design":
                            design_name,

                        "window_size":
                            window_size,

                        "step":
                            step,

                        "overlap_ratio":
                            overlap_ratio,

                        "window_id":
                            window_id,

                        "window_start":
                            window.index[
                                0
                            ],

                        "network_date":
                            window.index[
                                -1
                            ],

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
                            same_industry,

                        "selected":
                            selected,

                        "partial_correlation":
                            partial_ij
                    }
                )


        mean_abs_partial = (
            np.mean(
                abs_partial_values
            )
            if len(
                abs_partial_values
            ) > 0
            else np.nan
        )


        network_rows.append(
            {
                "design":
                    design_name,

                "window_size":
                    window_size,

                "step":
                    step,

                "replacement_ratio":
                    step
                    /
                    window_size,

                "overlap_ratio":
                    overlap_ratio,

                "window_id":
                    window_id,

                "window_start":
                    window.index[
                        0
                    ],

                "network_date":
                    window.index[
                        -1
                    ],

                "edge_count":
                    edge_count,

                "density":
                    edge_count
                    /
                    N_PAIRS,

                "same_edges":
                    same_edges,

                "cross_edges":
                    cross_edges,

                "same_industry_ratio":
                    (
                        same_edges
                        /
                        edge_count
                        if edge_count > 0
                        else np.nan
                    ),

                "mean_abs_partial":
                    mean_abs_partial
            }
        )


        diagnostic_rows.append(
            {
                "design":
                    design_name,

                "window_size":
                    window_size,

                "step":
                    step,

                "window_id":
                    window_id,

                "network_date":
                    window.index[
                        -1
                    ],

                "n_iter":
                    getattr(
                        model,
                        "n_iter_",
                        np.nan
                    ),

                "convergence_warning":
                    convergence_warning
            }
        )


    return (
        pd.DataFrame(
            network_rows
        ),
        pd.DataFrame(
            edge_rows
        ),
        pd.DataFrame(
            diagnostic_rows
        )
    )


# ============================================================
# 13. 运行全部实验
# ============================================================

all_network = []

all_edges = []

all_diagnostics = []


for design_name, settings in DESIGNS.items():

    for window_size, step in settings.items():

        print(
            "\n运行：",
            design_name,
            "W =",
            window_size,
            "STEP =",
            step,
            "Overlap =",
            round(
                1
                -
                step
                /
                window_size,
                4
            )
        )


        network_df, edge_df, diag_df = (
            run_rolling_glasso(
                design_name,
                window_size,
                step
            )
        )


        all_network.append(
            network_df
        )

        all_edges.append(
            edge_df
        )

        all_diagnostics.append(
            diag_df
        )


network_df = pd.concat(
    all_network,
    ignore_index=True
)


edge_history_df = pd.concat(
    all_edges,
    ignore_index=True
)


diagnostics_df = pd.concat(
    all_diagnostics,
    ignore_index=True
)


# ============================================================
# 14. 计算相邻网络Turnover
# ============================================================

turnover_rows = []


for (
    design,
    window_size
), group in edge_history_df.groupby(
    [
        "design",
        "window_size"
    ]
):

    group = (
        group
        .sort_values(
            [
                "network_date",
                "stock_1",
                "stock_2"
            ]
        )
    )


    dates = (
        group[
            "network_date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


    for t in range(
        1,
        len(
            dates
        )
    ):

        previous_date = (
            dates[
                t - 1
            ]
        )

        current_date = (
            dates[
                t
            ]
        )


        previous = (
            group[
                group[
                    "network_date"
                ]
                ==
                previous_date
            ]
            .set_index(
                [
                    "stock_1",
                    "stock_2"
                ]
            )
        )


        current = (
            group[
                group[
                    "network_date"
                ]
                ==
                current_date
            ]
            .set_index(
                [
                    "stock_1",
                    "stock_2"
                ]
            )
        )


        merged = (
            previous[
                [
                    "selected",
                    "same_industry"
                ]
            ]
            .rename(
                columns={
                    "selected":
                        "selected_before"
                }
            )
            .join(
                current[
                    [
                        "selected"
                    ]
                ]
                .rename(
                    columns={
                        "selected":
                            "selected_after"
                    }
                ),

                how="inner"
            )
        )


        before = (
            merged[
                "selected_before"
            ]
            .astype(bool)
        )


        after = (
            merged[
                "selected_after"
            ]
            .astype(bool)
        )


        common = int(
            (
                before
                &
                after
            )
            .sum()
        )


        lost = int(
            (
                before
                &
                ~after
            )
            .sum()
        )


        gained = int(
            (
                ~before
                &
                after
            )
            .sum()
        )


        union = (
            common
            +
            lost
            +
            gained
        )


        jaccard = (
            common
            /
            union
            if union > 0
            else 1.0
        )


        turnover = (
            1
            -
            jaccard
        )


        changed = (
            before
            !=
            after
        )


        same_mask = (
            merged[
                "same_industry"
            ]
            .astype(bool)
        )


        same_changes = int(
            (
                changed
                &
                same_mask
            )
            .sum()
        )


        cross_changes = int(
            (
                changed
                &
                ~same_mask
            )
            .sum()
        )


        gross_changes = (
            lost
            +
            gained
        )


        turnover_rows.append(
            {
                "design":
                    design,

                "window_size":
                    window_size,

                "step":
                    int(
                        group[
                            "step"
                        ]
                        .iloc[0]
                    ),

                "overlap_ratio":
                    float(
                        group[
                            "overlap_ratio"
                        ]
                        .iloc[0]
                    ),

                "previous_date":
                    previous_date,

                "network_date":
                    current_date,

                "common_edges":
                    common,

                "lost_edges":
                    lost,

                "gained_edges":
                    gained,

                "gross_edge_changes":
                    gross_changes,

                "net_edge_change":
                    gained
                    -
                    lost,

                "jaccard":
                    jaccard,

                "turnover":
                    turnover,

                "same_edge_changes":
                    same_changes,

                "cross_edge_changes":
                    cross_changes,

                "cross_change_share":
                    (
                        cross_changes
                        /
                        gross_changes
                        if gross_changes > 0
                        else np.nan
                    )
            }
        )


turnover_df = pd.DataFrame(
    turnover_rows
)


# ============================================================
# 15. Edge Persistence
# ============================================================

persistence_rows = []


for (
    design,
    window_size,
    stock_1,
    stock_2
), group in edge_history_df.groupby(
    [
        "design",
        "window_size",
        "stock_1",
        "stock_2"
    ]
):

    n_windows = len(
        group
    )


    windows_selected = int(
        group[
            "selected"
        ]
        .sum()
    )


    persistence = (
        windows_selected
        /
        n_windows
    )


    selected_group = (
        group[
            group[
                "selected"
            ]
        ]
    )


    persistence_rows.append(
        {
            "design":
                design,

            "window_size":
                window_size,

            "step":
                int(
                    group[
                        "step"
                    ]
                    .iloc[0]
                ),

            "overlap_ratio":
                float(
                    group[
                        "overlap_ratio"
                    ]
                    .iloc[0]
                ),

            "stock_1":
                stock_1,

            "name_1":
                group[
                    "name_1"
                ]
                .iloc[0],

            "industry_1":
                group[
                    "industry_1"
                ]
                .iloc[0],

            "stock_2":
                stock_2,

            "name_2":
                group[
                    "name_2"
                ]
                .iloc[0],

            "industry_2":
                group[
                    "industry_2"
                ]
                .iloc[0],

            "same_industry":
                bool(
                    group[
                        "same_industry"
                    ]
                    .iloc[0]
                ),

            "n_windows":
                n_windows,

            "windows_selected":
                windows_selected,

            "persistence":
                persistence,

            "persistent_08":
                persistence
                >=
                PERSISTENCE_THRESHOLD,

            "always_selected":
                np.isclose(
                    persistence,
                    1.0
                ),

            "mean_abs_partial_when_selected":
                (
                    selected_group[
                        "partial_correlation"
                    ]
                    .abs()
                    .mean()
                    if len(
                        selected_group
                    ) > 0
                    else np.nan
                )
        }
    )


persistence_df = pd.DataFrame(
    persistence_rows
)


# ============================================================
# 16. Same / Cross Persistence Summary
# ============================================================

industry_rows = []


for (
    design,
    window_size,
    same_industry
), group in persistence_df.groupby(
    [
        "design",
        "window_size",
        "same_industry"
    ]
):

    relation = (
        "Same industry"
        if same_industry
        else
        "Cross industry"
    )


    industry_rows.append(
        {
            "design":
                design,

            "window_size":
                window_size,

            "step":
                int(
                    group[
                        "step"
                    ]
                    .iloc[0]
                ),

            "overlap_ratio":
                float(
                    group[
                        "overlap_ratio"
                    ]
                    .iloc[0]
                ),

            "industry_relation":
                relation,

            "n_candidate_pairs":
                len(
                    group
                ),

            "mean_persistence":
                group[
                    "persistence"
                ]
                .mean(),

            "median_persistence":
                group[
                    "persistence"
                ]
                .median(),

            "persistent_rate_08":
                group[
                    "persistent_08"
                ]
                .mean(),

            "always_rate":
                group[
                    "always_selected"
                ]
                .mean(),

            "ever_selected_rate":
                (
                    group[
                        "windows_selected"
                    ]
                    >
                    0
                )
                .mean(),

            "mean_abs_partial_when_selected":
                group[
                    "mean_abs_partial_when_selected"
                ]
                .mean()
        }
    )


industry_summary_df = pd.DataFrame(
    industry_rows
)


# ============================================================
# 17. Design × Window汇总
# ============================================================

summary_rows = []


for (
    design,
    window_size
), network_group in network_df.groupby(
    [
        "design",
        "window_size"
    ]
):

    transition_group = (
        turnover_df[
            (
                turnover_df[
                    "design"
                ]
                ==
                design
            )
            &
            (
                turnover_df[
                    "window_size"
                ]
                ==
                window_size
            )
        ]
    )


    persistence_group = (
        persistence_df[
            (
                persistence_df[
                    "design"
                ]
                ==
                design
            )
            &
            (
                persistence_df[
                    "window_size"
                ]
                ==
                window_size
            )
        ]
    )


    same_pairs = (
        persistence_group[
            persistence_group[
                "same_industry"
            ]
        ]
    )


    cross_pairs = (
        persistence_group[
            ~persistence_group[
                "same_industry"
            ]
        ]
    )


    n_transitions = len(
        transition_group
    )


    total_same_changes = (
        transition_group[
            "same_edge_changes"
        ]
        .sum()
    )


    total_cross_changes = (
        transition_group[
            "cross_edge_changes"
        ]
        .sum()
    )


    same_state_change_rate = (
        total_same_changes
        /
        (
            len(
                same_pairs
            )
            *
            n_transitions
        )
        if (
            len(
                same_pairs
            ) > 0
            and
            n_transitions > 0
        )
        else np.nan
    )


    cross_state_change_rate = (
        total_cross_changes
        /
        (
            len(
                cross_pairs
            )
            *
            n_transitions
        )
        if (
            len(
                cross_pairs
            ) > 0
            and
            n_transitions > 0
        )
        else np.nan
    )


    summary_rows.append(
        {
            "design":
                design,

            "window_size":
                window_size,

            "step":
                int(
                    network_group[
                        "step"
                    ]
                    .iloc[0]
                ),

            "replacement_ratio":
                float(
                    network_group[
                        "replacement_ratio"
                    ]
                    .iloc[0]
                ),

            "overlap_ratio":
                float(
                    network_group[
                        "overlap_ratio"
                    ]
                    .iloc[0]
                ),

            "n_networks":
                len(
                    network_group
                ),

            "n_transitions":
                n_transitions,

            "mean_edge_count":
                network_group[
                    "edge_count"
                ]
                .mean(),

            "sd_edge_count":
                network_group[
                    "edge_count"
                ]
                .std(
                    ddof=1
                ),

            "mean_density":
                network_group[
                    "density"
                ]
                .mean(),

            "mean_abs_partial":
                network_group[
                    "mean_abs_partial"
                ]
                .mean(),

            "mean_same_edges":
                network_group[
                    "same_edges"
                ]
                .mean(),

            "mean_cross_edges":
                network_group[
                    "cross_edges"
                ]
                .mean(),

            "mean_same_industry_ratio":
                network_group[
                    "same_industry_ratio"
                ]
                .mean(),

            "mean_turnover":
                transition_group[
                    "turnover"
                ]
                .mean(),

            "mean_gross_edge_changes":
                transition_group[
                    "gross_edge_changes"
                ]
                .mean(),

            "mean_cross_change_share":
                transition_group[
                    "cross_change_share"
                ]
                .mean(),

            "same_state_change_rate":
                same_state_change_rate,

            "cross_state_change_rate":
                cross_state_change_rate,

            "same_mean_persistence":
                same_pairs[
                    "persistence"
                ]
                .mean(),

            "cross_mean_persistence":
                cross_pairs[
                    "persistence"
                ]
                .mean(),

            "same_persistent_rate_08":
                same_pairs[
                    "persistent_08"
                ]
                .mean(),

            "cross_persistent_rate_08":
                cross_pairs[
                    "persistent_08"
                ]
                .mean(),

            "n_persistent_edges_08":
                int(
                    persistence_group[
                        "persistent_08"
                    ]
                    .sum()
                ),

            "n_always_edges":
                int(
                    persistence_group[
                        "always_selected"
                    ]
                    .sum()
                )
        }
    )


window_summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 18. Same-step vs Fixed-overlap直接比较
# ============================================================

same_step_summary = (
    window_summary_df[
        window_summary_df[
            "design"
        ]
        ==
        "same_step"
    ]
    .drop(
        columns=[
            "design"
        ]
    )
    .copy()
)


fixed_overlap_summary = (
    window_summary_df[
        window_summary_df[
            "design"
        ]
        ==
        "fixed_overlap"
    ]
    .drop(
        columns=[
            "design"
        ]
    )
    .copy()
)


comparison_df = (
    same_step_summary
    .merge(
        fixed_overlap_summary,
        on="window_size",
        suffixes=(
            "_same_step",
            "_fixed_overlap"
        )
    )
)


# ------------------------------------------------------------
# 重点比较Turnover变化
# ------------------------------------------------------------

comparison_df[
    "delta_mean_turnover_fixed_minus_same"
] = (
    comparison_df[
        "mean_turnover_fixed_overlap"
    ]
    -
    comparison_df[
        "mean_turnover_same_step"
    ]
)


comparison_df[
    "turnover_ratio_fixed_to_same"
] = (
    comparison_df[
        "mean_turnover_fixed_overlap"
    ]
    /
    comparison_df[
        "mean_turnover_same_step"
    ]
)


comparison_df[
    "delta_mean_gross_changes_fixed_minus_same"
] = (
    comparison_df[
        "mean_gross_edge_changes_fixed_overlap"
    ]
    -
    comparison_df[
        "mean_gross_edge_changes_same_step"
    ]
)


# ============================================================
# 19. 保存结果
# ============================================================

network_df.to_csv(
    NETWORK_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


edge_history_df.to_csv(
    EDGE_HISTORY_FILE,
    index=False,
    encoding="utf-8-sig"
)


turnover_df.to_csv(
    TURNOVER_FILE,
    index=False,
    encoding="utf-8-sig"
)


persistence_df.to_csv(
    EDGE_PERSISTENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


industry_summary_df.to_csv(
    INDUSTRY_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


window_summary_df.to_csv(
    WINDOW_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


comparison_df.to_csv(
    DESIGN_COMPARISON_FILE,
    index=False,
    encoding="utf-8-sig"
)


diagnostics_df.to_csv(
    DIAGNOSTICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. 输出核心汇总
# ============================================================

print(
    "\n======================================"
)

print(
    "Window / Overlap Robustness Summary"
)

print(
    "======================================"
)


print(
    window_summary_df[
        [
            "design",
            "window_size",
            "step",
            "overlap_ratio",
            "n_networks",
            "mean_edge_count",
            "sd_edge_count",
            "mean_turnover",
            "mean_gross_edge_changes",
            "same_mean_persistence",
            "cross_mean_persistence",
            "same_state_change_rate",
            "cross_state_change_rate",
            "n_persistent_edges_08"
        ]
    ]
    .sort_values(
        [
            "design",
            "window_size"
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 21. 图1：Turnover
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


for design, group in window_summary_df.groupby(
    "design"
):

    group = (
        group
        .sort_values(
            "window_size"
        )
    )


    ax.plot(
        group[
            "window_size"
        ],
        group[
            "mean_turnover"
        ],
        marker="o",
        label=design
    )


ax.set_xlabel(
    "Window Size"
)

ax.set_ylabel(
    "Mean Turnover"
)

ax.set_title(
    "Window Size and Overlap Control: Mean Turnover"
)

ax.set_xticks(
    [
        126,
        252,
        504
    ]
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "overlap_control_turnover.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 22. 图2：Gross Edge Changes
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


for design, group in window_summary_df.groupby(
    "design"
):

    group = (
        group
        .sort_values(
            "window_size"
        )
    )


    ax.plot(
        group[
            "window_size"
        ],
        group[
            "mean_gross_edge_changes"
        ],
        marker="o",
        label=design
    )


ax.set_xlabel(
    "Window Size"
)

ax.set_ylabel(
    "Mean Gross Edge Changes"
)

ax.set_title(
    "Window Size and Overlap Control: Gross Changes"
)

ax.set_xticks(
    [
        126,
        252,
        504
    ]
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "overlap_control_gross_changes.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. 图3：Fixed-overlap下Same/Cross Persistence
# ============================================================

fixed_industry = (
    industry_summary_df[
        industry_summary_df[
            "design"
        ]
        ==
        "fixed_overlap"
    ]
    .copy()
)


fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


for relation, group in fixed_industry.groupby(
    "industry_relation"
):

    group = (
        group
        .sort_values(
            "window_size"
        )
    )


    ax.plot(
        group[
            "window_size"
        ],
        group[
            "mean_persistence"
        ],
        marker="o",
        label=relation
    )


ax.set_xlabel(
    "Window Size"
)

ax.set_ylabel(
    "Mean Edge Persistence"
)

ax.set_title(
    "Fixed-overlap: Same vs Cross-industry Persistence"
)

ax.set_xticks(
    [
        126,
        252,
        504
    ]
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


fig.savefig(
    FIGURE_DIR
    / "fixed_overlap_industry_persistence.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. 图4：Fixed-overlap下Same/Cross State-change Rate
# ============================================================

fixed_summary = (
    window_summary_df[
        window_summary_df[
            "design"
        ]
        ==
        "fixed_overlap"
    ]
    .sort_values(
        "window_size"
    )
)


fig, ax = plt.subplots(
    figsize=(
        9,
        6
    )
)


ax.plot(
    fixed_summary[
        "window_size"
    ],
    fixed_summary[
        "same_state_change_rate"
    ],
    marker="o",
    label="Same industry"
)


ax.plot(
    fixed_summary[
        "window_size"
    ],
    fixed_summary[
        "cross_state_change_rate"
    ],
    marker="o",
    label="Cross industry"
)


ax.set_xlabel(
    "Window Size"
)

ax.set_ylabel(
    "Normalized State-change Rate"
)

ax.set_title(
    "Fixed-overlap: Same vs Cross-industry Dynamic Changes"
)

ax.set_xticks(
    [
        126,
        252,
        504
    ]
)

ax.legend()

ax.grid(
    alpha=0.3
)

fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "fixed_overlap_state_change_rate.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. Convergence检查
# ============================================================

warning_count = int(
    diagnostics_df[
        "convergence_warning"
    ]
    .sum()
)


print(
    "\nGraphical Lasso convergence warnings：",
    warning_count
)


# ============================================================
# 26. 完成
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


for path in [
    NETWORK_SUMMARY_FILE,
    EDGE_HISTORY_FILE,
    TURNOVER_FILE,
    EDGE_PERSISTENCE_FILE,
    INDUSTRY_SUMMARY_FILE,
    WINDOW_SUMMARY_FILE,
    DESIGN_COMPARISON_FILE,
    DIAGNOSTICS_FILE
]:

    print(
        path
    )

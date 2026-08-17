from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning
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

            plt.rcParams["font.sans-serif"] = [
                font_name
            ]

            plt.rcParams["axes.unicode_minus"] = False

            print(
                f"使用中文字体：{font_name}"
            )

            return

    print(
        "警告：没有找到常见中文字体。"
    )


set_chinese_font()


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


# ------------------------------------------------------------
# 输入
# ------------------------------------------------------------

RETURN_FILE = (
    PROCESSED_DIR
    / "stock_returns.csv"
)

SELECTION_FILE = (
    PROCESSED_DIR
    / "glasso_1se_selection.csv"
)

BASELINE_EDGE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_edges.csv"
)

# 阶段五的9条跨方法共同强边
STRONG_COMMON_FILE = (
    PROCESSED_DIR
    / "partial_vs_glasso_common_edges.csv"
)


# ------------------------------------------------------------
# 输出
# ------------------------------------------------------------

EDGE_STABILITY_FILE = (
    PROCESSED_DIR
    / "block_length_edge_stability.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "block_length_robustness_summary.csv"
)

JACCARD_FILE = (
    PROCESSED_DIR
    / "block_length_pairwise_jaccard.csv"
)

ROBUST_EDGE_FILE = (
    PROCESSED_DIR
    / "block_length_robust_edges.csv"
)

STRONG_EDGE_FILE = (
    PROCESSED_DIR
    / "strong_common_edge_block_robustness.csv"
)

DIAGNOSTICS_FILE = (
    PROCESSED_DIR
    / "block_length_run_diagnostics.csv"
)


# ============================================================
# 2. 参数
# ============================================================

BLOCK_LENGTHS = [
    10,
    20,
    40
]

N_RESAMPLES = 200

SUBSAMPLE_RATIO = 0.80

STABLE_THRESHOLD = 0.80

CORE_THRESHOLD = 0.90

ZERO_TOL = 1e-8

MAX_ITER = 5000

OUTER_TOL = 1e-4

ENET_TOL = 1e-6

BASE_RANDOM_SEED = 20260817


# ============================================================
# 3. 工具函数
# ============================================================

def normalize_code(x):

    return str(
        x
    ).strip().zfill(6)


def edge_key(a, b):

    return tuple(
        sorted(
            [
                normalize_code(a),
                normalize_code(b)
            ]
        )
    )


# ============================================================
# 4. 读取收益率
# ============================================================

returns = pd.read_csv(
    RETURN_FILE,
    index_col=0,
    parse_dates=True
)


returns.columns = [
    normalize_code(x)
    for x in returns.columns
]


returns = (
    returns
    .sort_index()
    .dropna(
        axis=0,
        how="any"
    )
)


codes = (
    returns.columns
    .tolist()
)


X = (
    returns.values
    .astype(float)
)


n_samples = X.shape[0]

n_stocks = X.shape[1]


print(
    "收益率数据维度：",
    X.shape
)


# ============================================================
# 5. 读取固定的1-SE alpha
# ============================================================

selection_df = pd.read_csv(
    SELECTION_FILE
)


ALPHA_1SE = float(
    selection_df[
        "alpha_1se"
    ].iloc[0]
)


print(
    "固定 alpha_1SE =",
    ALPHA_1SE
)


# ============================================================
# 6. 读取全样本1-SE网络
# ============================================================

baseline_df = pd.read_csv(
    BASELINE_EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


baseline_df["stock_1"] = (
    baseline_df["stock_1"]
    .apply(normalize_code)
)

baseline_df["stock_2"] = (
    baseline_df["stock_2"]
    .apply(normalize_code)
)


E_BASELINE = {
    edge_key(
        row.stock_1,
        row.stock_2
    )
    for row
    in baseline_df.itertuples()
}


print(
    "全样本1-SE边数：",
    len(E_BASELINE)
)


# ============================================================
# 7. Moving-block bootstrap
# ============================================================

def moving_block_resample_indices(
    n,
    target_size,
    block_length,
    rng
):

    if block_length > n:

        raise ValueError(
            "block_length不能大于样本量。"
        )


    possible_starts = np.arange(
        0,
        n - block_length + 1
    )


    indices = []


    while len(indices) < target_size:

        start = int(
            rng.choice(
                possible_starts
            )
        )


        block = range(
            start,
            start + block_length
        )


        indices.extend(
            block
        )


    return np.asarray(
        indices[:target_size],
        dtype=int
    )


# ============================================================
# 8. 单个 block length 的稳定性分析
# ============================================================

def run_stability_for_block_length(
    block_length
):

    print(
        "\n======================================"
    )

    print(
        f"开始 block length = {block_length}"
    )

    print(
        "======================================"
    )


    # 每个L使用独立且可重复的随机种子
    rng = np.random.default_rng(
        BASE_RANDOM_SEED
        +
        block_length
    )


    target_size = int(
        np.floor(
            SUBSAMPLE_RATIO
            *
            n_samples
        )
    )


    selected_count = np.zeros(
        (
            n_stocks,
            n_stocks
        ),
        dtype=int
    )


    positive_count = np.zeros_like(
        selected_count
    )


    negative_count = np.zeros_like(
        selected_count
    )


    partial_sum = np.zeros(
        (
            n_stocks,
            n_stocks
        ),
        dtype=float
    )


    diagnostic_rows = []

    valid_runs = 0


    for b in range(
        N_RESAMPLES
    ):

        # ----------------------------------------------------
        # 8.1 block resampling
        # ----------------------------------------------------

        sample_idx = (
            moving_block_resample_indices(
                n=
                    n_samples,

                target_size=
                    target_size,

                block_length=
                    block_length,

                rng=
                    rng
            )
        )


        X_b = X[
            sample_idx,
            :
        ]


        # ----------------------------------------------------
        # 8.2 每次重采样内部标准化
        # ----------------------------------------------------

        mean_b = (
            X_b.mean(
                axis=0
            )
        )


        std_b = (
            X_b.std(
                axis=0,
                ddof=0
            )
        )


        if np.any(
            std_b
            <
            1e-12
        ):

            diagnostic_rows.append(
                {
                    "block_length":
                        block_length,

                    "resample":
                        b + 1,

                    "valid":
                        False,

                    "converged":
                        False,

                    "n_iter":
                        np.nan,

                    "dual_gap":
                        np.nan,

                    "n_edges":
                        np.nan,

                    "reason":
                        "near_zero_std"
                }
            )

            continue


        Z_b = (
            X_b
            -
            mean_b
        ) / std_b


        # ----------------------------------------------------
        # 8.3 Graphical Lasso
        # ----------------------------------------------------

        model = GraphicalLasso(

            alpha=
                ALPHA_1SE,

            mode=
                "cd",

            tol=
                OUTER_TOL,

            enet_tol=
                ENET_TOL,

            max_iter=
                MAX_ITER,

            assume_centered=
                True
        )


        with warnings.catch_warnings(
            record=True
        ) as caught:

            warnings.simplefilter(
                "always"
            )


            try:

                model.fit(
                    Z_b
                )

                fit_error = None

            except Exception as exc:

                fit_error = str(
                    exc
                )


        # ----------------------------------------------------
        # 8.4 拟合失败
        # ----------------------------------------------------

        if fit_error is not None:

            diagnostic_rows.append(
                {
                    "block_length":
                        block_length,

                    "resample":
                        b + 1,

                    "valid":
                        False,

                    "converged":
                        False,

                    "n_iter":
                        np.nan,

                    "dual_gap":
                        np.nan,

                    "n_edges":
                        np.nan,

                    "reason":
                        fit_error
                }
            )

            continue


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


        finite_solution = np.all(
            np.isfinite(
                precision
            )
        )


        # ----------------------------------------------------
        # 8.5 dual gap
        # ----------------------------------------------------

        if (
            hasattr(
                model,
                "costs_"
            )
            and
            len(model.costs_) > 0
        ):

            final_gap = float(
                model.costs_[-1][1]
            )

        else:

            final_gap = np.nan


        converged = (
            finite_solution
            and
            not convergence_warning
        )


        # ----------------------------------------------------
        # 8.6 当前网络边数
        # ----------------------------------------------------

        n_edges_current = 0


        for i in range(
            n_stocks
        ):

            for j in range(
                i + 1,
                n_stocks
            ):

                if (
                    abs(
                        precision[i, j]
                    )
                    >
                    ZERO_TOL
                ):

                    n_edges_current += 1


        diagnostic_rows.append(
            {
                "block_length":
                    block_length,

                "resample":
                    b + 1,

                "valid":
                    bool(
                        finite_solution
                    ),

                "converged":
                    bool(
                        converged
                    ),

                "n_iter":
                    model.n_iter_,

                "dual_gap":
                    final_gap,

                "n_edges":
                    n_edges_current,

                "reason":
                    (
                        ""
                        if converged
                        else
                        "ConvergenceWarning"
                    )
            }
        )


        # 不收敛结果不计入selection frequency
        if not converged:

            continue


        # ----------------------------------------------------
        # 8.7 Precision -> partial correlation
        # ----------------------------------------------------

        diag = np.sqrt(
            np.diag(
                precision
            )
        )


        partial = (
            -precision
            /
            np.outer(
                diag,
                diag
            )
        )


        np.fill_diagonal(
            partial,
            1.0
        )


        valid_runs += 1


        # ----------------------------------------------------
        # 8.8 更新边选择统计
        # ----------------------------------------------------

        for i in range(
            n_stocks
        ):

            for j in range(
                i + 1,
                n_stocks
            ):

                if (
                    abs(
                        precision[i, j]
                    )
                    >
                    ZERO_TOL
                ):

                    selected_count[
                        i,
                        j
                    ] += 1


                    rho = float(
                        partial[
                            i,
                            j
                        ]
                    )


                    partial_sum[
                        i,
                        j
                    ] += rho


                    if rho > 0:

                        positive_count[
                            i,
                            j
                        ] += 1

                    elif rho < 0:

                        negative_count[
                            i,
                            j
                        ] += 1


    # ========================================================
    # 9. 生成该L下的105条边结果
    # ========================================================

    if valid_runs == 0:

        raise RuntimeError(
            f"L={block_length}没有有效重复。"
        )


    edge_rows = []


    for i in range(
        n_stocks
    ):

        for j in range(
            i + 1,
            n_stocks
        ):

            count = int(
                selected_count[
                    i,
                    j
                ]
            )


            frequency = (
                count
                /
                valid_runs
            )


            if count > 0:

                mean_partial = (
                    partial_sum[
                        i,
                        j
                    ]
                    /
                    count
                )


                pos_fraction = (
                    positive_count[
                        i,
                        j
                    ]
                    /
                    count
                )


                neg_fraction = (
                    negative_count[
                        i,
                        j
                    ]
                    /
                    count
                )


                sign_consistency = max(
                    pos_fraction,
                    neg_fraction
                )

            else:

                mean_partial = np.nan

                sign_consistency = np.nan


            code_i = codes[i]

            code_j = codes[j]


            edge = edge_key(
                code_i,
                code_j
            )


            edge_rows.append(
                {
                    "block_length":
                        block_length,

                    "stock_1":
                        code_i,

                    "stock_2":
                        code_j,

                    "selected_count":
                        count,

                    "valid_runs":
                        valid_runs,

                    "selection_frequency":
                        frequency,

                    "mean_partial_when_selected":
                        mean_partial,

                    "sign_consistency":
                        sign_consistency,

                    "baseline_1se_selected":
                        edge
                        in
                        E_BASELINE
                }
            )


    edge_df = pd.DataFrame(
        edge_rows
    )


    # --------------------------------------------------------
    # Stable / core sets
    # --------------------------------------------------------

    stable_df = edge_df[
        edge_df[
            "selection_frequency"
        ]
        >=
        STABLE_THRESHOLD
    ]


    core_df = edge_df[
        edge_df[
            "selection_frequency"
        ]
        >=
        CORE_THRESHOLD
    ]


    E_stable = {

        edge_key(
            row.stock_1,
            row.stock_2
        )

        for row
        in stable_df.itertuples()
    }


    E_core = {

        edge_key(
            row.stock_1,
            row.stock_2
        )

        for row
        in core_df.itertuples()
    }


    # --------------------------------------------------------
    # baseline 1-SE retention
    # --------------------------------------------------------

    common_with_baseline = (
        E_BASELINE
        &
        E_stable
    )


    baseline_retention = (
        len(
            common_with_baseline
        )
        /
        len(
            E_BASELINE
        )
    )


    diagnostics_df = pd.DataFrame(
        diagnostic_rows
    )


    summary = {

        "block_length":
            block_length,

        "n_resamples_requested":
            N_RESAMPLES,

        "n_valid_resamples":
            valid_runs,

        "valid_rate":
            valid_runs
            /
            N_RESAMPLES,

        "mean_edges_per_resample":
            diagnostics_df.loc[
                diagnostics_df[
                    "converged"
                ],
                "n_edges"
            ].mean(),

        "sd_edges_per_resample":
            diagnostics_df.loc[
                diagnostics_df[
                    "converged"
                ],
                "n_edges"
            ].std(),

        "stable_edges":
            len(
                E_stable
            ),

        "core_stable_edges":
            len(
                E_core
            ),

        "baseline_1se_edges":
            len(
                E_BASELINE
            ),

        "baseline_edges_retained":
            len(
                common_with_baseline
            ),

        "baseline_edge_retention":
            baseline_retention
    }


    return (
        edge_df,
        diagnostics_df,
        summary,
        E_stable,
        E_core
    )


# ============================================================
# 10. 对 L = 10,20,40 分别运行
# ============================================================

all_edge_results = []

all_diagnostics = []

summary_rows = []

stable_sets = {}

core_sets = {}


for L in BLOCK_LENGTHS:

    (
        edge_df_L,
        diagnostics_df_L,
        summary_L,
        E_stable_L,
        E_core_L

    ) = run_stability_for_block_length(
        L
    )


    all_edge_results.append(
        edge_df_L
    )


    all_diagnostics.append(
        diagnostics_df_L
    )


    summary_rows.append(
        summary_L
    )


    stable_sets[L] = (
        E_stable_L
    )


    core_sets[L] = (
        E_core_L
    )


# ============================================================
# 11. 保存所有边稳定性
# ============================================================

all_edge_df = pd.concat(
    all_edge_results,
    ignore_index=True
)


all_edge_df.to_csv(
    EDGE_STABILITY_FILE,
    index=False,
    encoding="utf-8-sig"
)


all_diagnostics_df = pd.concat(
    all_diagnostics,
    ignore_index=True
)


all_diagnostics_df.to_csv(
    DIAGNOSTICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n======================================"
)

print(
    "不同Block Length摘要"
)

print(
    "======================================"
)


print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 12. Stable network pairwise Jaccard
# ============================================================

jaccard_rows = []


for i in range(
    len(BLOCK_LENGTHS)
):

    for j in range(
        i + 1,
        len(BLOCK_LENGTHS)
    ):

        L1 = BLOCK_LENGTHS[i]

        L2 = BLOCK_LENGTHS[j]


        E1 = stable_sets[L1]

        E2 = stable_sets[L2]


        intersection = (
            E1
            &
            E2
        )


        union = (
            E1
            |
            E2
        )


        jaccard = (
            len(
                intersection
            )
            /
            len(
                union
            )

            if len(
                union
            )
            >
            0

            else np.nan
        )


        jaccard_rows.append(
            {
                "block_length_1":
                    L1,

                "block_length_2":
                    L2,

                "edges_1":
                    len(
                        E1
                    ),

                "edges_2":
                    len(
                        E2
                    ),

                "common_edges":
                    len(
                        intersection
                    ),

                "union_edges":
                    len(
                        union
                    ),

                "jaccard":
                    jaccard
            }
        )


jaccard_df = pd.DataFrame(
    jaccard_rows
)


jaccard_df.to_csv(
    JACCARD_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n======================================"
)

print(
    "Stable Networks的Pairwise Jaccard"
)

print(
    "======================================"
)


print(
    jaccard_df.to_string(
        index=False
    )
)


# ============================================================
# 13. 将三个L的selection frequency放到同一张表
# ============================================================

wide_list = []


for L in BLOCK_LENGTHS:

    temp = all_edge_df[
        all_edge_df[
            "block_length"
        ]
        ==
        L
    ][
        [
            "stock_1",
            "stock_2",
            "selection_frequency",
            "mean_partial_when_selected",
            "sign_consistency"
        ]
    ].copy()


    temp = temp.rename(
        columns={
            "selection_frequency":
                f"frequency_L{L}",

            "mean_partial_when_selected":
                f"mean_partial_L{L}",

            "sign_consistency":
                f"sign_consistency_L{L}"
        }
    )


    wide_list.append(
        temp
    )


robust_df = wide_list[0]


for temp in wide_list[1:]:

    robust_df = robust_df.merge(
        temp,
        on=[
            "stock_1",
            "stock_2"
        ],
        how="outer"
    )


frequency_cols = [
    f"frequency_L{L}"
    for L in BLOCK_LENGTHS
]


robust_df[
    "min_frequency"
] = (
    robust_df[
        frequency_cols
    ]
    .min(
        axis=1
    )
)


robust_df[
    "mean_frequency"
] = (
    robust_df[
        frequency_cols
    ]
    .mean(
        axis=1
    )
)


robust_df[
    "max_frequency"
] = (
    robust_df[
        frequency_cols
    ]
    .max(
        axis=1
    )
)


robust_df[
    "frequency_range"
] = (
    robust_df[
        "max_frequency"
    ]
    -
    robust_df[
        "min_frequency"
    ]
)


# 三个L全部>=0.8
robust_df[
    "stable_all_block_lengths"
] = (
    robust_df[
        "min_frequency"
    ]
    >=
    STABLE_THRESHOLD
)


# 三个L全部>=0.9
robust_df[
    "core_all_block_lengths"
] = (
    robust_df[
        "min_frequency"
    ]
    >=
    CORE_THRESHOLD
)


robust_df = (
    robust_df
    .sort_values(
        [
            "stable_all_block_lengths",
            "min_frequency",
            "mean_frequency"
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


robust_df.to_csv(
    ROBUST_EDGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 最终block-length robust边
# ============================================================

robust_stable_df = robust_df[
    robust_df[
        "stable_all_block_lengths"
    ]
]


robust_core_df = robust_df[
    robust_df[
        "core_all_block_lengths"
    ]
]


print(
    "\n======================================"
)

print(
    "Block-length Robustness"
)

print(
    "======================================"
)


print(
    "三个L下均 stable 的边数：",
    len(
        robust_stable_df
    )
)


print(
    "三个L下均 core-stable 的边数：",
    len(
        robust_core_df
    )
)


print(
    "\n最稳健的20条边："
)


print(
    robust_df[
        [
            "stock_1",
            "stock_2",
            "frequency_L10",
            "frequency_L20",
            "frequency_L40",
            "min_frequency",
            "frequency_range"
        ]
    ]
    .head(
        20
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 15. 检查阶段五的9条跨方法共同强边
# ============================================================

if STRONG_COMMON_FILE.exists():

    strong_df = pd.read_csv(
        STRONG_COMMON_FILE,
        dtype={
            "stock_1": str,
            "stock_2": str
        }
    )


    strong_df[
        "stock_1"
    ] = (
        strong_df[
            "stock_1"
        ]
        .apply(
            normalize_code
        )
    )


    strong_df[
        "stock_2"
    ] = (
        strong_df[
            "stock_2"
        ]
        .apply(
            normalize_code
        )
    )


    strong_robust_df = (
        strong_df[
            [
                "stock_1",
                "stock_2"
            ]
        ]
        .merge(
            robust_df,
            on=[
                "stock_1",
                "stock_2"
            ],
            how="left"
        )
    )


    strong_robust_df.to_csv(
        STRONG_EDGE_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "\n======================================"
    )

    print(
        "9条跨方法共同强边"
    )

    print(
        "======================================"
    )


    print(
        strong_robust_df[
            [
                "stock_1",
                "stock_2",
                "frequency_L10",
                "frequency_L20",
                "frequency_L40",
                "min_frequency",
                "stable_all_block_lengths"
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# 16. 图1：Stable / Core-stable边数
# ============================================================

plt.figure(
    figsize=(
        8,
        6
    )
)


plt.plot(
    summary_df[
        "block_length"
    ],

    summary_df[
        "stable_edges"
    ],

    marker="o",

    label=
        "Stable edges (>=0.80)"
)


plt.plot(
    summary_df[
        "block_length"
    ],

    summary_df[
        "core_stable_edges"
    ],

    marker="s",

    label=
        "Core-stable edges (>=0.90)"
)


plt.xlabel(
    "Block length"
)

plt.ylabel(
    "Number of edges"
)

plt.title(
    "不同Block Length下的稳定边数量"
)


plt.xticks(
    BLOCK_LENGTHS
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "block_length_stable_edge_counts.png",

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# 17. 图2：Baseline 1-SE edge retention
# ============================================================

plt.figure(
    figsize=(
        8,
        6
    )
)


plt.plot(
    summary_df[
        "block_length"
    ],

    summary_df[
        "baseline_edge_retention"
    ],

    marker="o"
)


plt.ylim(
    0,
    1.05
)


plt.xlabel(
    "Block length"
)

plt.ylabel(
    "1-SE edge retention"
)

plt.title(
    "不同Block Length下的1-SE网络边保留率"
)


plt.xticks(
    BLOCK_LENGTHS
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "block_length_1se_edge_retention.png",

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# 18. 完成
# ============================================================

print(
    "\n======================================"
)

print(
    "Block length稳健性分析完成"
)

print(
    "======================================"
)


print(
    "\n输出文件："
)


for path in [

    EDGE_STABILITY_FILE,
    SUMMARY_FILE,
    JACCARD_FILE,
    ROBUST_EDGE_FILE,
    STRONG_EDGE_FILE,
    DIAGNOSTICS_FILE,

    FIGURE_DIR
    / "block_length_stable_edge_counts.png",

    FIGURE_DIR
    / "block_length_1se_edge_retention.png"

]:

    print(
        path
    )
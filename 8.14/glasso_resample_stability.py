from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.lines import Line2D

from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# 0. 中文字体
# ============================================================

def set_chinese_font():

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC"
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
# 输入文件
# ------------------------------------------------------------

RETURN_FILE = (
    PROCESSED_DIR
    / "stock_returns.csv"
)

STOCK_INFO_FILE = (
    PROCESSED_DIR
    / "stock_info.csv"
)

SELECTION_FILE = (
    PROCESSED_DIR
    / "glasso_1se_selection.csv"
)

BASELINE_EDGE_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_edges.csv"
)

BASELINE_PARTIAL_FILE = (
    PROCESSED_DIR
    / "graphical_lasso_1se_partial_correlation.csv"
)


# ------------------------------------------------------------
# 输出文件
# ------------------------------------------------------------

EDGE_STABILITY_FILE = (
    PROCESSED_DIR
    / "glasso_edge_stability.csv"
)

STABLE_EDGE_FILE = (
    PROCESSED_DIR
    / "stable_glasso_edges.csv"
)

RUN_DIAGNOSTICS_FILE = (
    PROCESSED_DIR
    / "glasso_stability_run_diagnostics.csv"
)

STABLE_SUMMARY_FILE = (
    PROCESSED_DIR
    / "stable_glasso_network_summary.csv"
)

BASELINE_STABLE_COMPARE_FILE = (
    PROCESSED_DIR
    / "glasso_1se_vs_stable_summary.csv"
)


# ============================================================
# 2. 参数
# ============================================================

RANDOM_SEED = 20260814

# 重采样次数
N_RESAMPLES = 200

# 每次重采样约使用原样本的80%
SUBSAMPLE_RATIO = 0.80

# Moving-block长度
BLOCK_LENGTH = 20

# 稳定边阈值
STABLE_THRESHOLD = 0.80

# 核心稳定边阈值
CORE_THRESHOLD = 0.90

# 判断precision是否非零
ZERO_TOL = 1e-8

# Graphical Lasso数值参数
MAX_ITER = 5000

OUTER_TOL = 1e-4

ENET_TOL = 1e-6

# 是否严格剔除出现ConvergenceWarning的重复
REQUIRE_CONVERGENCE = True


# ============================================================
# 3. 工具：股票代码标准化
# ============================================================

def normalize_code(x):

    return str(
        x
    ).strip().zfill(6)


# ============================================================
# 4. 读取收益率
# ============================================================

returns = pd.read_csv(
    RETURN_FILE,
    index_col=0,
    parse_dates=True
)


returns.columns = [
    normalize_code(
        x
    )

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
    returns
    .values
    .astype(float)
)


n_samples = (
    X.shape[0]
)

n_stocks = (
    X.shape[1]
)


print(
    "原始收益率维度：",
    X.shape
)


# ============================================================
# 5. 读取alpha_1SE
# ============================================================

selection_df = pd.read_csv(
    SELECTION_FILE
)


if (
    "alpha_1se"
    not in selection_df.columns
):

    raise ValueError(
        "glasso_1se_selection.csv 中不存在 alpha_1se。"
    )


ALPHA_1SE = float(
    selection_df[
        "alpha_1se"
    ].iloc[0]
)


print(
    "固定的 1-SE alpha：",
    ALPHA_1SE
)


# ============================================================
# 6. 股票名称与行业
# ============================================================

stock_info = pd.read_csv(
    STOCK_INFO_FILE,
    dtype={
        "code": str
    }
)


stock_info[
    "code"
] = (
    stock_info[
        "code"
    ]
    .apply(
        normalize_code
    )
)


name_map = {}

industry_map = {}


if (
    "name"
    in stock_info.columns
):

    name_map = dict(
        zip(
            stock_info[
                "code"
            ],

            stock_info[
                "name"
            ]
        )
    )


if (
    "industry"
    in stock_info.columns
):

    industry_map = dict(
        zip(
            stock_info[
                "code"
            ],

            stock_info[
                "industry"
            ]
        )
    )


def get_name(code):

    value = name_map.get(
        code,
        code
    )

    if pd.isna(
        value
    ):

        return code

    return str(
        value
    )


def get_industry(code):

    value = industry_map.get(
        code,
        "未知行业"
    )

    if pd.isna(
        value
    ):

        return "未知行业"

    return str(
        value
    )


# ============================================================
# 7. Moving Block Bootstrap
# ============================================================

def moving_block_resample_indices(
    n,
    target_size,
    block_length,
    rng
):

    """
    从时间序列中随机抽取连续block，
    直到样本长度达到target_size。

    block之间允许重复。
    """

    if block_length > n:

        raise ValueError(
            "BLOCK_LENGTH不能大于样本长度。"
        )


    possible_starts = np.arange(
        0,
        n - block_length + 1
    )


    indices = []


    while len(
        indices
    ) < target_size:

        start = int(
            rng.choice(
                possible_starts
            )
        )


        block = list(
            range(
                start,
                start + block_length
            )
        )


        indices.extend(
            block
        )


    # 截取所需长度
    indices = np.asarray(
        indices[
            :target_size
        ],
        dtype=int
    )


    return indices


# ============================================================
# 8. Precision -> Partial Correlation
# ============================================================

def precision_to_partial(
    precision
):

    diagonal = np.sqrt(
        np.diag(
            precision
        )
    )


    partial = (
        -precision
        /
        np.outer(
            diagonal,
            diagonal
        )
    )


    np.fill_diagonal(
        partial,
        1.0
    )


    return partial


# ============================================================
# 9. 初始化稳定性统计
# ============================================================

selected_count = np.zeros(
    (
        n_stocks,
        n_stocks
    ),
    dtype=int
)


partial_sum = np.zeros(
    (
        n_stocks,
        n_stocks
    ),
    dtype=float
)


partial_sq_sum = np.zeros(
    (
        n_stocks,
        n_stocks
    ),
    dtype=float
)


positive_count = np.zeros(
    (
        n_stocks,
        n_stocks
    ),
    dtype=int
)


negative_count = np.zeros(
    (
        n_stocks,
        n_stocks
    ),
    dtype=int
)


rng = np.random.default_rng(
    RANDOM_SEED
)


target_size = int(
    np.floor(
        SUBSAMPLE_RATIO
        *
        n_samples
    )
)


print(
    "每次重采样样本量：",
    target_size
)


# ============================================================
# 10. 重采样主循环
# ============================================================

diagnostic_rows = []

valid_runs = 0


for b in range(
    N_RESAMPLES
):

    # --------------------------------------------------------
    # 10.1 Moving-block resample
    # --------------------------------------------------------

    sample_idx = moving_block_resample_indices(

        n=
            n_samples,

        target_size=
            target_size,

        block_length=
            BLOCK_LENGTH,

        rng=
            rng
    )


    X_b = (
        X[
            sample_idx,
            :
        ]
    )


    # --------------------------------------------------------
    # 10.2 每次重采样内部重新标准化
    #
    # 避免直接使用全样本的均值和标准差
    # --------------------------------------------------------

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


    # 若某变量方差异常为0，则跳过
    if np.any(
        std_b
        <
        1e-12
    ):

        diagnostic_rows.append(
            {
                "resample":
                    b + 1,

                "valid":
                    False,

                "converged":
                    False,

                "reason":
                    "near_zero_std",

                "n_iter":
                    np.nan,

                "dual_gap":
                    np.nan,

                "n_edges":
                    np.nan
            }
        )

        continue


    Z_b = (
        X_b
        -
        mean_b
    ) / std_b


    # --------------------------------------------------------
    # 10.3 拟合固定alpha的Graphical Lasso
    # --------------------------------------------------------

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
    ) as caught_warnings:

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


    # --------------------------------------------------------
    # 10.4 拟合失败
    # --------------------------------------------------------

    if fit_error is not None:

        diagnostic_rows.append(
            {
                "resample":
                    b + 1,

                "valid":
                    False,

                "converged":
                    False,

                "reason":
                    fit_error,

                "n_iter":
                    np.nan,

                "dual_gap":
                    np.nan,

                "n_edges":
                    np.nan
            }
        )

        continue


    # --------------------------------------------------------
    # 10.5 检查ConvergenceWarning
    # --------------------------------------------------------

    convergence_warning = any(

        issubclass(
            w.category,
            ConvergenceWarning
        )

        for w
        in caught_warnings
    )


    # --------------------------------------------------------
    # 10.6 最终dual gap
    # --------------------------------------------------------

    if (
        hasattr(
            model,
            "costs_"
        )
        and
        len(
            model.costs_
        )
        >
        0
    ):

        final_gap = float(
            model.costs_[
                -1
            ][1]
        )

    else:

        final_gap = np.nan


    precision = (
        model.precision_
    )


    # --------------------------------------------------------
    # 10.7 检查矩阵是否有限
    # --------------------------------------------------------

    finite_solution = np.all(
        np.isfinite(
            precision
        )
    )


    converged = (
        finite_solution
        and
        (
            not convergence_warning
        )
    )


    # --------------------------------------------------------
    # 10.8 边数
    # --------------------------------------------------------

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
                    precision[
                        i,
                        j
                    ]
                )
                >
                ZERO_TOL
            ):

                n_edges_current += 1


    # --------------------------------------------------------
    # 10.9 保存诊断
    # --------------------------------------------------------

    diagnostic_rows.append(
        {
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

            "reason":
                (
                    ""
                    if converged
                    else
                    "ConvergenceWarning"
                ),

            "n_iter":
                model.n_iter_,

            "dual_gap":
                final_gap,

            "n_edges":
                n_edges_current
        }
    )


    # --------------------------------------------------------
    # 若要求严格收敛，则不把警告重复用于频率统计
    # --------------------------------------------------------

    if (
        REQUIRE_CONVERGENCE
        and
        not converged
    ):

        continue


    # --------------------------------------------------------
    # 10.10 Partial correlation
    # --------------------------------------------------------

    partial = precision_to_partial(
        precision
    )


    # --------------------------------------------------------
    # 10.11 更新边选择统计
    # --------------------------------------------------------

    valid_runs += 1


    for i in range(
        n_stocks
    ):

        for j in range(
            i + 1,
            n_stocks
        ):

            if (
                abs(
                    precision[
                        i,
                        j
                    ]
                )
                >
                ZERO_TOL
            ):

                rho = float(
                    partial[
                        i,
                        j
                    ]
                )


                selected_count[
                    i,
                    j
                ] += 1


                partial_sum[
                    i,
                    j
                ] += rho


                partial_sq_sum[
                    i,
                    j
                ] += (
                    rho ** 2
                )


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


    if (
        (b + 1)
        % 20
        ==
        0
    ):

        print(
            f"完成 {b + 1}/{N_RESAMPLES} 次重采样，"
            f"当前有效重复数 = {valid_runs}"
        )


# ============================================================
# 11. 保存运行诊断
# ============================================================

diagnostic_df = pd.DataFrame(
    diagnostic_rows
)


diagnostic_df.to_csv(
    RUN_DIAGNOSTICS_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========================================"
)

print(
    "重采样运行情况"
)

print(
    "========================================"
)


print(
    "计划重采样次数：",
    N_RESAMPLES
)

print(
    "有效重复次数：",
    valid_runs
)

print(
    "有效比例：",
    round(
        valid_runs
        /
        N_RESAMPLES,
        4
    )
)


if valid_runs == 0:

    raise RuntimeError(
        "没有有效Graphical Lasso重采样结果。"
    )


# ============================================================
# 12. 读取全样本1-SE基准网络
# ============================================================

baseline_edges = pd.read_csv(
    BASELINE_EDGE_FILE,
    dtype={
        "stock_1": str,
        "stock_2": str
    }
)


baseline_edges[
    "stock_1"
] = (
    baseline_edges[
        "stock_1"
    ]
    .apply(
        normalize_code
    )
)


baseline_edges[
    "stock_2"
] = (
    baseline_edges[
        "stock_2"
    ]
    .apply(
        normalize_code
    )
)


def edge_key(
    a,
    b
):

    return tuple(
        sorted(
            [
                normalize_code(
                    a
                ),
                normalize_code(
                    b
                )
            ]
        )
    )


E_baseline = {

    edge_key(
        row.stock_1,
        row.stock_2
    )

    for row
    in baseline_edges.itertuples()
}


# ============================================================
# 13. 读取全样本1-SE偏相关矩阵
# ============================================================

baseline_partial = pd.read_csv(
    BASELINE_PARTIAL_FILE,
    index_col=0
)


baseline_partial.index = [
    normalize_code(
        x
    )
    for x in baseline_partial.index
]


baseline_partial.columns = [
    normalize_code(
        x
    )
    for x in baseline_partial.columns
]


baseline_partial = (
    baseline_partial
    .loc[
        codes,
        codes
    ]
)


# ============================================================
# 14. 计算每条边的selection frequency
# ============================================================

edge_rows = []


selection_frequency_matrix = np.zeros(
    (
        n_stocks,
        n_stocks
    )
)


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


        selection_frequency_matrix[
            i,
            j
        ] = frequency

        selection_frequency_matrix[
            j,
            i
        ] = frequency


        # ----------------------------------------------------
        # 只在该边被选中时统计partial
        # ----------------------------------------------------

        if count > 0:

            mean_partial = (
                partial_sum[
                    i,
                    j
                ]
                /
                count
            )


            mean_sq = (
                partial_sq_sum[
                    i,
                    j
                ]
                /
                count
            )


            variance_partial = max(
                mean_sq
                -
                mean_partial ** 2,
                0.0
            )


            sd_partial = np.sqrt(
                variance_partial
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


            dominant_sign = (
                "positive"
                if pos_fraction >= neg_fraction
                else "negative"
            )

        else:

            mean_partial = np.nan

            sd_partial = np.nan

            pos_fraction = np.nan

            neg_fraction = np.nan

            sign_consistency = np.nan

            dominant_sign = "never_selected"


        code_i = (
            codes[i]
        )

        code_j = (
            codes[j]
        )


        edge = edge_key(
            code_i,
            code_j
        )


        baseline_selected = (
            edge in E_baseline
        )


        baseline_rho = float(
            baseline_partial.loc[
                code_i,
                code_j
            ]
        )


        # ----------------------------------------------------
        # 稳定性分类
        # ----------------------------------------------------

        if (
            frequency
            >=
            CORE_THRESHOLD
        ):

            stability_class = (
                "core_stable"
            )

        elif (
            frequency
            >=
            STABLE_THRESHOLD
        ):

            stability_class = (
                "stable"
            )

        elif (
            frequency
            >=
            0.50
        ):

            stability_class = (
                "moderate"
            )

        else:

            stability_class = (
                "unstable"
            )


        edge_rows.append(
            {
                "stock_1":
                    code_i,

                "name_1":
                    get_name(
                        code_i
                    ),

                "industry_1":
                    get_industry(
                        code_i
                    ),

                "stock_2":
                    code_j,

                "name_2":
                    get_name(
                        code_j
                    ),

                "industry_2":
                    get_industry(
                        code_j
                    ),

                "selected_count":
                    count,

                "selection_frequency":
                    frequency,

                "mean_partial_when_selected":
                    mean_partial,

                "sd_partial_when_selected":
                    sd_partial,

                "positive_fraction_when_selected":
                    pos_fraction,

                "negative_fraction_when_selected":
                    neg_fraction,

                "sign_consistency":
                    sign_consistency,

                "dominant_sign":
                    dominant_sign,

                "baseline_1se_selected":
                    baseline_selected,

                "baseline_1se_partial":
                    baseline_rho,

                "stability_class":
                    stability_class,

                "same_industry":
                    (
                        get_industry(
                            code_i
                        )
                        ==
                        get_industry(
                            code_j
                        )
                    )
            }
        )


edge_stability_df = pd.DataFrame(
    edge_rows
)


edge_stability_df = (
    edge_stability_df
    .sort_values(
        [
            "selection_frequency",
            "sign_consistency"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


edge_stability_df.to_csv(
    EDGE_STABILITY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Stable edges
# ============================================================

stable_edges_df = edge_stability_df[
    edge_stability_df[
        "selection_frequency"
    ]
    >=
    STABLE_THRESHOLD
].copy()


stable_edges_df.to_csv(
    STABLE_EDGE_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========================================"
)

print(
    "稳定边统计"
)

print(
    "========================================"
)


print(
    f"Stable threshold = {STABLE_THRESHOLD:.2f}"
)

print(
    "稳定边数：",
    len(
        stable_edges_df
    )
)


print(
    f"Core threshold = {CORE_THRESHOLD:.2f}"
)

print(
    "核心稳定边数：",
    (
        edge_stability_df[
            "selection_frequency"
        ]
        >=
        CORE_THRESHOLD
    ).sum()
)


# ============================================================
# 16. 构建Stable Network
# ============================================================

G_stable = nx.Graph()


for code in codes:

    G_stable.add_node(

        code,

        name=
            get_name(
                code
            ),

        industry=
            get_industry(
                code
            )
    )


for row in stable_edges_df.itertuples():

    G_stable.add_edge(

        row.stock_1,
        row.stock_2,

        selection_frequency=
            float(
                row.selection_frequency
            ),

        mean_partial=
            float(
                row.mean_partial_when_selected
            ),

        sign_consistency=
            float(
                row.sign_consistency
            )
    )


# ============================================================
# 17. Stable Network统计
# ============================================================

n_nodes = (
    G_stable.number_of_nodes()
)

n_edges = (
    G_stable.number_of_edges()
)

density = (
    nx.density(
        G_stable
    )
)

n_components = (
    nx.number_connected_components(
        G_stable
    )
)

n_isolated = len(
    list(
        nx.isolates(
            G_stable
        )
    )
)


degrees = [
    d
    for _, d
    in G_stable.degree()
]


mean_degree = (
    np.mean(
        degrees
    )
)

max_degree = (
    np.max(
        degrees
    )
)


# ------------------------------------------------------------
# 同行业边比例
# ------------------------------------------------------------

same_industry_edges = sum(

    get_industry(
        u
    )
    ==
    get_industry(
        v
    )

    for u, v
    in G_stable.edges()
)


same_industry_ratio = (

    same_industry_edges
    /
    n_edges

    if n_edges > 0

    else np.nan
)


stable_summary_df = pd.DataFrame(
    [
        {
            "alpha_fixed":
                ALPHA_1SE,

            "n_resamples_requested":
                N_RESAMPLES,

            "n_valid_resamples":
                valid_runs,

            "subsample_ratio":
                SUBSAMPLE_RATIO,

            "block_length":
                BLOCK_LENGTH,

            "stable_threshold":
                STABLE_THRESHOLD,

            "core_threshold":
                CORE_THRESHOLD,

            "n_nodes":
                n_nodes,

            "n_edges":
                n_edges,

            "density":
                density,

            "mean_degree":
                mean_degree,

            "max_degree":
                max_degree,

            "n_components":
                n_components,

            "n_isolated":
                n_isolated,

            "same_industry_edges":
                same_industry_edges,

            "same_industry_edge_ratio":
                same_industry_ratio
        }
    ]
)


stable_summary_df.to_csv(
    STABLE_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. Baseline 1-SE vs Stable Network
# ============================================================

E_stable = {

    edge_key(
        row.stock_1,
        row.stock_2
    )

    for row
    in stable_edges_df.itertuples()
}


common = (
    E_baseline
    &
    E_stable
)

baseline_only = (
    E_baseline
    -
    E_stable
)

stable_only = (
    E_stable
    -
    E_baseline
)

union = (
    E_baseline
    |
    E_stable
)


jaccard = (

    len(
        common
    )
    /
    len(
        union
    )

    if len(
        union
    ) > 0

    else np.nan
)


baseline_retention = (

    len(
        common
    )
    /
    len(
        E_baseline
    )

    if len(
        E_baseline
    ) > 0

    else np.nan
)


stable_supported_by_baseline = (

    len(
        common
    )
    /
    len(
        E_stable
    )

    if len(
        E_stable
    ) > 0

    else np.nan
)


baseline_stable_df = pd.DataFrame(
    [
        {
            "baseline_1se_edges":
                len(
                    E_baseline
                ),

            "stable_edges":
                len(
                    E_stable
                ),

            "common_edges":
                len(
                    common
                ),

            "baseline_only_edges":
                len(
                    baseline_only
                ),

            "stable_only_edges":
                len(
                    stable_only
                ),

            "jaccard":
                jaccard,

            "baseline_edge_retention":
                baseline_retention,

            "stable_edge_supported_by_baseline":
                stable_supported_by_baseline
        }
    ]
)


baseline_stable_df.to_csv(
    BASELINE_STABLE_COMPARE_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========================================"
)

print(
    "全样本1-SE vs Stable Network"
)

print(
    "========================================"
)


print(
    baseline_stable_df
    .to_string(
        index=False
    )
)


# ============================================================
# 19. 输出最稳定的边
# ============================================================

print(
    "\n========================================"
)

print(
    "选择频率最高的20条边"
)

print(
    "========================================"
)


print(
    edge_stability_df[
        [
            "stock_1",
            "name_1",
            "stock_2",
            "name_2",
            "selection_frequency",
            "mean_partial_when_selected",
            "sign_consistency",
            "baseline_1se_selected"
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
# 20. 图1：Selection Frequency Histogram
# ============================================================

plt.figure(
    figsize=(
        8,
        6
    )
)


plt.hist(
    edge_stability_df[
        "selection_frequency"
    ],

    bins=
        np.linspace(
            0,
            1,
            21
        ),

    edgecolor=
        "black"
)


plt.axvline(
    STABLE_THRESHOLD,

    linestyle=
        "--",

    label=
        f"Stable threshold = {STABLE_THRESHOLD:.2f}"
)


plt.axvline(
    CORE_THRESHOLD,

    linestyle=
        ":",

    label=
        f"Core threshold = {CORE_THRESHOLD:.2f}"
)


plt.xlabel(
    "Edge selection frequency"
)

plt.ylabel(
    "Number of stock pairs"
)

plt.title(
    "Graphical Lasso边选择频率分布"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "glasso_edge_selection_frequency_histogram.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 21. 图2：Selection Frequency Heatmap
# ============================================================

heatmap_matrix = (
    selection_frequency_matrix.copy()
)


# 对角线不是边，设为NaN
np.fill_diagonal(
    heatmap_matrix,
    np.nan
)


labels = [

    f"{code}\n{get_name(code)}"

    for code in codes
]


fig, ax = plt.subplots(
    figsize=(
        13,
        11
    )
)


im = ax.imshow(
    heatmap_matrix,

    vmin=
        0,

    vmax=
        1,

    aspect=
        "auto"
)


ax.set_xticks(
    np.arange(
        n_stocks
    )
)

ax.set_yticks(
    np.arange(
        n_stocks
    )
)


ax.set_xticklabels(
    labels,

    rotation=
        90,

    fontsize=
        7
)


ax.set_yticklabels(
    labels,

    fontsize=
        7
)


cbar = plt.colorbar(
    im,
    ax=ax
)


cbar.set_label(
    "Selection frequency"
)


ax.set_title(
    "Graphical Lasso边选择频率矩阵"
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "glasso_edge_selection_frequency_heatmap.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 22. 图3：Stable Graphical Lasso Network
# ============================================================

industries = sorted(
    {
        get_industry(
            code
        )

        for code in codes
    }
)


cmap = plt.get_cmap(
    "tab20"
)


industry_color = {

    industry:
        cmap(
            i % 20
        )

    for i, industry
    in enumerate(
        industries
    )
}


node_colors = [

    industry_color[
        get_industry(
            code
        )
    ]

    for code in G_stable.nodes()
]


node_labels = {

    code:
        f"{code}\n{get_name(code)}"

    for code in G_stable.nodes()
}


# ------------------------------------------------------------
# 使用selection frequency作为布局权重
# 仅用于视觉展示
# ------------------------------------------------------------

pos = nx.spring_layout(

    G_stable,

    seed=
        42,

    weight=
        "selection_frequency"
)


fig, ax = plt.subplots(
    figsize=(
        15,
        12
    )
)


nx.draw_networkx_nodes(

    G_stable,
    pos,

    node_color=
        node_colors,

    node_size=
        1900,

    edgecolors=
        "black",

    linewidths=
        0.8,

    ax=
        ax
)


# ------------------------------------------------------------
# 逐边绘制
# 实线：平均偏相关>0
# 虚线：平均偏相关<0
# 边宽：selection frequency
# ------------------------------------------------------------

for u, v, data in G_stable.edges(
    data=True
):

    frequency = (
        data[
            "selection_frequency"
        ]
    )


    mean_partial = (
        data[
            "mean_partial"
        ]
    )


    # 0.8 -> 较细
    # 1.0 -> 较粗
    width = (
        1.5
        +
        5.0
        *
        (
            frequency
            -
            STABLE_THRESHOLD
        )
        /
        max(
            1.0
            -
            STABLE_THRESHOLD,
            1e-12
        )
    )


    style = (
        "solid"
        if mean_partial >= 0
        else "dashed"
    )


    nx.draw_networkx_edges(

        G_stable,
        pos,

        edgelist=[
            (
                u,
                v
            )
        ],

        width=
            width,

        style=
            style,

        alpha=
            0.8,

        ax=
            ax
    )


nx.draw_networkx_labels(

    G_stable,
    pos,

    labels=
        node_labels,

    font_size=
        7,

    ax=
        ax
)


# ------------------------------------------------------------
# 边标签：selection frequency
# ------------------------------------------------------------

edge_labels = {

    (
        u,
        v
    ):
        f"{data['selection_frequency']:.2f}"

    for u, v, data
    in G_stable.edges(
        data=True
    )
}


nx.draw_networkx_edge_labels(

    G_stable,
    pos,

    edge_labels=
        edge_labels,

    font_size=
        7,

    ax=
        ax
)


# ============================================================
# 23. 图例
# ============================================================

industry_handles = [

    Line2D(
        [0],
        [0],

        marker=
            "o",

        linestyle=
            "",

        markerfacecolor=
            color,

        markeredgecolor=
            "black",

        markersize=
            9,

        label=
            industry
    )

    for industry, color
    in industry_color.items()
]


edge_handles = [

    Line2D(
        [0],
        [0],

        linestyle=
            "solid",

        linewidth=
            2,

        label=
            "平均正偏相关"
    ),

    Line2D(
        [0],
        [0],

        linestyle=
            "dashed",

        linewidth=
            2,

        label=
            "平均负偏相关"
    )
]


legend1 = ax.legend(

    handles=
        industry_handles,

    title=
        "行业",

    loc=
        "upper left",

    bbox_to_anchor=
        (
            1.01,
            1.0
        )
)


ax.add_artist(
    legend1
)


ax.legend(

    handles=
        edge_handles,

    title=
        "关联方向",

    loc=
        "lower left",

    bbox_to_anchor=
        (
            1.01,
            0.0
        )
)


ax.set_title(
    "重采样稳定 Graphical Lasso 股票网络\n"
    f"alpha={ALPHA_1SE:.4f}, "
    f"selection frequency >= {STABLE_THRESHOLD:.2f}"
)


ax.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    FIGURE_DIR
    / "stable_graphical_lasso_network.png",

    dpi=
        300,

    bbox_inches=
        "tight"
)


plt.show()


# ============================================================
# 24. 完成
# ============================================================

print(
    "\n========================================"
)

print(
    "阶段六完成"
)

print(
    "========================================"
)


print(
    "\n输出文件："
)


for path in [

    EDGE_STABILITY_FILE,

    STABLE_EDGE_FILE,

    RUN_DIAGNOSTICS_FILE,

    STABLE_SUMMARY_FILE,

    BASELINE_STABLE_COMPARE_FILE,

    FIGURE_DIR
    / "glasso_edge_selection_frequency_histogram.png",

    FIGURE_DIR
    / "glasso_edge_selection_frequency_heatmap.png",

    FIGURE_DIR
    / "stable_graphical_lasso_network.png"
]:

    print(
        path
    )
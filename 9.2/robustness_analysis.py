from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# 0. Project directory
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. Paths
# ============================================================

# Stage 1 outputs:
#
# regime_samples/
#   R3_returns.csv
#   R4_returns.csv
#
REGIME_SAMPLE_DIR = (
    BASE_DIR
    / "regime_samples"
)


# Robustness outputs
OUTPUT_DIR = (
    BASE_DIR
    / "stage4_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. General settings
# ============================================================

R3 = "R3"
R4 = "R4"


# ------------------------------------------------------------
# Bootstrap size
#
# First debugging:
#     B = 100
#
# Final analysis:
#     B = 1000
# ------------------------------------------------------------

BOOTSTRAP_B = 1000


SIGNIFICANCE_LEVEL = 0.05

RANDOM_SEED = 20260902


# GLasso numerical settings
MAX_ITER = 5000
TOL = 1e-4


# ------------------------------------------------------------
# Preprocessing
#
# Keep the same definition as Stage 2-4:
# standardize each Regime separately.
#
# Hence the test concerns conditional-association structure
# after removing stage-specific marginal mean and scale.
# ------------------------------------------------------------

PREPROCESSING = (
    "within_stage_standardize"
)


# ------------------------------------------------------------
# Descriptive alpha from earlier Rolling / Regime analysis
# ------------------------------------------------------------

ALPHA_1SE = 0.216910


# ============================================================
# 3. Helper functions
# ============================================================

def normalize_code(x) -> str:
    """
    Convert stock codes to six-digit strings.
    """

    x = str(x).strip()

    if x.endswith(".0"):
        x = x[:-2]

    if x.isdigit():
        x = x.zfill(6)

    return x


def detect_date_column(
    df: pd.DataFrame,
) -> str | None:

    candidates = [
        "日期",
        "date",
        "Date",
        "DATE",
        "network_date",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


# ============================================================
# 4. Load Regime sample
# ============================================================

def load_regime_sample(
    regime: str,
) -> tuple[np.ndarray, list[str], pd.DatetimeIndex | None]:

    file = (
        REGIME_SAMPLE_DIR
        / f"{regime}_returns.csv"
    )

    if not file.exists():

        raise FileNotFoundError(
            f"Cannot find:\n{file}"
        )


    df = pd.read_csv(
        file
    )


    date_col = detect_date_column(
        df
    )


    dates = None


    if date_col is not None:

        df[date_col] = pd.to_datetime(
            df[date_col]
        )

        df = (
            df
            .sort_values(
                date_col
            )
            .drop_duplicates(
                subset=date_col
            )
            .reset_index(drop=True)
        )

        dates = pd.DatetimeIndex(
            df[date_col]
        )


    stock_cols_original = [
        col
        for col in df.columns
        if col != date_col
    ]


    rename_map = {
        col:
            normalize_code(col)

        for col
        in stock_cols_original
    }


    df = df.rename(
        columns=rename_map
    )


    stock_cols = [
        rename_map[col]
        for col
        in stock_cols_original
    ]


    df[stock_cols] = (
        df[stock_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )


    complete_mask = (
        df[stock_cols]
        .notna()
        .all(axis=1)
    )


    if not complete_mask.all():

        n_drop = int(
            (~complete_mask).sum()
        )

        print(
            f"{regime}: dropping "
            f"{n_drop} incomplete observations."
        )


        df = (
            df.loc[
                complete_mask
            ]
            .reset_index(drop=True)
        )


        if date_col is not None:

            dates = pd.DatetimeIndex(
                df[date_col]
            )


    X = (
        df[stock_cols]
        .to_numpy(
            dtype=float
        )
    )


    return (
        X,
        stock_cols,
        dates,
    )


# ============================================================
# 5. Preprocessing
# ============================================================

def preprocess_stage(
    X: np.ndarray,
) -> np.ndarray:

    X = np.asarray(
        X,
        dtype=float,
    )


    mean = X.mean(
        axis=0,
        keepdims=True,
    )


    sd = X.std(
        axis=0,
        ddof=0,
        keepdims=True,
    )


    if np.any(
        sd <= 0
    ):

        raise ValueError(
            "Zero-variance variable detected."
        )


    Z = (
        X
        -
        mean
    ) / sd


    return Z


# ============================================================
# 6. Empirical covariance
# ============================================================

def empirical_covariance(
    X: np.ndarray,
) -> np.ndarray:

    n = X.shape[0]

    return (
        X.T
        @
        X
    ) / n


# ============================================================
# 7. GLasso fit
# ============================================================

def fit_glasso(
    S: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, bool]:

    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )


        covariance_hat, precision_hat = (
            graphical_lasso(
                emp_cov=S,
                alpha=alpha,
                max_iter=MAX_ITER,
                tol=TOL,
                mode="cd",
            )
        )


    convergence_warning = any(
        issubclass(
            w.category,
            ConvergenceWarning,
        )
        for w in caught
    )


    return (
        covariance_hat,
        precision_hat,
        convergence_warning,
    )


# ============================================================
# 8. Gaussian likelihood loss
# ============================================================

def gaussian_loss(
    S: np.ndarray,
    precision: np.ndarray,
) -> float:

    sign, logdet = (
        np.linalg.slogdet(
            precision
        )
    )


    if sign <= 0:

        raise ValueError(
            "Precision matrix is not positive definite."
        )


    return float(
        np.trace(
            S
            @
            precision
        )
        -
        logdet
    )


# ============================================================
# 9. Edge count
# ============================================================

def count_edges(
    precision: np.ndarray,
    tol: float = 1e-8,
) -> int:

    p = precision.shape[0]

    upper = np.triu_indices(
        p,
        k=1,
    )


    return int(
        np.sum(
            np.abs(
                precision[
                    upper
                ]
            )
            >
            tol
        )
    )


# ============================================================
# 10. Fit H0 and H1 and calculate D
# ============================================================

def fit_null_and_alternative(
    samples: list[np.ndarray],
    alpha: float,
) -> dict:

    n_list = [
        X.shape[0]
        for X in samples
    ]


    p = samples[0].shape[1]

    N = sum(
        n_list
    )


    # --------------------------------------------------------
    # Stage-specific covariance
    # --------------------------------------------------------

    stage_covs = [
        empirical_covariance(
            X
        )
        for X in samples
    ]


    # --------------------------------------------------------
    # H0: common precision matrix
    # --------------------------------------------------------

    pooled = np.vstack(
        samples
    )


    pooled_cov = (
        empirical_covariance(
            pooled
        )
    )


    (
        _,
        precision_null,
        warning_null,
    ) = fit_glasso(
        pooled_cov,
        alpha,
    )


    # --------------------------------------------------------
    # H1: stage-specific precision matrices
    # --------------------------------------------------------

    precision_alt = []

    warning_alt = []


    for S in stage_covs:

        (
            _,
            precision_t,
            warning_t,
        ) = fit_glasso(
            S,
            alpha,
        )


        precision_alt.append(
            precision_t
        )

        warning_alt.append(
            warning_t
        )


    # --------------------------------------------------------
    # Test statistic
    #
    # D =
    #
    # 1/(N p^2)
    # sum_t n_t[
    #    L(S_t, Omega_0)
    #    -
    #    L(S_t, Omega_t)
    # ]
    # --------------------------------------------------------

    numerator = 0.0

    null_losses = []

    alt_losses = []


    for (
        n_t,
        S_t,
        precision_t,
    ) in zip(
        n_list,
        stage_covs,
        precision_alt,
    ):

        loss_null = (
            gaussian_loss(
                S_t,
                precision_null,
            )
        )


        loss_alt = (
            gaussian_loss(
                S_t,
                precision_t,
            )
        )


        null_losses.append(
            loss_null
        )

        alt_losses.append(
            loss_alt
        )


        numerator += (
            n_t
            *
            (
                loss_null
                -
                loss_alt
            )
        )


    D = (
        numerator
        /
        (
            N
            *
            p**2
        )
    )


    return {
        "D":
            float(D),

        "precision_null":
            precision_null,

        "precision_alt":
            precision_alt,

        "warning_null":
            warning_null,

        "warning_alt":
            warning_alt,

        "null_losses":
            null_losses,

        "alt_losses":
            alt_losses,
    }


# ============================================================
# 11. Generic bootstrap equality test
# ============================================================

def run_bootstrap_test(
    X3_raw: np.ndarray,
    X4_raw: np.ndarray,
    alpha: float,
    B: int,
    seed: int,
    label: str,
) -> tuple[dict, pd.DataFrame]:

    """
    Run one R3/R4 whole-network equality test.

    Returns
    -------
    summary_dict
    bootstrap_dataframe
    """

    # --------------------------------------------------------
    # 11.1 Standardize observed stages independently
    # --------------------------------------------------------

    X3 = preprocess_stage(
        X3_raw
    )

    X4 = preprocess_stage(
        X4_raw
    )


    n3 = X3.shape[0]
    n4 = X4.shape[0]

    p = X3.shape[1]

    N = (
        n3
        +
        n4
    )


    # --------------------------------------------------------
    # 11.2 Observed test statistic
    # --------------------------------------------------------

    observed_fit = (
        fit_null_and_alternative(
            [
                X3,
                X4,
            ],
            alpha,
        )
    )


    D_obs = (
        observed_fit[
            "D"
        ]
    )


    null_edges = (
        count_edges(
            observed_fit[
                "precision_null"
            ]
        )
    )


    r3_edges = (
        count_edges(
            observed_fit[
                "precision_alt"
            ][0]
        )
    )


    r4_edges = (
        count_edges(
            observed_fit[
                "precision_alt"
            ][1]
        )
    )


    # --------------------------------------------------------
    # 11.3 Null pooled population
    # --------------------------------------------------------

    pooled = np.vstack(
        [
            X3,
            X4,
        ]
    )


    rng = (
        np.random.default_rng(
            seed
        )
    )


    bootstrap_rows = []

    failed = 0

    convergence_warnings = 0


    # --------------------------------------------------------
    # 11.4 Bootstrap
    # --------------------------------------------------------

    for b in range(
        1,
        B + 1,
    ):

        try:

            # Draw N observations under the pooled null
            idx = rng.integers(
                0,
                N,
                size=N,
            )


            X_star = (
                pooled[
                    idx,
                    :
                ]
            )


            # Preserve original stage sizes
            X3_star = (
                X_star[
                    :n3,
                    :
                ]
                .copy()
            )


            X4_star = (
                X_star[
                    n3:,
                    :
                ]
                .copy()
            )


            # Mimic observed stage-wise preprocessing
            X3_star = (
                preprocess_stage(
                    X3_star
                )
            )


            X4_star = (
                preprocess_stage(
                    X4_star
                )
            )


            fit_star = (
                fit_null_and_alternative(
                    [
                        X3_star,
                        X4_star,
                    ],
                    alpha,
                )
            )


            D_star = (
                fit_star[
                    "D"
                ]
            )


            warning_flag = (
                fit_star[
                    "warning_null"
                ]
                or
                any(
                    fit_star[
                        "warning_alt"
                    ]
                )
            )


            if warning_flag:

                convergence_warnings += 1


            bootstrap_rows.append(
                {
                    "analysis":
                        label,

                    "alpha":
                        alpha,

                    "bootstrap_id":
                        b,

                    "D_star":
                        D_star,

                    "success":
                        True,

                    "convergence_warning":
                        warning_flag,
                }
            )


        except Exception as exc:

            failed += 1


            bootstrap_rows.append(
                {
                    "analysis":
                        label,

                    "alpha":
                        alpha,

                    "bootstrap_id":
                        b,

                    "D_star":
                        np.nan,

                    "success":
                        False,

                    "convergence_warning":
                        False,

                    "error":
                        str(exc),
                }
            )


        if (
            b % 100 == 0
            or
            b == B
        ):

            print(
                f"{label}: "
                f"{b}/{B}"
            )


    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )


    valid = (
        bootstrap_df.loc[
            bootstrap_df[
                "success"
            ]
            &
            bootstrap_df[
                "D_star"
            ]
            .notna(),

            "D_star",
        ]
        .to_numpy()
    )


    B_valid = len(
        valid
    )


    if B_valid == 0:

        raise RuntimeError(
            f"No valid bootstrap samples for {label}."
        )


    # --------------------------------------------------------
    # 11.5 Critical value and p-value
    # --------------------------------------------------------

    q95 = float(
        np.quantile(
            valid,
            1
            -
            SIGNIFICANCE_LEVEL,
        )
    )


    exceed = int(
        np.sum(
            valid
            >
            D_obs
        )
    )


    p_boot = (
        exceed
        /
        B_valid
    )


    p_plus_one = (
        exceed
        +
        1
    ) / (
        B_valid
        +
        1
    )


    reject = bool(
        D_obs
        >
        q95
    )


    # --------------------------------------------------------
    # 11.6 Summary
    # --------------------------------------------------------

    result = {
        "analysis":
            label,

        "n_R3":
            n3,

        "n_R4":
            n4,

        "p":
            p,

        "alpha":
            alpha,

        "D_observed":
            D_obs,

        "critical_value_95":
            q95,

        "bootstrap_pvalue":
            p_boot,

        "plus_one_pvalue":
            p_plus_one,

        "reject_H0":
            reject,

        "null_edges":
            null_edges,

        "R3_edges_H1":
            r3_edges,

        "R4_edges_H1":
            r4_edges,

        "B_requested":
            B,

        "B_valid":
            B_valid,

        "B_failed":
            failed,

        "convergence_warnings":
            convergence_warnings,

        "bootstrap_mean":
            float(
                np.mean(
                    valid
                )
            ),

        "bootstrap_sd":
            float(
                np.std(
                    valid,
                    ddof=1,
                )
            ),
    }


    return (
        result,
        bootstrap_df,
    )


# ============================================================
# 12. Load original R3/R4 data
# ============================================================

(
    X3_raw,
    stock3,
    dates3,
) = load_regime_sample(
    R3
)


(
    X4_raw,
    stock4,
    dates4,
) = load_regime_sample(
    R4
)


if stock3 != stock4:

    raise ValueError(
        "R3 and R4 have different stock columns/order."
    )


n3 = X3_raw.shape[0]

n4 = X4_raw.shape[0]

p = X3_raw.shape[1]


print("=" * 78)
print("Stage 4 Robustness Analysis")
print("=" * 78)


print(
    f"Full R3: n = {n3}"
)

print(
    f"Full R4: n = {n4}"
)

print(
    f"p = {p}"
)


# ============================================================
# 13. Baseline inference alpha
# ============================================================

baseline_alpha = (
    np.log(p)
    /
    min(
        n3,
        n4,
    )
)


print(
    f"\nBaseline inference alpha = "
    f"{baseline_alpha:.8f}"
)


# ============================================================
# 14. Stage 4R-1:
#     Penalty sensitivity
# ============================================================

# Use np.unique to avoid duplicate alpha values.
ALPHA_GRID = np.unique(
    np.array(
        [
            baseline_alpha,
            0.06,
            0.08,
            0.10,
            0.15,
            ALPHA_1SE,
        ],
        dtype=float,
    )
)


print("\n" + "=" * 78)
print("Stage 4R-1: Penalty Sensitivity")
print("=" * 78)


penalty_results = []

penalty_bootstraps = []


for k, alpha in enumerate(
    ALPHA_GRID
):

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        f"Testing alpha = "
        f"{alpha:.8f}"
    )


    result, boot = (
        run_bootstrap_test(
            X3_raw=X3_raw,
            X4_raw=X4_raw,
            alpha=float(alpha),
            B=BOOTSTRAP_B,

            # Different but reproducible seed per alpha
            seed=(
                RANDOM_SEED
                +
                k
                *
                10000
            ),

            label=(
                f"full_R3_R4_"
                f"alpha_{alpha:.6f}"
            ),
        )
    )


    penalty_results.append(
        result
    )

    penalty_bootstraps.append(
        boot
    )


penalty_summary = pd.DataFrame(
    penalty_results
)


penalty_bootstrap_df = pd.concat(
    penalty_bootstraps,
    ignore_index=True,
)


penalty_summary.to_csv(
    OUTPUT_DIR
    / "penalty_sensitivity_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


penalty_bootstrap_df.to_csv(
    OUTPUT_DIR
    / "penalty_sensitivity_bootstrap_statistics.csv",

    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 78)
print("Penalty sensitivity results")
print("=" * 78)


print(
    penalty_summary[
        [
            "alpha",
            "R3_edges_H1",
            "R4_edges_H1",
            "D_observed",
            "critical_value_95",
            "bootstrap_pvalue",
            "reject_H0",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 15. Stage 4R-2:
#     Equal-length R3 vs R4
# ============================================================

print("\n" + "=" * 78)
print("Stage 4R-2: Equal-length Comparison")
print("=" * 78)


# ------------------------------------------------------------
# Use n_equal = smaller stage length
# ------------------------------------------------------------

n_equal = min(
    n3,
    n4,
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Use the LAST n_equal observations of R3.
#
# Reason:
# They are the R3 observations closest to the R3 -> R4
# transition and therefore give a more local pre/post
# comparison.
# ------------------------------------------------------------

X3_equal = (
    X3_raw[
        -n_equal:,
        :
    ]
    .copy()
)


X4_equal = (
    X4_raw[
        :n_equal,
        :
    ]
    .copy()
)


if dates3 is not None:

    dates3_equal = (
        dates3[
            -n_equal:
        ]
    )

    print(
        "Equal-length R3 period:"
    )

    print(
        dates3_equal.min().date(),
        "to",
        dates3_equal.max().date(),
    )


if dates4 is not None:

    dates4_equal = (
        dates4[
            :n_equal
        ]
    )

    print(
        "Equal-length R4 period:"
    )

    print(
        dates4_equal.min().date(),
        "to",
        dates4_equal.max().date(),
    )


print(
    f"\nn_equal = "
    f"{n_equal}"
)


# ------------------------------------------------------------
# Very useful feature:
#
# Because the original smaller sample is already R4 = 61,
#
#     alpha = log(p) / 61
#
# is IDENTICAL to the baseline full-sample Stage-4 alpha.
#
# Thus this comparison changes the sample balance but does
# not change the baseline penalty.
# ------------------------------------------------------------

equal_length_alpha = (
    np.log(p)
    /
    n_equal
)


print(
    f"Equal-length alpha = "
    f"{equal_length_alpha:.8f}"
)


equal_result, equal_bootstrap = (
    run_bootstrap_test(
        X3_raw=X3_equal,
        X4_raw=X4_equal,
        alpha=equal_length_alpha,
        B=BOOTSTRAP_B,
        seed=(
            RANDOM_SEED
            +
            999999
        ),
        label="equal_length_R3late_vs_R4",
    )
)


equal_summary = pd.DataFrame(
    [
        equal_result
    ]
)


equal_summary.to_csv(
    OUTPUT_DIR
    / "equal_length_R3_R4_test.csv",

    index=False,
    encoding="utf-8-sig",
)


equal_bootstrap.to_csv(
    OUTPUT_DIR
    / "equal_length_bootstrap_statistics.csv",

    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 78)
print("Equal-length result")
print("=" * 78)


print(
    equal_summary[
        [
            "n_R3",
            "n_R4",
            "alpha",
            "R3_edges_H1",
            "R4_edges_H1",
            "D_observed",
            "critical_value_95",
            "bootstrap_pvalue",
            "reject_H0",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 16. Compare original full-sample baseline
#     vs equal-length test
# ============================================================

baseline_row = (
    penalty_summary
    .iloc[
        np.argmin(
            np.abs(
                penalty_summary[
                    "alpha"
                ]
                -
                baseline_alpha
            )
        )
    ]
)


comparison = pd.DataFrame(
    [
        {
            "design":
                "Full R3 vs R4",

            "n_R3":
                int(
                    baseline_row[
                        "n_R3"
                    ]
                ),

            "n_R4":
                int(
                    baseline_row[
                        "n_R4"
                    ]
                ),

            "alpha":
                baseline_row[
                    "alpha"
                ],

            "D_observed":
                baseline_row[
                    "D_observed"
                ],

            "critical_value_95":
                baseline_row[
                    "critical_value_95"
                ],

            "bootstrap_pvalue":
                baseline_row[
                    "bootstrap_pvalue"
                ],

            "reject_H0":
                baseline_row[
                    "reject_H0"
                ],
        },

        {
            "design":
                "Equal-length late R3 vs R4",

            "n_R3":
                equal_result[
                    "n_R3"
                ],

            "n_R4":
                equal_result[
                    "n_R4"
                ],

            "alpha":
                equal_result[
                    "alpha"
                ],

            "D_observed":
                equal_result[
                    "D_observed"
                ],

            "critical_value_95":
                equal_result[
                    "critical_value_95"
                ],

            "bootstrap_pvalue":
                equal_result[
                    "bootstrap_pvalue"
                ],

            "reject_H0":
                equal_result[
                    "reject_H0"
                ],
        },
    ]
)


comparison.to_csv(
    OUTPUT_DIR
    / "full_vs_equal_length_comparison.csv",

    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 78)
print("Full vs Equal-length comparison")
print("=" * 78)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 17. Plot 1:
#     Penalty vs bootstrap p-value
# ============================================================

plot_penalty = (
    penalty_summary
    .sort_values(
        "alpha"
    )
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)


ax.plot(
    plot_penalty[
        "alpha"
    ],
    plot_penalty[
        "bootstrap_pvalue"
    ],
    marker="o",
)


ax.axhline(
    SIGNIFICANCE_LEVEL,
    linestyle="--",
    label="5% significance level",
)


ax.set_xlabel(
    "Graphical Lasso penalty alpha"
)


ax.set_ylabel(
    "Bootstrap p-value"
)


ax.set_title(
    "Penalty Sensitivity of the R3-R4 "
    "Network Equality Test"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "penalty_sensitivity_pvalues.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 18. Plot 2:
#     Observed D vs 95% critical value across penalties
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)


ax.plot(
    plot_penalty[
        "alpha"
    ],
    plot_penalty[
        "D_observed"
    ],
    marker="o",
    label="Observed D",
)


ax.plot(
    plot_penalty[
        "alpha"
    ],
    plot_penalty[
        "critical_value_95"
    ],
    marker="o",
    label="95% bootstrap critical value",
)


ax.set_xlabel(
    "Graphical Lasso penalty alpha"
)


ax.set_ylabel(
    "Test statistic"
)


ax.set_title(
    "Observed Statistic and Bootstrap Critical Value "
    "Across Penalties"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "penalty_sensitivity_D_vs_critical.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 19. Plot 3:
#     Equal-length bootstrap distribution
# ============================================================

valid_equal_D = (
    equal_bootstrap.loc[
        equal_bootstrap[
            "success"
        ]
        &
        equal_bootstrap[
            "D_star"
        ]
        .notna(),

        "D_star",
    ]
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)


ax.hist(
    valid_equal_D,
    bins=30,
    alpha=0.75,
)


ax.axvline(
    equal_result[
        "critical_value_95"
    ],
    linestyle="--",
    linewidth=2,
    label="95% critical value",
)


ax.axvline(
    equal_result[
        "D_observed"
    ],
    linewidth=2,
    label="Observed D",
)


ax.set_xlabel(
    "Bootstrap D*"
)


ax.set_ylabel(
    "Frequency"
)


ax.set_title(
    "Equal-length R3 vs R4 Bootstrap Distribution"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "equal_length_bootstrap_distribution.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 20. Final status
# ============================================================

print("\n" + "=" * 78)
print("Stage 4 robustness analysis completed")
print("=" * 78)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR.resolve()
)


print(
    "\nMain files:"
)

print(
    "  penalty_sensitivity_summary.csv"
)

print(
    "  penalty_sensitivity_bootstrap_statistics.csv"
)

print(
    "  equal_length_R3_R4_test.csv"
)

print(
    "  equal_length_bootstrap_statistics.csv"
)

print(
    "  full_vs_equal_length_comparison.csv"
)

print(
    "  penalty_sensitivity_pvalues.png"
)

print(
    "  penalty_sensitivity_D_vs_critical.png"
)

print(
    "  equal_length_bootstrap_distribution.png"
)
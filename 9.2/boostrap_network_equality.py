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

# Stage 1 outputs
REGIME_SAMPLE_DIR = (
    BASE_DIR
    / "regime_samples"
)

OUTPUT_DIR = (
    BASE_DIR
    / "bootstrap_network_inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Main settings
# ============================================================

REGIMES_TO_TEST = [
    "R3",
    "R4",
]


# Bootstrap size:
#
# First debugging run:
#     B = 100 or 200
#
# Final main result:
#     B = 1000
#
# Zhang et al. use B = 1000.
BOOTSTRAP_B = 1000


# Significance level
SIGNIFICANCE_LEVEL = 0.05


# Random seed for reproducibility
RANDOM_SEED = 20260902


# GLasso numerical settings
MAX_ITER = 5000
TOL = 1e-4


# ============================================================
# 3. Preprocessing
# ============================================================

# Stage 2/3 standardized each Regime separately.
#
# Therefore the primary Stage-4 analysis below tests equality
# of the conditional-association structures after removing
# stage-specific marginal location and scale.
#
# Available choices:
#
#   "within_stage_standardize"
#       Main recommendation:
#       consistent with Stage 2/3.
#
#   "center_only"
#       Closer to Zhang et al.'s covariance-precision
#       formulation.
#
PREPROCESSING = "within_stage_standardize"


# In the bootstrap, repeat the same stage-wise standardization
# used in the observed estimator.
#
# This makes the bootstrap procedure mimic the estimator used
# in the observed Stage-2/3 analysis.
RESTANDARDIZE_BOOTSTRAP_STAGES = True


# ============================================================
# 4. Inference penalty
# ============================================================

# IMPORTANT:
#
# Do NOT use alpha_1SE = 0.216910 as the primary formal
# inference penalty.
#
# Zhang et al. use:
#
#       lambda = log(p) / n
#
# Their stages have equal sample sizes.
#
# Here R3 and R4 have unequal n, so we adopt a COMMON
# preselected lambda based on the smaller Regime sample:
#
#       lambda = log(p) / min(n_R3, n_R4)
#
# Using the same lambda in the null and alternative models
# avoids mechanically generating differences by changing
# penalties across stages.
#
# This is an adaptation for our unequal-stage design.
PENALTY_RULE = "logp_over_min_n"


# Optional fixed penalty for sensitivity analysis only.
FIXED_ALPHA = 0.216910


# ============================================================
# 5. Helper: detect date column
# ============================================================

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
# 6. Helper: normalize stock code
# ============================================================

def normalize_code(x) -> str:

    x = str(x).strip()

    if x.endswith(".0"):
        x = x[:-2]

    if x.isdigit():
        x = x.zfill(6)

    return x


# ============================================================
# 7. Load one Stage-1 Regime sample
# ============================================================

def load_regime_sample(
    regime: str,
) -> tuple[pd.DataFrame, list[str]]:

    file = (
        REGIME_SAMPLE_DIR
        / f"{regime}_returns.csv"
    )

    if not file.exists():

        raise FileNotFoundError(
            f"Missing Stage-1 file:\n{file}"
        )


    df = pd.read_csv(
        file
    )


    date_col = detect_date_column(
        df
    )


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


    stock_cols = [
        col
        for col in df.columns
        if col != date_col
    ]


    rename_map = {
        col:
            normalize_code(col)

        for col in stock_cols
    }


    df = df.rename(
        columns=rename_map
    )


    stock_cols = [
        rename_map[col]
        for col in stock_cols
    ]


    df[stock_cols] = (
        df[stock_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )


    n_before = len(df)


    df = (
        df
        .dropna(
            subset=stock_cols
        )
        .reset_index(drop=True)
    )


    n_after = len(df)


    if n_after < n_before:

        print(
            f"{regime}: dropped "
            f"{n_before - n_after} incomplete rows."
        )


    return (
        df,
        stock_cols,
    )


# ============================================================
# 8. Preprocess one stage
# ============================================================

def preprocess_stage(
    X: np.ndarray,
    mode: str,
) -> np.ndarray:

    X = np.asarray(
        X,
        dtype=float,
    )


    if mode == "center_only":

        mean = X.mean(
            axis=0,
            keepdims=True,
        )

        return (
            X
            -
            mean
        )


    if mode == "within_stage_standardize":

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
                "Zero-variance variable detected "
                "during standardization."
            )


        return (
            X
            -
            mean
        ) / sd


    raise ValueError(
        f"Unknown preprocessing mode: {mode}"
    )


# ============================================================
# 9. Empirical covariance / second moment
#
# Paper:
#
#       S = (1/n) Y'Y
#
# because the sample is assumed centered.
#
# We deliberately use denominator n, not n-1.
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
# 10. GLasso estimation from empirical covariance
# ============================================================

def fit_glasso(
    S: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, bool]:

    """
    Returns
    -------
    covariance_hat
    precision_hat
    convergence_warning
    """

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
            warning.category,
            ConvergenceWarning,
        )
        for warning
        in caught
    )


    return (
        covariance_hat,
        precision_hat,
        convergence_warning,
    )


# ============================================================
# 11. Negative Gaussian log-likelihood loss
#
# L(S; Omega)
# =
# trace(S Omega) - log det(Omega)
#
# IMPORTANT:
# The test statistic uses this UNPENALIZED likelihood loss.
# The precision matrix itself is estimated using penalization.
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
            "Precision matrix is not "
            "positive definite."
        )


    trace_term = np.trace(
        S
        @
        precision
    )


    return float(
        trace_term
        -
        logdet
    )


# ============================================================
# 12. Count nonzero GLasso edges
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
# 13. Fit null and alternative models
# ============================================================

def fit_null_and_alternative(
    stage_samples: list[np.ndarray],
    alpha: float,
) -> dict:

    """
    H0:
        All stages share one common precision matrix.

    H1:
        Each stage has its own precision matrix.
    """

    n_list = [
        X.shape[0]
        for X in stage_samples
    ]


    p = (
        stage_samples[0]
        .shape[1]
    )


    N = sum(
        n_list
    )


    # --------------------------------------------------------
    # Stage-specific covariance matrices
    # --------------------------------------------------------

    stage_covariances = [
        empirical_covariance(
            X
        )
        for X
        in stage_samples
    ]


    # --------------------------------------------------------
    # H0:
    # pooled sample -> common precision matrix
    # --------------------------------------------------------

    pooled = np.vstack(
        stage_samples
    )


    pooled_covariance = (
        empirical_covariance(
            pooled
        )
    )


    (
        covariance_null,
        precision_null,
        warning_null,
    ) = fit_glasso(
        pooled_covariance,
        alpha,
    )


    # --------------------------------------------------------
    # H1:
    # each stage gets its own precision matrix
    # --------------------------------------------------------

    precision_alt = []

    covariance_alt = []

    warning_alt = []


    for S_t in stage_covariances:

        (
            covariance_t,
            precision_t,
            warning_t,
        ) = fit_glasso(
            S_t,
            alpha,
        )


        covariance_alt.append(
            covariance_t
        )

        precision_alt.append(
            precision_t
        )

        warning_alt.append(
            warning_t
        )


    # --------------------------------------------------------
    # Calculate the observed likelihood-ratio statistic
    #
    # Zhang et al.:
    #
    # D =
    # 1/(N p^2)
    # sum_t n_t[
    #   L(S_t; Omega_0)
    #   -
    #   L(S_t; Omega_t)
    # ]
    #
    # --------------------------------------------------------

    null_losses = []

    alt_losses = []


    numerator = 0.0


    for (
        n_t,
        S_t,
        precision_t,
    ) in zip(
        n_list,
        stage_covariances,
        precision_alt,
    ):

        loss_null_t = (
            gaussian_loss(
                S_t,
                precision_null,
            )
        )


        loss_alt_t = (
            gaussian_loss(
                S_t,
                precision_t,
            )
        )


        null_losses.append(
            loss_null_t
        )

        alt_losses.append(
            loss_alt_t
        )


        numerator += (
            n_t
            *
            (
                loss_null_t
                -
                loss_alt_t
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

        "covariance_null":
            covariance_null,

        "covariance_alt":
            covariance_alt,

        "stage_covariances":
            stage_covariances,

        "null_losses":
            null_losses,

        "alt_losses":
            alt_losses,

        "warning_null":
            warning_null,

        "warning_alt":
            warning_alt,

        "n_list":
            n_list,

        "p":
            p,

        "N":
            N,
    }


# ============================================================
# 14. Load R3 and R4
# ============================================================

raw_stage_samples = []

stock_reference = None


for regime in REGIMES_TO_TEST:

    df, stock_cols = (
        load_regime_sample(
            regime
        )
    )


    if stock_reference is None:

        stock_reference = (
            stock_cols
        )

    else:

        if stock_cols != stock_reference:

            raise ValueError(
                "R3 and R4 do not have "
                "the same stock columns/order."
            )


    X = (
        df[
            stock_cols
        ]
        .to_numpy(
            dtype=float
        )
    )


    raw_stage_samples.append(
        X
    )


# ============================================================
# 15. Basic dimensions
# ============================================================

n_list = [
    X.shape[0]
    for X in raw_stage_samples
]


p = (
    raw_stage_samples[0]
    .shape[1]
)


N = sum(
    n_list
)


print("=" * 78)
print("Stage 4: Bootstrap Network Equality Test")
print("=" * 78)


for regime, n in zip(
    REGIMES_TO_TEST,
    n_list,
):

    print(
        f"{regime}: n = {n}"
    )


print(
    f"p = {p}"
)

print(
    f"N = {N}"
)


# ============================================================
# 16. Choose a COMMON inference penalty
# ============================================================

if PENALTY_RULE == "logp_over_min_n":

    n_reference = min(
        n_list
    )


    inference_alpha = (
        np.log(p)
        /
        n_reference
    )


elif PENALTY_RULE == "fixed":

    n_reference = np.nan

    inference_alpha = (
        FIXED_ALPHA
    )


else:

    raise ValueError(
        f"Unknown PENALTY_RULE: "
        f"{PENALTY_RULE}"
    )


print(
    f"Inference alpha = "
    f"{inference_alpha:.8f}"
)

print(
    f"Penalty rule = "
    f"{PENALTY_RULE}"
)


# ============================================================
# 17. Preprocess observed R3/R4 samples
# ============================================================

observed_samples = [
    preprocess_stage(
        X,
        PREPROCESSING,
    )

    for X
    in raw_stage_samples
]


# ============================================================
# 18. Observed statistic
# ============================================================

observed_fit = (
    fit_null_and_alternative(
        stage_samples=observed_samples,
        alpha=inference_alpha,
    )
)


D_observed = (
    observed_fit[
        "D"
    ]
)


print("\n" + "=" * 78)
print("Observed test statistic")
print("=" * 78)


print(
    f"D_observed = "
    f"{D_observed:.10f}"
)


# ============================================================
# 19. Observed model diagnostics
# ============================================================

null_edges = count_edges(
    observed_fit[
        "precision_null"
    ]
)


alt_edges = [
    count_edges(
        precision
    )

    for precision
    in observed_fit[
        "precision_alt"
    ]
]


print(
    f"H0 common network edges = "
    f"{null_edges}"
)


for regime, edges in zip(
    REGIMES_TO_TEST,
    alt_edges,
):

    print(
        f"H1 {regime} edges = "
        f"{edges}"
    )


# ============================================================
# 20. Construct pooled null sample
# ============================================================

# Under H0, all stage observations come from one common
# graphical structure.
#
# Following Zhang et al., bootstrap observations are drawn
# from the combined sample.
pooled_observed = np.vstack(
    observed_samples
)


rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# 21. Bootstrap
# ============================================================

bootstrap_rows = []

n_failed = 0
n_convergence_warning = 0


print("\n" + "=" * 78)
print(
    f"Running {BOOTSTRAP_B} bootstrap replications..."
)
print("=" * 78)


for b in range(
    1,
    BOOTSTRAP_B + 1,
):

    try:

        # ----------------------------------------------------
        # 21.1 Sample N rows with replacement from pooled data
        # ----------------------------------------------------

        bootstrap_indices = (
            rng.integers(
                low=0,
                high=N,
                size=N,
            )
        )


        bootstrap_full = (
            pooled_observed[
                bootstrap_indices,
                :
            ]
        )


        # ----------------------------------------------------
        # 21.2 Partition into original stage sizes
        # ----------------------------------------------------

        bootstrap_stages = []

        start = 0


        for n_t in n_list:

            end = (
                start
                +
                n_t
            )


            X_star_t = (
                bootstrap_full[
                    start:end,
                    :
                ]
                .copy()
            )


            # -----------------------------------------------
            # Repeat Stage-2 preprocessing inside bootstrap
            # -----------------------------------------------

            if (
                PREPROCESSING
                ==
                "within_stage_standardize"
                and
                RESTANDARDIZE_BOOTSTRAP_STAGES
            ):

                X_star_t = (
                    preprocess_stage(
                        X_star_t,
                        PREPROCESSING,
                    )
                )


            elif (
                PREPROCESSING
                ==
                "center_only"
            ):

                # Optional:
                # keep article-like bootstrap second moments.
                #
                # To enforce exact centering instead, replace
                # with:
                #
                # X_star_t = preprocess_stage(
                #     X_star_t,
                #     "center_only",
                # )
                pass


            bootstrap_stages.append(
                X_star_t
            )


            start = end


        # ----------------------------------------------------
        # 21.3 Estimate H0/H1 and calculate D*
        # ----------------------------------------------------

        bootstrap_fit = (
            fit_null_and_alternative(
                stage_samples=bootstrap_stages,
                alpha=inference_alpha,
            )
        )


        D_star = (
            bootstrap_fit[
                "D"
            ]
        )


        warning_flag = (
            bootstrap_fit[
                "warning_null"
            ]
            or
            any(
                bootstrap_fit[
                    "warning_alt"
                ]
            )
        )


        if warning_flag:

            n_convergence_warning += 1


        bootstrap_rows.append(
            {
                "bootstrap_id":
                    b,

                "D_star":
                    D_star,

                "convergence_warning":
                    warning_flag,

                "success":
                    True,
            }
        )


    except Exception as exc:

        n_failed += 1


        bootstrap_rows.append(
            {
                "bootstrap_id":
                    b,

                "D_star":
                    np.nan,

                "convergence_warning":
                    False,

                "success":
                    False,

                "error":
                    str(exc),
            }
        )


    # --------------------------------------------------------
    # Progress report
    # --------------------------------------------------------

    if (
        b % 100 == 0
        or
        b == BOOTSTRAP_B
    ):

        print(
            f"Completed "
            f"{b}/{BOOTSTRAP_B}"
        )


# ============================================================
# 22. Bootstrap distribution
# ============================================================

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


bootstrap_df.to_csv(
    OUTPUT_DIR
    / "R3_R4_bootstrap_statistics.csv",

    index=False,
    encoding="utf-8-sig",
)


valid_D = (
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
    valid_D
)


if B_valid == 0:

    raise RuntimeError(
        "All bootstrap replications failed."
    )


# ============================================================
# 23. Critical value and p-value
# ============================================================

critical_value = (
    np.quantile(
        valid_D,
        1
        -
        SIGNIFICANCE_LEVEL,
    )
)


# ------------------------------------------------------------
# Paper p-value:
#
#     #{D_b* > D} / B
# ------------------------------------------------------------

n_exceed = int(
    np.sum(
        valid_D
        >
        D_observed
    )
)


paper_pvalue = (
    n_exceed
    /
    B_valid
)


# ------------------------------------------------------------
# Plus-one version:
#
# Often convenient in finite Monte Carlo samples because it
# avoids reporting an exact p-value of zero.
#
# This is NOT the formula reported by Zhang et al.;
# therefore both values are saved separately.
# ------------------------------------------------------------

plus_one_pvalue = (
    n_exceed
    +
    1
) / (
    B_valid
    +
    1
)


reject_null = bool(
    D_observed
    >
    critical_value
)


# ============================================================
# 24. Observed model diagnostics table
# ============================================================

diagnostic_rows = []


for idx, regime in enumerate(
    REGIMES_TO_TEST
):

    diagnostic_rows.append(
        {
            "model":
                regime,

            "n":
                n_list[idx],

            "alpha":
                inference_alpha,

            "edges_H1":
                alt_edges[idx],

            "loss_H0":
                observed_fit[
                    "null_losses"
                ][idx],

            "loss_H1":
                observed_fit[
                    "alt_losses"
                ][idx],

            "loss_improvement":
                (
                    observed_fit[
                        "null_losses"
                    ][idx]
                    -
                    observed_fit[
                        "alt_losses"
                    ][idx]
                ),

            "convergence_warning_H1":
                observed_fit[
                    "warning_alt"
                ][idx],
        }
    )


diagnostics = pd.DataFrame(
    diagnostic_rows
)


diagnostics.to_csv(
    OUTPUT_DIR
    / "R3_R4_observed_model_diagnostics.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 25. Final test summary
# ============================================================

summary = pd.DataFrame(
    [
        {
            "test":
                "R3_vs_R4_precision_equality",

            "null_hypothesis":
                "Omega_R3 = Omega_R4",

            "alternative_hypothesis":
                "Omega_R3 != Omega_R4",

            "preprocessing":
                PREPROCESSING,

            "penalty_rule":
                PENALTY_RULE,

            "alpha_glasso":
                inference_alpha,

            "n_R3":
                n_list[0],

            "n_R4":
                n_list[1],

            "p":
                p,

            "N":
                N,

            "B_requested":
                BOOTSTRAP_B,

            "B_valid":
                B_valid,

            "B_failed":
                n_failed,

            "B_convergence_warning":
                n_convergence_warning,

            "D_observed":
                D_observed,

            "critical_value_95":
                critical_value,

            "bootstrap_exceedances":
                n_exceed,

            "paper_bootstrap_pvalue":
                paper_pvalue,

            "plus_one_pvalue":
                plus_one_pvalue,

            "significance_level":
                SIGNIFICANCE_LEVEL,

            "reject_H0":
                reject_null,

            "null_network_edges":
                null_edges,

            "R3_network_edges_inference":
                alt_edges[0],

            "R4_network_edges_inference":
                alt_edges[1],
        }
    ]
)


summary.to_csv(
    OUTPUT_DIR
    / "R3_R4_bootstrap_network_test.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 26. Histogram
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8.5,
        5.5,
    )
)


ax.hist(
    valid_D,
    bins=30,
    alpha=0.75,
)


ax.axvline(
    critical_value,
    linestyle="--",
    linewidth=2,
    label=(
        f"95% critical value "
        f"= {critical_value:.5g}"
    ),
)


ax.axvline(
    D_observed,
    linestyle="-",
    linewidth=2,
    label=(
        f"Observed D "
        f"= {D_observed:.5g}"
    ),
)


ax.set_xlabel(
    "Bootstrap test statistic D*"
)


ax.set_ylabel(
    "Frequency"
)


ax.set_title(
    "Bootstrap Distribution under "
    "H0: Omega_R3 = Omega_R4"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "R3_R4_bootstrap_distribution.png",

    dpi=300,
    bbox_inches="tight",
)


plt.close()


# ============================================================
# 27. Print results
# ============================================================

print("\n" + "=" * 78)
print("Bootstrap Network Equality Test Result")
print("=" * 78)


print(
    f"D observed            = "
    f"{D_observed:.10f}"
)

print(
    f"95% critical value    = "
    f"{critical_value:.10f}"
)

print(
    f"Exceedances           = "
    f"{n_exceed}/{B_valid}"
)

print(
    f"Paper bootstrap p     = "
    f"{paper_pvalue:.6f}"
)

print(
    f"Plus-one p            = "
    f"{plus_one_pvalue:.6f}"
)

print(
    f"Reject H0 at 5%       = "
    f"{reject_null}"
)


print(
    f"Failed bootstrap runs = "
    f"{n_failed}"
)

print(
    f"Convergence warnings  = "
    f"{n_convergence_warning}"
)


print("\nInterpretation:")


if reject_null:

    print(
        "The observed R3-R4 whole-network difference "
        "lies beyond the 95% bootstrap null distribution."
    )

    print(
        "Under the current bootstrap-MGGM specification, "
        "H0: Omega_R3 = Omega_R4 is rejected."
    )

else:

    print(
        "The observed R3-R4 difference does not exceed "
        "the 95% bootstrap critical value."
    )

    print(
        "Under the current bootstrap-MGGM specification, "
        "there is insufficient evidence to reject "
        "H0: Omega_R3 = Omega_R4."
    )


print("\nOutputs written to:")

print(
    OUTPUT_DIR.resolve()
)
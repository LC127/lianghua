from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.covariance import GraphicalLasso


# ============================================================
# 1. Paths
# ============================================================

INPUT_DIR = Path("regime_samples")
OUTPUT_DIR = Path("regime_networks")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Main settings
# ============================================================

REGIMES = [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
]

# Keep the same GLasso penalty used in the previous
# Rolling Graphical Lasso analysis.
ALPHA_1SE = 0.216910

MAX_ITER = 5000
TOL = 1e-4

# GLasso should theoretically give exact zeros.
# This tolerance is only used to avoid numerical floating-point
# noise when deciding whether an edge exists.
EDGE_TOL = 1e-8


# ============================================================
# 3. Helper: identify date column
# ============================================================

def detect_date_column(df: pd.DataFrame) -> str:
    """
    Detect the date column in a regime-sample CSV.
    """

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

    raise ValueError(
        "No date column found. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# 4. Helper: load one regime sample
# ============================================================

def load_regime_sample(
    regime: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Load R1_returns.csv, ..., R5_returns.csv.
    """

    path = INPUT_DIR / f"{regime}_returns.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage-1 file: {path}"
        )

    df = pd.read_csv(path)

    date_col = detect_date_column(df)

    df[date_col] = pd.to_datetime(
        df[date_col]
    )

    df = (
        df
        .sort_values(date_col)
        .drop_duplicates(subset=date_col)
        .reset_index(drop=True)
    )

    stock_cols = [
        c
        for c in df.columns
        if c != date_col
    ]

    # Keep stock codes as 6-digit strings whenever possible.
    stock_cols_new = []

    rename_map = {}

    for col in stock_cols:

        col_str = str(col)

        if col_str.isdigit():
            new_col = col_str.zfill(6)
        else:
            new_col = col_str

        rename_map[col] = new_col
        stock_cols_new.append(new_col)

    df = df.rename(
        columns=rename_map
    )

    stock_cols = stock_cols_new

    # Numeric conversion.
    df[stock_cols] = df[stock_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Require complete observations for all stocks.
    n_before = len(df)

    df = (
        df
        .dropna(subset=stock_cols)
        .reset_index(drop=True)
    )

    n_after = len(df)

    if n_after < n_before:
        print(
            f"{regime}: dropped "
            f"{n_before - n_after} incomplete rows."
        )

    return df, date_col, stock_cols


# ============================================================
# 5. Helper: standardize returns within each regime
# ============================================================

def standardize_returns(
    X: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Standardize each stock return series within the regime.

    Z_ij = (X_ij - mean_j) / sd_j

    ddof=0 is used to be consistent with the common
    StandardScaler convention used in rolling estimation.
    """

    means = X.mean(axis=0)

    stds = X.std(
        axis=0,
        ddof=0,
    )

    if (stds <= 0).any():

        bad = list(
            stds[stds <= 0].index
        )

        raise ValueError(
            "Zero-variance stock return series detected: "
            f"{bad}"
        )

    Z = (
        (X - means)
        / stds
    )

    return (
        Z.to_numpy(dtype=float),
        means,
        stds,
    )


# ============================================================
# 6. Helper: precision matrix -> partial correlation matrix
# ============================================================

def precision_to_partial(
    precision: np.ndarray,
) -> np.ndarray:
    """
    Convert precision matrix Omega into the partial-correlation
    matrix:

        rho_ij = - Omega_ij /
                 sqrt(Omega_ii * Omega_jj)

    Diagonal entries are set to 1.
    """

    diag = np.diag(
        precision
    )

    if np.any(diag <= 0):
        raise ValueError(
            "Precision matrix has non-positive "
            "diagonal entries."
        )

    scale = np.sqrt(
        np.outer(
            diag,
            diag,
        )
    )

    partial = (
        -precision
        / scale
    )

    np.fill_diagonal(
        partial,
        1.0,
    )

    return partial


# ============================================================
# 7. Helper: construct edge list
# ============================================================

def build_edge_list(
    precision: np.ndarray,
    partial: np.ndarray,
    stock_cols: list[str],
) -> pd.DataFrame:
    """
    An undirected edge is retained when the corresponding
    off-diagonal precision element is nonzero.
    """

    p = len(stock_cols)

    rows = []

    for i in range(p):

        for j in range(
            i + 1,
            p,
        ):

            omega_ij = precision[i, j]

            rho_ij = partial[i, j]

            if abs(omega_ij) > EDGE_TOL:

                rows.append(
                    {
                        "stock_i":
                            stock_cols[i],

                        "stock_j":
                            stock_cols[j],

                        "precision":
                            omega_ij,

                        "partial_corr":
                            rho_ij,

                        "abs_partial_corr":
                            abs(rho_ij),

                        "sign":
                            (
                                "positive"
                                if rho_ij > 0
                                else "negative"
                            ),
                    }
                )

    edges = pd.DataFrame(rows)

    if not edges.empty:

        edges = edges.sort_values(
            "abs_partial_corr",
            ascending=False,
        ).reset_index(drop=True)

    return edges


# ============================================================
# 8. Main estimation loop
# ============================================================

summary_rows = []

reference_lambda_rows = []

for regime in REGIMES:

    print("\n" + "=" * 80)
    print(f"Estimating {regime}")
    print("=" * 80)

    # --------------------------------------------------------
    # Step 8.1 Load Stage-1 sample
    # --------------------------------------------------------

    (
        df,
        date_col,
        stock_cols,
    ) = load_regime_sample(
        regime
    )

    X_df = df[stock_cols].copy()

    n = len(X_df)
    p = len(stock_cols)

    if p != 15:

        print(
            f"WARNING: expected p=15, "
            f"but found p={p}."
        )

    print(
        f"Sample size n = {n}"
    )

    print(
        f"Number of stocks p = {p}"
    )

    print(
        f"Date range: "
        f"{df[date_col].min().date()} "
        f"to "
        f"{df[date_col].max().date()}"
    )


    # --------------------------------------------------------
    # Step 8.2 Standardize within regime
    # --------------------------------------------------------

    (
        Z,
        means,
        stds,
    ) = standardize_returns(
        X_df
    )

    # Check numerical standardization.
    max_abs_mean = np.max(
        np.abs(
            Z.mean(axis=0)
        )
    )

    max_sd_error = np.max(
        np.abs(
            Z.std(
                axis=0,
                ddof=0,
            )
            - 1.0
        )
    )

    print(
        f"Max |standardized mean| = "
        f"{max_abs_mean:.3e}"
    )

    print(
        f"Max |standardized SD - 1| = "
        f"{max_sd_error:.3e}"
    )


    # --------------------------------------------------------
    # Step 8.3 Estimate Regime-specific GLasso
    # --------------------------------------------------------

    model = GraphicalLasso(
        alpha=ALPHA_1SE,
        max_iter=MAX_ITER,
        tol=TOL,

        # Z has already been centered.
        assume_centered=True,
    )

    model.fit(Z)

    covariance = (
        model.covariance_
        .copy()
    )

    precision = (
        model.precision_
        .copy()
    )


    # --------------------------------------------------------
    # Step 8.4 Convert precision to partial correlation
    # --------------------------------------------------------

    partial = precision_to_partial(
        precision
    )


    # --------------------------------------------------------
    # Step 8.5 Build edge list
    # --------------------------------------------------------

    edges = build_edge_list(
        precision=precision,
        partial=partial,
        stock_cols=stock_cols,
    )


    # --------------------------------------------------------
    # Step 8.6 Network summary
    # --------------------------------------------------------

    n_possible_edges = (
        p * (p - 1) // 2
    )

    n_edges = len(edges)

    density = (
        n_edges
        / n_possible_edges
    )

    if n_edges > 0:

        mean_abs_partial = (
            edges[
                "abs_partial_corr"
            ].mean()
        )

        median_abs_partial = (
            edges[
                "abs_partial_corr"
            ].median()
        )

        max_abs_partial = (
            edges[
                "abs_partial_corr"
            ].max()
        )

        n_positive = int(
            (
                edges["partial_corr"]
                > 0
            ).sum()
        )

        n_negative = int(
            (
                edges["partial_corr"]
                < 0
            ).sum()
        )

    else:

        mean_abs_partial = np.nan
        median_abs_partial = np.nan
        max_abs_partial = np.nan

        n_positive = 0
        n_negative = 0


    # --------------------------------------------------------
    # Step 8.7 Numerical diagnostics
    # --------------------------------------------------------

    eig_precision = np.linalg.eigvalsh(
        precision
    )

    min_precision_eig = (
        eig_precision.min()
    )

    max_precision_eig = (
        eig_precision.max()
    )

    condition_precision = (
        max_precision_eig
        / min_precision_eig
    )

    eig_covariance = np.linalg.eigvalsh(
        covariance
    )

    min_covariance_eig = (
        eig_covariance.min()
    )


    # --------------------------------------------------------
    # Step 8.8 Save matrices
    # --------------------------------------------------------

    covariance_df = pd.DataFrame(
        covariance,
        index=stock_cols,
        columns=stock_cols,
    )

    precision_df = pd.DataFrame(
        precision,
        index=stock_cols,
        columns=stock_cols,
    )

    partial_df = pd.DataFrame(
        partial,
        index=stock_cols,
        columns=stock_cols,
    )


    covariance_df.to_csv(
        OUTPUT_DIR
        / f"{regime}_covariance.csv",

        encoding="utf-8-sig",
    )

    precision_df.to_csv(
        OUTPUT_DIR
        / f"{regime}_precision.csv",

        encoding="utf-8-sig",
    )

    partial_df.to_csv(
        OUTPUT_DIR
        / f"{regime}_partial.csv",

        encoding="utf-8-sig",
    )

    edges.to_csv(
        OUTPUT_DIR
        / f"{regime}_edges.csv",

        index=False,
        encoding="utf-8-sig",
    )


    # Save standardization parameters so that the entire
    # estimation is reproducible.
    scaling_df = pd.DataFrame(
        {
            "stock":
                stock_cols,

            "mean_return":
                means.values,

            "std_return":
                stds.values,
        }
    )

    scaling_df.to_csv(
        OUTPUT_DIR
        / f"{regime}_standardization.csv",

        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # Step 8.9 Save summary row
    # --------------------------------------------------------

    summary_rows.append(
        {
            "regime":
                regime,

            "start_date":
                df[date_col].min().date(),

            "end_date":
                df[date_col].max().date(),

            "n_days":
                n,

            "p_stocks":
                p,

            "alpha":
                ALPHA_1SE,

            "n_edges":
                n_edges,

            "n_possible_edges":
                n_possible_edges,

            "density":
                density,

            "mean_abs_partial_selected":
                mean_abs_partial,

            "median_abs_partial_selected":
                median_abs_partial,

            "max_abs_partial_selected":
                max_abs_partial,

            "n_positive_edges":
                n_positive,

            "n_negative_edges":
                n_negative,

            "n_iter":
                model.n_iter_,

            "min_precision_eigenvalue":
                min_precision_eig,

            "precision_condition_number":
                condition_precision,

            "min_covariance_eigenvalue":
                min_covariance_eig,
        }
    )


    # --------------------------------------------------------
    # Step 8.10 Paper-inspired lambda diagnostic
    # --------------------------------------------------------

    lambda_reference = (
        np.log(p)
        / n
    )

    reference_lambda_rows.append(
        {
            "regime":
                regime,

            "n_days":
                n,

            "p_stocks":
                p,

            "log_p_over_n":
                lambda_reference,
        }
    )


    print(
        f"GLasso iterations = "
        f"{model.n_iter_}"
    )

    print(
        f"Edges = "
        f"{n_edges}/{n_possible_edges}"
    )

    print(
        f"Density = "
        f"{density:.4f}"
    )

    print(
        f"Mean |partial| = "
        f"{mean_abs_partial:.4f}"
        if n_edges > 0
        else
        "Mean |partial| = NA"
    )

    print(
        f"min eig(Omega) = "
        f"{min_precision_eig:.6f}"
    )


# ============================================================
# 9. Save overall network summary
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUTPUT_DIR
    / "regime_network_summary.csv",

    index=False,
    encoding="utf-8-sig",
)


reference_lambda = pd.DataFrame(
    reference_lambda_rows
)

reference_lambda.to_csv(
    OUTPUT_DIR
    / "regime_lambda_reference.csv",

    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. Print final comparison
# ============================================================

print("\n" + "=" * 80)
print("Regime-specific GLasso summary")
print("=" * 80)

display_cols = [
    "regime",
    "n_days",
    "n_edges",
    "density",
    "mean_abs_partial_selected",
    "n_positive_edges",
    "n_negative_edges",
    "n_iter",
]

print(
    summary[
        display_cols
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("Reference lambda = log(p) / n")
print("=" * 80)

print(
    reference_lambda.to_string(
        index=False
    )
)


print("\nOutputs written to:")
print(
    OUTPUT_DIR.resolve()
)
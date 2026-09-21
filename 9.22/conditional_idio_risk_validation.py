from __future__ import annotations

from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
OUT = ROOT / "output"

DAY7_ROOT = OUT / "M1_day7"
DAY9_ROOT = OUT / "M1_day9"

REGIME_DIR = (
    DAY9_ROOT
    / "02_stage2_pit_regime_classification"
)

REGIME_PATH = (
    REGIME_DIR / "pit_regime_panel.parquet"
)

REGIME_CSV = (
    REGIME_DIR / "pit_regime_panel.csv"
)

REGIME_QA = (
    REGIME_DIR / "pit_regime_qa.csv"
)

OUTPUT_DIR = (
    DAY9_ROOT
    / "04_stage4_regime_conditional_idio_risk"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "conditional_idio_risk_summary.csv"
)

FOCUS_PATH = (
    OUTPUT_DIR
    / "delta_degree_idio_focus.csv"
)

SOURCE_PATH = (
    OUTPUT_DIR
    / "conditional_idio_risk_source_manifest.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "conditional_idio_risk_qa.csv"
)

META_PATH = (
    OUTPUT_DIR
    / "day9_step4_metadata.json"
)


# ============================================================
# 1. Frozen design
# ============================================================

TARGET = "future_idio_vol_annualized"

FACTORS = [
    "residual_degree_percentile",
    "delta_degree_percentile",
    "cross_industry_degree_percentile",
    "cross_industry_degree_ratio",
    "delta_residual_degree_percentile_1m",
    "delta_cross_industry_degree_percentile_1m",
    "neighbor_jaccard_1m",
    "neighbor_retention_1m",
    "outside_community_degree_percentile",
]

METHODS = [
    "RAW_COMMON",
    "INDUSTRY_SIZE_COMMON",
    "FULL_CHARACTERISTIC_NEUTRAL",
]

# Main Day-9 question:
# network state and network reconfiguration only.
REGIMES = {
    "NETWORK":
        "network_stress_regime",

    "RECONFIG":
        "network_reconfig_regime",
}

REGIME_ORDER = [
    "LOW",
    "MEDIUM",
    "HIGH",
]

HAC_LAGS = 3
SMALL_SAMPLE_N = 12

EXPECTED_GROUPS = (
    len(FACTORS)
    * 3
    * len(METHODS)
)  # 81

# Set a path here only if automatic discovery picks the wrong file.
RISK_IC_INPUT = None


# ============================================================
# 2. Aliases
# ============================================================

KEY_ALIASES = {
    "analysis_date":
        ["analysis_date", "date"],

    "window":
        ["window", "network_window"],

    "factor_name":
        ["factor_name", "factor"],

    "method":
        [
            "method",
            "control_method",
            "neutralization_method",
            "specification",
            "spec",
        ],
}

IC_ALIASES = [
    "risk_rank_ic",
    "rank_ic",
    "risk_ic",
    "spearman_ic",
    "ic",
    "ic_value",
]

TARGET_ALIASES = [
    "target_name",
    "risk_metric",
    "risk_name",
    "label",
    "label_name",
    "target",
    "outcome",
    "dependent_variable",
]

METHOD_MAP = {
    "RAW": "RAW_COMMON",
    "RAW_COMMON": "RAW_COMMON",

    "INDUSTRY_SIZE":
        "INDUSTRY_SIZE_COMMON",

    "INDUSTRY_SIZE_COMMON":
        "INDUSTRY_SIZE_COMMON",

    "FULL":
        "FULL_CHARACTERISTIC_NEUTRAL",

    "FULL_CONTROLS":
        "FULL_CHARACTERISTIC_NEUTRAL",

    "FULL_CHARACTERISTIC_NEUTRAL":
        "FULL_CHARACTERISTIC_NEUTRAL",
}


# ============================================================
# 3. Utilities
# ============================================================

def save_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )
    os.replace(tmp, path)


def parse_bool(x):
    return str(x).strip().lower() in {
        "true", "1", "yes"
    }


def find_col(columns, aliases):
    lookup = {
        str(c).strip().lower(): c
        for c in columns
    }

    for x in aliases:
        if x.lower() in lookup:
            return lookup[x.lower()]

    return None


def file_columns(path):
    if path.suffix.lower() == ".csv":
        return list(
            pd.read_csv(
                path,
                nrows=0,
            ).columns
        )

    return list(
        pq.ParquetFile(path)
        .schema_arrow
        .names
    )


# ============================================================
# 4. NumPy Newey-West HAC
# ============================================================

def newey_west_ols(y, X, maxlags=3):

    y = np.asarray(
        y,
        dtype=float,
    )

    X = np.asarray(
        X,
        dtype=float,
    )

    n = len(y)

    inv = np.linalg.pinv(
        X.T @ X
    )

    beta = (
        inv
        @ X.T
        @ y
    )

    resid = (
        y
        -
        X @ beta
    )

    xu = (
        X
        *
        resid[:, None]
    )

    S = xu.T @ xu

    L = min(
        maxlags,
        n - 1,
    )

    for lag in range(
        1,
        L + 1,
    ):

        w = (
            1.0
            -
            lag / (L + 1.0)
        )

        G = (
            xu[lag:].T
            @
            xu[:-lag]
        )

        S += w * (
            G + G.T
        )

    cov = (
        inv
        @ S
        @ inv
    )

    return beta, cov


# ============================================================
# 5. Locate frozen Day-7 IVOL IC
# ============================================================

def discover_risk_ic():

    if RISK_IC_INPUT is not None:

        path = Path(
            RISK_IC_INPUT
        )

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    candidates = []

    for path in DAY7_ROOT.rglob("*"):

        if (
            not path.is_file()
            or
            path.suffix.lower()
            not in {".csv", ".parquet"}
        ):
            continue

        try:
            cols = file_columns(path)
        except Exception:
            continue

        keys_ok = all(
            find_col(
                cols,
                aliases,
            ) is not None
            for aliases
            in KEY_ALIASES.values()
        )

        ic_col = find_col(
            cols,
            IC_ALIASES,
        )

        if not keys_ok or ic_col is None:
            continue

        name = path.name.lower()

        score = 0

        if "risk" in name:
            score += 10

        if "idio" in name:
            score += 10

        if "ic" in name:
            score += 5

        if "controlled" in name:
            score += 3

        candidates.append(
            (
                score,
                path,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Cannot locate frozen Day-7 risk IC file."
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            str(x[1]),
        ),
        reverse=True,
    )

    return candidates[0][1]


# ============================================================
# 6. Load IVOL IC
# ============================================================

def load_risk_ic():

    path = discover_risk_ic()

    print(
        f"Risk IC source: {path}"
    )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    original = list(
        df.columns
    )

    rename = {}

    for canonical, aliases in (
        KEY_ALIASES.items()
    ):

        col = find_col(
            original,
            aliases,
        )

        if col is None:
            raise RuntimeError(
                f"Missing column: {canonical}"
            )

        rename[col] = canonical

    ic_col = find_col(
        original,
        IC_ALIASES,
    )

    if ic_col is None:
        raise RuntimeError(
            "Risk IC column not found."
        )

    rename[ic_col] = "risk_ic"

    df = df.rename(
        columns=rename
    )

    # --------------------------------------------------------
    # Require exact idiosyncratic-volatility target whenever
    # the source contains multiple risk outcomes.
    # --------------------------------------------------------

    target_original = find_col(
        original,
        TARGET_ALIASES,
    )

    if target_original is not None:

        target_col = rename.get(
            target_original,
            target_original,
        )

        values = (
            df[target_col]
            .astype(str)
            .str.strip()
        )

        if not (
            values == TARGET
        ).any():

            raise RuntimeError(
                f"Selected source does not contain {TARGET}."
            )

        df = df[
            values == TARGET
        ].copy()

    elif "idio" not in path.name.lower():

        raise RuntimeError(
            "Source has no risk-target column and its "
            "filename does not identify IVOL. "
            "Set RISK_IC_INPUT manually."
        )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(int)

    df["factor_name"] = (
        df["factor_name"]
        .astype(str)
        .str.strip()
    )

    raw_method = (
        df["method"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["method"] = (
        raw_method
        .map(METHOD_MAP)
        .fillna(raw_method)
    )

    df["risk_ic"] = pd.to_numeric(
        df["risk_ic"],
        errors="coerce",
    )

    df = df[
        df["factor_name"].isin(FACTORS)
        &
        df["method"].isin(METHODS)
    ].copy()

    key = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
    ]

    if df[key].duplicated().any():

        raise RuntimeError(
            "Duplicate IVOL IC date-factor-window-method "
            "rows remain after target filtering."
        )

    return (
        df[
            key + ["risk_ic"]
        ],
        path,
        ic_col,
    )


# ============================================================
# 7. Load Step-2 regimes
# ============================================================

def load_regimes():

    qa = pd.read_csv(
        REGIME_QA
    )

    q = dict(
        zip(
            qa["qa_name"],
            qa["qa_value"],
        )
    )

    if not parse_bool(
        q["all_formal_qa_pass"]
    ):
        raise RuntimeError(
            "Step 2 formal QA is not PASS."
        )

    if REGIME_PATH.exists():
        x = pd.read_parquet(
            REGIME_PATH
        )
    else:
        x = pd.read_csv(
            REGIME_CSV
        )

    x["analysis_date"] = pd.to_datetime(
        x["analysis_date"],
        errors="raise",
    )

    x["window"] = pd.to_numeric(
        x["window"],
        errors="raise",
    ).astype(int)

    keep = [
        "analysis_date",
        "window",
        *REGIMES.values(),
    ]

    x = x[keep].copy()

    if x[
        ["analysis_date", "window"]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate regime keys."
        )

    return x


# ============================================================
# 8. Conditional regime estimation
# ============================================================

def fit_group(
    g,
    regime_col,
    regime_type,
    scope,
):

    g = (
        g[
            g[regime_col]
            .isin(REGIME_ORDER)
            &
            g["risk_ic"].notna()
        ]
        .sort_values("analysis_date")
        .copy()
    )

    if len(g) < 6:
        return []

    D = (
        pd.get_dummies(
            g[regime_col],
            dtype=float,
        )
        .reindex(
            columns=REGIME_ORDER,
            fill_value=0.0,
        )
    )

    active = [
        r
        for r in REGIME_ORDER
        if D[r].sum() > 0
    ]

    beta, cov = newey_west_ols(
        g["risk_ic"],
        D[active],
        HAC_LAGS,
    )

    beta = pd.Series(
        beta,
        index=active,
    )

    cov = pd.DataFrame(
        cov,
        index=active,
        columns=active,
    )

    first = g.iloc[0]

    base = {
        "target":
            TARGET,

        "regime_type":
            regime_type,

        "sample_scope":
            scope,

        "factor_name":
            first["factor_name"],

        "method":
            first["method"],

        "window":
            int(first["window"]),

        "total_date_n":
            int(len(g)),
    }

    counts = (
        g[regime_col]
        .value_counts()
    )

    rows = []

    for r in active:

        est = float(
            beta[r]
        )

        se = float(
            np.sqrt(
                max(
                    cov.loc[r, r],
                    0.0,
                )
            )
        )

        n_r = int(
            counts.get(r, 0)
        )

        rows.append(
            {
                **base,

                "estimate_type":
                    "REGIME_MEAN",

                "regime":
                    r,

                "estimate":
                    est,

                "hac_se":
                    se,

                "hac_t":
                    est / se
                    if se > 0
                    else np.nan,

                "regime_date_n":
                    n_r,

                "min_component_n":
                    n_r,

                "small_sample":
                    n_r
                    <
                    SMALL_SAMPLE_N,
            }
        )

    # --------------------------------------------------------
    # Primary test: HIGH - LOW
    # --------------------------------------------------------

    if (
        "HIGH" in active
        and
        "LOW" in active
    ):

        est = float(
            beta["HIGH"]
            -
            beta["LOW"]
        )

        var = (
            cov.loc[
                "HIGH",
                "HIGH",
            ]
            +
            cov.loc[
                "LOW",
                "LOW",
            ]
            -
            2
            *
            cov.loc[
                "HIGH",
                "LOW",
            ]
        )

        se = float(
            np.sqrt(
                max(
                    float(var),
                    0.0,
                )
            )
        )

        low_n = int(
            counts.get(
                "LOW",
                0,
            )
        )

        high_n = int(
            counts.get(
                "HIGH",
                0,
            )
        )

        t = (
            est / se
            if se > 0
            else np.nan
        )

        rows.append(
            {
                **base,

                "estimate_type":
                    "HIGH_MINUS_LOW",

                "regime":
                    "HIGH_MINUS_LOW",

                "estimate":
                    est,

                "hac_se":
                    se,

                "hac_t":
                    t,

                "regime_date_n":
                    low_n + high_n,

                "min_component_n":
                    min(
                        low_n,
                        high_n,
                    ),

                "small_sample":
                    min(
                        low_n,
                        high_n,
                    )
                    <
                    SMALL_SAMPLE_N,
            }
        )

    return rows


# ============================================================
# 9. Run Native + Common-Date samples
# ============================================================

def run_analysis(
    risk,
    regimes,
):

    x = risk.merge(
        regimes,
        on=[
            "analysis_date",
            "window",
        ],
        how="left",
        validate="many_to_one",
    )

    rows = []

    groups = [
        "factor_name",
        "method",
        "window",
    ]

    n_windows = (
        regimes[
            "window"
        ]
        .nunique()
    )

    for regime_type, regime_col in (
        REGIMES.items()
    ):

        c = (
            regimes[
                regimes[regime_col]
                .notna()
            ]
            .groupby(
                "analysis_date"
            )["window"]
            .nunique()
        )

        common_dates = set(
            c[
                c == n_windows
            ].index
        )

        for scope in [
            "NATIVE",
            "COMMON_DATE",
        ]:

            z = x

            if scope == "COMMON_DATE":

                z = z[
                    z["analysis_date"]
                    .isin(common_dates)
                ]

            for _, g in z.groupby(
                groups,
                observed=True,
                sort=False,
            ):

                rows.extend(
                    fit_group(
                        g,
                        regime_col,
                        regime_type,
                        scope,
                    )
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. BH multiple-testing correction
# ============================================================

def add_bh_qvalues(df):

    df["p_nominal"] = np.nan
    df["q_bh"] = np.nan

    mask = (
        df["estimate_type"]
        ==
        "HIGH_MINUS_LOW"
    )

    df.loc[
        mask,
        "p_nominal",
    ] = df.loc[
        mask,
        "hac_t",
    ].map(
        lambda t:
            math.erfc(
                abs(t)
                /
                math.sqrt(2)
            )
        if np.isfinite(t)
        else np.nan
    )

    group_cols = [
        "regime_type",
        "sample_scope",
        "method",
    ]

    for _, idx in (
        df[mask]
        .groupby(
            group_cols
        )
        .groups.items()
    ):

        idx = list(idx)

        p = df.loc[
            idx,
            "p_nominal",
        ]

        valid = p.notna()

        if not valid.any():
            continue

        pv = p[
            valid
        ].to_numpy()

        order = np.argsort(pv)

        ranked = pv[
            order
        ]

        m = len(ranked)

        q = (
            ranked
            *
            m
            /
            np.arange(
                1,
                m + 1,
            )
        )

        q = np.minimum.accumulate(
            q[::-1]
        )[::-1]

        q = np.minimum(
            q,
            1.0,
        )

        back = np.empty_like(q)

        back[
            order
        ] = q

        df.loc[
            p[valid].index,
            "q_bh",
        ] = back

    return df


# ============================================================
# 11. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 9 - Step 4")
    print("Regime-Conditional Idiosyncratic-Risk Validation")
    print("=" * 72)

    regimes = load_regimes()

    risk, source, value_col = (
        load_risk_ic()
    )

    print(
        f"IVOL IC rows: {len(risk):,}"
    )

    result = run_analysis(
        risk,
        regimes,
    )

    result = add_bh_qvalues(
        result
    )

    save_csv(
        result,
        SUMMARY_PATH,
    )

    # Focus table for previously important delta-degree result.
    focus = result[
        result["factor_name"]
        ==
        "delta_degree_percentile"
    ].copy()

    save_csv(
        focus,
        FOCUS_PATH,
    )

    save_csv(
        pd.DataFrame(
            [
                {
                    "target":
                        TARGET,

                    "source_path":
                        str(source),

                    "value_column":
                        value_col,
                }
            ]
        ),
        SOURCE_PATH,
    )

    key = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
    ]

    group_n = (
        risk[
            [
                "factor_name",
                "window",
                "method",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    contrast = result[
        result["estimate_type"]
        ==
        "HIGH_MINUS_LOW"
    ]

    qa = {
        "step2_formal_qa_pass":
            True,

        "target":
            TARGET,

        "frozen_factor_count":
            len(FACTORS),

        "expected_group_count":
            EXPECTED_GROUPS,

        "actual_group_count":
            int(group_n),

        "duplicate_ic_key_count":
            int(
                risk[key]
                .duplicated()
                .sum()
            ),

        "result_row_count":
            int(len(result)),

        "contrast_row_count":
            int(len(contrast)),

        "small_sample_result_count":
            int(
                result[
                    "small_sample"
                ].sum()
            ),

        "minimum_component_n":
            int(
                result[
                    "min_component_n"
                ].min()
            ),

        "factor_sign_flipped":
            False,

        "factor_selected_from_results":
            False,

        "window_selected_from_results":
            False,

        "regime_reestimated":
            False,

        "risk_target_changed":
            False,

        "future_information_used":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["actual_group_count"]
        ==
        EXPECTED_GROUPS
        and
        qa[
            "duplicate_ic_key_count"
        ]
        == 0
        and
        qa["result_row_count"]
        > 0
    )

    save_csv(
        pd.DataFrame(
            [
                {
                    "qa_name": k,
                    "qa_value": v,
                }
                for k, v in qa.items()
            ]
        ),
        QA_PATH,
    )

    metadata = {
        "research_day":
            9,

        "step":
            (
                "Step4_Regime_Conditional_"
                "Idiosyncratic_Risk_Validation"
            ),

        "target":
            TARGET,

        "regime_dimensions":
            REGIMES,

        "primary_statistic":
            "HIGH_MINUS_LOW",

        "hac_lags":
            HAC_LAGS,

        "small_sample_threshold":
            SMALL_SAMPLE_N,

        "multiple_testing":
            (
                "BH correction applied to HIGH-LOW "
                "contrasts within each "
                "regime_type x sample_scope x method."
            ),

        "sample_scopes": [
            "NATIVE",
            "COMMON_DATE",
        ],

        "important_note":
            (
                "No risk model is re-estimated. "
                "Frozen Day-7 monthly IVOL IC is "
                "conditioned on frozen Day-9 PIT regimes."
            ),

        "formal_qa":
            qa,
    }

    with open(
        META_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print()
    print("Formal QA:")

    for k, v in qa.items():
        print(f"  {k}: {v}")

    if not qa[
        "all_formal_qa_pass"
    ]:
        raise RuntimeError(
            "Day 9 Step 4 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 9 STEP 4 COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()

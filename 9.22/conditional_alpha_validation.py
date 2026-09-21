from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
OUTPUT_ROOT = ROOT / "output"

# Research Day 7 
DAY7_ROOT = OUTPUT_ROOT / "M1_day7"

# Research Day 9 
DAY9_ROOT = OUTPUT_ROOT / "M1_day9"

REGIME_PATH = (
    DAY9_ROOT
    / "02_stage2_pit_regime_classification"
    / "pit_regime_panel.parquet"
)

REGIME_CSV = (
    DAY9_ROOT
    / "02_stage2_pit_regime_classification"
    / "pit_regime_panel.csv"
)

STEP2_QA = (
    DAY9_ROOT
    / "02_stage2_pit_regime_classification"
    / "pit_regime_qa.csv"
)

OUTPUT_DIR = (
    DAY9_ROOT
    / "03_stage3_conditional_alpha"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "conditional_alpha_summary.csv"
QA_PATH = OUTPUT_DIR / "conditional_alpha_qa.csv"
SOURCE_PATH = OUTPUT_DIR / "conditional_alpha_source_manifest.csv"
META_PATH = OUTPUT_DIR / "day9_step3_metadata.json"


# ============================================================
# 1. Frozen design
# ============================================================

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

REGIME_COLUMNS = {
    "MARKET":
        "market_stress_regime",

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

# Monthly HAC inference; fixed before seeing Step-3 results.
HAC_LAGS = 3

EXPECTED_GROUPS = (
    len(FACTORS)
    * 3          # W60, W120, W252
    * len(METHODS)
)

# Optional manual override.
# Leave as None for automatic discovery.
IC_INPUT = None
SPREAD_INPUT = None


# ============================================================
# 2. Column aliases for frozen Day-7 outputs
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

VALUE_ALIASES = {
    "IC": [
        "alpha_rank_ic",
        "ic",
        "rank_ic",
        "spearman_ic",
        "alpha_ic",
        "ic_value",
        "future_excess_return_ic",
    ],

    "SPREAD": [
        "q5_minus_q1",
        "q5_q1_spread",
        "spread_q5_minus_q1",
        "spread",
        "spread_bp",
        "spread_bps",
        "spread_bp_month",
    ],
}

TARGET_ALIASES = [
    "target_name",
    "label",
    "label_name",
    "target",
    "outcome",
    "dependent_variable",
]

METHOD_MAP = {
    "RAW": "RAW_COMMON",
    "RAW_COMMON": "RAW_COMMON",

    "INDUSTRY_SIZE": "INDUSTRY_SIZE_COMMON",
    "INDUSTRY_SIZE_COMMON": "INDUSTRY_SIZE_COMMON",

    "FULL": "FULL_CHARACTERISTIC_NEUTRAL",
    "FULL_CONTROLS": "FULL_CHARACTERISTIC_NEUTRAL",
    "FULL_CHARACTERISTIC_NEUTRAL":
        "FULL_CHARACTERISTIC_NEUTRAL",
}


# ============================================================
# 3. Utilities
# ============================================================

def save_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def parse_bool(x):
    return str(x).strip().lower() in {
        "true", "1", "yes"
    }


def file_columns(path):
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)

    return list(
        pq.ParquetFile(path)
        .schema_arrow
        .names
    )


def find_column(columns, aliases):
    lookup = {
        str(c).strip().lower(): c
        for c in columns
    }

    for x in aliases:
        if x.lower() in lookup:
            return lookup[x.lower()]

    return None


# ============================================================
# 4. Automatically locate frozen Day-7 date-level outputs
# ============================================================

def discover_metric_file(kind, override=None):

    if override is not None:
        path = Path(override)

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    candidates = []

    for path in DAY7_ROOT.rglob("*"):

        if (
            not path.is_file()
            or path.suffix.lower()
            not in {".csv", ".parquet"}
        ):
            continue

        try:
            cols = file_columns(path)
        except Exception:
            continue

        keys_ok = all(
            find_column(
                cols,
                aliases,
            ) is not None
            for aliases in KEY_ALIASES.values()
        )

        value_col = find_column(
            cols,
            VALUE_ALIASES[kind],
        )

        if not keys_ok or value_col is None:
            continue

        name = path.name.lower()

        score = 0

        if kind.lower() in name:
            score += 10

        if "alpha" in name:
            score += 4

        if kind == "SPREAD" and "spread" in name:
            score += 8

        if "date" in name or "monthly" in name:
            score += 2

        candidates.append(
            (
                score,
                path,
            )
        )

    if not candidates:
        raise RuntimeError(
            f"Cannot locate Day-7 {kind} date-level file "
            f"under:\n{DAY7_ROOT}"
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            str(x[1]),
        ),
        reverse=True,
    )

    path = candidates[0][1]

    print(
        f"{kind} source: {path}"
    )

    return path


# ============================================================
# 5. Load frozen date-level IC / spread
# ============================================================

def load_metric(kind, override=None):

    path = discover_metric_file(
        kind,
        override,
    )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    original_columns = list(df.columns)

    rename = {}

    for canonical, aliases in KEY_ALIASES.items():

        col = find_column(
            original_columns,
            aliases,
        )

        if col is None:
            raise RuntimeError(
                f"{path} missing {canonical}"
            )

        rename[col] = canonical

    value_original = find_column(
        original_columns,
        VALUE_ALIASES[kind],
    )

    if value_original is None:
        raise RuntimeError(
            f"{path} missing a supported {kind} value column. "
            f"Expected one of: {VALUE_ALIASES[kind]}"
        )

    rename[value_original] = "metric_value"

    df = df.rename(columns=rename)

    # --------------------------------------------------------
    # If the Day-7 file contains multiple outcomes,
    # explicitly retain future_excess_return only.
    # --------------------------------------------------------

    target_col = find_column(
        original_columns,
        TARGET_ALIASES,
    )

    if target_col is not None:

        target_col = rename.get(
            target_col,
            target_col,
        )

        target = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if not (
            target
            ==
            "future_excess_return"
        ).any():
            raise RuntimeError(
                f"{path} contains no future_excess_return target rows."
            )

        df = df[
            target
            ==
            "future_excess_return"
        ].copy()

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

    method_raw = (
        df["method"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["method"] = (
        method_raw
        .map(METHOD_MAP)
        .fillna(method_raw)
    )

    df["metric_value"] = pd.to_numeric(
        df["metric_value"],
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
            f"{kind}: duplicate date-factor-window-method rows. "
            "Check whether the source contains multiple labels."
        )

    df["metric"] = kind

    unit = (
        "bp/month"
        if (
            kind == "SPREAD"
            and "bp" in value_original.lower()
        )
        else "source_native"
    )

    return (
        df[
            key
            +
            [
                "metric",
                "metric_value",
            ]
        ],
        path,
        value_original,
        unit,
    )


# ============================================================
# 6. Load frozen Step-2 regimes
# ============================================================

def load_regimes():

    qa = pd.read_csv(STEP2_QA)

    qa_map = dict(
        zip(
            qa["qa_name"],
            qa["qa_value"],
        )
    )

    if not parse_bool(
        qa_map[
            "all_formal_qa_pass"
        ]
    ):
        raise RuntimeError(
            "Step 2 formal QA is not PASS."
        )

    if REGIME_PATH.exists():
        reg = pd.read_parquet(REGIME_PATH)
    else:
        reg = pd.read_csv(REGIME_CSV)

    reg["analysis_date"] = pd.to_datetime(
        reg["analysis_date"],
        errors="raise",
    )

    reg["window"] = pd.to_numeric(
        reg["window"],
        errors="raise",
    ).astype(int)

    keep = [
        "analysis_date",
        "window",
        *REGIME_COLUMNS.values(),
    ]

    reg = reg[keep].copy()

    if reg[
        ["analysis_date", "window"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Step-2 regime keys."
        )

    return reg


# ============================================================
# 7. HAC regime means
#
# y_t =
#   beta_L * 1{LOW}
# + beta_M * 1{MEDIUM}
# + beta_H * 1{HIGH}
# + error_t
#
# No intercept.
#
# beta_r is exactly the mean conditional on regime r.
# ============================================================

def fit_one_group(
    group,
    regime_col,
    regime_type,
    sample_scope,
):

    g = (
        group[
            group[regime_col]
            .isin(REGIME_ORDER)
            &
            group["metric_value"].notna()
        ]
        .sort_values("analysis_date")
        .copy()
    )

    if len(g) < 6:
        return []

    X = pd.get_dummies(
        g[regime_col],
        dtype=float,
    ).reindex(
        columns=REGIME_ORDER,
        fill_value=0.0,
    )

    active = [
        c for c in REGIME_ORDER
        if X[c].sum() > 0
    ]

    X = X[active]

    fit = sm.OLS(
        g["metric_value"].to_numpy(float),
        X.to_numpy(float),
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags":
                min(
                    HAC_LAGS,
                    len(g) - 1,
                )
        },
    )

    beta = pd.Series(
        fit.params,
        index=active,
    )

    cov = pd.DataFrame(
        fit.cov_params(),
        index=active,
        columns=active,
    )

    first = g.iloc[0]

    base = {
        "metric":
            first["metric"],

        "regime_type":
            regime_type,

        "sample_scope":
            sample_scope,

        "factor_name":
            first["factor_name"],

        "method":
            first["method"],

        "window":
            int(first["window"]),

        "total_date_n":
            int(len(g)),
    }

    rows = []

    for r in REGIME_ORDER:

        if r not in active:
            continue

        se = float(
            np.sqrt(
                max(
                    cov.loc[r, r],
                    0.0,
                )
            )
        )

        estimate = float(
            beta[r]
        )

        rows.append(
            {
                **base,

                "estimate_type":
                    "REGIME_MEAN",

                "regime":
                    r,

                "estimate":
                    estimate,

                "hac_se":
                    se,

                "hac_t":
                    (
                        estimate / se
                        if se > 0
                        else np.nan
                    ),

                "regime_date_n":
                    int(
                        (
                            g[regime_col]
                            == r
                        ).sum()
                    ),
            }
        )

    # --------------------------------------------------------
    # Main regime contrast: HIGH - LOW
    # --------------------------------------------------------

    if (
        "HIGH" in active
        and
        "LOW" in active
    ):

        contrast = float(
            beta["HIGH"]
            -
            beta["LOW"]
        )

        var = (
            cov.loc["HIGH", "HIGH"]
            +
            cov.loc["LOW", "LOW"]
            -
            2.0
            *
            cov.loc["HIGH", "LOW"]
        )

        se = float(
            np.sqrt(
                max(
                    var,
                    0.0,
                )
            )
        )

        rows.append(
            {
                **base,

                "estimate_type":
                    "HIGH_MINUS_LOW",

                "regime":
                    "HIGH_MINUS_LOW",

                "estimate":
                    contrast,

                "hac_se":
                    se,

                "hac_t":
                    (
                        contrast / se
                        if se > 0
                        else np.nan
                    ),

                "regime_date_n":
                    int(
                        (
                            g[regime_col]
                            .isin(
                                [
                                    "LOW",
                                    "HIGH",
                                ]
                            )
                        ).sum()
                    ),
            }
        )

    return rows


# ============================================================
# 8. Native + common-date conditional validation
# ============================================================

def run_conditional(
    data,
    regimes,
):

    data = data.merge(
        regimes,
        on=[
            "analysis_date",
            "window",
        ],
        how="left",
        validate="many_to_one",
    )

    rows = []

    group_cols = [
        "metric",
        "factor_name",
        "method",
        "window",
    ]

    n_windows = regimes[
        "window"
    ].nunique()

    for regime_type, regime_col in (
        REGIME_COLUMNS.items()
    ):

        # Dates classified for every W.
        tmp = (
            regimes[
                regimes[regime_col]
                .notna()
            ]
            .groupby("analysis_date")[
                "window"
            ]
            .nunique()
        )

        common_dates = set(
            tmp[
                tmp == n_windows
            ].index
        )

        for scope in [
            "NATIVE",
            "COMMON_DATE",
        ]:

            x = data

            if scope == "COMMON_DATE":
                x = x[
                    x["analysis_date"]
                    .isin(common_dates)
                ]

            for _, g in x.groupby(
                group_cols,
                observed=True,
                sort=False,
            ):

                rows.extend(
                    fit_one_group(
                        g,
                        regime_col,
                        regime_type,
                        scope,
                    )
                )

    return pd.DataFrame(rows)


# ============================================================
# 9. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 9 - Step 3")
    print("Conditional Alpha Validation")
    print("=" * 72)

    regimes = load_regimes()

    ic, ic_path, ic_col, ic_unit = (
        load_metric(
            "IC",
            IC_INPUT,
        )
    )

    spread, spread_path, spread_col, spread_unit = (
        load_metric(
            "SPREAD",
            SPREAD_INPUT,
        )
    )

    print(
        f"IC rows     : {len(ic):,}"
    )

    print(
        f"Spread rows : {len(spread):,}"
    )

    data = pd.concat(
        [
            ic,
            spread,
        ],
        ignore_index=True,
    )

    result = run_conditional(
        data,
        regimes,
    )

    save_csv(
        result,
        SUMMARY_PATH,
    )

    # --------------------------------------------------------
    # Source manifest
    # --------------------------------------------------------

    sources = pd.DataFrame(
        [
            {
                "metric": "IC",
                "path": str(ic_path),
                "value_column": ic_col,
                "unit": ic_unit,
            },
            {
                "metric": "SPREAD",
                "path": str(spread_path),
                "value_column": spread_col,
                "unit": spread_unit,
            },
        ]
    )

    save_csv(
        sources,
        SOURCE_PATH,
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    key = [
        "analysis_date",
        "window",
        "factor_name",
        "method",
    ]

    ic_groups = int(
        ic[
            [
                "factor_name",
                "window",
                "method",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    spread_groups = int(
        spread[
            [
                "factor_name",
                "window",
                "method",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    qa = {
        "step2_formal_qa_pass":
            True,

        "frozen_factor_count":
            len(FACTORS),

        "expected_groups_per_metric":
            EXPECTED_GROUPS,

        "ic_group_count":
            ic_groups,

        "spread_group_count":
            spread_groups,

        "ic_duplicate_key_count":
            int(
                ic[key]
                .duplicated()
                .sum()
            ),

        "spread_duplicate_key_count":
            int(
                spread[key]
                .duplicated()
                .sum()
            ),

        "result_row_count":
            int(len(result)),

        "native_result_row_count":
            int(
                (
                    result[
                        "sample_scope"
                    ]
                    ==
                    "NATIVE"
                ).sum()
            ),

        "common_date_result_row_count":
            int(
                (
                    result[
                        "sample_scope"
                    ]
                    ==
                    "COMMON_DATE"
                ).sum()
            ),

        "minimum_regime_date_n":
            int(
                result.loc[
                    result[
                        "estimate_type"
                    ]
                    ==
                    "REGIME_MEAN",
                    "regime_date_n",
                ]
                .min()
            ),

        "factor_sign_flipped":
            False,

        "factor_selected_from_regime_results":
            False,

        "window_selected_from_regime_results":
            False,

        "regime_reestimated":
            False,

        "future_information_used":
            False,

        "alpha_definition_changed":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["ic_group_count"]
        ==
        EXPECTED_GROUPS
        and
        qa["spread_group_count"]
        ==
        EXPECTED_GROUPS
        and
        qa["ic_duplicate_key_count"]
        == 0
        and
        qa["spread_duplicate_key_count"]
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
            "Step3_Conditional_Alpha_Validation",

        "alpha_target":
            "future_excess_return",

        "frozen_factors":
            FACTORS,

        "frozen_methods":
            METHODS,

        "regime_dimensions":
            REGIME_COLUMNS,

        "primary_comparison":
            "HIGH_MINUS_LOW",

        "hac_lags":
            HAC_LAGS,

        "sample_scopes": [
            "NATIVE",
            "COMMON_DATE",
        ],

        "important_note":
            (
                "Step 3 conditions frozen Day-7 "
                "date-level Alpha IC and portfolio-spread "
                "statistics on frozen Day-9 Step-2 PIT "
                "regimes. No stock-level Alpha model is "
                "re-estimated."
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
            "Day 9 Step 3 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 9 STEP 3 COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()

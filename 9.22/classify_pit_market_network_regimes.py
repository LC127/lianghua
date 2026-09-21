from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
DAY9_ROOT = ROOT / "output" / "M1_day9"

STEP1_DIR = (
    DAY9_ROOT
    / "01_stage1_market_network_state"
)

INPUT_PARQUET = (
    STEP1_DIR
    / "market_network_state_panel.parquet"
)

INPUT_CSV = (
    STEP1_DIR
    / "market_network_state_panel.csv"
)

OUTPUT_DIR = (
    DAY9_ROOT
    / "02_stage2_pit_regime_classification"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PANEL_PATH = (
    OUTPUT_DIR
    / "pit_regime_panel.parquet"
)

PANEL_CSV_PATH = (
    OUTPUT_DIR
    / "pit_regime_panel.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "pit_regime_summary.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "pit_regime_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day9_step2_metadata.json"
)


# ============================================================
# 1. Frozen regime design
# ============================================================

# Need 24 prior monthly observations before a component
# receives a PIT z-score.
MIN_Z_HISTORY = 24

# Need at least 12 prior valid scores before Low/Medium/High
# thresholds are estimated.
MIN_SCORE_HISTORY = 12

LOW_Q = 1.0 / 3.0
HIGH_Q = 2.0 / 3.0


# ------------------------------------------------------------
# Direction:
#
# +1 : larger value -> higher stress
# -1 : smaller value -> higher stress
# ------------------------------------------------------------

MARKET_COMPONENTS = {
    "market_vol_20d_ann": +1,
    "market_dispersion_20d": +1,
    "market_return_20d": -1,
    "market_breadth_20d": -1,
}

NETWORK_COMPONENTS = {
    "market_r2_mean": +1,
    "residual_edge_threshold": +1,
    "residual_cross_industry_edge_share": +1,
    "raw_residual_overlap_fraction": -1,
}

RECONFIG_COMPONENTS = {
    "abs_delta_market_r2_mean_1m": +1,
    "abs_delta_residual_edge_threshold_1m": +1,
    "abs_delta_residual_cross_industry_edge_share_1m": +1,
    "abs_delta_raw_residual_overlap_fraction_1m": +1,
}


# ============================================================
# 2. Small utilities
# ============================================================

def save_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def save_parquet(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_parquet(
        tmp,
        index=False,
        compression="zstd",
    )
    os.replace(tmp, path)


def load_panel():

    if INPUT_PARQUET.exists():
        df = pd.read_parquet(INPUT_PARQUET)

    elif INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV)

    else:
        raise FileNotFoundError(
            "Step-1 state panel not found."
        )

    df["analysis_date"] = pd.to_datetime(
        df["analysis_date"],
        errors="raise",
    )

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(int)

    key = ["analysis_date", "window"]

    if df[key].duplicated().any():
        raise RuntimeError(
            "Duplicate analysis_date × window rows."
        )

    return df.sort_values(key).reset_index(drop=True)


# ============================================================
# 3. PIT score + expanding PIT regime
# ============================================================

def classify_one_series(
    df,
    components,
    prefix,
):

    """
    For each date t:

    1. estimate mean / sd using dates strictly before t;
    2. standardize current state X_t;
    3. average signed component z-scores;
    4. classify current score using q33/q67 of PRIOR scores.

    Therefore current/future observations never enter
    historical means, standard deviations or thresholds.
    """

    x = df.sort_values(
        "analysis_date"
    ).copy()

    zcols = []

    for col, direction in components.items():

        if col not in x.columns:
            raise RuntimeError(
                f"Missing state variable: {col}"
            )

        values = pd.to_numeric(
            x[col],
            errors="coerce",
        )

        history = values.shift(1)

        mean = history.expanding(
            min_periods=MIN_Z_HISTORY
        ).mean()

        std = history.expanding(
            min_periods=MIN_Z_HISTORY
        ).std(ddof=1)

        zcol = f"{prefix}_z_{col}"

        x[zcol] = (
            direction
            *
            (values - mean)
            /
            std
        ).where(
            std > 0
        )

        zcols.append(zcol)

    # All components must be available.
    score_col = f"{prefix}_score"

    x[score_col] = (
        x[zcols]
        .mean(
            axis=1,
            skipna=False,
        )
    )

    # --------------------------------------------------------
    # Regime thresholds use only previously available scores.
    # --------------------------------------------------------

    score_history = (
        x[score_col]
        .shift(1)
    )

    history_n_col = (
        f"{prefix}_score_history_n"
    )

    qlow_col = (
        f"{prefix}_q33_pit"
    )

    qhigh_col = (
        f"{prefix}_q67_pit"
    )

    x[history_n_col] = (
        score_history
        .expanding()
        .count()
    )

    x[qlow_col] = (
        score_history
        .expanding(
            min_periods=MIN_SCORE_HISTORY
        )
        .quantile(
            LOW_Q
        )
    )

    x[qhigh_col] = (
        score_history
        .expanding(
            min_periods=MIN_SCORE_HISTORY
        )
        .quantile(
            HIGH_Q
        )
    )

    valid = (
        x[score_col].notna()
        &
        x[qlow_col].notna()
        &
        x[qhigh_col].notna()
    )

    regime = pd.Series(
        pd.NA,
        index=x.index,
        dtype="string",
    )

    regime.loc[
        valid
        &
        (
            x[score_col]
            <=
            x[qlow_col]
        )
    ] = "LOW"

    regime.loc[
        valid
        &
        (
            x[score_col]
            >
            x[qlow_col]
        )
        &
        (
            x[score_col]
            <=
            x[qhigh_col]
        )
    ] = "MEDIUM"

    regime.loc[
        valid
        &
        (
            x[score_col]
            >
            x[qhigh_col]
        )
    ] = "HIGH"

    x[f"{prefix}_regime"] = regime

    return x


# ============================================================
# 4. Market regime
#
# Market state is M_t, not M_{t,W}.
# Therefore classify once per date and merge back.
# ============================================================

def build_market_regime(panel):

    market_cols = list(
        MARKET_COMPONENTS.keys()
    )

    # Check market state is identical across W.
    variation = (
        panel.groupby("analysis_date")[
            market_cols
        ]
        .nunique(
            dropna=False
        )
    )

    if (variation > 1).any().any():
        raise RuntimeError(
            "Market state differs across network windows "
            "on the same analysis date."
        )

    market = (
        panel[
            ["analysis_date"] + market_cols
        ]
        .drop_duplicates("analysis_date")
        .sort_values("analysis_date")
        .reset_index(drop=True)
    )

    market = classify_one_series(
        market,
        MARKET_COMPONENTS,
        "market_stress",
    )

    keep = [
        c
        for c in market.columns
        if (
            c == "analysis_date"
            or
            c.startswith("market_stress_")
        )
    ]

    return market[keep]


# ============================================================
# 5. Network regimes
#
# Network levels differ mechanically across W, so all
# standardization and regime thresholds are estimated
# separately within each window.
# ============================================================

def classify_within_window(
    panel,
    components,
    prefix,
):

    parts = []

    for _, group in panel.groupby(
        "window",
        sort=True,
    ):

        parts.append(
            classify_one_series(
                group,
                components,
                prefix,
            )
        )

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "analysis_date",
                "window",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 6. Summary
# ============================================================

def build_summary(panel):

    rows = []

    regime_cols = [
        "market_stress_regime",
        "network_stress_regime",
        "network_reconfig_regime",
    ]

    for col in regime_cols:

        for window, g in panel.groupby(
            "window"
        ):

            counts = (
                g[col]
                .value_counts(
                    dropna=False
                )
            )

            for regime in [
                "LOW",
                "MEDIUM",
                "HIGH",
            ]:

                rows.append(
                    {
                        "regime_type":
                            col,

                        "window":
                            window,

                        "regime":
                            regime,

                        "count":
                            int(
                                counts.get(
                                    regime,
                                    0,
                                )
                            ),
                    }
                )

            rows.append(
                {
                    "regime_type":
                        col,

                    "window":
                        window,

                    "regime":
                        "UNCLASSIFIED",

                    "count":
                        int(
                            g[col]
                            .isna()
                            .sum()
                        ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 7. QA
# ============================================================

def run_qa(panel, input_n):

    regime_cols = [
        "market_stress_regime",
        "network_stress_regime",
        "network_reconfig_regime",
    ]

    valid_labels = {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    invalid_label_n = sum(
        (
            ~panel[col]
            .dropna()
            .isin(valid_labels)
        ).sum()
        for col in regime_cols
    )

    early_classification_n = 0
    threshold_order_violation_n = 0

    for prefix in [
        "market_stress",
        "network_stress",
        "network_reconfig",
    ]:

        regime = panel[
            f"{prefix}_regime"
        ]

        history_n = panel[
            f"{prefix}_score_history_n"
        ]

        q33 = panel[
            f"{prefix}_q33_pit"
        ]

        q67 = panel[
            f"{prefix}_q67_pit"
        ]

        early_classification_n += int(
            (
                regime.notna()
                &
                (
                    history_n
                    <
                    MIN_SCORE_HISTORY
                )
            ).sum()
        )

        threshold_order_violation_n += int(
            (
                q33.notna()
                &
                q67.notna()
                &
                (
                    q33 > q67
                )
            ).sum()
        )

    # Market regime must be identical across W on same date.
    market_regime_window_mismatch_n = int(
        (
            panel.groupby("analysis_date")[
                "market_stress_regime"
            ]
            .nunique(
                dropna=False
            )
            >
            1
        ).sum()
    )

    qa = {
        "input_row_count":
            int(input_n),

        "output_row_count":
            int(len(panel)),

        "row_count_preserved":
            bool(
                len(panel)
                ==
                input_n
            ),

        "duplicate_date_window_count":
            int(
                panel[
                    [
                        "analysis_date",
                        "window",
                    ]
                ]
                .duplicated()
                .sum()
            ),

        "invalid_regime_label_count":
            int(invalid_label_n),

        "classification_before_min_history_count":
            int(
                early_classification_n
            ),

        "threshold_order_violation_count":
            int(
                threshold_order_violation_n
            ),

        "market_regime_window_mismatch_date_count":
            int(
                market_regime_window_mismatch_n
            ),

        "market_classified_row_count":
            int(
                panel[
                    "market_stress_regime"
                ]
                .notna()
                .sum()
            ),

        "network_classified_row_count":
            int(
                panel[
                    "network_stress_regime"
                ]
                .notna()
                .sum()
            ),

        "reconfiguration_classified_row_count":
            int(
                panel[
                    "network_reconfig_regime"
                ]
                .notna()
                .sum()
            ),

        "future_information_used":
            False,

        "full_sample_quantiles_used":
            False,

        "current_observation_used_in_threshold":
            False,

        "outcome_data_used":
            False,

        "factor_performance_used":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["row_count_preserved"]
        and
        qa[
            "duplicate_date_window_count"
        ] == 0
        and
        qa[
            "invalid_regime_label_count"
        ] == 0
        and
        qa[
            "classification_before_min_history_count"
        ] == 0
        and
        qa[
            "threshold_order_violation_count"
        ] == 0
        and
        qa[
            "market_regime_window_mismatch_date_count"
        ] == 0
        and
        qa[
            "market_classified_row_count"
        ] > 0
        and
        qa[
            "network_classified_row_count"
        ] > 0
        and
        qa[
            "reconfiguration_classified_row_count"
        ] > 0
    )

    return qa


# ============================================================
# 8. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 9 - Step 2")
    print("PIT Regime Classification")
    print("=" * 72)

    panel = load_panel()
    input_n = len(panel)

    print(
        f"Input rows: {input_n:,}"
    )

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------

    print("[1] Market stress regime")

    market = build_market_regime(
        panel
    )

    panel = panel.merge(
        market,
        on="analysis_date",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Network level regime
    # --------------------------------------------------------

    print("[2] Network stress regime")

    panel = classify_within_window(
        panel,
        NETWORK_COMPONENTS,
        "network_stress",
    )

    # --------------------------------------------------------
    # Network reconfiguration regime
    # --------------------------------------------------------

    print("[3] Network reconfiguration regime")

    panel = classify_within_window(
        panel,
        RECONFIG_COMPONENTS,
        "network_reconfig",
    )

    # --------------------------------------------------------
    # Optional joint descriptive state
    # --------------------------------------------------------

    valid_joint = (
        panel[
            "market_stress_regime"
        ].notna()
        &
        panel[
            "network_stress_regime"
        ].notna()
    )

    panel[
        "market_network_joint_regime"
    ] = pd.NA

    panel.loc[
        valid_joint,
        "market_network_joint_regime",
    ] = (
        panel.loc[
            valid_joint,
            "market_stress_regime",
        ]
        +
        "_MARKET__"
        +
        panel.loc[
            valid_joint,
            "network_stress_regime",
        ]
        +
        "_NETWORK"
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    qa = run_qa(
        panel,
        input_n,
    )

    summary = build_summary(
        panel
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_parquet(
        panel,
        PANEL_PATH,
    )

    save_csv(
        panel,
        PANEL_CSV_PATH,
    )

    save_csv(
        summary,
        SUMMARY_PATH,
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
            "Step2_PIT_Regime_Classification",

        "min_z_history":
            MIN_Z_HISTORY,

        "min_score_history":
            MIN_SCORE_HISTORY,

        "regime_quantiles":
            [
                LOW_Q,
                HIGH_Q,
            ],

        "market_components":
            MARKET_COMPONENTS,

        "network_components":
            NETWORK_COMPONENTS,

        "reconfiguration_components":
            RECONFIG_COMPONENTS,

        "market_regime_scope":
            (
                "One regime per analysis date; "
                "independent of network window."
            ),

        "network_regime_scope":
            (
                "Standardization and PIT thresholds "
                "estimated separately within each "
                "network window."
            ),

        "standardization":
            (
                "Current value standardized using "
                "mean and standard deviation of "
                "strictly prior observations only."
            ),

        "classification":
            (
                "LOW/MEDIUM/HIGH based on expanding "
                "33.3%/66.7% quantiles of strictly "
                "prior valid scores."
            ),

        "future_information_used":
            False,

        "outcome_information_used":
            False,

        "formal_qa":
            qa,
    }

    with open(
        METADATA_PATH,
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
            "Day 9 Step 2 QA failed."
        )

    print()
    print("=" * 72)
    print("DAY 9 STEP 2 COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
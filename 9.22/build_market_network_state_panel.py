from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. Paths
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
OUTPUT_ROOT = ROOT / "output"

# ------------------------------------------------------------
# Day 4 frozen network outputs
#
# Current frozen project convention:
# Research Day 4 -> physical M1_day4
# ------------------------------------------------------------

DAY4_NETWORK_DESIGN_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "01_step1_network_design"
)

DAY4_NETWORK_DIR = (
    OUTPUT_ROOT
    / "M1_day4"
    / "02_step2_matched_networks"
)

NETWORK_CONFIG_PATH = (
    DAY4_NETWORK_DESIGN_DIR
    / "day4_network_design_config.json"
)

NETWORK_SUMMARY_PATH = (
    DAY4_NETWORK_DIR
    / "network_construction_summary.csv"
)

DENSITY_SUMMARY_PATH = (
    DAY4_NETWORK_DIR
    / "matched_density_network_summary.csv"
)


# ------------------------------------------------------------
# Day 1 frozen return panel
# ------------------------------------------------------------

RETURN_PANEL_PATH = (
    OUTPUT_ROOT
    / "M1_day1"
    / "03_stage3_return_validation"
    / "daily_return_panel.parquet"
)


# ------------------------------------------------------------
# Day 9 Step 1 outputs
# ------------------------------------------------------------

OUTPUT_DIR = (
    OUTPUT_ROOT
    / "M1_day9"
    / "01_stage1_market_network_state"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PANEL_PATH = (
    OUTPUT_DIR
    / "market_network_state_panel.parquet"
)

PANEL_CSV_PATH = (
    OUTPUT_DIR
    / "market_network_state_panel.csv"
)

DAILY_MARKET_PATH = (
    OUTPUT_DIR
    / "daily_market_state.csv"
)

QA_PATH = (
    OUTPUT_DIR
    / "market_network_state_qa.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "day9_step1_metadata.json"
)


# ============================================================
# 1. Fixed design
# ============================================================

MARKET_WINDOWS = [20, 60]

PARQUET_BATCH_SIZE = 500_000

CORE_NETWORK_STATE_COLUMNS = [
    "market_r2_mean",
    "residual_edge_threshold",
    "residual_cross_industry_edge_share",
    "residual_industry_enrichment",
    "raw_residual_overlap_fraction",
    "residual_positive_pair_share",
]


# ============================================================
# 2. Small utilities
# ============================================================

def save_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def load_primary_density() -> float:

    if not NETWORK_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Network config not found:\n{NETWORK_CONFIG_PATH}"
        )

    with open(
        NETWORK_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    return float(
        config[
            "primary_network_comparison"
        ][
            "primary_density"
        ]
    )


# ============================================================
# 3. Build daily market state
#
# Uses only information available on or before each date.
# ============================================================

def build_daily_market_state() -> pd.DataFrame:

    print("[1] Build daily market state")

    if not RETURN_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Return panel not found:\n{RETURN_PANEL_PATH}"
        )

    parquet = pq.ParquetFile(
        RETURN_PANEL_PATH
    )

    schema = parquet.schema_arrow.names

    # --------------------------------------------------------
    # Prefer simple network return.
    # If only log network return exists, convert with expm1.
    # --------------------------------------------------------

    if "return_network_simple" in schema:
        return_col = "return_network_simple"
        log_return = False

    elif "return_network" in schema:
        return_col = "return_network"
        log_return = True

    else:
        raise RuntimeError(
            "Neither return_network_simple nor "
            "return_network exists."
        )

    partials = []

    for batch in parquet.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=[
            "trade_date",
            return_col,
        ],
    ):

        x = batch.to_pandas()

        x["trade_date"] = pd.to_datetime(
            x["trade_date"],
            errors="coerce",
        )

        r = pd.to_numeric(
            x[return_col],
            errors="coerce",
        )

        if log_return:
            r = np.expm1(r)

        x["r"] = r

        x = x[
            x["trade_date"].notna()
            &
            x["r"].notna()
            &
            np.isfinite(x["r"])
        ].copy()

        if x.empty:
            continue

        x["r2"] = x["r"] ** 2
        x["positive"] = (x["r"] > 0).astype(np.int64)

        part = (
            x.groupby(
                "trade_date",
                as_index=False,
            )
            .agg(
                valid_stock_n=("r", "size"),
                sum_r=("r", "sum"),
                sum_r2=("r2", "sum"),
                positive_n=("positive", "sum"),
            )
        )

        partials.append(part)

    if not partials:
        raise RuntimeError(
            "No valid return observations found."
        )

    daily = (
        pd.concat(
            partials,
            ignore_index=True,
        )
        .groupby(
            "trade_date",
            as_index=False,
        )
        .agg(
            valid_stock_n=("valid_stock_n", "sum"),
            sum_r=("sum_r", "sum"),
            sum_r2=("sum_r2", "sum"),
            positive_n=("positive_n", "sum"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    n = daily["valid_stock_n"].astype(float)

    daily["market_ew_return_1d"] = (
        daily["sum_r"] / n
    )

    daily["market_breadth_1d"] = (
        daily["positive_n"] / n
    )

    var = (
        daily["sum_r2"]
        -
        daily["sum_r"] ** 2 / n
    ) / (n - 1)

    daily["cross_sectional_dispersion_1d"] = np.sqrt(
        np.maximum(
            var,
            0.0,
        )
    )

    # --------------------------------------------------------
    # PIT trailing market state.
    #
    # Window ends at current market date t.
    # No future observations.
    # --------------------------------------------------------

    for w in MARKET_WINDOWS:

        daily[f"market_return_{w}d"] = (
            (1.0 + daily["market_ew_return_1d"])
            .rolling(
                w,
                min_periods=w,
            )
            .apply(
                np.prod,
                raw=True,
            )
            - 1.0
        )

        daily[f"market_vol_{w}d_ann"] = (
            daily["market_ew_return_1d"]
            .rolling(
                w,
                min_periods=w,
            )
            .std()
            *
            np.sqrt(252.0)
        )

        daily[f"market_dispersion_{w}d"] = (
            daily["cross_sectional_dispersion_1d"]
            .rolling(
                w,
                min_periods=w,
            )
            .mean()
        )

        daily[f"market_breadth_{w}d"] = (
            daily["market_breadth_1d"]
            .rolling(
                w,
                min_periods=w,
            )
            .mean()
        )

    return daily


# ============================================================
# 4. Load primary-density network state
# ============================================================

def build_network_state(
    primary_q: float,
) -> pd.DataFrame:

    print("[2] Load frozen network summaries")

    for path in [
        NETWORK_SUMMARY_PATH,
        DENSITY_SUMMARY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen network output:\n{path}"
            )

    jobs = pd.read_csv(
        NETWORK_SUMMARY_PATH
    )

    density = pd.read_csv(
        DENSITY_SUMMARY_PATH
    )

    for df in [jobs, density]:

        df["analysis_date"] = pd.to_datetime(
            df["analysis_date"],
            errors="raise",
        )

        df["window"] = pd.to_numeric(
            df["window"],
            errors="raise",
        ).astype(int)

    density["density_fraction"] = pd.to_numeric(
        density["density_fraction"],
        errors="raise",
    )

    density = density[
        np.isclose(
            density["density_fraction"],
            primary_q,
            rtol=0,
            atol=1e-12,
        )
    ].copy()

    if density.empty:
        raise RuntimeError(
            f"No rows found for primary density q={primary_q}."
        )

    key = [
        "analysis_date",
        "window",
    ]

    if jobs[key].duplicated().any():
        raise RuntimeError(
            "Duplicate date-window rows in network summary."
        )

    if density[key].duplicated().any():
        raise RuntimeError(
            "Duplicate date-window rows at primary density."
        )

    # --------------------------------------------------------
    # Build interpretable network-state variables.
    #
    # Density itself is fixed by construction, therefore it is
    # NOT used as a time-varying state variable.
    # --------------------------------------------------------

    density[
        "residual_cross_industry_edge_share"
    ] = (
        1.0
        -
        density[
            "residual_same_industry_edge_share"
        ]
    )

    density[
        "residual_positive_pair_share"
    ] = (
        density[
            "residual_positive_pair_count"
        ]
        /
        density[
            "common_valid_pair_count"
        ]
    )

    density[
        "raw_positive_pair_share"
    ] = (
        density[
            "raw_positive_pair_count"
        ]
        /
        density[
            "common_valid_pair_count"
        ]
    )

    keep_density = [
        "analysis_date",
        "window",
        "density_fraction",
        "common_node_count",
        "common_valid_pair_count",
        "raw_edge_threshold",
        "residual_edge_threshold",
        "raw_residual_overlap_fraction",
        "raw_residual_jaccard",
        "raw_same_industry_edge_share",
        "residual_same_industry_edge_share",
        "residual_cross_industry_edge_share",
        "raw_industry_enrichment",
        "residual_industry_enrichment",
        "raw_positive_pair_share",
        "residual_positive_pair_share",
    ]

    keep_jobs = [
        "analysis_date",
        "window",
        "common_valid_pair_ratio",
        "market_beta_mean",
        "market_beta_median",
        "market_r2_mean",
        "market_r2_median",
    ]

    network = density[
        keep_density
    ].merge(
        jobs[
            keep_jobs
        ],
        on=key,
        how="left",
        validate="one_to_one",
    )

    network = network.sort_values(
        [
            "window",
            "analysis_date",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # One-rebalance changes.
    #
    # Analysis dates are monthly formation dates, therefore
    # these represent approximately one-month changes.
    # --------------------------------------------------------

    change_vars = [
        "market_r2_mean",
        "residual_edge_threshold",
        "residual_cross_industry_edge_share",
        "raw_residual_overlap_fraction",
    ]

    for col in change_vars:

        dcol = f"delta_{col}_1m"

        network[dcol] = (
            network.groupby(
                "window",
                observed=True,
            )[col]
            .diff()
        )

        network[f"abs_{dcol}"] = (
            network[dcol].abs()
        )

    return network


# ============================================================
# 5. Merge market and network state
# ============================================================

def build_state_panel(
    daily: pd.DataFrame,
    network: pd.DataFrame,
) -> pd.DataFrame:

    print("[3] Merge market + network state")

    market_columns = [
        "trade_date",
        "valid_stock_n",
        "market_ew_return_1d",
        "market_breadth_1d",
        "cross_sectional_dispersion_1d",
        "market_return_20d",
        "market_vol_20d_ann",
        "market_dispersion_20d",
        "market_breadth_20d",
        "market_return_60d",
        "market_vol_60d_ann",
        "market_dispersion_60d",
        "market_breadth_60d",
    ]

    panel = network.merge(
        daily[
            market_columns
        ].rename(
            columns={
                "trade_date":
                    "analysis_date"
            }
        ),
        on="analysis_date",
        how="left",
        validate="many_to_one",
    )

    panel = panel.sort_values(
        [
            "analysis_date",
            "window",
        ]
    ).reset_index(drop=True)

    return panel


# ============================================================
# 6. Formal QA
# ============================================================

def run_qa(
    panel: pd.DataFrame,
    network: pd.DataFrame,
    primary_q: float,
) -> dict:

    key = [
        "analysis_date",
        "window",
    ]

    market_required = [
        "market_return_20d",
        "market_vol_20d_ann",
        "market_dispersion_20d",
        "market_breadth_20d",
        "market_return_60d",
        "market_vol_60d_ann",
        "market_dispersion_60d",
        "market_breadth_60d",
    ]

    network_missing = int(
        panel[
            CORE_NETWORK_STATE_COLUMNS
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    market_missing = int(
        panel[
            market_required
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    qa = {
        "primary_density":
            primary_q,

        "panel_row_count":
            int(len(panel)),

        "network_row_count":
            int(len(network)),

        "unique_analysis_dates":
            int(
                panel["analysis_date"].nunique()
            ),

        "window_count":
            int(
                panel["window"].nunique()
            ),

        "duplicate_date_window_count":
            int(
                panel[key]
                .duplicated()
                .sum()
            ),

        "core_network_state_missing_row_count":
            network_missing,

        "market_state_missing_row_count":
            market_missing,

        "density_is_primary_for_all_rows":
            bool(
                np.isclose(
                    panel["density_fraction"],
                    primary_q,
                    rtol=0,
                    atol=1e-12,
                ).all()
            ),

        "future_market_data_used":
            False,

        "future_network_data_used":
            False,

        "market_windows_end_at_analysis_date":
            True,

        "regime_classification_performed":
            False,

        "network_stress_score_constructed":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["duplicate_date_window_count"] == 0
        and
        qa[
            "core_network_state_missing_row_count"
        ] == 0
        and
        qa[
            "market_state_missing_row_count"
        ] == 0
        and
        qa[
            "density_is_primary_for_all_rows"
        ]
    )

    return qa


# ============================================================
# 7. Main
# ============================================================

def main():

    print("=" * 72)
    print("M1 - Research Day 9 - Step 1")
    print("Market / Network State Panel")
    print("=" * 72)

    primary_q = load_primary_density()

    print(
        f"Primary network density q = {primary_q}"
    )

    daily = build_daily_market_state()

    network = build_network_state(
        primary_q
    )

    panel = build_state_panel(
        daily,
        network,
    )

    qa = run_qa(
        panel,
        network,
        primary_q,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_csv(
        daily,
        DAILY_MARKET_PATH,
    )

    save_parquet(
        panel,
        PANEL_PATH,
    )

    save_csv(
        panel,
        PANEL_CSV_PATH,
    )

    qa_df = pd.DataFrame(
        [
            {
                "qa_name": k,
                "qa_value": v,
            }
            for k, v in qa.items()
        ]
    )

    save_csv(
        qa_df,
        QA_PATH,
    )

    metadata = {
        "research_day":
            9,

        "step":
            "Step1_Market_Network_State_Panel",

        "primary_density":
            primary_q,

        "market_return_source":
            (
                "return_network_simple; "
                "fallback exp(return_network)-1"
            ),

        "market_windows":
            MARKET_WINDOWS,

        "market_state_definition":
            (
                "Trailing PIT market statistics ending "
                "at the current analysis date."
            ),

        "network_state_definition":
            (
                "Frozen Day-4 primary-density residual "
                "network summaries; density itself is not "
                "treated as a state variable because it is "
                "fixed by construction."
            ),

        "regime_classification":
            "NOT_PERFORMED_IN_STEP1",

        "future_information_used":
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

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            "Day 9 Step 1 QA failed. "
            "Do not proceed to regime classification."
        )

    print()
    print("=" * 72)
    print("DAY 9 STEP 1 COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
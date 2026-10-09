from pathlib import Path
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# Day 13 - Step 4
# Frozen Network Risk-State Construction
# Semantic-fixed version
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")

PRED = (
    ROOT / "output" / "M1_day11"
    / "03_stage3_strict_oos_forecasting"
    / "oos_predictions.parquet"
)

STEP3_QA = (
    ROOT / "output" / "M1_day13"
    / "03_step3_risk_alpha_separation_summary"
    / "day13_step3_qa.csv"
)

OUT = (
    ROOT / "output" / "M1_day13"
    / "04_step4_frozen_network_risk_state"
)

WINDOWS = [60, 120, 252]

# Frozen Day-11 strict-OOS score columns.
# These are risk-model scores, NOT raw future-IVOL units.
BASELINE_COL = "baseline_score"
NETWORK_COL = "network_score"

OUTPUT_NAMES = (
    "frozen_network_risk_state_by_window.csv",
    "frozen_network_risk_state.csv",
    "day13_step4_qa.csv",
    "day13_step4_metadata.json",
)


# ============================================================
# Helpers
# ============================================================

def output_permission_error(path, pending=None):
    message = (
        f"Cannot update output file (locked or not writable):\n{path}\n"
        "Save and close this file in Excel/WPS and other applications, "
        "then rerun the script. If it persists, check the file's read-only "
        "attribute and the output-folder permissions. "
        "Do not forcibly terminate Excel or delete the existing results."
    )
    if pending is not None:
        message += (
            f"\nThe completed new file was retained at:\n{pending}\n"
            "The destination was not replaced. This run is incomplete; "
            "close the locked files and rerun to refresh all outputs."
        )
    return PermissionError(message)


def check_output_access(paths):
    """Fail before computation if an output is locked; never truncate it."""
    for path in paths:
        if path.exists():
            if not path.is_file():
                raise IsADirectoryError(f"Output path is not a file: {path}")
            try:
                # Unlike 'w', r+b checks write access without changing bytes.
                with path.open("r+b"):
                    pass
            except PermissionError as exc:
                raise output_permission_error(path) from exc

    for parent in {path.parent for path in paths}:
        try:
            with tempfile.TemporaryFile(dir=parent):
                pass
        except PermissionError as exc:
            raise output_permission_error(parent) from exc


def save_atomic(path, writer, encoding):
    """Replace one output only after a complete write on the same volume.

    A late lock keeps the complete pending file for recovery. Atomicity is
    per file, not a transaction across the four output files.
    """
    pending = None
    complete = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding=encoding, newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".pending", delete=False,
        ) as handle:
            pending = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        complete = True
        os.replace(pending, path)
    except PermissionError as exc:
        raise output_permission_error(
            path, pending if complete else None,
        ) from exc
    finally:
        # Discard only our own incomplete temporary file, never the output.
        if pending is not None and not complete:
            pending.unlink(missing_ok=True)


def save_csv_atomic(frame, path):
    save_atomic(
        path, lambda handle: frame.to_csv(handle, index=False), "utf-8-sig",
    )


def save_json_atomic(data, path):
    save_atomic(
        path,
        lambda handle: json.dump(
            data, handle, ensure_ascii=False, indent=2, default=str,
        ),
        "utf-8",
    )


def as_bool(x):
    return str(x).strip().lower() in {
        "true", "1", "yes"
    }


def expanding_midrank(x):
    """
    PIT historical percentile.

    At date t, only observations available up to and
    including t are used. No future dates enter.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty(len(x))

    for i, value in enumerate(x):
        hist = x[: i + 1]

        out[i] = (
            (hist < value).sum()
            + 0.5 * (hist == value).sum()
        ) / len(hist)

    return out


# ============================================================
# Main
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = [OUT / name for name in OUTPUT_NAMES]
    window_out, state_out, qa_out, meta_out = output_paths
    check_output_access(output_paths)

    # --------------------------------------------------------
    # 1. Upstream QA
    # --------------------------------------------------------

    if not PRED.exists():
        raise FileNotFoundError(PRED)

    if not STEP3_QA.exists():
        raise FileNotFoundError(STEP3_QA)

    step3_qa = pd.read_csv(STEP3_QA)

    step3_qa = dict(
        zip(
            step3_qa["qa_name"],
            step3_qa["qa_value"],
        )
    )

    if not as_bool(
        step3_qa.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Day13 Step3 formal QA has not passed."
        )

    # --------------------------------------------------------
    # 2. Validate frozen Day-11 prediction schema
    # --------------------------------------------------------

    cols = (
        pq.ParquetFile(PRED)
        .schema_arrow
        .names
    )

    required = [
        "security_id",
        "analysis_date",
        "window",
        BASELINE_COL,
        NETWORK_COL,
    ]

    missing = [
        c for c in required
        if c not in cols
    ]

    if missing:
        raise RuntimeError(
            f"Missing prediction columns: {missing}"
        )

    print(f"Baseline score : {BASELINE_COL}")
    print(f"Network score  : {NETWORK_COL}")

    # --------------------------------------------------------
    # 3. Read frozen strict-OOS scores only
    # --------------------------------------------------------

    x = pd.read_parquet(
        PRED,
        columns=required,
    )

    original_n = len(x)

    x["analysis_date"] = pd.to_datetime(
        x["analysis_date"],
        errors="raise",
    )

    x["window"] = pd.to_numeric(
        x["window"],
        errors="raise",
    ).astype(int)

    key = [
        "security_id",
        "analysis_date",
        "window",
    ]

    duplicate_n = int(
        x[key].duplicated().sum()
    )

    complete = (
        np.isfinite(x[BASELINE_COL])
        & np.isfinite(x[NETWORK_COL])
    )

    incomplete_n = int(
        (~complete).sum()
    )

    x = x.loc[complete].copy()

    if x.empty:
        raise RuntimeError(
            "No complete frozen prediction-score rows."
        )

    # Diagnostic only.
    # This variable is NOT used in Risk-State construction.
    x["network_minus_baseline_score"] = (
        x[NETWORK_COL]
        - x[BASELINE_COL]
    )

    # --------------------------------------------------------
    # 4. Date × Window market risk-score summary
    # --------------------------------------------------------

    dw = (
        x.groupby(
            [
                "window",
                "analysis_date",
            ],
            as_index=False,
        )
        .agg(
            stock_n=(
                "security_id",
                "size",
            ),
            baseline_risk_score_median=(
                BASELINE_COL,
                "median",
            ),
            network_risk_score_median=(
                NETWORK_COL,
                "median",
            ),
            network_minus_baseline_score_median=(
                "network_minus_baseline_score",
                "median",
            ),
        )
        .sort_values(
            [
                "window",
                "analysis_date",
            ]
        )
        .reset_index(drop=True)
    )

    if sorted(
        dw["window"]
        .unique()
        .tolist()
    ) != WINDOWS:
        raise RuntimeError(
            "Frozen window set mismatch."
        )

    # The risk score must contain time variation.
    score_std = (
        dw.groupby("window")[
            "network_risk_score_median"
        ]
        .std()
    )

    if (
        score_std.isna().any()
        or (score_std <= 1e-12).any()
    ):
        raise RuntimeError(
            "Network risk score has insufficient "
            "time variation."
        )

    # --------------------------------------------------------
    # 5. PIT expanding historical percentile
    # --------------------------------------------------------

    dw["risk_score_pct_pit"] = np.nan

    for _, idx in (
        dw.groupby(
            "window",
            sort=False,
        )
        .groups
        .items()
    ):

        idx = list(idx)

        dw.loc[
            idx,
            "risk_score_pct_pit",
        ] = expanding_midrank(
            dw.loc[
                idx,
                "network_risk_score_median",
            ]
        )

    # --------------------------------------------------------
    # 6. Combine W60 / W120 / W252
    # --------------------------------------------------------

    state = (
        dw.pivot(
            index="analysis_date",
            columns="window",
            values="risk_score_pct_pit",
        )
        .reindex(columns=WINDOWS)
        .dropna()
        .reset_index()
    )

    if state.empty:
        raise RuntimeError(
            "No common dates across W60/W120/W252."
        )

    state.columns = [
        "analysis_date",
        "risk_score_pct_W60",
        "risk_score_pct_W120",
        "risk_score_pct_W252",
    ]

    pct_cols = [
        "risk_score_pct_W60",
        "risk_score_pct_W120",
        "risk_score_pct_W252",
    ]

    # Equal-weight frozen composite.
    state["network_risk_state_score"] = (
        state[pct_cols]
        .mean(axis=1)
    )

    # Diagnostic only.
    state["window_dispersion"] = (
        state[pct_cols]
        .std(
            axis=1,
            ddof=0,
        )
    )

    # --------------------------------------------------------
    # 7. Fixed LOW / MID / HIGH definition
    # --------------------------------------------------------

    score = state[
        "network_risk_state_score"
    ]

    state["network_risk_state"] = np.select(
        [
            score <= 1 / 3,
            score <= 2 / 3,
        ],
        [
            "LOW_NETWORK_RISK",
            "MID_NETWORK_RISK",
        ],
        default="HIGH_NETWORK_RISK",
    )

    state["previous_risk_state"] = (
        state["network_risk_state"]
        .shift(1)
    )

    state.loc[
        state.index[0],
        "previous_risk_state",
    ] = state.loc[
        state.index[0],
        "network_risk_state",
    ]

    state["risk_state_changed"] = (
        state["network_risk_state"]
        != state["previous_risk_state"]
    )

    # --------------------------------------------------------
    # 8. Formal QA
    # --------------------------------------------------------

    qa = {
        "step3_qa_pass":
            True,

        "prediction_row_count":
            int(original_n),

        "prediction_incomplete_row_count":
            int(incomplete_n),

        "duplicate_stock_date_window_count":
            int(duplicate_n),

        "prediction_columns_are_scores":
            True,

        "raw_future_ivol_interpreted_as_state_input":
            False,

        "realized_target_required_for_state":
            False,

        "window_count":
            int(
                dw["window"].nunique()
            ),

        "date_window_row_count":
            int(len(dw)),

        "duplicate_date_window_count":
            int(
                dw[
                    [
                        "window",
                        "analysis_date",
                    ]
                ]
                .duplicated()
                .sum()
            ),

        "common_state_date_count":
            int(len(state)),

        "state_missing_count":
            int(
                state[
                    "network_risk_state"
                ]
                .isna()
                .sum()
            ),

        "state_score_in_unit_interval":
            bool(
                state[
                    "network_risk_state_score"
                ]
                .between(0, 1)
                .all()
            ),

        "future_target_used_in_state":
            False,

        "network_minus_baseline_used_in_state":
            False,

        "window_dispersion_used_in_state":
            False,

        "model_refitted":
            False,

        "factor_reestimated":
            False,

        "factor_selected_from_results":
            False,

        "window_selected_from_results":
            False,

        "state_threshold_selected_from_results":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        duplicate_n == 0
        and qa["window_count"] == 3
        and qa[
            "duplicate_date_window_count"
        ] == 0
        and qa[
            "common_state_date_count"
        ] > 0
        and qa[
            "state_missing_count"
        ] == 0
        and qa[
            "state_score_in_unit_interval"
        ]
        and not qa[
            "future_target_used_in_state"
        ]
        and not qa[
            "realized_target_required_for_state"
        ]
        and not qa[
            "network_minus_baseline_used_in_state"
        ]
    )

    if not qa[
        "all_formal_qa_pass"
    ]:
        raise RuntimeError(
            f"Day13 Step4 QA failed: {qa}"
        )

    # --------------------------------------------------------
    # 9. Save outputs
    # --------------------------------------------------------

    save_csv_atomic(dw, window_out)
    save_csv_atomic(state, state_out)

    qa_frame = pd.DataFrame(
        [
            {
                "qa_name": k,
                "qa_value": v,
            }
            for k, v in qa.items()
        ]
    )
    save_csv_atomic(qa_frame, qa_out)

    # --------------------------------------------------------
    # 10. Semantic-fixed metadata
    # --------------------------------------------------------

    metadata = {
        "research_day": 13,

        "step":
            "Step4_Frozen_Network_Risk_State_Construction",

        "source":
            str(PRED),

        "prediction_columns": {
            "baseline":
                BASELINE_COL,
            "network":
                NETWORK_COL,
        },

        "prediction_score_semantics": (
            "baseline_score and network_score are frozen "
            "strict-OOS future-IVOL risk-model scores. "
            "They are used ordinally and are not interpreted "
            "as future idiosyncratic volatility in raw "
            "volatility units."
        ),

        "windows":
            WINDOWS,

        "date_window_risk_measure": (
            "Cross-sectional median of the frozen strict-OOS "
            "network-enhanced future-IVOL risk score."
        ),

        "normalization": (
            "Expanding historical midrank percentile within "
            "each window. At date t, only score observations "
            "dated at or before t enter the normalization; "
            "no future date is used."
        ),

        "composite": (
            "Equal-weight mean of W60/W120/W252 PIT "
            "risk-score percentiles on common dates."
        ),

        "states": {
            "LOW_NETWORK_RISK":
                "score <= 1/3",

            "MID_NETWORK_RISK":
                "1/3 < score <= 2/3",

            "HIGH_NETWORK_RISK":
                "score > 2/3",
        },

        "sample_scope_note": (
            "Risk-state construction uses all frozen strict-OOS "
            "prediction dates with available baseline_score and "
            "network_score. Realized future-IVOL availability is "
            "not required. Therefore the number of state "
            "date-window observations may exceed the Day-11 "
            "realized-target OOS evaluation count."
        ),

        "diagnostic_variables": {
            "network_minus_baseline_score_median": (
                "Median difference between network_score and "
                "baseline_score. Diagnostic only; not used "
                "to define the Risk-State."
            ),

            "window_dispersion": (
                "Cross-window dispersion of the three PIT "
                "risk-score percentiles. Diagnostic only; "
                "not used to define the Risk-State."
            ),
        },

        "interpretation_boundary": (
            "The resulting state is a network-enhanced "
            "strict-OOS future-risk-score state. It is not "
            "a direct estimate of market IVOL in raw units, "
            "and it is not a pure network-only state because "
            "the Day-11 network model augments the frozen "
            "traditional baseline with the nine network factors."
        ),

        "future_target_used_in_state":
            False,

        "realized_target_required_for_state":
            False,

        "model_refitted":
            False,

        "factor_or_window_selected_from_results":
            False,

        "formal_qa":
            qa,
    }

    save_json_atomic(metadata, meta_out)

    # --------------------------------------------------------
    # 11. Console summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "Day 13 Step 4 - Frozen Network Risk-State"
    )
    print("=" * 72)

    print(
        f"Prediction rows : {original_n}"
    )
    print(
        f"Date-window rows: {len(dw)}"
    )
    print(
        f"Common dates    : {len(state)}"
    )

    print()
    print("State distribution:")
    print(
        state[
            "network_risk_state"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Formal QA       : PASS")
    print(f"Output           : {OUT}")


if __name__ == "__main__":
    main()

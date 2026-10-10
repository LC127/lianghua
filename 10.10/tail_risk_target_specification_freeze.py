from pathlib import Path
from datetime import datetime, timezone
import json
import os

import pandas as pd


# ============================================================
# Day 14 - Step 1
# M4 Tail-Risk Target Specification Freeze
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")

M1_META = (
    ROOT / "output" / "M1_day13"
    / "06_step6_m1_v1_final_freeze"
    / "m1_v1_0_freeze_metadata.json"
)

OUT = (
    ROOT / "output" / "M1_day14"
    / "01_step1_m4_tail_risk_target_specification_freeze"
)
OUT.mkdir(parents=True, exist_ok=True)

SPEC_OUT = OUT / "m4_tail_risk_target_specification.csv"
QA_OUT = OUT / "day14_step1_qa.csv"
META_OUT = OUT / "day14_step1_metadata.json"


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    os.replace(tmp, path)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Verify frozen M1 v1.0
    # --------------------------------------------------------

    if not M1_META.exists():
        raise FileNotFoundError(M1_META)

    m1 = load_json(M1_META)

    if (
        m1.get("freeze_version") != "M1_v1.0"
        or m1.get("status") != "FROZEN"
        or not m1.get(
            "formal_qa", {}
        ).get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "M1 v1.0 final freeze is not valid."
        )

    interface_path = Path(
        m1["canonical_m1_to_m4_interface"]
    )

    if not interface_path.exists():
        raise FileNotFoundError(
            interface_path
        )

    interface = pd.read_csv(
        interface_path,
        usecols=["analysis_date"],
    )

    interface["analysis_date"] = pd.to_datetime(
        interface["analysis_date"],
        errors="raise",
    )

    if (
        interface.empty
        or interface[
            "analysis_date"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Invalid frozen M1-to-M4 interface."
        )

    first_m1_date = (
        interface["analysis_date"]
        .min()
        .date()
        .isoformat()
    )

    # --------------------------------------------------------
    # 2. Freeze target definitions
    # --------------------------------------------------------

    spec = pd.DataFrame([
        {
            "target_id": "T1",
            "tier": "PRIMARY",
            "target_name":
                "fwd_20d_max_drawdown_loss",
            "horizon_td": 20,
            "definition":
                "Maximum peak-to-trough drawdown over "
                "P_t,...,P_t+20; expressed as positive loss.",
            "direction":
                "HIGHER_IS_WORSE",
            "threshold_rule": "",
        },
        {
            "target_id": "T2",
            "tier": "ROBUSTNESS",
            "target_name":
                "fwd_5d_max_drawdown_loss",
            "horizon_td": 5,
            "definition":
                "5-trading-day forward maximum drawdown loss.",
            "direction":
                "HIGHER_IS_WORSE",
            "threshold_rule": "",
        },
        {
            "target_id": "T3",
            "tier": "ROBUSTNESS",
            "target_name":
                "fwd_60d_max_drawdown_loss",
            "horizon_td": 60,
            "definition":
                "60-trading-day forward maximum drawdown loss.",
            "direction":
                "HIGHER_IS_WORSE",
            "threshold_rule": "",
        },
        {
            "target_id": "T4",
            "tier": "SECONDARY",
            "target_name":
                "fwd_20d_realized_vol_annualized",
            "horizon_td": 20,
            "definition":
                "Annualized realized volatility from "
                "daily returns t+1,...,t+20.",
            "direction":
                "HIGHER_IS_WORSE",
            "threshold_rule": "",
        },
        {
            "target_id": "T5",
            "tier": "SECONDARY",
            "target_name":
                "fwd_20d_worst_daily_loss",
            "horizon_td": 20,
            "definition":
                "Negative of the minimum daily return "
                "over t+1,...,t+20.",
            "direction":
                "HIGHER_IS_WORSE",
            "threshold_rule": "",
        },
        {
            "target_id": "T6",
            "tier": "SECONDARY",
            "target_name":
                "fwd_20d_downside_semideviation_annualized",
            "horizon_td": 20,
            "definition":
                "Annualized square-root mean squared "
                "negative daily returns over t+1,...,t+20.",
            "direction":
                "HIGHER_IS_WORSE",
            "threshold_rule": "",
        },
        {
            "target_id": "T7",
            "tier": "DERIVED_EVENT",
            "target_name":
                "fwd_20d_tail_event_p90",
            "horizon_td": 20,
            "definition":
                "Indicator that T1 exceeds its frozen "
                "pre-M1 calibration 90th percentile.",
            "direction":
                "1_IS_TAIL_EVENT",
            "threshold_rule":
                "PRE_M1_FULLY_REALIZED_CALIBRATION_P90",
        },
        {
            "target_id": "T8",
            "tier": "DERIVED_EVENT",
            "target_name":
                "fwd_20d_tail_event_p95",
            "horizon_td": 20,
            "definition":
                "Indicator that T1 exceeds its frozen "
                "pre-M1 calibration 95th percentile.",
            "direction":
                "1_IS_TAIL_EVENT",
            "threshold_rule":
                "PRE_M1_FULLY_REALIZED_CALIBRATION_P95",
        },
    ])

    # --------------------------------------------------------
    # 3. Immutable specification check
    # --------------------------------------------------------

    if SPEC_OUT.exists():

        old = (
            pd.read_csv(SPEC_OUT)
            .fillna("")
            .astype(str)
        )

        new = (
            spec.fillna("")
            .astype(str)
        )

        if not old.equals(new):
            raise RuntimeError(
                "Frozen Day14 Step1 target "
                "specification changed. "
                "Do not overwrite silently."
            )

    else:
        atomic_csv(spec, SPEC_OUT)

    # --------------------------------------------------------
    # 4. Formal QA
    # --------------------------------------------------------

    qa = {
        "m1_v1_0_frozen":
            True,

        "m1_v1_0_qa_pass":
            True,

        "interface_row_count":
            int(len(interface)),

        "interface_duplicate_date_count":
            int(
                interface[
                    "analysis_date"
                ]
                .duplicated()
                .sum()
            ),

        "first_m1_state_date":
            first_m1_date,

        "target_count":
            int(len(spec)),

        "primary_target_count":
            int(
                (
                    spec["tier"]
                    == "PRIMARY"
                ).sum()
            ),

        "derived_event_count":
            int(
                (
                    spec["tier"]
                    == "DERIVED_EVENT"
                ).sum()
            ),

        "all_horizons_positive":
            bool(
                (
                    spec["horizon_td"]
                    > 0
                ).all()
            ),

        "target_definition_missing_count":
            int(
                (
                    spec["definition"]
                    .str.len()
                    == 0
                ).sum()
            ),

        "m1_state_used_to_define_target":
            False,

        "m4_outcome_computed":
            False,

        "target_selected_from_results":
            False,

        "threshold_selected_from_results":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["m1_v1_0_frozen"]
        and qa["m1_v1_0_qa_pass"]
        and qa[
            "interface_duplicate_date_count"
        ] == 0
        and qa[
            "primary_target_count"
        ] == 1
        and qa[
            "derived_event_count"
        ] == 2
        and qa[
            "all_horizons_positive"
        ]
        and qa[
            "target_definition_missing_count"
        ] == 0
        and not qa[
            "m1_state_used_to_define_target"
        ]
        and not qa[
            "m4_outcome_computed"
        ]
        and not qa[
            "target_selected_from_results"
        ]
    )

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            f"Day14 Step1 QA failed: {qa}"
        )

    # --------------------------------------------------------
    # 5. Metadata
    # --------------------------------------------------------

    metadata = {
        "research_day": 14,

        "step":
            "Step1_M4_Tail_Risk_Target_Specification_Freeze",

        "status":
            "FROZEN",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(
                timespec="seconds"
            ),

        "m1_version":
            "M1_v1.0",

        "m1_interface":
            str(interface_path),

        "primary_market_proxy":
            "CSI300",

        "robustness_market_proxies": [
            "CSI500",
            "SSE_COMPOSITE",
        ],

        "primary_target":
            "fwd_20d_max_drawdown_loss",

        "primary_horizon_td":
            20,

        "tail_event_threshold_rule": (
            "Use only pre-M1 calibration observations "
            "whose full forward label is realized before "
            f"the first M1 state date ({first_m1_date}). "
            "Freeze both P90 and P95 thresholds before "
            "M1-M4 validation."
        ),

        "timing_rule":
            (
                "All target returns must occur strictly "
                "after the formation date. Future target "
                "information must never enter M1 state "
                "construction or target selection."
            ),

        "data_source_resolution":
            "DEFERRED_TO_DAY14_STEP2",

        "selection_policy":
            (
                "No target, horizon, market proxy, or "
                "tail threshold may be selected using "
                "M1 Risk-State validation results."
            ),

        "m4_outcome_computed":
            False,

        "formal_qa":
            qa,
    }

    atomic_csv(
        pd.DataFrame([
            {
                "qa_name": k,
                "qa_value": v,
            }
            for k, v in qa.items()
        ]),
        QA_OUT,
    )

    atomic_json(
        metadata,
        META_OUT,
    )

    # --------------------------------------------------------
    # 6. Console summary
    # --------------------------------------------------------

    print("=" * 72)
    print(
        "Day 14 Step 1 - "
        "M4 Tail-Risk Target Specification Freeze"
    )
    print("=" * 72)

    print(
        f"M1 interface dates : {len(interface)}"
    )

    print(
        f"First M1 state     : {first_m1_date}"
    )

    print(
        f"Frozen targets     : {len(spec)}"
    )

    print(
        "Primary target     : "
        "fwd_20d_max_drawdown_loss"
    )

    print(
        "Formal QA          : PASS"
    )

    print(
        f"Output             : {OUT}"
    )


if __name__ == "__main__":
    main()
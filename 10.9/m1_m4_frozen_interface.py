from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# Day 13 - Step 5
# M1-to-M4 Frozen Interface Panel
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")

S4 = (
    ROOT / "output" / "M1_day13"
    / "04_step4_frozen_network_risk_state"
)

OUT = (
    ROOT / "output" / "M1_day13"
    / "05_step5_m1_m4_frozen_interface"
)

STATE_IN = S4 / "frozen_network_risk_state.csv"
QA_IN = S4 / "day13_step4_qa.csv"
META_IN = S4 / "day13_step4_metadata.json"

PANEL_OUT = OUT / "m1_to_m4_frozen_interface_panel.csv"
QA_OUT = OUT / "day13_step5_qa.csv"
META_OUT = OUT / "day13_step5_metadata.json"

VALID_STATES = {
    "LOW_NETWORK_RISK",
    "MID_NETWORK_RISK",
    "HIGH_NETWORK_RISK",
}


def as_bool(x):
    return str(x).strip().lower() in {
        "true", "1", "yes"
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Upstream integrity
    # ========================================================

    for p in [
        STATE_IN,
        QA_IN,
        META_IN,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    q = pd.read_csv(QA_IN)

    q = dict(
        zip(
            q["qa_name"],
            q["qa_value"],
        )
    )

    if not as_bool(
        q.get(
            "all_formal_qa_pass",
            False,
        )
    ):
        raise RuntimeError(
            "Day13 Step4 QA has not passed."
        )

    with open(
        META_IN,
        "r",
        encoding="utf-8",
    ) as f:
        meta4 = json.load(f)

    if (
        meta4.get("step")
        !=
        "Step4_Frozen_Network_Risk_State_Construction"
    ):
        raise RuntimeError(
            "Unexpected Step4 metadata."
        )

    if (
        meta4.get(
            "future_target_used_in_state"
        )
        is not False
    ):
        raise RuntimeError(
            "Step4 state is not future-target clean."
        )

    # ========================================================
    # 2. Load frozen Step4 state
    # ========================================================

    x = pd.read_csv(STATE_IN)

    required = [
        "analysis_date",
        "risk_score_pct_W60",
        "risk_score_pct_W120",
        "risk_score_pct_W252",
        "network_risk_state_score",
        "window_dispersion",
        "network_risk_state",
        "previous_risk_state",
        "risk_state_changed",
    ]

    missing = [
        c for c in required
        if c not in x.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing Step4 columns: {missing}"
        )

    x = x[required].copy()

    x["analysis_date"] = pd.to_datetime(
        x["analysis_date"],
        errors="raise",
    )

    x = (
        x.sort_values("analysis_date")
        .reset_index(drop=True)
    )

    if x[
        "analysis_date"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Step4 analysis_date."
        )

    if not set(
        x["network_risk_state"]
    ).issubset(VALID_STATES):
        raise RuntimeError(
            "Unexpected Risk-State label."
        )

    # ========================================================
    # 3. Build M1 -> M4 interface schema
    # ========================================================

    panel = x.rename(
        columns={
            "risk_score_pct_W60":
                "m1_risk_score_pct_W60",

            "risk_score_pct_W120":
                "m1_risk_score_pct_W120",

            "risk_score_pct_W252":
                "m1_risk_score_pct_W252",

            "network_risk_state_score":
                "m1_network_risk_state_score",

            "window_dispersion":
                "m1_window_dispersion",

            "network_risk_state":
                "m1_network_risk_state",

            "previous_risk_state":
                "m1_previous_risk_state",

            "risk_state_changed":
                "m1_risk_state_changed",
        }
    ).copy()

    # State validity interval:
    # [current state date, next state date)
    panel[
        "m1_state_valid_from"
    ] = panel["analysis_date"]

    panel[
        "m1_state_valid_to_exclusive"
    ] = (
        panel["analysis_date"]
        .shift(-1)
    )

    panel[
        "m1_state_open_ended"
    ] = (
        panel[
            "m1_state_valid_to_exclusive"
        ]
        .isna()
    )

    panel = panel[
        [
            "analysis_date",

            "m1_state_valid_from",
            "m1_state_valid_to_exclusive",
            "m1_state_open_ended",

            "m1_network_risk_state_score",
            "m1_network_risk_state",
            "m1_previous_risk_state",
            "m1_risk_state_changed",

            "m1_risk_score_pct_W60",
            "m1_risk_score_pct_W120",
            "m1_risk_score_pct_W252",

            "m1_window_dispersion",
        ]
    ]

    # ========================================================
    # 4. QA
    # ========================================================

    closed = (
        panel[
            "m1_state_valid_to_exclusive"
        ]
        .notna()
    )

    invalid_interval_n = int(
        (
            panel.loc[
                closed,
                "m1_state_valid_to_exclusive",
            ]
            <=
            panel.loc[
                closed,
                "m1_state_valid_from",
            ]
        ).sum()
    )

    pairs = [
        (
            "m1_risk_score_pct_W60",
            "risk_score_pct_W60",
        ),
        (
            "m1_risk_score_pct_W120",
            "risk_score_pct_W120",
        ),
        (
            "m1_risk_score_pct_W252",
            "risk_score_pct_W252",
        ),
        (
            "m1_network_risk_state_score",
            "network_risk_state_score",
        ),
        (
            "m1_window_dispersion",
            "window_dispersion",
        ),
    ]

    repro = max(
        float(
            np.nanmax(
                np.abs(
                    panel[a].to_numpy(float)
                    -
                    x[b].to_numpy(float)
                )
            )
        )
        for a, b in pairs
    )

    state_mismatch = int(
        (
            panel[
                "m1_network_risk_state"
            ].to_numpy()
            !=
            x[
                "network_risk_state"
            ].to_numpy()
        ).sum()
    )

    qa = {
        "step4_qa_pass":
            True,

        "interface_row_count":
            int(len(panel)),

        "step4_row_count":
            int(len(x)),

        "duplicate_analysis_date_count":
            int(
                panel[
                    "analysis_date"
                ]
                .duplicated()
                .sum()
            ),

        "missing_core_count":
            int(
                panel.drop(
                    columns=[
                        "m1_state_valid_to_exclusive"
                    ]
                )
                .isna()
                .sum()
                .sum()
            ),

        "valid_state_label_set":
            bool(
                set(
                    panel[
                        "m1_network_risk_state"
                    ]
                ).issubset(
                    VALID_STATES
                )
            ),

        "state_score_in_unit_interval":
            bool(
                panel[
                    "m1_network_risk_state_score"
                ]
                .between(
                    0,
                    1,
                )
                .all()
            ),

        "closed_interval_invalid_count":
            invalid_interval_n,

        "open_ended_interval_count":
            int(
                panel[
                    "m1_state_open_ended"
                ]
                .sum()
            ),

        "step4_reproduction_max_abs_diff":
            repro,

        "state_label_mismatch_count":
            state_mismatch,

        "m4_data_joined":
            False,

        "m4_outcome_used":
            False,

        "risk_state_reestimated":
            False,

        "factor_or_window_selected_from_results":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["interface_row_count"]
        == qa["step4_row_count"]

        and qa[
            "duplicate_analysis_date_count"
        ] == 0

        and qa[
            "missing_core_count"
        ] == 0

        and qa[
            "valid_state_label_set"
        ]

        and qa[
            "state_score_in_unit_interval"
        ]

        and qa[
            "closed_interval_invalid_count"
        ] == 0

        and qa[
            "open_ended_interval_count"
        ] == 1

        and qa[
            "step4_reproduction_max_abs_diff"
        ] <= 1e-12

        and qa[
            "state_label_mismatch_count"
        ] == 0
    )

    if not qa[
        "all_formal_qa_pass"
    ]:
        raise RuntimeError(
            f"Day13 Step5 QA failed: {qa}"
        )

    # ========================================================
    # 5. Save
    # ========================================================

    panel.to_csv(
        PANEL_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "qa_name": k,
                "qa_value": v,
            }
            for k, v in qa.items()
        ]
    ).to_csv(
        QA_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "research_day": 13,

        "step":
            "Step5_M1_to_M4_Frozen_Interface_Panel",

        "interface_version":
            1,

        "source":
            str(STATE_IN),

        "purpose": (
            "Expose the frozen M1 network-enhanced "
            "strict-OOS future-risk-score state to M4 "
            "without joining M4 outcomes or "
            "re-estimating M1."
        ),

        "canonical_key":
            "analysis_date",

        "asof_interval":
            (
                "[m1_state_valid_from, "
                "m1_state_valid_to_exclusive)"
            ),

        "same_frequency_join_rule":
            (
                "For M4 data on the M1 analysis-date "
                "calendar, join exactly on analysis_date."
            ),

        "daily_event_join_rule":
            (
                "For M4 daily/event data, use the most "
                "recent M1 state only inside a closed "
                "interface interval. The final open-ended "
                "state must not be automatically carried "
                "beyond the frozen M1 sample."
            ),

        "interpretation_boundary":
            (
                "The M1 state is a network-enhanced "
                "risk-score state, not raw IVOL, not a "
                "pure network-only state, and not causal "
                "evidence of tail-risk transmission."
            ),

        "m4_data_joined":
            False,

        "m4_outcome_used":
            False,

        "formal_qa":
            qa,
    }

    with open(
        META_OUT,
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

    print("=" * 72)
    print(
        "Day 13 Step 5 - "
        "M1-to-M4 Frozen Interface Panel"
    )
    print("=" * 72)

    print(
        f"Rows      : {len(panel)}"
    )

    print(
        "Date range: "
        f"{panel['analysis_date'].min().date()}"
        " -> "
        f"{panel['analysis_date'].max().date()}"
    )

    print(
        "Formal QA : PASS"
    )

    print(
        f"Output    : {OUT}"
    )


if __name__ == "__main__":
    main()
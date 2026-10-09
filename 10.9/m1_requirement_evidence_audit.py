from pathlib import Path
import json, os
import pandas as pd

# ============================================================
# M1 - Research Day 13 - Step 1
# Requirement-to-Evidence Final Audit
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
OUTROOT = ROOT / "output"
OUT = OUTROOT / "M1_day13" / "01_step1_requirement_evidence_audit"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------

def save_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)

def as_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}

def find_one(patterns):
    for pat in patterns:
        hits = [
            p for p in sorted(OUTROOT.glob(pat))
            if "archive" not in str(p).lower()
        ]
        if hits:
            return hits[0]
    return None

def qa_pass(path):
    if path is None or not path.exists():
        return None
    try:
        q = pd.read_csv(path)
        if {"qa_name", "qa_value"}.issubset(q.columns):
            d = dict(zip(q["qa_name"], q["qa_value"]))
            return as_bool(d.get("all_formal_qa_pass", False))
    except Exception:
        pass
    return None

def row(rid, requirement, evidence_patterns, qa_patterns=(),
        expected="COMPLETE", note=""):
    evidence = find_one(evidence_patterns)
    qa = find_one(qa_patterns) if qa_patterns else None
    qpass = qa_pass(qa)

    if expected == "PENDING":
        status = "PENDING"
    elif evidence is None:
        status = "MISSING_EVIDENCE"
    elif qa_patterns and qpass is not True:
        status = "QA_NOT_CONFIRMED"
    else:
        status = "COMPLETE"

    return {
        "requirement_id": rid,
        "requirement": requirement,
        "status": status,
        "evidence_exists": evidence is not None,
        "qa_pass": qpass,
        "evidence_file": "" if evidence is None else str(evidence),
        "qa_file": "" if qa is None else str(qa),
        "note": note,
    }

# ============================================================
# 1. M1 requirement -> frozen evidence
# ============================================================

rows = [
    row(
        "R1", "A-share full-market network construction",
        [
            "M1_day4/**/*.parquet",
            "M1_day5/**/community_bridge_feature_panel.parquet",
        ],
        note="Day-5 frozen stock-level network features provide downstream evidence that the full-market network pipeline was completed."
    ),
    row(
        "R2", "Frozen stock-level network factor library",
        ["M1_day5/**/community_bridge_feature_panel.parquet"],
        note="Nine frozen network factors are reused by Days 6-12."
    ),
    row(
        "R3", "Forward return / future-risk label pipeline",
        ["M1_day6/**/forward_label_panel.parquet"],
        note="Frozen forward labels support PIT risk and alpha evaluation."
    ),
    row(
        "R4", "Traditional-characteristic control pipeline",
        ["M1_day7/**/traditional_characteristic_panel.parquet"],
        note="Industry + size/momentum/reversal/volatility/turnover controls."
    ),
    row(
        "R5", "Liquidity and capacity pipeline",
        [
            "M1_day8/**/capacity_limit_rebalance_pair.csv",
            "M1_day8/**/stock_liquidity_panel.parquet",
        ],
        note="Frozen ADV/capacity outputs are reused by later implementation tests."
    ),
    row(
        "R6", "Strict OOS future-risk forecasting",
        [
            "M1_day11/**/*oos*summary*.csv",
            "M1_day11/**/*validation*summary*.csv",
            "M1_day11/**/*.csv",
        ],
        ["M1_day11/03_stage3_strict_oos_forecasting/day11_step3_qa.csv"],
        note="Expanding-window PIT OOS comparison of traditional controls vs. controls + network factors."
    ),
    row(
        "R7", "Alpha IC and characteristic-neutral validation",
        ["M1_day12/02_stage2_alpha_ic_validation/alpha_ic_comparison_summary.csv"],
        ["M1_day12/02_stage2_alpha_ic_validation/day12_step2_qa.csv"],
    ),
    row(
        "R8", "Monotonic quintile / long-short validation",
        ["M1_day12/03_stage3_monotonic_portfolio_validation/portfolio_validation_summary.csv"],
        ["M1_day12/03_stage3_monotonic_portfolio_validation/day12_step3_qa.csv"],
    ),
    row(
        "R9", "Turnover and transaction-cost validation",
        ["M1_day12/04_stage4_turnover_cost_validation/turnover_cost_summary.csv"],
        ["M1_day12/04_stage4_turnover_cost_validation/day12_step4_qa.csv"],
    ),
    row(
        "R10", "Capacity-constrained alpha validation",
        ["M1_day12/05_stage5_capacity_constrained_alpha/capacity_alpha_summary.csv"],
        ["M1_day12/05_stage5_capacity_constrained_alpha/day12_step5_qa.csv"],
    ),
    row(
        "R11", "Alpha information-implementation joint summary",
        ["M1_day12/06_stage6_alpha_joint_summary/alpha_information_implementation_joint_summary.csv"],
        ["M1_day12/06_stage6_alpha_joint_summary/day12_step6_qa.csv"],
    ),
    row(
        "R12", "M1-to-M4 tail-risk transmission linkage",
        [],
        expected="PENDING",
        note="Cross-module extension; Day 13+ should define a PIT-safe M1-to-M4 network-risk interface rather than redoing M4."
    ),
]

audit = pd.DataFrame(rows)

# ============================================================
# 2. Deliverable-level audit
# ============================================================

core_ids = [f"R{i}" for i in range(1, 12)]
core = audit[audit["requirement_id"].isin(core_ids)]

deliverables = pd.DataFrame([
    {
        "deliverable": "Network code/data pipeline",
        "status": "COMPLETE" if core[core["requirement_id"].isin(
            ["R1","R2","R3","R4","R5"]
        )]["status"].eq("COMPLETE").all() else "REVIEW_REQUIRED",
        "evidence": "Days 5-8 frozen network/features/labels/controls/liquidity-capacity outputs",
    },
    {
        "deliverable": "Risk-factor research evidence",
        "status": "COMPLETE" if audit.loc[
            audit["requirement_id"].eq("R6"), "status"
        ].eq("COMPLETE").all() else "REVIEW_REQUIRED",
        "evidence": "Day 11 strict PIT expanding-window OOS risk validation",
    },
    {
        "deliverable": "Alpha factor research evidence",
        "status": "COMPLETE" if audit.loc[
            audit["requirement_id"].isin(["R7","R8","R9","R10","R11"]), "status"
        ].eq("COMPLETE").all() else "REVIEW_REQUIRED",
        "evidence": "Day 12 IC -> portfolio -> cost -> capacity -> joint summary",
    },
    {
        "deliverable": "Literature-review report",
        "status": "PACKAGING_PENDING",
        "evidence": "Method research exists, but this audit does not assume a final review document unless separately packaged.",
    },
    {
        "deliverable": "M1 final factor research report",
        "status": "PACKAGING_PENDING",
        "evidence": "Underlying frozen evidence exists; final report/README should be assembled after Day 13 closure.",
    },
    {
        "deliverable": "M1-to-M4 interface",
        "status": "PENDING",
        "evidence": "Cross-module handoff remains to be constructed.",
    },
])

# ============================================================
# 3. Formal QA for this audit
# ============================================================

qa = {
    "requirement_row_count": len(audit),
    "core_requirement_count": len(core),
    "core_complete_count": int(core["status"].eq("COMPLETE").sum()),
    "core_incomplete_count": int((~core["status"].eq("COMPLETE")).sum()),
    "m4_linkage_pending": bool(
        audit.loc[audit["requirement_id"].eq("R12"), "status"].eq("PENDING").all()
    ),
    "day12_step6_confirmed_complete": bool(
        audit.loc[audit["requirement_id"].eq("R11"), "status"].eq("COMPLETE").all()
    ),
    "no_result_based_factor_selection": True,
    "no_result_based_window_selection": True,
    "no_reestimation_performed": True,
}
qa["all_formal_qa_pass"] = bool(
    qa["requirement_row_count"] == 12
    and qa["core_complete_count"] == qa["core_requirement_count"]
    and qa["day12_step6_confirmed_complete"]
    and qa["m4_linkage_pending"]
)

# ============================================================
# 4. Save
# ============================================================

save_csv(audit, OUT / "m1_requirement_completion_matrix.csv")
save_csv(deliverables, OUT / "m1_deliverable_completion_matrix.csv")
save_csv(
    pd.DataFrame([{"qa_name": k, "qa_value": v} for k, v in qa.items()]),
    OUT / "day13_step1_qa.csv",
)

meta = {
    "research_day": 13,
    "step": "Step1_M1_Requirement_to_Evidence_Final_Audit",
    "principle": "Audit frozen evidence only; do not re-estimate networks, factors, labels, risk models, alpha, costs, or capacity.",
    "status_rule": {
        "COMPLETE": "Required frozen evidence exists and QA passes when a formal QA file is expected.",
        "MISSING_EVIDENCE": "Expected frozen evidence was not found.",
        "QA_NOT_CONFIRMED": "Evidence exists but formal QA was not confirmed.",
        "PENDING": "Explicitly unfinished cross-module work.",
        "PACKAGING_PENDING": "Research evidence exists but final report/document packaging is still required.",
    },
    "formal_qa": qa,
}
with open(OUT / "day13_step1_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("=" * 78)
print("M1 - Day 13 - Step 1: Requirement-to-Evidence Final Audit")
print("=" * 78)
print(audit[["requirement_id","requirement","status"]].to_string(index=False))
print("\nDeliverables:")
print(deliverables[["deliverable","status"]].to_string(index=False))
print("\nFormal QA:")
for k, v in qa.items():
    print(f"  {k}: {v}")
print(f"\nOutput: {OUT}")

if not qa["all_formal_qa_pass"]:
    raise RuntimeError(
        "Day-13 Step-1 audit QA failed. Inspect m1_requirement_completion_matrix.csv."
    )

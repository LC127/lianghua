"""Day14 Step1: freeze M4 tail-risk target definitions; no future outcome is read."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import pandas as pd

ROOT = Path(os.getenv("M1_ROOT", r"D:\M1_StockNetwork"))
D13 = ROOT / "output" / "M1_day13"
M1_META = D13 / "06_step6_m1_v1_final_freeze" / "m1_v1_0_freeze_metadata.json"
M1_LIST = D13 / "06_step6_m1_v1_final_freeze" / "m1_v1_0_final_freeze_manifest.csv"
OUT = ROOT / "output" / "M1_day14" / "01_step1_m4_tail_risk_target_specification_freeze"
SPEC = OUT / "m4_tail_risk_target_specification.csv"
QA = OUT / "day14_step1_qa.csv"
META = OUT / "day14_step1_metadata.json"

FIELDS = ["target_id", "tier", "target_name", "horizon_td",
          "definition", "direction", "threshold_rule"]
TARGETS = [
    ("T1", "PRIMARY", "fwd_20d_max_drawdown_loss", 20,
     "Maximum peak-to-trough drawdown over P_t,...,P_t+20; expressed as positive loss.", "HIGHER_IS_WORSE", ""),
    ("T2", "ROBUSTNESS", "fwd_5d_max_drawdown_loss", 5,
     "5-trading-day forward maximum drawdown loss.", "HIGHER_IS_WORSE", ""),
    ("T3", "ROBUSTNESS", "fwd_60d_max_drawdown_loss", 60,
     "60-trading-day forward maximum drawdown loss.", "HIGHER_IS_WORSE", ""),
    ("T4", "SECONDARY", "fwd_20d_realized_vol_annualized", 20,
     "Annualized realized volatility from daily returns t+1,...,t+20.", "HIGHER_IS_WORSE", ""),
    ("T5", "SECONDARY", "fwd_20d_worst_daily_loss", 20,
     "Negative of the minimum daily return over t+1,...,t+20.", "HIGHER_IS_WORSE", ""),
    ("T6", "SECONDARY", "fwd_20d_downside_semideviation_annualized", 20,
     "Annualized square-root mean squared negative daily returns over t+1,...,t+20.", "HIGHER_IS_WORSE", ""),
    ("T7", "DERIVED_EVENT", "fwd_20d_tail_event_p90", 20,
     "Indicator that T1 exceeds its frozen pre-M1 calibration 90th percentile.",
     "1_IS_TAIL_EVENT", "PRE_M1_FULLY_REALIZED_CALIBRATION_P90"),
    ("T8", "DERIVED_EVENT", "fwd_20d_tail_event_p95", 20,
     "Indicator that T1 exceeds its frozen pre-M1 calibration 95th percentile.",
     "1_IS_TAIL_EVENT", "PRE_M1_FULLY_REALIZED_CALIBRATION_P95"),
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    # Verify the M1 freeze and the exact interface bytes recorded by Step6.
    m1 = json.loads(M1_META.read_text(encoding="utf-8"))
    require(m1.get("freeze_version") == "M1_v1.0"
            and m1.get("status") == "FROZEN"
            and m1.get("formal_qa", {}).get("all_formal_qa_pass") is True,
            "M1 freeze invalid")
    interface = Path(m1["canonical_m1_to_m4_interface"])
    record = pd.read_csv(M1_LIST, dtype=str)
    rel = str(interface.relative_to(ROOT))
    matched = record.loc[record["relative_path"] == rel, "sha256"]
    require(len(matched) == 1, "M1 interface absent from freeze manifest")
    require(hashlib.sha256(interface.read_bytes()).hexdigest() == matched.iloc[0], "M1 interface hash drift")

    dates = pd.to_datetime(pd.read_csv(interface, usecols=["analysis_date"])["analysis_date"], errors="raise")
    require(len(dates) > 0 and dates.notna().all() and not dates.duplicated().any(), "M1 dates invalid")
    first = dates.min().date().isoformat()
    spec = pd.DataFrame(TARGETS, columns=FIELDS)
    qa = {
        "m1_v1_0_frozen": True, "m1_v1_0_qa_pass": True,
        "interface_row_count": len(dates), "interface_duplicate_date_count": 0,
        "first_m1_state_date": first, "target_count": len(spec),
        "primary_target_count": int((spec.tier == "PRIMARY").sum()),
        "derived_event_count": int((spec.tier == "DERIVED_EVENT").sum()),
        "all_horizons_positive": bool((spec.horizon_td > 0).all()),
        "target_definition_missing_count": int(spec.definition.eq("").sum()),
        "m1_state_used_to_define_target": False, "m4_outcome_computed": False,
        "target_selected_from_results": False, "threshold_selected_from_results": False,
    }
    qa["all_formal_qa_pass"] = bool(
        qa["primary_target_count"] == 1 and qa["derived_event_count"] == 2
        and qa["all_horizons_positive"] and qa["target_definition_missing_count"] == 0
        and not spec.target_id.duplicated().any() and not spec.target_name.duplicated().any()
    )
    require(qa["all_formal_qa_pass"], "Target specification QA failed")
    metadata = {
        "research_day": 14, "step": "Step1_M4_Tail_Risk_Target_Specification_Freeze",
        "status": "FROZEN", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "m1_version": "M1_v1.0", "m1_interface": str(interface),
        "primary_market_proxy": "CSI300", "robustness_market_proxies": ["CSI500", "SSE_COMPOSITE"],
        "primary_target": "fwd_20d_max_drawdown_loss", "primary_horizon_td": 20,
        "tail_event_threshold_rule": (
            "Use only pre-M1 calibration observations whose full forward label is realized before "
            f"the first M1 state date ({first}). Freeze both P90 and P95 thresholds before M1-M4 validation."
        ),
        "timing_rule": (
            "All target returns must occur strictly after the formation date. Future target "
            "information must never enter M1 state construction or target selection."
        ),
        "data_source_resolution": "DEFERRED_TO_DAY14_STEP2",
        "selection_policy": (
            "No target, horizon, market proxy, or tail threshold may be selected using "
            "M1 Risk-State validation results."
        ),
        "m4_outcome_computed": False, "formal_qa": qa,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [SPEC, QA, META]
    if any(p.exists() for p in paths):
        require(all(p.exists() for p in paths), "Incomplete existing freeze; manual audit required")
        old_spec = pd.read_csv(SPEC, dtype=str).fillna("")
        new_spec = spec.fillna("").astype(str)
        require(old_spec.equals(new_spec), "Frozen target specification changed")
        old_qa = dict(zip(*[pd.read_csv(QA, dtype=str)[c] for c in ["qa_name", "qa_value"]]))
        require(old_qa == {k: str(v) for k, v in qa.items()}, "Frozen QA changed")
        old_meta = json.loads(META.read_text(encoding="utf-8"))
        require(all(old_meta.get(k) == v for k, v in metadata.items() if k != "created_utc"),
                "Frozen metadata changed")
        print(f"Step1 verified, existing freeze unchanged: {len(spec)} targets; M1 dates={len(dates)}")
        return
    # Write metadata last; incomplete files on interruption are never silently repaired.
    spec.to_csv(SPEC, index=False, encoding="utf-8-sig")
    pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"]).to_csv(QA, index=False, encoding="utf-8-sig")
    write_json(META, metadata)
    print(f"Step1 FROZEN / QA PASS: {len(spec)} targets; M1 dates={len(dates)}; {OUT}")


if __name__ == "__main__":
    main()

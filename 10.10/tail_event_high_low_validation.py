"""Day14 Step5 | Frozen M1-M4 tail-event hit rate and HIGH-LOW validation.
Run: python day14_step5_tail_event_high_low_validation.py --root D:\\M1_StockNetwork
Dependencies: pandas, numpy, statsmodels. No change to upstream freezes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

MARKETS = ("CSI300", "CSI500", "SSE_COMPOSITE")
STATES = ("LOW", "MID", "HIGH")
LAGS = {5: 1, 20: 2, 60: 4}  # Same fixed HAC convention as Step4


def require(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metadata(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verified(path, expected):
    require(path.is_file() and sha(path) == expected, f"SHA256 mismatch: {path}")


def write_csv(frame, path):
    tmp = Path(str(path) + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.15g")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\M1_StockNetwork")
    root = Path(parser.parse_args().root)
    day = root / "output" / "M1_day14"
    s1 = day / "01_step1_m4_tail_risk_target_specification_freeze"
    s2 = day / "02_step2_m4_forward_tail_risk_labels"
    s3 = day / "03_step3_m1_m4_pit_alignment_summary"
    s4 = day / "04_step4_continuous_risk_state_validation"
    audit = day / "04a_step4_code_version_audit" / "step4_code_version_audit.json"
    out = day / "05_step5_tail_event_high_low_validation"

    spec_path = s1 / "m4_tail_risk_target_specification.csv"
    panel_path = s3 / "m1_m4_pit_aligned_panel.csv"
    contrast_path = s3 / "m1_m4_high_low_contrast.csv"
    calib_path = s2 / "m4_tail_event_calibration.csv"
    m2p, m3p, m4p = [s / f"day14_step{k}_metadata.json" for k, s in ((2, s2), (3, s3), (4, s4))]
    for p in (m2p, m3p, m4p, audit):
        require(p.is_file(), f"Missing frozen audit: {p}")
    m2, m3, m4, a = metadata(m2p), metadata(m3p), metadata(m4p), metadata(audit)
    for m in (m2, m3, m4):
        require(m.get("m1_version") == "M1_v1.0" and m.get("status") == "FROZEN"
                and m.get("formal_qa", {}).get("all_formal_qa_pass") is True, "Upstream QA invalid")
    require(a.get("status") == "PASS" and a.get("script_exact_match") is True
            and a.get("frozen_outputs_match") is True and a.get("frozen_inputs_match") is True,
            "Step4 code-version audit must be PASS")

    # Check the exact frozen source bytes; do not re-estimate labels or states.
    for p in (panel_path, contrast_path):
        verified(p, m3["output_sha256"][p.name])
    verified(calib_path, m2["output_sha256"][calib_path.name])
    verified(spec_path, next(v for k, v in m3["input_sha256"].items()
                             if k.replace("\\", "/").endswith("/" + spec_path.name)))
    for name, digest in m4["output_sha256"].items():
        verified(s4 / name, digest)
    for p in (m2p, m3p):
        expected = m4["input_sha256"] if p == m3p else m3["input_sha256"]
        verified(p, next(v for k, v in expected.items()
                         if k.replace("\\", "/").endswith("/" + p.name)))
    inputs = (spec_path, panel_path, contrast_path, calib_path, m2p, m3p, m4p, audit,
              *(s4 / name for name in m4["output_sha256"]))
    signatures = {str(p): sha(p) for p in inputs}
    names = ("m1_m4_high_low_hac_validation.csv", "m1_m4_tail_event_by_state.csv",
             "m1_m4_tail_event_hit_rate.csv", "day14_step5_qa.csv")
    files = [out / n for n in names]
    meta_file = out / "day14_step5_metadata.json"
    if any(p.exists() for p in (*files, meta_file)):
        require(all(p.is_file() for p in (*files, meta_file)), "Partial old Step5 output")
        old = metadata(meta_file)
        require(old.get("input_sha256") == signatures and old.get("script_sha256") == sha(Path(__file__))
                and old.get("output_sha256") == {p.name: sha(p) for p in files}
                and old.get("formal_qa", {}).get("all_formal_qa_pass") is True,
                "Step5 frozen artifact/code drift: refusing overwrite")
        print("Day14 Step5: existing freeze VERIFIED, unchanged")
        return

    spec = pd.read_csv(spec_path)
    cont = spec.loc[spec.tier != "DERIVED_EVENT"].copy()
    ev = spec.loc[spec.tier == "DERIVED_EVENT"].copy()
    require(len(cont) == 6 and len(ev) == 2 and list(ev.target_id) == ["T7", "T8"]
            and list(cont.target_id) == [f"T{k}" for k in range(1, 7)], "Target set drift")
    require(spec.set_index("target_id").loc["T1", "target_name"] == "fwd_20d_max_drawdown_loss",
            "Primary target drift")
    require((ev["threshold_rule"].astype(str).str.contains("PRE_M1_FULLY_REALIZED_CALIBRATION")).all(),
            "Event calibration rule drift")

    panel = pd.read_csv(panel_path, parse_dates=["analysis_date", "label_end_5d",
                                                   "label_end_20d", "label_end_60d"])
    require(len(panel) == 312 and not panel.duplicated(["market_proxy", "analysis_date"]).any()
            and set(panel.market_proxy) == set(MARKETS), "Unexpected aligned panel")
    panel["state"] = panel.m1_network_risk_state.str.replace("_NETWORK_RISK", "", regex=False)
    require(set(panel.state) == set(STATES) and panel.analysis_date.nunique() == 104,
            "Invalid M1 states")
    require(panel.groupby("analysis_date").state.nunique().eq(1).all(), "State differs by market")
    for h in (5, 20, 60):
        valid = panel[f"available_{h}d"].astype(bool)
        end = panel[f"label_end_{h}d"]
        require((end.notna() == valid).all()
                and (end.loc[valid] > panel.loc[valid, "analysis_date"]).all(),
                f"Forward-label timing error: {h}d")

    calib = pd.read_csv(calib_path)
    require(len(calib) == 3 and set(calib.market_proxy) == set(MARKETS)
            and (pd.to_datetime(calib.last_label_end_date)
                 < pd.to_datetime(calib.first_m1_state_date)).all(), "Invalid calibration")
    for m in MARKETS:
        d = panel.loc[panel.market_proxy == m]
        q = calib.set_index("market_proxy").loc[m]
        y = d.fwd_20d_max_drawdown_loss
        for p, col in ((90, "fwd_20d_tail_event_p90"), (95, "fwd_20d_tail_event_p95")):
            valid = y.notna()
            require((d[col].notna() == valid).all()
                    and (d.loc[valid, col].to_numpy() == (y.loc[valid] > q[f"q{p}"]).astype(int).to_numpy()).all(),
                    f"Event labels/calibration mismatch: {m} P{p}")

    # HIGH minus LOW: LOW is regression reference; MID included, not discarded.
    contrasts, event_state, hits = [], [], []
    frozen = pd.read_csv(contrast_path).set_index(["market_proxy", "target_id"])
    for m in MARKETS:
        d = panel.loc[panel.market_proxy == m].sort_values("analysis_date")
        for z in cont.itertuples(index=False):
            v = d.dropna(subset=[z.target_name]).copy()
            x = pd.DataFrame({"const": 1.0,
                              "MID": v.state.eq("MID").astype(float),
                              "HIGH": v.state.eq("HIGH").astype(float)})
            fit = sm.OLS(v[z.target_name].to_numpy(float), x.to_numpy()).fit(
                cov_type="HAC", cov_kwds={"maxlags": LAGS[int(z.horizon_td)]})
            ci = fit.conf_int()[2]
            means = v.groupby("state")[z.target_name].mean()
            diff = float(means["HIGH"] - means["LOW"])
            require(abs(diff - float(frozen.loc[(m, z.target_id), "high_minus_low"])) < 1e-12,
                    f"Step3 HIGH-LOW not reproduced: {m} {z.target_id}")
            require(abs(diff - fit.params[2]) < 1e-10, "HAC regression contrast mismatch")
            contrasts.append(dict(market_proxy=m, target_id=z.target_id, tier=z.tier,
                                  target_name=z.target_name, n_valid=len(v),
                                  n_low=int(v.state.eq("LOW").sum()), n_high=int(v.state.eq("HIGH").sum()),
                                  mean_low=float(means["LOW"]), mean_high=float(means["HIGH"]),
                                  high_minus_low=diff, hac_lag=LAGS[int(z.horizon_td)],
                                  hac_se=float(fit.bse[2]), hac_t=float(fit.tvalues[2]),
                                  hac_p_two_sided=float(fit.pvalues[2]),
                                  ci95_low=float(ci[0]), ci95_high=float(ci[1])))

        for p, col in ((90, "fwd_20d_tail_event_p90"), (95, "fwd_20d_tail_event_p95")):
            v = d.dropna(subset=[col]).copy()
            actual = v[col].astype(int)
            n_event = int(actual.sum())
            for state in STATES:
                mask = v.state.eq(state)
                n = int(mask.sum()); k = int(actual[mask].sum())
                event_state.append(dict(market_proxy=m, event=f"P{p}", state=state,
                                        n_valid=n, n_events=k, event_rate=k / n if n else np.nan))
            alarm = v.state.eq("HIGH")
            tp = int((alarm & actual.eq(1)).sum())
            fp = int((alarm & actual.eq(0)).sum())
            fn = int((~alarm & actual.eq(1)).sum())
            tn = int((~alarm & actual.eq(0)).sum())
            hits.append(dict(market_proxy=m, event=f"P{p}", n_valid=len(v),
                             n_events=n_event, n_high_alerts=int(alarm.sum()),
                             tp=tp, fp=fp, fn=fn, tn=tn,
                             precision=tp / (tp + fp) if tp + fp else np.nan,
                             recall=tp / n_event if n_event else np.nan,
                             false_alarm_rate=fp / (fp + tn) if fp + tn else np.nan,
                             base_event_rate=n_event / len(v),
                             inference_status="ZERO_EVENTS" if n_event == 0 else
                                              "SPARSE_DESCRIPTIVE_ONLY" if n_event < 10 else
                                              "DESCRIPTIVE_ONLY_OVERLAPPING_HORIZONS"))

    hl = pd.DataFrame(contrasts)
    hl["bh_q_18"] = multipletests(hl.hac_p_two_sided, method="fdr_bh")[1]
    states = pd.DataFrame(event_state)
    hit = pd.DataFrame(hits)
    qa = dict(m1_v1_0_and_step1_step4_verified=True, step4_code_audit_pass=True,
              aligned_rows=len(panel), market_count=len(MARKETS),
              continuous_contrasts=len(hl), event_state_rows=len(states), event_hit_rows=len(hit),
              exact_20d_event_calibration_reproduced=True, step3_high_low_reproduced=True,
              p95_subset_p90=bool((panel.fwd_20d_tail_event_p95.fillna(0)
                                      <= panel.fwd_20d_tail_event_p90.fillna(0)).all()),
              observed_p95_event_count=int(hit.loc[hit.event == "P95", "n_events"].sum()),
              no_target_or_threshold_selection=True, no_m1_reestimation=True)
    qa["all_formal_qa_pass"] = bool(len(hl) == 18 and len(states) == 18 and len(hit) == 6
                                  and hl.hac_p_two_sided.notna().all()
                                  and qa["p95_subset_p90"])
    require(qa["all_formal_qa_pass"], f"Step5 QA failed: {qa}")
    out.mkdir(parents=True, exist_ok=True)
    for p, df in zip(files, (hl, states, hit, pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"]))):
        write_csv(df, p)
    meta = dict(research_day=14, step="Step5_Tail_Event_Hit_Rate_High_Low_Validation",
                status="FROZEN", m1_version="M1_v1.0", primary_market="CSI300",
                primary_target="T1", hac_lags=LAGS,
                high_low_method="OLS with LOW reference, MID/HIGH indicators; Newey-West HAC",
                event_method="HIGH alert; descriptive precision/recall, no rare-event p-values",
                interpretation="Exploratory; sparse events; overlapping forward horizons; not causality",
                input_sha256=signatures, script_sha256=sha(Path(__file__)),
                output_sha256={p.name: sha(p) for p in files}, formal_qa=qa)
    tmp = Path(str(meta_file) + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, meta_file)
    row = hl.loc[(hl.market_proxy == "CSI300") & (hl.target_id == "T1")].iloc[0]
    print(f"Day14 Step5 FROZEN / QA PASS | HAC contrasts={len(hl)}, event summaries={len(hit)}")
    print(f"CSI300 T1: HIGH-LOW={row.high_minus_low:.5f}, HAC p={row.hac_p_two_sided:.4g}, BH q={row.bh_q_18:.4g}")
    print(f"P90 events by market: {dict(zip(hit.loc[hit.event == 'P90', 'market_proxy'], hit.loc[hit.event == 'P90', 'n_events']))}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()

r"""Day15 Step5-D: causal feasibility/robustness audit of frozen Step5-A/B/C.
Observational diagnostics only; NO causal effect estimate.
Run: python day15_step5d_causal_feasibility_audit.py --root D:\M1_StockNetwork
Requires numpy, pandas, scipy, statsmodels. No Parquet or network rebuild.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

HAC, MARGIN = 2, 0.005  # 0.5 pp PLACEBO equivalence margin: exploratory, not pre-registered
NETWORKS = ("B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE")
MEASURES = ("post_raw", "post_sector", "pre_raw", "pre_sector")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def csv_bytes(df):
    return df.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")


def hac(x):
    z = np.asarray(x, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 30 or np.std(z) < 1e-14:
        return dict(n_dates=len(z), mean=float(z.mean()) if len(z) else np.nan,
                    se=np.nan, p_two_sided=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    fit = sm.OLS(z, np.ones((len(z), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC})
    return dict(n_dates=len(z), mean=float(fit.params[0]), se=float(fit.bse[0]),
                p_two_sided=float(fit.pvalues[0]), ci95_low=float(fit.conf_int()[0, 0]),
                ci95_high=float(fit.conf_int()[0, 1]))


def verified_step(directory, meta_name, expected_step, inputs):
    mp = directory / meta_name
    require(mp.is_file(), f"Missing frozen metadata: {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    require(meta.get("status") == "FROZEN" and meta.get("step") == expected_step
            and meta.get("formal_qa", {}).get("all_formal_qa_pass") is True,
            f"Invalid upstream freeze / QA: {mp}")
    inputs[meta_name] = sha(mp)
    for name, digest in meta["output_sha256"].items():
        file = directory / name
        require(file.is_file() and sha(file) == digest, f"SHA256 mismatch: {file}")
        inputs[name] = digest
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    a = ap.parse_args()
    base = Path(a.root) / "output" / "M1_day15"
    da = base / "05_step5_temporal_order_matched_null"
    db = base / "05b_step5b_common_shock_placebo"
    dc = base / "05c_step5c_rolling_oos"
    out = base / "05d_step5d_causal_feasibility_audit"
    inputs = {}
    ma = verified_step(da, "day15_step5_metadata.json",
                       "Step5_Temporal_Ordering_Matched_Null", inputs)
    mb = verified_step(db, "day15_step5b_metadata.json",
                       "Step5B_Common_Shock_Controls_Placebo", inputs)
    mc = verified_step(dc, "day15_step5c_metadata.json",
                       "Step5C_Retrospective_Rolling_OOS", inputs)
    for source, earlier, newer in [
        ("day15_step5_metadata.json", ma, mb),
        ("day15_step5b_metadata.json", mb, mc)]:
        require(newer["input_sha256"].get(source) == inputs[source],
                f"Upstream hash chain broken: {source}")
    for parent, child in ((ma, mb), (mb, mc)):
        for filename, digest in parent["output_sha256"].items():
            require(child["input_sha256"].get(filename) == digest,
                    f"Upstream output hash chain broken: {filename}")
    require(mb.get("analyzed_windows") == [120] and mb.get("event_threshold") == -.05
            and mb.get("source_days") == [1, 10] and mb.get("target_followup") == [1, 5]
            and mb.get("pre_placebo_days") == 5 and mb.get("repeats") == 3,
            "Step5-B design mismatch")
    require(mc.get("purge_market_sessions") == 20 and mc.get("min_train_dates") == 36
            and mc["formal_qa"].get("all_prediction_training_purged") is True,
            "Step5-C purge / training mismatch")
    d = pd.read_csv(db / "temporal_common_shock_placebo_daily.csv", low_memory=False)
    s = pd.read_csv(db / "temporal_common_shock_placebo_hac.csv", low_memory=False)
    o = pd.read_csv(dc / "rolling_oos_score_summary.csv", low_memory=False)
    od = pd.read_csv(dc / "rolling_oos_daily.csv", low_memory=False)
    require(len(d) == 208 and not d.duplicated(["analysis_date", "network", "window"]).any()
            and set(d.network) == set(NETWORKS) and set(d.window) == {120},
            "Step5-B daily cells invalid")
    require(len(s) == 9 and len(o) == 16, "Unexpected predefined test counts")
    require(len(od) == 832 and (od.status == "OOS_EVALUATED").sum() == 520,
            "Step5-C prediction coverage drift")
    d["analysis_date"] = pd.to_datetime(d.analysis_date, errors="raise")
    for m in MEASURES:
        n, p, q = (pd.to_numeric(d[m + suffix], errors="coerce").to_numpy(float)
                   for suffix in ("_n", "_edge", "_null"))
        delta = pd.to_numeric(d[m + "_delta"], errors="coerce").to_numpy(float)
        good = n > 0
        require(np.isfinite(n).all() and (n >= 0).all()
                and ((p[good] >= 0) & (p[good] <= 1) & (q[good] >= 0) & (q[good] <= 1)).all()
                and np.allclose(p[good] - q[good], delta[good], atol=1e-8)
                and np.isnan(delta[~good]).all(), f"Invalid Step5-B metric: {m}")
    # Check the published HAC results against daily data, without adding new tests.
    for net in NETWORKS:
        sub = d[d.network == net].sort_values("analysis_date")
        require(len(sub) == 104, f"Incomplete dates: {net}")
        for m in MEASURES:
            row = s[(s.network == net) & (s.test == m.upper())]
            require(len(row) == 1, f"Missing frozen HAC row: {net}/{m}")
            calc = hac(sub[m + "_delta"])
            require(calc["n_dates"] == int(row.iloc[0].n_dates)
                    and np.isclose(calc["mean"], row.iloc[0].mean_delta, atol=2e-10)
                    and np.isclose(calc["p_two_sided"], row.iloc[0].p_two_sided, atol=2e-9),
                    f"HAC summary mismatch: {net}/{m}")
    # Descriptive early/late split, not an unseen OOS replication.
    periods = []
    for net in NETWORKS:
        sub = d[d.network == net].sort_values("analysis_date")
        for m in MEASURES:
            active = sub[sub[m + "_delta"].notna()].copy()
            require(len(active) == 102, f"Unexpected effective dates: {net}/{m}")
            for name, seg in (("FULL", active), ("EARLY_51", active.iloc[:51]),
                              ("LATE_51", active.iloc[51:])):
                est = hac(seg[m + "_delta"])
                periods.append(dict(network=net, measure=m.upper(), period=name,
                                    first_date=str(seg.analysis_date.iloc[0].date()),
                                    last_date=str(seg.analysis_date.iloc[-1].date()),
                                    mean_matched_share=float(seg[m + "_share"].mean()),
                                    mean_sampled_edge_share=float(seg.sampled_share.mean()),
                                    mean_match_n=float(seg[m + "_n"].mean()), **est))
    periods = pd.DataFrame(periods)
    # Negative-control equivalence: absence of significance != equivalence.
    # This is a NEW exploratory tolerance; it must not be called confirmatory.
    placebo = []
    for net in NETWORKS:
        for m in ("pre_sector", "pre_raw"):
            v = periods[(periods.network == net) & (periods.measure == m.upper())
                        & (periods.period == "FULL")].iloc[0]
            se = float(v.se)
            p_lo = norm.sf((v["mean"] + MARGIN) / se)
            p_hi = norm.cdf((v["mean"] - MARGIN) / se)
            p_tost = max(float(p_lo), float(p_hi))
            placebo.append(dict(network=net, measure=m.upper(), n_dates=int(v.n_dates),
                                mean_delta=float(v["mean"]), hac_se=se, tolerance=MARGIN,
                                p_tost=p_tost, equivalent_at_5pct=bool(p_tost < .05),
                                interpretation="exploratory only; different post/pre risk sets"))
    placebo = pd.DataFrame(placebo)
    # Link to the already frozen OOS scores; do not invent a new prediction test.
    require(set(o.metric) == {"POST_RAW", "POST_SECTOR"}, "OOS outcome mismatch")
    oos_main = o[(o.network == NETWORKS[1]) & (o.metric == "POST_SECTOR")]
    oos_bh_significant = int((o.bh_q_16_tests < .05).sum())
    require(len(oos_main) == 4 and (oos_main.n_oos_dates == 65).all(), "Main OOS rows invalid")
    robust_main = s[(s.network == NETWORKS[1]) & (s.test == "POST_SECTOR")].iloc[0]
    pre_main = s[(s.network == NETWORKS[1]) & (s.test == "PRE_SECTOR")].iloc[0]
    raw_comparator = s[(s.network == "RESIDUAL_MINUS_RAW") &
                       (s.test == "POST_SECTOR_CONTRAST")].iloc[0]
    # No source-shock exogeneity / IV / shock-level covariate design is available.
    checks = [
        ("time_order", "PARTIAL", "Source event precedes target event; undirected edges do not imply causation"),
        ("formation_PIT", "SUPPORTED_BY_UPSTREAM_QA", "Frozen edge and matching strata use formation-time information"),
        ("industry_common_shock", "PARTIAL", "Contemporaneous formation-industry leave-one-out mean; no market-beta or shock-specific controls"),
        ("size_vol_matching", "PARTIAL", "Sector + historical volatility quintile + size tercile; residual confounding possible"),
        ("overlap_positivity", "LIMITED", "Only matched sample, report coverage and sampled-edge proportion"),
        ("pre_event_negative_control", "PARTIAL", "PRE_SECTOR placebo exists but has a different risk set from POST_SECTOR"),
        ("no_pretrend_equivalence", "NOT_ESTABLISHED", "Zero-effect nonsignificance does not establish pre-event equivalence"),
        ("shock_exogeneity", "NOT_IDENTIFIED", "The -5% source-stock shock is endogenous; no external shock or instrument"),
        ("no_unobserved_confounding", "NOT_IDENTIFIED", "Unmeasured market-news and delayed common reactions remain"),
        ("interference_exclusion", "NOT_IDENTIFIED", "Same network contains multiple shared nodes and simultaneous shocks"),
        ("out_of_sample_gain", "NOT_ESTABLISHED", "16 historical walk-forward scoring tests; none pass BH q<0.05"),
        ("fresh_independent_holdout", "NOT_AVAILABLE", "All 104 formation dates were previously inspected"),
        ("causal_effect_identified", "NO", "Observational data do not meet a credible causal identification design")]
    checklist = pd.DataFrame(checks, columns=["criterion", "assessment", "reason"])
    main_periods = periods[(periods.network == NETWORKS[1]) &
                           (periods.measure == "POST_SECTOR")]
    pval = placebo[(placebo.network == NETWORKS[1]) &
                   (placebo.measure == "PRE_SECTOR")].iloc[0]
    qa = dict(upstream_a_b_c_frozen_sha256_pass=True, step5b_graph_cells=len(d),
              step5b_tests_checked=8, step5c_oos_rows=520, step5c_tests=16,
              diagnostics_rows=len(periods), placebo_equivalence_rows=len(placebo),
              identification_criteria=len(checklist), date_split_descriptive_only=True,
              pre_post_risk_sets_differ=True, causal_claim_prohibited=True,
              full_sample_post_sector_delta=float(robust_main.mean_delta),
              full_sample_pre_sector_delta=float(pre_main.mean_delta),
              residual_raw_post_sector_q=float(raw_comparator.bh_q_9_tests),
              placebo_pre_equivalent_at_margin=bool(pval.equivalent_at_5pct),
              oos_post_sector_any_bh_significant=bool((oos_main.bh_q_16_tests < .05).any()),
              causal_identification_status="NOT_IDENTIFIED")
    qa["all_formal_qa_pass"] = (len(periods) == 24 and len(placebo) == 4
                                and len(checklist) == 13)  # No significance-based QA gate
    require(qa["all_formal_qa_pass"], f"Formal QA blocked: {qa}")
    summary = (
        "# Day15 Step5-D — Causal Identification Feasibility Audit\n\n"
        "**Verdict: CAUSAL_IDENTIFICATION_NOT_ESTABLISHED.** This is a robustness audit, not a causal-effect estimate.\n\n"
        f"- Residual W120 POST_SECTOR association: {float(robust_main.mean_delta):.6f} "
        f"({100*float(robust_main.mean_delta):.3f} percentage points), "
        f"BH q={float(robust_main.bh_q_9_tests):.6g}.\n"
        f"- Residual W120 PRE_SECTOR placebo: {float(pre_main.mean_delta):.6f}, "
        f"BH q={float(pre_main.bh_q_9_tests):.6g}; exploratory +/-0.5 pp equivalence: "
        f"{'WITHIN EXPLORATORY TOLERANCE' if bool(pval.equivalent_at_5pct) else 'NOT ESTABLISHED'} "
        f"(TOST p={float(pval.p_tost):.4g}).\n"
        f"- Residual–Raw POST_SECTOR contrast BH q={float(raw_comparator.bh_q_9_tests):.6g}.\n"
        f"- First 51 dates POST_SECTOR mean: {float(main_periods[main_periods.period=='EARLY_51']['mean'].iloc[0]):.6f}; "
        f"last 51: {float(main_periods[main_periods.period=='LATE_51']['mean'].iloc[0]):.6f}; "
        "this retrospective split is NOT independent out-of-sample confirmation.\n"
        f"- Step5-C: {oos_bh_significant}/16 walk-forward forecast tests pass BH 5%; fresh untouched holdout unavailable.\n"
        "- Source shocks are endogenous, market/news common shocks and interference remain, "
        "and post/pre have different risk sets. Do not subtract them as a causal DiD estimator.\n\n"
        "**Next research requirement:** externally justified stock-specific shocks, suitable untreated controls, "
        "credible exclusion/exchangeability arguments, and genuinely unseen validation dates.\n"
    )
    outputs = {"causal_robustness_periods.csv": csv_bytes(periods),
               "pre_event_placebo_equivalence.csv": csv_bytes(placebo),
               "causal_identification_checklist.csv": csv_bytes(checklist),
               "day15_step5d_qa.csv": csv_bytes(pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"])),
               "day15_step5d_conclusion.md": summary.encode("utf-8")}
    current_script_sha = sha(Path(__file__))
    metadata = dict(research_day=15, step="Step5D_Causal_Feasibility_Robustness_Audit",
                    status="FROZEN", conclusion="CAUSAL_IDENTIFICATION_NOT_ESTABLISHED",
                    no_causal_effect_estimate=True, networks=list(NETWORKS), window=120,
                    hac_lags=HAC, placebo_equivalence_margin_exploratory=MARGIN,
                    split="first 51 vs last 51 valid dates; descriptive only",
                    source="previously frozen Step5-A/B/C aggregated outputs",
                    input_sha256=inputs, script_sha256=current_script_sha,
                    output_sha256={k:hashlib.sha256(v).hexdigest() for k,v in outputs.items()},
                    formal_qa=qa)
    mp = out / "day15_step5d_metadata.json"
    if out.exists() and any(out.iterdir()):
        require(mp.is_file(), "Partial output exists; refusing overwrite")
        prior = json.loads(mp.read_text(encoding="utf-8"))
        require(prior.get("status") == "FROZEN" and prior.get("script_sha256") == current_script_sha
                and prior.get("input_sha256") == inputs, "Existing freeze code/input drift")
        for name, digest in prior["output_sha256"].items():
            require((out / name).is_file() and sha(out / name) == digest,
                    f"Existing frozen output drift: {name}")
        print("Day15 Step5-D: EXISTING FROZEN OUTPUT VERIFIED; no overwrite")
        return
    out.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        (out / name).write_bytes(value)  # byte writes avoid Windows newline conversion
    mp.write_bytes((json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print("Day15 Step5-D: FROZEN / PASS | CAUSAL IDENTIFICATION NOT ESTABLISHED")
    print(f"Residual W120 adjusted association: {100*float(robust_main.mean_delta):.3f} pp")
    print(f"Main placebo equivalence p: {float(pval.p_tost):.4g}")
    print("Saved:", out)


if __name__ == "__main__":
    main()

"""Day16 Step5: PIT structural attribution + hypothetical M1/M4 portfolio interface.
Run: python day16_step5_incremental_risk_attribution_m4_interface.py --root D:\\M1_StockNetwork
Needs: numpy, pandas, pyarrow. No forward labels / predictive or causal claims.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW, RES = "B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE"
PORTS = ("SYNTHETIC_HIGH_RESIDUAL_NEIGHBOR_RISK", "SECTOR_BALANCED_ID_CONTROL")
COVER_MIN = 0.8
COLS = ["analysis_date", "network", "window", "master_index", "security_id",
        "industry_id1", "hist_valid_sessions", "hist_downside_vol_60d", "degree",
        "neighbor_valid_weight_share", "neighbor_downside_exposure", "connection_hhi",
        "cross_industry_weight_share", "cross_industry_downside_exposure", "risk_status"]


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen(directory, name, qa_key):
    file = directory / name
    m = json.loads(file.read_text(encoding="utf-8"))
    need(m.get("status") == "FROZEN" and m.get("formal_qa", {}).get(qa_key) is True,
         f"Unfrozen / QA fail: {file}")
    for n, digest in m["output_sha256"].items():
        need(sha(directory / n) == digest, f"Frozen bytes changed: {directory / n}")
    return m


def unique(df, keys, n, label):
    need(len(df) == n and not df.duplicated(keys).any(), f"{label}: rows/keys mismatch")


def wmean(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.dot(v[ok], w[ok]) / w[ok].sum()) if np.any(ok) else np.nan


def risk_group(f, date, net, port, weight_map):
    ids = f.master_index.to_numpy(np.int64)
    ix = pd.Index(ids)
    need(ix.is_unique, "Duplicate node in frozen risk panel")
    idx = ix.get_indexer(weight_map.master_index.to_numpy(np.int64))
    need((idx >= 0).all(), f"Holdings absent from network: {date}/{port}/{net}")
    a = f.iloc[idx].copy()
    w = weight_map.weight.to_numpy(float)
    own = a.hist_downside_vol_60d.to_numpy(float)
    nbr = a.neighbor_downside_exposure.to_numpy(float)
    valid = np.isfinite(own) & np.isfinite(nbr)
    industry = a.industry_id1.to_numpy(np.int64)
    need(np.array_equal(industry, weight_map.industry_id1.to_numpy(np.int64)),
         "Holding PIT industry does not match frozen risk node")
    own_cover = float(w[np.isfinite(own)].sum())
    nbr_cover = float(w[valid].sum())
    sectors = pd.Series(w).groupby(industry).sum().to_numpy(float)
    baseline = dict(holding_hhi=float(np.square(w).sum()),
        sector_hhi=float(np.square(sectors).sum()), sector_count=len(sectors),
        own_hist_downside_vol_60d=wmean(own, w), own_risk_weight_coverage=own_cover)
    metrics = dict(analysis_date=date, network=net, window=120, portfolio=port,
        **baseline, neighbor_risk_weight_coverage=nbr_cover,
        neighbor_downside_exposure=wmean(nbr[valid], w[valid]),
        own_risk_same_covered_stocks=wmean(own[valid], w[valid]),
        connection_hhi_covered=wmean(a.connection_hhi.to_numpy(float)[valid], w[valid]),
        cross_industry_weight_share_covered=wmean(
            a.cross_industry_weight_share.to_numpy(float)[valid], w[valid]),
        cross_industry_downside_exposure_covered=wmean(
            a.cross_industry_downside_exposure.to_numpy(float)[valid], w[valid]),
        source="FROZEN_STEP2_FORMATION_CLOSE", portfolio_source="SYNTHETIC_NOT_REAL")
    metrics["neighbor_minus_own_matched"] = (
        metrics["neighbor_downside_exposure"]-metrics["own_risk_same_covered_stocks"]
        if nbr_cover >= COVER_MIN else np.nan)
    metrics["m4_exposure_status"] = ("PIT_EXPOSURE_READY" if
        own_cover >= COVER_MIN and nbr_cover >= COVER_MIN else "PARTIAL_COVERAGE")
    a.insert(0, "portfolio", port)
    a.insert(0, "weight", w)
    a["real_holdings"] = False
    return metrics, a


def build(root):
    base = root / "output/M1_day16"
    d2 = base / "02_step2_network_risk_exposure_factors"
    d3 = base / "03_step3_network_concentration_crowding"
    d4 = base / "04_step4_network_stress_scenarios"
    out = base / "05_step5_incremental_risk_attribution_m4_interface"
    m2 = frozen(d2, "day16_step2_metadata.json", "formal_qa_pass")
    m3 = frozen(d3, "day16_step3_metadata.json", "all_formal_qa_pass")
    m4 = frozen(d4, "day16_step4_metadata.json", "all_formal_qa_pass")
    names = ["day16_step2_metadata.json", "day16_step3_metadata.json", "day16_step4_metadata.json"]
    dirs = [d2, d3, d4]
    inputs = {n: sha(d / n) for d, n in zip(dirs, names)}
    need(m3["input_sha256"][names[0]] == inputs[names[0]] and
         m4["input_sha256"][names[0]] == inputs[names[0]] and
         m4["input_sha256"][names[1]] == inputs[names[1]], "Broken Step2->3->4 chain")
    need(not m2["formal_qa"]["future_labels_used"] and
         not m3["formal_qa"]["future_labels_used"] and
         m4["formal_qa"]["no_future_labels"] is True and
         not m4["formal_qa"]["actual_portfolios"], "PIT / synthetic prerequisites violated")
    script_sha = sha(Path(__file__))
    if out.exists():
        old = frozen(out, "day16_step5_metadata.json", "all_formal_qa_pass")
        need(old["script_sha256"] == script_sha and old["input_sha256"] == inputs,
             "Step5 frozen version / upstream drift")
        print("Day16 Step5: FROZEN RECHECK_PASS")
        return

    holdings = pd.read_csv(d4 / "stress_portfolio_holdings.csv")
    stress = pd.read_csv(d4 / "network_stress_daily.csv")
    contrast = pd.read_csv(d4 / "network_stress_contrast.csv")
    prior = pd.read_csv(d3 / "synthetic_portfolio_crowding.csv")
    sm = pd.read_csv(d2 / "network_risk_exposure_daily_summary.csv")
    unique(holdings, ["analysis_date", "portfolio", "master_index"], 6240, "Holdings")
    unique(stress, ["analysis_date", "network", "portfolio", "shock_scenario", "gamma"],
           4992, "Stress")
    unique(contrast, ["analysis_date", "portfolio", "shock_scenario", "gamma"],
           2496, "Contrast")
    unique(prior, ["analysis_date", "network"], 208, "Step3 portfolio")
    unique(sm, ["analysis_date", "network", "window"], 624, "Step2 summary")
    dates = sorted(sm.analysis_date.unique())
    need(len(dates) == 104 and set(stress.analysis_date) == set(dates) and
         set(holdings.analysis_date) == set(dates) and
         set(stress.network) == {RAW, RES} and
         set(holdings.portfolio) == set(PORTS), "Date/network/portfolio drift")
    need(not holdings.real_holdings.astype(bool).any() and
         not stress.real_holdings.astype(bool).any() and
         stress.hypothetical_only.astype(bool).all(), "Real holdings or non-hypothetical claims")
    need(stress.gamma.eq(0).sum() == 1248 and
         (stress.loc[stress.gamma == 0, "incremental_loss"].abs() < 1e-10).all(),
         "Gamma=0 baseline mismatch")
    pf = pq.ParquetFile(d2 / "network_risk_exposure_panel.parquet")
    need(pf.num_row_groups == len(sm) and set(COLS).issubset(pf.schema_arrow.names),
         "Frozen row group / schema mismatch")
    groups = holdings.groupby(["analysis_date", "portfolio"], sort=False)
    need(len(groups) == 208, "Holding groups missing")
    for (_, _), h in groups:
        need(len(h) == 30 and h.master_index.nunique() == 30 and
             np.isfinite(h.weight.to_numpy(float)).all() and (h.weight > 0).all() and
             abs(h.weight.sum()-1) < 1e-9 and h.industry_id1.nunique() == 15,
             "Synthetic 30-name, 15-sector portfolio changed")

    records, stocks = [], []
    selected = sm.loc[sm.window == 120].reset_index()
    need(len(selected) == 208, "Expected 208 W120 rows")
    for r in selected.itertuples(index=False):
        # Original row group order equals frozen Step2 summary row order.
        frame = pf.read_row_group(int(r.index), columns=COLS).to_pandas()
        need(len(frame) == int(r.nodes) and frame.master_index.is_unique and
             (frame.analysis_date.astype(str) == str(r.analysis_date)).all() and
             (frame.network == r.network).all() and
             (frame.window == 120).all(), "Parquet row-group identity drift")
        for port in PORTS:
            h = groups.get_group((r.analysis_date, port))
            metrics, sample = risk_group(frame, r.analysis_date, r.network, port, h)
            records.append(metrics)
            stocks.append(sample)
    ag = pd.DataFrame(records)
    stock = pd.concat(stocks, ignore_index=True)
    unique(ag, ["analysis_date", "network", "portfolio"], 416, "PIT group")
    unique(stock, ["analysis_date", "network", "portfolio", "master_index"], 12480,
           "Stock M4 interface")
    p = ag.pivot(index=["analysis_date", "portfolio"], columns="network",
                 values=["holding_hhi", "sector_hhi", "own_hist_downside_vol_60d"])
    for key in ("holding_hhi", "sector_hhi", "own_hist_downside_vol_60d"):
        need(np.allclose(p[(key, RAW)], p[(key, RES)], atol=1e-12,
                         rtol=0, equal_nan=True), f"Traditional baseline changed: {key}")
    need(np.allclose(ag.holding_hhi, 1/30, atol=1e-10) and
         np.allclose(ag.sector_hhi, 1/15, atol=1e-10),
         "Synthetic HHI invariants failed")
    joined = stress.merge(ag, on=["analysis_date", "network", "portfolio"],
                          how="left", validate="many_to_one", indicator=True)
    need((joined._merge == "both").all(), "Stress/risk interface join missing")
    joined = joined.drop(columns="_merge")
    # Step3 internal density available only for its own selected portfolio.
    density = prior[["analysis_date", "network", "selected_pair_density",
                     "sector_matched_null_mean", "selected_minus_null"]].copy()
    density["portfolio"] = PORTS[0]
    joined = joined.merge(density, how="left", on=["analysis_date", "network", "portfolio"],
                          validate="many_to_one")
    need(joined.loc[joined.portfolio == PORTS[1], "selected_pair_density"].isna().all(),
         "Control portfolio density must be unavailable, not invented")
    joined["network_scenario_incremental_per_5pct_shock"] = joined.incremental_loss / 0.05
    joined["m4_interface_type"] = "PIT_STRUCTURE_PLUS_HYPOTHETICAL_SCENARIO"
    joined["m4_prediction_validated"] = False
    joined["m4_causal_identified"] = False
    unique(joined, ["analysis_date", "network", "portfolio", "shock_scenario", "gamma"],
           4992, "M4 scenario interface")
    diff = joined.pivot(index=["analysis_date", "portfolio", "shock_scenario", "gamma"],
                        columns="network", values="stress_loss")
    chk = diff[RES].sub(diff[RAW]).rename("residual_minus_raw_loss").reset_index()
    chk = chk.merge(contrast, on=["analysis_date", "portfolio", "shock_scenario", "gamma"],
                    suffixes=("_recomputed", "_frozen"), validate="one_to_one")
    need(len(chk) == 2496 and np.allclose(chk.residual_minus_raw_loss_recomputed,
         chk.residual_minus_raw_loss_frozen, rtol=0, atol=2e-12) and
         (chk.loc[chk.gamma == 0, "residual_minus_raw_loss_recomputed"].abs() < 1e-12).all(),
         "Step4 Raw/Residual stress contrast changed")
    delta = ag.pivot(index=["analysis_date", "portfolio"], columns="network",
                     values="neighbor_downside_exposure")
    delta = (delta[RES]-delta[RAW]).rename("neighbor_risk_res_minus_raw").reset_index()
    chk = chk.merge(delta, on=["analysis_date", "portfolio"], how="left", validate="many_to_one")
    chk = chk.drop(columns="residual_minus_raw_loss_frozen").rename(
        columns={"residual_minus_raw_loss_recomputed": "residual_minus_raw_loss"})
    qa = dict(all_formal_qa_pass=True, formation_dates=104, risk_group_rows=416,
              m4_stock_interface_rows=len(stock), m4_scenario_interface_rows=len(joined),
              paired_attribution_rows=len(chk), synthetic_portfolio_groups=len(groups),
              downstream_qa_chain_pass=True, gamma_zero_equal=True,
              same_stocks_and_traditional_hhi_across_networks=True,
              input_step4_stress_unchanged=True, traditional_hhi_constant_by_design=True,
              actual_holdings=False, predictions_tested=False, causal_identification=False,
              stock_factors_formation_close_only=True,
              partial_coverage_groups=int((ag.m4_exposure_status != "PIT_EXPOSURE_READY").sum()))
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="day16_step5_", dir=out.parent))
    try:
        files = {"m1_m4_stock_risk_exposure.csv": stock,
                 "m1_m4_portfolio_risk_profile.csv": ag,
                 "m1_m4_hypothetical_stress_interface.csv": joined,
                 "incremental_structural_attribution.csv": chk,
                 "day16_step5_qa.csv": pd.DataFrame(qa.items(), columns=["metric", "value"])}
        for name, df in files.items():
            (temp / name).write_bytes(df.to_csv(index=False, lineterminator="\n",
                float_format="%.12g").encode("utf-8"))
        meta = dict(research_day=16, step="Step5_Incremental_Structural_Risk_Attribution_M4_Interface",
                    status="FROZEN", formal_qa=qa, input_sha256=inputs,
                    script_sha256=script_sha,
                    output_sha256={k: sha(temp / k) for k in files},
                    method="Frozen PIT 60d risk plus hypothetical W120 gamma stress; no refit",
                    comparator="Same 30 stocks and matching sector/holding HHI; structural differences only",
                    interpretation="SIMULATED portfolio, no verified risk prediction or causal propagation",
                    m4_ready_policy=f"neighbor / own weighted coverage >= {COVER_MIN}")
        (temp / "day16_step5_metadata.json").write_bytes((json.dumps(
            meta, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode("utf-8"))
        need(not out.exists(), "Refuse overwrite of existing Step5 output")
        temp.rename(out)
        print(f"Day16 Step5: FROZEN / PASS | {len(stock)} stock | {len(joined)} M4 scenarios")
        print(f"PIT group partial coverage: {qa['partial_coverage_groups']}/{len(ag)}")
    finally:
        if temp.exists():
            shutil.rmtree(temp)


if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--root", default=r"D:\M1_StockNetwork")
        build(Path(ap.parse_args().root).resolve())
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, IndexError) as e:
        sys.exit(f"Day16 Step5 BLOCKED: {e}")

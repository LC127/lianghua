"""Day16 Step3: frozen-network concentration / crowding PROXIES, not actual holdings.
Run: python day16_step3_network_concentration_crowding.py --root D:\\M1_StockNetwork
Requires numpy, pandas, pyarrow. Reads frozen Step2 panel and W120 edge files.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW, RES = "B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE"
SEED, SECTORS, PER_SECTOR, REPS = 20261014, 15, 2, 50
COLS = ["analysis_date", "network", "window", "master_index", "industry_id1",
        "hist_downside_vol_60d", "degree", "neighbor_downside_exposure",
        "connection_hhi", "cross_industry_weight_share", "cross_industry_downside_exposure"]


def need(test, msg):
    if not test:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate(root, s):
    p = (root / str(s).replace("\\", "/")).resolve()
    need(p.is_relative_to(root), f"External path forbidden: {s}")
    return p


def gini(v):
    x = np.sort(np.asarray(v, dtype=np.float64))
    total = x.sum()
    return float((2 * np.dot(np.arange(1, len(x)+1), x) / (len(x)*total)
                  - (len(x)+1)/len(x)) if total else 0.)


def mean(v):
    x = np.asarray(v, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


def analyze(f, expected):
    n = len(f)
    d = f.degree.to_numpy(dtype=np.int64)
    h = f.connection_hhi.to_numpy(float)
    ne = f.neighbor_downside_exposure.to_numpy(float)
    ce = f.cross_industry_downside_exposure.to_numpy(float)
    cs = f.cross_industry_weight_share.to_numpy(float)
    risk = f.hist_downside_vol_60d.to_numpy(float)
    need(n == int(expected.nodes) and len(np.unique(f.master_index)) == n,
         "Node count / duplicate master_index")
    need((d >= 0).all() and int(d.sum()) == 2 * int(expected.edges), "Degree-edge identity")
    need(int((d == 0).sum()) == int(expected.isolated_nodes), "Isolate count mismatch")
    need(int(np.isfinite(risk).sum()) == int(expected.risk_valid) and
         int(np.isfinite(ne).sum()) == int(expected.neighbor_valid) and
         int(np.isfinite(ce).sum()) == int(expected.cross_valid), "Risk coverage mismatch")
    need(np.all(np.isnan(h[d == 0])) and np.isfinite(h[d > 0]).all() and
         np.all((h[d > 0] > 0) & (h[d > 0] <= 1+1e-9)), "HHI invalid")
    need(np.all(np.isnan(cs[d == 0])) and
         np.all((cs[np.isfinite(cs)] >= 0) & (cs[np.isfinite(cs)] <= 1+1e-9)), "Cross-sector share invalid")
    need(np.all(risk[np.isfinite(risk)] >= 0) and
         np.all(ne[np.isfinite(ne)] >= 0) and np.all(ce[np.isfinite(ce)] >= 0), "Risk negative")
    k = max(1, int(math.ceil(0.10*n)))
    return dict(analysis_date=expected.analysis_date, network=expected.network,
        window=int(expected.window), nodes=n, edges=int(expected.edges),
        isolated_share=float(np.mean(d == 0)), connected_nodes=int((d > 0).sum()),
        degree_gini=gini(d), top10_degree_share=float(np.sort(d)[-k:].sum()/max(1, d.sum())),
        avg_degree=float(d.mean()), hhi_connected_mean=mean(h),
        effective_neighbors_median=float(np.median(1/h[np.isfinite(h)])) if np.isfinite(h).any() else np.nan,
        cross_sector_share_connected=mean(cs[d > 0]),
        neighbor_risk_valid_share=float(np.isfinite(ne).mean()),
        neighbor_risk_mean=mean(ne), own_hist_risk_mean=mean(risk))


def paired(raw, res, date, window):
    a, b = (t.sort_values("master_index").reset_index(drop=True) for t in (raw, res))
    need(np.array_equal(a.master_index, b.master_index) and
         np.array_equal(a.industry_id1, b.industry_id1), "Raw/Residual node universe differs")
    own_a, own_b = (t.hist_downside_vol_60d.to_numpy(float) for t in (a, b))
    need(np.allclose(own_a, own_b, equal_nan=True, rtol=0, atol=1e-12), "Common PIT risk differs")
    ne_a, ne_b = (t.neighbor_downside_exposure.to_numpy(float) for t in (a, b))
    h_a, h_b = (t.connection_hhi.to_numpy(float) for t in (a, b))
    common = np.isfinite(ne_a) & np.isfinite(ne_b)
    both_h = np.isfinite(h_a) & np.isfinite(h_b)
    return dict(analysis_date=date, window=int(window), nodes=len(a),
        both_neighbor_valid=int(common.sum()), both_connected=int(both_h.sum()),
        raw_neighbor_valid_share=float(np.isfinite(ne_a).mean()),
        residual_neighbor_valid_share=float(np.isfinite(ne_b).mean()),
        mean_neighbor_risk_res_minus_raw=mean((ne_b-ne_a)[common]),
        mean_hhi_res_minus_raw=mean((h_b-h_a)[both_h]),
        residual_minus_raw_isolated_share=float(np.mean(b.degree == 0)-np.mean(a.degree == 0)))


def selected_portfolio(raw, res):
    """PIT diagnostic: 2 highest residual neighbor-risk stocks in 15 largest sectors;
    eligible pool requires neighbor risk observed in BOTH networks."""
    a = raw.sort_values("master_index").reset_index(drop=True)
    b = res.sort_values("master_index").reset_index(drop=True)
    common = np.isfinite(a.neighbor_downside_exposure.to_numpy(float)) & \
             np.isfinite(b.neighbor_downside_exposure.to_numpy(float))
    g = b.loc[common & (b.industry_id1 >= 0)].copy()
    sectors = (g.groupby("industry_id1").size().loc[lambda x: x >= PER_SECTOR]
               .sort_values(ascending=False, kind="stable"))
    # Ties in sector sizes resolved deterministically by industry ID.
    sectors = sorted(sectors.index, key=lambda s: (-int(sectors.loc[s]), int(s)))[:SECTORS]
    need(len(sectors) == SECTORS, "Too few common-eligible industries for synthetic portfolio")
    pools, chosen = [], []
    for s in sectors:
        t = g.loc[g.industry_id1 == s].sort_values(
            ["neighbor_downside_exposure", "master_index"], ascending=[False, True])
        pool = t.master_index.to_numpy(np.int64)
        pools.append(pool)
        chosen.extend(pool[:PER_SECTOR])
    return np.array(chosen, dtype=np.int64), pools, int(len(g))


def connectivity(holdings, edges, base):
    x = np.asarray(holdings, dtype=np.int64)
    ii, jj = np.triu_indices(len(x), 1)
    u, v = np.minimum(x[ii], x[jj]), np.maximum(x[ii], x[jj])
    return float(sum((int(t) in edges) for t in (u*base+v)) / len(ii))


def synthetic(date, net, chosen, pools, eligible, edge_row, root):
    path = locate(root, edge_row.source)
    need(sha(path) == str(edge_row.sha256), f"Edge drift: {path}")
    t = pd.read_parquet(path, columns=["master_index_i", "master_index_j", "rank_fraction_common"])
    t = t.loc[pd.to_numeric(t.rank_fraction_common).to_numpy(float) <= 0.01+1e-12]
    need(len(t) == int(edge_row.selected_1pct_edges), "Frozen 1pct edge count differs")
    a, b = t.master_index_i.to_numpy(np.int64), t.master_index_j.to_numpy(np.int64)
    need((a >= 0).all() and (b >= 0).all() and (a != b).all(), "Invalid edge endpoints")
    base = int(max(a.max(), b.max(), max(v.max() for v in pools))) + 1
    keys = set((np.minimum(a, b)*base+np.maximum(a, b)).tolist())
    need(len(keys) == len(t), "Duplicate frozen edge pairs")
    obs = connectivity(chosen, keys, base)
    rng = np.random.default_rng(SEED + int(str(date).replace("-", "")))
    null = np.array([connectivity(np.concatenate([rng.choice(x, PER_SECTOR, replace=False)
                         for x in pools]), keys, base) for _ in range(REPS)])
    return dict(analysis_date=date, network=net, window=120,
        portfolio_type="SYNTHETIC_HIGH_RESIDUAL_NEIGHBOR_RISK",
        holding_count=len(chosen), sectors=SECTORS, common_eligible_nodes=eligible,
        selected_pair_density=obs, sector_matched_null_mean=float(null.mean()),
        selected_minus_null=float(obs-null.mean()), null_p95=float(np.quantile(null, .95)),
        null_repetitions=REPS, real_holdings=False, prediction_or_causality=False)


def run(root):
    prev = root / "output/M1_day16/02_step2_network_risk_exposure_factors"
    mpath = prev / "day16_step2_metadata.json"
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    need(meta["status"] == "FROZEN" and meta["formal_qa"]["formal_qa_pass"] and
         meta["formal_qa"]["network_date_cells"] == 624 and
         not meta["formal_qa"]["future_labels_used"], "Step2 source not formally frozen")
    for name, expected in meta["output_sha256"].items():
        need(sha(prev / name) == expected, f"Step2 output drift: {name}")
    step1 = root / "output/M1_day16/01_step1_frozen_network_portfolio_source_audit/source_only/day16_step1_metadata.json"
    need(sha(step1) == meta["input_sha256"][step1.name], "Step1 frozen source drift")
    manifest = root / "output/M1_day15/03_step3_network_edge_codistress/day15_step3_edge_manifest.csv"
    need(sha(manifest) == meta["input_sha256"][manifest.name], "Edge manifest drift")
    em = pd.read_csv(manifest).set_index(["analysis_date", "window", "network"])
    summary = pd.read_csv(prev / "network_risk_exposure_daily_summary.csv")
    need(len(summary) == 624 and summary.analysis_date.nunique() == 104 and
         not summary.duplicated(["analysis_date", "network", "window"]).any(), "Step2 daily keys invalid")
    out = root / "output/M1_day16/03_step3_network_concentration_crowding"
    inputs = {"day16_step2_metadata.json": sha(mpath),
              "day15_step3_edge_manifest.csv": sha(manifest)}
    script_hash = sha(Path(__file__))
    if out.exists():
        old = json.loads((out / "day16_step3_metadata.json").read_text(encoding="utf-8"))
        need(old["status"] == "FROZEN" and old["script_sha256"] == script_hash and
             old["input_sha256"] == inputs and
             all(sha(out / k) == v for k, v in old["output_sha256"].items()),
             "Step3 frozen drift: refuse overwrite")
        print("Day16 Step3: FROZEN RECHECK_PASS")
        return
    pf = pq.ParquetFile(prev / "network_risk_exposure_panel.parquet")
    need(pf.num_row_groups == 624 and set(COLS).issubset(pf.schema_arrow.names),
         "Step2 Parquet row groups / columns invalid")
    diagnostics, matched, portfolio = [], [], []
    raw = None
    for i, exp in enumerate(summary.itertuples(index=False)):
        f = pf.read_row_group(i, columns=COLS).to_pandas()
        need(len(f) and f.analysis_date.nunique() == 1 and f.network.nunique() == 1 and
             f.window.nunique() == 1 and str(f.analysis_date.iloc[0]) == str(exp.analysis_date) and
             f.network.iloc[0] == exp.network and int(f.window.iloc[0]) == exp.window,
             f"Row group/order mismatch #{i}")
        diagnostics.append(analyze(f, exp))
        if exp.network == RAW:
            raw = f
            continue
        need(raw is not None and raw.analysis_date.iloc[0] == exp.analysis_date and
             int(raw.window.iloc[0]) == int(exp.window), "Network pair order mismatch")
        matched.append(paired(raw, f, exp.analysis_date, exp.window))
        if exp.window == 120:
            chosen, pools, eligible = selected_portfolio(raw, f)
            for net in (RAW, RES):
                er = em.loc[(exp.analysis_date, 120, net)]
                portfolio.append(synthetic(exp.analysis_date, net, chosen, pools, eligible, er, root))
        raw = None
    need(raw is None and len(diagnostics) == 624 and len(matched) == 312 and
         len(portfolio) == 208, "Step3 coverage incomplete")
    stats = pd.DataFrame(diagnostics)
    pair = pd.DataFrame(matched)
    demo = pd.DataFrame(portfolio)
    need(((stats.isolated_share >= 0) & (stats.isolated_share <= 1)).all() and
         demo.selected_pair_density.between(0, 1).all() and
         demo.sector_matched_null_mean.between(0, 1).all(), "Concentration metric invalid")
    qa = dict(all_formal_qa_pass=True, network_date_cells=len(stats), paired_cells=len(pair),
              synthetic_diagnostic_cells=len(demo), formation_dates=int(stats.analysis_date.nunique()),
              common_stock_comparison=True, formation_close_pit=True, future_labels_used=False,
              actual_portfolio_holdings=False, real_institutional_crowding_measured=False,
              causal_or_predictive_claim=False, synthetic_sector_matched_reps=REPS,
              frozen_w120_edge_snapshots_rehashed=len(demo))
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(dir=out.parent, prefix="day16_step3_"))
    try:
        files = {"network_concentration_daily.csv": stats,
                 "raw_residual_common_sample.csv": pair,
                 "synthetic_portfolio_crowding.csv": demo,
                 "day16_step3_qa.csv": pd.DataFrame(qa.items(), columns=["metric", "value"])}
        for name, tab in files.items():
            (temp / name).write_bytes(tab.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8"))
        new_meta = dict(research_day=16, step="Step3_Network_Concentration_Crowding",
            status="FROZEN", formal_qa=qa, script_sha256=script_hash,
            input_sha256=inputs, output_sha256={x: sha(temp / x) for x in files},
            scope="Structural concentrations and sector-matched synthetic diagnostic only; no actual institution holdings",
            portfolio_selection="top 2 Residual neighbor-risk common-valid stocks in 15 largest sectors, at each close",
            comparator="50 seeded equal-sector random portfolios from same common-valid universe",
            graph_scope="624 cell node-factor panels; actual W120 Raw/Residual frozen edges")
        (temp / "day16_step3_metadata.json").write_bytes((json.dumps(new_meta,
                        sort_keys=True, ensure_ascii=False, indent=2)+"\n").encode("utf-8"))
        need(not out.exists(), "Output already exists; refuse replacement")
        temp.rename(out)
        print(f"Day16 Step3: FROZEN / PASS | {len(stats)} structural | {len(pair)} paired | {len(demo)} synthetic")
        print(f"Output: {out}")
    finally:
        if temp.exists():
            shutil.rmtree(temp)


if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser()
        p.add_argument("--root", default=r"D:\M1_StockNetwork")
        run(Path(p.parse_args().root).resolve())
    except (OSError, RuntimeError, ValueError, KeyError, IndexError, ImportError) as e:
        sys.exit(f"Day16 Step3 BLOCKED: {e}")

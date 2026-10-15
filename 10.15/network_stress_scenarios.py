"""Day16 Step4: frozen W120 network *hypothetical* stress, not causal contagion.
Run: python day16_step4_network_stress_scenarios.py --root D:\\M1_StockNetwork
Requires numpy, pandas, scipy, pyarrow. Reconstructs Step3 selected 30-stock portfolio.
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
from scipy.sparse import csr_matrix

RAW, RES = "B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE"
GAMMAS = (0.0, 0.1, 0.2, 0.3)
SCENARIOS = ("STOCK", "SECTOR", "RESIDUAL_EGO15")
PORTS = ("SYNTHETIC_HIGH_RESIDUAL_NEIGHBOR_RISK", "SECTOR_BALANCED_ID_CONTROL")
SHOCK, TOP_SECTORS, N_PER, EGO = 0.05, 15, 2, 15
COLS = ["analysis_date", "network", "window", "master_index", "industry_id1",
        "neighbor_downside_exposure"]


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def src(root, path):
    p = (root / str(path).replace("\\", "/")).resolve()
    require(p.is_relative_to(root), f"Path outside project: {path}")
    return p


def verified_outputs(directory, filename, frozen=True):
    m = json.loads((directory / filename).read_text(encoding="utf-8"))
    require(not frozen or m.get("status") == "FROZEN", f"Unfrozen: {filename}")
    for k, v in m["output_sha256"].items():
        require(digest(directory / k) == v, f"Frozen output drift: {directory / k}")
    return m


def portfolio(raw, res):
    """Reconstruct Step3 selection; independent control uses all-node sector sizes/IDs."""
    a, b = (x.sort_values("master_index").reset_index(drop=True) for x in (raw, res))
    require(np.array_equal(a.master_index, b.master_index) and
            np.array_equal(a.industry_id1, b.industry_id1), "Raw/Residual node mismatch")
    good = np.isfinite(a.neighbor_downside_exposure.to_numpy(float)) & \
           np.isfinite(b.neighbor_downside_exposure.to_numpy(float))
    eligible = b.loc[good & (b.industry_id1 >= 0)].copy()
    sizes = eligible.groupby("industry_id1").size()
    sectors = sorted((int(s) for s in sizes.index if sizes[s] >= N_PER),
                     key=lambda s: (-int(sizes[s]), s))[:TOP_SECTORS]
    require(len(sectors) == TOP_SECTORS, "Insufficient sectors for frozen portfolio")
    selected, control = [], []
    for s in sectors:
        t = eligible.loc[eligible.industry_id1 == s].sort_values(
            ["neighbor_downside_exposure", "master_index"], ascending=[False, True])
        selected.extend(t.master_index.to_numpy(np.int64)[:N_PER])
    all_sizes = b.loc[b.industry_id1 >= 0].groupby("industry_id1").size()
    all_sectors = sorted((int(s) for s in all_sizes.index if all_sizes[s] >= N_PER),
                         key=lambda s: (-int(all_sizes[s]), s))[:TOP_SECTORS]
    require(len(all_sectors) == TOP_SECTORS, "Too few industries for ID control")
    for s in all_sectors:
        pool = b.loc[b.industry_id1 == s, "master_index"].to_numpy(np.int64)
        control.extend(np.sort(pool)[:N_PER])
    p, q = np.array(selected, np.int64), np.array(control, np.int64)
    require(len(np.unique(p)) == 30 and len(np.unique(q)) == 30,
            "Duplicate synthetic portfolio holdings")
    return a, b, {PORTS[0]: p, PORTS[1]: q}, len(eligible)


def network(root, edge, node_ids):
    p = src(root, edge.source)
    require(digest(p) == str(edge.sha256), f"Edge SHA drift: {p}")
    f = pd.read_parquet(p, columns=["master_index_i", "master_index_j",
                                    "rank_fraction_common", "edge_weight"])
    f = f.loc[pd.to_numeric(f.rank_fraction_common).to_numpy(float) <= .01 + 1e-12]
    require(len(f) == int(edge.selected_1pct_edges), f"Selected edge count drift: {p}")
    i, j = (pd.Index(node_ids).get_indexer(f[k].to_numpy(np.int64))
            for k in ("master_index_i", "master_index_j"))
    w = pd.to_numeric(f.edge_weight, errors="coerce").to_numpy(float)
    n = len(node_ids)
    require((i >= 0).all() and (j >= 0).all() and (i != j).all() and
            np.isfinite(w).all() and (w > 0).all(), f"Invalid edges: {p}")
    key = np.minimum(i, j).astype(np.int64)*n + np.maximum(i, j)
    require(len(np.unique(key)) == len(key), f"Duplicate edge: {p}")
    u, v = np.r_[i, j], np.r_[j, i]
    weights = np.r_[w, w]
    totals = np.bincount(u, weights=weights, minlength=n)
    mat = csr_matrix((weights / totals[u], (u, v)), shape=(n, n))
    return mat, set(key.tolist()), (i, j, w)


def internal_density(holdings, node_idx, keys, n):
    loc = node_idx.get_indexer(holdings)
    require((loc >= 0).all(), "Holding outside frozen graph")
    ix, jx = np.triu_indices(len(loc), 1)
    return sum(int(k) in keys for k in
               (np.minimum(loc[ix], loc[jx])*n + np.maximum(loc[ix], loc[jx]))) / len(ix)


def shock_vectors(ids, sectors, anchor, res_edges):
    """Same three absolute shock vectors across Raw and Residual."""
    n = len(ids)
    idx = pd.Index(ids)
    anchor_local = idx.get_loc(anchor)
    sec = sectors[anchor_local]
    i, j, w = res_edges
    nbr = np.r_[j[i == anchor_local], i[j == anchor_local]]
    vals = np.r_[w[i == anchor_local], w[j == anchor_local]]
    order = np.lexsort((ids[nbr], -vals))
    cluster = np.r_[anchor_local, nbr[order][:EGO-1]]
    z = np.zeros((n, len(SCENARIOS)), float)
    z[anchor_local, 0] = SHOCK
    z[sectors == sec, 1] = SHOCK
    z[cluster, 2] = SHOCK
    return z, [int((z[:, k] > 0).sum()) for k in range(3)], sec, cluster


def propagate(p, z):
    """Neumann fixed point: x=z+gamma P x; row-substochastic, gamma<=0.3."""
    positive = np.array(GAMMAS[1:])
    gamma = np.repeat(positive, len(SCENARIOS))
    rhs = np.tile(z, (1, len(positive)))
    x = rhs.copy()
    for _ in range(40):
        new = rhs + (p @ x)*gamma[None, :]
        if np.max(np.abs(new-x)) < 1e-12:
            x = new
            break
        x = new
    require(np.max(np.abs(x-rhs-(p @ x)*gamma[None, :])) < 2e-11,
            "Stress fixed point residual too large")
    require(np.isfinite(x).all() and np.min(x) >= -1e-12 and
            np.max(x) <= SHOCK/(1-max(GAMMAS)) + 1e-10,
            "Stress scenario bound failed")
    return {0.0: z, **{float(g): x[:, k*3:(k+1)*3]
                      for k, g in enumerate(positive)}}


def build(root):
    d2 = root / "output/M1_day16/02_step2_network_risk_exposure_factors"
    d3 = root / "output/M1_day16/03_step3_network_concentration_crowding"
    m2 = verified_outputs(d2, "day16_step2_metadata.json")
    m3 = verified_outputs(d3, "day16_step3_metadata.json")
    require(m2["formal_qa"]["formal_qa_pass"] and
            m3["formal_qa"]["all_formal_qa_pass"] and
            m3["formal_qa"]["synthetic_diagnostic_cells"] == 208 and
            not m3["formal_qa"]["future_labels_used"], "Step2/3 QA not frozen")
    require(digest(d2 / "day16_step2_metadata.json") ==
            m3["input_sha256"]["day16_step2_metadata.json"], "Step2->Step3 drift")
    emfile = root / "output/M1_day15/03_step3_network_edge_codistress/day15_step3_edge_manifest.csv"
    require(digest(emfile) == m3["input_sha256"][emfile.name], "Edge manifest drift")
    em = pd.read_csv(emfile).set_index(["analysis_date", "window", "network"])
    sm = pd.read_csv(d2 / "network_risk_exposure_daily_summary.csv")
    prior = pd.read_csv(d3 / "synthetic_portfolio_crowding.csv").set_index(
        ["analysis_date", "network"])
    require(len(sm) == 624 and len(prior) == 208 and
            not sm.duplicated(["analysis_date", "network", "window"]).any(),
            "Source daily keys invalid")
    pf = pq.ParquetFile(d2 / "network_risk_exposure_panel.parquet")
    require(pf.num_row_groups == len(sm) and set(COLS).issubset(pf.schema_arrow.names),
            "Source Parquet structure changed")
    dates = sorted(sm.analysis_date.unique())
    require(len(dates) == 104, "Expected 104 frozen formation dates")
    output = root / "output/M1_day16/04_step4_network_stress_scenarios"
    script_sha = digest(Path(__file__))
    inputs = {"day16_step2_metadata.json": digest(d2 / "day16_step2_metadata.json"),
              "day16_step3_metadata.json": digest(d3 / "day16_step3_metadata.json"),
              emfile.name: digest(emfile)}
    if output.exists():
        old = verified_outputs(output, "day16_step4_metadata.json")
        require(old["script_sha256"] == script_sha and old["input_sha256"] == inputs,
                "Frozen Step4 version drift")
        for day in dates:
            for net in (RAW, RES):
                e = em.loc[(day, 120, net)]
                require(digest(src(root, e.source)) == str(e.sha256),
                        f"Frozen W120 network drift on recheck: {day}/{net}")
        print("Day16 Step4: FROZEN RECHECK_PASS")
        return
    rows, holdings, nedge = [], [], 0
    for date in dates:
        subset = sm[(sm.analysis_date == date) & (sm.window == 120)]
        require(set(subset.network) == {RAW, RES}, f"Missing W120 network: {date}")
        f = {}
        for net in (RAW, RES):
            positions = sm.index[(sm.analysis_date == date) & (sm.window == 120) &
                                 (sm.network == net)].to_numpy()
            require(len(positions) == 1, "Duplicate source row group")
            frame = pf.read_row_group(int(positions[0]), columns=COLS).to_pandas()
            require(len(frame) == int(subset.loc[subset.network == net, "nodes"].iloc[0]) and
                    (frame.analysis_date.astype(str) == str(date)).all() and
                    (frame.network == net).all() and (frame.window == 120).all(),
                    f"Parquet row group drift: {date}/{net}")
            f[net] = frame
        raw, res, ports, common_count = portfolio(f[RAW], f[RES])
        ids = raw.master_index.to_numpy(np.int64)
        idx = pd.Index(ids)
        sector = raw.industry_id1.to_numpy(np.int64)
        for name, stocks in ports.items():
            for stock in stocks:
                holdings.append(dict(analysis_date=date, portfolio=name,
                                     master_index=int(stock), weight=1/30,
                                     industry_id1=int(sector[idx.get_loc(stock)]), real_holdings=False))
        matrices, keys = {}, {}
        residual_edges = None
        for net in (RAW, RES):
            er = em.loc[(date, 120, net)]
            p, keyset, edge_data = network(root, er, ids)
            matrices[net], keys[net] = p, keyset
            nedge += 1
            expected = prior.loc[(date, net)]
            require(int(expected.holding_count) == 30 and
                    int(expected.common_eligible_nodes) == common_count and
                    abs(internal_density(ports[PORTS[0]], idx, keyset, len(ids)) -
                        float(expected.selected_pair_density)) < 1e-10,
                    f"Step3 selected portfolio reconstruction differs: {date}/{net}")
            if net == RES:
                residual_edges = edge_data
        anchor = int(ports[PORTS[1]][0])
        z, counts, sector_id, _ = shock_vectors(ids, sector, anchor, residual_edges)
        for net in (RAW, RES):
            solved = propagate(matrices[net], z)
            for name, stocks in ports.items():
                loc = idx.get_indexer(stocks)
                base = z[loc].mean(axis=0)
                for gamma in GAMMAS:
                    vals = solved[gamma][loc].mean(axis=0)
                    require(np.all(vals >= base-1e-11), "Amplification negative")
                    for k, shock_name in enumerate(SCENARIOS):
                        rows.append(dict(analysis_date=date, network=net, window=120,
                            portfolio=name, shock_scenario=shock_name,
                            gamma=gamma, shock_size=SHOCK, anchor_master_index=anchor,
                            anchor_industry_id1=int(sector_id), shocked_nodes=counts[k],
                            portfolio_n=30, initial_loss=float(base[k]),
                            stress_loss=float(vals[k]),
                            incremental_loss=float(vals[k]-base[k]),
                            amplification_ratio=float(vals[k]/base[k]) if base[k] > 0 else np.nan,
                            hypothetical_only=True, real_holdings=False))
    result = pd.DataFrame(rows)
    require(len(result) == 104*2*2*3*4 and len(holdings) == 104*2*30 and nedge == 208,
            "Stress output size mismatch")
    pivot = result.pivot(index=["analysis_date", "portfolio", "shock_scenario", "gamma"],
                         columns="network", values="stress_loss")
    contrast = (pivot[RES]-pivot[RAW]).rename("residual_minus_raw_loss").reset_index()
    paired = result.pivot(index=["analysis_date", "portfolio", "shock_scenario", "gamma"],
                          columns="network", values="initial_loss")
    require(np.allclose(paired[RES], paired[RAW], rtol=0, atol=1e-12) and
            (contrast.loc[contrast.gamma == 0, "residual_minus_raw_loss"].abs() < 1e-12).all(),
            "Unequal initial shocks / gamma-zero network contrast")
    summary = result.groupby(["network", "portfolio", "shock_scenario", "gamma"],
                             as_index=False).agg(mean_stress_loss=("stress_loss", "mean"),
                                mean_initial_loss=("initial_loss", "mean"),
                                mean_incremental_loss=("incremental_loss", "mean"),
                                median_incremental_loss=("incremental_loss", "median"))
    require(len(summary) == 48 and len(contrast) == 2496 and
            result.incremental_loss.min() >= -1e-10, "Stress QA failed")
    qa = dict(all_formal_qa_pass=True, formation_dates=104, frozen_w120_edges_rehashed=nedge,
              stress_rows=len(result), paired_network_comparisons=len(contrast),
              fixed_portfolio_holding_rows=len(holdings), summary_rows=len(summary),
              reconstructed_step3_portfolio_verified=True, identical_network_shock_vectors=True,
              gamma_zero_network_invariance=True, nonnegative_amplification=True,
              no_future_labels=True, actual_portfolios=False, causal_identification=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="day16_step4_", dir=output.parent))
    try:
        tables = {"network_stress_daily.csv": result,
                  "network_stress_contrast.csv": contrast,
                  "network_stress_summary.csv": summary,
                  "stress_portfolio_holdings.csv": pd.DataFrame(holdings),
                  "day16_step4_qa.csv": pd.DataFrame(qa.items(), columns=["metric", "value"])}
        for fn, t in tables.items():
            (tmp / fn).write_bytes(t.to_csv(index=False, lineterminator="\n",
                                            float_format="%.12g").encode("utf-8"))
        meta = dict(research_day=16, step="Step4_Network_Based_Hypothetical_Stress",
                    status="FROZEN", formal_qa=qa, script_sha256=script_sha,
                    input_sha256=inputs,
                    output_sha256={k: digest(tmp / k) for k in tables},
                    main="M1_RESIDUAL_POSITIVE_W120", gammas=list(GAMMAS),
                    scenarios=list(SCENARIOS), initial_per_shocked_stock=SHOCK,
                    transition="positive undirected weight row-normalized; (I-gamma P)^-1",
                    ego_definition="anchor + top 14 Residual W120 neighbors by weight; NOT Louvain community",
                    control_portfolio="lowest master_index 2 in 15 largest full node-universe sectors",
                    interpretation="descriptive hypothetical stress, no institutional crowding, prediction or causality")
        (tmp / "day16_step4_metadata.json").write_bytes(
            (json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False)+"\n").encode("utf-8"))
        require(not output.exists(), "Refuse overwrite of existing output")
        tmp.rename(output)
        print(f"Day16 Step4: FROZEN / PASS | {len(result)} stress rows | {len(contrast)} paired contrasts")
        print(f"Output: {output}")
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", default=r"D:\M1_StockNetwork")
        build(Path(parser.parse_args().root).resolve())
    except (RuntimeError, OSError, ValueError, KeyError, IndexError, ImportError) as exc:
        sys.exit(f"Day16 Step4 BLOCKED: {exc}")

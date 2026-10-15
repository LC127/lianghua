"""Day16 Step2: PIT historical downside risk and frozen network exposures.
Run: python day16_step2_network_risk_exposure_factors.py --root D:\\M1_StockNetwork
Requires: numpy, pandas, pyarrow. Does NOT read future labels or refit networks.
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
import pyarrow as pa
import pyarrow.parquet as pq

HIST, MIN_OBS, MIN_COVER, DENSITY = 60, 40, 0.80, 0.01
NETS = ("B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE")
WINS = (60, 120, 252)


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source(root, rel):
    p = (root / str(rel).replace("\\", "/")).resolve()
    need(p.is_relative_to(root.resolve()), f"Illegal source path: {rel}")
    return p


def load_inputs(root):
    d16 = root / "output/M1_day16/01_step1_frozen_network_portfolio_source_audit/source_only"
    s1 = d16 / "day16_step1_metadata.json"
    m = json.loads(s1.read_text(encoding="utf-8"))
    need(m["formal_qa"]["all_source_audit_pass"] and
         m["formal_qa"]["portfolio_status"] == "PORTFOLIO_PENDING", "Day16 Step1 source_only QA failed")
    for name, expected in m["output_sha256"].items():
        need(sha(d16 / name) == expected, f"Step1 drift: {name}")
    d15 = root / "output/M1_day15"
    s2 = d15 / "02_step2_stock_downside_target_freeze"
    s3 = d15 / "03_step3_network_edge_codistress"
    m2 = json.loads((s2 / "day15_step2_metadata.json").read_text(encoding="utf-8"))
    m3 = json.loads((s3 / "day15_step3_metadata.json").read_text(encoding="utf-8"))
    nm = s2 / "day15_step2_node_snapshot_manifest.csv"
    em = s3 / "day15_step3_edge_manifest.csv"
    need(sha(nm) == m2["output_sha256"][nm.name] and
         sha(em) == m3["output_sha256"][em.name], "Frozen manifests drift")
    nodes, edges = pd.read_csv(nm), pd.read_csv(em)
    need(len(nodes) == 312 and len(edges) == 624 and
         nodes.analysis_date.nunique() == 104 and
         not nodes.duplicated(["analysis_date", "window"]).any() and
         not edges.duplicated(["analysis_date", "window", "network"]).any(), "Frozen manifest keys invalid")
    keys = {
        "calendar": "output/M1_day1/01_stage1_final/trade_calendar_used.csv",
        "returns": "output/M1_day1/03_stage3_return_validation/daily_return_panel.parquet",
        "mapping": "output/M1_day4/02_step2_matched_networks/stock_master_mapping.parquet",
    }
    for rel in keys.values():
        need(sha(source(root, rel)) == m2["input_sha256"][rel.replace("/", "\\")],
             f"Day1/Day4 source drift: {rel}")
    return m, nodes, edges, {k: source(root, p) for k, p in keys.items()}, s1


def historical_risk(paths):
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(paths["calendar"])["trade_date"]).sort_values())
    need(dates.is_unique and len(dates), "Invalid trading calendar")
    mp = pd.read_parquet(paths["mapping"], columns=["master_index", "security_id"])
    mp = mp.sort_values("master_index").reset_index(drop=True)
    ids = pd.Index(mp.security_id.astype("string").str.strip())
    need(ids.is_unique and np.array_equal(mp.master_index.to_numpy(), np.arange(len(mp))),
         "Day4 mapping is not a unique dense master index")
    width = len(ids)
    ret = np.full((len(dates), width), np.nan, dtype=np.float32)
    seen = np.zeros(ret.shape, dtype=bool)
    pf = pq.ParquetFile(paths["returns"])
    cols = ["trade_date", "security_id", "return_network_simple", "is_open", "is_consecutive_open_pair"]
    need(set(cols).issubset(pf.schema_arrow.names), f"Daily return schema missing: {set(cols)-set(pf.schema_arrow.names)}")
    masked = 0
    for batch in pf.iter_batches(batch_size=250_000, columns=cols):
        b = batch.to_pandas()
        ti = dates.get_indexer(pd.to_datetime(b.trade_date).dt.normalize())
        si = ids.get_indexer(b.security_id.astype("string").str.strip())
        need((ti >= 0).all() and (si >= 0).all(), "Return row not in frozen calendar/mapping")
        k = ti.astype(np.int64) * width + si
        need(len(np.unique(k)) == len(k) and not seen[ti, si].any(), "Duplicate date-security return")
        val = pd.to_numeric(b.return_network_simple, errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan, copy=True)
        need(not (np.isinf(val) | (val < -1)).any(), "Invalid simple daily return")
        good = b.is_open.eq(True).fillna(False).to_numpy(bool) & \
               b.is_consecutive_open_pair.eq(True).fillna(False).to_numpy(bool)
        masked += int((~good & np.isfinite(val)).sum())
        val[~good] = np.nan
        ret[ti, si] = val.astype(np.float32)
        seen[ti, si] = True
    return dates, ids, ret, masked


def graph_factors(a, b, weight, sector, q, degree_width):
    n = degree_width
    u, v = np.r_[a, b], np.r_[b, a]
    w = np.r_[weight, weight]
    total = np.bincount(u, weights=w, minlength=n)
    degree = np.bincount(u, minlength=n)
    square = np.bincount(u, weights=w*w, minlength=n)
    risk_valid = np.isfinite(q[v])
    valid_weight = np.bincount(u[risk_valid], weights=w[risk_valid], minlength=n)
    value_sum = np.bincount(u[risk_valid], weights=w[risk_valid]*q[v[risk_valid]], minlength=n)
    cov = np.divide(valid_weight, total, out=np.zeros(n), where=total > 0)
    exposure = np.divide(value_sum, valid_weight, out=np.full(n, np.nan), where=valid_weight > 0)
    exposure[(cov < MIN_COVER) | (degree == 0)] = np.nan
    hhi = np.divide(square, total*total, out=np.full(n, np.nan), where=total > 0)
    sector_ok = (sector[u] >= 0) & (sector[v] >= 0)
    known = np.bincount(u[sector_ok], weights=w[sector_ok], minlength=n)
    cross = sector_ok & (sector[u] != sector[v])
    cross_w = np.bincount(u[cross], weights=w[cross], minlength=n)
    cross_share = np.divide(cross_w, total, out=np.full(n, np.nan), where=total > 0)
    cross_share[np.divide(known, total, out=np.zeros(n), where=total > 0) < MIN_COVER] = np.nan
    valid_cross = cross & risk_valid
    cross_sum = np.bincount(u[valid_cross], weights=w[valid_cross]*q[v[valid_cross]], minlength=n)
    cross_risk = np.divide(cross_sum, valid_weight, out=np.full(n, np.nan), where=valid_weight > 0)
    cross_risk[(cov < MIN_COVER) | np.isnan(cross_share)] = np.nan
    return degree, hhi, cov, exposure, cross_share, cross_risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    s1meta, nm, em, paths, s1path = load_inputs(root)
    out = root / "output/M1_day16/02_step2_network_risk_exposure_factors"
    script_hash = sha(Path(__file__))
    inputs = {"day16_step1_metadata.json": sha(s1path),
              "day15_step2_node_snapshot_manifest.csv": sha(root / "output/M1_day15/02_step2_stock_downside_target_freeze/day15_step2_node_snapshot_manifest.csv"),
              "day15_step3_edge_manifest.csv": sha(root / "output/M1_day15/03_step3_network_edge_codistress/day15_step3_edge_manifest.csv")}
    if out.exists():
        meta = json.loads((out / "day16_step2_metadata.json").read_text(encoding="utf-8"))
        need(meta["status"] == "FROZEN" and meta["script_sha256"] == script_hash and
             meta["input_sha256"] == inputs and
             all(sha(out / k) == v for k, v in meta["output_sha256"].items()),
             "Frozen Step2 drift; do not overwrite")
        for records, field in ((nm, "path"), (em, "source")):
            for row in records.itertuples(index=False):
                need(sha(source(root, getattr(row, field))) == str(row.sha256),
                     f"Frozen network source drift: {getattr(row, field)}")
        print("Day16 Step2: FROZEN RECHECK_PASS")
        return
    dates, securities, returns, masked = historical_risk(paths)
    date_index = {str(d): int(dates.get_indexer([pd.Timestamp(d)])[0]) for d in nm.analysis_date.unique()}
    need(all(i >= 0 for i in date_index.values()), "Formation date outside calendar")
    width = len(securities)
    nlookup = nm.set_index(["analysis_date", "window"])
    elookup = em.set_index(["analysis_date", "window", "network"])
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="day16_step2_", dir=out.parent))
    writer = None
    daily = []
    try:
        for day in sorted(date_index):
            ix = date_index[day]
            x = returns[max(0, ix - HIST + 1):ix + 1].astype(np.float64)
            counts = np.isfinite(x).sum(axis=0)
            downside = np.minimum(x, 0.0)
            q = np.sqrt(np.divide(np.nansum(downside*downside, axis=0), counts,
                                  out=np.full(width, np.nan), where=counts >= MIN_OBS))
            for win in WINS:
                nr = nlookup.loc[(day, win)]
                npth = source(root, nr["path"])
                need(sha(npth) == str(nr.sha256), f"Node SHA256 mismatch: {npth}")
                node = pd.read_parquet(npth, columns=["master_index", "security_id", "industry_id1"])
                ni = node.master_index.to_numpy(np.int64)
                need(len(ni) == int(nr.n_nodes) and len(np.unique(ni)) == len(ni) and
                     np.array_equal(securities.take(ni).to_numpy(), node.security_id.astype("string").str.strip().to_numpy()),
                     f"Node identity mismatch: {day}/W{win}")
                sector = np.full(width, -1, dtype=np.int64)
                sector[ni] = pd.to_numeric(node.industry_id1, errors="coerce").fillna(-1).to_numpy(np.int64)
                flag = np.zeros(width, dtype=bool)
                flag[ni] = True
                for net in NETS:
                    er = elookup.loc[(day, win, net)]
                    epth = source(root, er["source"])
                    need(sha(epth) == str(er.sha256), f"Edge SHA256 mismatch: {epth}")
                    edge = pd.read_parquet(epth, columns=["master_index_i", "master_index_j", "edge_weight", "rank_fraction_common"])
                    rank_all = pd.to_numeric(edge.rank_fraction_common, errors="coerce").to_numpy(float)
                    need(np.isfinite(rank_all).all() and (rank_all > 0).all(), f"Invalid frozen ranks: {epth}")
                    edge = edge.loc[rank_all <= DENSITY + 1e-12]
                    need(len(edge) == int(er.selected_1pct_edges), f"1% edge count mismatch: {epth}")
                    a = edge.master_index_i.to_numpy(np.int64)
                    b = edge.master_index_j.to_numpy(np.int64)
                    w = pd.to_numeric(edge.edge_weight, errors="coerce").to_numpy(float)
                    rank = pd.to_numeric(edge.rank_fraction_common, errors="coerce").to_numpy(float)
                    need(len(a) > 0 and np.all((a >= 0) & (b >= 0) & (a < width) & (b < width)) and
                         flag[a].all() and flag[b].all() and (a != b).all() and
                         np.isfinite(w).all() and (w > 0).all() and
                         np.isfinite(rank).all() and (rank > 0).all() and (rank <= DENSITY + 1e-12).all(),
                         f"Edge integrity failure: {epth}")
                    edges_key = np.minimum(a, b)*width + np.maximum(a, b)
                    need(len(np.unique(edges_key)) == len(edges_key), f"Repeated edge: {epth}")
                    deg, hhi, cov, ne, cross_share, cross_risk = graph_factors(a, b, w, sector, q, width)
                    status = np.where(deg[ni] == 0, "ISOLATED",
                                      np.where(cov[ni] < MIN_COVER, "LOW_NEIGHBOR_COVERAGE", "OK"))
                    frame = pd.DataFrame({"analysis_date": day, "network": net, "window": win,
                        "master_index": ni, "security_id": securities.take(ni).to_numpy(),
                        "industry_id1": sector[ni], "hist_valid_sessions": counts[ni].astype(np.int16),
                        "hist_downside_vol_60d": q[ni], "degree": deg[ni].astype(np.int32),
                        "neighbor_valid_weight_share": cov[ni], "neighbor_downside_exposure": ne[ni],
                        "connection_hhi": hhi[ni], "cross_industry_weight_share": cross_share[ni],
                        "cross_industry_downside_exposure": cross_risk[ni], "risk_status": status})
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temp / "network_risk_exposure_panel.parquet", table.schema, compression="snappy")
                    writer.write_table(table)
                    daily.append({"analysis_date": day, "network": net, "window": win,
                        "nodes": len(ni), "edges": len(a), "risk_valid": int(np.isfinite(q[ni]).sum()),
                        "neighbor_valid": int(np.isfinite(ne[ni]).sum()),
                        "cross_valid": int(np.isfinite(cross_risk[ni]).sum()),
                        "isolated_nodes": int((deg[ni] == 0).sum())})
        writer.close()
        writer = None
        ds = pd.DataFrame(daily)
        need(len(ds) == 624 and ds.analysis_date.nunique() == 104 and
             int(ds.neighbor_valid.sum()) > 0, "Missing network/date coverage or all neighbor risks unavailable")
        qa = dict(formal_qa_pass=True, formation_dates=104, network_date_cells=len(ds),
                  network_node_rows=int(ds.nodes.sum()), risk_valid_node_rows=int(ds.risk_valid.sum()),
                  valid_neighbor_exposure_rows=int(ds.neighbor_valid.sum()),
                  valid_cross_exposure_rows=int(ds.cross_valid.sum()),
                  masked_nontrade_returns=masked, historical_sessions=HIST,
                  min_valid_sessions=MIN_OBS, min_neighbor_weight_coverage=MIN_COVER,
                  future_labels_used=False, frozen_graph_hashes_verified=624,
                  frozen_node_hashes_verified=312,
                  portfolio_status=s1meta["formal_qa"]["portfolio_status"])
        files = {"network_risk_exposure_daily_summary.csv": ds.to_csv(index=False, lineterminator="\n").encode("utf-8"),
                 "day16_step2_qa.csv": pd.DataFrame([{"metric": k, "value": v} for k, v in qa.items()]).to_csv(index=False, lineterminator="\n").encode("utf-8")}
        for name, content in files.items():
            (temp / name).write_bytes(content)
        names = ["network_risk_exposure_panel.parquet", *files]
        metadata = dict(research_day=16, step="Step2_Network_Risk_Exposure_Factor_Construction",
                        status="FROZEN", formal_qa=qa, input_sha256=inputs,
                        script_sha256=script_hash, output_sha256={n: sha(temp / n) for n in names},
                        risk_definition="sqrt(mean(min(simple_return,0)^2 over valid last 60 market sessions))",
                        time_policy="formation-close PIT; 40/60 min; suspended/nonconsecutive masked; no future labels",
                        network_policy="undirected positive 1pct frozen edge_weight; full node/edge SHA256 verified")
        (temp / "day16_step2_metadata.json").write_bytes((json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        need(not out.exists(), "Another Step2 output exists; refuse replacement")
        temp.rename(out)
        print(f"Day16 Step2: FROZEN / PASS | {len(ds)} cells | {qa['network_node_rows']:,} rows")
        print(f"Neighbor exposure coverage = {qa['valid_neighbor_exposure_rows']/qa['network_node_rows']:.2%}")
        print(f"Output: {out}")
    finally:
        if writer is not None:
            writer.close()
        if temp.exists():
            shutil.rmtree(temp)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, IndexError) as exc:
        sys.exit(f"Day16 Step2 BLOCKED: {exc}")

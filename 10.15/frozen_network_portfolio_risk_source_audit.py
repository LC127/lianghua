"""Day16 Step1: read-only PIT network/risk/optional portfolio source audit.
Run: python day16_step1_frozen_network_portfolio_source_audit.py --root D:\\M1_StockNetwork
Optional: --holdings D:\\path\\to\\holdings.csv (analysis_date,security_id,weight[,portfolio_id]).
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("请先在运行脚本的 Python 环境安装 pyarrow：python -m pip install pyarrow") from exc

STEPS = {
    "S1": ("01_step1_frozen_network_risk_source_audit", "day15_step1_metadata.json"),
    "S2": ("02_step2_stock_downside_target_freeze", "day15_step2_metadata.json"),
    "S3": ("03_step3_network_edge_codistress", "day15_step3_metadata.json"),
    "S4": ("04_step4_community_cross_industry_pathways", "day15_step4_metadata.json"),
    "S5A": ("05_step5_temporal_order_matched_null", "day15_step5_metadata.json"),
    "S5B": ("05b_step5b_common_shock_placebo", "day15_step5b_metadata.json"),
    "S5C": ("05c_step5c_rolling_oos", "day15_step5c_metadata.json"),
    "S5D": ("05d_step5d_causal_feasibility_audit", "day15_step5d_metadata.json"),
}
RISK_COLS = ["trade_date", "security_id", "return_network_simple",
             "industry_id1", "market_value", "turnover_value", "is_open"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def scan_formation(panel_path, dates):
    """Batch-scan 11M+ daily rows; retain formation-date rows only."""
    pf = pq.ParquetFile(panel_path)
    missing = set(RISK_COLS) - set(pf.schema_arrow.names)
    require(not missing, f"Day1 日频面板缺少字段: {sorted(missing)}")
    parts = []
    for batch in pf.iter_batches(batch_size=250_000, columns=RISK_COLS):
        df = batch.to_pandas()
        day = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        keep = day.isin(dates)
        if keep.any():
            x = df.loc[keep, RISK_COLS[1:]].copy()
            x.insert(0, "analysis_date", day.loc[keep].to_numpy())
            parts.append(x)
    require(bool(parts), "日收益面板未找到任何冻结形成日期")
    df = pd.concat(parts, ignore_index=True)
    df["security_id"] = df["security_id"].astype("string").str.strip()
    require(not df.duplicated(["analysis_date", "security_id"]).any(),
            "日频面板存在重复的形成日期×股票，请先审计上游")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    ap.add_argument("--holdings", help="可选真实组合持仓 CSV；缺省则标记 PORTFOLIO_PENDING")
    args = ap.parse_args()
    root = Path(args.root)
    d15 = root / "output/M1_day15"
    step6 = d15 / "06_step6_final_qa_research_summary"
    final = json.loads((step6 / "day15_step6_metadata.json").read_text(encoding="utf-8"))
    require(final.get("status") == "FROZEN" and final["formal_qa"]["all_formal_qa_pass"],
            "Day15 Step6 未正式通过冻结 QA")

    inv, qa = [], {}
    def check(role, path, expected, full=True):
        require(path.is_file(), f"缺失来源文件: {path}")
        actual = digest(path) if full else "NOT_REHASHED"
        if expected and full:
            require(actual == expected, f"冻结 SHA256 不匹配: {path}")
        inv.append(dict(role=role, path=str(path), expected_sha256=expected or "",
                        actual_sha256=actual, verified=bool(full)))

    # Step6 four frozen outputs and 8 stage-metadata fingerprints.
    for name, expected in final["output_sha256"].items():
        check("day15_final", step6 / name, expected)
    for key, (folder, fname) in STEPS.items():
        meta = d15 / folder / fname
        check(key + "_metadata", meta, final["input_metadata_sha256"][key])
        require(json.loads(meta.read_text(encoding="utf-8"))["formal_qa"]["all_formal_qa_pass"],
                f"上游未通过 Formal QA: {key}")
    qa["frozen_chain_pass"] = True

    s2dir = d15 / STEPS["S2"][0]
    s3dir = d15 / STEPS["S3"][0]
    m2 = json.loads((s2dir / STEPS["S2"][1]).read_text(encoding="utf-8"))
    m3 = json.loads((s3dir / STEPS["S3"][1]).read_text(encoding="utf-8"))
    nm = s2dir / "day15_step2_node_snapshot_manifest.csv"
    em = s3dir / "day15_step3_edge_manifest.csv"
    check("node_manifest", nm, m2["output_sha256"][nm.name])
    check("edge_manifest", em, m3["output_sha256"][em.name])
    nodes, edges = pd.read_csv(nm), pd.read_csv(em)
    require(len(nodes) == 312 and len(edges) == 624, "冻结节点/边快照数量不是312/624")
    require(nodes.analysis_date.nunique() == edges.analysis_date.nunique() == 104,
            "冻结104个形成日期不一致")
    require(not nodes.duplicated(["analysis_date", "window"]).any() and
            not edges.duplicated(["analysis_date", "window", "network"]).any(),
            "快照 Manifest 出现重复键")
    require(set(nodes.window) == set(edges.window) == {60, 120, 252}, "窗口不是60/120/252")
    require(set(edges.network) == {"B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE"}, "网络类型不匹配")
    paired = edges.pivot(index=["analysis_date", "window"],
                         columns="network", values="selected_1pct_edges")
    require(paired.notna().all().all() and paired.iloc[:, 0].eq(paired.iloc[:, 1]).all()
            and (paired.iloc[:, 0] > 0).all(), "Raw/Residual 1%边预算不一致")

    def source(rel):
        p = (root / Path(str(rel).replace("\\", "/"))).resolve()
        require(p.is_relative_to(root.resolve()), f"非法相对路径: {rel}")
        return p

    for r in nodes.itertuples(index=False):
        check("frozen_node", source(r.path), str(r.sha256))
    require(all(source(rel).is_file() for rel in edges.source), "至少一份冻结边文件缺失")
    sampled = edges.sort_values("analysis_date").groupby(["network", "window"]).head(1)
    for r in sampled.itertuples(index=False):
        p = source(r.source)
        check("sample_edge", p, str(r.sha256))
        cols = set(pq.ParquetFile(p).schema_arrow.names)
        require({"master_index_i", "master_index_j", "rank_fraction_common",
                 "edge_weight"}.issubset(cols), f"边快照字段缺失: {p}")
    qa.update(node_snapshots_hashed=len(nodes), edge_snapshots_present=len(edges),
              representative_edges_hashed=len(sampled), matched_density_pass=True)

    # Day1 sources must equal the SHA256 explicitly pinned by Day15 Step2.
    src = {
        "stock_master": "output/M1_day1/01_stage1_final/stock_master.parquet",
        "trade_calendar": "output/M1_day1/01_stage1_final/trade_calendar_used.csv",
        "daily_returns": "output/M1_day1/03_stage3_return_validation/daily_return_panel.parquet",
        "day4_mapping": "output/M1_day4/02_step2_matched_networks/stock_master_mapping.parquet",
    }
    for role, rel in src.items():
        key = rel.replace("/", "\\")
        check(role, root / rel, m2["input_sha256"][key])
    mapping = pd.read_parquet(root / src["day4_mapping"], columns=["master_index", "security_id"])
    require(not mapping.duplicated("security_id").any() and
            not mapping.duplicated("master_index").any(), "Day4股票映射存在重复")
    mapping["security_id"] = mapping.security_id.astype("string").str.strip()
    formation_dates = set(nodes.analysis_date.astype(str))
    f = scan_formation(root / src["daily_returns"], formation_dates)
    require(set(f.analysis_date) == formation_dates, "部分形成日期无法匹配日频行情")
    mv = pd.to_numeric(f.market_value, errors="coerce")
    turn = pd.to_numeric(f.turnover_value, errors="coerce")
    qa.update(formation_dates=len(formation_dates), formation_stock_rows=len(f),
              valid_market_cap_share=float(((mv > 0) & np.isfinite(mv)).mean()),
              valid_industry_share=float(f.industry_id1.notna().mean()),
              valid_turnover_share=float(((turn >= 0) & np.isfinite(turn)).mean()),
              portfolio_provided=bool(args.holdings))
    require((mv > 0).any() and f.industry_id1.notna().any(),
            "历史市值或行业字段缺少有效值")

    coverage = []
    portfolio_status = "PORTFOLIO_PENDING"
    if args.holdings:
        hpath = Path(args.holdings)
        require(hpath.is_file(), f"持仓文件不存在: {hpath}")
        h_sha = digest(hpath)
        check("portfolio_holdings", hpath, h_sha)
        h = pd.read_csv(hpath, dtype={"security_id": "string"}, low_memory=False)
        require({"analysis_date", "security_id", "weight"}.issubset(h),
                "持仓 CSV 必须包括 analysis_date,security_id,weight")
        if "portfolio_id" not in h:
            h["portfolio_id"] = "P1"
        h["analysis_date"] = pd.to_datetime(h.analysis_date, errors="coerce").dt.strftime("%Y-%m-%d")
        h["security_id"] = h.security_id.astype("string").str.strip()
        h["weight"] = pd.to_numeric(h.weight, errors="coerce")
        require(h.analysis_date.notna().all() and h.analysis_date.isin(formation_dates).all(),
                "持仓日期必须属于104个冻结网络形成日期")
        require(h.security_id.notna().all() and h.portfolio_id.notna().all(), "持仓股票/组合ID不能缺失")
        require(h.weight.notna().all() and np.isfinite(h.weight).all() and h.weight.between(0, 1).all(),
                "当前方案仅允许0-1之间的多头组合权重")
        key = ["portfolio_id", "analysis_date", "security_id"]
        require(not h.duplicated(key).any(), "同一组合日期股票存在重复持仓")
        sums = h.groupby(key[:2]).weight.sum()
        require(np.allclose(sums, 1.0, atol=1e-4), "每个组合形成日的权重总和必须为1")
        h = h.merge(mapping, on="security_id", how="left", validate="many_to_one")
        info = f[["analysis_date", "security_id", "market_value", "industry_id1", "turnover_value"]]
        h = h.merge(info, on=["analysis_date", "security_id"], how="left", validate="many_to_one")
        # Membership uses frozen nodes at time t, never a future complete-case universe.
        universe = {}
        for date, group in nodes.groupby("analysis_date"):
            u = set()
            for r in group.itertuples(index=False):
                arr = pd.read_parquet(source(r.path), columns=["master_index"])
                u.update(arr.master_index.tolist())
            universe[date] = u
        h["in_network"] = [bool(pd.notna(idx) and int(idx) in universe[d])
                           for idx, d in zip(h.master_index, h.analysis_date)]
        h["valid_cap"] = pd.to_numeric(h.market_value, errors="coerce").gt(0)
        h["valid_industry"] = h.industry_id1.notna()
        for (pid, date), group in h.groupby(key[:2], sort=True):
            coverage.append(dict(portfolio_id=str(pid), analysis_date=date,
                holdings=len(group), weight_sum=float(group.weight.sum()),
                mapped_weight=float(group.loc[group.master_index.notna(), "weight"].sum()),
                network_weight=float(group.loc[group.in_network, "weight"].sum()),
                cap_weight=float(group.loc[group.valid_cap, "weight"].sum()),
                industry_weight=float(group.loc[group.valid_industry, "weight"].sum())))
        portfolio_status = ("PORTFOLIO_SOURCE_PASS" if coverage and all(
            min(x["mapped_weight"], x["network_weight"], x["cap_weight"],
                x["industry_weight"]) > 0.9999 for x in coverage)
            else "PORTFOLIO_COVERAGE_INCOMPLETE")
    qa["portfolio_status"] = portfolio_status
    qa["all_source_audit_pass"] = bool(qa["frozen_chain_pass"] and
        qa["node_snapshots_hashed"] == 312 and qa["edge_snapshots_present"] == 624)

    out = root / "output/M1_day16/01_step1_frozen_network_portfolio_source_audit" / (
        "source_only" if not args.holdings else "holdings_" + h_sha[:12])
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "day16_step1_source_inventory.csv": pd.DataFrame(inv).to_csv(index=False).encode("utf-8"),
        "day16_step1_portfolio_coverage.csv": pd.DataFrame(coverage, columns=[
            "portfolio_id", "analysis_date", "holdings", "weight_sum", "mapped_weight",
            "network_weight", "cap_weight", "industry_weight"]).to_csv(index=False).encode("utf-8"),
        "day16_step1_qa.csv": pd.DataFrame([{"metric": k, "value": v} for k, v in qa.items()]).to_csv(index=False).encode("utf-8"),
    }
    meta = dict(research_day=16, step="Step1_Frozen_Network_Portfolio_Source_Audit",
                status="NETWORK_SOURCE_PASS__" + portfolio_status, formal_qa=qa,
                source_policy="Day15 M1 frozen; return/size/turnover observed by formation-day close",
                portfolio_policy="Optional long-only holdings, weights sum 1; no synthetic portfolio generated",
                hash_limit="312 nodes fully hashed; 624 edges path-checked, six edges hash-sampled; inherited Day15 full manifest",
                code_sha256=digest(Path(__file__)),
                output_sha256={k: hashlib.sha256(v).hexdigest() for k, v in payload.items()})
    payload["day16_step1_metadata.json"] = (json.dumps(meta, ensure_ascii=False,
                                                         indent=2, sort_keys=True) + "\n").encode("utf-8")
    if any((out / name).exists() for name in payload):
        require(all((out / name).is_file() and (out / name).read_bytes() == blob
                    for name, blob in payload.items()), "冻结输出不一致，禁止覆盖；如需更改请另开版本")
        action = "RECHECK_PASS"
    else:
        for name, blob in payload.items():
            (out / name).write_bytes(blob)
        action = "WRITTEN"
    print(f"Day16 Step1: NETWORK_SOURCE_PASS | {portfolio_status} | {action}")
    print(f"Dates={len(formation_dates)} | Nodes={len(nodes)} hashed | Edges={len(edges)} present")
    print(f"Market-cap PIT coverage={qa['valid_market_cap_share']:.2%}; Industry={qa['valid_industry_share']:.2%}")
    print(f"Output: {out}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, ValueError, OSError) as exc:
        sys.exit(f"Day16 Step1 BLOCKED: {exc}")

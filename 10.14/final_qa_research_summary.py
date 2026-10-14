# -*- coding: utf-8 -*-
"""Day15 Step6: readonly frozen-chain QA + final research synthesis."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STEPS = {
    "S1": ("01_step1_frozen_network_risk_source_audit", "day15_step1", []),
    "S2": ("02_step2_stock_downside_target_freeze", "day15_step2",
           ["day15_step2_node_snapshot_manifest.csv", "stock_forward_downside_20d_panel.csv"]),
    "S3": ("03_step3_network_edge_codistress", "day15_step3",
           ["day15_step3_edge_manifest.csv", "network_edge_codistress_daily.csv", "network_edge_codistress_hac_summary.csv"]),
    "S4": ("04_step4_community_cross_industry_pathways", "day15_step4",
           ["network_pathway_group_daily.csv", "network_pathway_hac_summary.csv",
            "candidate_cross_industry_community_bridges.csv"]),
    "S5A": ("05_step5_temporal_order_matched_null", "day15_step5",
            ["temporal_order_daily.csv", "temporal_order_hac_summary.csv"]),
    "S5B": ("05b_step5b_common_shock_placebo", "day15_step5b",
            ["temporal_common_shock_placebo_daily.csv", "temporal_common_shock_placebo_hac.csv"]),
    "S5C": ("05c_step5c_rolling_oos", "day15_step5c",
            ["rolling_oos_daily.csv", "rolling_oos_score_summary.csv"]),
    "S5D": ("05d_step5d_causal_feasibility_audit", "day15_step5d",
            ["causal_robustness_periods.csv", "pre_event_placebo_equivalence.csv",
             "causal_identification_checklist.csv", "day15_step5d_conclusion.md"]),
}
RAW, RES = "B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def normalized(v):
    s = str(v).strip().lower()
    if s in ("true", "false", "none", "nan"):
        return s
    try:
        return float(s)
    except ValueError:
        return s


def same(a, b):
    a, b = normalized(a), normalized(b)
    if isinstance(a, float) and isinstance(b, float):
        return np.isclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=True)
    return a == b


def required_row(df, **filters):
    for c, v in filters.items():
        df = df.loc[df[c] == v]
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row: {filters}; got {len(df)}")
    return df.iloc[0]


def tobytes(df):
    return df.to_csv(index=False, lineterminator="\n", float_format="%.14g").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    a = ap.parse_args()
    root = Path(a.root)
    day = root / "output" / "M1_day15"
    out = day / "06_step6_final_qa_research_summary"
    evidence_dir, metadata, file_paths, checks = {}, {}, {}, []

    def qa(k, passed, value=""):
        checks.append({"qa_name": k, "pass": bool(passed), "detail": str(value)})
        if not passed:
            raise RuntimeError(f"FINAL QA BLOCKED: {k} | {value}")

    # 1. Verify each frozen step, its own output hashes, and formal QA rows.
    for tag, (folder, prefix, extras) in STEPS.items():
        src = day / folder
        mp, qp = src / f"{prefix}_metadata.json", src / f"{prefix}_qa.csv"
        qa(f"{tag}_metadata_exists", mp.is_file(), mp)
        m = json.loads(mp.read_text(encoding="utf-8"))
        q = pd.read_csv(qp, dtype=str, low_memory=False)
        qa(f"{tag}_frozen", m.get("status") == ("SOURCE_AUDIT_PASS" if tag == "S1" else "FROZEN"))
        qa(f"{tag}_formal", m.get("formal_qa", {}).get("all_formal_qa_pass") is True)
        qa(f"{tag}_qa_unique", q.qa_name.is_unique)
        qa(f"{tag}_qa_agrees", all(
            k in m["formal_qa"] and same(v, m["formal_qa"][k])
            for k, v in zip(q.qa_name, q.qa_value)))
        named = [mp, qp] + [src / x for x in extras]
        for p in named:
            qa(f"{tag}_exists_{p.name}", p.is_file(), p)
            qa(f"{tag}_unique_filename_{p.name}", p.name not in file_paths)
            file_paths[p.name] = p
        for name, expected in m.get("output_sha256", {}).items():
            p = src / name
            qa(f"{tag}_output_hash_{name}", p.is_file() and sha(p) == expected, name)
        metadata[tag], evidence_dir[tag] = m, src

    # 2. Verify the inter-step hash chain (Day15 inputs, not raw giant Parquet data).
    chain = []
    for tag, m in metadata.items():
        for name, expected in m.get("input_sha256", {}).items():
            basename = name.replace("\\", "/").split("/")[-1]
            p = file_paths.get(basename)
            if p is None:
                chain.append({"step": tag, "input": name, "scope": "EXTERNAL_SOURCE_NOT_REHASHED",
                              "sha256_expected": expected, "verified": False})
                continue
            ok = sha(p) == expected
            qa(f"{tag}_upstream_{basename}", ok)
            chain.append({"step": tag, "input": name, "scope": "DAY15_HASH_CHAIN",
                          "sha256_expected": expected, "verified": ok})
    qa("S2_20d_label", metadata["S2"].get("horizon_market_sessions") == 20)
    m2 = metadata["S2"]["formal_qa"]
    qa("S2_label_accounting", m2["available_labels"] + m2["unavailable_labels"] == m2["universe_rows"])
    qa("S3_624_cells", metadata["S3"]["formal_qa"].get("graph_cells") == 624)
    qa("S4_21_tests", metadata["S4"].get("tests") == 21)
    qa("S5A_9_tests", metadata["S5A"].get("tests") == 9)
    qa("S5B_9_tests", metadata["S5B"].get("formal_qa", {}).get("estimable_tests") == 9)
    qa("S5C_16_tests", metadata["S5C"].get("number_of_bh_tests") == 16)
    qa("S5D_no_causal_claim", metadata["S5D"].get("conclusion") == "CAUSAL_IDENTIFICATION_NOT_ESTABLISHED")

    # 3. Extract evidence from frozen summary tables; do not re-estimate.
    def read(tag, name):
        return pd.read_csv(evidence_dir[tag] / name, low_memory=False)

    s3 = required_row(read("S3", "network_edge_codistress_hac_summary.csv"), network=RES, window=120)
    s4 = read("S4", "network_pathway_hac_summary.csv")
    ind = required_row(s4, network=RES, window=120, test_type="BETWEEN_GROUP_CONTRAST",
                       group="CROSS_INDUSTRY_MINUS_SAME_INDUSTRY")
    comm = required_row(s4, network=RES, window=120, test_type="BETWEEN_GROUP_CONTRAST",
                        group="CROSS_COMMUNITY_MINUS_WITHIN_COMMUNITY")
    s5a = required_row(read("S5A", "temporal_order_hac_summary.csv"), network=RES,
                       window=120, test="EDGE_MINUS_NULL")
    b = read("S5B", "temporal_common_shock_placebo_hac.csv")
    post = required_row(b, network=RES, window=120, test="POST_SECTOR")
    pre = required_row(b, network=RES, window=120, test="PRE_SECTOR")
    contrast = required_row(b, network="RESIDUAL_MINUS_RAW", window=120,
                            test="POST_SECTOR_CONTRAST")
    c = read("S5C", "rolling_oos_score_summary.csv")
    primary = required_row(c, network=RES, window=120, metric="POST_SECTOR",
                           mode="expanding", score="brier")
    qa("S5C_all_16_nonsignificant", len(c) == 16 and (c.bh_q_16_tests >= 0.05).all())
    qa("S5D_13_conditions", metadata["S5D"]["formal_qa"].get("identification_criteria") == 13)
    qa("S5D_checklist_rows", len(read("S5D", "causal_identification_checklist.csv")) == 13)
    qa("S3_daily_rows", len(read("S3", "network_edge_codistress_daily.csv")) == 624)
    qa("S4_daily_rows", len(read("S4", "network_pathway_group_daily.csv")) == 1456)
    qa("S5B_daily_rows", len(read("S5B", "temporal_common_shock_placebo_daily.csv")) == 208)
    qa("S5C_oos_rows", (read("S5C", "rolling_oos_daily.csv")["status"] == "OOS_EVALUATED").sum() == 520)
    qa("S5D_periods_24", len(read("S5D", "causal_robustness_periods.csv")) == 24)
    qa("S5D_tost_4", len(read("S5D", "pre_event_placebo_equivalence.csv")) == 4)

    ev = []
    def add(stage, metric, val, q, meaning, source):
        ev.append({"stage": stage, "metric": metric, "estimate": float(val),
                   "bh_q": float(q) if pd.notna(q) else np.nan,
                   "interpretation": meaning, "source": source})
    add("S3", "Residual W120 co-distress edge-null", s3.delta_mean, s3.bh_q_six_tests,
        "Observed co-distress association; not contagion", "network_edge_codistress_hac_summary.csv")
    add("S4", "cross-sector minus within-sector", ind.mean_delta, ind.bh_q_21_tests,
        "Subgroup contrast not statistically robust", "network_pathway_hac_summary.csv")
    add("S4", "cross-community minus within-community", comm.mean_delta, comm.bh_q_21_tests,
        "Subgroup contrast not statistically robust", "network_pathway_hac_summary.csv")
    add("S5A", "post-source shock edge-null probability", s5a.mean_delta, s5a.bh_q_9_tests,
        "Temporal ordering association only", "temporal_order_hac_summary.csv")
    add("S5B", "post-sector-adjusted edge-null probability", post.mean_delta, post.bh_q_9_tests,
        "Conditional association survives stronger controls", "temporal_common_shock_placebo_hac.csv")
    add("S5B", "pre-sector placebo edge-null probability", pre.mean_delta, pre.bh_q_9_tests,
        "Non-significance does not establish equality", "temporal_common_shock_placebo_hac.csv")
    add("S5B", "Residual minus Raw post-sector", contrast.mean_delta, contrast.bh_q_9_tests,
        "Residual superiority not confirmed", "temporal_common_shock_placebo_hac.csv")
    add("S5C", "rolling OOS Brier gain (expanding, sector)", primary.mean_daily_gain, primary.bh_q_16_tests,
        "No robust retrospective OOS predictive gain", "rolling_oos_score_summary.csv")

    # 4. Day14 market evidence: independently audited if frozen file exists.
    d14 = root / "output/M1_day14/06_step6_final_m1_m4_linkage_summary"
    dm = d14 / "day14_step6_metadata.json"
    day14_note = "未提供 Day14 正式文件：市场层面结论未在本次独立核验。"
    if dm.exists():
        m14 = json.loads(dm.read_text(encoding="utf-8"))
        qa("Day14_frozen", m14.get("status") == "FROZEN" and m14.get("formal_qa", {}).get("all_formal_qa_pass") is True)
        for name, expected in m14.get("outputs_sha256", {}).items():
            qa(f"Day14_hash_{name}", sha(d14 / name) == expected)
        market = read_csv = pd.read_csv(d14 / "m1_m4_final_linkage_evidence.csv")
        x = required_row(market, market_proxy="CSI300")
        add("Day14", "CSI300 forward MDD high-low", x.high_minus_low, x.high_low_bh_q,
            "Market-level tail warning not validated", "m1_m4_final_linkage_evidence.csv")
        day14_note = (f"CSI300 HIGH–LOW = {100*x.high_minus_low:.4f} pp，BH q={x.high_low_bh_q:.4g}；"
                      f"调整后 HAC p={x.adjusted_hac_p:.4g}。市场级尾部预警未获稳健支持。")
        d14_lines = (d14 / "m1_m4_final_linkage_summary.md").read_text(encoding="utf-8").splitlines()
        day14_note += " " + " ".join(v.strip("- ") for v in d14_lines if "递推 OOS" in v)


    # 5. Produce deterministic, non-overwriting, byte-verified frozen outputs.
    s2 = metadata["S2"]["formal_qa"]
    md = f"""# Day15 Step6｜最终 QA 与研究总结\n\n**定位：**M1 v1.0 冻结网络的尾部风险关联审计；不重新估计网络、不新增统计假设。\n\n## 数据与程序复核\n\n- 冻结形成日期：{s2['frozen_formation_dates']}；节点快照312份、边快照624份（来源审计基于上游QA/指纹）。\n- 股票–形成日期：{s2['universe_rows']:,}；有效20日标签：{s2['available_labels']:,}（{100*s2['available_labels']/s2['universe_rows']:.2f}%）。\n- 末端右截尾：{s2['calendar_right_censored_n']:,}；缺失或无效收益：{s2['missing_or_invalid_return_n']:,}；退市截尾：{s2['delist_censored_n']:,}。\n- 已核验 Day15 输出字节 SHA256、步骤间输入指纹、Formal QA；不会将上游原始 Parquet 全量重哈希误写为本轮独立完成。\n\n## 冻结实证证据\n\n1. **Step3 共同损失：**Residual W120 真实边减行业匹配非边 = {s3.delta_mean:.9f}，BH q={s3.bh_q_six_tests:.4g}；支持风险共聚集，不证明风险传染。\n2. **Step4 结构差异：**跨行业−同行业 = {ind.mean_delta:.9f}（q={ind.bh_q_21_tests:.4g}）；跨社区−社区内 = {comm.mean_delta:.9f}（q={comm.bh_q_21_tests:.4g}）；主网络两项组间差异均未获显著支持。\n3. **Step5-A 时间顺序：**源股票先跌后，关联目标的事件概率增量 = {100*s5a.mean_delta:.3f} pp（q={s5a.bh_q_9_tests:.4g}）。\n4. **Step5-B 控制/安慰剂：**行业调整后增量 = {100*post.mean_delta:.3f} pp（q={post.bh_q_9_tests:.4g}）；事前行业调整安慰剂 = {100*pre.mean_delta:.3f} pp（q={pre.bh_q_9_tests:.4g}）；Residual−Raw行业调整差异 q={contrast.bh_q_9_tests:.4g}。事前和事后风险集不同，不能作因果DiD。\n5. **Step5-C 回顾性滚动评价：**主设定行业调整后 Expanding Brier gain={primary.mean_daily_gain:.9g}（BH q={primary.bh_q_16_tests:.4g}）；16项指标均未通过5% BH门槛，不支持稳健预测增量；不是全新独立留出样本。\n6. **Step5-D 因果审计：**`CAUSAL_IDENTIFICATION_NOT_ESTABLISHED`。源下跌事件可能内生于市场/行业共同消息，存在未观测混杂及节点干扰；探索性TOST与历史分段不能构成因果识别。\n\n## Day14 市场级独立背景\n\n{day14_note}\n\n## 研究结论及后续工作\n\n**已获得：**股票网络下行损失共聚集及条件时间顺序关联的统计证据。\n\n**未获得：**稳定的行业调整后历史样本外概率预测改善、市场指数尾部风险可靠预警、因果风险传导方向的识别。\n\n下一阶段应锁定当前口径，在真正未见日期前瞻验证，并另行寻找有可信外生性的公司特异性冲击开展准实验；不得据本轮 q 值再选阈值/网络窗口。\n"""
    output = {"day15_step6_evidence_matrix.csv": tobytes(pd.DataFrame(ev)),
              "day15_step6_hash_chain.csv": tobytes(pd.DataFrame(chain)),
              "day15_step6_qa.csv": tobytes(pd.DataFrame(checks)),
              "day15_step6_final_summary.md": md.encode("utf-8")}
    meta = {"research_day": 15, "step": "Step6_Final_QA_Research_Summary", "status": "FROZEN",
            "formal_qa": {"all_formal_qa_pass": True, "checks_passed": len(checks),
                          "day15_step_count": len(STEPS), "external_inputs_not_rehashed": sum(x["scope"] != "DAY15_HASH_CHAIN" for x in chain),
                          "day14_context_verified": dm.exists(), "causal_identification": "NOT_ESTABLISHED"},
            "input_metadata_sha256": {tag: sha(evidence_dir[tag] / f"{STEPS[tag][1]}_metadata.json") for tag in STEPS},
            "day14_metadata_sha256": sha(dm) if dm.exists() else None,
            "script_sha256": sha(Path(__file__)),
            "output_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in output.items()}}
    output["day15_step6_metadata.json"] = (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in output.items():
        p = out / name
        if p.exists() and p.read_bytes() != content:
            raise RuntimeError(f"FROZEN OUTPUT DRIFT: {p}")
    for name, content in output.items():
        p = out / name
        if not p.exists():
            p.write_bytes(content)
    print(f"Day15 Step6: FINAL_QA_PASS / FROZEN | {len(checks)} checks | {len(chain)} chained inputs")
    print("Output:", out)


if __name__ == "__main__":
    main()

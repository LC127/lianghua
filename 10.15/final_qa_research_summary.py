"""Day16 Step6: read-only frozen QA, structural evidence, and M1->M4 report.
Run: python day16_step6_final_qa_research_summary.py --root D:\\M1_StockNetwork
Needs: pandas and numpy. No new models, forward labels, or Parquet decoding.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW, RES = "B0_RAW_POSITIVE", "M1_RESIDUAL_POSITIVE"
DIRS = ["01_step1_frozen_network_portfolio_source_audit",
        "02_step2_network_risk_exposure_factors", "03_step3_network_concentration_crowding",
        "04_step4_network_stress_scenarios", "05_step5_incremental_risk_attribution_m4_interface"]
PORTS = ("SYNTHETIC_HIGH_RESIDUAL_NEIGHBOR_RISK", "SECTOR_BALANCED_ID_CONTROL")
GAMMAS = {0.0, 0.1, 0.2, 0.3}


def need(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bytes(path, data):
    path.write_bytes(data)


def write_csv(path, df):
    write_bytes(path, df.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def check(ok, title, results):
    need(ok, f"FINAL QA BLOCKED: {title}")
    results.append({"check": title, "status": "PASS"})


def exact(df, keys, n, name, checks):
    check(len(df) == n and not df.duplicated(keys).any(), f"{name}: {n} unique rows", checks)


def load_sources(root, checks):
    base = root / "output/M1_day16"
    dirs = []
    for i, sub in enumerate(DIRS, 1):
        d = base / sub
        if i == 1:
            found = list(d.rglob("day16_step1_metadata.json"))
            check(len(found) == 1, "one Step1 frozen source variant", checks)
            d = found[0].parent
        dirs.append(d)
    meta, hashes, ledger = [], {}, []
    for i, d in enumerate(dirs, 1):
        mn = f"day16_step{i}_metadata.json"
        mp = d / mn
        m = json.loads(mp.read_text(encoding="utf-8"))
        good = (m.get("formal_qa", {}).get("all_source_audit_pass") if i == 1 else
                m.get("formal_qa", {}).get("formal_qa_pass" if i == 2 else "all_formal_qa_pass"))
        check(good is True, f"Step{i} formal QA", checks)
        check(m.get("status") == ("NETWORK_SOURCE_PASS__PORTFOLIO_PENDING" if i == 1 else "FROZEN"),
              f"Step{i} frozen source status", checks)
        q = pd.read_csv(d / f"day16_step{i}_qa.csv", dtype=str).set_index("metric")["value"].to_dict()
        check(all(k in q and str(v).lower() == str(q[k]).lower()
                  for k, v in m["formal_qa"].items()), f"Step{i} QA/metadata agreement", checks)
        hashes[mn] = sha(mp)
        ledger.append({"stage": f"Step{i}", "file": mn, "sha256": hashes[mn], "kind": "metadata"})
        for name, expected in m["output_sha256"].items():
            actual = sha(d / name)
            check(actual == expected, f"Step{i} frozen bytes: {name}", checks)
            ledger.append({"stage": f"Step{i}", "file": name, "sha256": actual, "kind": "output"})
        meta.append(m)
    for i, m in enumerate(meta, 1):
        for fn, h in m.get("input_sha256", {}).items():
            if fn.startswith("day16_step") and fn.endswith("_metadata.json"):
                check(hashes.get(fn) == h, f"Step{i} parent link: {fn}", checks)
    check(meta[0]["formal_qa"]["portfolio_provided"] is False and
          meta[4]["formal_qa"]["actual_holdings"] is False,
          "synthetic portfolios only", checks)
    check(meta[1]["formal_qa"]["future_labels_used"] is False and
          meta[3]["formal_qa"]["no_future_labels"] is True and
          meta[4]["formal_qa"]["predictions_tested"] is False,
          "PIT descriptive / not predictive", checks)
    return base, dirs, meta, hashes, ledger


def summarize(dirs, meta, checks):
    inv = pd.read_csv(dirs[0] / "day16_step1_source_inventory.csv")
    check(len(inv) > 0 and inv.verified.astype(bool).all() and
          (inv.expected_sha256 == inv.actual_sha256).all(),
          "Step1 inherited source inventory internally consistent", checks)
    s2 = pd.read_csv(dirs[1] / "network_risk_exposure_daily_summary.csv")
    s3 = pd.read_csv(dirs[2] / "network_concentration_daily.csv")
    pair = pd.read_csv(dirs[2] / "raw_residual_common_sample.csv")
    crowd = pd.read_csv(dirs[2] / "synthetic_portfolio_crowding.csv")
    stress = pd.read_csv(dirs[3] / "network_stress_daily.csv")
    contrasts = pd.read_csv(dirs[3] / "network_stress_contrast.csv")
    holdings = pd.read_csv(dirs[3] / "stress_portfolio_holdings.csv")
    profile = pd.read_csv(dirs[4] / "m1_m4_portfolio_risk_profile.csv")
    m4 = pd.read_csv(dirs[4] / "m1_m4_hypothetical_stress_interface.csv")
    attr = pd.read_csv(dirs[4] / "incremental_structural_attribution.csv")
    stock = pd.read_csv(dirs[4] / "m1_m4_stock_risk_exposure.csv")
    exact(s2, ["analysis_date", "network", "window"], 624, "Step2", checks)
    exact(s3, ["analysis_date", "network", "window"], 624, "Step3 structure", checks)
    exact(pair, ["analysis_date", "window"], 312, "Step3 matched stocks", checks)
    exact(crowd, ["analysis_date", "network"], 208, "Step3 portfolios", checks)
    exact(stress, ["analysis_date", "network", "portfolio", "shock_scenario", "gamma"], 4992, "Step4 stress", checks)
    exact(contrasts, ["analysis_date", "portfolio", "shock_scenario", "gamma"], 2496, "Step4 contrasts", checks)
    exact(holdings, ["analysis_date", "portfolio", "master_index"], 6240, "Step4 holdings", checks)
    exact(profile, ["analysis_date", "network", "portfolio"], 416, "Step5 profile", checks)
    exact(m4, ["analysis_date", "network", "portfolio", "shock_scenario", "gamma"], 4992, "Step5 M4", checks)
    exact(attr, ["analysis_date", "portfolio", "shock_scenario", "gamma"], 2496, "Step5 attribution", checks)
    exact(stock, ["analysis_date", "network", "portfolio", "master_index"], 12480, "Step5 stocks", checks)
    check(s2.analysis_date.nunique() == 104 and set(s2.network) == {RAW, RES} and
          set(s2.window) == {60, 120, 252}, "104 dates and six network settings", checks)
    check(int(s2.nodes.sum()) == int(meta[1]["formal_qa"]["network_node_rows"]) and
          int(s2.neighbor_valid.sum()) == int(meta[1]["formal_qa"]["valid_neighbor_exposure_rows"]),
          "Step2 row/coverage arithmetic", checks)
    check(set(stress.gamma) == GAMMAS and set(stress.shock_scenario) ==
          {"STOCK", "SECTOR", "RESIDUAL_EGO15"} and set(stress.portfolio) == set(PORTS),
          "fixed stress scenarios and gamma", checks)
    check(np.allclose(stress.stress_loss - stress.initial_loss, stress.incremental_loss,
                      atol=1e-11, rtol=0), "Step4 pressure-loss identity", checks)
    check(np.allclose(stress.loc[stress.gamma == 0, "incremental_loss"], 0, atol=1e-11),
          "gamma zero baseline", checks)
    k = ["analysis_date", "network", "portfolio", "shock_scenario", "gamma"]
    merged = stress.merge(m4, on=k, how="inner", validate="one_to_one", suffixes=("_s4", "_m4"))
    check(len(merged) == 4992 and all(np.allclose(merged[f"{f}_s4"], merged[f"{f}_m4"],
                                                  atol=1e-11, rtol=0)
             for f in ("initial_loss", "stress_loss", "incremental_loss")),
          "M4 stress exact carry-through", checks)
    kc = ["analysis_date", "portfolio", "shock_scenario", "gamma"]
    merged_c = contrasts.merge(attr, on=kc, how="inner", validate="one_to_one", suffixes=("_s4", "_m4"))
    check(len(merged_c) == 2496 and np.allclose(
          merged_c.residual_minus_raw_loss_s4, merged_c.residual_minus_raw_loss_m4,
          atol=1e-11, rtol=0), "M4 paired contrasts unchanged", checks)
    partial = profile.m4_exposure_status.eq("PARTIAL_COVERAGE")
    check(int(partial.sum()) == int(meta[4]["formal_qa"]["partial_coverage_groups"]) and
          (profile.loc[partial, "network"] == RAW).all() and
          (profile.loc[partial, "portfolio"] == PORTS[1]).all(),
          "partial coverage explicitly flagged", checks)
    check(not m4.m4_prediction_validated.astype(bool).any() and
          not m4.m4_causal_identified.astype(bool).any() and
          m4.hypothetical_only.astype(bool).all() and
          not stock.real_holdings.astype(bool).any(),
          "no predictive, causal or real holdings claim", checks)
    check(np.allclose(profile.holding_hhi, 1/30, atol=1e-9) and
          np.allclose(profile.sector_hhi, 1/15, atol=1e-9),
          "traditional HHI constant by construction", checks)
    evidence = []
    def add(stage, metric, setting, value, unit, interpretation):
        evidence.append(dict(stage=stage, metric=metric, setting=setting,
                             value=float(value), unit=unit, interpretation=interpretation))
    add("Step1", "formation_rows", "PIT daily source", meta[0]["formal_qa"]["formation_stock_rows"],
        "stock-date rows", "input universe, not network-node count")
    add("Step2", "risk_valid_share", "all networks", s2.risk_valid.sum()/s2.nodes.sum(),
        "fraction", "historical stock downside risk coverage")
    add("Step2", "neighbor_valid_share", "all networks", s2.neighbor_valid.sum()/s2.nodes.sum(),
        "fraction", "neighbor exposure coverage, includes isolated nodes")
    for net in (RAW, RES):
        f = s3[(s3.window == 120) & (s3.network == net)]
        for col in ("isolated_share", "degree_gini", "hhi_connected_mean", "neighbor_risk_valid_share"):
            add("Step3", col, f"{net}/W120", f[col].mean(), "fraction", "date-equal structural diagnostic")
        f = crowd[crowd.network == net]
        add("Step3", "selected_minus_sector_null", f"{net}/W120 synthetic 30-stock",
            f.selected_minus_null.mean(), "fraction", "sector-matched null; exploratory selection")
    f = pair[pair.window == 120]
    add("Step3", "paired_hhi_res_minus_raw", "W120 common stocks",
        f.mean_hhi_res_minus_raw.mean(), "difference", "identical-date/stock comparison")
    for port in PORTS:
        for net in (RAW, RES):
            a = profile[(profile.portfolio == port) & (profile.network == net)]
            add("Step5", "neighbor_risk_weight_coverage", f"{port}/{net}",
                a.neighbor_risk_weight_coverage.mean(), "fraction", "coverage before M4 use")
            st = stress[(stress.portfolio == port) & (stress.network == net) &
                        (stress.shock_scenario == "SECTOR") & (stress.gamma == 0.3)]
            add("Step4", "sector_shock_incremental", f"{port}/{net}/gamma=.3",
                st.incremental_loss.mean(), "portfolio loss fraction", "hypothetical model, not observed contagion")
    add("Step5", "partial_coverage_groups", "M4 profile", partial.sum(),
        "groups", "cannot treat these as full historical network exposure")
    evidence = pd.DataFrame(evidence)
    contract = pd.DataFrame([
        ("m1_m4_stock_risk_exposure.csv", "stock PIT", 12480, "formation-close only; synthetic holdings"),
        ("m1_m4_portfolio_risk_profile.csv", "portfolio PIT", 416, "m4_exposure_status=PIT_EXPOSURE_READY required for full profile"),
        ("m1_m4_hypothetical_stress_interface.csv", "hypothetical scenario", 4992, "window_x=window_y=120; normalize downstream"),
        ("incremental_structural_attribution.csv", "network contrast", 2496, "not a causal or predictive contribution"),
    ], columns=["source_file", "grain", "rows", "usage_constraint"])
    return evidence, contract, partial.sum(), s2, s3, profile


def report(e, partial, s2, s3, profile, checks):
    def val(metric, setting=None):
        f = e[e.metric == metric]
        if setting is not None:
            f = f[f.setting == setting]
        need(len(f) == 1, f"Missing summary {metric}/{setting}")
        return float(f.value.iloc[0])
    res = f"{RES}/W120"
    raw = f"{RAW}/W120"
    h = profile.groupby(["network", "portfolio"], sort=False).neighbor_risk_weight_coverage.mean()
    main = PORTS[0]; ctrl = PORTS[1]
    return f"""# Day16｜M1全市场关联网络风险研究：最终QA与科研总结

**最终状态：FINAL_QA_PASS / FROZEN**（共 {len(checks)} 项本轮最终QA）。本报告只汇总冻结结果，不重新构网、不重新拟合或新增显著性检验。

## 1. 研究目标与来源

Day16承接Day15的统计关联证据，建立形成时点可观测的网络风险暴露、结构集中度、研究性模拟组合压力测试和M1→M4接口。104个冻结形成日期；Step1形成日期原始股票记录为 {int(val('formation_rows')):,} 条；网络六组（Raw/Residual × W60/W120/W252）。真实组合持仓尚未提供。

## 2. Step1–Step5 冻结成果

| 阶段 | 正式交付与关键口径 |
|---|---|
| Step1 | 网络来源及收益/行业/市值审计通过；312份节点重新哈希，624份边存在、其中6份重新哈希；继承Day15边文件完整审计；真实持仓 PENDING。 |
| Step2 | 624个网络–日期单元，{int(s2.nodes.sum()):,}条股票网络节点记录；历史风险有效率 {val('risk_valid_share'):.2%}，邻居风险有效率 {val('neighbor_valid_share'):.2%}；全部312节点及624边指纹由Step2核验。 |
| Step3 | 624期结构统计、312期共同股票比较、208期研究性组合结构诊断；W120 Raw孤立率 {val('isolated_share',raw):.2%}，Residual {val('isolated_share',res):.2%}。 |
| Step4 | 4,992条假设性压力情景、2,496条网络配对差异；同一组合/冲击下γ=0损失一致；参数γ∈{{0,.1,.2,.3}}，5%初始股票冲击。 |
| Step5 | 12,480条股票级M4暴露、416条组合画像、4,992条M4情景接口和2,496条差异归因；其中 {partial} 组历史网络风险暴露覆盖不达80%。 |

## 3. 网络结构及增量风险描述

W120两网络密度和平均度相同，但Raw的度Gini为 {val('degree_gini',raw):.4f}，Residual为 {val('degree_gini',res):.4f}；Raw非孤立节点连接HHI均值 {val('hhi_connected_mean',raw):.4f}，Residual {val('hhi_connected_mean',res):.4f}。W120双方共同有效股票的连接HHI差（Residual−Raw）日期均值为 {val('paired_hhi_res_minus_raw'):.4f}。这些差异表明网络边的**分布和覆盖**有区别，而非总体密度不同。

研究性高Residual邻居风险30股组合的内部连边密度，相对同行业随机参考的平均超额值：Raw {val('selected_minus_sector_null',raw+' synthetic 30-stock'):.2%}，Residual {val('selected_minus_sector_null',res+' synthetic 30-stock'):.2%}。该组合依Residual网络暴露选股，因此只能解释为结构诊断，不能据此声称真实机构交易拥挤或未来预测更优。

两类组合均为30只等权、15行业各2只，持仓HHI=1/30、行业HHI=1/15，**这些传统指标相同是构造设定**。即使如此，网络邻居风险、连接集中程度及给定冲击下的假设损失可能不同。其“增量”仅指额外结构描述，非增量样本外预测证据。

## 4. 假设性压力模型与M4接口

压力模型 `loss = (I - gamma * P)^(-1) * z`，行标准化正边权；γ为事先规定的**情景参数**，并非估计出的实际传染系数。行业初始冲击、γ=0.3时，30股模拟组合的平均**增量压力损失**：

| 模拟组合 | Raw | Residual |
|---|---:|---:|
| 高Residual邻居风险 | {val('sector_shock_incremental',f'{main}/{RAW}/gamma=.3'):.4%} | {val('sector_shock_incremental',f'{main}/{RES}/gamma=.3'):.4%} |
| 行业均衡ID对照 | {val('sector_shock_incremental',f'{ctrl}/{RAW}/gamma=.3'):.4%} | {val('sector_shock_incremental',f'{ctrl}/{RES}/gamma=.3'):.4%} |

上述百分数是组合损失比例，并非百分点标签。Raw与Residual差异方向取决于冲击与持仓位置，不能简单解释成某一网络风险更高。

**M4接口约束：** `m1_m4_portfolio_risk_profile.csv` 的 `m4_exposure_status` 应为 `PIT_EXPOSURE_READY` 才能当作完整历史网络暴露；共有 {partial}/416 个 `PARTIAL_COVERAGE`，均位于Raw/行业均衡ID对照组合。`m1_m4_hypothetical_stress_interface.csv` 中 `window_x`,`window_y` 都为120，下游可合并为单个 `window`；接口必须保留 `hypothetical_only=True`、`m4_prediction_validated=False`、`m4_causal_identified=False`。

## 5. 与Day15边界一致的研究结论

- **获得的结果：** 全市场PIT网络风险特征、Raw/Residual结构差异、固定模拟组合的压力情景敏感性，以及规范的M1–M4接口。
- **未获得的结果：** 真实机构共同持仓拥挤程度、网络风险指标对未来损失的新增预测能力、因果风险传染系数。Day15的行业调整后回顾性OOS预测增量也未得到稳健支持。
- **不能外推：** 形成时点网络情景差异不等于实际组合未来损失；相关与先后发生不证明股票间因果传导。模拟持仓不代表客户真实组合。

## 6. 复现范围与下一步

本轮Step6重新哈希Day16各阶段metadata和已冻结输出，对关键规模、压力恒等式、Step4→Step5情景逐行传递进行复核；**并未重新解码或逐行重算Step2的约139MB原始Parquet因子，也未重新哈希Day1和Day4全部原始来源**，依赖上游QA。建议锁定本轮成果后，在真实持仓和新日期数据具备时分别开展组合级真实风险归因及独立新时期验证；外生冲击因果设计须另立项目。
"""


def main(root):
    checks = []
    base, dirs, metas, hashes, ledger = load_sources(root, checks)
    evidence, contract, partial, s2, s3, profile = summarize(dirs, metas, checks)
    out = base / "06_step6_final_qa_research_summary"
    inputs = dict(hashes)
    script_digest = sha(Path(__file__))
    if out.exists():
        m = json.loads((out / "day16_step6_metadata.json").read_text(encoding="utf-8"))
        need(m["status"] == "FROZEN" and m["script_sha256"] == script_digest and
             m["input_metadata_sha256"] == inputs, "Frozen Step6 code/input drift")
        for filename, digest in m["output_sha256"].items():
            need(sha(out / filename) == digest, f"Frozen Step6 result drift: {filename}")
        print(f"Day16 Step6: FROZEN RECHECK_PASS ({len(checks)} upstream checks)")
        return
    tmp = Path(tempfile.mkdtemp(prefix="day16_step6_", dir=base))
    try:
        write_csv(tmp / "day16_step6_evidence_matrix.csv", evidence)
        write_csv(tmp / "day16_step6_m4_contract.csv", contract)
        write_csv(tmp / "day16_step6_hash_chain.csv", pd.DataFrame(ledger))
        write_csv(tmp / "day16_step6_qa.csv", pd.DataFrame(checks))
        write_bytes(tmp / "day16_step6_final_summary.md",
                    report(evidence, partial, s2, s3, profile, checks).encode("utf-8"))
        names = ["day16_step6_evidence_matrix.csv", "day16_step6_m4_contract.csv",
                 "day16_step6_hash_chain.csv", "day16_step6_qa.csv", "day16_step6_final_summary.md"]
        m = dict(research_day=16, step="Step6_Final_QA_Research_Summary", status="FROZEN",
                 formal_qa=dict(all_final_qa_pass=True, checks_passed=len(checks),
                                stages=5, portfolio_status="SYNTHETIC_ONLY",
                                m4_partial_coverage_groups=int(partial),
                                no_new_prediction_or_causal_test=True),
                 source_limit="Step6 verifies frozen bytes/summaries; upstream raw graph QA inherited",
                 script_sha256=script_digest, input_metadata_sha256=inputs,
                 output_sha256={fn: sha(tmp / fn) for fn in names})
        write_bytes(tmp / "day16_step6_metadata.json",
                    (json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        tmp.rename(out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(f"Day16 Step6: FINAL_QA_PASS / FROZEN | {len(checks)} checks | {len(ledger)} frozen hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root.expanduser().resolve())

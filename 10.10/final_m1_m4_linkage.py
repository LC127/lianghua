"""Day14 Step6 | Frozen M1-M4 linkage audit and conclusion (no refitting)."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    ap.add_argument("--step5-script", default="tail_event_high_low_validation.py")
    a = ap.parse_args()
    root = Path(a.root)
    base = root / "output" / "M1_day14"
    dirs = {
        1: "01_step1_m4_tail_risk_target_specification_freeze",
        2: "02_step2_m4_forward_tail_risk_labels",
        3: "03_step3_m1_m4_pit_alignment_summary",
        4: "04_step4_continuous_risk_state_validation",
        5: "05_step5_tail_event_high_low_validation",
    }
    stages = {i: base / name for i, name in dirs.items()}
    dest = base / "06_step6_final_m1_m4_linkage_summary"
    meta_name = {i: f"day14_step{i}_metadata.json" for i in stages}
    qa_name = {i: f"day14_step{i}_qa.csv" for i in stages}
    m1_dir = root / "output/M1_day13/06_step6_m1_v1_final_freeze"
    m1_meta = m1_dir / "m1_v1_0_freeze_metadata.json"
    manifest = m1_dir / "m1_v1_0_final_freeze_manifest.csv"
    audit = base / "04a_step4_code_version_audit/step4_code_version_audit.json"
    script = Path(a.step5_script)
    if not script.is_absolute():
        script = root / script

    # Map canonical Windows paths to --root (also enables isolated tests).
    def local(original):
        name = str(original).replace("\\", "/")
        prefix = "D:/M1_StockNetwork/"
        require(name.lower().startswith(prefix.lower()), f"Unexpected source: {original}")
        return root / name[len(prefix):]

    inputs = {}

    def check_file(path, expected=None):
        require(path.is_file(), f"Missing file: {path}")
        actual = sha(path)
        if expected is not None:
            require(actual == expected, f"SHA256 drift: {path}")
        inputs[str(path.relative_to(root))] = actual
        return path

    m1 = json.loads(check_file(m1_meta).read_text(encoding="utf-8"))
    require(m1.get("status") == "FROZEN" and m1.get("freeze_version") == "M1_v1.0"
            and m1.get("formal_qa", {}).get("all_formal_qa_pass") is True,
            "M1 v1.0 freeze invalid")
    mf = pd.read_csv(check_file(manifest))
    require(len(mf) == 18 and mf.relative_path.nunique() == 18, "M1 manifest invalid")
    # Recheck the canonical M1 interface, as recorded in the manifest.
    interface = local(m1["canonical_m1_to_m4_interface"])
    entry = mf[mf.relative_path.str.replace("\\", "/", regex=False) ==
               str(interface.relative_to(root)).replace("\\", "/")]
    require(len(entry) == 1, "M1 interface absent from manifest")
    check_file(interface, entry.iloc[0]["sha256"])
    context = root / "output/M1_day13/03_step3_risk_alpha_separation_summary/risk_system_oos_context.csv"
    context_row = mf[mf.relative_path.str.replace("\\", "/", regex=False) ==
                     str(context.relative_to(root)).replace("\\", "/")]
    require(len(context_row) == 1, "M1 OOS context absent from manifest")
    check_file(context, context_row.iloc[0]["sha256"])
    risk_context = pd.read_csv(context).sort_values("window")
    require(set(risk_context.window) == {60, 120, 252}, "M1 risk OOS windows invalid")

    step_meta = {}
    for i, folder in stages.items():
        metadata = json.loads(check_file(folder / meta_name[i]).read_text(encoding="utf-8"))
        require(metadata.get("status") == "FROZEN" and metadata.get("m1_version") == "M1_v1.0",
                f"Step{i} not frozen")
        qa = metadata.get("formal_qa", {})
        require(qa.get("all_formal_qa_pass") is True, f"Step{i} QA not PASS")
        q = pd.read_csv(check_file(folder / qa_name[i]), keep_default_na=False)
        observed = dict(zip(q.qa_name, q.qa_value))
        require(len(observed) == len(qa) and all(
            str(observed.get(k, "")).lower() == str(v).lower() for k, v in qa.items()
        ), f"Step{i} CSV QA / metadata mismatch")
        for key, expected in metadata.get("input_sha256", {}).items():
            check_file(local(key), expected)
        for name, expected in metadata.get("output_sha256", {}).items():
            check_file(folder / name, expected)
        step_meta[i] = metadata

    audit_json = json.loads(check_file(audit).read_text(encoding="utf-8"))
    require(audit_json.get("status") == "PASS" and
            audit_json.get("script_exact_match") is True and
            audit_json.get("frozen_outputs_match") is True and
            audit_json.get("frozen_inputs_match") is True,
            "Step4 independent code audit is not PASS")
    check_file(script, step_meta[5]["script_sha256"])

    def read(i, filename):
        return pd.read_csv(stages[i] / filename)

    reg = read(4, "m1_m4_continuous_regression.csv")
    oos = read(4, "m1_m4_primary_oos_summary.csv")
    hl = read(5, "m1_m4_high_low_hac_validation.csv")
    events = read(5, "m1_m4_tail_event_hit_rate.csv")
    contrasts = read(3, "m1_m4_high_low_contrast.csv")
    require(len(reg) == len(hl) == 18 and len(events) == 6 and len(oos) == 1,
            "Unexpected result dimensions")
    require(reg[["market_proxy", "target_id"]].duplicated().sum() == 0 and
            hl[["market_proxy", "target_id"]].duplicated().sum() == 0, "Duplicate tests")
    require(set(hl.market_proxy) == {"CSI300", "CSI500", "SSE_COMPOSITE"} and
            set(hl.target_id) == {f"T{i}" for i in range(1, 7)}, "Test universe drift")
    require(set(events.event) == {"P90", "P95"}, "Tail events drift")
    require(step_meta[1]["primary_market_proxy"] == "CSI300" and
            step_meta[1]["primary_target"] == "fwd_20d_max_drawdown_loss", "Primary changed")
    require(all(events.n_events == events.tp + events.fn) and
            all(events.n_valid == events.tp + events.fp + events.fn + events.tn),
            "Event confusion matrix invalid")
    check = hl.merge(contrasts, on=["market_proxy", "target_id"], suffixes=("", "_step3"))
    require(len(check) == 18 and
            (check.high_minus_low - check.high_minus_low_step3).abs().max() < 1e-10,
            "Step3/Step5 HIGH-LOW mismatch")

    rows = []
    for market in ["CSI300", "CSI500", "SSE_COMPOSITE"]:
        r = reg.query("market_proxy == @market and target_id == 'T1'").iloc[0]
        h = hl.query("market_proxy == @market and target_id == 'T1'").iloc[0]
        e90 = events.query("market_proxy == @market and event == 'P90'").iloc[0]
        e95 = events.query("market_proxy == @market and event == 'P95'").iloc[0]
        rows.append(dict(market_proxy=market, role="PRIMARY" if market == "CSI300" else "ROBUSTNESS",
                         target_id="T1", n=int(h.n_valid), high_minus_low=float(h.high_minus_low),
                         high_low_hac_p=float(h.hac_p_two_sided), high_low_bh_q=float(h.bh_q_18),
                         adjusted_beta_per_0p1=float(r.adjusted_beta_per_0p1),
                         adjusted_hac_p=float(r.adjusted_p), adjusted_bh_q=float(r.adjusted_bh_q_18),
                         p90_events=int(e90.n_events), p90_tp=int(e90.tp),
                         p90_precision=float(e90.precision), p95_events=int(e95.n_events)))
    evidence = pd.DataFrame(rows)
    primary = evidence.iloc[0]
    out_oos = oos.iloc[0]
    flags = dict(m1_v1_frozen=True, day14_steps_1_to_5_pass=True,
                 step4_code_audit_pass=True, step5_script_exact_match=True,
                 primary_preserved=True, primary_result_unique=True,
                 high_low_reproduced=True, no_reestimation=True,
                 no_result_based_selection=True,
                 p95_event_count=int(events.query("event == 'P95'").n_events.sum()),
                 primary_oos_points=int(out_oos.n_oos),
                 all_formal_qa_pass=True)
    qa = pd.DataFrame([{"qa_name": k, "qa_value": v} for k, v in flags.items()])
    change = 100 * float(primary.high_minus_low)
    oos_pct = 100 * float(out_oos.oos_mse_improvement)
    lines = [
        "# Day14 Step6｜M1–M4 Final QA & Linkage Summary", "",
        "**版本：M1 v1.0；Day14 Step1–Step5 已冻结、QA 通过。**", "",
        "## 主目标（CSI300，未来20交易日最大回撤）", "",
        f"- HIGH–LOW：**{change:+.4f} 个百分点**；HAC p={primary.high_low_hac_p:.4f}；BH q={primary.high_low_bh_q:.4f}。",
        f"- 传统市场风险控制后，网络评分系数（每增加0.1）={100*primary.adjusted_beta_per_0p1:+.4f} 个百分点；HAC p={primary.adjusted_hac_p:.4f}；BH q={primary.adjusted_bh_q:.4f}。",
        f"- 递推 OOS：{int(out_oos.n_oos)} 个预测点；MSE 改善={oos_pct:+.2f}%；损失差 HAC p={float(out_oos.loss_gain_hac_p_two_sided):.4f}。",
        f"- P90：{int(primary.p90_events)} 次事件，HIGH 命中 {int(primary.p90_tp)} 次；precision={primary.p90_precision:.2%}。P95：{int(primary.p95_events)} 次。", "",
        "## 跨市场结果（冻结的 T1）", "",
        "| 指数 | HIGH–LOW (pp) | HAC p | 控制后 HAC p | P90事件数 | P95事件数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['market_proxy']} | {r['high_minus_low']*100:+.4f} | {r['high_low_hac_p']:.4f} | {r['adjusted_hac_p']:.4f} | {r['p90_events']} | {r['p95_events']} |")
    lines += ["", "## M1 冻结的严格样本外风险信息（不同预测任务）", "",
              "| 网络窗口 | future-IVOL 增量 OOS IC | HAC t |",
              "|---|---:|---:|"]
    for _, rr in risk_context.iterrows():
        lines.append(f"| W{int(rr.window)} | {rr.oos_all_network_delta_ic:+.5f} | {rr.oos_all_network_hac_t:.3f} |")
    lines += ["", "## 严格结论与限制", "", 
              "- M1 网络变量在个股未来风险预测中具有既有的严格 OOS 增量证据；这一结论不能自动外推到 M4 市场尾部事件预测。",
              "- CSI300 主目标描述性 HIGH–LOW 差异为正，但控制后未达5%显著，递推样本外 MSE 未改善；不能宣称已验证市场尾部风险增量预警能力。",
              "- P90 事件极少且 P95 无事件；高 Recall 不代表可靠预警。CSI500 未呈现一致的 HIGH–LOW 方向。",
              "- Step4/Step5 推断属探索性，预测窗口可能重叠；不构成因果风险传染或具体传导路径识别。",
              "- 不改写 M1 v1.0、目标、阈值、指数、成本、方向或已冻结研究结果。", "",
              "**状态：DAY14_STEP6_FINAL_QA_COMPLETE；M1–M4 市场尾部联动验证已形成阶段性结论，但‘尾部风险传导路径’仍未识别。**", ""]
    report = "\n".join(lines)
    dest.mkdir(parents=True, exist_ok=True)
    outputs = {"m1_m4_final_linkage_evidence.csv": lambda p: evidence.to_csv(p, index=False, encoding="utf-8-sig"),
               "day14_step6_qa.csv": lambda p: qa.to_csv(p, index=False, encoding="utf-8-sig"),
               "m1_m4_final_linkage_summary.md": lambda p: p.write_text(report, encoding="utf-8")}
    frozen = dest / "day14_step6_metadata.json"
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding="utf-8"))
        require(old.get("inputs_sha256") == inputs and old.get("script_sha256") == sha(Path(__file__)),
                "Frozen Step6 input/code drift")
        for name, digest in old["outputs_sha256"].items():
            require(sha(dest / name) == digest, f"Step6 output drift: {name}")
        require(old.get("formal_qa", {}).get("all_formal_qa_pass") is True, "Frozen QA invalid")
        print("Step6 verified; existing outputs unchanged:", dest)
        return
    require(not any((dest / name).exists() for name in outputs), "Partial Step6 freeze exists")
    for name, save in outputs.items():
        tmp = dest / (name + ".tmp")
        save(tmp)
        os.replace(tmp, dest / name)
    metadata = {"research_day": 14, "step": "Step6_Final_QA_M1_M4_Linkage_Summary",
                "status": "FROZEN", "created_utc": datetime.now(timezone.utc).isoformat(),
                "m1_version": "M1_v1.0", "primary": "CSI300_T1",
                "interpretation": "Exploratory; market tail-risk warning not validated; no causal transmission claim",
                "script_sha256": sha(Path(__file__)), "inputs_sha256": inputs,
                "outputs_sha256": {name: sha(dest / name) for name in outputs}, "formal_qa": flags}
    tmp = dest / "day14_step6_metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, frozen)
    print("Day14 Step6 PASS | evidence=3 markets | frozen:", dest)


if __name__ == "__main__":
    main()

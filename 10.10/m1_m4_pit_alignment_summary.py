from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

STATES = ("LOW_NETWORK_RISK", "MID_NETWORK_RISK", "HIGH_NETWORK_RISK")


def require(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_digest(hashes, path):
    found = [v for k, v in hashes.items()
             if k.replace("\\", "/").split("/")[-1] == path.name]
    require(len(found) == 1 and sha(path) == found[0], f"SHA256 mismatch: {path}")


def dates(df, col):
    df[col] = pd.to_datetime(df[col], errors="raise").dt.normalize()
    require(df[col].notna().all(), f"Invalid date in {col}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    root = Path(ap.parse_args().root)
    base = root / "output"
    s5 = base / "M1_day13" / "05_step5_m1_m4_frozen_interface"
    fz = base / "M1_day13" / "06_step6_m1_v1_final_freeze"
    s1 = base / "M1_day14" / "01_step1_m4_tail_risk_target_specification_freeze"
    s2 = base / "M1_day14" / "02_step2_m4_forward_tail_risk_labels"
    out = base / "M1_day14" / "03_step3_m1_m4_pit_alignment_summary"

    interface = s5 / "m1_to_m4_frozen_interface_panel.csv"
    manifest = fz / "m1_v1_0_final_freeze_manifest.csv"
    m1meta = fz / "m1_v1_0_freeze_metadata.json"
    specfile = s1 / "m4_tail_risk_target_specification.csv"
    specmeta = s1 / "day14_step1_metadata.json"
    labelsfile = s2 / "m4_forward_tail_risk_labels.csv"
    calfile = s2 / "m4_tail_event_calibration.csv"
    qa2file = s2 / "day14_step2_qa.csv"
    meta2file = s2 / "day14_step2_metadata.json"
    inputs = [interface, manifest, m1meta, specfile, specmeta,
              labelsfile, calfile, qa2file, meta2file]
    for p in inputs:
        require(p.is_file(), f"Missing frozen input: {p}")

    # Freeze gates: verify canonical M1 interface and every frozen Step2 artifact.
    m1, sm, m2 = map(read_json, (m1meta, specmeta, meta2file))
    for meta, label in ((m1, "M1"), (sm, "Step1"), (m2, "Step2")):
        require(meta.get("status") == "FROZEN"
                and meta.get("formal_qa", {}).get("all_formal_qa_pass") is True,
                f"{label}: freeze QA failed")
    require(m1.get("freeze_version") == "M1_v1.0"
            and sm.get("m1_version") == "M1_v1.0"
            and m2.get("m1_version") == "M1_v1.0", "Wrong M1 version")
    man = pd.read_csv(manifest, dtype=str)
    rel = interface.relative_to(root).as_posix()
    matches = man.loc[man.relative_path.str.replace("\\", "/", regex=False) == rel, "sha256"]
    require(len(matches) == 1 and matches.iloc[0] == sha(interface),
            "M1 interface differs from v1.0 manifest")
    for p in (interface, specfile, specmeta):
        verify_digest(m2["input_sha256"], p)
    for p in (labelsfile, calfile, qa2file):
        verify_digest(m2["output_sha256"], p)

    # Second run verifies locked inputs/outputs without overwriting the freeze.
    outfiles = [out / "m1_m4_pit_aligned_panel.csv",
                out / "m1_m4_conditional_tail_risk_summary.csv",
                out / "m1_m4_high_low_contrast.csv",
                out / "day14_step3_qa.csv"]
    metafile = out / "day14_step3_metadata.json"
    signatures = {str(p): sha(p) for p in inputs}
    if any(p.exists() for p in [*outfiles, metafile]):
        require(all(p.is_file() for p in [*outfiles, metafile]),
                "Incomplete Step3 freeze; manual audit needed")
        old = read_json(metafile)
        require(old.get("input_sha256") == signatures
                and old.get("output_sha256") == {p.name: sha(p) for p in outfiles}
                and old.get("formal_qa", {}).get("all_formal_qa_pass") is True,
                "Step3 frozen inputs/outputs drifted; refusing overwrite")
        print("Day14 Step3: existing freeze verified, unchanged")
        return

    state = pd.read_csv(interface, encoding="utf-8-sig")
    label = pd.read_csv(labelsfile, encoding="utf-8-sig")
    spec = pd.read_csv(specfile, encoding="utf-8-sig")
    for df in (state, label):
        dates(df, "analysis_date")
    require(state.analysis_date.is_unique
            and not label.duplicated(["market_proxy", "analysis_date"]).any(),
            "Duplicate alignment keys")
    require(len(spec) == 8 and spec.target_name.is_unique
            and (spec.tier == "PRIMARY").sum() == 1,
            "Step1 targets no longer have the frozen structure")
    markets = [sm["primary_market_proxy"], *sm["robustness_market_proxies"]]
    require(set(label.market_proxy) == set(markets)
            and len(label) == len(state) * len(markets), "Market/date coverage mismatch")
    for market in markets:
        require(set(label.loc[label.market_proxy == market, "analysis_date"])
                == set(state.analysis_date), f"{market}: missing formation dates")

    score = pd.to_numeric(state.m1_network_risk_state_score, errors="raise")
    require(score.notna().all() and score.between(0, 1).all(), "Invalid state score")
    expected = np.select([score <= 1/3, score <= 2/3], STATES[:2], default=STATES[2])
    require((state.m1_network_risk_state == expected).all(), "State threshold mismatch")
    dates(state, "m1_state_valid_from")
    state["m1_state_valid_to_exclusive"] = pd.to_datetime(
        state.m1_state_valid_to_exclusive, errors="coerce")
    opened = state.m1_state_open_ended.astype(str).str.lower().map(
        {"true": True, "false": False})
    require(opened.notna().all() and opened.sum() == 1
            and (state.m1_state_valid_from == state.analysis_date).all()
            and state.loc[~opened, "m1_state_valid_to_exclusive"].gt(
                state.loc[~opened, "analysis_date"]).all()
            and state.loc[opened, "m1_state_valid_to_exclusive"].isna().all(),
            "Invalid frozen state validity intervals")

    for horizon in (5, 20, 60):
        flag = f"available_{horizon}d"
        available = label[flag].astype(str).str.lower().map({"true": True, "false": False})
        end = f"label_end_{horizon}d"
        label[end] = pd.to_datetime(label[end], errors="coerce")
        columns = [end, *spec.loc[spec.horizon_td == horizon, "target_name"]]
        require(available.notna().all() and all(
            label[c].notna().eq(available).all() for c in columns)
            and label.loc[available, end].gt(label.loc[available, "analysis_date"]).all(),
            f"{horizon}d availability / future timing mismatch")
    joined = label.merge(state, on="analysis_date", how="left",
                         validate="many_to_one", indicator=True, sort=False)
    require(len(joined) == len(label) and joined._merge.eq("both").all(),
            "PIT exact-date join failed")
    joined = joined.drop(columns="_merge")

    # Descriptive statistics only: never choose a target/window based on outcomes.
    rows, contrasts = [], []
    for market in markets:
        panel = joined.loc[joined.market_proxy == market]
        for target in spec.itertuples(index=False):
            means, counts = {}, {}
            for st in STATES:
                sample = pd.to_numeric(panel.loc[
                    panel.m1_network_risk_state == st, target.target_name],
                    errors="raise").dropna()
                n = int((panel.m1_network_risk_state == st).sum())
                means[st], counts[st] = (float(sample.mean()) if len(sample) else np.nan), len(sample)
                rows.append(dict(market_proxy=market, target_id=target.target_id,
                                 tier=target.tier, target_name=target.target_name,
                                 horizon_td=int(target.horizon_td), risk_state=st,
                                 n_total=n, n_valid=len(sample), n_missing=n-len(sample),
                                 mean=means[st],
                                 median=float(sample.median()) if len(sample) else np.nan))
            low, high = STATES[0], STATES[2]
            contrasts.append(dict(market_proxy=market, target_id=target.target_id,
                                  tier=target.tier, target_name=target.target_name,
                                  n_low=counts[low], n_high=counts[high],
                                  mean_low=means[low], mean_high=means[high],
                                  high_minus_low=means[high]-means[low]))
    summary, contrast = pd.DataFrame(rows), pd.DataFrame(contrasts)
    qa = dict(m1_v1_verified=True, step1_step2_frozen_verified=True,
              market_count=len(markets), m1_date_count=len(state), aligned_rows=len(joined),
              duplicate_market_date_count=int(joined.duplicated([
                  "market_proxy", "analysis_date"]).sum()),
              summary_rows=len(summary), contrast_rows=len(contrast),
              available_20d_rows=int(joined.available_20d.astype(str).str.lower().eq("true").sum()),
              exact_date_join_only=True, future_end_strictly_after_formation=True,
              result_based_selection=False, inference_performed=False)
    qa["all_formal_qa_pass"] = (len(summary) == len(markets)*len(spec)*len(STATES)
                                 and len(contrast) == len(markets)*len(spec)
                                 and qa["duplicate_market_date_count"] == 0)
    require(qa["all_formal_qa_pass"], f"Step3 QA failed: {qa}")

    out.mkdir(parents=True, exist_ok=True)
    for p, df in zip(outfiles, (joined, summary, contrast,
                                pd.DataFrame(qa.items(), columns=["qa_name", "qa_value"]))):
        df.to_csv(p, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    metadata = dict(research_day=14, step="Step3_M1_M4_PIT_Alignment_Conditional_Summary",
                    status="FROZEN", m1_version="M1_v1.0",
                    join_rule="Exact match on analysis_date; no asof/forward-fill",
                    interpretation="Descriptive association only; overlapping horizons; no inference or causality",
                    primary_market=markets[0], robustness_markets=markets[1:],
                    input_sha256=signatures,
                    output_sha256={p.name: sha(p) for p in outfiles}, formal_qa=qa)
    metafile.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Day14 Step3 FROZEN / QA PASS | aligned={len(joined)}, "
          f"summary={len(summary)}, contrast={len(contrast)} | {out}")


if __name__ == "__main__":
    main()

"""Day15 Step2: PIT-safe exact-20-market-session stock downside labels; no network refit."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WINS = (60, 120, 252)
H = 20


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    args = ap.parse_args()
    root = Path(args.root)
    day1 = root / "output/M1_day1"
    day4 = root / "output/M1_day4/02_step2_matched_networks"
    step1 = root / "output/M1_day15/01_step1_frozen_network_risk_source_audit"
    out = root / "output/M1_day15/02_step2_stock_downside_target_freeze"
    out.mkdir(parents=True, exist_ok=True)
    src = {
        "step1_meta": step1 / "day15_step1_metadata.json",
        "master": day1 / "01_stage1_final/stock_master.parquet",
        "mapping": day4 / "stock_master_mapping.parquet",
        "returns": day1 / "03_stage3_return_validation/daily_return_panel.parquet",
        "calendar": day1 / "01_stage1_final/trade_calendar_used.csv",
        "design": root / "output/M1_day4/01_step1_network_design/day4_network_design_config.json",
        "interface": root / "output/M1_day13/05_step5_m1_m4_frozen_interface/m1_to_m4_frozen_interface_panel.csv",
    }
    for k, p in src.items():
        require(p.is_file(), f"Missing {k}: {p}")
    s1 = json.loads(src["step1_meta"].read_text(encoding="utf-8"))
    require(s1.get("status") == "SOURCE_AUDIT_PASS"
            and s1.get("formal_qa", {}).get("all_formal_qa_pass") is True,
            "Day15 Step1 is not PASS")
    for rel, sha in s1["source_sha256"].items():
        p = root / rel.replace("\\", "/")
        require(p.is_file() and digest(p) == sha, f"Step1 frozen source drift: {p}")
    cfg = json.loads(src["design"].read_text(encoding="utf-8"))
    network_rcol = cfg["data_design"]["return_column"]
    rcol = "return_network_simple"  # Day1 SIMPLE return; Day4 may use LOG return
    cal = pd.to_datetime(pd.read_csv(src["calendar"])["trade_date"], errors="coerce")
    require(cal.notna().all() and not cal.duplicated().any(), "Invalid trade calendar")
    dates = pd.DatetimeIndex(cal.sort_values().dt.normalize())
    require(dates.is_unique, "Duplicate normalized trading dates")
    formation = pd.to_datetime(pd.read_csv(src["interface"])["analysis_date"])
    require(formation.notna().all() and formation.is_unique and len(formation) == 104,
            "M1 analysis dates are not the frozen 104 unique dates")
    fi = dates.get_indexer(formation)
    require((fi >= 0).all(), "M1 formation date absent from Day1 trading calendar")

    mapping = pd.read_parquet(src["mapping"], columns=["master_index", "security_id"])
    mapping = mapping.sort_values("master_index").reset_index(drop=True)
    require(mapping.master_index.tolist() == list(range(len(mapping)))
            and mapping.security_id.is_unique, "Day4 mapping invalid")
    security = pd.Index(mapping.security_id.astype("string").str.strip())
    master = pd.read_parquet(src["master"], columns=["security_id", "delist_date"])
    master["security_id"] = master.security_id.astype("string").str.strip()
    require(master.security_id.is_unique, "Duplicate master security_id")
    delist = pd.to_datetime(master.set_index("security_id").reindex(security).delist_date,
                            errors="coerce").to_numpy(dtype="datetime64[ns]")

    # Frozen node sets only: build formation universe BEFORE consulting any future return.
    rows, node_manifest = [], []
    for t in formation.sort_values():
        memberships = []
        for w in WINS:
            p = day4 / "node_universe" / f"W{w}" / f"{t:%Y-%m-%d}.parquet"
            require(p.is_file(), f"Missing frozen node snapshot: {p}")
            n = pd.read_parquet(p, columns=["master_index", "security_id", "analysis_date", "window"])
            ids = n.master_index.to_numpy(dtype=np.int32)
            require((pd.to_datetime(n.analysis_date) == t).all() and n.window.eq(w).all(),
                    f"Wrong frozen node snapshot identity: {p}")
            require(len(ids) and len(ids) == len(np.unique(ids))
                    and np.all((ids >= 0) & (ids < len(security)))
                    and np.array_equal(security.take(ids).to_numpy(),
                                       n.security_id.astype("string").str.strip().to_numpy()),
                    f"Day4 node mapping mismatch: {p}")
            memberships.append(ids)
            node_manifest.append(dict(path=str(p.relative_to(root)), window=w,
                                      analysis_date=f"{t:%Y-%m-%d}", n_nodes=len(ids),
                                      sha256=digest(p)))
        ids = np.unique(np.concatenate(memberships))
        date_panel = pd.DataFrame({"analysis_date": t, "master_index": ids,
                                   "security_id": security.take(ids).to_numpy()})
        for w, members in zip(WINS, memberships):
            date_panel[f"in_w{w}"] = np.isin(ids, members).astype(np.int8)
        rows.append(date_panel)
    panel = pd.concat(rows, ignore_index=True)

    # Read Day1 simple return in batches; dense float64 ~140 MB.
    # Do not compound Day4 log-return column as if it were a simple return.
    pf = pq.ParquetFile(src["returns"])
    required_ret_cols = {"trade_date", "security_id", rcol, "is_open", "is_consecutive_open_pair"}
    require(required_ret_cols.issubset(pf.schema_arrow.names),
            f"Daily-return schema lacks trade_date/security_id/{rcol}")
    R = np.full((len(dates), len(security)), np.nan, dtype=np.float64)
    seen = np.zeros(R.shape, dtype=bool)
    dup = invalid = masked = 0
    for batch in pf.iter_batches(batch_size=250_000,
                                 columns=["trade_date", "security_id", rcol,
                                          "is_open", "is_consecutive_open_pair"]):
        b = batch.to_pandas()
        ti = dates.get_indexer(pd.to_datetime(b.trade_date, errors="coerce").dt.normalize())
        si = security.get_indexer(b.security_id.astype("string").str.strip())
        keep = (ti >= 0) & (si >= 0)
        require(keep.all(), "Returns include unknown security or non-calendar date")
        ti, si = ti[keep], si[keep]
        v = pd.to_numeric(b.loc[keep, rcol], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan, copy=True
        )  # Writable copy: Arrow-backed arrays can be read-only
        valid_session = (b.loc[keep, "is_open"].eq(True).to_numpy(dtype=bool)
                         & b.loc[keep, "is_consecutive_open_pair"].eq(True).to_numpy(dtype=bool))
        masked += int((~valid_session & np.isfinite(v)).sum())
        v[~valid_session] = np.nan
        invalid += int(((v < -1) | np.isinf(v)).sum())
        keys = ti.astype(np.int64) * len(security) + si
        dup += int(seen[ti, si].sum()) + (len(keys) - len(np.unique(keys)))
        R[ti, si] = v
        seen[ti, si] = True
    require(dup == 0 and invalid == 0, f"Bad return input: duplicate={dup}, ret<-1={invalid}")

    # Exactly (t, t+20] market sessions, NOT next 20 observed stock returns.
    results = []
    for t, ix in zip(formation.sort_values(), dates.get_indexer(formation.sort_values())):
        group = panel.loc[panel.analysis_date.eq(t)].copy()
        ids = group.master_index.to_numpy(dtype=np.int32)
        if ix + H >= len(dates):
            group["label_end_date"] = pd.NaT
            group["observed_sessions"] = 0
            group["forward_return_20d"] = np.nan
            group["downside_loss_20d"] = np.nan
            group["label_status"] = "CALENDAR_RIGHT_CENSORED"
        else:
            end = dates[ix + H]
            x = R[ix + 1:ix + H + 1, :][:, ids]
            counts = np.isfinite(x).sum(axis=0)
            gross = np.prod(1.0 + x, axis=0, dtype=np.float64) - 1.0
            censored = (~pd.isna(delist[ids])) & (delist[ids] > np.datetime64(t)) & (delist[ids] <= np.datetime64(end))
            good = (counts == H) & np.isfinite(gross) & ~censored
            y = np.where(good, gross, np.nan)
            group["label_end_date"] = end
            group["observed_sessions"] = counts
            group["forward_return_20d"] = y
            group["downside_loss_20d"] = np.maximum(0.0, -y)
            group["label_status"] = np.where(censored, "DELISTED_WITHIN_HORIZON",
                                              np.where((counts < H) | ~np.isfinite(gross), "MISSING_OR_INVALID_RETURN", "OK"))
        results.append(group)
    labels = pd.concat(results, ignore_index=True)
    labels["label_available"] = labels.label_status.eq("OK")
    qa = {
        "step1_source_audit_pass": True,
        "frozen_formation_dates": len(formation),
        "frozen_node_snapshots": len(node_manifest),
        "duplicate_formation_stock": int(labels.duplicated(["analysis_date", "security_id"]).sum()),
        "bad_forward_label_dates": int(((labels.label_end_date <= labels.analysis_date)
                                         & labels.label_available).sum()),
        "bad_loss_values": int(((labels.downside_loss_20d < 0)
                                & labels.label_available).sum()),
        "universe_rows": len(labels),
        "available_labels": int(labels.label_available.sum()),
        "unavailable_labels": int((~labels.label_available).sum()),
        "calendar_right_censored_n": int(labels.label_status.eq("CALENDAR_RIGHT_CENSORED").sum()),
        "delist_censored_n": int(labels.label_status.eq("DELISTED_WITHIN_HORIZON").sum()),
        "missing_or_invalid_return_n": int(labels.label_status.eq("MISSING_OR_INVALID_RETURN").sum()),
        "simple_return_column": rcol,
        "network_return_column_from_day4_design": network_rcol,
        "nontrade_or_reopen_returns_masked": masked,
        "no_forward_universe_filtering": True,
        "no_network_reestimation": True,
    }
    for w in WINS:
        qa[f"w{w}_nodes_preserved"] = (int(labels[f"in_w{w}"].sum()) ==
                                     sum(x["n_nodes"] for x in node_manifest if x["window"] == w))
    qa["all_formal_qa_pass"] = bool(qa["duplicate_formation_stock"] == 0
        and qa["bad_forward_label_dates"] == 0 and qa["bad_loss_values"] == 0
        and qa["universe_rows"] > 0 and qa["available_labels"] > 0
        and all(qa[f"w{w}_nodes_preserved"] for w in WINS))
    require(qa["all_formal_qa_pass"], f"Formal QA BLOCKED: {qa}")

    inputs = {str(p.relative_to(root)): digest(p) for p in src.values()}
    script_sha = digest(Path(__file__))
    oldpath = out / "day15_step2_metadata.json"
    old = json.loads(oldpath.read_text(encoding="utf-8")) if oldpath.exists() else None
    if old and old.get("status") == "FROZEN":
        require(old["script_sha256"] == script_sha and old["input_sha256"] == inputs,
                "Frozen Step2 script/input drift")
    products = {
        "stock_forward_downside_20d_panel.csv": labels.to_csv(index=False, float_format="%.10g"),
        "day15_step2_node_snapshot_manifest.csv": pd.DataFrame(node_manifest).to_csv(index=False),
        "day15_step2_qa.csv": pd.DataFrame([{"qa_name": k, "qa_value": v} for k, v in qa.items()]).to_csv(index=False),
    }
    hashes = {}
    for name, content in products.items():
        b = content.encode("utf-8")
        hashes[name] = hashlib.sha256(b).hexdigest()
        p = out / name
        if old and old.get("status") == "FROZEN":
            require(p.is_file() and digest(p) == hashes[name]
                    and old["output_sha256"][name] == hashes[name], f"Frozen Step2 output drift: {p}")
        else:
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(b)
            os.replace(tmp, p)
    metadata = dict(research_day=15, step="Step2_Stock_Forward_Downside_Target_Freeze",
        status="FROZEN", horizon_market_sessions=H,
        label_definition="max(0,-prod(1+daily_return[t+1:t+20])+1)",
        return_column=rcol, day4_network_return_column=network_rcol,
        primary_label="downside_loss_20d",
        delist_policy="mark NA if delisting occurs inside horizon; no invented delisting return",
        missing_policy="all 20 exchange sessions required; no forward-fill/zero-fill",
        universe_policy="union of PIT frozen Day4 node universes; window membership retained",
        no_reestimation=True, input_sha256=inputs,
        script_sha256=script_sha, output_sha256=hashes, formal_qa=qa,
        limitation="One-sided end-to-end holding loss, not intrahorizon MDD; missing sessions remain censored.")
    if old and old.get("status") == "FROZEN":
        require(old == metadata, "Existing frozen Step2 metadata drift")
    else:
        tmp = oldpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, oldpath)
    print(f"Day15 Step2: FROZEN / PASS | stocks x dates={len(labels):,} | valid={qa['available_labels']:,}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()

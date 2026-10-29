
# -*- coding: utf-8 -*-
"""Day18: Final stock-master downstream consistency audit."""
import argparse, os, re, csv
from pathlib import Path
from datetime import datetime
import pandas as pd

SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules", "M1_day18"}
KEYS = ["security_id", "ts_code", "stock_code", "security_code",
        "wind_code", "symbol", "ticker", "stock_id"]
PAIRS = [("source","target"), ("src","dst"), ("from","to"),
         ("node_i","node_j"), ("stock_i","stock_j"),
         ("stock_a","stock_b"), ("stock1","stock2"),
         ("security_id_i","security_id_j"), ("source_id","target_id"),
         ("u","v"), ("i","j")]


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def canon(s):
    return (s.astype("string").str.strip().replace("", pd.NA)
            .str.replace(r"^(\d+)\.0$", r"\1", regex=True))


def keycol(df):
    cols = {norm(c): c for c in df.columns}
    return next((cols[norm(k)] for k in KEYS if norm(k) in cols), None)


def read(p):
    return (pd.read_parquet(p) if p.suffix.lower() == ".parquet"
            else pd.read_csv(p, dtype="string"))


def locate(root, name, parent=None):
    hits = [p for p in root.rglob(name) if p.is_file()
            and not any(s in p.parts for s in SKIP)]
    if parent:
        hits = [p for p in hits if p.parent.name.lower() == parent.lower()]
    return (hits[0] if len(hits) == 1 else None), hits


def save(path, rows):
    fields = list(rows[0]) if rows else ["status", "detail"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Final downstream consistency audit")
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    ap.add_argument("--return-panel", default="")
    ap.add_argument("--mapping", default="")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"BLOCKED: root not found: {root}")

    out = root / "output" / "M1_day18" / "04_final_consistency_audit"
    out /= datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)

    # 1. Load the Final stock master.
    master_path = root / "output" / "M1_day1" / "01_stage1_final" / "stock_master.parquet"
    if not master_path.is_file():
        master_path, hits = locate(root, "stock_master.parquet", "01_stage1_final")
        if master_path is None:
            raise SystemExit(f"BLOCKED: Final master missing/ambiguous: {hits}")

    master = pd.read_parquet(master_path)
    mk = keycol(master)
    if not mk:
        raise SystemExit(f"BLOCKED: no security ID column: {master_path}")
    mids = set(canon(master[mk]).dropna())
    checks = []

    # 2. Compare the return panel and mapping table with Final.
    for kind, given, name in [
        ("return_panel", args.return_panel, "daily_return_panel.parquet"),
        ("mapping", args.mapping, "stock_master_mapping.parquet")
    ]:
        if given:
            p = Path(given).resolve()
            hits = [p] if p.is_file() else []
        else:
            p, hits = locate(root, name)

        if p is None or not hits:
            checks.append(dict(
                object=kind, status="MISSING_OR_AMBIGUOUS", path="",
                master_key=mk, object_key="", master_unique=len(mids),
                object_unique="", overlap="", master_only="", object_only="",
                detail=f"{len(hits)} candidates; pass explicit path"
            ))
            continue

        try:
            df = read(p)
            k = keycol(df)
            if not k:
                checks.append(dict(
                    object=kind, status="NO_SECURITY_ID_COLUMN", path=str(p),
                    master_key=mk, object_key="", master_unique=len(mids),
                    object_unique="", overlap="", master_only="", object_only="",
                    detail=str(list(df.columns))
                ))
                continue

            ids = set(canon(df[k]).dropna())
            checks.append(dict(
                object=kind, status="CHECKED", path=str(p),
                master_key=mk, object_key=k, master_unique=len(mids),
                object_unique=len(ids), overlap=len(mids & ids),
                master_only=len(mids - ids), object_only=len(ids - mids),
                detail="Interpret differences using universe/date rules."
            ))

            if kind == "mapping":
                ic = next((c for c in df.columns if norm(c) == "masterindex"), None)
                if ic:
                    v = canon(df[ic])
                    nn = v.dropna()
                    checks.append(dict(
                        object="mapping_master_index", status="CHECKED",
                        path=str(p), master_key=mk, object_key=ic,
                        master_unique=len(mids), object_unique=nn.nunique(),
                        overlap="", master_only="", object_only="",
                        detail=f"missing={v.isna().sum()}; duplicates={nn.duplicated().sum()}"
                    ))
        except Exception as e:
            checks.append(dict(
                object=kind, status=f"READ_ERROR: {e}", path=str(p),
                master_key=mk, object_key="", master_unique=len(mids),
                object_unique="", overlap="", master_only="", object_only="",
                detail=""
            ))

    # 3. Load mapping IDs and master_index, if available.
    mapping_ids, mapping_idx = set(), set()
    mc = next((x for x in checks
               if x["object"] == "mapping" and x["status"] == "CHECKED"), None)
    if mc:
        try:
            md = read(Path(mc["path"]))
            k = keycol(md)
            if k:
                mapping_ids = set(canon(md[k]).dropna())
            ic = next((c for c in md.columns if norm(c) == "masterindex"), None)
            if ic:
                mapping_idx = set(canon(md[ic]).dropna())
        except Exception:
            pass

    # 4. Find Day4 network/edge files and inspect recognized endpoint pairs.
    files4 = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            p = Path(folder) / name
            low = str(p).lower()
            if (p.suffix.lower() in {".parquet", ".csv"}
                    and "day4" in low
                    and any(t in low for t in ("network", "edge"))):
                files4.append(p)

    net = []
    for p in sorted(set(files4)):
        try:
            df = read(p)
            cols = {norm(c): c for c in df.columns}
            pair = next(
                ((cols[norm(x)], cols[norm(y)]) for x, y in PAIRS
                 if norm(x) in cols and norm(y) in cols), None
            )
            if not pair:
                continue

            nodes = set(pd.concat([
                canon(df[pair[0]]), canon(df[pair[1]])
            ], ignore_index=True).dropna())

            if not nodes:
                mode, unknown, status = "EMPTY", "", "REVIEW_EMPTY"
            elif nodes <= mids:
                mode, unknown, status = "security_id", 0, "PASS"
            elif mapping_idx and nodes <= mapping_idx:
                mode, unknown, status = "master_index", 0, "PASS"
            elif mapping_ids and nodes <= mapping_ids:
                mode, unknown, status = "mapping_security_id", 0, "PASS"
            else:
                mode, unknown, status = "UNRESOLVED", len(nodes - mids), "REVIEW"

            net.append(dict(
                file=str(p), rows=len(df), endpoint_a=pair[0],
                endpoint_b=pair[1], unique_nodes=len(nodes),
                endpoint_type=mode, unknown_vs_master=unknown,
                status=status
            ))
        except Exception as e:
            net.append(dict(
                file=str(p), rows="", endpoint_a="", endpoint_b="",
                unique_nodes="", endpoint_type="READ_ERROR",
                unknown_vs_master="", status=str(e)
            ))

    # 5. Save the audit results.
    save(out / "final_panel_mapping_checks.csv", checks)
    save(out / "final_network_node_checks.csv", net)
    (out / "day18_final_consistency_report.md").write_text(
        "# Day18 Final 下游一致性核验\n\n"
        f"- Final 主表：`{master_path}`\n"
        f"- 证券键：`{mk}`；唯一证券数：{len(mids)}\n"
        f"- Day4 网络候选：{len(files4)}；识别并检查的边文件：{len(net)}\n\n"
        "输出：`final_panel_mapping_checks.csv`（收益面板/映射覆盖）；"
        "`final_network_node_checks.csv`（网络节点映射）。\n\n"
        "注意：主表独有或面板独有证券不自动判错，须结合股票池、"
        "日期范围和网络节点定义解释。未识别端点列的文件不会进入节点检查。"
        "差异全部解释清楚后才能冻结 Final。\n",
        encoding="utf-8"
    )
    print(f"Audit complete | Final securities={len(mids)}")
    print(f"Day4 candidates={len(files4)} | checked edge files={len(net)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()

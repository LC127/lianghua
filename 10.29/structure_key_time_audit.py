from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib, json, re
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"D:\M1_StockNetwork")
MANIFEST = ROOT / "output/M1_day18/03_step3_frozen_fingerprints/day18_frozen_source_manifest.csv"
OUT = ROOT / "output/M1_day18/04_step4_structure_key_time_audit" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
DATE_NAMES = {"analysisdate", "date", "tradedate", "datetime", "timestamp",
              "labelenddate", "nextanalysisdate", "startdate", "enddate",
              "eventdate", "reportdate", "asofdate", "effectivedate",
              "periodstart", "periodend", "targetdate", "monthend"}
EDGE_PAIRS = [("sourceindex","targetindex"), ("sourcemasterindex","targetmasterindex"),
              ("srcindex","dstindex"), ("sourceid","targetid"), ("src","dst"),
              ("source","target"), ("fromindex","toindex"), ("from","to"),
              ("u","v"), ("nodei","nodej"), ("i","j")]
ENTITY = ["masterindex","securityid","stockcode","ticker","symbol","permno",
          "assetid","instrumentid","stockid"]
ID_COLS = ["recordid","rowid","edgeid","id"]


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def parse_date(s):
    raw = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    out = pd.to_datetime(raw, errors="coerce")
    for pat, fmt, unit in [(r"\d{8}", "%Y%m%d", None), (r"\d{6}", "%Y%m", None),
                           (r"\d{10}", None, "s"), (r"\d{13}", None, "ms")]:
        m = raw.str.fullmatch(pat, na=False)
        if m.any():
            v = pd.to_numeric(raw[m], errors="coerce") if unit else raw[m]
            out.loc[m] = (pd.to_datetime(v, unit=unit, errors="coerce") if unit
                          else pd.to_datetime(v, format=fmt, errors="coerce"))
    return out


def infer(cols):
    d = {norm(c): c for c in cols}
    dates = [c for c in cols if norm(c) in DATE_NAMES or
             norm(c).endswith(("date", "datetime", "timestamp"))]
    date = next((d[x] for x in ["analysisdate","date","tradedate","datetime","timestamp"] if x in d), None)
    window = d.get("window") or d.get("windowdays")
    pair = next(((d[a], d[b]) for a,b in EDGE_PAIRS if a in d and b in d), None)
    if pair:
        key = ([date] if date else []) + ([window] if window else []) + list(pair)
    else:
        entity = next((d[x] for x in ENTITY if x in d), None)
        if entity:
            key = ([date] if date else []) + ([window] if window else []) + [entity]
        else:
            entity = next((d[x] for x in ID_COLS if x in d), None)
            key = [entity] if entity else []
    return key, dates, window, pair


def audit_file(p):
    r = dict(path=str(p), file_type=p.suffix.lower(), row_count="", column_count=0,
             columns="", column_types="", schema_sha256="", candidate_primary_key="", key_status="NOT_INFERRED",
             duplicate_key_rows="", null_key_rows="", self_loops="", duplicate_undirected_pairs="",
             date_columns="", time_min="", time_max="", invalid_date_values=0,
             window_values="", status="REVIEW", error="")
    if not p.is_file():
        r.update(status="ERROR", error="FILE_NOT_FOUND")
        return r, [], [("FILE_NOT_FOUND", "")]
    ext = p.suffix.lower()
    try:
        if ext == ".json":
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
            cols = list(obj) if isinstance(obj, dict) else []
            types = [type(obj[c]).__name__ for c in cols] if isinstance(obj, dict) else []
            sig = "|".join(f"{c}:{t}" for c,t in zip(cols,types))
            r.update(row_count=len(obj) if isinstance(obj, list) else 1,
                     column_count=len(cols), columns="|".join(cols), column_types="|".join(types),
                     schema_sha256=hashlib.sha256(sig.encode()).hexdigest(),
                     key_status="NOT_APPLICABLE", status="INFO")
            return r, [], []
        if ext in {".parquet", ".pq"}:
            pf = pq.ParquetFile(p); cols = pf.schema_arrow.names
            types = [str(f.type) for f in pf.schema_arrow]
            r["row_count"] = pf.metadata.num_rows
            enc = None
        elif ext == ".csv":
            enc = "utf-8-sig"
            try: cols = list(pd.read_csv(p, nrows=0, encoding=enc).columns)
            except UnicodeDecodeError:
                enc = "gb18030"; cols = list(pd.read_csv(p, nrows=0, encoding=enc).columns)
            sample = pd.read_csv(p, nrows=1000, encoding=enc, low_memory=False)
            types = [str(sample[c].dtype) for c in cols]
        else:
            return r, [], []

        key, dates, window, pair = infer(cols)
        need = list(dict.fromkeys(key + dates + ([window] if window else [])))
        if ext == ".csv" and not need and cols: need = [cols[0]]
        df = (pd.read_parquet(p, columns=need) if ext in {".parquet",".pq"} and need
              else pd.read_csv(p, usecols=need, encoding=enc, low_memory=False)
              if ext == ".csv" and need else None)
        if ext == ".csv": r["row_count"] = len(df) if df is not None else 0
        sig = "|".join(f"{c}:{t}" for c,t in zip(cols,types))
        r.update(column_count=len(cols), columns="|".join(map(str,cols)),
                 column_types="|".join(types),
                 schema_sha256=hashlib.sha256(sig.encode()).hexdigest(),
                 candidate_primary_key="|".join(key), key_status="INFERRED" if key else "NOT_INFERRED",
                 date_columns="|".join(dates))
        issues, times = [], []
        if df is not None and key:
            r["null_key_rows"] = int(df[key].isna().any(axis=1).sum())
            r["duplicate_key_rows"] = int(df.duplicated(key).sum())
            if r["null_key_rows"]: issues.append(("NULL_PRIMARY_KEY", r["null_key_rows"]))
            if r["duplicate_key_rows"]: issues.append(("DUPLICATE_PRIMARY_KEY", r["duplicate_key_rows"]))
        if df is not None and window and window in df:
            vals = sorted(df[window].dropna().astype(str).unique())
            r["window_values"] = "|".join(vals[:20]) + ("|..." if len(vals)>20 else "")
        mins, maxs = [], []
        if df is not None:
            for c in dates:
                parsed = parse_date(df[c]); valid = parsed.dropna()
                bad = int((df[c].notna() & parsed.isna()).sum())
                lo = valid.min().strftime("%Y-%m-%d") if len(valid) else ""
                hi = valid.max().strftime("%Y-%m-%d") if len(valid) else ""
                times.append(dict(path=str(p), date_column=c, valid_dates=len(valid),
                                  invalid_dates=bad, min_date=lo, max_date=hi))
                r["invalid_date_values"] += bad
                if lo: mins.append(lo)
                if hi: maxs.append(hi)
                if bad: issues.append(("INVALID_DATE_VALUES", f"{c}:{bad}"))
            r["time_min"], r["time_max"] = min(mins, default=""), max(maxs, default="")
            if pair:
                a, b = df[pair[0]], df[pair[1]]
                r["self_loops"] = int((a == b).sum())
                a, b = a.astype("string").fillna("<NA>"), b.astype("string").fillna("<NA>")
                lo, hi = a.where(a <= b, b), b.where(a <= b, a)
                r["duplicate_undirected_pairs"] = int(pd.DataFrame({"a":lo,"b":hi}).duplicated().sum())
                if r["self_loops"]: issues.append(("SELF_LOOPS", r["self_loops"]))
                if r["duplicate_undirected_pairs"]:
                    issues.append(("DUPLICATE_UNDIRECTED_PAIRS", r["duplicate_undirected_pairs"]))
        if not key: issues.append(("KEY_NOT_INFERRED", "需按表语义人工确认"))
        r["status"] = "REVIEW" if not key or any(k != "KEY_NOT_INFERRED" for k,_ in issues) else "PASS"
        return r, times, issues
    except Exception as e:
        r.update(status="ERROR", error=f"{type(e).__name__}: {e}")
        return r, [], [("READ_ERROR", r["error"])]


def main():
    if not MANIFEST.is_file(): raise FileNotFoundError(f"Frozen manifest not found: {MANIFEST}")
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    if "path" not in m: raise ValueError(f"Manifest needs 'path'; columns={list(m.columns)}")
    raw_paths = m["path"].dropna().astype(str).tolist()
    files = list(dict.fromkeys(Path(x) for x in raw_paths
                 if Path(x).suffix.lower() in {".parquet",".pq",".csv",".json"}))
    OUT.mkdir(parents=True, exist_ok=True)
    audits, times, issues = [], [], []
    for i,p in enumerate(files,1):
        r,t,found = audit_file(p); audits.append(r); times.extend(t)
        issues.extend({"path":str(p),"issue":k,"detail":str(v)} for k,v in found)
        if i % 250 == 0: print(f"Audited {i:,}/{len(files):,}: {p.name}")
    adf = pd.DataFrame(audits)
    adf.to_csv(OUT/"data_structure_key_audit.csv",index=False,encoding="utf-8-sig")
    families = []
    for (parent, ext), g in adf.groupby([
        adf["path"].map(lambda x: str(Path(x).parent)),
        adf["file_type"]
    ]):
        sigs = g.loc[g["schema_sha256"].astype(str).ne(""), "schema_sha256"].nunique()
        status = "INFO" if len(g) == 1 else "PASS" if sigs <= 1 else "REVIEW"
        families.append({"parent": parent, "file_type": ext, "file_count": len(g),
                         "schema_versions": int(sigs), "status": status})
        if status == "REVIEW":
            issues.append({"path": parent, "issue": "SCHEMA_VARIATION_IN_DIRECTORY",
                           "detail": f"{ext}: {sigs} schema signatures across {len(g)} files"})
    pd.DataFrame(families, columns=["parent","file_type","file_count","schema_versions","status"]).to_csv(
        OUT/"schema_consistency_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(times,columns=["path","date_column","valid_dates","invalid_dates","min_date","max_date"]).to_csv(
        OUT/"time_range_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(issues,columns=["path","issue","detail"]).to_csv(OUT/"step4_issues.csv",index=False,encoding="utf-8-sig")
    summary = {
        "manifest_rows":len(m), "duplicate_manifest_paths":len(raw_paths)-m["path"].dropna().astype(str).nunique(),
        "data_files_audited":len(files), "schema_families":len(families),
        "schema_families_with_variation":sum(x["status"]=="REVIEW" for x in families),
        "status_counts":dict(Counter(x["status"] for x in audits)),
        "files_with_inferred_key":sum(x["key_status"]=="INFERRED" for x in audits),
        "files_with_duplicate_keys":sum(isinstance(x["duplicate_key_rows"],int) and x["duplicate_key_rows"]>0 for x in audits),
        "files_with_invalid_dates":sum(x["invalid_date_values"]>0 for x in audits),
        "time_range_rows":len(times), "issue_rows":len(issues),
        "note":"主键为按列名推断的候选主键；需结合各表数据契约解释 REVIEW。",
        "output_dir":str(OUT)}
    (OUT/"day18_step4_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()

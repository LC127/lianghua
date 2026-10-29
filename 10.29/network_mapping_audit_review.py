
from pathlib import Path
import argparse
import re
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "network_mapping_audit_reviewed"
INPUT_NAMES = (
    "global_master_mapping_check.csv",
    "node_universe_checks.csv",
    "edge_endpoint_checks.csv",
)


def resolve_audit_dir(base, explicit=None):
    """Use all three reports from one run; never mix audit batches."""
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        missing = [name for name in INPUT_NAMES if not (candidate / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete audit directory: {candidate}\nMissing: {', '.join(missing)}"
            )
        return candidate

    base = Path(base).resolve()
    audit_root = base / "output" / "M1_day18" / "06_final_network_mapping_audit"
    locations = [base, audit_root]
    if audit_root.is_dir():
        locations.extend(sorted(p for p in audit_root.glob("run_*") if p.is_dir()))
    candidates = [
        p for p in locations if all((p / name).is_file() for name in INPUT_NAMES)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple complete audit directories found. Select one using "
            "--audit-dir PATH:\n" + "\n".join(str(p) for p in candidates)
        )
    raise FileNotFoundError(
        "No complete mapping audit found. Expected all three files together:\n"
        + "\n".join(INPUT_NAMES)
        + f"\nSearched: {base} and {audit_root} (including run_* directories)."
        + "\nRun network_mapping_audit.py first, or specify --audit-dir PATH."
    )


def load_report(audit_dir, name):
    path = audit_dir / name
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Audit report is empty: {path}") from exc
    if df.empty:
        raise ValueError(f"Audit report contains no rows: {path}")
    return df


def window_num(value):
    m = re.fullmatch(r"W?(\d+)(?:\.0+)?", str(value).strip(), re.I)
    return int(m.group(1)) if m else None


def verify_source(file_path):
    p = Path(file_path)
    try:
        d = pd.read_parquet(p, columns=["window", "analysis_date"])
        expected_w = window_num(p.parent.name)
        actual_w = {window_num(x) for x in d["window"].dropna().unique()}
        w_ok = bool(
            not d.empty and d["window"].notna().all()
            and expected_w is not None and actual_w == {expected_w}
        )

        dates = pd.to_datetime(d["analysis_date"], errors="coerce")
        date_text = dates.dt.strftime("%Y-%m-%d")
        d_ok = bool(
            not d.empty and dates.notna().all()
            and date_text.nunique() == 1
            and date_text.iloc[0] == p.stem
        )
        return w_ok, d_ok, ""
    except Exception as e:
        return False, False, f"{type(e).__name__}: {e}"


def zero_checks(row, columns):
    return all(pd.to_numeric(row[c], errors="coerce") == 0 for c in columns)


def review(df, kind, global_ok):
    checked = df["file"].map(verify_source)
    df[["window_match", "date_match", "source_read_error"]] = pd.DataFrame(
        checked.tolist(), index=df.index
    )

    def passed(r):
        common = (
            global_ok and bool(r["window_match"]) and bool(r["date_match"])
            and not r["source_read_error"]
        )
        if kind == "node":
            return common and r["rows"] == r["unique_security_ids"] and zero_checks(
                r, ["unknown_final_ids", "unknown_mapping_ids",
                    "index_mismatch_rows", "duplicate_security_ids",
                    "duplicate_indices"]
            )
        return common and bool(r["node_snapshot_found"]) and zero_checks(
            r, ["missing_endpoint_rows", "unknown_mapping_indices",
                "unknown_date_node_indices", "self_loops",
                "duplicate_undirected_pairs"]
        )

    df["status"] = df.apply(
        lambda r: "PASS" if passed(r) else "REVIEW", axis=1
    )
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Review network mapping audit reports.")
    parser.add_argument("--audit-dir", type=Path, help="Directory containing all three audit CSVs.")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Reviewed report output directory.")
    args = parser.parse_args(argv)
    audit_dir = resolve_audit_dir(BASE, args.audit_dir)
    out = args.output_dir.expanduser().resolve()
    print("Audit input directory:", audit_dir, flush=True)

    # Read all reports from the same validated directory before writing output.
    global_report = load_report(audit_dir, INPUT_NAMES[0])
    if len(global_report) != 1:
        raise ValueError("Global mapping report must contain exactly one row.")
    node_input = load_report(audit_dir, INPUT_NAMES[1])
    edge_input = load_report(audit_dir, INPUT_NAMES[2])
    g = global_report.iloc[0]
    global_ok = (
        g["final_count"] == g["mapping_count"] == g["overlap"]
        and zero_checks(g, [
            "final_only", "mapping_only", "final_missing_ids",
            "mapping_missing_ids", "mapping_missing_index",
            "mapping_duplicate_ids", "mapping_duplicate_indices",
        ])
    )
    node = review(node_input, "node", global_ok)
    edge = review(edge_input, "edge", global_ok)

    out.mkdir(parents=True, exist_ok=True)
    node.to_csv(out / "node_universe_checks_reviewed.csv",
                index=False, encoding="utf-8-sig")
    edge.to_csv(out / "edge_endpoint_checks_reviewed.csv",
                index=False, encoding="utf-8-sig")

    print("Global mapping:", "PASS" if global_ok else "REVIEW")
    for name, df in [("Nodes", node), ("Edges", edge)]:
        print(
            f"{name}: PASS={(df.status == 'PASS').sum()}, "
            f"REVIEW={(df.status == 'REVIEW').sum()}, "
            f"window_mismatch={(~df.window_match).sum()}, "
            f"date_mismatch={(~df.date_match).sum()}, "
            f"source_errors={(df.source_read_error != '').sum()}"
        )
    print("Reviewed files saved to:", out)


if __name__ == "__main__":
    main()

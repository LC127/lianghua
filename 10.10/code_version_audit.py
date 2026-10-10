"""Day14 Step4 source-code fingerprint audit (read-only for frozen artifacts).
Run: python day14_step4_code_version_audit.py --root D:\\M1_StockNetwork
Optional: --archive-dir D:\\M1_StockNetwork\\archive --check-inputs
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


def sha256(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(key, root):
    """Map frozen Windows absolute paths to this project's root."""
    p = PureWindowsPath(key)
    if p.is_absolute():
        if len(p.parts) < 3 or p.parts[1].casefold() != root.name.casefold():
            raise ValueError(f"Unexpected root in metadata: {key}")
        return root.joinpath(*p.parts[2:])
    return root / key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\M1_StockNetwork")
    ap.add_argument("--script", default=None, help="Step4 .py file; default: project root")
    ap.add_argument("--archive-dir", default=None, help="Optional folder to search for the recorded .py")
    ap.add_argument("--check-inputs", action="store_true", help="Also verify all frozen input hashes")
    args = ap.parse_args()

    root = Path(args.root)
    step4 = root / "output/M1_day14/04_step4_continuous_risk_state_validation"
    meta_file = step4 / "day14_step4_metadata.json"
    script = Path(args.script) if args.script else root / "day14_step4_continuous_risk_state_validation.py"
    metadata = json.loads(meta_file.read_text(encoding="utf-8-sig"))
    expected = metadata["script_sha256"].lower()
    actual = sha256(script)

    checks = []
    for filename, recorded in metadata["output_sha256"].items():
        p = step4 / filename
        found = sha256(p)
        checks.append({"kind": "output", "file": str(p),
                       "expected": recorded, "actual": found, "match": found == recorded})

    if args.check_inputs:
        for key, recorded in metadata["input_sha256"].items():
            p = source_path(key, root)
            found = sha256(p)
            checks.append({"kind": "input", "file": str(p),
                           "expected": recorded, "actual": found, "match": found == recorded})

    # Diagnose line endings; a normalized match is not an exact SHA256 pass.
    normalized_match = False
    if actual is not None:
        raw = script.read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        crlf = lf.replace(b"\n", b"\r\n")
        normalized_match = expected in {
            hashlib.sha256(lf).hexdigest(), hashlib.sha256(crlf).hexdigest()
        }

    archived_matches = []
    if args.archive_dir:
        archive = Path(args.archive_dir)
        if not archive.is_dir():
            raise FileNotFoundError(f"Archive directory missing: {archive}")
        for p in archive.rglob("*.py"):
            if p.is_file() and sha256(p) == expected:
                archived_matches.append(str(p))

    outputs_ok = all(x["match"] for x in checks if x["kind"] == "output")
    inputs_ok = all(x["match"] for x in checks if x["kind"] == "input")
    meta_ok = (metadata.get("status") == "FROZEN"
               and metadata.get("formal_qa", {}).get("all_formal_qa_pass") is True)
    passed = bool(actual == expected and outputs_ok and inputs_ok and meta_ok)

    report = {
        "audit_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if passed else "FAIL",
        "metadata_file": str(meta_file),
        "source_script": str(script),
        "expected_script_sha256": expected,
        "actual_script_sha256": actual,
        "script_exact_match": actual == expected,
        "line_ending_normalized_match_only": normalized_match and actual != expected,
        "archive_matching_script_paths": archived_matches,
        "step4_metadata_qa_pass": meta_ok,
        "frozen_outputs_match": outputs_ok,
        "frozen_inputs_match": inputs_ok if args.check_inputs else "NOT_CHECKED",
        "file_checks": checks,
        "policy": "Never alter original Step4 script, QA, metadata or outputs during audit."
    }
    audit_dir = root / "output/M1_day14/04a_step4_code_version_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_file = audit_dir / "step4_code_version_audit.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Step4 code-version audit: {report['status']}")
    print(f"  script exact match : {actual == expected}")
    print(f"  output hashes pass : {outputs_ok}")
    print(f"  input hashes pass  : {report['frozen_inputs_match']}")
    print(f"  archived candidates: {len(archived_matches)}")
    print(f"  report             : {report_file}")
    if not passed:
        print("FAIL means no version certification; retain the frozen Step4 outputs unchanged.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

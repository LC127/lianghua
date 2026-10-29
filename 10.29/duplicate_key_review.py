
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/M1_day18/04_step4_duplicate_key_review" / (
    f"run_{datetime.now():%Y%m%d_%H%M%S}"
)

# 文件路径、候选主键、辅助查看字段
FILES = [
    ("day16_risk", "output/M1_day16/02_step2_network_risk_exposure_factors/network_risk_exposure_panel.parquet", [
        ["analysis_date", "network", "window", "master_index"],
        ["analysis_date", "network", "window", "security_id"]],
        ["security_id", "risk_status", "degree"]),

    ("stock_code_history", "output/M1_day1/01_stage1/stock_code_history.parquet", [
        ["security_id", "stock_code", "list_date", "delist_date"],
        ["security_id", "stock_code", "exchange", "list_date", "delist_date"]],
        ["stock_name", "exchange", "board"]),

    ("stock_master_issues", "output/M1_day1/01_stage1_fixed/stock_master_issues.csv", [
        ["security_id", "issue_type", "severity", "note"],
        ["security_id", "stock_code", "issue_type", "severity", "note"]],
        ["stock_name", "exchange", "board", "code_count"]),

    ("core_data_anomalies", "output/M1_day1/02_stage2_daily_market_qa/core_data_anomalies.parquet", [
        ["trade_date", "security_id", "issue_type", "severity", "note"],
        ["trade_date", "security_id", "issue_type", "severity"]],
        ["stock_code", "stock_name"]),

    ("stock_feasibility", "output/M1_day1/04_step4_network_feasibility/stock_level_feasibility_summary.parquet", [
        ["window", "security_id"],
        ["window", "security_id", "stock_code", "exchange"]],
        ["min_valid_ratio", "current_universe_days", "eligible_days"]),

    ("target_weights", "output/M1_day8/02_stage2_trade_reconstruction/portfolio_target_weights.parquet", [
        ["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
        ["analysis_date", "window", "factor_name", "method", "security_id"]],
        ["signal_value", "target_weight"]),

    ("trade_weights", "output/M1_day8/02_stage2_trade_reconstruction/portfolio_trade_weight_panel.parquet", [
        ["analysis_date", "previous_analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
        ["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
        ["analysis_date", "window", "factor_name", "method", "security_id"]],
        ["target_weight", "delta_weight", "trade_side"]),

    ("dedicated_drift", "output/M1_day8/02c_stage2_rerun_with_dedicated_drift/portfolio_trade_weight_panel_dedicated_drift.parquet", [
        ["analysis_date", "previous_analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
        ["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
        ["analysis_date", "window", "factor_name", "method", "security_id"]],
        ["dedicated_drift_status", "target_weight", "delta_weight"]),

    ("canonical_trade", "output/M1_day8/02e_stage2_final_canonical_trade_reconstruction/canonical_portfolio_trade_weight_panel.parquet", [
        ["analysis_date", "previous_analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
        ["analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
        ["analysis_date", "window", "factor_name", "method", "security_id"]],
        ["dedicated_drift_status", "target_weight", "delta_weight"]),

    ("participation_base", "output/M1_day8/03_stage3_participation_capacity/stock_participation_base_panel.parquet", [
        ["analysis_date", "previous_analysis_date", "window", "factor_name", "method", "portfolio_leg", "security_id"],
        ["analysis_date", "window", "factor_name", "method", "security_id"]],
        ["dedicated_drift_status", "capacity_delta_weight", "capacity_trade_side"]),
]


def read_columns(path):
    if path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path).schema_arrow.names
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return list(pd.read_csv(path, nrows=0, encoding=enc).columns)
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode CSV header")


def read_data(path, columns):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns, dtype_backend="pyarrow")
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(
                path, usecols=columns, encoding=enc,
                low_memory=False, dtype_backend="pyarrow"
            )
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode CSV data")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tests, samples, errors = [], [], []

    for name, rel, candidates, extra in FILES:
        path = ROOT / rel
        print(f"Checking {name} ...", flush=True)

        try:
            if not path.is_file():
                raise FileNotFoundError(path)

            available = read_columns(path)
            requested = list(dict.fromkeys(
                c for key in candidates for c in key
            ))
            requested += [c for c in extra if c not in requested]
            cols = [c for c in requested if c in available]
            df = read_data(path, cols)

            for i, key in enumerate(candidates, 1):
                missing = [c for c in key if c not in df.columns]

                result = {
                    "file": name,
                    "path": str(path),
                    "candidate": f"K{i}",
                    "key_columns": "|".join(key),
                    "rows": len(df),
                    "missing_columns": "|".join(missing),
                    "null_key_rows": "",
                    "duplicate_rows": "",
                    "status": ""
                }

                if missing:
                    result["status"] = "MISSING_COLUMNS"
                    tests.append(result)
                    continue

                nulls = int(df[key].isna().any(axis=1).sum())
                duplicate = df.duplicated(key)
                dup_count = int(duplicate.sum())

                result.update(
                    null_key_rows=nulls,
                    duplicate_rows=dup_count,
                    status="UNIQUE" if nulls == 0 and dup_count == 0
                    else "DUPLICATE_OR_NULL"
                )
                tests.append(result)

                # 每个候选键最多输出 20 条重复记录样本
                if dup_count:
                    mask = df.duplicated(key, keep=False)
                    show = list(dict.fromkeys(key + [
                        c for c in extra if c in df.columns
                    ]))
                    sample = df.loc[mask, show].head(20).copy()
                    sample.insert(0, "candidate", f"K{i}")
                    sample.insert(0, "file", name)
                    samples.extend(sample.to_dict("records"))

        except Exception as e:
            errors.append({
                "file": name,
                "path": str(path),
                "error": f"{type(e).__name__}: {e}"
            })

    pd.DataFrame(tests).to_csv(
        OUT / "candidate_key_tests.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(samples).to_csv(
        OUT / "duplicate_key_samples.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(errors, columns=["file", "path", "error"]).to_csv(
        OUT / "read_errors.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "files_targeted": len(FILES),
        "files_read_successfully": len(FILES) - len(errors),
        "candidate_tests": len(tests),
        "unique_candidates": sum(x["status"] == "UNIQUE" for x in tests),
        "duplicate_or_null_candidates": sum(
            x["status"] == "DUPLICATE_OR_NULL" for x in tests
        ),
        "missing_column_candidates": sum(
            x["status"] == "MISSING_COLUMNS" for x in tests
        ),
        "read_errors": len(errors),
        "output_dir": str(OUT)
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import pandas as pd


# ============================================================
# Day 13 - Step 6
# M1 v1.0 Final Freeze Manifest
# ============================================================

ROOT = Path(r"D:\M1_StockNetwork")
D13 = ROOT / "output" / "M1_day13"

OUT = D13 / "06_step6_m1_v1_final_freeze"
OUT.mkdir(parents=True, exist_ok=True)

FREEZE_VERSION = "M1_v1.0"

MANIFEST_OUT = OUT / "m1_v1_0_final_freeze_manifest.csv"
QA_OUT = OUT / "day13_step6_qa.csv"
META_OUT = OUT / "m1_v1_0_freeze_metadata.json"


# ============================================================
# 1. Frozen Step directories
# ============================================================

S1 = D13 / "01_step1_requirement_evidence_audit"
S2 = D13 / "02_step2_frozen_9factor_evidence"
S3 = D13 / "03_step3_risk_alpha_separation_summary"
S4 = D13 / "04_step4_frozen_network_risk_state"
S5 = D13 / "05_step5_m1_m4_frozen_interface"

STEPS = {
    1: (
        S1 / "day13_step1_qa.csv",
        S1 / "day13_step1_metadata.json",
    ),
    2: (
        S2 / "day13_step2_qa.csv",
        S2 / "day13_step2_metadata.json",
    ),
    3: (
        S3 / "day13_step3_qa.csv",
        S3 / "day13_step3_metadata.json",
    ),
    4: (
        S4 / "day13_step4_qa.csv",
        S4 / "day13_step4_metadata.json",
    ),
    5: (
        S5 / "day13_step5_qa.csv",
        S5 / "day13_step5_metadata.json",
    ),
}


# ============================================================
# 2. Canonical M1 v1.0 artifacts
# ============================================================

ARTIFACTS = [
    # Step 1
    (1, "requirement_completion_matrix",
     S1 / "m1_requirement_completion_matrix.csv"),
    (1, "qa",
     S1 / "day13_step1_qa.csv"),
    (1, "metadata",
     S1 / "day13_step1_metadata.json"),

    # Step 2
    (2, "full_9factor_evidence_matrix",
     S2 / "m1_frozen_9factor_final_evidence_matrix.csv"),
    (2, "compact_9factor_reporting_matrix",
     S2 / "m1_frozen_9factor_final_evidence_matrix_compact.csv"),
    (2, "qa",
     S2 / "day13_step2_qa.csv"),
    (2, "metadata",
     S2 / "day13_step2_metadata.json"),

    # Step 3
    (3, "risk_alpha_factor_summary",
     S3 / "risk_alpha_separation_final_summary.csv"),
    (3, "system_oos_context",
     S3 / "risk_system_oos_context.csv"),
    (3, "qa",
     S3 / "day13_step3_qa.csv"),
    (3, "metadata",
     S3 / "day13_step3_metadata.json"),

    # Step 4
    (4, "frozen_network_risk_state",
     S4 / "frozen_network_risk_state.csv"),
    (4, "risk_state_by_window",
     S4 / "frozen_network_risk_state_by_window.csv"),
    (4, "qa",
     S4 / "day13_step4_qa.csv"),
    (4, "metadata",
     S4 / "day13_step4_metadata.json"),

    # Step 5
    (5, "m1_to_m4_frozen_interface",
     S5 / "m1_to_m4_frozen_interface_panel.csv"),
    (5, "qa",
     S5 / "day13_step5_qa.csv"),
    (5, "metadata",
     S5 / "day13_step5_metadata.json"),
]


# ============================================================
# Helpers
# ============================================================

def as_bool(x):
    return str(x).strip().lower() in {
        "true", "1", "yes"
    }


def qa_pass(path):
    q = pd.read_csv(path)
    d = dict(zip(q["qa_name"], q["qa_value"]))
    return as_bool(
        d.get("all_formal_qa_pass", False)
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metadata_pass(meta):
    if "formal_qa" in meta:
        return bool(
            meta["formal_qa"].get(
                "all_formal_qa_pass",
                False,
            )
        )

    return bool(
        meta.get(
            "all_formal_qa_pass",
            False,
        )
    )


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = Path(str(path) + ".tmp")

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    os.replace(tmp, path)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 3. Upstream Step1-5 QA gates
    # --------------------------------------------------------

    step_status = {}
    step_meta = {}

    for step, (qa_file, meta_file) in STEPS.items():

        for p in [qa_file, meta_file]:
            if not p.exists():
                raise FileNotFoundError(p)

        qpass = qa_pass(qa_file)
        meta = load_json(meta_file)
        mpass = metadata_pass(meta)

        if not (qpass and mpass):
            raise RuntimeError(
                f"Step{step} QA/metadata not passed."
            )

        step_status[step] = {
            "qa_pass": qpass,
            "metadata_qa_pass": mpass,
        }

        step_meta[step] = meta

    # --------------------------------------------------------
    # 4. M4 boundary consistency
    # --------------------------------------------------------

    step1_m4_pending = bool(
        step_meta[1]
        .get("formal_qa", {})
        .get("m4_linkage_pending", False)
    )

    step5_m4_clean = (
        step_meta[5].get(
            "m4_data_joined"
        ) is False
        and
        step_meta[5].get(
            "m4_outcome_used"
        ) is False
    )

    m4_status = (
        "INTERFACE_FROZEN_ANALYSIS_PENDING"
        if step1_m4_pending and step5_m4_clean
        else "STATUS_INCONSISTENT"
    )

    # --------------------------------------------------------
    # 5. Build artifact manifest
    # --------------------------------------------------------

    rows = []

    for step, role, path in ARTIFACTS:

        exists = path.exists()
        size = (
            path.stat().st_size
            if exists else 0
        )

        rows.append({
            "freeze_version":
                FREEZE_VERSION,

            "step":
                step,

            "artifact_role":
                role,

            "relative_path":
                str(
                    path.relative_to(ROOT)
                ),

            "size_bytes":
                size,

            "sha256":
                sha256(path)
                if exists and size > 0
                else "",

            "step_qa_pass":
                step_status[step][
                    "qa_pass"
                ],

            "step_metadata_qa_pass":
                step_status[step][
                    "metadata_qa_pass"
                ],
        })

    manifest = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 6. Formal final-freeze QA
    # --------------------------------------------------------

    qa = {
        "freeze_version":
            FREEZE_VERSION,

        "step1_to_step5_qa_pass":
            all(
                x["qa_pass"]
                for x in step_status.values()
            ),

        "step1_to_step5_metadata_qa_pass":
            all(
                x["metadata_qa_pass"]
                for x in step_status.values()
            ),

        "expected_artifact_count":
            len(ARTIFACTS),

        "artifact_count":
            len(manifest),

        "missing_artifact_count":
            int(
                (
                    manifest["size_bytes"]
                    == 0
                ).sum()
            ),

        "duplicate_relative_path_count":
            int(
                manifest[
                    "relative_path"
                ]
                .duplicated()
                .sum()
            ),

        "sha256_missing_count":
            int(
                (
                    manifest["sha256"]
                    == ""
                ).sum()
            ),

        "step1_core_m1_complete":
            (
                step_meta[1]
                .get("formal_qa", {})
                .get(
                    "core_incomplete_count",
                    1,
                )
                == 0
            ),

        "step4_future_target_used_in_state":
            step_meta[4].get(
                "future_target_used_in_state"
            ),

        "step5_m4_data_joined":
            step_meta[5].get(
                "m4_data_joined"
            ),

        "step5_m4_outcome_used":
            step_meta[5].get(
                "m4_outcome_used"
            ),

        "m4_linkage_status":
            m4_status,

        "research_result_reestimated":
            False,
    }

    qa["all_formal_qa_pass"] = bool(
        qa["step1_to_step5_qa_pass"]
        and qa[
            "step1_to_step5_metadata_qa_pass"
        ]
        and qa[
            "artifact_count"
        ] == qa[
            "expected_artifact_count"
        ]
        and qa[
            "missing_artifact_count"
        ] == 0
        and qa[
            "duplicate_relative_path_count"
        ] == 0
        and qa[
            "sha256_missing_count"
        ] == 0
        and qa[
            "step1_core_m1_complete"
        ]
        and qa[
            "step4_future_target_used_in_state"
        ] is False
        and qa[
            "step5_m4_data_joined"
        ] is False
        and qa[
            "step5_m4_outcome_used"
        ] is False
        and m4_status
        == "INTERFACE_FROZEN_ANALYSIS_PENDING"
    )

    if not qa["all_formal_qa_pass"]:
        raise RuntimeError(
            f"M1 v1.0 freeze QA failed: {qa}"
        )

    # --------------------------------------------------------
    # 7. Existing freeze = immutable
    # --------------------------------------------------------

    if MANIFEST_OUT.exists():

        old = pd.read_csv(
            MANIFEST_OUT,
            dtype=str,
        )

        old_map = dict(
            zip(
                old["relative_path"],
                old["sha256"],
            )
        )

        new_map = dict(
            zip(
                manifest["relative_path"],
                manifest["sha256"],
            )
        )

        if old_map != new_map:
            raise RuntimeError(
                "M1_v1.0 is already frozen, but "
                "one or more canonical artifacts changed. "
                "Do NOT overwrite v1.0; create a new version."
            )

        if not (
            QA_OUT.exists()
            and META_OUT.exists()
        ):
            raise RuntimeError(
                "Existing freeze manifest found but "
                "Step6 QA/metadata is incomplete."
            )

        print(
            "Existing M1_v1.0 freeze verified: "
            "no artifact drift."
        )
        return

    # --------------------------------------------------------
    # 8. Freeze metadata
    # --------------------------------------------------------

    metadata = {
        "research_day":
            13,

        "step":
            "Step6_M1_v1_0_Final_Freeze_Manifest",

        "freeze_version":
            FREEZE_VERSION,

        "freeze_created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(
                timespec="seconds"
            ),

        "status":
            "FROZEN",

        "hash_algorithm":
            "SHA256",

        "artifact_count":
            len(manifest),

        "canonical_manifest":
            str(MANIFEST_OUT),

        "canonical_m1_to_m4_interface":
            str(
                S5
                / "m1_to_m4_frozen_interface_panel.csv"
            ),

        "m4_linkage_status":
            m4_status,

        "freeze_principle": (
            "Freeze existing canonical M1 evidence only. "
            "No network, factor, risk, alpha, cost, "
            "capacity, Risk-State, or M4 outcome is "
            "re-estimated or selected."
        ),

        "versioning_rule": (
            "Any post-freeze modification to a canonical "
            "artifact requires a new M1 version. "
            "M1_v1.0 must not be silently overwritten."
        ),

        "formal_qa":
            qa,
    }

    # Write manifest LAST: its existence acts as freeze lock.
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "qa_name": k,
                    "qa_value": v,
                }
                for k, v in qa.items()
            ]
        ),
        QA_OUT,
    )

    atomic_json(
        metadata,
        META_OUT,
    )

    atomic_csv(
        manifest,
        MANIFEST_OUT,
    )

    print("=" * 72)
    print("M1 v1.0 Final Freeze")
    print("=" * 72)
    print(
        f"Artifacts : {len(manifest)}"
    )
    print(
        f"M4 status : {m4_status}"
    )
    print(
        "Formal QA : PASS"
    )
    print(
        "Freeze    : FROZEN"
    )
    print(
        f"Output    : {OUT}"
    )


if __name__ == "__main__":
    main()
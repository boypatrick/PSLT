#!/usr/bin/env python3
"""
Build a compact artifact-status registry for reviewer-facing output hygiene.

The registry marks which artifacts are canonical baseline evidence versus
diagnostic / legacy comparators, so old files are not misread as current
headline numbers.

Outputs:
  - output/repro/artifact_status.csv
  - paper/artifact_status.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "repro"
PAPER_DIR = ROOT / "paper"


def rows() -> List[Dict[str, object]]:
    return [
        {
            "artifact_path": "paper/main.pdf",
            "status": "canonical_baseline",
            "category": "manuscript",
            "note": "Current submission target PDF.",
        },
        {
            "artifact_path": "paper/hll_signal_strength_summary.csv",
            "status": "canonical_baseline",
            "category": "observable_map",
            "note": "Baseline UV+LL-RG H->ll summary used in main text.",
        },
        {
            "artifact_path": "paper/hll_uv_to_eft_summary.csv",
            "status": "canonical_baseline",
            "category": "uv_to_eft",
            "note": "Baseline UV-tree -> finite-match -> LL-RG summary.",
        },
        {
            "artifact_path": "paper/core_param_robustness.csv",
            "status": "canonical_baseline",
            "category": "robustness",
            "note": "Main-text local one-at-a-time robustness table source.",
        },
        {
            "artifact_path": "paper/hll_rge_sensitivity.csv",
            "status": "canonical_baseline",
            "category": "robustness",
            "note": "Main-text UV+LL-RG control sensitivity table source.",
        },
        {
            "artifact_path": "paper/chain_mode_full_direct_audit_Dgrid21_Egrid41.csv",
            "status": "canonical_baseline",
            "category": "chain_mode_audit",
            "note": "Current strict chain_mode auto vs full_direct audit source.",
        },
        {
            "artifact_path": "paper/chain_mode_uv_audit_Dgrid21_Egrid21.csv",
            "status": "canonical_baseline",
            "category": "chain_mode_audit",
            "note": "Current UV map parity audit source.",
        },
        {
            "artifact_path": "paper/hll_uv_envelope_summary.csv",
            "status": "canonical_baseline",
            "category": "uv_envelope",
            "note": "UV control envelope summary (nonzero finite-match center).",
        },
        {
            "artifact_path": "paper/hll_uv_envelope_cases.csv",
            "status": "canonical_baseline",
            "category": "uv_envelope",
            "note": "Per-case UV envelope run details.",
        },
        {
            "artifact_path": "paper/hll_uv_envelope_map.csv",
            "status": "canonical_baseline",
            "category": "uv_envelope",
            "note": "Pointwise mu_mumu envelope bounds over the active grid.",
        },
        {
            "artifact_path": "output/kinetic_action_chain/action_chain_consistency_summary.csv",
            "status": "legacy_comparator",
            "category": "chain_mode_audit",
            "note": "Older surrogate-vs-action comparator; not cited in current main text.",
        },
        {
            "artifact_path": "output/kinetic_action_chain/action_chain_consistency.png",
            "status": "legacy_comparator",
            "category": "chain_mode_audit",
            "note": "Legacy comparator figure; retained for history only.",
        },
        {
            "artifact_path": "output/hll_signal_strength/hll_signal_strength_summary_chi2_best.csv",
            "status": "diagnostic_variant",
            "category": "anchor_sensitivity",
            "note": "Dynamic-anchor sensitivity variant, not baseline headline.",
        },
        {
            "artifact_path": "output/hll_signal_strength/hll_signal_strength_summary_robust_center.csv",
            "status": "diagnostic_variant",
            "category": "anchor_sensitivity",
            "note": "Robust-center anchor variant, not baseline headline.",
        },
        {
            "artifact_path": "output/hll_signal_strength/hll_signal_strength_summary_uv_tree_calibrated.csv",
            "status": "diagnostic_variant",
            "category": "uv_modes",
            "note": "UV-tree calibrated mode; comparator only.",
        },
        {
            "artifact_path": "output/hll_signal_strength/hll_signal_strength_summary_uv_rge_smoke.csv",
            "status": "diagnostic_variant",
            "category": "uv_modes",
            "note": "Smoke-mode output for quick checks only.",
        },
    ]


def write_csv(path: Path, rows_in: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_in[0].keys()))
        writer.writeheader()
        writer.writerows(rows_in)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, object]] = []
    for row in rows():
        abs_path = ROOT / str(row["artifact_path"])
        out_rows.append(
            {
                **row,
                "exists": bool(abs_path.exists()),
            }
        )

    out_csv = OUTDIR / "artifact_status.csv"
    paper_csv = PAPER_DIR / "artifact_status.csv"
    write_csv(out_csv, out_rows)
    write_csv(paper_csv, out_rows)

    print(f"[saved] {out_csv}")
    print(f"[saved] {paper_csv}")
    for row in out_rows:
        print(row)


if __name__ == "__main__":
    main()

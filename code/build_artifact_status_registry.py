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
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid21_Egrid41.csv",
            "status": "canonical_baseline",
            "category": "chain_mode_audit",
            "note": "Current strict chain_mode full_direct vs cell_direct_runtime audit source.",
        },
        {
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21.csv",
            "status": "canonical_baseline",
            "category": "chain_mode_audit",
            "note": "Large-surface release gate: full_direct vs cell_direct_runtime parity audit.",
        },
        {
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_extreme.csv",
            "status": "diagnostic_variant",
            "category": "chain_mode_audit",
            "note": "Large-surface stress gate: full_direct vs cell_direct_runtime_extreme all-direct audit.",
        },
        {
            "artifact_path": "paper/chain_mode_uv_audit_Dgrid21_Egrid21.csv",
            "status": "canonical_baseline",
            "category": "chain_mode_audit",
            "note": "Current UV map parity audit source.",
        },
        {
            "artifact_path": "paper/full_direct_map_release_summary.csv",
            "status": "canonical_baseline",
            "category": "chain_mode_audit",
            "note": "Integrated release summary: main full_direct map + direct-bias checks + large runtime/extreme chain gates.",
        },
        {
            "artifact_path": "output/kinetic_action_chain/full_direct_map_release_meta.json",
            "status": "canonical_baseline",
            "category": "chain_mode_audit",
            "note": "Release gate metadata for full_direct_map summary assembly.",
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
            "artifact_path": "paper/open_system_micro_baseline_candidate.csv",
            "status": "diagnostic_variant",
            "category": "open_system_gate",
            "note": "Open-system micro baseline-candidate gate decision (diagnostic track).",
        },
        {
            "artifact_path": "paper/kappa_env_anchor_calibration.csv",
            "status": "diagnostic_variant",
            "category": "open_system_gate",
            "note": "Multi-anchor kappa_env calibration summary for open-system micro mode.",
        },
        {
            "artifact_path": "paper/kappa_env_anchor_holdout.csv",
            "status": "diagnostic_variant",
            "category": "open_system_gate",
            "note": "Holdout validation metrics for multi-anchor kappa_env calibration.",
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

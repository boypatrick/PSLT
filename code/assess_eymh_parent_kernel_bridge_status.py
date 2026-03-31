#!/usr/bin/env python3
"""
Bridge-status checklist for the EYMH projected parent-kernel derivation.

This script does not rescan any maps. It consolidates the exact D21xE21
parent-kernel audit summaries into a single structural-closure checklist,
highlighting which ambiguities are already closed and which derivation-level
lemmas are still missing before the roadmap item can be considered fully
derived from the parent action itself.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "output" / "kinetic_action_chain"

SUMMARY_CSV = OUT / "eymh_parent_kernel_bridge_status_summary.csv"
GATES_CSV = OUT / "eymh_parent_kernel_bridge_status_gates.csv"
GAPS_CSV = OUT / "eymh_parent_kernel_bridge_status_gaps.csv"
JSON_OUT = OUT / "eymh_parent_kernel_bridge_status.json"
PNG_OUT = OUT / "eymh_parent_kernel_bridge_status.png"
RUN_META = OUT / "eymh_parent_kernel_bridge_status_run_meta.json"

PAPER_SUMMARY = PAPER / SUMMARY_CSV.name
PAPER_GATES = PAPER / GATES_CSV.name
PAPER_GAPS = PAPER / GAPS_CSV.name
PAPER_PNG = PAPER / PNG_OUT.name

STRUCTURAL_EPS = 1e-12
SELECTION_GAP_MIN = 1e-7
ONSET_GAP_MIN = 1e-3


@dataclass(frozen=True)
class AuditSpec:
    key: str
    filename: str
    max_residual_field: str
    selection_gap_field: str | None
    label: str
    note: str
    corr_field: str | None = None
    corr_threshold: float | None = None


AUDITS: List[AuditSpec] = [
    AuditSpec(
        key="parent_kernel_statement",
        filename="hll_uv_action_loop_eymh_parent_kernel_statement_summary_D21E21_fix.csv",
        max_residual_field="max_abs_objective_identity_residual",
        selection_gap_field="selection_gap_parent_objective",
        label="Exact parent-kernel excess functional",
        note="The mismatch functional is reproduced exactly by the projected parent-kernel excess objective.",
    ),
    AuditSpec(
        key="block_split",
        filename="hll_uv_action_loop_eymh_block_split_audit_summary_D21E21_fix.csv",
        max_residual_field="max_abs_canonical_action_split_residual",
        selection_gap_field=None,
        label="Participation/tree block split",
        note="The response action closes as an exact split between participation and shell-background/tree sectors.",
        corr_field="corr(split_action, response_action)",
        corr_threshold=1.0 - 1e-12,
    ),
    AuditSpec(
        key="parent_blockdet",
        filename="hll_uv_action_loop_eymh_parent_blockdet_audit_summary_D21E21_fix.csv",
        max_residual_field="max_abs_canonical_action_residual",
        selection_gap_field="selection_gap_blockdet_objective",
        label="Background-normalized parent block determinant",
        note="The response action equals a block-determinant / Schur-complement factorization to machine precision.",
    ),
    AuditSpec(
        key="parent_mix_geomean",
        filename="hll_uv_action_loop_eymh_parent_mix_geomean_audit_summary_D21E21_fix.csv",
        max_residual_field="canonical_max_abs_det_ratio_residual",
        selection_gap_field="selection_gap_det_ratio_residual",
        label="Geometric-mean mixed block",
        note="The symmetric excess family uniquely selects the geometric-mean mixed entry.",
    ),
    AuditSpec(
        key="parent_ratio_warp",
        filename="hll_uv_action_loop_eymh_parent_ratio_warp_audit_summary_D21E21_fix.csv",
        max_residual_field="canonical_max_abs_det_ratio_residual",
        selection_gap_field="selection_gap_det_ratio_residual",
        label="Reject ratio-warp extension",
        note="The first ratio-warped deformation of the mixed block is rejected.",
    ),
    AuditSpec(
        key="parent_symnorm",
        filename="hll_uv_action_loop_eymh_parent_symnorm_audit_summary_D21E21_fix.csv",
        max_residual_field="canonical_max_abs_det_ratio_residual",
        selection_gap_field="selection_gap_det_ratio_residual",
        label="Symmetry/normalization coordinates fixed",
        note="The canonical point enforces zero normalization shift, unit symmetric degree, and zero tilt.",
    ),
    AuditSpec(
        key="parent_generator_affinity",
        filename="hll_uv_action_loop_eymh_parent_generator_affinity_audit_summary_D21E21_fix.csv",
        max_residual_field="canonical_max_abs_det_ratio_residual",
        selection_gap_field="selection_gap_det_ratio_residual",
        label="Local generator-affinity closed",
        note="First local log-curvature corrections are rejected; the surviving class is locally affine.",
    ),
    AuditSpec(
        key="excess_coordinate",
        filename="hll_uv_action_loop_eymh_excess_coordinate_audit_summary_D21E21_fix.csv",
        max_residual_field="canonical_max_abs_det_ratio_residual",
        selection_gap_field="selection_gap_part_onset_residual",
        label="Reference offsets fixed at identity/background",
        note="The excess variables are anchored exactly on the identity participation block and shell-background tree block.",
    ),
    AuditSpec(
        key="fixedpoint_boxcox",
        filename="hll_uv_action_loop_eymh_fixedpoint_boxcox_audit_summary_D21E21_fix.csv",
        max_residual_field="canonical_max_abs_det_ratio_residual",
        selection_gap_field="selection_gap_part_onset_residual",
        label="Linear excess variables selected",
        note="Even within tangent-normalized Box-Cox warps, the linear additive excess coordinates remain uniquely exact.",
    ),
    AuditSpec(
        key="normal_coordinate",
        filename="hll_uv_action_loop_eymh_normal_coordinate_audit_summary_D21E21_fix.csv",
        max_residual_field="canonical_max_abs_det_ratio_residual",
        selection_gap_field="selection_gap_part_onset_residual",
        label="Zero-second-jet normal coordinates",
        note="The normal-coordinate audit selects zero second jet, i.e. the linear excess variables themselves.",
    ),
]


DERIVATION_GAPS = [
    {
        "gap_id": "derive_projected_hessian",
        "status": "OPEN",
        "priority": 1,
        "label": "Derive the projected 2x2 kernel from the EYMH fluctuation Hessian",
        "why_open": "The audits show the projected log-det / Schur form is exact and uniquely selected, but they still start from exported witnesses rather than a parent-action-side Hessian reduction.",
        "target_statement": "Show that the low-mode projected fluctuation operator of the parent EYMH action reduces to the observed participation block, shell-background/tree block, and background-normalized Schur response.",
    },
    {
        "gap_id": "derive_geomean_mixed_entry",
        "status": "OPEN",
        "priority": 2,
        "label": "Explain the geometric-mean mixed block from symmetry/minimality",
        "why_open": "The geometric-mean mixed entry is uniquely selected by the minimal symmetric-excess and ratio-warp rejection audits, but it is not yet derived as the unavoidable mixed bilinear of the parent action.",
        "target_statement": "Derive why the mixed block must scale as sqrt((K11-1)(K22-Kbg)) once the parent block is projected and normalized.",
    },
    {
        "gap_id": "derive_linear_excess_coordinates",
        "status": "OPEN",
        "priority": 3,
        "label": "Derive linear excess variables as natural fixed-point normal coordinates",
        "why_open": "The excess-coordinate, Box-Cox, and normal-coordinate audits uniquely select the identity/background anchors and zero-second-jet coordinates, but this remains an exact selection statement rather than a parent-action derivation.",
        "target_statement": "Show that the parent action naturally organizes the projected kernel around identity/background fixed points in zero-second-jet tangent-normalized linear excess coordinates.",
    },
]


def read_one_row_csv(path: Path) -> Dict[str, str]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mirror_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text())


def main() -> None:
    gate_rows: List[Dict[str, object]] = []
    residual_plot = []
    gap_plot = []

    for spec in AUDITS:
        row = read_one_row_csv(PAPER / spec.filename)
        max_residual = float(row[spec.max_residual_field])
        if spec.selection_gap_field is not None:
            selection_gap = float(row[spec.selection_gap_field])
            threshold = (
                ONSET_GAP_MIN
                if "onset" in spec.selection_gap_field
                else SELECTION_GAP_MIN
            )
            gap_ok = selection_gap >= threshold
        else:
            selection_gap = float("nan")
            threshold = float("nan")
            gap_ok = True

        corr_value = float(row[spec.corr_field]) if spec.corr_field is not None else float("nan")
        corr_ok = True if spec.corr_field is None else corr_value >= float(spec.corr_threshold)
        passed = max_residual <= STRUCTURAL_EPS and gap_ok and corr_ok
        gate_rows.append(
            {
                "gate": spec.label,
                "passed": int(passed),
                "max_residual": max_residual,
                "residual_threshold": STRUCTURAL_EPS,
                "selection_gap": selection_gap,
                "selection_gap_threshold": threshold,
                "corr_value": corr_value,
                "corr_threshold": spec.corr_threshold if spec.corr_threshold is not None else "",
                "note": spec.note,
                "source_csv": str((PAPER / spec.filename).relative_to(ROOT)),
            }
        )
        residual_plot.append(max(max_residual, 1e-18))
        if np.isfinite(selection_gap):
            gap_plot.append(max(selection_gap, 1e-18))
        else:
            gap_plot.append(max(corr_value, 1e-18))

    n_passed = sum(int(r["passed"]) for r in gate_rows)
    n_gates = len(gate_rows)
    go_flag = int(n_passed == n_gates)
    decision = (
        "STRUCTURAL_BRIDGE_CLOSED_DERIVATION_LEMMA_PENDING"
        if go_flag
        else "STRUCTURAL_BRIDGE_INCOMPLETE"
    )

    summary_row = {
        "decision": decision,
        "go_flag": go_flag,
        "n_gates": n_gates,
        "n_passed": n_passed,
        "n_failed": n_gates - n_passed,
        "max_structural_residual": max(float(r["max_residual"]) for r in gate_rows),
        "min_selection_gap": min(
            float(r["selection_gap"])
            for r in gate_rows
            if np.isfinite(float(r["selection_gap"]))
        ),
        "open_derivation_gaps": len(DERIVATION_GAPS),
        "next_lemma": DERIVATION_GAPS[0]["label"],
        "next_lemma_gap_id": DERIVATION_GAPS[0]["gap_id"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = [r["gate"] for r in gate_rows]

    axes[0].barh(labels, np.log10(np.array(residual_plot)))
    axes[0].axvline(np.log10(STRUCTURAL_EPS), color="crimson", linestyle="--", linewidth=1.2)
    axes[0].set_title("EYMH Parent-Kernel Structural Residuals")
    axes[0].set_xlabel("log10(max residual)")

    axes[1].barh(labels, np.log10(np.array(gap_plot)))
    axes[1].axvline(np.log10(SELECTION_GAP_MIN), color="crimson", linestyle="--", linewidth=1.2)
    axes[1].set_title("Runner-Up / Onset Gap Size")
    axes[1].set_xlabel("log10(selection gap)")

    fig.suptitle("EYMH Parent-Kernel Bridge Status")
    fig.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    write_csv(SUMMARY_CSV, list(summary_row.keys()), [summary_row])
    write_csv(GATES_CSV, list(gate_rows[0].keys()), gate_rows)
    write_csv(GAPS_CSV, list(DERIVATION_GAPS[0].keys()), DERIVATION_GAPS)

    payload = {
        "summary": summary_row,
        "gates": gate_rows,
        "derivation_gaps": DERIVATION_GAPS,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    RUN_META.write_text(
        json.dumps(
            {
                "script": str(Path(__file__).relative_to(ROOT)),
                "structural_eps": STRUCTURAL_EPS,
                "selection_gap_min": SELECTION_GAP_MIN,
                "onset_gap_min": ONSET_GAP_MIN,
                "audits": [spec.filename for spec in AUDITS],
            },
            indent=2,
        )
    )

    mirror_text(SUMMARY_CSV, PAPER_SUMMARY)
    mirror_text(GATES_CSV, PAPER_GATES)
    mirror_text(GAPS_CSV, PAPER_GAPS)
    PAPER_PNG.write_bytes(PNG_OUT.read_bytes())

    print(json.dumps(summary_row, indent=2))


if __name__ == "__main__":
    main()

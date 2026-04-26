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
            "artifact_path": "ROADMAP_STATUS.md",
            "status": "canonical_certificate",
            "category": "submission_status",
            "note": "Operational roadmap: current closed/monitor/diagnostic reading for reviewer-facing theorem and profile gates.",
        },
        {
            "artifact_path": "FAMILY_DECISION_TABLE.md",
            "status": "canonical_certificate",
            "category": "submission_status",
            "note": "One-page family decision map: canonical reviewer branch, closed negatives, research-only ingredients, and reopen rules.",
        },
        {
            "artifact_path": "paper/artifact_status.csv",
            "status": "canonical_certificate",
            "category": "submission_status",
            "note": "Generated artifact-status registry used to prevent diagnostic or legacy outputs from being read as baseline claims.",
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
            "note": "Current strict chain_mode full_direct vs cell_direct_runtime (runtime_profile) audit source.",
        },
        {
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21.csv",
            "status": "canonical_baseline",
            "category": "chain_mode_audit",
            "note": "Large-surface release gate: full_direct vs cell_direct_runtime (runtime_profile) parity audit.",
        },
        {
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid21_Egrid41_cell_direct_runtime_release.csv",
            "status": "diagnostic_variant",
            "category": "chain_mode_audit",
            "note": "Small-surface release-candidate all-direct gate: full_direct vs cell_direct_runtime_release (runtime_bnorm_release_candidate).",
        },
        {
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_release.csv",
            "status": "diagnostic_variant",
            "category": "chain_mode_audit",
            "note": "Large-surface release-candidate all-direct gate: full_direct vs cell_direct_runtime_release (runtime_bnorm_release_candidate).",
        },
        {
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_extreme.csv",
            "status": "diagnostic_variant",
            "category": "chain_mode_audit",
            "note": "Large-surface stress gate: full_direct vs cell_direct_runtime_extreme (runtime_bnorm_extreme) all-direct audit.",
        },
        {
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid21_Egrid41_cell_direct_runtime_release_tuned.csv",
            "status": "diagnostic_variant",
            "category": "chain_mode_audit",
            "note": "Small-surface risk-weighted profile-anchored runtime-direct visibility parity gate: full_direct vs cell_direct_runtime_release_tuned.",
        },
        {
            "artifact_path": "paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_release_tuned.csv",
            "status": "diagnostic_variant",
            "category": "chain_mode_audit",
            "note": "Large-surface risk-weighted profile-anchored runtime-direct visibility parity gate: full_direct vs cell_direct_runtime_release_tuned.",
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
            "note": "Integrated release summary: main full_direct map + direct-bias checks + runtime_profile / runtime_bnorm_release_tuned_profile_anchored / runtime_bnorm_extreme chain gates.",
        },
        {
            "artifact_path": "paper/full_direct_b_module_diagnostics_D60E21.csv",
            "status": "diagnostic_variant",
            "category": "chain_mode_audit",
            "note": "B-module drift attribution table (B_N, y_raw, m2, width_ratio) at max-drift and anchor points.",
        },
        {
            "artifact_path": "paper/runtime_direct_b_tuning_best.json",
            "status": "diagnostic_variant",
            "category": "chain_mode_audit",
            "note": "Two-stage runtime-direct B tuning winner (D21xE41 -> D60xE21 gate).",
        },
        {
            "artifact_path": "paper/nonhermitian_rank2_diagnostic_D21E41_kdiag0p10.csv",
            "status": "diagnostic_variant",
            "category": "rank2_nonhermitian",
            "note": "Non-Hermitian rank-2 diagnostic map drift summary (kappa_diag_scale=0.10).",
        },
        {
            "artifact_path": "output/source_projection/source_projection_fit_gaussian_w0p2.json",
            "status": "diagnostic_variant",
            "category": "source_projection",
            "note": "High-dimensional parent-profile to 3D projected-source fit summary (gaussian example).",
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
            "artifact_path": "output/cap_resonance_1d/static_width_whittaker_special_interval_certificate_d6rc400_r60_r80_c640_summary.csv",
            "status": "canonical_certificate",
            "category": "static_width",
            "note": "S25 local strict static-width parent-tube certificate; adopted only on D=6, R_C=400, R=60->80, c=640.",
        },
        {
            "artifact_path": "output/cap_resonance_1d/static_width_whittaker_special_interval_certificate_d6rc400_r60_r80_c640_aggregate_summary.csv",
            "status": "canonical_certificate",
            "category": "static_width",
            "note": "S25 aggregate certificate for the parity-uniform Whittaker special-function interval check.",
        },
        {
            "artifact_path": "output/dirac_frame_check/minimal_dirac_conformal_check.csv",
            "status": "canonical_certificate",
            "category": "fermionic_lift",
            "note": "F1 conformal Dirac-frame certificate for the projected fermionic lift.",
        },
        {
            "artifact_path": "output/gn_fp_2d/gn_lowN_microstate_count_summary.csv",
            "status": "canonical_certificate",
            "category": "lowN_counting",
            "note": "G1 finite-volume low-N projector-rank certificate: ranks 1,2,3.",
        },
        {
            "artifact_path": "output/highN_decoupling/highN_threshold_theorem_summary.csv",
            "status": "canonical_certificate",
            "category": "highN_threshold",
            "note": "N1 finite-volume Sturm/min-max threshold certificate; no fourth bound layer on the audited chain.",
        },
        {
            "artifact_path": "output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_trace_synthesis_source_summary.csv",
            "status": "canonical_certificate",
            "category": "open_system_exact_bridge",
            "note": "O71 strict-slab physical trace synthesis certificate; closes the exact-bridge bottleneck for reviewer purposes.",
        },
        {
            "artifact_path": "output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_gs_rate_source_summary.csv",
            "status": "canonical_certificate",
            "category": "open_system_exact_bridge",
            "note": "O72 secondary GS-rate certificate for the monotone adjoint-barrier sweep counts.",
        },
        {
            "artifact_path": "output/hll_absolute_normalization/hll_absolute_normalization_synthesis_source_summary.csv",
            "status": "canonical_certificate",
            "category": "eymh_normalization",
            "note": "U9 structural absolute-normalization synthesis; ratio constants, amplitude factorization, tree/source identities, kernel, and width profile closed on the audited domain.",
        },
        {
            "artifact_path": "output/hll_absolute_normalization/hll_y2raw_adoption_impact_gate_summary.csv",
            "status": "diagnostic_gate",
            "category": "eymh_normalization",
            "note": "U10 direct y2raw(D*) impact gate; parent-point audited but diagnostic-only because it changes D21xE21 acceptance topology.",
        },
        {
            "artifact_path": "paper/hll_uv_action_loop_eymh_feshbach_complement_summary_D21E21_fix.csv",
            "status": "canonical_certificate",
            "category": "eymh_normalization",
            "note": "U11 projected parent-kernel Feshbach-Schur complement budget; full-prefactor eps_F^(eq)<=2.33e-15 under the 1e-12 guard.",
        },
        {
            "artifact_path": "output/superrad_fp_1d/channel_resolved_A_l_tensor_D4-20full_eta3_N1-2-3_l1-2_summary.csv",
            "status": "monitor_artifact",
            "category": "localized_channel_tensor",
            "note": "A1 localized channel tensor extraction; useful extraction upgrade but not an all-valid production replacement.",
        },
        {
            "artifact_path": "output/superrad_fp_1d/channel_A_tensor_impact_Dgrid60_Egrid21_summary.csv",
            "status": "monitor_artifact",
            "category": "localized_channel_tensor",
            "note": "A1 D60/E21 impact gate; bound-sector fallback is production-safe, strict all-valid tensor lookup is not.",
        },
        {
            "artifact_path": "output/tcoh_fp_1d/tcoh_impact_gate_Dgrid60_Egrid21_summary.csv",
            "status": "diagnostic_gate",
            "category": "profile_gate",
            "note": "t_coh^(deph)(D) impact gate; diagnostic-only because it changes H->mumu acceptance and winner topology.",
        },
        {
            "artifact_path": "output/eta_fp_1d/eta_impact_gate_Dgrid60_Egrid21_summary.csv",
            "status": "monitor_artifact",
            "category": "profile_gate",
            "note": "eta_fp(D) impact gate; profile-scaled branches are adoption-safe, fully closed eta-axis variants remain diagnostic-only.",
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
        writer = csv.DictWriter(f, fieldnames=list(rows_in[0].keys()), lineterminator="\n")
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

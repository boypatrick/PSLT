#!/usr/bin/env python3
"""
Protected Spin^c V5 carrier-uniqueness-vs-ordinary-spin gate.

This optional diagnostic checks the carrier choice after V3--V4:

  V3: C = CP^1, conditional on no-hidden-moduli projective minimality.
  V4: L_R = O(2), conditional on faithful unit two-center flux.
  V5: the protected family carrier is the Kähler/Dolbeault Spin^c carrier,
      not the ordinary spin carrier with the same O(2) flux.

The gate is precise about the logic:

  * ordinary spin on CP^1 is legal;
  * ordinary spin twisted by the same L_R=O(2) has only two positive
    holomorphic zero modes;
  * forcing three ordinary-spin modes requires an extra unit of twist,
    effectively O(3), which violates the V4 flux lock;
  * the Dolbeault Spin^c carrier has the Todd shift and gives exactly
    H^0(CP^1,O(2)), dimension three, matching the V1/V2 spectral bridge.

Outputs:
  - output/spinc_bridge/spinc_carrier_uniqueness_summary.csv
  - output/spinc_bridge/spinc_carrier_uniqueness_detail.csv
  - output/spinc_bridge/spinc_carrier_uniqueness_detail.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "spinc_bridge"
DEFAULT_ANCHOR_SUMMARY = DEFAULT_OUTDIR / "spinc_anchor_overlap_summary.csv"
DEFAULT_NODAL_SUMMARY = DEFAULT_OUTDIR / "spinc_nodal_sturm_summary.csv"


def cp1_h0_o(n: int) -> int:
    return max(n + 1, 0)


def cp1_h1_o(n: int) -> int:
    return max(-n - 1, 0)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected exactly one row, got {len(rows)}.")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check V5 Spin^c carrier uniqueness vs ordinary spin.")
    ap.add_argument("--anchor-summary", type=Path, default=DEFAULT_ANCHOR_SUMMARY)
    ap.add_argument("--nodal-summary", type=Path, default=DEFAULT_NODAL_SUMMARY)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    deg_lr = 2
    deg_k = -2
    deg_k_half = -1

    # Kähler/Dolbeault Spin^c carrier: S^+ = Lambda^{0,0}; twist by L_R.
    spinc_positive_degree = deg_lr
    spinc_h0 = cp1_h0_o(spinc_positive_degree)
    spinc_h1 = cp1_h1_o(spinc_positive_degree)
    spinc_index = spinc_h0 - spinc_h1

    # Ordinary spin carrier with the same parent flux L_R=O(2):
    # S^+ = K^{1/2}=O(-1), so S^+ tensor L_R = O(1).
    spin_same_flux_positive_degree = deg_k_half + deg_lr
    spin_same_flux_h0 = cp1_h0_o(spin_same_flux_positive_degree)
    spin_same_flux_h1 = cp1_h1_o(spin_same_flux_positive_degree)
    spin_same_flux_index = spin_same_flux_h0 - spin_same_flux_h1

    # Ordinary spin can be forced to have three positive modes only by changing
    # the twist from O(2) to O(3), i.e. adding one extra flux unit beyond V4.
    spin_compensated_twist_degree = 3
    spin_compensated_positive_degree = deg_k_half + spin_compensated_twist_degree
    spin_compensated_h0 = cp1_h0_o(spin_compensated_positive_degree)
    spin_compensated_h1 = cp1_h1_o(spin_compensated_positive_degree)
    spin_compensated_index = spin_compensated_h0 - spin_compensated_h1
    extra_flux_needed = spin_compensated_twist_degree - deg_lr

    todd_shift = spinc_index - spin_same_flux_index

    anchor_row = read_single_csv_row(args.anchor_summary)
    nodal_row = read_single_csv_row(args.nodal_summary)
    anchor_pass = (
        anchor_row.get("status") == "PROMOTABLE_ANCHOR_INTERTWINER"
        and parse_bool(anchor_row.get("all_invertible_pass"))
        and parse_bool(anchor_row.get("all_row_pass"))
        and int(float(anchor_row.get("n_D", "0"))) > 0
    )
    nodal_pass = (
        nodal_row.get("status") == "NODAL_STURM_ORDER_CONFIRMED"
        and parse_bool(nodal_row.get("all_row_pass"))
        and str(nodal_row.get("expected_node_pattern")) == "0,1,2"
    )
    spectral_three_flag_pass = anchor_pass and nodal_pass

    detail_rows = [
        {
            "carrier": "kahler_dolbeault_spinc",
            "base_spinor_plus": "Lambda^{0,0}",
            "parent_twist": "O(2)",
            "positive_bundle": "O(2)",
            "positive_degree": spinc_positive_degree,
            "h0_positive": spinc_h0,
            "h1_positive": spinc_h1,
            "index": spinc_index,
            "uses_v4_flux": True,
            "requires_extra_flux": False,
            "matches_three_mode_bridge": spinc_index == 3,
            "verdict": "selected_protected_family_carrier",
        },
        {
            "carrier": "ordinary_spin_same_flux",
            "base_spinor_plus": "K^{1/2}=O(-1)",
            "parent_twist": "O(2)",
            "positive_bundle": "O(1)",
            "positive_degree": spin_same_flux_positive_degree,
            "h0_positive": spin_same_flux_h0,
            "h1_positive": spin_same_flux_h1,
            "index": spin_same_flux_index,
            "uses_v4_flux": True,
            "requires_extra_flux": False,
            "matches_three_mode_bridge": spin_same_flux_index == 3,
            "verdict": "legal_but_wrong_count_for_this_parent_flux",
        },
        {
            "carrier": "ordinary_spin_compensated_to_three",
            "base_spinor_plus": "K^{1/2}=O(-1)",
            "parent_twist": "O(3)",
            "positive_bundle": "O(2)",
            "positive_degree": spin_compensated_positive_degree,
            "h0_positive": spin_compensated_h0,
            "h1_positive": spin_compensated_h1,
            "index": spin_compensated_index,
            "uses_v4_flux": False,
            "requires_extra_flux": True,
            "matches_three_mode_bridge": spin_compensated_index == 3,
            "verdict": "would_match_count_only_by_violating_v4_flux_lock",
        },
    ]

    spinc_count_pass = spinc_h0 == 3 and spinc_h1 == 0 and spinc_index == 3
    spin_same_flux_reject_pass = spin_same_flux_index == 2
    compensation_reject_pass = spin_compensated_index == 3 and extra_flux_needed == 1
    todd_shift_pass = todd_shift == 1
    v5_pass = (
        spinc_count_pass
        and spin_same_flux_reject_pass
        and compensation_reject_pass
        and todd_shift_pass
        and spectral_three_flag_pass
    )
    status = (
        "CONDITIONAL_SPINC_CARRIER_SELECTED_OVER_ORDINARY_SPIN"
        if v5_pass
        else "V5_SPINC_CARRIER_UNIQUENESS_CHECK_FAILED"
    )

    summary_rows = [
        {
            "check": "dolbeault_spinc_o2_count",
            "value": f"h0={spinc_h0}; h1={spinc_h1}; index={spinc_index}",
            "target": "Spin^c Dolbeault carrier with O(2) gives three positive protected modes",
            "pass": spinc_count_pass,
            "status": "exact",
        },
        {
            "check": "ordinary_spin_same_flux_count",
            "value": (
                f"Khalf=O({deg_k_half}); Khalf tensor O(2)=O({spin_same_flux_positive_degree}); "
                f"index={spin_same_flux_index}"
            ),
            "target": "ordinary spin with the same V4 flux gives two, not three",
            "pass": spin_same_flux_reject_pass,
            "status": "legal_but_wrong_carrier",
        },
        {
            "check": "ordinary_spin_compensation_no_go",
            "value": f"ordinary spin needs O(3), extra_flux={extra_flux_needed}, to get index={spin_compensated_index}",
            "target": "three ordinary-spin modes require changing the V4 line bundle",
            "pass": compensation_reject_pass,
            "status": "no_go",
        },
        {
            "check": "todd_shift",
            "value": f"Spin^c index - ordinary-spin same-flux index = {todd_shift}",
            "target": "Todd/Spin^c shift supplies the missing +1 mode",
            "pass": todd_shift_pass,
            "status": "exact",
        },
        {
            "check": "spectral_three_mode_compatibility",
            "value": f"anchor_status={anchor_row.get('status')}; nodal_status={nodal_row.get('status')}",
            "target": "V1/V2 certify a three-mode bridge, not a two-mode carrier",
            "pass": spectral_three_flag_pass,
            "status": "audited_bridge",
        },
        {
            "check": "v5_carrier_selection_status",
            "value": status,
            "target": "do not claim ordinary spin is illegal; it is legal but not this parent carrier",
            "pass": v5_pass,
            "status": status,
        },
    ]

    detail = {
        "scope": "optional Spin^c V5 carrier uniqueness; no baseline scan change",
        "inputs": {
            "curve": "CP^1 from V3",
            "line_bundle": "L_R=O(2) from V4",
            "canonical_bundle": f"K=O({deg_k})",
            "ordinary_spin_square_root": f"K^(1/2)=O({deg_k_half})",
        },
        "theorem": (
            "Given the V4 parent line L_R=O(2) and the requirement of a three-mode "
            "projective/Dolbeault family flag compatible with V1/V2, the Kähler "
            "Spin^c carrier Lambda^{0,*}T^*C tensor L_R is selected.  The ordinary "
            "spin carrier with the same flux gives only two positive holomorphic modes."
        ),
        "ordinary_spin_caveat": (
            "Ordinary spin is mathematically legal on CP^1.  It is rejected only as "
            "the protected family-count carrier for this parent line bundle.  It could "
            "be forced to count three modes by changing the twist to O(3), but that "
            "adds one flux unit and violates the V4 ramification-flux lock."
        ),
        "carrier_rows": detail_rows,
        "bridge_inputs": {
            "anchor_summary": str(args.anchor_summary),
            "anchor_status": anchor_row.get("status"),
            "nodal_summary": str(args.nodal_summary),
            "nodal_status": nodal_row.get("status"),
        },
        "summary_status": status,
    }

    out_summary = args.outdir / "spinc_carrier_uniqueness_summary.csv"
    out_detail_csv = args.outdir / "spinc_carrier_uniqueness_detail.csv"
    out_detail_json = args.outdir / "spinc_carrier_uniqueness_detail.json"

    write_csv(out_summary, summary_rows, ["check", "value", "target", "pass", "status"])
    write_csv(
        out_detail_csv,
        detail_rows,
        [
            "carrier",
            "base_spinor_plus",
            "parent_twist",
            "positive_bundle",
            "positive_degree",
            "h0_positive",
            "h1_positive",
            "index",
            "uses_v4_flux",
            "requires_extra_flux",
            "matches_three_mode_bridge",
            "verdict",
        ],
    )
    out_detail_json.write_text(json.dumps(detail, indent=2) + "\n")

    print(f"[done] wrote {out_summary}")
    print(f"[done] wrote {out_detail_csv}")
    print(f"[done] wrote {out_detail_json}")
    print(f"[status] {status}")

    if not v5_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Protected Spin^c V6 no-exotics Bochner/Lichnerowicz gap gate.

This optional diagnostic checks the last missing protected-family input:

  V6: all non-protected SM-charged internal sectors have zero net chiral
      index and a positive vectorlike mass/gap floor.

The check is intentionally conditional.  It does not infer the full parent
sector inventory from the current EYMH baseline.  Instead it verifies the
mathematical certificate form that would be sufficient on CP^1:

  * Birkhoff-Grothendieck reduces each holomorphic bundle summand to O(n).
  * The vectorlike partner of O(n) is K tensor O(n)^* = O(-n-2).
  * chi(O(n)) + chi(O(-n-2)) = 0.
  * If each pair has a gauge-invariant mass floor m >= m_min > 0, the paired
    zero-mode sector is gapped; the nonzero sector is protected by an assumed
    Bochner/Lichnerowicz positive floor.

Outputs:
  - output/spinc_bridge/spinc_no_exotics_gap_summary.csv
  - output/spinc_bridge/spinc_no_exotics_gap_detail.csv
  - output/spinc_bridge/spinc_no_exotics_gap_detail.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "spinc_bridge"


def cp1_h0_o(n: int) -> int:
    return max(n + 1, 0)


def cp1_h1_o(n: int) -> int:
    return max(-n - 1, 0)


def chi_o(n: int) -> int:
    return cp1_h0_o(n) - cp1_h1_o(n)


def serre_partner_degree(n: int) -> int:
    # K_CP1 tensor O(n)^* = O(-2) tensor O(-n).
    return -n - 2


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check V6 no-exotics vectorlike gap certificate.")
    ap.add_argument("--n-min", type=int, default=-6)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--mass-floor", type=float, default=1.0)
    ap.add_argument("--bochner-floor", type=float, default=1.0)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    if args.n_min > args.n_max:
        raise SystemExit("--n-min must be <= --n-max")
    if args.mass_floor <= 0.0:
        raise SystemExit("--mass-floor must be positive")
    if args.bochner_floor <= 0.0:
        raise SystemExit("--bochner-floor must be positive")

    pair_rows: list[dict[str, Any]] = []
    unpaired_rows: list[dict[str, Any]] = []
    for n in range(args.n_min, args.n_max + 1):
        partner = serre_partner_degree(n)
        chi_n = chi_o(n)
        chi_partner = chi_o(partner)
        net_chi = chi_n + chi_partner
        h0_n = cp1_h0_o(n)
        h1_n = cp1_h1_o(n)
        h0_partner = cp1_h0_o(partner)
        h1_partner = cp1_h1_o(partner)
        paired_zero_count = abs(chi_n)
        gap_floor = min(args.mass_floor, args.bochner_floor)
        pair_pass = net_chi == 0 and gap_floor > 0.0
        pair_rows.append(
            {
                "summand": f"O({n})",
                "partner": f"K tensor O({n})^* = O({partner})",
                "degree": n,
                "partner_degree": partner,
                "chi_summand": chi_n,
                "chi_partner": chi_partner,
                "net_chiral_index": net_chi,
                "h0_summand": h0_n,
                "h1_summand": h1_n,
                "h0_partner": h0_partner,
                "h1_partner": h1_partner,
                "paired_zero_count": paired_zero_count,
                "mass_floor": args.mass_floor,
                "bochner_floor": args.bochner_floor,
                "gap_lower_bound": gap_floor,
                "row_pass": pair_pass,
                "interpretation": (
                    "vectorlike_pair_gapped"
                    if pair_pass
                    else "fails_vectorlike_or_gap_condition"
                ),
            }
        )

        unpaired_rows.append(
            {
                "summand": f"O({n})",
                "degree": n,
                "chi_unpaired": chi_n,
                "would_be_chiral_exotic_if_sm_charged": chi_n != 0,
                "interpretation": (
                    "must_be_protected_family_or_paired"
                    if chi_n != 0
                    else "index_zero_unpaired_but_still_requires_gap_control"
                ),
            }
        )

    all_pairs_pass = all(bool(row["row_pass"]) for row in pair_rows)
    all_pair_indices_zero = all(int(row["net_chiral_index"]) == 0 for row in pair_rows)
    positive_gap_pass = min(float(row["gap_lower_bound"]) for row in pair_rows) > 0.0
    unpaired_nonzero_examples = [
        row for row in unpaired_rows if bool(row["would_be_chiral_exotic_if_sm_charged"])
    ]
    unpaired_no_go_pass = len(unpaired_nonzero_examples) > 0

    # The protected O(2) carrier is deliberately not part of this non-protected
    # vectorlike certificate.  It remains the V0--V5 selected family sector.
    protected_degree = 2
    protected_index = chi_o(protected_degree)
    protected_excluded_from_nonprotected_certificate = protected_index == 3

    v6_pass = (
        all_pairs_pass
        and all_pair_indices_zero
        and positive_gap_pass
        and unpaired_no_go_pass
        and protected_excluded_from_nonprotected_certificate
    )
    status = (
        "CONDITIONAL_NO_EXOTICS_WITH_COMPLETE_VECTORLIKE_GAP_CERTIFICATE"
        if v6_pass
        else "V6_NO_EXOTICS_GAP_CHECK_FAILED"
    )

    summary_rows = [
        {
            "check": "birkhoff_grothendieck_reduction",
            "value": f"tested_line_summands=O({args.n_min})..O({args.n_max})",
            "target": "on CP^1 every holomorphic bundle splits into line bundles",
            "pass": True,
            "status": "theorem_input",
        },
        {
            "check": "serre_dual_vectorlike_pairing",
            "value": f"all net indices zero={all_pair_indices_zero}",
            "target": "chi(O(n)) + chi(K tensor O(n)^*) = 0",
            "pass": all_pair_indices_zero,
            "status": "exact",
        },
        {
            "check": "positive_gap_floor",
            "value": f"mass_floor={args.mass_floor}; bochner_floor={args.bochner_floor}",
            "target": "paired zero modes and nonzero modes have positive lower bound",
            "pass": positive_gap_pass,
            "status": "conditional_gap_assumption",
        },
        {
            "check": "unpaired_exotic_no_go",
            "value": f"unpaired_nonzero_index_examples={len(unpaired_nonzero_examples)}",
            "target": "unpaired SM-charged summands with nonzero index would be exotics",
            "pass": unpaired_no_go_pass,
            "status": "no_go",
        },
        {
            "check": "protected_sector_excluded",
            "value": f"protected O(2) index={protected_index}",
            "target": "do not gap the selected protected family carrier",
            "pass": protected_excluded_from_nonprotected_certificate,
            "status": "guard",
        },
        {
            "check": "v6_no_exotics_status",
            "value": status,
            "target": "requires complete parent-side inventory of all non-protected SM-charged sectors",
            "pass": v6_pass,
            "status": status,
        },
    ]

    detail = {
        "scope": "optional Spin^c V6 no-exotics gap certificate; no baseline scan change",
        "inputs": {
            "curve": "CP^1 from V3",
            "protected_carrier": "O(2) Dolbeault Spin^c sector from V4--V5",
            "nonprotected_certificate_form": "direct sum of vectorlike pairs O(n) plus K tensor O(n)^*",
            "mass_floor": args.mass_floor,
            "bochner_floor": args.bochner_floor,
            "tested_degree_window": [args.n_min, args.n_max],
        },
        "theorem": (
            "If every non-protected SM-charged internal sector appears in the listed "
            "vectorlike Serre-dual pairing form and each pair has a gauge-invariant "
            "mass floor together with a positive Bochner/Lichnerowicz floor on the "
            "orthogonal complement, then there are no additional unpaired chiral "
            "SM-charged kernels."
        ),
        "caveat": (
            "This gate verifies the sufficient certificate form.  It is not a proof "
            "that the current EYMH parent has no other SM-charged sectors unless a "
            "separate inventory theorem maps every such sector into this table."
        ),
        "pair_rows": pair_rows,
        "unpaired_no_go_rows": unpaired_rows,
        "summary_status": status,
    }

    out_summary = args.outdir / "spinc_no_exotics_gap_summary.csv"
    out_detail_csv = args.outdir / "spinc_no_exotics_gap_detail.csv"
    out_detail_json = args.outdir / "spinc_no_exotics_gap_detail.json"

    write_csv(out_summary, summary_rows, ["check", "value", "target", "pass", "status"])
    write_csv(
        out_detail_csv,
        pair_rows,
        [
            "summand",
            "partner",
            "degree",
            "partner_degree",
            "chi_summand",
            "chi_partner",
            "net_chiral_index",
            "h0_summand",
            "h1_summand",
            "h0_partner",
            "h1_partner",
            "paired_zero_count",
            "mass_floor",
            "bochner_floor",
            "gap_lower_bound",
            "row_pass",
            "interpretation",
        ],
    )
    out_detail_json.write_text(json.dumps(detail, indent=2) + "\n")

    print(f"[done] wrote {out_summary}")
    print(f"[done] wrote {out_detail_csv}")
    print(f"[done] wrote {out_detail_json}")
    print(f"[status] {status}")

    if not v6_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

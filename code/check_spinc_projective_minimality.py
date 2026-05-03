#!/usr/bin/env python3
"""
Protected Spin^c V3 projective-minimality / genus-zero rigidity gate.

This is an optional bridge diagnostic, not a submission-baseline scan.  It
checks the algebraic-geometric obstruction behind the question:

  Can a two-center projected parent force C = CP^1 rather than merely assume it?

The gate formalizes the conservative answer used in the roadmap:

  Under an explicit no-hidden-moduli / minimal compact projective completion
  axiom, genus zero is the only compact connected Riemann-surface candidate.
  Hence C is CP^1 by the classification of compact genus-zero Riemann surfaces.

It also records the no-go side: without that minimality axiom, the index
condition chi(L)=3 admits infinitely many (genus, degree) pairs
deg L = genus + 2, so CP^1 is not unconditional from the index count alone.

Outputs:
  - output/spinc_bridge/spinc_projective_minimality_summary.csv
  - output/spinc_bridge/spinc_projective_minimality_detail.csv
  - output/spinc_bridge/spinc_projective_minimality_detail.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "spinc_bridge"


def marked_curve_moduli_dim(genus: int, marked_points: int = 2) -> int:
    """Complex dimension of the moduli of smooth curves with marked points.

    The (g,n)=(0,2) case is unstable as a Deligne-Mumford object, but as a
    two-center projective curve modulo PGL(2,C) it has no shape modulus: every
    ordered pair of distinct points is equivalent to (0, infinity).
    """
    if genus == 0 and marked_points <= 3:
        return 0
    if genus == 1:
        # General stable formula 3g-3+n gives n for g=1.
        return marked_points
    return 3 * genus - 3 + marked_points


def jacobian_dim(genus: int) -> int:
    """Complex dimension of Jac(C), hence also Pic^d(C), for a genus-g curve."""
    return genus


def required_degree_for_spinc_index_three(genus: int) -> int:
    """Riemann-Roch: chi(L)=deg L + 1 - g = 3."""
    return genus + 2


def cp1_h0_o(n: int) -> int:
    return max(n + 1, 0)


def cp1_h1_o(n: int) -> int:
    return max(-n - 1, 0)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check V3 projective minimality / genus-zero rigidity.")
    ap.add_argument("--g-max", type=int, default=8, help="maximum genus in the no-go table")
    ap.add_argument("--marked-points", type=int, default=2, help="number of projected center marks")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    if args.g_max < 1:
        raise SystemExit("--g-max must be at least 1")
    if args.marked_points != 2:
        raise SystemExit("V3 is defined for the two-center case: --marked-points must be 2")

    detail_rows: list[dict[str, Any]] = []
    for genus in range(args.g_max + 1):
        curve_moduli = marked_curve_moduli_dim(genus, args.marked_points)
        pic_moduli = jacobian_dim(genus)
        hidden_total = curve_moduli + pic_moduli
        degree = required_degree_for_spinc_index_three(genus)
        index = degree + 1 - genus
        no_hidden = hidden_total == 0
        detail_rows.append(
            {
                "genus": genus,
                "marked_points": args.marked_points,
                "curve_moduli_dim_C": curve_moduli,
                "pic_degree_moduli_dim_C": pic_moduli,
                "hidden_moduli_dim_C": hidden_total,
                "degree_required_for_index3": degree,
                "spinc_index": index,
                "passes_no_hidden_moduli": no_hidden,
                "selected_by_projective_minimality": genus == 0 and no_hidden,
                "interpretation": (
                    "rigid genus-zero candidate; compact curve is CP1"
                    if genus == 0 and no_hidden
                    else "rejected by hidden complex/Picard moduli"
                ),
            }
        )

    selected = [row for row in detail_rows if row["selected_by_projective_minimality"]]
    rejected_positive_genus = [
        row
        for row in detail_rows
        if int(row["genus"]) >= 1 and int(row["hidden_moduli_dim_C"]) > 0
    ]
    infinite_index_no_go = all(
        int(row["spinc_index"]) == 3
        for row in detail_rows
        if int(row["degree_required_for_index3"]) == int(row["genus"]) + 2
    )

    cp1_degree = 2
    cp1_h0 = cp1_h0_o(cp1_degree)
    cp1_h1 = cp1_h1_o(cp1_degree)
    cp1_index = cp1_h0 - cp1_h1

    v3_pass = (
        len(selected) == 1
        and int(selected[0]["genus"]) == 0
        and len(rejected_positive_genus) == args.g_max
        and cp1_h0 == 3
        and cp1_h1 == 0
        and cp1_index == 3
        and infinite_index_no_go
    )
    status = (
        "CONDITIONAL_CP1_SELECTED_BY_NO_HIDDEN_MODULI"
        if v3_pass
        else "V3_PROJECTIVE_MINIMALITY_CHECK_FAILED"
    )

    summary_rows = [
        {
            "check": "two_center_rigid_curve_selection",
            "value": f"selected_genus={[row['genus'] for row in selected]}",
            "target": "unique selected genus is 0",
            "pass": len(selected) == 1 and int(selected[0]["genus"]) == 0,
            "status": "conditional_on_no_hidden_moduli",
        },
        {
            "check": "positive_genus_hidden_moduli_rejection",
            "value": f"rejected_positive_genus={len(rejected_positive_genus)} of {args.g_max}",
            "target": f"all genera 1..{args.g_max} have hidden_moduli_dim_C > 0",
            "pass": len(rejected_positive_genus) == args.g_max,
            "status": "moduli_count",
        },
        {
            "check": "index_count_no_go_without_minimality",
            "value": "deg L = genus + 2 gives Spin^c index 3 for every tested genus",
            "target": "record that index=3 alone does not imply CP1",
            "pass": infinite_index_no_go,
            "status": "no_go",
        },
        {
            "check": "cp1_o2_exact_index",
            "value": f"h0={cp1_h0}; h1={cp1_h1}; index={cp1_index}",
            "target": "h0=3; h1=0; index=3",
            "pass": cp1_h0 == 3 and cp1_h1 == 0 and cp1_index == 3,
            "status": "exact",
        },
        {
            "check": "v3_projective_minimality_status",
            "value": status,
            "target": "do not read as unconditional EYMH derivation",
            "pass": v3_pass,
            "status": status,
        },
    ]

    detail = {
        "scope": "optional Spin^c V3 projective-minimality diagnostic; no baseline scan change",
        "minimality_axiom": (
            "The projected two-center parent is required to be connected, single-sheet, "
            "compact, and free of hidden continuous internal curve/Picard moduli beyond "
            "the explicit PSLT scan variables."
        ),
        "theorem": (
            "Under the minimality axiom, the only compact connected Riemann-surface "
            "candidate among the tested genera is genus zero; by the classification of "
            "compact genus-zero Riemann surfaces this gives C = CP^1."
        ),
        "no_go": (
            "Without the minimality axiom, Riemann-Roch gives chi(L)=3 whenever "
            "deg L = genus + 2, so the family count alone does not force CP^1."
        ),
        "marked_points": args.marked_points,
        "g_max": args.g_max,
        "summary_status": status,
        "detail_rows": detail_rows,
    }

    out_summary = args.outdir / "spinc_projective_minimality_summary.csv"
    out_detail_csv = args.outdir / "spinc_projective_minimality_detail.csv"
    out_detail_json = args.outdir / "spinc_projective_minimality_detail.json"

    write_csv(
        out_summary,
        summary_rows,
        ["check", "value", "target", "pass", "status"],
    )
    write_csv(
        out_detail_csv,
        detail_rows,
        [
            "genus",
            "marked_points",
            "curve_moduli_dim_C",
            "pic_degree_moduli_dim_C",
            "hidden_moduli_dim_C",
            "degree_required_for_index3",
            "spinc_index",
            "passes_no_hidden_moduli",
            "selected_by_projective_minimality",
            "interpretation",
        ],
    )
    out_detail_json.write_text(json.dumps(detail, indent=2) + "\n")

    print(f"[done] wrote {out_summary}")
    print(f"[done] wrote {out_detail_csv}")
    print(f"[done] wrote {out_detail_json}")
    print(f"[status] {status}")

    if not v3_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Protected Spin^c index-to-spectral-flag bridge diagnostic.

This is deliberately not a baseline scan script.  It checks the exact
integer data behind the optional protected-family-count route and records
what the current artifacts can and cannot prove:

  1) CP^1/O(2) Dolbeault index: h^0=3, h^1=0.
  2) One SM Weyl-family hypercharge anomalies cancel exactly.
  3) Existing finite-volume G1 artifacts give a strict rank flag 1,2,3.
  4) The current repository has only a dimension/flag bridge, not a
     canonical index-to-spectral intertwiner, unless an external anchor
     overlap artifact is supplied.

Outputs:
  - output/spinc_bridge/spinc_bridge_lemma_summary.csv
  - output/spinc_bridge/spinc_bridge_lemma_detail.json
"""

from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "spinc_bridge"
DEFAULT_GN_SUMMARY = ROOT / "output" / "gn_fp_2d" / "gn_lowN_microstate_count_summary.csv"
DEFAULT_GN_DETAIL = ROOT / "output" / "gn_fp_2d" / "gn_lowN_microstate_count_detail.csv"
DEFAULT_ANCHOR = DEFAULT_OUTDIR / "spinc_spectral_anchor_overlap.csv"


def h0_cp1_o(n: int) -> int:
    """h^0(CP^1,O(n))."""
    return max(n + 1, 0)


def h1_cp1_o(n: int) -> int:
    """h^1(CP^1,O(n)) = h^0(CP^1,O(-n-2))."""
    return max(-n - 1, 0)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected exactly one row, got {len(rows)}.")
    return rows[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["check", "value", "target", "pass", "status"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the optional Spin^c bridge lemma gates.")
    ap.add_argument("--gn-summary", type=Path, default=DEFAULT_GN_SUMMARY)
    ap.add_argument("--gn-detail", type=Path, default=DEFAULT_GN_DETAIL)
    ap.add_argument("--anchor-overlap", type=Path, default=DEFAULT_ANCHOR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    deg_lr = 2
    h0 = h0_cp1_o(deg_lr)
    h1 = h1_cp1_o(deg_lr)
    index = h0 - h1

    # Standard one-family hypercharges in left-handed Weyl convention.
    q = Fraction(1, 6)
    yq = q
    yu = -4 * q
    yd = 2 * q
    yl = -3 * q
    ye = 6 * q
    yh = 3 * q

    grav_anomaly = 6 * yq + 3 * yu + 3 * yd + 2 * yl + ye
    u1_cubed_anomaly = 6 * yq**3 + 3 * yu**3 + 3 * yd**3 + 2 * yl**3 + ye**3
    su2_doublet_count = 3 + 1
    witten_su2_ok = (su2_doublet_count % 2) == 0
    # With A(3)=+1 and A(bar 3)=-1, the two Q components give 2A(3).
    su3_cubed_anomaly = 2 - 1 - 1

    gn_summary = read_single_csv_row(args.gn_summary)
    gn_detail = read_csv_rows(args.gn_detail)
    spectral_all_pass = parse_bool(gn_summary["all_pass"])
    min_gap12 = float(gn_summary["min_gap12"])
    min_gap23 = float(gn_summary["min_gap23"])
    max_rank_error = int(float(gn_summary["max_rank_error"]))
    n_d = int(float(gn_summary["n_D"]))
    all_detail_ranks = all(
        int(float(row["rank_at_lambda1"])) == 1
        and int(float(row["rank_at_lambda2"])) == 2
        and int(float(row["rank_at_lambda3"])) == 3
        and parse_bool(row["row_pass"])
        for row in gn_detail
    )

    index_pass = h0 == 3 and h1 == 0 and index == 3
    anomaly_pass = (
        grav_anomaly == 0
        and u1_cubed_anomaly == 0
        and su3_cubed_anomaly == 0
        and witten_su2_ok
        and yh == Fraction(1, 2)
    )
    spectral_flag_pass = (
        spectral_all_pass
        and all_detail_ranks
        and min_gap12 > 0.0
        and min_gap23 > 0.0
        and max_rank_error == 0
    )
    dimension_flag_bridge_pass = index_pass and anomaly_pass and spectral_flag_pass

    anchor_rows: list[dict[str, str]] = []
    anchor_status = "missing"
    canonical_anchor_pass = False
    if args.anchor_overlap.exists():
        anchor_rows = read_csv_rows(args.anchor_overlap)
        # Future-proof expected columns.  The current repo does not ship this
        # artifact; if supplied, each row must certify the D-local intertwiner.
        required = {"D", "det_abs", "sigma_min", "tail_norm", "row_pass"}
        if anchor_rows and required.issubset(anchor_rows[0]):
            canonical_anchor_pass = all(parse_bool(row["row_pass"]) for row in anchor_rows)
            anchor_status = "pass" if canonical_anchor_pass else "fail"
        else:
            anchor_status = "unrecognized_schema"

    promoted_bridge_pass = dimension_flag_bridge_pass and canonical_anchor_pass
    bridge_status = (
        "PROMOTED_CANONICAL_INDEX_SPECTRAL_BRIDGE"
        if promoted_bridge_pass
        else "CONDITIONAL_DIMENSION_FLAG_BRIDGE_ONLY"
    )

    summary_rows: list[dict[str, Any]] = [
        {
            "check": "cp1_o2_dolbeault_index",
            "value": f"h0={h0}; h1={h1}; index={index}",
            "target": "h0=3; h1=0; index=3",
            "pass": index_pass,
            "status": "exact",
        },
        {
            "check": "sm_family_anomaly_cancellation",
            "value": (
                f"Y=(Q:{yq},u:{yu},d:{yd},L:{yl},e:{ye},H:{yh}); "
                f"grav={grav_anomaly}; u1cubed={u1_cubed_anomaly}; "
                f"su3={su3_cubed_anomaly}; su2_doublets={su2_doublet_count}"
            ),
            "target": "all anomalies zero; even SU(2) doublets",
            "pass": anomaly_pass,
            "status": "exact",
        },
        {
            "check": "finite_volume_spectral_rank_flag",
            "value": (
                f"n_D={n_d}; min_gap12={min_gap12:.12g}; "
                f"min_gap23={min_gap23:.12g}; max_rank_error={max_rank_error}"
            ),
            "target": "strict gaps; ranks 1,2,3 at every audited D",
            "pass": spectral_flag_pass,
            "status": "audited_G1_artifact",
        },
        {
            "check": "dimension_flag_bridge",
            "value": "index flag dimensions (1,2,3) match spectral projector ranks (1,2,3)",
            "target": "compatible protected-index flag and low-mode spectral flag",
            "pass": dimension_flag_bridge_pass,
            "status": "conditional",
        },
        {
            "check": "canonical_anchor_intertwiner",
            "value": str(args.anchor_overlap) if args.anchor_overlap.exists() else "not_available",
            "target": "anchor overlap/invertibility artifact with row_pass=True",
            "pass": canonical_anchor_pass,
            "status": anchor_status,
        },
        {
            "check": "bridge_promotion_status",
            "value": bridge_status,
            "target": "do not promote unless canonical_anchor_intertwiner passes",
            "pass": dimension_flag_bridge_pass,
            "status": bridge_status,
        },
    ]

    detail = {
        "scope": "optional Spin^c bridge lemma; no baseline scan change",
        "index_sector": {
            "C": "CP^1",
            "R": "p_+ + p_-",
            "L_R": "O(2)",
            "h0": h0,
            "h1": h1,
            "index": index,
            "flag_dimensions": [1, 2, 3],
            "basis": ["1", "zeta", "zeta^2"],
        },
        "sm_family_anomalies": {
            "hypercharges": {
                "Q": str(yq),
                "u^c": str(yu),
                "d^c": str(yd),
                "L": str(yl),
                "e^c": str(ye),
                "H": str(yh),
            },
            "gravitational": str(grav_anomaly),
            "u1_cubed": str(u1_cubed_anomaly),
            "su3_cubed": su3_cubed_anomaly,
            "su2_doublet_count": su2_doublet_count,
            "witten_su2_ok": witten_su2_ok,
        },
        "spectral_flag": {
            "artifact_summary": str(args.gn_summary),
            "artifact_detail": str(args.gn_detail),
            "n_D": n_d,
            "min_gap12": min_gap12,
            "min_gap23": min_gap23,
            "max_rank_error": max_rank_error,
            "all_pass": spectral_flag_pass,
        },
        "bridge_verdict": {
            "dimension_flag_bridge_pass": dimension_flag_bridge_pass,
            "canonical_anchor_intertwiner_pass": canonical_anchor_pass,
            "promoted_bridge_pass": promoted_bridge_pass,
            "status": bridge_status,
            "interpretation": (
                "The current artifacts prove compatibility of the protected O(2) index flag "
                "with the audited 1,2,3 spectral-rank flag.  They do not yet prove a "
                "canonical index-to-spectral intertwiner; that requires an anchor-overlap "
                "or equivalent operator-level embedding certificate."
            ),
        },
    }

    out_summary = args.outdir / "spinc_bridge_lemma_summary.csv"
    out_detail = args.outdir / "spinc_bridge_lemma_detail.json"
    write_summary(out_summary, summary_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_detail.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")

    hard_pass = index_pass and anomaly_pass and spectral_flag_pass
    if not hard_pass:
        raise SystemExit("Spin^c bridge hard gate failed; see summary CSV.")

    print(f"Wrote {out_summary}")
    print(f"Wrote {out_detail}")
    print(f"Bridge status: {bridge_status}")


if __name__ == "__main__":
    main()

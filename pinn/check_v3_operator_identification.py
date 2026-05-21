#!/usr/bin/env python3
"""V3.2 operator-identification gate for the PINN action bridge.

V3.1 proves that the axial action adapter exactly reproduces the canonical
single-track WKB artifact.  This gate asks a different question: whether the
V3 self-adjoint cylindrical PINN spectrum is the same operator object as that
canonical negative-energy single-track artifact.

The expected safe outcome is negative.  A negative gate is useful because it
prevents a validated numerical adapter from being silently promoted into a
physical Gamma_N/P_N map for the wrong spectral object.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECTRAL = ROOT / "pinn" / "v3_downstream_spectral_table.csv"
DEFAULT_ACTION_SUMMARY = ROOT / "pinn" / "v3_energy_to_action_summary.json"
DEFAULT_ACTION_DETAIL = ROOT / "pinn" / "v3_energy_to_action_detail.csv"
DEFAULT_TRUE = ROOT / "output" / "true_single_track" / "true_results.json"
DEFAULT_ANCHORS = ROOT / "pinn" / "v2_augmented_anchor_summary.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spectral-table", type=Path, default=DEFAULT_SPECTRAL)
    p.add_argument("--action-summary", type=Path, default=DEFAULT_ACTION_SUMMARY)
    p.add_argument("--action-detail", type=Path, default=DEFAULT_ACTION_DETAIL)
    p.add_argument("--true-artifact", type=Path, default=DEFAULT_TRUE)
    p.add_argument("--anchor-summary", type=Path, default=DEFAULT_ANCHORS)
    p.add_argument("--out-prefix", default="v3_operator_identification")
    return p.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    spectral_rows = read_csv_rows(args.spectral_table)
    action_summary = json.loads(args.action_summary.read_text())
    action_rows = read_csv_rows(args.action_detail)
    true_data = json.loads(args.true_artifact.read_text())
    anchor_rows = read_csv_rows(args.anchor_summary)

    true_E = [float(x) for x in true_data["E_bound"]]
    true_turning = [int(x) for x in true_data["n_turning"]]
    spectral_E = []
    for row in spectral_rows:
        for mode in range(3):
            spectral_E.append(float(row[f"E{mode}_ritz"]))

    anchor_n_negative = [int(float(row["n_negative_selfadjoint"])) for row in anchor_rows]
    statuses = [row["status"] for row in action_rows]
    n_total = len(statuses)
    n_central = statuses.count("CENTRAL_BARRIER")
    n_nearest = statuses.count("NEAREST_FORBIDDEN_ISLAND")
    central_fraction = n_central / n_total if n_total else 0.0

    true_all_negative = all(E < 0.0 for E in true_E)
    selfadjoint_all_positive = all(E > 0.0 for E in spectral_E)
    anchor_no_negative = max(anchor_n_negative) == 0
    true_all_four_turning = all(n == 4 for n in true_turning)
    action_adapter_valid = bool(action_summary.get("true_crosscheck_pass", False))
    central_uniform = n_total > 0 and n_central == n_total
    sign_sector_compatible = not (true_all_negative and selfadjoint_all_positive)

    direct_operator_identification_pass = (
        action_adapter_valid
        and sign_sector_compatible
        and central_uniform
        and not anchor_no_negative
    )

    evidence = [
        {
            "check": "canonical_single_track_energy_sign",
            "value": "all_negative" if true_all_negative else "not_all_negative",
            "pass_for_direct_identification": not true_all_negative,
            "interpretation": "Canonical WKB artifact is a negative-energy bound-state chain.",
        },
        {
            "check": "pinn_selfadjoint_energy_sign",
            "value": "all_positive" if selfadjoint_all_positive else "not_all_positive",
            "pass_for_direct_identification": not selfadjoint_all_positive,
            "interpretation": "V3 PINN export is a positive finite-volume self-adjoint branch table.",
        },
        {
            "check": "finite_volume_anchor_negative_count",
            "value": max(anchor_n_negative),
            "pass_for_direct_identification": not anchor_no_negative,
            "interpretation": "Augmented self-adjoint anchors have no negative eigenbranches.",
        },
        {
            "check": "canonical_turning_structure",
            "value": "all_four_turning" if true_all_four_turning else "mixed",
            "pass_for_direct_identification": true_all_four_turning,
            "interpretation": "Canonical single-track action uses a fixed four-turning-point bound geometry.",
        },
        {
            "check": "pinn_action_component_uniformity",
            "value": f"{n_central}/{n_total} central",
            "pass_for_direct_identification": central_uniform,
            "interpretation": "PINN V3 action rows are mixed central-barrier/nearest-island components.",
        },
        {
            "check": "v3p1_adapter_validation",
            "value": action_adapter_valid,
            "pass_for_direct_identification": action_adapter_valid,
            "interpretation": "The adapter itself is valid on the canonical deterministic artifact.",
        },
    ]

    status = (
        "DIRECT_OPERATOR_IDENTIFICATION_PASS"
        if direct_operator_identification_pass
        else "CLOSED_NEGATIVE_DIAGNOSTIC_ONLY"
    )
    verdict = (
        "PINN_SELFADJOINT_IDENTIFIED_WITH_SINGLE_TRACK_WKB"
        if direct_operator_identification_pass
        else "NO_DIRECT_OPERATOR_IDENTIFICATION_DIAGNOSTIC_ONLY"
    )
    policy = (
        "The V3.1 S adapter may feed Gamma_N/P_N only after an independent operator theorem."
        if direct_operator_identification_pass
        else "Do not feed V3.1 PINN S_adapter values into r_N, Gamma_N, P_N, or release maps."
    )

    summary = {
        "target": "PINN V3.2 operator-identification gate",
        "status": status,
        "verdict": verdict,
        "direct_operator_identification_pass": direct_operator_identification_pass,
        "action_adapter_valid_on_true_artifact": action_adapter_valid,
        "true_energy_min": min(true_E),
        "true_energy_max": max(true_E),
        "true_all_negative": true_all_negative,
        "true_all_four_turning": true_all_four_turning,
        "selfadjoint_energy_min": min(spectral_E),
        "selfadjoint_energy_max": max(spectral_E),
        "selfadjoint_all_positive": selfadjoint_all_positive,
        "selfadjoint_anchor_n_negative_max": max(anchor_n_negative),
        "selfadjoint_anchor_no_negative": anchor_no_negative,
        "n_action_rows": n_total,
        "n_central_barrier_rows": n_central,
        "n_nearest_forbidden_island_rows": n_nearest,
        "central_barrier_fraction": central_fraction,
        "central_barrier_uniform": central_uniform,
        "policy": policy,
        "inputs": {
            "spectral_table": str(args.spectral_table.relative_to(ROOT)),
            "action_summary": str(args.action_summary.relative_to(ROOT)),
            "action_detail": str(args.action_detail.relative_to(ROOT)),
            "true_artifact": str(args.true_artifact.relative_to(ROOT)),
            "anchor_summary": str(args.anchor_summary.relative_to(ROOT)),
        },
    }

    out_dir = ROOT / "pinn"
    detail_path = out_dir / f"{args.out_prefix}_detail.csv"
    summary_path = out_dir / f"{args.out_prefix}_summary.json"
    write_csv(detail_path, evidence)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

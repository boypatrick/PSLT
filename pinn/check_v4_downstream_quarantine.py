#!/usr/bin/env python3
"""V4 downstream quarantine gate for PINN-derived actions.

This gate formalizes the policy consequence of V3.2.  If the PINN
self-adjoint spectrum is not identified with the canonical WKB object, then
the sandbox must not produce PINN-derived r_N, Gamma_N, P_N, or release-map
artifacts.  Passing V4 means the sandbox is safely quarantined as diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_V32 = ROOT / "pinn" / "v3_operator_identification_summary.json"


DISALLOWED_ARTIFACTS = [
    "pinn/v3_gamma_map.csv",
    "pinn/v3_probability_map.csv",
    "pinn/v3_pn_map.csv",
    "pinn/v4_gamma_map.csv",
    "pinn/v4_probability_map.csv",
    "pinn/v4_pn_map.csv",
    "pinn/pinn_release_map.csv",
    "pinn/pinn_gamma_map.csv",
    "pinn/pinn_probability_map.csv",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator-summary", type=Path, default=DEFAULT_V32)
    p.add_argument("--out", type=Path, default=ROOT / "pinn" / "v4_downstream_quarantine_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    operator_summary = json.loads(args.operator_summary.read_text())
    direct_pass = bool(operator_summary.get("direct_operator_identification_pass", False))
    missing_required_negative_gate = operator_summary.get("verdict") != "NO_DIRECT_OPERATOR_IDENTIFICATION_DIAGNOSTIC_ONLY"

    existing_disallowed = []
    for rel in DISALLOWED_ARTIFACTS:
        path = ROOT / rel
        if path.exists():
            existing_disallowed.append(rel)

    quarantine_pass = (not direct_pass) and (not missing_required_negative_gate) and (not existing_disallowed)
    summary = {
        "target": "PINN V4 downstream quarantine gate",
        "status": "QUARANTINE_PASS_DIAGNOSTIC_ONLY" if quarantine_pass else "QUARANTINE_FAIL_REVIEW_REQUIRED",
        "quarantine_pass": quarantine_pass,
        "operator_identification_verdict": operator_summary.get("verdict"),
        "direct_operator_identification_pass": direct_pass,
        "blocked_quantities": ["r_N", "Gamma_N", "P_N", "release maps", "paper baseline figures"],
        "allowed_quantities": [
            "D -> E_k(D) spectral export",
            "D,E -> S diagnostic adapter",
            "finite-volume cross-check prompts",
            "visual/diagnostic emulator plots",
        ],
        "existing_disallowed_artifacts": existing_disallowed,
        "policy": (
            "PINN outputs remain outside the manuscript baseline.  They may "
            "suggest where to run deterministic finite-volume checks, but they "
            "must not define Gamma_N, P_N, or release-map acceptance."
        ),
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V5 final closure gate for the PINN sandbox.

V5 is a metadata/freeze gate.  It checks that the sandbox has a positive V2.6
spectral-emulator freeze, a valid V3.0 spectral export, a validated V3.1
energy-to-action adapter, a negative V3.2 operator-identification gate, and a
passing V4 quarantine policy.  The intended final state is a closed diagnostic
PINN package, not a manuscript-baseline replacement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v26", type=Path, default=ROOT / "pinn" / "v2_seed_stability_metrics.json")
    p.add_argument("--v30", type=Path, default=ROOT / "pinn" / "v3_downstream_manifest.json")
    p.add_argument("--v31", type=Path, default=ROOT / "pinn" / "v3_energy_to_action_summary.json")
    p.add_argument("--v32", type=Path, default=ROOT / "pinn" / "v3_operator_identification_summary.json")
    p.add_argument("--v40", type=Path, default=ROOT / "pinn" / "v4_downstream_quarantine_summary.json")
    p.add_argument("--out", type=Path, default=ROOT / "pinn" / "v5_pinn_closure_summary.json")
    return p.parse_args()


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    v26 = load(args.v26)
    v30 = load(args.v30)
    v31 = load(args.v31)
    v32 = load(args.v32)
    v40 = load(args.v40)

    checks = [
        {
            "name": "V2.6 seed-stability freeze",
            "pass": bool(v26.get("gate_pass", False)),
            "status": v26.get("verdict"),
        },
        {
            "name": "V3.0 downstream spectral export",
            "pass": v30.get("status") == "OPENED_POSITIVE_EXPORT_ONLY" and not bool(v30.get("needs_finite_volume_check", True)),
            "status": v30.get("status"),
        },
        {
            "name": "V3.1 energy-to-action adapter",
            "pass": bool(v31.get("true_crosscheck_pass", False)) and v31.get("status") == "DIAGNOSTIC_BRIDGE_ONLY",
            "status": v31.get("status"),
        },
        {
            "name": "V3.2 operator-identification gate",
            "pass": v32.get("verdict") == "NO_DIRECT_OPERATOR_IDENTIFICATION_DIAGNOSTIC_ONLY",
            "status": v32.get("status"),
        },
        {
            "name": "V4 downstream quarantine",
            "pass": bool(v40.get("quarantine_pass", False)),
            "status": v40.get("status"),
        },
    ]
    closure_pass = all(bool(row["pass"]) for row in checks)
    summary = {
        "target": "PINN V5 final sandbox closure",
        "status": "PINN_SANDBOX_CLOSED_DIAGNOSTIC_ONLY" if closure_pass else "PINN_SANDBOX_OPEN_REVIEW_REQUIRED",
        "closure_pass": closure_pass,
        "checks": checks,
        "final_policy": (
            "The PINN line is frozen as an optional differentiable spectral "
            "emulator/diagnostic package.  It does not alter paper/main.tex, "
            "the release baseline, Gamma_N, P_N, or submission figures."
        ),
        "next_allowed_use": (
            "Use PINN only to nominate deterministic finite-volume follow-up "
            "points or to produce explicitly diagnostic companion plots."
        ),
        "inputs": {
            "v2_6": str(args.v26.relative_to(ROOT)),
            "v3_0": str(args.v30.relative_to(ROOT)),
            "v3_1": str(args.v31.relative_to(ROOT)),
            "v3_2": str(args.v32.relative_to(ROOT)),
            "v4_0": str(args.v40.relative_to(ROOT)),
        },
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit V0.4 reference-operator alignment before training spectral PINNs.

The existing canonical single-track artifact comes from code/true_single_track_solver.py.
Before using it as a neural holdout, this script checks two things:

1. whether the finite-difference Hamiltonian is symmetric in the ordinary
   Euclidean matrix inner product expected by scipy.sparse.linalg.eigsh;
2. whether the current solver output at a selected D matches the checked-in
   artifact output/true_single_track/true_results.json.

This is an alignment audit, not a paper result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "pinn" / "runs"
TRUE_SINGLE_TRACK = ROOT / "output" / "true_single_track" / "true_results.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--D", type=float, default=12.0)
    p.add_argument("--n-rho", type=int, default=50)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--L-rho", type=float, default=4.0)
    p.add_argument("--L-z", type=float, default=20.0)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_artifact(D):
    data = json.loads(TRUE_SINGLE_TRACK.read_text())
    if D not in data["D"]:
        return None
    i = data["D"].index(D)
    return {
        "E_artifact": float(data["E_bound"][i]),
        "omega_artifact": float(data["omega"][i]),
        "n_bound_artifact": int(data["n_bound"][i]),
        "S_artifact": float(data["S_N"][i]),
        "source": str(TRUE_SINGLE_TRACK.relative_to(ROOT)),
    }


def main():
    args = parse_args()
    run_name = args.run_name or f"v0p4_reference_alignment_D{args.D:g}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(run_dir / ".mplconfig"))
    (run_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "code"))
    from true_single_track_solver import TrueSolver  # noqa: WPS433

    solver = TrueSolver(
        args.D,
        n_rho=args.n_rho,
        n_z=args.n_z,
        L_rho=args.L_rho,
        L_z=args.L_z,
    )
    H = solver.build_hamiltonian()
    asym = H - H.T
    asym_fro = float((asym.multiply(asym)).sum() ** 0.5)
    norm_fro = float((H.multiply(H)).sum() ** 0.5)
    eigsh_values = [float(x) for x in solver.solve(n_eigenvalues=args.k)]
    artifact = load_artifact(args.D)
    negative = [x for x in eigsh_values if x < 0]
    current_first_negative = negative[0] if negative else None
    metrics = {
        "target": "V0.4 reference alignment audit",
        "D": args.D,
        "n_rho": args.n_rho,
        "n_z": args.n_z,
        "L_rho": args.L_rho,
        "L_z": args.L_z,
        "matrix_shape": list(H.shape),
        "matrix_nnz": int(H.nnz),
        "fro_norm": norm_fro,
        "asym_fro": asym_fro,
        "asym_rel": asym_fro / norm_fro if norm_fro else None,
        "eigsh_values_current_code": eigsh_values,
        "current_first_negative": current_first_negative,
        "artifact": artifact,
    }
    if artifact is not None and current_first_negative is not None:
        metrics["current_vs_artifact_E_abs_delta"] = abs(current_first_negative - artifact["E_artifact"])
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

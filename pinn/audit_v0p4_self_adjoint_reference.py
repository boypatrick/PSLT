#!/usr/bin/env python3
"""Construct a self-adjoint weighted cylindrical reference for V0.4.

The axisymmetric shifted operator is

    H_U u = -rho^{-1} d_rho(rho d_rho u) - d_z^2 u + U u.

It is self-adjoint in <u,v> = int rho u v d rho dz.  This script builds a
finite-volume/energy discretization as a generalized symmetric eigenproblem

    K u = E M u,

where M is the positive cylindrical mass matrix.  It compares the result to the
checked-in true_single_track artifact and to the current legacy nonsymmetric
solver output.
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
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--m0", type=float, default=1.0)
    p.add_argument("--xi", type=float, default=0.0)
    p.add_argument("--n-rho", type=int, default=50)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--L-rho", type=float, default=4.0)
    p.add_argument("--L-z", type=float, default=20.0)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--skip-legacy", action="store_true", help="Skip the legacy nonsymmetric solver comparison.")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_artifact(D):
    data = json.loads(TRUE_SINGLE_TRACK.read_text())
    if D not in data["D"]:
        return None
    i = data["D"].index(D)
    omega = float(data["omega"][i])
    return {
        "E_artifact": float(data["E_bound"][i]),
        "omega_artifact": omega,
        "lambda_artifact": omega * omega,
        "n_bound_artifact": int(data["n_bound"][i]),
        "S_artifact": float(data["S_N"][i]),
        "source": str(TRUE_SINGLE_TRACK.relative_to(ROOT)),
    }


def main():
    args = parse_args()
    try:
        import numpy as np
        from scipy.sparse import coo_matrix, diags
        from scipy.sparse.linalg import eigsh
    except ModuleNotFoundError as exc:
        raise SystemExit("This audit requires scipy in the active Python environment.") from exc

    def omega(rho, z):
        rp = np.sqrt(rho * rho + (z - args.D / 2.0) ** 2 + args.eps * args.eps)
        rm = np.sqrt(rho * rho + (z + args.D / 2.0) ** 2 + args.eps * args.eps)
        return 1.0 + args.a * (1.0 / rp + 1.0 / rm)

    def lap_omega(rho, z):
        rp2 = rho * rho + (z - args.D / 2.0) ** 2 + args.eps * args.eps
        rm2 = rho * rho + (z + args.D / 2.0) ** 2 + args.eps * args.eps
        return args.a * (-3.0 * args.eps * args.eps / (rp2 ** 2.5) - 3.0 * args.eps * args.eps / (rm2 ** 2.5))

    def U(rho, z):
        om = omega(rho, z)
        return args.m0 * args.m0 * (om * om - 1.0) + (1.0 - 6.0 * args.xi) * lap_omega(rho, z) / om

    nr, nz = args.n_rho, args.n_z
    dr = args.L_rho / nr
    dz = 2.0 * args.L_z / nz
    rho = (np.arange(nr) + 0.5) * dr
    z = -args.L_z + (np.arange(nz) + 0.5) * dz
    RHO, Z = np.meshgrid(rho, z, indexing="ij")
    U_grid = U(RHO, Z)
    N = nr * nz

    def idx(i, j):
        return i * nz + j

    rows = []
    cols = []
    vals = []
    m_diag = np.zeros(N)

    def add(a, b, value):
        rows.append(a)
        cols.append(b)
        vals.append(value)

    for i in range(nr):
        for j in range(nz):
            p = idx(i, j)
            mass = rho[i] * dr * dz
            m_diag[p] = mass
            add(p, p, mass * U_grid[i, j])

    # Radial stiffness: natural at rho=0, Dirichlet at rho=L_rho.
    for i in range(nr - 1):
        rho_face = (i + 1) * dr
        c = rho_face * dz / dr
        for j in range(nz):
            p, q = idx(i, j), idx(i + 1, j)
            add(p, p, c); add(q, q, c)
            add(p, q, -c); add(q, p, -c)
    rho_outer = args.L_rho
    c_outer = 2.0 * rho_outer * dz / dr
    for j in range(nz):
        p = idx(nr - 1, j)
        add(p, p, c_outer)

    # z stiffness: Dirichlet at z=+-L_z.
    for i in range(nr):
        c = rho[i] * dr / dz
        for j in range(nz - 1):
            p, q = idx(i, j), idx(i, j + 1)
            add(p, p, c); add(q, q, c)
            add(p, q, -c); add(q, p, -c)
        c_boundary = 2.0 * rho[i] * dr / dz
        add(idx(i, 0), idx(i, 0), c_boundary)
        add(idx(i, nz - 1), idx(i, nz - 1), c_boundary)

    K = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    M = diags(m_diag, 0, shape=(N, N), format="csr")
    asym = K - K.T
    asym_fro = float((asym.multiply(asym)).sum() ** 0.5)
    norm_fro = float((K.multiply(K)).sum() ** 0.5)

    evals, _ = eigsh(K, k=args.k, M=M, which="SA", tol=1.0e-9)
    evals = np.sort(evals)
    omegas = np.sqrt(np.maximum(evals + args.m0 * args.m0, 0.0))
    artifact = load_artifact(args.D)

    legacy = None
    if args.skip_legacy:
        run_name = args.run_name or f"v0p4_selfadjoint_reference_D{args.D:g}"
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        legacy = {"skipped": True}
    else:
        try:
            run_name = args.run_name or f"v0p4_selfadjoint_reference_D{args.D:g}"
            run_dir = RUNS_DIR / run_name
            os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
            os.environ.setdefault("XDG_CACHE_HOME", str(run_dir / ".mplconfig"))
            (run_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
            sys.path.insert(0, str(ROOT / "code"))
            from true_single_track_solver import TrueSolver  # noqa: WPS433

            solver = TrueSolver(args.D, n_rho=nr, n_z=nz, L_rho=args.L_rho, L_z=args.L_z)
            legacy_vals = solver.solve(n_eigenvalues=args.k)
            legacy = [float(x) for x in legacy_vals]
        except Exception as exc:  # pragma: no cover - audit only
            legacy = {"error": repr(exc)}
            run_name = args.run_name or f"v0p4_selfadjoint_reference_D{args.D:g}"
            run_dir = RUNS_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "target": "V0.4.1 self-adjoint cylindrical reference audit",
        "D": args.D,
        "n_rho": nr,
        "n_z": nz,
        "L_rho": args.L_rho,
        "L_z": args.L_z,
        "dr": dr,
        "dz": dz,
        "matrix_shape": [N, N],
        "K_nnz": int(K.nnz),
        "K_asym_fro": asym_fro,
        "K_asym_rel": asym_fro / norm_fro if norm_fro else None,
        "M_min": float(m_diag.min()),
        "M_max": float(m_diag.max()),
        "E_selfadjoint": [float(x) for x in evals],
        "omega_selfadjoint": [float(x) for x in omegas],
        "n_negative_selfadjoint": int(np.sum(evals < 0.0)),
        "artifact": artifact,
        "legacy_current_solver_E": legacy,
    }
    if artifact is not None:
        metrics.update({
            "selfadjoint_first_vs_artifact_E_abs_delta": abs(float(evals[0]) - artifact["E_artifact"]),
            "selfadjoint_first_vs_artifact_omega_abs_delta": abs(float(omegas[0]) - artifact["omega_artifact"]),
        })
    if isinstance(legacy, list):
        neg = [x for x in legacy if x < 0]
        if neg:
            metrics["selfadjoint_first_vs_legacy_first_negative_abs_delta"] = abs(float(evals[0]) - neg[0])

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V0.4 localized 2D Ritz-basis validation for the frozen PSLT geometry.

This is a deterministic variational smoke test, not a neural proof.  It checks
whether a localized two-center basis in (s=rho^2,z) can see the bound sector
that the plain 1D axial MLP misses.

Axisymmetric energy with measure rho d rho dz = 1/2 ds dz:

    E[u] = int (2 s |u_s|^2 + 1/2 |u_z|^2 + 1/2 U |u|^2) ds dz
           / int (1/2 |u|^2) ds dz.

The basis is made of localized Gaussians near the two centers, multiplied by a
hard box envelope.  The result is compared to output/true_single_track, but the
finite-difference/Sturm chain remains the source of truth.
"""

from __future__ import annotations

import argparse
import json
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
    p.add_argument("--rho-max", type=float, default=4.0)
    p.add_argument("--z-max", type=float, default=20.0)
    p.add_argument("--n-s", type=int, default=120)
    p.add_argument("--n-z", type=int, default=600)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_reference(D):
    if not TRUE_SINGLE_TRACK.exists():
        return None
    data = json.loads(TRUE_SINGLE_TRACK.read_text())
    if D not in data["D"]:
        return None
    i = data["D"].index(D)
    omega = float(data["omega"][i])
    return {
        "E_ref": float(data["E_bound"][i]),
        "omega_ref": omega,
        "lambda_ref": omega * omega,
        "n_bound_ref": int(data["n_bound"][i]),
        "source": str(TRUE_SINGLE_TRACK.relative_to(ROOT)),
    }


def main():
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch required: python3 -m pip install -r pinn/requirements.txt") from exc

    dtype = torch.float64
    s_max = args.rho_max * args.rho_max
    s = torch.linspace(0.0, s_max, args.n_s, dtype=dtype).reshape(-1, 1)
    z = torch.linspace(-args.z_max, args.z_max, args.n_z, dtype=dtype).reshape(1, -1)
    ds = s_max / (args.n_s - 1)
    dz = 2.0 * args.z_max / (args.n_z - 1)
    ws = torch.ones(args.n_s, 1, dtype=dtype) * ds
    wz = torch.ones(1, args.n_z, dtype=dtype) * dz
    ws[0] *= 0.5; ws[-1] *= 0.5
    wz[:, 0] *= 0.5; wz[:, -1] *= 0.5
    W = ws @ wz

    def omega(s, z):
        rp = torch.sqrt(s + (z - args.D / 2.0) ** 2 + args.eps * args.eps)
        rm = torch.sqrt(s + (z + args.D / 2.0) ** 2 + args.eps * args.eps)
        return 1.0 + args.a * (1.0 / rp + 1.0 / rm)

    def lap_omega(s, z):
        rp2 = s + (z - args.D / 2.0) ** 2 + args.eps * args.eps
        rm2 = s + (z + args.D / 2.0) ** 2 + args.eps * args.eps
        return args.a * (-3.0 * args.eps * args.eps / (rp2 ** 2.5) - 3.0 * args.eps * args.eps / (rm2 ** 2.5))

    om = omega(s, z)
    U = args.m0 * args.m0 * (om * om - 1.0) + (1.0 - 6.0 * args.xi) * lap_omega(s, z) / om

    env_s = 1.0 - s / s_max
    env_z = 1.0 - (z / args.z_max) ** 2
    envelope = torch.clamp(env_s, min=0.0) * torch.clamp(env_z, min=0.0)

    basis = []
    labels = []
    sig_s_values = [0.015, 0.025, 0.04, 0.07, 0.12, 0.2, 0.35, 0.6]
    sig_z_values = [0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.1]
    for c in [-args.D / 2.0, args.D / 2.0]:
        for sig_s in sig_s_values:
            for sig_z in sig_z_values:
                g = torch.exp(-0.5 * s / sig_s - 0.5 * ((z - c) / sig_z) ** 2)
                basis.append((envelope * g).reshape(-1))
                labels.append((c, sig_s, sig_z))
                x = (z - c) / sig_z
                basis.append((envelope * x * g).reshape(-1))
                labels.append((c, sig_s, sig_z, "odd_z"))
    Phi = torch.stack(basis, dim=1)  # points x M
    M = Phi.shape[1]

    # Finite-difference derivatives on the tensor grid for each basis function.
    Phi_grid = Phi.T.reshape(M, args.n_s, args.n_z)
    dPhi_ds = torch.zeros_like(Phi_grid)
    dPhi_dz = torch.zeros_like(Phi_grid)
    dPhi_ds[:, 1:-1, :] = (Phi_grid[:, 2:, :] - Phi_grid[:, :-2, :]) / (2.0 * ds)
    dPhi_ds[:, 0, :] = (Phi_grid[:, 1, :] - Phi_grid[:, 0, :]) / ds
    dPhi_ds[:, -1, :] = (Phi_grid[:, -1, :] - Phi_grid[:, -2, :]) / ds
    dPhi_dz[:, :, 1:-1] = (Phi_grid[:, :, 2:] - Phi_grid[:, :, :-2]) / (2.0 * dz)
    dPhi_dz[:, :, 0] = (Phi_grid[:, :, 1] - Phi_grid[:, :, 0]) / dz
    dPhi_dz[:, :, -1] = (Phi_grid[:, :, -1] - Phi_grid[:, :, -2]) / dz

    W_flat = W.reshape(-1, 1)
    S = 0.5 * (Phi.T @ (W_flat * Phi))
    dS_flat = dPhi_ds.reshape(M, -1).T
    dZ_flat = dPhi_dz.reshape(M, -1).T
    s_flat = s.repeat(1, args.n_z).reshape(-1, 1)
    U_flat = U.reshape(-1, 1)
    H = (
        dS_flat.T @ (W_flat * (2.0 * s_flat) * dS_flat)
        + dZ_flat.T @ (W_flat * 0.5 * dZ_flat)
        + Phi.T @ (W_flat * 0.5 * U_flat * Phi)
    )
    H = 0.5 * (H + H.T)
    S = 0.5 * (S + S.T)

    eval_S, evec_S = torch.linalg.eigh(S)
    mask = eval_S > 1.0e-10
    Sinvhalf = evec_S[:, mask] @ torch.diag(1.0 / torch.sqrt(eval_S[mask]))
    Hw = Sinvhalf.T @ H @ Sinvhalf
    evals, coeffs_w = torch.linalg.eigh(0.5 * (Hw + Hw.T))
    E0 = float(evals[0])
    omega0 = float(torch.sqrt(torch.clamp(evals[0] + args.m0 * args.m0, min=0.0)))
    ref = load_reference(args.D)
    metrics = {
        "target": "2D localized Gaussian Ritz basis",
        "D": args.D,
        "rho_max": args.rho_max,
        "z_max": args.z_max,
        "n_s": args.n_s,
        "n_z": args.n_z,
        "basis_size": int(M),
        "retained_basis_rank": int(mask.sum()),
        "E0_ritz": E0,
        "omega0_ritz": omega0,
        "lambda0_from_E": E0 + args.m0 * args.m0,
        "lowest_E_values": [float(x) for x in evals[:8]],
        "reference": ref,
    }
    if ref is not None:
        metrics.update({
            "E_abs_error_ref": abs(E0 - ref["E_ref"]),
            "omega_abs_error_ref": abs(omega0 - ref["omega_ref"]),
            "lambda_abs_error_ref": abs((E0 + args.m0 * args.m0) - ref["lambda_ref"]),
        })

    run_name = args.run_name or f"v0p4_ritz_2d_localized_D{args.D:g}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

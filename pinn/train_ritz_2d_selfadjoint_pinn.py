#!/usr/bin/env python3
"""V0.5 weighted Ritz-PINN for the self-adjoint cylindrical PSLT operator.

This script targets the repaired V0.4.1 reference operator, not the legacy
single-track artifact.  In axisymmetric variables s=rho^2, the shifted
self-adjoint operator is

    H_U u = -4 s u_ss - 4 u_s - u_zz + U(s,z) u,
    U = V_eff - m0^2.

The Rayleigh quotient is evaluated in the cylindrical measure
rho d rho dz = (1/2) ds dz, so the common 1/2 cancels:

    E[u] = int (4s |u_s|^2 + |u_z|^2 + U |u|^2) ds dz
           / int |u|^2 ds dz.

A hard Dirichlet envelope enforces the finite-box boundary at s=s_max and
z=+-L_z, while the s=0 axis is left natural.  This is a sandbox spectral PINN
experiment; deterministic self-adjoint finite-volume audits remain the reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

from accelerator_utils import select_torch_device


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "pinn" / "runs"
SELFADJOINT_REFERENCE = RUNS_DIR / "v0p4p1_selfadjoint_D12_n50x500" / "metrics.json"


def format_d_label(D: float) -> str:
    if abs(D - round(D)) < 1.0e-12:
        return str(int(round(D)))
    return str(D).replace(".", "p")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--D", type=float, default=12.0)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--m0", type=float, default=1.0)
    p.add_argument("--xi", type=float, default=0.0)
    p.add_argument("--L-rho", type=float, default=4.0)
    p.add_argument("--L-z", type=float, default=20.0)
    p.add_argument("--n-s", type=int, default=64)
    p.add_argument("--n-z", type=int, default=160)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--base-mode", choices=["box", "two-lobe", "mixed"], default="box")
    p.add_argument("--rho-width", type=float, default=0.8)
    p.add_argument("--z-width", type=float, default=1.2)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--n-residual", type=int, default=4096)
    p.add_argument("--reference-E", type=float, default=None)
    p.add_argument("--make-plots", action="store_true")
    p.add_argument("--plot-grid-s", type=int, default=140)
    p.add_argument("--plot-grid-z", type=int, default=220)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_reference_energy(D: float, explicit: float | None):
    if explicit is not None:
        return {"E0_selfadjoint_reference": float(explicit), "source": "--reference-E"}
    d_label = format_d_label(D)
    reference_candidates = [
        RUNS_DIR / f"v0p5p1_selfadjoint_D{d_label}_n50x500" / "metrics.json",
        RUNS_DIR / f"v1_selfadjoint_D{d_label}_n50x500_k4" / "metrics.json",
    ]
    for reference_path in reference_candidates:
        if not reference_path.exists():
            continue
        data = json.loads(reference_path.read_text())
        values = data.get("E_selfadjoint") or []
        if values:
            return {
                "E0_selfadjoint_reference": float(values[0]),
                "omega0_selfadjoint_reference": float(data.get("omega_selfadjoint", [math.nan])[0]),
                "source": str(reference_path.relative_to(ROOT)),
            }
    if abs(D - 12.0) < 1.0e-12 and SELFADJOINT_REFERENCE.exists():
        data = json.loads(SELFADJOINT_REFERENCE.read_text())
        values = data.get("E_selfadjoint") or []
        if values:
            return {
                "E0_selfadjoint_reference": float(values[0]),
                "omega0_selfadjoint_reference": float(data.get("omega_selfadjoint", [math.nan])[0]),
                "source": str(SELFADJOINT_REFERENCE.relative_to(ROOT)),
            }
    if abs(D - 12.0) < 1.0e-12:
        return {
            "E0_selfadjoint_reference": 0.7898376934,
            "omega0_selfadjoint_reference": math.sqrt(1.7898376934),
            "source": "V0.4.1 documented fallback",
        }
    return None


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install dependencies with: "
            "python3 -m pip install -r pinn/requirements.txt"
        ) from exc

    torch.manual_seed(args.seed)
    try:
        device, accelerator = select_torch_device(torch, args.device)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    dtype = torch.float32
    s_max = args.L_rho * args.L_rho

    class TrialNet(nn.Module):
        def __init__(self):
            super().__init__()
            modules = [nn.Linear(2, args.hidden), nn.Tanh()]
            for _ in range(args.layers - 1):
                modules += [nn.Linear(args.hidden, args.hidden), nn.Tanh()]
            modules += [nn.Linear(args.hidden, 1)]
            self.net = nn.Sequential(*modules)
            # Keep a nonzero broad mode present from step one.  The MLP learns
            # deviations from this finite-box ground-shape ansatz.
            self.base = nn.Parameter(torch.tensor(1.0, dtype=dtype))

        def forward(self, s, z):
            x = 2.0 * s / s_max - 1.0
            y = z / args.L_z
            envelope = torch.clamp(1.0 - s / s_max, min=0.0) * torch.clamp(1.0 - y * y, min=0.0)
            mlp = self.net(torch.cat([x, y], dim=1))
            box = torch.ones_like(z)
            rho_part = torch.exp(-s / max(args.rho_width * args.rho_width, 1.0e-12))
            z_width2 = max(args.z_width * args.z_width, 1.0e-12)
            lobe = rho_part * (
                torch.exp(-((z - args.D / 2.0) ** 2) / z_width2)
                + torch.exp(-((z + args.D / 2.0) ** 2) / z_width2)
            )
            if args.base_mode == "box":
                base_shape = box
            elif args.base_mode == "two-lobe":
                base_shape = lobe
            else:
                base_shape = box + lobe
            return envelope * (self.base * base_shape + args.residual_scale * mlp)

    def omega_sz(s, z):
        rp2 = s + (z - args.D / 2.0) ** 2 + args.eps * args.eps
        rm2 = s + (z + args.D / 2.0) ** 2 + args.eps * args.eps
        return 1.0 + args.a * (torch.rsqrt(rp2) + torch.rsqrt(rm2))

    def lap_omega_sz(s, z):
        rp2 = s + (z - args.D / 2.0) ** 2 + args.eps * args.eps
        rm2 = s + (z + args.D / 2.0) ** 2 + args.eps * args.eps
        return args.a * (-3.0 * args.eps * args.eps * rp2.pow(-2.5) - 3.0 * args.eps * args.eps * rm2.pow(-2.5))

    def U_sz(s, z):
        om = omega_sz(s, z)
        return args.m0 * args.m0 * (om * om - 1.0) + (1.0 - 6.0 * args.xi) * lap_omega_sz(s, z) / om

    def quadrature_grid(n_s: int, n_z: int, requires_grad: bool = True):
        # Midpoint rule avoids evaluating exactly on the hard boundary and on
        # the Plummer center.  The common ds dz factor cancels in the quotient.
        s1 = (torch.arange(n_s, device=device, dtype=dtype) + 0.5) * (s_max / n_s)
        z1 = -args.L_z + (torch.arange(n_z, device=device, dtype=dtype) + 0.5) * (2.0 * args.L_z / n_z)
        S, Z = torch.meshgrid(s1, z1, indexing="ij")
        s = S.reshape(-1, 1).detach().clone().requires_grad_(requires_grad)
        z = Z.reshape(-1, 1).detach().clone().requires_grad_(requires_grad)
        return s, z

    def rayleigh(model, s, z):
        u = model(s, z)
        du_ds, du_dz = torch.autograd.grad(u, (s, z), torch.ones_like(u), create_graph=True)
        U = U_sz(s, z)
        norm = torch.mean(u * u)
        kinetic_s = torch.mean(4.0 * s * du_ds * du_ds)
        kinetic_z = torch.mean(du_dz * du_dz)
        potential = torch.mean(U * u * u)
        energy = (kinetic_s + kinetic_z + potential) / torch.clamp(norm, min=1.0e-12)
        return energy, norm, kinetic_s, kinetic_z, potential

    def residual_metrics(model, energy_value: float):
        n_total = args.n_s * args.n_z
        n_res = min(args.n_residual, n_total)
        # Deterministic subsampling of the same midpoint grid.
        s_full, z_full = quadrature_grid(args.n_s, args.n_z, requires_grad=False)
        idx = torch.linspace(0, n_total - 1, n_res, device=device).round().long()
        s = s_full[idx].detach().clone().requires_grad_(True)
        z = z_full[idx].detach().clone().requires_grad_(True)
        u = model(s, z)
        du_ds, du_dz = torch.autograd.grad(u, (s, z), torch.ones_like(u), create_graph=True)
        d2u_ds2 = torch.autograd.grad(du_ds, s, torch.ones_like(du_ds), create_graph=True)[0]
        d2u_dz2 = torch.autograd.grad(du_dz, z, torch.ones_like(du_dz), create_graph=True)[0]
        H_u = -4.0 * s * d2u_ds2 - 4.0 * du_ds - d2u_dz2 + U_sz(s, z) * u
        E = torch.tensor(energy_value, device=device, dtype=dtype)
        res = H_u - E * u
        norm = torch.sqrt(torch.mean(u * u))
        res_l2 = torch.sqrt(torch.mean(res * res)) / torch.clamp(norm, min=1.0e-12)
        return {
            "strong_residual_l2_over_rms_u": float(res_l2.detach().cpu()),
            "strong_residual_median_abs_over_rms_u": float((torch.median(torch.abs(res)) / torch.clamp(norm, min=1.0e-12)).detach().cpu()),
            "u_rms_residual_grid": float(norm.detach().cpu()),
            "u_max_abs_residual_grid": float(torch.max(torch.abs(u.detach())).cpu()),
            "n_residual": int(n_res),
        }

    model = TrialNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    s_quad, z_quad = quadrature_grid(args.n_s, args.n_z, requires_grad=True)
    history = []

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        energy, norm, kinetic_s, kinetic_z, potential = rayleigh(model, s_quad, z_quad)
        # The quotient is scale invariant.  This tiny monitor discourages only
        # pathological collapse of the represented trial amplitude.
        loss_norm = 1.0e-5 * torch.log(torch.clamp(norm, min=1.0e-12)).pow(2)
        loss = energy + loss_norm
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "E": float(energy.detach().cpu()),
                "omega": float(torch.sqrt(torch.clamp(energy.detach() + args.m0 * args.m0, min=0.0)).cpu()),
                "norm": float(norm.detach().cpu()),
                "kinetic_s": float(kinetic_s.detach().cpu()),
                "kinetic_z": float(kinetic_z.detach().cpu()),
                "potential": float(potential.detach().cpu()),
                "base": float(model.base.detach().cpu()),
                "loss_norm": float(loss_norm.detach().cpu()),
            }
            history.append(row)
            print(json.dumps(row))

    energy, norm, kinetic_s, kinetic_z, potential = rayleigh(model, s_quad, z_quad)
    E_pinn = float(energy.detach().cpu())
    reference = load_reference_energy(args.D, args.reference_E)
    metrics = {
        "target": "V0.5 weighted Ritz-PINN self-adjoint cylindrical reference",
        "operator": "H_U=-4s d_s^2 -4 d_s - d_z^2 + U(s,z)",
        "functional": "E=int(4s*u_s^2 + u_z^2 + U*u^2) ds dz / int(u^2) ds dz",
        "D": float(args.D),
        "L_rho": float(args.L_rho),
        "L_z": float(args.L_z),
        "s_max": float(s_max),
        "n_s": int(args.n_s),
        "n_z": int(args.n_z),
        "steps": int(args.steps),
        "base_mode": args.base_mode,
        "rho_width": float(args.rho_width),
        "z_width": float(args.z_width),
        "E_pinn": E_pinn,
        "omega_pinn": math.sqrt(max(E_pinn + args.m0 * args.m0, 0.0)),
        "norm": float(norm.detach().cpu()),
        "kinetic_s": float(kinetic_s.detach().cpu()),
        "kinetic_z": float(kinetic_z.detach().cpu()),
        "potential": float(potential.detach().cpu()),
        "base": float(model.base.detach().cpu()),
        "device": str(device),
        "accelerator": accelerator,
        "reference": reference,
    }
    if reference is not None:
        E_ref = reference["E0_selfadjoint_reference"]
        metrics["E_abs_error_selfadjoint_reference"] = abs(E_pinn - E_ref)
        metrics["E_rel_error_selfadjoint_reference"] = abs(E_pinn - E_ref) / max(abs(E_ref), 1.0e-12)
    metrics.update(residual_metrics(model, E_pinn))

    run_name = args.run_name or datetime.now().strftime("v0p5_ritz_2d_selfadjoint_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (run_dir / "history.csv").open("w", newline="") as f:
        fields = ["step", "E", "omega", "norm", "kinetic_s", "kinetic_z", "potential", "base", "loss_norm"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    if args.make_plots:
        cache_dir = run_dir / ".mplconfig"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        s_plot = (torch.arange(args.plot_grid_s, device=device, dtype=dtype) + 0.5) * (s_max / args.plot_grid_s)
        z_plot = -args.L_z + (torch.arange(args.plot_grid_z, device=device, dtype=dtype) + 0.5) * (2.0 * args.L_z / args.plot_grid_z)
        S, Z = torch.meshgrid(s_plot, z_plot, indexing="ij")
        s_flat = S.reshape(-1, 1)
        z_flat = Z.reshape(-1, 1)
        with torch.no_grad():
            u = model(s_flat, z_flat).detach().cpu().numpy().reshape(args.plot_grid_s, args.plot_grid_z)
            U = U_sz(s_flat, z_flat).detach().cpu().numpy().reshape(args.plot_grid_s, args.plot_grid_z)
        rho = np.sqrt(s_plot.detach().cpu().numpy())
        z_np = z_plot.detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=180, constrained_layout=True)
        im0 = axes[0].pcolormesh(z_np, rho, U, shading="auto", cmap="viridis")
        axes[0].set_title(r"Shifted potential $U=V_{eff}-m_0^2$")
        axes[0].set_xlabel("z")
        axes[0].set_ylabel(r"$\rho$")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].pcolormesh(z_np, rho, u / (np.max(np.abs(u)) + 1.0e-12), shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_title(r"Ritz-PINN trial $u/\max|u|$")
        axes[1].set_xlabel("z")
        axes[1].set_ylabel(r"$\rho$")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        fig.savefig(run_dir / "diagnostics.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

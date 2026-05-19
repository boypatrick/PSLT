#!/usr/bin/env python3
"""V0.4 axial single-track Ritz PINN smoke test.

This is the first spectral PINN experiment in the sandbox.  It intentionally
starts with the 1D axial operator, because the repository already has a
single-track deterministic reference in output/true_single_track/true_results.json.

The operator is

    H_D = -d_z^2 + U(0,z;D),
    U = V_eff - m0^2
      = m0^2 (Omega^2 - 1) + (1 - 6 xi) (Delta Omega) / Omega,

with the same analytic two-center Plummer background as V0.2.  The trial wave
function uses a hard Dirichlet envelope on [-z_max, z_max], and the loss is the
Rayleigh quotient plus a soft normalization monitor.

This is a PINN/surrogate smoke test only.  Deterministic finite-difference /
Sturm certificates remain the source of truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "pinn" / "runs"
TRUE_SINGLE_TRACK = ROOT / "output" / "true_single_track" / "true_results.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=float, default=12.0)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--m0", type=float, default=1.0)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=14.0)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--n-quad", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--plot-grid", type=int, default=800)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def load_reference(D: float):
    if not TRUE_SINGLE_TRACK.exists():
        return None
    data = json.loads(TRUE_SINGLE_TRACK.read_text())
    Ds = data.get("D", [])
    if D not in Ds:
        return None
    idx = Ds.index(D)
    omega = float(data["omega"][idx])
    return {
        "D": float(D),
        "omega_ref": omega,
        "lambda_ref": omega * omega,
        "E_ref": float(data["E_bound"][idx]),
        "S_ref": float(data["S_N"][idx]),
        "n_bound_ref": int(data["n_bound"][idx]),
        "source": str(TRUE_SINGLE_TRACK.relative_to(ROOT)),
    }


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    class TrialNet(nn.Module):
        def __init__(self):
            super().__init__()
            modules = [nn.Linear(1, args.hidden), nn.Tanh()]
            for _ in range(args.layers - 1):
                modules += [nn.Linear(args.hidden, args.hidden), nn.Tanh()]
            modules += [nn.Linear(args.hidden, 1)]
            self.net = nn.Sequential(*modules)

        def forward(self, z):
            x = z / args.z_max
            envelope = torch.clamp(1.0 - x * x, min=0.0)
            return envelope * self.net(x)

    def omega_axis(z):
        rp = torch.sqrt((z - args.D / 2.0) ** 2 + args.eps * args.eps)
        rm = torch.sqrt((z + args.D / 2.0) ** 2 + args.eps * args.eps)
        return 1.0 + args.a * (1.0 / rp + 1.0 / rm)

    def lap_omega_axis(z):
        rp2 = (z - args.D / 2.0) ** 2 + args.eps * args.eps
        rm2 = (z + args.D / 2.0) ** 2 + args.eps * args.eps
        return args.a * (-3.0 * args.eps * args.eps / (rp2 ** 2.5) - 3.0 * args.eps * args.eps / (rm2 ** 2.5))

    def veff_axis(z):
        om = omega_axis(z)
        return args.m0 * args.m0 * om * om + (1.0 - 6.0 * args.xi) * lap_omega_axis(z) / om

    def u_axis(z):
        return veff_axis(z) - args.m0 * args.m0

    def quadrature_grid(n):
        z = torch.linspace(-args.z_max, args.z_max, n, device=device, dtype=dtype).reshape(-1, 1)
        dz = 2.0 * args.z_max / (n - 1)
        weights = torch.ones_like(z) * dz
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return z, weights

    def rayleigh(model, z, w):
        z = z.detach().clone().requires_grad_(True)
        psi = model(z)
        dpsi = torch.autograd.grad(psi, z, torch.ones_like(psi), create_graph=True)[0]
        u = u_axis(z)
        norm = torch.sum(w * psi * psi)
        kinetic = torch.sum(w * dpsi * dpsi)
        potential = torch.sum(w * u * psi * psi)
        energy = (kinetic + potential) / torch.clamp(norm, min=1.0e-12)
        return energy, norm, kinetic, potential

    def residual_metrics(model, z, w, energy_value):
        z = z.detach().clone().requires_grad_(True)
        psi = model(z)
        dpsi = torch.autograd.grad(psi, z, torch.ones_like(psi), create_graph=True)[0]
        ddpsi = torch.autograd.grad(dpsi, z, torch.ones_like(dpsi), create_graph=True)[0]
        res = -ddpsi + u_axis(z) * psi - energy_value * psi
        norm = torch.sqrt(torch.sum(w * psi * psi))
        weighted_abs = torch.abs(res) / torch.clamp(norm, min=1.0e-12)
        return {
            "residual_l2_over_norm": float(torch.sqrt(torch.sum(w * res * res)) / torch.clamp(norm, min=1.0e-12)),
            "residual_median_abs_over_norm": float(torch.median(weighted_abs.detach().flatten()).cpu()),
            "psi_max_abs": float(torch.max(torch.abs(psi.detach())).cpu()),
        }

    model = TrialNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    z_quad, w_quad = quadrature_grid(args.n_quad)
    history = []

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        energy, norm, kinetic, potential = rayleigh(model, z_quad, w_quad)
        # Rayleigh quotient is scale invariant; this term prevents pathological tiny norms.
        loss_norm = (torch.log(torch.clamp(norm, min=1.0e-12)) ** 2) * 1.0e-4
        loss = energy + loss_norm
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "E": float(energy.detach().cpu()),
                "omega": float(torch.sqrt(torch.clamp(energy.detach() + args.m0 * args.m0, min=0.0)).cpu()),
                "norm": float(norm.detach().cpu()),
                "kinetic": float(kinetic.detach().cpu()),
                "potential": float(potential.detach().cpu()),
                "loss_norm": float(loss_norm.detach().cpu()),
            }
            history.append(row)
            print(json.dumps(row))

    energy, norm, kinetic, potential = rayleigh(model, z_quad, w_quad)
    energy_value = float(energy.detach().cpu())
    reference = load_reference(args.D)
    omega_pinn = math.sqrt(max(energy_value + args.m0 * args.m0, 0.0))
    metrics = {
        "target": "1D axial single-track Ritz PINN",
        "operator": "H_U = -d_z^2 + U, U = V_eff - m0^2",
        "D": float(args.D),
        "z_max": float(args.z_max),
        "n_quad": int(args.n_quad),
        "E_pinn": energy_value,
        "omega_pinn": omega_pinn,
        "lambda_from_E_pinn": energy_value + args.m0 * args.m0,
        "norm": float(norm.detach().cpu()),
        "kinetic": float(kinetic.detach().cpu()),
        "potential": float(potential.detach().cpu()),
        "device": str(device),
        "reference": reference,
    }
    if reference is not None:
        metrics.update(
            {
                "lambda_abs_error_ref": abs((energy_value + args.m0 * args.m0) - reference["lambda_ref"]),
                "omega_abs_error_ref": abs(omega_pinn - reference["omega_ref"]),
                "E_abs_error_ref": abs(energy_value - reference["E_ref"]),
            }
        )
    metrics.update(residual_metrics(model, z_quad, w_quad, energy.detach()))

    run_name = args.run_name or datetime.now().strftime("ritz_1d_single_track_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (run_dir / "history.csv").open("w", newline="") as f:
        fields = ["step", "E", "omega", "norm", "kinetic", "potential", "loss_norm"]
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

        z_plot = torch.linspace(-args.z_max, args.z_max, args.plot_grid, device=device, dtype=dtype).reshape(-1, 1)
        with torch.no_grad():
            psi = model(z_plot).detach().cpu().numpy().flatten()
            u = u_axis(z_plot).detach().cpu().numpy().flatten()
        z_np = z_plot.detach().cpu().numpy().flatten()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=180, constrained_layout=True)
        axes[0].plot(z_np, u, color="#1b4d89", lw=1.5)
        axes[0].axhline(energy_value, color="#d05a28", lw=1.2, ls="--", label=rf"$E_{{PINN}}={energy_value:.5f}$")
        if reference is not None:
            axes[0].axhline(reference["E_ref"], color="#2f8f46", lw=1.2, ls=":", label=rf"$E_{{ref}}={reference['E_ref']:.5f}$")
        axes[0].set_xlabel("z")
        axes[0].set_ylabel(r"$U(0,z)=V_{eff}-m_0^2$")
        axes[0].set_title("Axial shifted potential")
        axes[0].legend(fontsize=8)
        axes[1].plot(z_np, psi / (np.max(np.abs(psi)) + 1.0e-12), color="#44216b", lw=1.5)
        axes[1].set_xlabel("z")
        axes[1].set_ylabel("normalized trial psi")
        axes[1].set_title("Ritz-PINN ground-state trial")
        fig.savefig(run_dir / "diagnostics.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train/check an analytic-singular-split PINN for the two-center Poisson factor.

V0.2 target:
    Omega = Omega_sing + delta_theta,

where Omega_sing is the analytic Plummer Green-function solution of the exact
V0 two-center Poisson problem.  Therefore delta_theta should solve the
homogeneous Laplace equation with zero boundary data.  For the exact V0 test,
the mathematically correct correction is delta_theta == 0.

The trainable coordinate is s=rho^2, so

    Delta delta = 4 s delta_ss + 4 delta_s + delta_zz.

Generated artifacts are written only to pinn/runs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "pinn" / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=float, default=12.0)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--rho-max", type=float, default=8.0)
    parser.add_argument("--z-max", type=float, default=14.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-interior", type=int, default=2048)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-boundary", type=float, default=20.0)
    parser.add_argument("--w-data", type=float, default=5.0)
    parser.add_argument("--source-sample-frac", type=float, default=0.35)
    parser.add_argument("--core-radius", type=float, default=None)
    parser.add_argument("--random-final", action="store_true", help="Do not zero-initialize the final correction layer.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--plot-grid", type=int, default=120)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def safe_stat(prefix, values, out):
    if values.numel() == 0:
        out[f"{prefix}_count"] = 0
        out[f"{prefix}_rmse"] = None
        out[f"{prefix}_max_abs"] = None
        out[f"{prefix}_median_abs"] = None
        return
    out[f"{prefix}_count"] = int(values.numel())
    out[f"{prefix}_rmse"] = float((values * values).mean().sqrt().detach().cpu())
    out[f"{prefix}_max_abs"] = float(values.abs().max().detach().cpu())
    out[f"{prefix}_median_abs"] = float(values.abs().median().detach().cpu())


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
    s_max = args.rho_max * args.rho_max
    core_radius = args.core_radius if args.core_radius is not None else 3.0 * args.eps

    class CorrectionMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            modules = [nn.Linear(2, args.hidden), nn.Tanh()]
            for _ in range(args.layers - 1):
                modules += [nn.Linear(args.hidden, args.hidden), nn.Tanh()]
            modules += [nn.Linear(args.hidden, 1)]
            self.net = nn.Sequential(*modules)
            if not args.random_final:
                final = self.net[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

        def forward(self, s_z):
            s = s_z[:, :1]
            z = s_z[:, 1:2]
            coords = torch.cat([2.0 * s / s_max - 1.0, z / args.z_max], dim=1)
            return self.net(coords)

    def omega_sing_s(s, z):
        r_plus = torch.sqrt(torch.clamp(s, min=0.0) + (z - args.D / 2.0) ** 2 + args.eps * args.eps)
        r_minus = torch.sqrt(torch.clamp(s, min=0.0) + (z + args.D / 2.0) ** 2 + args.eps * args.eps)
        return 1.0 + args.a * (1.0 / r_plus + 1.0 / r_minus)

    def omega_model(model, s, z):
        return omega_sing_s(s, z) + model(torch.cat([s, z], dim=1))

    def core_mask(s, z):
        d_plus = torch.sqrt(s + (z - args.D / 2.0) ** 2)
        d_minus = torch.sqrt(s + (z + args.D / 2.0) ** 2)
        return torch.minimum(d_plus, d_minus) <= core_radius

    def sample_uniform(n):
        s = s_max * torch.rand(n, 1, device=device, dtype=dtype)
        z = -args.z_max + 2.0 * args.z_max * torch.rand(n, 1, device=device, dtype=dtype)
        return s, z

    def sample_source_biased(n):
        if n <= 0:
            return sample_uniform(0)
        half = n // 2
        counts = [half, n - half]
        centers = [-args.D / 2.0, args.D / 2.0]
        s_parts = []
        z_parts = []
        for count, center in zip(counts, centers):
            radial = torch.distributions.Exponential(rate=1.0 / max(args.eps, 1.0e-6)).sample((count, 1)).to(device=device, dtype=dtype)
            s = torch.clamp(radial * radial, max=s_max)
            z = center + args.eps * torch.randn(count, 1, device=device, dtype=dtype)
            z = torch.clamp(z, min=-args.z_max, max=args.z_max)
            s_parts.append(s)
            z_parts.append(z)
        return torch.cat(s_parts, dim=0), torch.cat(z_parts, dim=0)

    def sample_interior(n):
        n_biased = int(round(n * max(0.0, min(1.0, args.source_sample_frac))))
        n_uniform = n - n_biased
        s_u, z_u = sample_uniform(n_uniform)
        s_b, z_b = sample_source_biased(n_biased)
        return torch.cat([s_u, s_b], dim=0), torch.cat([z_u, z_b], dim=0)

    def sample_boundary(n):
        n_each = max(1, n // 4)
        z_line = -args.z_max + 2.0 * args.z_max * torch.rand(n_each, 1, device=device, dtype=dtype)
        s_line = s_max * torch.rand(n_each, 1, device=device, dtype=dtype)
        parts = [
            (s_max * torch.ones_like(z_line), z_line),
            (s_line, args.z_max * torch.ones_like(s_line)),
            (s_line, -args.z_max * torch.ones_like(s_line)),
            (torch.zeros_like(z_line), z_line),
        ]
        return torch.cat([p[0] for p in parts], dim=0), torch.cat([p[1] for p in parts], dim=0)

    def correction_laplace_residual(model, s, z):
        s = s.detach().clone().requires_grad_(True)
        z = z.detach().clone().requires_grad_(True)
        delta = model(torch.cat([s, z], dim=1))
        grad_s = torch.autograd.grad(delta, s, torch.ones_like(delta), create_graph=True)[0]
        grad_z = torch.autograd.grad(delta, z, torch.ones_like(delta), create_graph=True)[0]
        grad_ss = torch.autograd.grad(grad_s, s, torch.ones_like(grad_s), create_graph=True)[0]
        grad_zz = torch.autograd.grad(grad_z, z, torch.ones_like(grad_z), create_graph=True)[0]
        return 4.0 * s * grad_ss + 4.0 * grad_s + grad_zz

    @torch.no_grad()
    def evaluate_values(model, s_eval, z_eval, prefix):
        delta = model(torch.cat([s_eval, z_eval], dim=1)).flatten()
        full_err = (omega_model(model, s_eval, z_eval) - omega_sing_s(s_eval, z_eval)).flatten()
        exact = omega_sing_s(s_eval, z_eval).flatten()
        abs_delta = delta.abs()
        rel_full = full_err.abs() / torch.clamp(exact.abs(), min=1.0e-12)
        mask = core_mask(s_eval, z_eval).flatten()
        out = {
            f"{prefix}_delta_rmse": float(torch.sqrt(torch.mean(delta * delta)).cpu()),
            f"{prefix}_delta_max_abs": float(torch.max(abs_delta).cpu()),
            f"{prefix}_delta_median_abs": float(torch.median(abs_delta).cpu()),
            f"{prefix}_omega_median_rel": float(torch.median(rel_full).cpu()),
            f"{prefix}_omega_max_rel": float(torch.max(rel_full).cpu()),
            f"{prefix}_core_radius": float(core_radius),
        }
        safe_stat(f"{prefix}_delta_core_abs", delta[mask], out)
        safe_stat(f"{prefix}_delta_bulk_abs", delta[~mask], out)
        return out

    def evaluate_residual(model, s_eval, z_eval, prefix):
        res = correction_laplace_residual(model, s_eval, z_eval).detach().flatten()
        mask = core_mask(s_eval, z_eval).flatten()
        out = {
            f"{prefix}_laplace_residual_rmse": float(torch.sqrt(torch.mean(res * res)).cpu()),
            f"{prefix}_laplace_residual_max_abs": float(torch.max(torch.abs(res)).cpu()),
            f"{prefix}_laplace_residual_median_abs": float(torch.median(torch.abs(res)).cpu()),
            f"{prefix}_core_radius": float(core_radius),
        }
        safe_stat(f"{prefix}_laplace_residual_core_abs", res[mask], out)
        safe_stat(f"{prefix}_laplace_residual_bulk_abs", res[~mask], out)
        return out

    def make_plots(model, run_dir):
        cache_dir = run_dir / ".mplconfig"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n = args.plot_grid
        rho = torch.linspace(0.0, args.rho_max, n, device=device, dtype=dtype)
        z_grid = torch.linspace(-args.z_max, args.z_max, n, device=device, dtype=dtype)
        RHO, Z = torch.meshgrid(rho, z_grid, indexing="ij")
        S = RHO * RHO
        flat_s = S.reshape(-1, 1)
        flat_z = Z.reshape(-1, 1)
        delta_chunks = []
        residual_chunks = []
        omega_chunks = []
        for start in range(0, flat_s.shape[0], 2048):
            ss = flat_s[start : start + 2048]
            zz = flat_z[start : start + 2048]
            with torch.no_grad():
                delta_chunks.append(model(torch.cat([ss, zz], dim=1)).detach().cpu())
                omega_chunks.append(omega_sing_s(ss, zz).detach().cpu())
            residual_chunks.append(correction_laplace_residual(model, ss, zz).detach().cpu())
        delta = torch.cat(delta_chunks, dim=0).numpy().reshape(n, n)
        omega = torch.cat(omega_chunks, dim=0).numpy().reshape(n, n)
        residual = torch.cat(residual_chunks, dim=0).numpy().reshape(n, n)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=180, constrained_layout=True)
        extent = [-args.z_max, args.z_max, 0.0, args.rho_max]
        im0 = axes[0].imshow(omega, extent=extent, origin="lower", aspect="auto", cmap="viridis")
        axes[0].set_title(r"Analytic $\Omega_{\rm sing}$")
        scale_delta = max(float(np.percentile(np.abs(delta), 98)), 1.0e-12)
        im1 = axes[1].imshow(delta, extent=extent, origin="lower", aspect="auto", cmap="coolwarm", vmin=-scale_delta, vmax=scale_delta)
        axes[1].set_title(r"Learned correction $\delta_\theta$")
        scale_res = max(float(np.percentile(np.abs(residual), 98)), 1.0e-12)
        im2 = axes[2].imshow(residual, extent=extent, origin="lower", aspect="auto", cmap="coolwarm", vmin=-scale_res, vmax=scale_res)
        axes[2].set_title(r"Homogeneous residual $\Delta\delta_\theta$")
        for ax in axes:
            ax.set_xlabel("z")
            ax.set_ylabel("rho")
            ax.axvline(-args.D / 2.0, color="white", lw=0.8, ls="--", alpha=0.8)
            ax.axvline(args.D / 2.0, color="white", lw=0.8, ls="--", alpha=0.8)
        fig.colorbar(im0, ax=axes[0], shrink=0.85)
        fig.colorbar(im1, ax=axes[1], shrink=0.85)
        fig.colorbar(im2, ax=axes[2], shrink=0.85)
        fig.savefig(run_dir / "diagnostics.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    model = CorrectionMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        s_i, z_i = sample_interior(args.n_interior)
        res = correction_laplace_residual(model, s_i, z_i)
        loss_pde = torch.mean(res * res)

        s_b, z_b = sample_boundary(args.n_boundary)
        delta_b = model(torch.cat([s_b, z_b], dim=1))
        loss_boundary = torch.mean(delta_b * delta_b)

        s_d, z_d = sample_interior(args.n_interior)
        delta_d = model(torch.cat([s_d, z_d], dim=1))
        loss_data = torch.mean(delta_d * delta_d)

        loss = args.w_pde * loss_pde + args.w_boundary * loss_boundary + args.w_data * loss_data
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_pde": float(loss_pde.detach().cpu()),
                "loss_boundary": float(loss_boundary.detach().cpu()),
                "loss_data": float(loss_data.detach().cpu()),
            }
            history.append(row)
            print(json.dumps(row))

    s_holdout, z_holdout = sample_uniform(args.n_eval)
    s_trainlike, z_trainlike = sample_interior(args.n_eval)
    metrics = {
        "split": "Omega = Omega_sing + delta_theta",
        "expected_delta": 0.0,
        "zero_initialized_final_layer": not args.random_final,
        "core_radius": float(core_radius),
        "device": str(device),
    }
    metrics.update(evaluate_values(model, s_holdout, z_holdout, "uniform"))
    metrics.update(evaluate_residual(model, s_holdout, z_holdout, "uniform"))
    metrics.update(evaluate_values(model, s_trainlike, z_trainlike, "trainlike"))
    metrics.update(evaluate_residual(model, s_trainlike, z_trainlike, "trainlike"))

    run_name = args.run_name or datetime.now().strftime("two_center_poisson_split_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (run_dir / "history.csv").open("w", newline="") as f:
        fields = ["step", "loss", "loss_pde", "loss_boundary", "loss_data"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    if args.make_plots:
        make_plots(model, run_dir)
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

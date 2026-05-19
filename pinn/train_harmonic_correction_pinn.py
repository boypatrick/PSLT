#!/usr/bin/env python3
"""Train a V0.3 split PINN on a known harmonic boundary deformation.

The analytic singular split removes the Plummer source:

    Omega = Omega_sing + delta.

V0.3 uses a nonzero but exactly harmonic correction

    delta_*(s,z) = A (2 z^2 - s) / L^2,

with s=rho^2 and L_s = 4 s d_s^2 + 4 d_s + d_z^2.  Since

    L_s(2 z^2 - s) = -4 + 4 = 0,

this is a controlled boundary/domain-deformation test: the boundary data are
nonzero, but the correction remains source-free.  Outputs are written only to
pinn/runs/.
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
    parser.add_argument("--deformation-amp", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--n-interior", type=int, default=2048)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--w-pde", type=float, default=1.0)
    parser.add_argument("--w-boundary", type=float, default=50.0)
    parser.add_argument("--w-data", type=float, default=20.0)
    parser.add_argument("--source-sample-frac", type=float, default=0.25)
    parser.add_argument("--core-radius", type=float, default=None)
    parser.add_argument("--basis-head", action="store_true", help="Add a trainable coefficient times (2 z^2 - s)/L^2.")
    parser.add_argument("--basis-only", action="store_true", help="Use only the harmonic basis coefficient, without a residual MLP.")
    parser.add_argument("--basis-init", type=float, default=0.0)
    parser.add_argument("--w-orth", type=float, default=0.0, help="Penalty for residual-MLP projection onto the harmonic basis.")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--plot-grid", type=int, default=100)
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
    l2 = args.z_max * args.z_max + s_max
    core_radius = args.core_radius if args.core_radius is not None else 3.0 * args.eps

    def harmonic_basis_s(s, z):
        return (2.0 * z * z - s) / l2

    class CorrectionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.coeff = None
            if args.basis_head or args.basis_only:
                self.coeff = nn.Parameter(torch.tensor(float(args.basis_init), device=device, dtype=dtype))
            self.net = None
            if not args.basis_only:
                modules = [nn.Linear(2, args.hidden), nn.Tanh()]
                for _ in range(args.layers - 1):
                    modules += [nn.Linear(args.hidden, args.hidden), nn.Tanh()]
                modules += [nn.Linear(args.hidden, 1)]
                self.net = nn.Sequential(*modules)
                final = self.net[-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

        def forward(self, s_z):
            s = s_z[:, :1]
            z = s_z[:, 1:2]
            out = torch.zeros_like(s)
            if self.coeff is not None:
                out = out + self.coeff * harmonic_basis_s(s, z)
            out = out + self.residual_component(s_z)
            return out

        def residual_component(self, s_z):
            s = s_z[:, :1]
            z = s_z[:, 1:2]
            if self.net is None:
                return torch.zeros_like(s)
            coords = torch.cat([2.0 * s / s_max - 1.0, z / args.z_max], dim=1)
            return self.net(coords)

    def omega_sing_s(s, z):
        r_plus = torch.sqrt(torch.clamp(s, min=0.0) + (z - args.D / 2.0) ** 2 + args.eps * args.eps)
        r_minus = torch.sqrt(torch.clamp(s, min=0.0) + (z + args.D / 2.0) ** 2 + args.eps * args.eps)
        return 1.0 + args.a * (1.0 / r_plus + 1.0 / r_minus)

    def delta_exact_s(s, z):
        return args.deformation_amp * harmonic_basis_s(s, z)

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
        grad_s = torch.autograd.grad(delta, s, torch.ones_like(delta), create_graph=True, allow_unused=True)[0]
        grad_z = torch.autograd.grad(delta, z, torch.ones_like(delta), create_graph=True, allow_unused=True)[0]
        if grad_s is None:
            grad_s = torch.zeros_like(s)
        if grad_z is None:
            grad_z = torch.zeros_like(z)
        if grad_s.requires_grad:
            grad_ss = torch.autograd.grad(grad_s, s, torch.ones_like(grad_s), create_graph=True, allow_unused=True)[0]
            if grad_ss is None:
                grad_ss = torch.zeros_like(s)
        else:
            grad_ss = torch.zeros_like(s)
        if grad_z.requires_grad:
            grad_zz = torch.autograd.grad(grad_z, z, torch.ones_like(grad_z), create_graph=True, allow_unused=True)[0]
            if grad_zz is None:
                grad_zz = torch.zeros_like(z)
        else:
            grad_zz = torch.zeros_like(z)
        return 4.0 * s * grad_ss + 4.0 * grad_s + grad_zz

    def exact_laplace_residual(s, z):
        # For delta_* = A(2 z^2 - s)/L^2, L_s delta_* = -4A/L^2 + 4A/L^2 = 0.
        return torch.zeros_like(s)

    def residual_basis_projection(model, s, z):
        residual = model.residual_component(torch.cat([s, z], dim=1))
        basis = harmonic_basis_s(s, z)
        denom = torch.mean(basis * basis) + 1.0e-12
        coeff = torch.mean(residual * basis) / denom
        loss_orth = coeff * coeff * denom
        return coeff, loss_orth

    @torch.no_grad()
    def evaluate_values(model, s_eval, z_eval, prefix):
        pred = model(torch.cat([s_eval, z_eval], dim=1)).flatten()
        exact = delta_exact_s(s_eval, z_eval).flatten()
        err = pred - exact
        abs_err = err.abs()
        denom = torch.clamp(exact.abs(), min=max(args.deformation_amp * 1.0e-3, 1.0e-8))
        rel_err = abs_err / denom
        omega_exact = omega_sing_s(s_eval, z_eval).flatten() + exact
        omega_pred = omega_sing_s(s_eval, z_eval).flatten() + pred
        omega_rel = (omega_pred - omega_exact).abs() / torch.clamp(omega_exact.abs(), min=1.0e-12)
        mask = core_mask(s_eval, z_eval).flatten()
        out = {
            f"{prefix}_delta_rmse": float(torch.sqrt(torch.mean(err * err)).cpu()),
            f"{prefix}_delta_max_abs": float(torch.max(abs_err).cpu()),
            f"{prefix}_delta_median_abs": float(torch.median(abs_err).cpu()),
            f"{prefix}_delta_median_rel": float(torch.median(rel_err).cpu()),
            f"{prefix}_omega_median_rel": float(torch.median(omega_rel).cpu()),
            f"{prefix}_omega_max_rel": float(torch.max(omega_rel).cpu()),
            f"{prefix}_exact_delta_max_abs": float(torch.max(exact.abs()).cpu()),
            f"{prefix}_core_radius": float(core_radius),
        }
        safe_stat(f"{prefix}_delta_core_abs", err[mask], out)
        safe_stat(f"{prefix}_delta_bulk_abs", err[~mask], out)
        residual_coeff, _ = residual_basis_projection(model, s_eval, z_eval)
        out[f"{prefix}_residual_basis_coeff"] = float(residual_coeff.detach().cpu())
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
        pred_chunks = []
        exact_chunks = []
        residual_chunks = []
        for start in range(0, flat_s.shape[0], 2048):
            ss = flat_s[start : start + 2048]
            zz = flat_z[start : start + 2048]
            with torch.no_grad():
                pred_chunks.append(model(torch.cat([ss, zz], dim=1)).detach().cpu())
                exact_chunks.append(delta_exact_s(ss, zz).detach().cpu())
            residual_chunks.append(correction_laplace_residual(model, ss, zz).detach().cpu())
        pred = torch.cat(pred_chunks, dim=0).numpy().reshape(n, n)
        exact = torch.cat(exact_chunks, dim=0).numpy().reshape(n, n)
        residual = torch.cat(residual_chunks, dim=0).numpy().reshape(n, n)
        err = pred - exact
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=180, constrained_layout=True)
        extent = [-args.z_max, args.z_max, 0.0, args.rho_max]
        scale_exact = max(float(np.percentile(np.abs(exact), 98)), 1.0e-12)
        im0 = axes[0].imshow(exact, extent=extent, origin="lower", aspect="auto", cmap="coolwarm", vmin=-scale_exact, vmax=scale_exact)
        axes[0].set_title(r"Exact harmonic $\delta_*$")
        scale_err = max(float(np.percentile(np.abs(err), 98)), 1.0e-12)
        im1 = axes[1].imshow(err, extent=extent, origin="lower", aspect="auto", cmap="coolwarm", vmin=-scale_err, vmax=scale_err)
        axes[1].set_title(r"PINN error $\delta_\theta-\delta_*$")
        scale_res = max(float(np.percentile(np.abs(residual), 98)), 1.0e-12)
        im2 = axes[2].imshow(residual, extent=extent, origin="lower", aspect="auto", cmap="coolwarm", vmin=-scale_res, vmax=scale_res)
        axes[2].set_title(r"Residual $\Delta\delta_\theta$")
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

    model = CorrectionModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        s_i, z_i = sample_interior(args.n_interior)
        res = correction_laplace_residual(model, s_i, z_i)
        loss_pde = torch.mean(res * res)
        _, loss_orth = residual_basis_projection(model, s_i, z_i)

        s_b, z_b = sample_boundary(args.n_boundary)
        delta_b = model(torch.cat([s_b, z_b], dim=1))
        target_b = delta_exact_s(s_b, z_b)
        loss_boundary = torch.mean((delta_b - target_b) ** 2)

        s_d, z_d = sample_interior(args.n_interior)
        delta_d = model(torch.cat([s_d, z_d], dim=1))
        target_d = delta_exact_s(s_d, z_d)
        loss_data = torch.mean((delta_d - target_d) ** 2)

        loss = (
            args.w_pde * loss_pde
            + args.w_boundary * loss_boundary
            + args.w_data * loss_data
            + args.w_orth * loss_orth
        )
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_pde": float(loss_pde.detach().cpu()),
                "loss_boundary": float(loss_boundary.detach().cpu()),
                "loss_data": float(loss_data.detach().cpu()),
                "loss_orth": float(loss_orth.detach().cpu()),
            }
            history.append(row)
            print(json.dumps(row))

    s_holdout, z_holdout = sample_uniform(args.n_eval)
    s_trainlike, z_trainlike = sample_interior(args.n_eval)
    exact_res = exact_laplace_residual(s_holdout, z_holdout).detach().flatten()
    metrics = {
        "split": "Omega = Omega_sing + delta_theta",
        "target_delta": "A * (2 z^2 - s) / L^2",
        "deformation_amp": float(args.deformation_amp),
        "basis_head": bool(args.basis_head or args.basis_only),
        "basis_only": bool(args.basis_only),
        "basis_init": float(args.basis_init),
        "w_orth": float(args.w_orth),
        "learned_basis_coeff": float(model.coeff.detach().cpu()) if model.coeff is not None else None,
        "basis_coeff_error": float((model.coeff.detach().cpu() - args.deformation_amp).abs()) if model.coeff is not None else None,
        "L2": float(l2),
        "exact_delta_laplace_residual_max_abs": float(exact_res.abs().max().cpu()),
        "exact_delta_laplace_residual_median_abs": float(exact_res.abs().median().cpu()),
        "core_radius": float(core_radius),
        "device": str(device),
    }
    metrics.update(evaluate_values(model, s_holdout, z_holdout, "uniform"))
    metrics.update(evaluate_residual(model, s_holdout, z_holdout, "uniform"))
    metrics.update(evaluate_values(model, s_trainlike, z_trainlike, "trainlike"))
    metrics.update(evaluate_residual(model, s_trainlike, z_trainlike, "trainlike"))

    run_name = args.run_name or datetime.now().strftime("harmonic_correction_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (run_dir / "history.csv").open("w", newline="") as f:
        fields = ["step", "loss", "loss_pde", "loss_boundary", "loss_data", "loss_orth"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    if args.make_plots:
        make_plots(model, run_dir)
    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

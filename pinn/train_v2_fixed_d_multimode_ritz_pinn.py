#!/usr/bin/env python3
"""V2 fixed-D multi-mode self-adjoint Ritz-PINN.

V1 only tracks the lowest self-adjoint branch.  V2 starts with the safer
fixed-D multi-mode problem before any parametric-D generalization.  The model
learns a K-dimensional trial subspace for

    H_U u = -4s u_ss - 4u_s - u_zz + U(s,z;D)u,

with the cylindrical measure written in s=rho^2 coordinates.  On each
quadrature grid we assemble the projected matrices

    G_ij = <u_i,u_j>,
    H_ij = <u_i,H_U u_j>,

using the symmetric energy form

    H_ij = int (4s u_{i,s}u_{j,s} + u_{i,z}u_{j,z} + U u_i u_j) ds dz.

The reported eigenvalues are the small generalized Ritz values

    H c = E G c.

This is a sandbox emulator.  The deterministic finite-volume generalized
eigenproblem remains the reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from accelerator_utils import select_torch_device


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "pinn" / "runs"


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
    p.add_argument("--n-s", type=int, default=36)
    p.add_argument("--n-z", type=int, default=96)
    p.add_argument("--modes", type=int, default=3)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--hidden", type=int, default=80)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--residual-scale", type=float, default=0.20)
    p.add_argument("--base-mode", choices=["polynomial", "two-lobe"], default="polynomial")
    p.add_argument("--parity-heads", action="store_true", help="Project network heads to even/odd/even parity in z.")
    p.add_argument("--rho-width", type=float, default=0.8)
    p.add_argument("--z-width", type=float, default=1.2)
    p.add_argument("--w-reference", type=float, default=50.0)
    p.add_argument("--w-orth", type=float, default=2.0)
    p.add_argument("--w-norm", type=float, default=1.0e-4)
    p.add_argument("--gram-ridge", type=float, default=1.0e-7)
    p.add_argument("--no-select-best", action="store_true", help="Use the final step instead of the best training checkpoint.")
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--n-residual", type=int, default=768)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_reference(D: float, modes: int):
    d_label = format_d_label(D)
    candidates = [
        RUNS_DIR / f"v0p5p1_selfadjoint_D{d_label}_n50x500" / "metrics.json",
        RUNS_DIR / f"v1_selfadjoint_D{d_label}_n50x500_k4" / "metrics.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        values = [float(x) for x in data.get("E_selfadjoint", [])[:modes]]
        if len(values) >= modes:
            return {
                "source": str(path.relative_to(ROOT)),
                "E_selfadjoint": values,
                "omega_selfadjoint": [float(x) for x in data.get("omega_selfadjoint", [])[:modes]],
                "K_asym_rel": float(data.get("K_asym_rel", math.nan)),
                "n_negative_selfadjoint": int(data.get("n_negative_selfadjoint", -1)),
            }
    raise FileNotFoundError(f"No self-adjoint reference with {modes} modes for D={D:g}")


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed in this Python environment.") from exc

    if args.modes < 1:
        raise SystemExit("--modes must be positive")
    reference = load_reference(args.D, args.modes)
    torch.manual_seed(args.seed)
    try:
        device, accelerator = select_torch_device(torch, args.device)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    dtype = torch.float32
    s_max = args.L_rho * args.L_rho

    class MultiModeTrialNet(nn.Module):
        def __init__(self):
            super().__init__()
            modules = [nn.Linear(2, args.hidden), nn.Tanh()]
            for _ in range(args.layers - 1):
                modules += [nn.Linear(args.hidden, args.hidden), nn.Tanh()]
            modules += [nn.Linear(args.hidden, args.modes)]
            self.net = nn.Sequential(*modules)
            self.base_scale = nn.Parameter(torch.ones(args.modes, dtype=dtype))

        def raw_net(self, s, z):
            x = 2.0 * s / s_max - 1.0
            y = z / args.L_z
            return self.net(torch.cat([x, y], dim=1))

        def parity_project(self, s, z):
            raw = self.raw_net(s, z)
            if not args.parity_heads:
                return raw
            raw_reflect = self.raw_net(s, -z)
            cols = []
            for k in range(args.modes):
                if k % 2 == 0:
                    cols.append(0.5 * (raw[:, k : k + 1] + raw_reflect[:, k : k + 1]))
                else:
                    cols.append(0.5 * (raw[:, k : k + 1] - raw_reflect[:, k : k + 1]))
            return torch.cat(cols, dim=1)

        def base_shapes(self, s, z):
            y = z / args.L_z
            if args.base_mode == "polynomial":
                bases = [torch.ones_like(z)]
                if args.modes >= 2:
                    bases.append(y)
                if args.modes >= 3:
                    bases.append(y * y - torch.mean(y * y))
                for k in range(3, args.modes):
                    bases.append(torch.cos(float(k) * math.pi * y / 2.0))
                return torch.cat(bases[: args.modes], dim=1)
            rho_part = torch.exp(-s / max(args.rho_width * args.rho_width, 1.0e-12))
            z_width2 = max(args.z_width * args.z_width, 1.0e-12)
            left = torch.exp(-((z + args.D / 2.0) ** 2) / z_width2)
            right = torch.exp(-((z - args.D / 2.0) ** 2) / z_width2)
            bases = []
            bases.append(rho_part * (left + right))
            if args.modes >= 2:
                bases.append(rho_part * (right - left))
            if args.modes >= 3:
                # A local radial/axial excitation seed.  The constant keeps it
                # linearly independent from the lowest even two-lobe seed.
                bases.append(rho_part * (left + right) * ((z / max(args.D, 1.0e-12)) ** 2 + s / s_max - 0.35))
            for k in range(3, args.modes):
                bases.append(torch.cos(float(k) * math.pi * y / 2.0))
            return torch.cat(bases[: args.modes], dim=1)

        def forward(self, s, z):
            y = z / args.L_z
            envelope = torch.clamp(1.0 - s / s_max, min=0.0) * torch.clamp(1.0 - y * y, min=0.0)
            mlp = self.parity_project(s, z)
            base = self.base_shapes(s, z)
            return envelope * (self.base_scale.reshape(1, -1) * base + args.residual_scale * mlp)

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

    def quadrature_grid(requires_grad: bool = True):
        s1 = (torch.arange(args.n_s, device=device, dtype=dtype) + 0.5) * (s_max / args.n_s)
        z1 = -args.L_z + (torch.arange(args.n_z, device=device, dtype=dtype) + 0.5) * (2.0 * args.L_z / args.n_z)
        S, Z = torch.meshgrid(s1, z1, indexing="ij")
        s = S.reshape(-1, 1).detach().clone().requires_grad_(requires_grad)
        z = Z.reshape(-1, 1).detach().clone().requires_grad_(requires_grad)
        return s, z

    def projected_matrices(model, s, z):
        B = model(s, z)
        U = U_sz(s, z)
        grads_s = []
        grads_z = []
        for k in range(args.modes):
            du_ds, du_dz = torch.autograd.grad(B[:, k : k + 1], (s, z), torch.ones_like(B[:, k : k + 1]), create_graph=True)
            grads_s.append(du_ds)
            grads_z.append(du_dz)
        G = torch.empty((args.modes, args.modes), device=device, dtype=dtype)
        H = torch.empty((args.modes, args.modes), device=device, dtype=dtype)
        for i in range(args.modes):
            ui = B[:, i : i + 1]
            for j in range(args.modes):
                uj = B[:, j : j + 1]
                G[i, j] = torch.mean(ui * uj)
                H[i, j] = torch.mean(
                    4.0 * s * grads_s[i] * grads_s[j]
                    + grads_z[i] * grads_z[j]
                    + U * ui * uj
                )
        return 0.5 * (H + H.T), 0.5 * (G + G.T), B

    def ritz_eigh(H, G):
        ridge = args.gram_ridge * torch.eye(args.modes, device=device, dtype=dtype)
        L = torch.linalg.cholesky(G + ridge)
        y = torch.linalg.solve_triangular(L, H, upper=False)
        c = torch.linalg.solve_triangular(L, y.T, upper=False).T
        c = 0.5 * (c + c.T)
        evals, q = torch.linalg.eigh(c)
        coeff = torch.linalg.solve_triangular(L.T, q, upper=True)
        return evals, coeff

    def gram_losses(G):
        diag = torch.clamp(torch.diag(G), min=1.0e-12)
        denom = torch.sqrt(diag[:, None] * diag[None, :])
        corr = G / denom
        eye = torch.eye(args.modes, device=device, dtype=dtype)
        orth = torch.mean((corr - eye) ** 2)
        norm = torch.mean(torch.log(diag) ** 2)
        return orth, norm, corr

    def residual_metrics(model, evals, coeff):
        n_total = args.n_s * args.n_z
        n_res = min(args.n_residual, n_total)
        s_full, z_full = quadrature_grid(requires_grad=False)
        idx = torch.linspace(0, n_total - 1, n_res, device=device).round().long()
        s = s_full[idx].detach().clone().requires_grad_(True)
        z = z_full[idx].detach().clone().requires_grad_(True)
        B = model(s, z)
        HB_cols = []
        for k in range(args.modes):
            uk = B[:, k : k + 1]
            du_ds, du_dz = torch.autograd.grad(uk, (s, z), torch.ones_like(uk), create_graph=True)
            d2u_ds2 = torch.autograd.grad(du_ds, s, torch.ones_like(du_ds), create_graph=True)[0]
            d2u_dz2 = torch.autograd.grad(du_dz, z, torch.ones_like(du_dz), create_graph=True)[0]
            HB_cols.append(-4.0 * s * d2u_ds2 - 4.0 * du_ds - d2u_dz2 + U_sz(s, z) * uk)
        HB = torch.cat(HB_cols, dim=1)
        U_modes = B @ coeff
        HU_modes = HB @ coeff
        rows = []
        for k in range(args.modes):
            u = U_modes[:, k : k + 1]
            res = HU_modes[:, k : k + 1] - evals[k] * u
            rms = torch.sqrt(torch.mean(u * u))
            rows.append({
                "mode": k,
                "strong_residual_l2_over_rms_u": float((torch.sqrt(torch.mean(res * res)) / torch.clamp(rms, min=1.0e-12)).detach().cpu()),
                "strong_residual_median_abs_over_rms_u": float((torch.median(torch.abs(res)) / torch.clamp(rms, min=1.0e-12)).detach().cpu()),
                "mode_rms": float(rms.detach().cpu()),
            })
        return rows

    ref_e = torch.tensor(reference["E_selfadjoint"], device=device, dtype=dtype)
    model = MultiModeTrialNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    s_quad, z_quad = quadrature_grid(requires_grad=True)
    history = []
    best_state = None
    best_score = float("inf")
    best_step = 0

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        H, G, _ = projected_matrices(model, s_quad, z_quad)
        evals, _ = ritz_eigh(H, G)
        orth_loss, norm_loss, corr = gram_losses(G)
        scale = torch.clamp(torch.abs(ref_e), min=1.0)
        reference_loss = torch.mean(((evals[: args.modes] - ref_e) / scale) ** 2)
        rel_vec = torch.abs((evals[: args.modes] - ref_e) / scale)
        loss = torch.sum(evals[: args.modes]) + args.w_reference * reference_loss + args.w_orth * orth_loss + args.w_norm * norm_loss
        loss.backward()
        opt.step()
        max_offdiag = (corr - torch.eye(args.modes, device=device, dtype=dtype)).abs().masked_fill(
            torch.eye(args.modes, device=device, dtype=torch.bool),
            0.0,
        ).max()
        gate_score = float((torch.max(rel_vec) + max_offdiag + 0.1 * orth_loss).detach().cpu())
        if not args.no_select_best and gate_score < best_score:
            best_score = gate_score
            best_step = step
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "trace_E": float(torch.sum(evals[: args.modes]).detach().cpu()),
                "reference_loss": float(reference_loss.detach().cpu()),
                "orth_loss": float(orth_loss.detach().cpu()),
                "norm_loss": float(norm_loss.detach().cpu()),
                "max_abs_corr_offdiag": float(max_offdiag.detach().cpu()),
                "max_rel_error_train": float(torch.max(rel_vec).detach().cpu()),
                "gate_score": gate_score,
            }
            for k in range(args.modes):
                row[f"E{k}_ritz"] = float(evals[k].detach().cpu())
            history.append(row)
            print(json.dumps(row))

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    H, G, B = projected_matrices(model, s_quad, z_quad)
    evals, coeff = ritz_eigh(H, G)
    orth_loss, norm_loss, corr = gram_losses(G)
    residual_rows = residual_metrics(model, evals.detach(), coeff.detach())
    E_values = [float(x) for x in evals[: args.modes].detach().cpu()]
    rel_errors = [
        abs(E_values[k] - reference["E_selfadjoint"][k]) / max(abs(reference["E_selfadjoint"][k]), 1.0e-12)
        for k in range(args.modes)
    ]
    metrics = {
        "target": "V2 fixed-D multi-mode self-adjoint Ritz-PINN",
        "operator": "H_U=-4s d_s^2 -4 d_s -d_z^2 + U(s,z;D)",
        "D": float(args.D),
        "modes": int(args.modes),
        "L_rho": float(args.L_rho),
        "L_z": float(args.L_z),
        "n_s": int(args.n_s),
        "n_z": int(args.n_z),
        "steps": int(args.steps),
        "base_mode": args.base_mode,
        "parity_heads": bool(args.parity_heads),
        "rho_width": float(args.rho_width),
        "z_width": float(args.z_width),
        "E_ritz": E_values,
        "omega_ritz": [math.sqrt(max(x + args.m0 * args.m0, 0.0)) for x in E_values],
        "reference": reference,
        "E_rel_errors": rel_errors,
        "max_E_rel_error": max(rel_errors),
        "median_E_rel_error": sorted(rel_errors)[len(rel_errors) // 2],
        "gram_diag": [float(x) for x in torch.diag(G).detach().cpu()],
        "gram_corr": [[float(x) for x in row] for row in corr.detach().cpu()],
        "max_abs_corr_offdiag": float((corr - torch.eye(args.modes, device=device, dtype=dtype)).abs().masked_fill(torch.eye(args.modes, device=device, dtype=torch.bool), 0.0).max().detach().cpu()),
        "orth_loss": float(orth_loss.detach().cpu()),
        "norm_loss": float(norm_loss.detach().cpu()),
        "residual_metrics": residual_rows,
        "max_strong_residual_l2_over_rms_u": max(row["strong_residual_l2_over_rms_u"] for row in residual_rows),
        "device": str(device),
        "accelerator": accelerator,
        "seed": int(args.seed),
        "selected_checkpoint_step": int(best_step if best_state is not None else args.steps),
        "selected_checkpoint_score": float(best_score if best_state is not None else math.nan),
        "select_best": not args.no_select_best,
    }

    run_name = args.run_name or datetime.now().strftime("v2_fixed_d_multimode_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (run_dir / "history.csv").open("w", newline="") as f:
        fields = sorted({key for row in history for key in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "mode_residuals.csv").open("w", newline="") as f:
        fields = ["mode", "strong_residual_l2_over_rms_u", "strong_residual_median_abs_over_rms_u", "mode_rms"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(residual_rows)

    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

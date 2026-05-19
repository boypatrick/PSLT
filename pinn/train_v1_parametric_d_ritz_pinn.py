#!/usr/bin/env python3
"""V1 parametric-D self-adjoint Ritz-PINN.

This script is the first parametric-D experiment after the fixed-D V1 anchors.
It targets only the lowest self-adjoint branch E0(D).  It uses the same shifted
operator and Rayleigh functional as V0.5,

    H_U(D)u = -4s u_ss - 4u_s - u_zz + U(s,z;D)u,

    E_D[u] = int(4s*u_s^2 + u_z^2 + U*u^2) ds dz / int(u^2) ds dz.

Branch safety is enforced by a checkpoint coherence loss against the finite-
volume/PINN anchor table in pinn/v1_anchor_summary.csv.  This remains a sandbox
emulator; finite-volume self-adjoint references are the source of truth.
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
ANCHOR_SUMMARY = ROOT / "pinn" / "v1_anchor_summary.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchors", default=str(ANCHOR_SUMMARY))
    p.add_argument("--D-min", type=float, default=6.0)
    p.add_argument("--D-max", type=float, default=18.0)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--m0", type=float, default=1.0)
    p.add_argument("--xi", type=float, default=0.0)
    p.add_argument("--L-rho", type=float, default=4.0)
    p.add_argument("--L-z", type=float, default=20.0)
    p.add_argument("--n-s", type=int, default=40)
    p.add_argument("--n-z", type=int, default=104)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=80)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--w-checkpoint", type=float, default=200.0)
    p.add_argument("--w-smooth", type=float, default=1.0e-4)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--init-from-run", default=None, help="Load model.pt from a previous pinn/runs entry or explicit path.")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--n-residual", type=int, default=1024)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_anchors(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "D": float(row["D"]),
                "E_ref": float(row["E_selfadjoint_ref"]),
                "E_anchor": float(row["E_pinn"]),
                "source": row.get("reference_run", ""),
            })
    if not rows:
        raise ValueError(f"No anchors found in {path}")
    return rows


def resolve_model_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        return path
    run_model = RUNS_DIR / value / "model.pt"
    if run_model.is_file():
        return run_model
    raise FileNotFoundError(f"Could not find model checkpoint for --init-from-run={value!r}")


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed in this Python environment.") from exc

    anchors = load_anchors(args.anchors)
    torch.manual_seed(args.seed)
    try:
        device, accelerator = select_torch_device(torch, args.device)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    dtype = torch.float32
    s_max = args.L_rho * args.L_rho
    D_span = max(args.D_max - args.D_min, 1.0e-12)

    class ParametricTrialNet(nn.Module):
        def __init__(self):
            super().__init__()
            modules = [nn.Linear(3, args.hidden), nn.Tanh()]
            for _ in range(args.layers - 1):
                modules += [nn.Linear(args.hidden, args.hidden), nn.Tanh()]
            modules += [nn.Linear(args.hidden, 1)]
            self.net = nn.Sequential(*modules)
            self.base = nn.Parameter(torch.tensor(1.0, dtype=dtype))

        def forward(self, s, z, D):
            x = 2.0 * s / s_max - 1.0
            y = z / args.L_z
            d = 2.0 * (D - args.D_min) / D_span - 1.0
            envelope = torch.clamp(1.0 - s / s_max, min=0.0) * torch.clamp(1.0 - y * y, min=0.0)
            mlp = self.net(torch.cat([x, y, d], dim=1))
            return envelope * (self.base + args.residual_scale * mlp)

    def omega_szD(s, z, D):
        rp2 = s + (z - D / 2.0) ** 2 + args.eps * args.eps
        rm2 = s + (z + D / 2.0) ** 2 + args.eps * args.eps
        return 1.0 + args.a * (torch.rsqrt(rp2) + torch.rsqrt(rm2))

    def lap_omega_szD(s, z, D):
        rp2 = s + (z - D / 2.0) ** 2 + args.eps * args.eps
        rm2 = s + (z + D / 2.0) ** 2 + args.eps * args.eps
        return args.a * (-3.0 * args.eps * args.eps * rp2.pow(-2.5) - 3.0 * args.eps * args.eps * rm2.pow(-2.5))

    def U_szD(s, z, D):
        om = omega_szD(s, z, D)
        return args.m0 * args.m0 * (om * om - 1.0) + (1.0 - 6.0 * args.xi) * lap_omega_szD(s, z, D) / om

    def quadrature_grid(n_s: int, n_z: int, D_value: float, requires_grad: bool = True):
        s1 = (torch.arange(n_s, device=device, dtype=dtype) + 0.5) * (s_max / n_s)
        z1 = -args.L_z + (torch.arange(n_z, device=device, dtype=dtype) + 0.5) * (2.0 * args.L_z / n_z)
        S, Z = torch.meshgrid(s1, z1, indexing="ij")
        s = S.reshape(-1, 1).detach().clone().requires_grad_(requires_grad)
        z = Z.reshape(-1, 1).detach().clone().requires_grad_(requires_grad)
        D = torch.full_like(s, float(D_value), requires_grad=requires_grad)
        return s, z, D

    def rayleigh(model, D_value: float, create_d_graph: bool = False):
        s, z, D = quadrature_grid(args.n_s, args.n_z, D_value, requires_grad=True)
        u = model(s, z, D)
        du_ds, du_dz = torch.autograd.grad(u, (s, z), torch.ones_like(u), create_graph=True)
        U = U_szD(s, z, D)
        norm = torch.mean(u * u)
        kinetic_s = torch.mean(4.0 * s * du_ds * du_ds)
        kinetic_z = torch.mean(du_dz * du_dz)
        potential = torch.mean(U * u * u)
        energy = (kinetic_s + kinetic_z + potential) / torch.clamp(norm, min=1.0e-12)
        if not create_d_graph:
            return energy, norm, kinetic_s, kinetic_z, potential
        dE_dD = torch.autograd.grad(energy, D, create_graph=True, retain_graph=True, allow_unused=True)[0]
        smooth = torch.mean(dE_dD * dE_dD) if dE_dD is not None else torch.zeros((), device=device, dtype=dtype)
        return energy, norm, kinetic_s, kinetic_z, potential, smooth

    def residual_metrics(model, D_value: float, energy_value: float):
        n_total = args.n_s * args.n_z
        n_res = min(args.n_residual, n_total)
        s_full, z_full, D_full = quadrature_grid(args.n_s, args.n_z, D_value, requires_grad=False)
        idx = torch.linspace(0, n_total - 1, n_res, device=device).round().long()
        s = s_full[idx].detach().clone().requires_grad_(True)
        z = z_full[idx].detach().clone().requires_grad_(True)
        D = D_full[idx].detach().clone().requires_grad_(False)
        u = model(s, z, D)
        du_ds, du_dz = torch.autograd.grad(u, (s, z), torch.ones_like(u), create_graph=True)
        d2u_ds2 = torch.autograd.grad(du_ds, s, torch.ones_like(du_ds), create_graph=True)[0]
        d2u_dz2 = torch.autograd.grad(du_dz, z, torch.ones_like(du_dz), create_graph=True)[0]
        H_u = -4.0 * s * d2u_ds2 - 4.0 * du_ds - d2u_dz2 + U_szD(s, z, D) * u
        E = torch.tensor(energy_value, device=device, dtype=dtype)
        res = H_u - E * u
        norm = torch.sqrt(torch.mean(u * u))
        return {
            "strong_residual_l2_over_rms_u": float((torch.sqrt(torch.mean(res * res)) / torch.clamp(norm, min=1.0e-12)).detach().cpu()),
            "strong_residual_median_abs_over_rms_u": float((torch.median(torch.abs(res)) / torch.clamp(norm, min=1.0e-12)).detach().cpu()),
        }

    model = ParametricTrialNet().to(device)
    init_model_path = resolve_model_path(args.init_from_run)
    if init_model_path is not None:
        state = torch.load(init_model_path, map_location=device)
        model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    anchor_D = [a["D"] for a in anchors]
    anchor_E = {a["D"]: a["E_ref"] for a in anchors}
    history = []

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        energies = []
        ritz_terms = []
        chk_terms = []
        smooth_terms = []
        for D_value in anchor_D:
            energy, norm, kinetic_s, kinetic_z, potential, smooth = rayleigh(model, D_value, create_d_graph=args.w_smooth > 0.0)
            ref = torch.tensor(anchor_E[D_value], device=device, dtype=dtype)
            scale = max(1.0, abs(anchor_E[D_value]))
            ritz_terms.append(energy)
            chk_terms.append(((energy - ref) / scale).pow(2))
            smooth_terms.append(smooth)
            energies.append(energy)
        loss_ritz = torch.stack(ritz_terms).mean()
        loss_chk = torch.stack(chk_terms).mean()
        loss_smooth = torch.stack(smooth_terms).mean() if smooth_terms else torch.zeros((), device=device, dtype=dtype)
        loss = loss_ritz + args.w_checkpoint * loss_chk + args.w_smooth * loss_smooth
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_ritz": float(loss_ritz.detach().cpu()),
                "loss_checkpoint": float(loss_chk.detach().cpu()),
                "loss_smooth": float(loss_smooth.detach().cpu()),
                "max_rel_error": max(float(abs(e.detach().cpu()) - anchor_E[D]) / max(abs(anchor_E[D]), 1.0e-12) for e, D in zip(energies, anchor_D)),
                "base": float(model.base.detach().cpu()),
            }
            for e, D in zip(energies, anchor_D):
                row[f"E_D{D:g}"] = float(e.detach().cpu())
            history.append(row)
            print(json.dumps(row))

    checkpoint_metrics = []
    for D_value in anchor_D:
        energy, norm, kinetic_s, kinetic_z, potential = rayleigh(model, D_value, create_d_graph=False)
        E_value = float(energy.detach().cpu())
        ref = anchor_E[D_value]
        row = {
            "D": float(D_value),
            "E_ref": float(ref),
            "E_parametric": E_value,
            "E_abs_error": abs(E_value - ref),
            "E_rel_error": abs(E_value - ref) / max(abs(ref), 1.0e-12),
            "omega_parametric": math.sqrt(max(E_value + args.m0 * args.m0, 0.0)),
            "norm": float(norm.detach().cpu()),
            "kinetic_s": float(kinetic_s.detach().cpu()),
            "kinetic_z": float(kinetic_z.detach().cpu()),
            "potential": float(potential.detach().cpu()),
        }
        row.update(residual_metrics(model, D_value, E_value))
        checkpoint_metrics.append(row)

    max_rel = max(r["E_rel_error"] for r in checkpoint_metrics)
    median_rel = sorted(r["E_rel_error"] for r in checkpoint_metrics)[len(checkpoint_metrics) // 2]
    max_res = max(r["strong_residual_l2_over_rms_u"] for r in checkpoint_metrics)
    metrics = {
        "target": "V1 parametric-D lowest-branch self-adjoint Ritz-PINN",
        "anchors": anchors,
        "checkpoint_metrics": checkpoint_metrics,
        "max_rel_error": max_rel,
        "median_rel_error": median_rel,
        "max_strong_residual_l2_over_rms_u": max_res,
        "gate_max_rel_lt_5e_minus_4": max_rel < 5.0e-4,
        "gate_median_rel_lt_3e_minus_4": median_rel < 3.0e-4,
        "gate_max_residual_lt_5e_minus_2": max_res < 5.0e-2,
        "device": str(device),
        "accelerator": accelerator,
        "init_model_path": str(init_model_path.relative_to(ROOT)) if init_model_path else None,
        "config": vars(args),
    }

    run_name = args.run_name or datetime.now().strftime("v1_parametric_d_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (run_dir / "checkpoint_metrics.csv").open("w", newline="") as f:
        fields = list(checkpoint_metrics[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checkpoint_metrics)
    with (run_dir / "history.csv").open("w", newline="") as f:
        fields = sorted({k for row in history for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

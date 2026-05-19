#!/usr/bin/env python3
"""Evaluate a trained V1 parametric-D Ritz-PINN on dense D points.

This is a diagnostic evaluator, not a proof certificate.  It loads a trained
parametric model, evaluates the Rayleigh quotient and strong-form residual at a
chosen D grid, and flags monotonicity/residual anomalies that should be checked
by the self-adjoint finite-volume solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from accelerator_utils import select_torch_device


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "pinn" / "runs"
DEFAULT_RUN = "v1_parametric_D6_18_n32_z80_continue_4000"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=DEFAULT_RUN, help="Run directory name under pinn/runs or explicit path.")
    p.add_argument("--D-values", nargs="*", type=float, default=None)
    p.add_argument("--D-min", type=float, default=6.0)
    p.add_argument("--D-max", type=float, default=18.0)
    p.add_argument("--D-step", type=float, default=1.5)
    p.add_argument("--n-residual", type=int, default=1024)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path
    run_dir = RUNS_DIR / value
    if run_dir.is_dir():
        return run_dir
    raise FileNotFoundError(f"Cannot find run directory: {value}")


def dense_values(D_min: float, D_max: float, step: float):
    values = []
    x = D_min
    while x <= D_max + 1.0e-9:
        values.append(round(x, 10))
        x += step
    if abs(values[-1] - D_max) > 1.0e-9:
        values.append(D_max)
    return values


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed in this Python environment.") from exc

    run_dir = resolve_run_dir(args.run)
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    try:
        device, accelerator = select_torch_device(torch, args.device)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    dtype = torch.float32
    D_min = float(config.get("D_min", 6.0))
    D_max = float(config.get("D_max", 18.0))
    D_span = max(D_max - D_min, 1.0e-12)
    L_rho = float(config.get("L_rho", 4.0))
    L_z = float(config.get("L_z", 20.0))
    s_max = L_rho * L_rho
    n_s = int(config.get("n_s", 32))
    n_z = int(config.get("n_z", 80))
    hidden = int(config.get("hidden", 80))
    layers = int(config.get("layers", 4))
    residual_scale = float(config.get("residual_scale", 0.25))
    a = float(config.get("a", 1.0))
    eps = float(config.get("eps", 0.2))
    m0 = float(config.get("m0", 1.0))
    xi = float(config.get("xi", 0.0))

    class ParametricTrialNet(nn.Module):
        def __init__(self):
            super().__init__()
            modules = [nn.Linear(3, hidden), nn.Tanh()]
            for _ in range(layers - 1):
                modules += [nn.Linear(hidden, hidden), nn.Tanh()]
            modules += [nn.Linear(hidden, 1)]
            self.net = nn.Sequential(*modules)
            self.base = nn.Parameter(torch.tensor(1.0, dtype=dtype))

        def forward(self, s, z, D):
            x = 2.0 * s / s_max - 1.0
            y = z / L_z
            d = 2.0 * (D - D_min) / D_span - 1.0
            envelope = torch.clamp(1.0 - s / s_max, min=0.0) * torch.clamp(1.0 - y * y, min=0.0)
            mlp = self.net(torch.cat([x, y, d], dim=1))
            return envelope * (self.base + residual_scale * mlp)

    def omega_szD(s, z, D):
        rp2 = s + (z - D / 2.0) ** 2 + eps * eps
        rm2 = s + (z + D / 2.0) ** 2 + eps * eps
        return 1.0 + a * (torch.rsqrt(rp2) + torch.rsqrt(rm2))

    def lap_omega_szD(s, z, D):
        rp2 = s + (z - D / 2.0) ** 2 + eps * eps
        rm2 = s + (z + D / 2.0) ** 2 + eps * eps
        return a * (-3.0 * eps * eps * rp2.pow(-2.5) - 3.0 * eps * eps * rm2.pow(-2.5))

    def U_szD(s, z, D):
        om = omega_szD(s, z, D)
        return m0 * m0 * (om * om - 1.0) + (1.0 - 6.0 * xi) * lap_omega_szD(s, z, D) / om

    def quadrature_grid(D_value: float, requires_grad: bool = True):
        s1 = (torch.arange(n_s, device=device, dtype=dtype) + 0.5) * (s_max / n_s)
        z1 = -L_z + (torch.arange(n_z, device=device, dtype=dtype) + 0.5) * (2.0 * L_z / n_z)
        S, Z = torch.meshgrid(s1, z1, indexing="ij")
        s = S.reshape(-1, 1).detach().clone().requires_grad_(requires_grad)
        z = Z.reshape(-1, 1).detach().clone().requires_grad_(requires_grad)
        D = torch.full_like(s, float(D_value), requires_grad=False)
        return s, z, D

    def evaluate(model, D_value: float):
        s, z, D = quadrature_grid(D_value, requires_grad=True)
        u = model(s, z, D)
        du_ds, du_dz = torch.autograd.grad(u, (s, z), torch.ones_like(u), create_graph=True)
        d2u_ds2 = torch.autograd.grad(du_ds, s, torch.ones_like(du_ds), create_graph=True)[0]
        d2u_dz2 = torch.autograd.grad(du_dz, z, torch.ones_like(du_dz), create_graph=True)[0]
        U = U_szD(s, z, D)
        norm_mean = torch.mean(u * u)
        kinetic_s = torch.mean(4.0 * s * du_ds * du_ds)
        kinetic_z = torch.mean(du_dz * du_dz)
        potential = torch.mean(U * u * u)
        energy = (kinetic_s + kinetic_z + potential) / torch.clamp(norm_mean, min=1.0e-12)
        H_u = -4.0 * s * d2u_ds2 - 4.0 * du_ds - d2u_dz2 + U * u
        res = H_u - energy * u
        rms_u = torch.sqrt(norm_mean)
        return {
            "D": float(D_value),
            "E_parametric": float(energy.detach().cpu()),
            "omega_parametric": math.sqrt(max(float(energy.detach().cpu()) + m0 * m0, 0.0)),
            "norm": float(norm_mean.detach().cpu()),
            "strong_residual_l2_over_rms_u": float((torch.sqrt(torch.mean(res * res)) / torch.clamp(rms_u, min=1.0e-12)).detach().cpu()),
            "strong_residual_median_abs_over_rms_u": float((torch.median(torch.abs(res)) / torch.clamp(rms_u, min=1.0e-12)).detach().cpu()),
        }

    model = ParametricTrialNet().to(device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    D_values = args.D_values if args.D_values else dense_values(args.D_min, args.D_max, args.D_step)
    rows = [evaluate(model, D) for D in D_values]
    for i, row in enumerate(rows):
        if i == 0:
            row["delta_E_prev"] = None
            row["monotone_increase_ok"] = True
        else:
            delta = row["E_parametric"] - rows[i - 1]["E_parametric"]
            row["delta_E_prev"] = delta
            row["monotone_increase_ok"] = delta >= -1.0e-6
        row["residual_ok_lt_5e_minus_2"] = row["strong_residual_l2_over_rms_u"] < 5.0e-2

    max_res = max(r["strong_residual_l2_over_rms_u"] for r in rows)
    monotone_ok = all(r["monotone_increase_ok"] for r in rows)
    residual_ok = all(r["residual_ok_lt_5e_minus_2"] for r in rows)
    out = {
        "target": "V1 dense-D parametric evaluator",
        "source_run": str(run_dir.relative_to(ROOT)),
        "source_checkpoint_gate_max_rel_error": metrics.get("max_rel_error"),
        "source_checkpoint_gate_max_residual": metrics.get("max_strong_residual_l2_over_rms_u"),
        "D_values": D_values,
        "rows": rows,
        "max_dense_residual_l2_over_rms_u": max_res,
        "monotone_ok": monotone_ok,
        "residual_ok_lt_5e_minus_2": residual_ok,
        "needs_finite_volume_check": (not monotone_ok) or (not residual_ok),
        "device": str(device),
        "accelerator": accelerator,
    }

    run_name = args.run_name or f"v1_dense_eval_{run_dir.name}"
    eval_dir = RUNS_DIR / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (eval_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (eval_dir / "dense_metrics.csv").open("w", newline="") as f:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"run_dir": str(eval_dir), **out}, indent=2))


if __name__ == "__main__":
    main()

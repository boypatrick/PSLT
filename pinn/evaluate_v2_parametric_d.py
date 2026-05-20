#!/usr/bin/env python3
"""Evaluate a saved V2.2 parametric-D multi-mode Ritz-PINN on dense D points.

The evaluator is diagnostic-only.  It checks intermediate-D Ritz spectra,
Gram stability, strong residuals, and anchor-consistent monotonicity.  It does
not replace finite-volume references; it only identifies suspicious cells that
would deserve deterministic follow-up.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Saved V2.2 run name or path under pinn/runs.")
    p.add_argument("--D-values", default="7.5,9,10.5,13.5,15,16.5")
    p.add_argument("--include-anchors", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--n-residual", type=int, default=None)
    p.add_argument("--gram-threshold", type=float, default=0.05)
    p.add_argument("--residual-threshold", type=float, default=0.30)
    p.add_argument("--e2-turning-D", type=float, default=13.5, help="Expected E2 turn-over location for anchor-consistent monotonicity.")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def parse_d_values(value: str):
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("--D-values produced no points")
    return out


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path
    run_dir = RUNS_DIR / value
    if run_dir.is_dir():
        return run_dir
    raise FileNotFoundError(f"Could not find run directory for {value!r}")


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed in this Python environment.") from exc

    source_run_dir = resolve_run_dir(args.run)
    config = json.loads((source_run_dir / "config.json").read_text())
    model_path = source_run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    try:
        device, accelerator = select_torch_device(torch, args.device)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    dtype = torch.float32

    modes = int(config["modes"])
    n_s = int(config["n_s"])
    n_z = int(config["n_z"])
    n_residual = min(args.n_residual or int(config.get("n_residual", 768)), n_s * n_z)
    D_min = float(config["D_min"])
    D_max = float(config["D_max"])
    L_rho = float(config["L_rho"])
    L_z = float(config["L_z"])
    a = float(config["a"])
    eps = float(config["eps"])
    m0 = float(config["m0"])
    xi = float(config["xi"])
    hidden = int(config["hidden"])
    layers = int(config["layers"])
    residual_scale = float(config["residual_scale"])
    gram_ridge = float(config["gram_ridge"])
    s_max = L_rho * L_rho
    D_span = max(D_max - D_min, 1.0e-12)

    class ParametricMultiModeTrialNet(nn.Module):
        def __init__(self):
            super().__init__()
            modules = [nn.Linear(3, hidden), nn.Tanh()]
            for _ in range(layers - 1):
                modules += [nn.Linear(hidden, hidden), nn.Tanh()]
            modules += [nn.Linear(hidden, modes)]
            self.net = nn.Sequential(*modules)
            self.base_scale = nn.Parameter(torch.ones(modes, dtype=dtype))

        def forward(self, s, z, D):
            x = 2.0 * s / s_max - 1.0
            y = z / L_z
            d = 2.0 * (D - D_min) / D_span - 1.0
            envelope = torch.clamp(1.0 - s / s_max, min=0.0) * torch.clamp(1.0 - y * y, min=0.0)
            mlp = self.net(torch.cat([x, y, d], dim=1))
            bases = [torch.ones_like(z)]
            if modes >= 2:
                bases.append(y)
            if modes >= 3:
                bases.append(y * y - torch.mean(y * y))
            for k in range(3, modes):
                bases.append(torch.cos(float(k) * math.pi * y / 2.0))
            base = torch.cat(bases[:modes], dim=1)
            return envelope * (self.base_scale.reshape(1, -1) * base + residual_scale * mlp)

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
        D = torch.full_like(s, float(D_value), requires_grad=requires_grad)
        return s, z, D

    def projected_matrices(model, D_value: float):
        s, z, D = quadrature_grid(D_value, requires_grad=True)
        B = model(s, z, D)
        U = U_szD(s, z, D)
        grads_s = []
        grads_z = []
        for k in range(modes):
            du_ds, du_dz = torch.autograd.grad(B[:, k : k + 1], (s, z), torch.ones_like(B[:, k : k + 1]), create_graph=True)
            grads_s.append(du_ds)
            grads_z.append(du_dz)
        G = torch.empty((modes, modes), device=device, dtype=dtype)
        H = torch.empty((modes, modes), device=device, dtype=dtype)
        for i in range(modes):
            ui = B[:, i : i + 1]
            for j in range(modes):
                uj = B[:, j : j + 1]
                G[i, j] = torch.mean(ui * uj)
                H[i, j] = torch.mean(
                    4.0 * s * grads_s[i] * grads_s[j]
                    + grads_z[i] * grads_z[j]
                    + U * ui * uj
                )
        return 0.5 * (H + H.T), 0.5 * (G + G.T)

    def ritz_eigh(H, G):
        ridge = gram_ridge * torch.eye(modes, device=device, dtype=dtype)
        L = torch.linalg.cholesky(G + ridge)
        y = torch.linalg.solve_triangular(L, H, upper=False)
        c = torch.linalg.solve_triangular(L, y.T, upper=False).T
        c = 0.5 * (c + c.T)
        evals, q = torch.linalg.eigh(c)
        coeff = torch.linalg.solve_triangular(L.T, q, upper=True)
        return evals, coeff

    def gram_offdiag(G):
        diag = torch.clamp(torch.diag(G), min=1.0e-12)
        denom = torch.sqrt(diag[:, None] * diag[None, :])
        corr = G / denom
        eye = torch.eye(modes, device=device, dtype=dtype)
        return float((corr - eye).abs().masked_fill(torch.eye(modes, device=device, dtype=torch.bool), 0.0).max().detach().cpu())

    def residual_metrics(model, D_value: float, evals, coeff):
        n_total = n_s * n_z
        idx = torch.linspace(0, n_total - 1, n_residual, device=device).round().long()
        s_full, z_full, D_full = quadrature_grid(D_value, requires_grad=False)
        s = s_full[idx].detach().clone().requires_grad_(True)
        z = z_full[idx].detach().clone().requires_grad_(True)
        D = D_full[idx].detach().clone().requires_grad_(False)
        B = model(s, z, D)
        HB_cols = []
        for k in range(modes):
            uk = B[:, k : k + 1]
            du_ds, du_dz = torch.autograd.grad(uk, (s, z), torch.ones_like(uk), create_graph=True)
            d2u_ds2 = torch.autograd.grad(du_ds, s, torch.ones_like(du_ds), create_graph=True)[0]
            d2u_dz2 = torch.autograd.grad(du_dz, z, torch.ones_like(du_dz), create_graph=True)[0]
            HB_cols.append(-4.0 * s * d2u_ds2 - 4.0 * du_ds - d2u_dz2 + U_szD(s, z, D) * uk)
        HB = torch.cat(HB_cols, dim=1)
        U_modes = B @ coeff
        HU_modes = HB @ coeff
        out = []
        for k in range(modes):
            u = U_modes[:, k : k + 1]
            res = HU_modes[:, k : k + 1] - evals[k] * u
            rms = torch.sqrt(torch.mean(u * u))
            out.append(float((torch.sqrt(torch.mean(res * res)) / torch.clamp(rms, min=1.0e-12)).detach().cpu()))
        return out

    model = ParametricMultiModeTrialNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    d_values = parse_d_values(args.D_values)
    if args.include_anchors:
        d_values = sorted(set(d_values + [D_min, 0.5 * (D_min + D_max), D_max]))
    rows = []
    for D_value in d_values:
        H, G = projected_matrices(model, D_value)
        evals, coeff = ritz_eigh(H, G)
        e_values = [float(x) for x in evals[:modes].detach().cpu()]
        res_values = residual_metrics(model, D_value, evals.detach(), coeff.detach())
        max_residual = max(res_values)
        max_corr = gram_offdiag(G)
        row = {
            "D": float(D_value),
            "max_abs_corr_offdiag": max_corr,
            "max_strong_residual_l2_over_rms_u": max_residual,
            "gram_ok": max_corr < args.gram_threshold,
            "residual_ok": max_residual < args.residual_threshold,
        }
        for k in range(modes):
            row[f"E{k}_ritz"] = e_values[k]
            row[f"omega{k}_ritz"] = math.sqrt(max(e_values[k] + m0 * m0, 0.0))
            row[f"residual{k}_l2_over_rms_u"] = res_values[k]
        rows.append(row)

    rows.sort(key=lambda row: row["D"])

    # Anchor-consistent monotonicity: E0/E1 are increasing across the endpoint
    # anchors.  E2 has a finite-volume turn-over near the augmented anchor
    # specified by --e2-turning-D; intervals straddling that turning point are
    # not forced to be monotone.
    monotone_flags = {
        "E0_anchor_increase_ok": True,
        "E1_anchor_increase_ok": True,
        "E2_left_increase_ok": True,
        "E2_right_decrease_ok": True,
    }
    for prev, cur in zip(rows[:-1], rows[1:]):
        if cur["E0_ritz"] + 1.0e-10 < prev["E0_ritz"]:
            monotone_flags["E0_anchor_increase_ok"] = False
        if cur["E1_ritz"] + 1.0e-10 < prev["E1_ritz"]:
            monotone_flags["E1_anchor_increase_ok"] = False
        if prev["D"] < args.e2_turning_D and cur["D"] <= args.e2_turning_D and cur["E2_ritz"] + 1.0e-10 < prev["E2_ritz"]:
            monotone_flags["E2_left_increase_ok"] = False
        if prev["D"] >= args.e2_turning_D and cur["D"] > args.e2_turning_D and cur["E2_ritz"] > prev["E2_ritz"] + 1.0e-10:
            monotone_flags["E2_right_decrease_ok"] = False

    monotone_ok = all(monotone_flags.values())
    suspicious = []
    for row in rows:
        reasons = []
        if not row["gram_ok"]:
            reasons.append("gram")
        if not row["residual_ok"]:
            reasons.append("residual")
        # Do not duplicate every row for a global monotonicity issue; record it
        # once in the metrics and flag the whole dense sweep as needing review.
        row["suspicious_reasons"] = ";".join(reasons)
        row["suspicious"] = bool(reasons)
        if row["suspicious"]:
            suspicious.append(row)

    run_name = args.run_name or datetime.now().strftime("v2_dense_eval_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "target": "V2.3 dense-D evaluator for parametric-D K=3 Ritz-PINN",
        "source_run": str(source_run_dir.relative_to(ROOT)),
        "D_values": [row["D"] for row in rows],
        "modes": modes,
        "n_s": n_s,
        "n_z": n_z,
        "gram_threshold": args.gram_threshold,
        "residual_threshold": args.residual_threshold,
        "e2_turning_D": args.e2_turning_D,
        "max_dense_corr_offdiag": max(row["max_abs_corr_offdiag"] for row in rows),
        "max_dense_strong_residual_l2_over_rms_u": max(row["max_strong_residual_l2_over_rms_u"] for row in rows),
        "monotone_flags": monotone_flags,
        "anchor_consistent_monotone_ok": monotone_ok,
        "suspicious_D": [row["D"] for row in suspicious],
        "needs_finite_volume_check": bool(suspicious or not monotone_ok),
        "device": str(device),
        "accelerator": accelerator,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (run_dir / "dense_summary.csv").open("w", newline="") as f:
        fields = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

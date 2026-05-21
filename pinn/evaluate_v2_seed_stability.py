#!/usr/bin/env python3
"""V2.6 Monte-Carlo seed-stability gate for the frozen V2 Ritz-PINN.

This is an evaluation-only freeze gate.  The trained V2 model is deterministic,
and the V2 training loss uses full tensor-product quadrature, so changing a
seed during same-checkpoint continuation would not be a meaningful stability
test.  Instead, this script probes whether the frozen model's projected
Ritz spectrum, Gram conditioning, and strong residual remain stable when the
quadrature/collocation points are resampled.

The result is a PINN-emulator robustness diagnostic only.  Deterministic
finite-volume references remain the certificate source of truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from accelerator_utils import select_torch_device


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "pinn" / "runs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="v2_parametric_D6_18_K3_augmented_final_800")
    p.add_argument("--D-values", default="6.75,8.25,9.75,11.25,12.75,14.25,15.75,17.25")
    p.add_argument("--seeds", default="101,202,303,404,505")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--sampler", choices=["grid-jitter", "iid"], default="grid-jitter")
    p.add_argument("--n-project", type=int, default=4096, help="IID Monte-Carlo points for H/G projection.")
    p.add_argument("--n-residual", type=int, default=1024, help="IID Monte-Carlo points for strong residual.")
    p.add_argument("--jitter-n-s", type=int, default=None, help="Stratified s-cells for grid-jitter sampling; defaults to run n_s.")
    p.add_argument("--jitter-n-z", type=int, default=None, help="Stratified z-cells for grid-jitter sampling; defaults to run n_z.")
    p.add_argument("--energy-spread-threshold", type=float, default=2.5e-2)
    p.add_argument("--gram-threshold", type=float, default=8.0e-2)
    p.add_argument("--residual-threshold", type=float, default=3.5e-1)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def parse_float_list(value: str):
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("empty float list")
    return out


def parse_int_list(value: str):
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("empty int list")
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
    config_n_s = int(config["n_s"])
    config_n_z = int(config["n_z"])
    jitter_n_s = int(args.jitter_n_s or config_n_s)
    jitter_n_z = int(args.jitter_n_z or config_n_z)
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

    def iid_points(D_value: float, n_points: int, seed: int, salt: int, requires_grad: bool = True):
        torch.manual_seed(int(seed) + int(salt))
        s = s_max * torch.rand((n_points, 1), device=device, dtype=dtype)
        z = -L_z + 2.0 * L_z * torch.rand((n_points, 1), device=device, dtype=dtype)
        s = s.detach().clone().requires_grad_(requires_grad)
        z = z.detach().clone().requires_grad_(requires_grad)
        D = torch.full_like(s, float(D_value), requires_grad=False)
        return s, z, D

    def jittered_grid_points(D_value: float, seed: int, salt: int, requires_grad: bool = True):
        torch.manual_seed(int(seed) + int(salt))
        i_s = torch.arange(jitter_n_s, device=device, dtype=dtype).reshape(-1, 1).repeat(1, jitter_n_z)
        i_z = torch.arange(jitter_n_z, device=device, dtype=dtype).reshape(1, -1).repeat(jitter_n_s, 1)
        # One randomly jittered point per tensor-product cell.  This probes
        # quadrature sensitivity while preserving the stratification of the
        # deterministic V2 projection grid.
        js = torch.rand((jitter_n_s, jitter_n_z), device=device, dtype=dtype)
        jz = torch.rand((jitter_n_s, jitter_n_z), device=device, dtype=dtype)
        s = ((i_s + js) * (s_max / jitter_n_s)).reshape(-1, 1)
        z = (-L_z + (i_z + jz) * (2.0 * L_z / jitter_n_z)).reshape(-1, 1)
        s = s.detach().clone().requires_grad_(requires_grad)
        z = z.detach().clone().requires_grad_(requires_grad)
        D = torch.full_like(s, float(D_value), requires_grad=False)
        return s, z, D

    def sample_points(D_value: float, n_points: int, seed: int, salt: int, requires_grad: bool = True):
        if args.sampler == "grid-jitter":
            return jittered_grid_points(D_value, seed, salt, requires_grad=requires_grad)
        return iid_points(D_value, n_points, seed, salt, requires_grad=requires_grad)

    def projected_matrices(model, D_value: float, seed: int):
        s, z, D = sample_points(D_value, args.n_project, seed, salt=0, requires_grad=True)
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

    def residual_metrics(model, D_value: float, seed: int, evals, coeff):
        s, z, D = sample_points(D_value, args.n_residual, seed, salt=1000003, requires_grad=True)
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
    model.eval()

    d_values = parse_float_list(args.D_values)
    seeds = parse_int_list(args.seeds)
    rows = []
    for seed in seeds:
        for D_value in d_values:
            H, G = projected_matrices(model, D_value, seed)
            evals, coeff = ritz_eigh(H, G)
            e_values = [float(x) for x in evals[:modes].detach().cpu()]
            res_values = residual_metrics(model, D_value, seed, evals.detach(), coeff.detach())
            max_corr = gram_offdiag(G)
            max_res = max(res_values)
            row = {
                "seed": int(seed),
                "D": float(D_value),
                "max_abs_corr_offdiag": max_corr,
                "max_strong_residual_l2_over_rms_u": max_res,
                "gram_ok": max_corr < args.gram_threshold,
                "residual_ok": max_res < args.residual_threshold,
            }
            for k in range(modes):
                row[f"E{k}_ritz"] = e_values[k]
                row[f"omega{k}_ritz"] = math.sqrt(max(e_values[k] + m0 * m0, 0.0))
                row[f"residual{k}_l2_over_rms_u"] = res_values[k]
            rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        for k in range(modes):
            grouped[(row["D"], k)].append(row[f"E{k}_ritz"])

    d_summary_rows = []
    max_energy_spread_rel = 0.0
    for D_value in d_values:
        out = {"D": float(D_value)}
        for k in range(modes):
            vals = grouped[(float(D_value), k)]
            mean_val = sum(vals) / len(vals)
            spread_abs = max(vals) - min(vals)
            spread_rel = spread_abs / max(abs(mean_val), 1.0)
            max_energy_spread_rel = max(max_energy_spread_rel, spread_rel)
            out[f"E{k}_mean"] = mean_val
            out[f"E{k}_min"] = min(vals)
            out[f"E{k}_max"] = max(vals)
            out[f"E{k}_spread_abs"] = spread_abs
            out[f"E{k}_spread_rel"] = spread_rel
        d_summary_rows.append(out)

    max_corr = max(row["max_abs_corr_offdiag"] for row in rows)
    max_residual = max(row["max_strong_residual_l2_over_rms_u"] for row in rows)
    all_seed_rows_pass = all(row["gram_ok"] and row["residual_ok"] for row in rows)
    gate_pass = (
        max_energy_spread_rel < args.energy_spread_threshold
        and max_corr < args.gram_threshold
        and max_residual < args.residual_threshold
        and all_seed_rows_pass
    )
    metrics = {
        "target": "V2.6 Monte-Carlo seed-stability freeze gate",
        "source_run": str(source_run_dir.relative_to(ROOT)),
        "D_values": d_values,
        "seeds": seeds,
        "modes": modes,
        "sampler": args.sampler,
        "n_project": int(jitter_n_s * jitter_n_z if args.sampler == "grid-jitter" else args.n_project),
        "n_residual": int(jitter_n_s * jitter_n_z if args.sampler == "grid-jitter" else args.n_residual),
        "jitter_n_s": int(jitter_n_s),
        "jitter_n_z": int(jitter_n_z),
        "energy_spread_threshold": float(args.energy_spread_threshold),
        "gram_threshold": float(args.gram_threshold),
        "residual_threshold": float(args.residual_threshold),
        "max_energy_spread_rel": float(max_energy_spread_rel),
        "max_corr_offdiag": float(max_corr),
        "max_strong_residual_l2_over_rms_u": float(max_residual),
        "all_seed_rows_pass": bool(all_seed_rows_pass),
        "gate_pass": bool(gate_pass),
        "verdict": "V2_6_SEED_STABILITY_PASS" if gate_pass else "V2_6_SEED_STABILITY_REVIEW",
        "device": str(device),
        "accelerator": accelerator,
    }

    run_name = args.run_name or datetime.now().strftime("v2_seed_stability_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with (run_dir / "seed_detail.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "D_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(d_summary_rows[0]))
        writer.writeheader()
        writer.writerows(d_summary_rows)

    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

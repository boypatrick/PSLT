#!/usr/bin/env python3
"""V2.2 parametric-D multi-mode self-adjoint Ritz-PINN.

This script starts only after the fixed-D K=3 V2 endpoint gates pass.  It
learns a D-dependent K-dimensional trial subspace and compares its generalized
Ritz values against finite-volume references at checkpoint anchors.

The neural model is a differentiable emulator, not a replacement for the
deterministic self-adjoint eigensolver.
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
ANCHOR_SUMMARY = ROOT / "pinn" / "v2_fixed_endpoint_summary.csv"


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
    p.add_argument("--n-s", type=int, default=32)
    p.add_argument("--n-z", type=int, default=96)
    p.add_argument("--modes", type=int, default=3)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--residual-scale", type=float, default=0.20)
    p.add_argument("--w-reference", type=float, default=160.0)
    p.add_argument("--w-orth", type=float, default=2.0)
    p.add_argument("--w-norm", type=float, default=1.0e-4)
    p.add_argument("--w-smooth", type=float, default=0.0)
    p.add_argument("--gram-ridge", type=float, default=1.0e-7)
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--init-from-run", default=None, help="Load model.pt from a previous pinn/runs entry or explicit path.")
    p.add_argument("--log-every", type=int, default=400)
    p.add_argument("--n-residual", type=int, default=768)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_anchors(path: str, modes: int):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            refs = []
            for k in range(modes):
                key = f"E{k}_ref"
                if key not in row:
                    raise ValueError(f"Anchor file {path} is missing {key}")
                refs.append(float(row[key]))
            rows.append({
                "D": float(row["D"]),
                "E_ref": refs,
                "source": row.get("run", "") or row.get("reference_run", ""),
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

    if args.modes < 1:
        raise SystemExit("--modes must be positive")
    anchors = load_anchors(args.anchors, args.modes)
    torch.manual_seed(args.seed)
    try:
        device, accelerator = select_torch_device(torch, args.device)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    dtype = torch.float32
    s_max = args.L_rho * args.L_rho
    D_span = max(args.D_max - args.D_min, 1.0e-12)

    class ParametricMultiModeTrialNet(nn.Module):
        def __init__(self):
            super().__init__()
            modules = [nn.Linear(3, args.hidden), nn.Tanh()]
            for _ in range(args.layers - 1):
                modules += [nn.Linear(args.hidden, args.hidden), nn.Tanh()]
            modules += [nn.Linear(args.hidden, args.modes)]
            self.net = nn.Sequential(*modules)
            self.base_scale = nn.Parameter(torch.ones(args.modes, dtype=dtype))

        def forward(self, s, z, D):
            x = 2.0 * s / s_max - 1.0
            y = z / args.L_z
            d = 2.0 * (D - args.D_min) / D_span - 1.0
            envelope = torch.clamp(1.0 - s / s_max, min=0.0) * torch.clamp(1.0 - y * y, min=0.0)
            mlp = self.net(torch.cat([x, y, d], dim=1))
            bases = [torch.ones_like(z)]
            if args.modes >= 2:
                bases.append(y)
            if args.modes >= 3:
                bases.append(y * y - torch.mean(y * y))
            for k in range(3, args.modes):
                bases.append(torch.cos(float(k) * math.pi * y / 2.0))
            base = torch.cat(bases[: args.modes], dim=1)
            return envelope * (self.base_scale.reshape(1, -1) * base + args.residual_scale * mlp)

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

    def quadrature_grid(D_value: float, requires_grad: bool = True):
        s1 = (torch.arange(args.n_s, device=device, dtype=dtype) + 0.5) * (s_max / args.n_s)
        z1 = -args.L_z + (torch.arange(args.n_z, device=device, dtype=dtype) + 0.5) * (2.0 * args.L_z / args.n_z)
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
        return 0.5 * (H + H.T), 0.5 * (G + G.T)

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
        offdiag = (corr - eye).abs().masked_fill(torch.eye(args.modes, device=device, dtype=torch.bool), 0.0)
        orth = torch.mean((corr - eye) ** 2)
        norm = torch.mean(torch.log(diag) ** 2)
        return orth, norm, corr, torch.max(offdiag)

    def residual_metrics(model, D_value: float, evals, coeff):
        n_total = args.n_s * args.n_z
        n_res = min(args.n_residual, n_total)
        s_full, z_full, D_full = quadrature_grid(D_value, requires_grad=False)
        idx = torch.linspace(0, n_total - 1, n_res, device=device).round().long()
        s = s_full[idx].detach().clone().requires_grad_(True)
        z = z_full[idx].detach().clone().requires_grad_(True)
        D = D_full[idx].detach().clone().requires_grad_(False)
        B = model(s, z, D)
        HB_cols = []
        for k in range(args.modes):
            uk = B[:, k : k + 1]
            du_ds, du_dz = torch.autograd.grad(uk, (s, z), torch.ones_like(uk), create_graph=True)
            d2u_ds2 = torch.autograd.grad(du_ds, s, torch.ones_like(du_ds), create_graph=True)[0]
            d2u_dz2 = torch.autograd.grad(du_dz, z, torch.ones_like(du_dz), create_graph=True)[0]
            HB_cols.append(-4.0 * s * d2u_ds2 - 4.0 * du_ds - d2u_dz2 + U_szD(s, z, D) * uk)
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
            })
        return rows

    model = ParametricMultiModeTrialNet().to(device)
    init_model_path = resolve_model_path(args.init_from_run)
    if init_model_path is not None:
        state = torch.load(init_model_path, map_location=device)
        model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_state = None
    best_score = float("inf")
    best_step = 0

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        trace_terms = []
        reference_terms = []
        orth_terms = []
        norm_terms = []
        max_rel_terms = []
        max_corr_terms = []
        for anchor in anchors:
            H, G = projected_matrices(model, anchor["D"])
            evals, _ = ritz_eigh(H, G)
            ref = torch.tensor(anchor["E_ref"], device=device, dtype=dtype)
            scale = torch.clamp(torch.abs(ref), min=1.0)
            rel = torch.abs((evals[: args.modes] - ref) / scale)
            orth_loss, norm_loss, _corr, max_corr = gram_losses(G)
            trace_terms.append(torch.sum(evals[: args.modes]))
            reference_terms.append(torch.mean(rel * rel))
            orth_terms.append(orth_loss)
            norm_terms.append(norm_loss)
            max_rel_terms.append(torch.max(rel))
            max_corr_terms.append(max_corr)
        loss_trace = torch.stack(trace_terms).mean()
        loss_reference = torch.stack(reference_terms).mean()
        loss_orth = torch.stack(orth_terms).mean()
        loss_norm = torch.stack(norm_terms).mean()
        # Smoothness is deliberately a tiny direct D-variation monitor.  It is
        # zero by default until a later V2.2 tightening pass needs it.
        loss_smooth = torch.zeros((), device=device, dtype=dtype)
        loss = (
            loss_trace
            + args.w_reference * loss_reference
            + args.w_orth * loss_orth
            + args.w_norm * loss_norm
            + args.w_smooth * loss_smooth
        )
        loss.backward()
        opt.step()
        max_rel = torch.max(torch.stack(max_rel_terms))
        max_corr = torch.max(torch.stack(max_corr_terms))
        gate_score = float((max_rel + max_corr + 0.1 * loss_orth).detach().cpu())
        if gate_score < best_score:
            best_score = gate_score
            best_step = step
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "loss_trace": float(loss_trace.detach().cpu()),
                "loss_reference": float(loss_reference.detach().cpu()),
                "loss_orth": float(loss_orth.detach().cpu()),
                "loss_norm": float(loss_norm.detach().cpu()),
                "max_rel_error_train": float(max_rel.detach().cpu()),
                "max_abs_corr_offdiag_train": float(max_corr.detach().cpu()),
                "gate_score": gate_score,
            }
            history.append(row)
            print(json.dumps(row))

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    anchor_rows = []
    all_residuals = []
    for anchor in anchors:
        H, G = projected_matrices(model, anchor["D"])
        evals, coeff = ritz_eigh(H, G)
        orth_loss, norm_loss, corr, max_corr = gram_losses(G)
        e_values = [float(x) for x in evals[: args.modes].detach().cpu()]
        rel_errors = [
            abs(e_values[k] - anchor["E_ref"][k]) / max(abs(anchor["E_ref"][k]), 1.0e-12)
            for k in range(args.modes)
        ]
        residual_rows = residual_metrics(model, anchor["D"], evals.detach(), coeff.detach())
        max_res = max(row["strong_residual_l2_over_rms_u"] for row in residual_rows)
        anchor_rows.append({
            "D": float(anchor["D"]),
            "source": anchor["source"],
            "max_E_rel_error": max(rel_errors),
            "median_E_rel_error": sorted(rel_errors)[len(rel_errors) // 2],
            "max_abs_corr_offdiag": float(max_corr.detach().cpu()),
            "max_strong_residual_l2_over_rms_u": max_res,
            **{f"E{k}_ritz": e_values[k] for k in range(args.modes)},
            **{f"E{k}_ref": anchor["E_ref"][k] for k in range(args.modes)},
        })
        for row in residual_rows:
            all_residuals.append({"D": float(anchor["D"]), **row})

    metrics = {
        "target": "V2.2 parametric-D multi-mode self-adjoint Ritz-PINN",
        "D_min": float(args.D_min),
        "D_max": float(args.D_max),
        "modes": int(args.modes),
        "n_s": int(args.n_s),
        "n_z": int(args.n_z),
        "steps": int(args.steps),
        "anchors": anchor_rows,
        "max_anchor_E_rel_error": max(row["max_E_rel_error"] for row in anchor_rows),
        "max_anchor_corr_offdiag": max(row["max_abs_corr_offdiag"] for row in anchor_rows),
        "max_anchor_strong_residual_l2_over_rms_u": max(row["max_strong_residual_l2_over_rms_u"] for row in anchor_rows),
        "selected_checkpoint_step": int(best_step),
        "selected_checkpoint_score": float(best_score),
        "init_from_run": args.init_from_run,
        "device": str(device),
        "accelerator": accelerator,
        "seed": int(args.seed),
    }

    run_name = args.run_name or datetime.now().strftime("v2_parametric_d_multimode_%Y%m%d_%H%M%S")
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
    with (run_dir / "anchor_summary.csv").open("w", newline="") as f:
        fields = list(anchor_rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(anchor_rows)
    with (run_dir / "mode_residuals.csv").open("w", newline="") as f:
        fields = list(all_residuals[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_residuals)

    print(json.dumps({"run_dir": str(run_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()

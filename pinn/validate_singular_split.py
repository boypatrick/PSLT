#!/usr/bin/env python3
"""Autograd validation of the V0.2 analytic singular split.

For s=rho^2 and one Plummer center at z0,

    L_s (s + (z-z0)^2 + eps^2)^(-1/2)
    = -3 eps^2 (s + (z-z0)^2 + eps^2)^(-5/2),

where L_s = 4 s d_s^2 + 4 d_s + d_z^2.  Therefore the exact two-center
background Omega_sing satisfies

    L_s Omega_sing + 4 pi sigma = 0.

This script checks that identity with the same autograd derivative structure
used by the split PINN training code, across several D values and source-biased
holdout samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=float, nargs="+", default=[6.0, 12.0, 18.0])
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--rho-max", type=float, default=8.0)
    parser.add_argument("--z-pad", type=float, default=8.0)
    parser.add_argument("--n-uniform", type=int, default=2048)
    parser.add_argument("--n-core", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This check requires numpy and torch. Install with: "
            "python3 -m pip install -r pinn/requirements.txt"
        ) from exc

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dtype = torch.float64
    device = torch.device("cpu")
    s_max = args.rho_max * args.rho_max

    def omega_sing_s(s, z, D):
        r_plus = torch.sqrt(torch.clamp(s, min=0.0) + (z - D / 2.0) ** 2 + args.eps * args.eps)
        r_minus = torch.sqrt(torch.clamp(s, min=0.0) + (z + D / 2.0) ** 2 + args.eps * args.eps)
        return 1.0 + args.a * (1.0 / r_plus + 1.0 / r_minus)

    def sigma_s(s, z, D):
        r2_plus = s + (z - D / 2.0) ** 2
        r2_minus = s + (z + D / 2.0) ** 2
        coeff = 3.0 * args.eps * args.eps / (4.0 * torch.pi)
        return args.a * (
            coeff / (r2_plus + args.eps * args.eps) ** 2.5
            + coeff / (r2_minus + args.eps * args.eps) ** 2.5
        )

    def residual_for_samples(s_np, z_np, D):
        s = torch.tensor(s_np.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True)
        z = torch.tensor(z_np.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True)
        omega = omega_sing_s(s, z, D)
        grad_s = torch.autograd.grad(omega, s, torch.ones_like(omega), create_graph=True)[0]
        grad_z = torch.autograd.grad(omega, z, torch.ones_like(omega), create_graph=True)[0]
        grad_ss = torch.autograd.grad(grad_s, s, torch.ones_like(grad_s), create_graph=True)[0]
        grad_zz = torch.autograd.grad(grad_z, z, torch.ones_like(grad_z), create_graph=True)[0]
        residual = 4.0 * s * grad_ss + 4.0 * grad_s + grad_zz + 4.0 * torch.pi * sigma_s(s, z, D)
        return residual.detach().cpu().numpy().reshape(-1)

    def stats(values):
        abs_values = np.abs(values)
        return {
            "count": int(values.size),
            "rmse": float(np.sqrt(np.mean(values * values))),
            "max_abs": float(np.max(abs_values)),
            "median_abs": float(np.median(abs_values)),
            "p99_abs": float(np.quantile(abs_values, 0.99)),
        }

    summaries = []
    for D in args.D:
        z_max = D / 2.0 + args.z_pad
        s_uniform = s_max * rng.random(args.n_uniform)
        z_uniform = -z_max + 2.0 * z_max * rng.random(args.n_uniform)

        counts = [args.n_core // 2, args.n_core - args.n_core // 2]
        s_core_parts = []
        z_core_parts = []
        for count, center in zip(counts, [-D / 2.0, D / 2.0]):
            radial = rng.exponential(scale=args.eps, size=count)
            s_core_parts.append(np.clip(radial * radial, 0.0, s_max))
            z_core_parts.append(np.clip(center + args.eps * rng.normal(size=count), -z_max, z_max))
        s_core = np.concatenate(s_core_parts)
        z_core = np.concatenate(z_core_parts)

        res_uniform = residual_for_samples(s_uniform, z_uniform, D)
        res_core = residual_for_samples(s_core, z_core, D)
        summary = {
            "D": float(D),
            "z_max": float(z_max),
            "uniform": stats(res_uniform),
            "core_biased": stats(res_core),
        }
        summaries.append(summary)

    out = {
        "check": "L_s Omega_sing + 4 pi sigma = 0",
        "operator": "L_s = 4 s d_s^2 + 4 d_s + d_z^2",
        "dtype": "float64",
        "a": args.a,
        "eps": args.eps,
        "rho_max": args.rho_max,
        "n_uniform": args.n_uniform,
        "n_core": args.n_core,
        "seed": args.seed,
        "summaries": summaries,
    }

    print("V0.2 singular-split autograd validation")
    print(f"eps={args.eps:g} a={args.a:g} rho_max={args.rho_max:g} seed={args.seed}")
    for item in summaries:
        D = item["D"]
        u = item["uniform"]
        c = item["core_biased"]
        print(
            f"D={D:g} uniform_median_abs={u['median_abs']:.3e} "
            f"uniform_max_abs={u['max_abs']:.3e} "
            f"core_median_abs={c['median_abs']:.3e} "
            f"core_max_abs={c['max_abs']:.3e}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Derive an effective 3D projected source profile from a higher-dimensional
parent profile and fit it to the Plummer regulator family used in PSLT.

This is a forward-modeling bridge for the "high-dimensional mother theory"
upgrade path:
  parent profile w(z)  ->  sigma_eff(r)  ->  (a_eff, eps_eff) candidate.

Model:
  sigma_eff(r) ∝ ∫ dz w(z) / (r^2 + z^2)^(3/2),
then normalized on R^3:
  4π ∫ r^2 sigma_eff(r) dr = 1.

Outputs:
  - output/source_projection/source_projection_profile_<tag>.csv
  - output/source_projection/source_projection_fit_<tag>.json
  - paper/source_projection_profile_<tag>.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "source_projection"
PAPER = ROOT / "paper"


def _parent_profile(z: np.ndarray, model: str, width: float) -> np.ndarray:
    if model == "gaussian":
        w = np.exp(-0.5 * (z / width) ** 2)
    elif model == "laplace":
        w = np.exp(-np.abs(z) / max(width, 1e-12))
    elif model == "sech2":
        w = 1.0 / np.cosh(z / max(width, 1e-12)) ** 2
    else:
        raise ValueError(f"Unsupported parent model: {model}")
    norm = np.trapezoid(w, z)
    return w / max(float(norm), 1e-30)


def _sigma_eff_from_parent(r: np.ndarray, z: np.ndarray, w: np.ndarray, core_reg: float) -> np.ndarray:
    rr = r[:, None]
    zz = z[None, :]
    core2 = float(max(core_reg, 0.0)) ** 2
    kern = w[None, :] / np.power(rr * rr + zz * zz + core2, 1.5)
    s = np.trapezoid(kern, z, axis=1)
    s = np.maximum(s, 0.0)
    # Normalize to 3D source convention: 4π ∫ r^2 sigma(r) dr = 1
    norm3 = 4.0 * np.pi * np.trapezoid((r * r) * s, r)
    return s / max(float(norm3), 1e-30)


def _plummer_sigma(r: np.ndarray, eps: float) -> np.ndarray:
    return (3.0 * eps * eps) / (4.0 * np.pi * np.power(r * r + eps * eps, 2.5))


def _fit_plummer_eps(r: np.ndarray, sigma_eff: np.ndarray, eps_min: float, eps_max: float, n_grid: int) -> Tuple[float, float]:
    eps_grid = np.linspace(float(eps_min), float(eps_max), int(n_grid))
    best_eps = float(eps_grid[0])
    best_rmse = float("inf")
    for eps in eps_grid:
        s_pl = _plummer_sigma(r, eps)
        rmse = float(np.sqrt(np.mean((sigma_eff - s_pl) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_eps = float(eps)
    return best_eps, best_rmse


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Project higher-dimensional parent profile to effective 3D source and fit Plummer epsilon.")
    ap.add_argument("--parent-model", choices=["gaussian", "laplace", "sech2"], default="gaussian")
    ap.add_argument("--parent-width", type=float, default=0.2)
    ap.add_argument("--z-max", type=float, default=6.0)
    ap.add_argument("--nz", type=int, default=4001)
    ap.add_argument("--r-max", type=float, default=8.0)
    ap.add_argument("--nr", type=int, default=1600)
    ap.add_argument("--fit-eps-min", type=float, default=0.02)
    ap.add_argument("--fit-eps-max", type=float, default=0.6)
    ap.add_argument("--fit-eps-grid", type=int, default=800)
    ap.add_argument(
        "--projection-core-reg",
        type=float,
        default=-1.0,
        help="Core regulator in projected kernel denominator. If <=0, use parent-width.",
    )
    ap.add_argument("--tag", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    z = np.linspace(-float(args.z_max), float(args.z_max), int(args.nz))
    r = np.linspace(0.0, float(args.r_max), int(args.nr))
    w = _parent_profile(z=z, model=str(args.parent_model), width=float(args.parent_width))
    core_reg = float(args.parent_width) if float(args.projection_core_reg) <= 0.0 else float(args.projection_core_reg)
    sigma_eff = _sigma_eff_from_parent(r=r, z=z, w=w, core_reg=core_reg)

    eps_fit, rmse = _fit_plummer_eps(
        r=r,
        sigma_eff=sigma_eff,
        eps_min=float(args.fit_eps_min),
        eps_max=float(args.fit_eps_max),
        n_grid=int(args.fit_eps_grid),
    )
    sigma_fit = _plummer_sigma(r, eps_fit)

    # Shape agreement proxy (log-space low-r weighted)
    w_r = 1.0 / np.maximum(r + 0.05, 0.05)
    log_rmse = float(
        np.sqrt(
            np.average(
                (np.log(np.maximum(sigma_eff, 1e-30)) - np.log(np.maximum(sigma_fit, 1e-30))) ** 2,
                weights=w_r,
            )
        )
    )

    tag = str(args.tag).strip()
    if not tag:
        tag = f"{args.parent_model}_w{float(args.parent_width):.3g}"
    tag = tag.replace(".", "p").replace("-", "m")

    out_csv = OUTDIR / f"source_projection_profile_{tag}.csv"
    out_json = OUTDIR / f"source_projection_fit_{tag}.json"
    paper_csv = PAPER / out_csv.name

    df = pd.DataFrame(
        {
            "r": r,
            "sigma_eff": sigma_eff,
            "sigma_plummer_fit": sigma_fit,
        }
    )
    df.to_csv(out_csv, index=False)
    paper_csv.write_text(out_csv.read_text())

    payload = {
        "parent_model": str(args.parent_model),
        "parent_width": float(args.parent_width),
        "z_max": float(args.z_max),
        "nz": int(args.nz),
        "r_max": float(args.r_max),
        "nr": int(args.nr),
        "eps_fit": float(eps_fit),
        "rmse_linear": float(rmse),
        "rmse_log_weighted": float(log_rmse),
        "fit_eps_range": [float(args.fit_eps_min), float(args.fit_eps_max)],
        "fit_eps_grid": int(args.fit_eps_grid),
        "projection_core_reg": float(core_reg),
    }
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {paper_csv}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

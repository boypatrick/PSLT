#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from extract_chi_localized_2d import PhysicalParams, u_potential  # noqa: E402
from extract_y_eff_2d_three_channel import chirality_profiles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit curvature/Hessian-controlled effective-width candidates "
            "for the canonical muon overlap kernel."
        )
    )
    parser.add_argument(
        "--overlap-detail-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/y_eff_2d/"
            "y_eff_2d_three_channel_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
        ),
    )
    parser.add_argument(
        "--summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_summary_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    parser.add_argument(
        "--core-h",
        type=float,
        default=1.0e-3,
        help="Finite-difference step for action-derived core-Hessian probes.",
    )
    return parser.parse_args()


def finite_box_amplitude(D: float, sigma: float, rho_max: float, z_margin: float) -> float:
    i_rho = math.pi * sigma * sigma * (1.0 - math.exp(-(rho_max * rho_max) / (sigma * sigma)))
    i_z = 0.5 * math.sqrt(math.pi) * sigma * (
        math.erf((D + z_margin) / sigma) + math.erf(z_margin / sigma)
    )
    return math.exp(-(D * D) / (4.0 * sigma * sigma)) / max(i_rho * i_z, 1.0e-300)


def finite_box_m2(sigma: float, rho_max: float, z_max: float) -> float:
    a = (rho_max * rho_max) / (sigma * sigma)
    i0_rho = math.pi * sigma * sigma * (1.0 - math.exp(-a))
    i2_rho = math.pi * sigma**4 * (1.0 - math.exp(-a) * (1.0 + a))

    b = z_max / sigma
    i0_z = math.sqrt(math.pi) * sigma * math.erf(b)
    i2_z = 0.5 * math.sqrt(math.pi) * sigma**3 * math.erf(b) - sigma * sigma * z_max * math.exp(-(b * b))
    return i2_rho / max(i0_rho, 1.0e-300) + i2_z / max(i0_z, 1.0e-300)


def invert_sigma_from_m2(target_m2: float, rho_max: float, z_max: float) -> float:
    lo, hi = 1.0e-3, 32.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if finite_box_m2(mid, rho_max, z_max) < target_m2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exact_midplane_amplitude(kernel: np.ndarray, sigma: float, rr: np.ndarray, zz: np.ndarray) -> float:
    base = np.exp(-(rr * rr + zz * zz) / (sigma * sigma))
    return float(np.sum(kernel * base) / max(np.sum(base * base), 1.0e-300))


def fit_sigma_from_log_hessian(kernel: np.ndarray, rr: np.ndarray, zz: np.ndarray) -> tuple[float, float, float]:
    q = (rr * rr + zz * zz).ravel()
    y = np.log(np.maximum(kernel.ravel(), 1.0e-300))
    design = np.column_stack([np.ones_like(q), q])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    sigma = float(math.sqrt(max(-1.0 / slope, 1.0e-300)))
    resid = y - (intercept + slope * q)
    return sigma, intercept, float(np.max(np.abs(resid)))


def rel_sup_error(target: np.ndarray, approx: np.ndarray) -> float:
    return float(np.max(np.abs(target - approx)) / max(float(np.max(np.abs(target))), 1.0e-300))


def core_hessian_curvatures(D: float, p: PhysicalParams, h: float) -> tuple[float, float]:
    z0 = 0.5 * D
    u00 = float(u_potential(np.array([[0.0]]), np.array([[z0]]), D, p)[0, 0])
    uzh = float(u_potential(np.array([[0.0]]), np.array([[z0 + h]]), D, p)[0, 0])
    uzm = float(u_potential(np.array([[0.0]]), np.array([[z0 - h]]), D, p)[0, 0])
    urh = float(u_potential(np.array([[h]]), np.array([[z0]]), D, p)[0, 0])
    urm = float(u_potential(np.array([[-h]]), np.array([[z0]]), D, p)[0, 0])
    kappa_z = (uzh - 2.0 * u00 + uzm) / (h * h)
    kappa_rho = (urh - 2.0 * u00 + urm) / (h * h)
    return float(kappa_rho), float(kappa_z)


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.overlap_detail_input).copy().sort_values("D").reset_index(drop=True)
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])
    p = PhysicalParams()

    ref_sigma_exact = 2.5
    kappa_rho_ref, kappa_z_ref = core_hessian_curvatures(ref_d, p, args.core_h)
    kappa_iso_ref = (2.0 * kappa_rho_ref + kappa_z_ref) / 3.0
    c_sigma = ref_sigma_exact * (kappa_iso_ref ** 0.25)

    rows: list[dict[str, float]] = []
    for _, row in detail.iterrows():
        D = float(row["D"])
        dr = float(row["dr"])
        dz = float(row["dz"])
        rho_max = float(row["rho_max"])
        z_max = float(row["z_max"])
        z_margin = z_max - D / 2.0
        sigma_l = float(row["sigma_l"])
        sigma_r = float(row["sigma_r"])
        sigma_exact = 0.5 * (sigma_l + sigma_r)

        nr = int(round(rho_max / dr))
        nz = int(round(2.0 * z_max / dz))
        rho = (np.arange(nr) + 0.5) * dr
        z = -z_max + (np.arange(nz) + 0.5) * dz
        rr, zz = np.meshgrid(rho, z, indexing="ij")
        f_l, f_r = chirality_profiles(
            rr=rr,
            zz=zz,
            rho=rho,
            dr=dr,
            dz=dz,
            d_val=D,
            sigma_l=sigma_l,
            sigma_r=sigma_r,
        )
        kernel = f_l * f_r

        sigma_hess, intercept, max_logfit_resid = fit_sigma_from_log_hessian(kernel, rr, zz)
        amp_hess = math.exp(intercept)

        w = 2.0 * math.pi * rho[:, None] * dr * dz
        q = rr * rr + zz * zz
        m2_obs = float(np.sum(w * kernel * q) / max(np.sum(w * kernel), 1.0e-300))
        sigma_moment = invert_sigma_from_m2(m2_obs, rho_max, z_max)

        kappa_rho, kappa_z = core_hessian_curvatures(D, p, args.core_h)
        kappa_iso = (2.0 * kappa_rho + kappa_z) / 3.0
        sigma_curv = c_sigma * (kappa_iso ** -0.25)

        amp_curv = finite_box_amplitude(D, sigma_curv, rho_max, z_margin)
        approx_curv = amp_curv * np.exp(-(rr * rr + zz * zz) / (sigma_curv * sigma_curv))
        rel_sup_curv = rel_sup_error(kernel, approx_curv)

        rows.append(
            {
                "D": D,
                "sigma_exact": sigma_exact,
                "sigma_hessian": sigma_hess,
                "sigma_moment": sigma_moment,
                "sigma_curvature": sigma_curv,
                "sigma_hessian_relerr": float(abs(sigma_hess - sigma_exact) / sigma_exact),
                "sigma_moment_relerr": float(abs(sigma_moment - sigma_exact) / sigma_exact),
                "sigma_curvature_relerr": float(abs(sigma_curv - sigma_exact) / sigma_exact),
                "logfit_amp": amp_hess,
                "logfit_max_resid": max_logfit_resid,
                "m2_obs": m2_obs,
                "rho_max": rho_max,
                "z_max": z_max,
                "z_margin": z_margin,
                "kappa_rho_core": kappa_rho,
                "kappa_z_core": kappa_z,
                "kappa_iso_core": kappa_iso,
                "core_anisotropy_ratio": float(kappa_rho / max(kappa_z, 1.0e-300)),
                "amp_curvature_box": amp_curv,
                "kernel_rel_sup_curvature_candidate": rel_sup_curv,
            }
        )

    out_detail = pd.DataFrame(rows)

    ref_sigma_curv = float(c_sigma * (((2.0 * kappa_rho_ref + kappa_z_ref) / 3.0) ** -0.25))
    ref_amp_curv = finite_box_amplitude(ref_d, ref_sigma_curv, float(out_detail["rho_max"].iloc[0]), float(out_detail["z_margin"].iloc[0]))

    summary = pd.DataFrame(
        [
            {
                "section": "exact_log_hessian_width",
                "max_sigma_hessian_relerr": float(out_detail["sigma_hessian_relerr"].max()),
                "mean_sigma_hessian_relerr": float(out_detail["sigma_hessian_relerr"].mean()),
                "max_logfit_resid": float(out_detail["logfit_max_resid"].max()),
            },
            {
                "section": "finite_box_moment_width",
                "max_sigma_moment_relerr": float(out_detail["sigma_moment_relerr"].max()),
                "mean_sigma_moment_relerr": float(out_detail["sigma_moment_relerr"].mean()),
            },
            {
                "section": "curvature_controlled_width",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "sigma_ref_target": ref_sigma_exact,
                "c_sigma": c_sigma,
                "kappa_iso_ref": kappa_iso_ref,
                "max_sigma_curvature_relerr": float(out_detail["sigma_curvature_relerr"].max()),
                "mean_sigma_curvature_relerr": float(out_detail["sigma_curvature_relerr"].mean()),
                "anisotropy_ratio_min": float(out_detail["core_anisotropy_ratio"].min()),
                "anisotropy_ratio_max": float(out_detail["core_anisotropy_ratio"].max()),
                "max_kernel_rel_sup_curvature_candidate": float(out_detail["kernel_rel_sup_curvature_candidate"].max()),
                "mean_kernel_rel_sup_curvature_candidate": float(out_detail["kernel_rel_sup_curvature_candidate"].mean()),
            },
            {
                "section": "reference_constants",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "sigma_exact_ref": ref_sigma_exact,
                "sigma_curvature_ref": ref_sigma_curv,
                "amp_curvature_box_ref": ref_amp_curv,
                "kappa_rho_ref": kappa_rho_ref,
                "kappa_z_ref": kappa_z_ref,
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "kernel_log_hessian_width",
                "formula": "log K_mu = c0 - (rho^2+z^2)/sigma_mu^2",
                "status": "exact current-chain reduction on the canonical kernel",
            },
            {
                "candidate": "finite_box_moment_width",
                "formula": "<rho^2+z^2>_K = M2_box(sigma_mu; rho_max, z_max)",
                "status": "robust finite-box cross-check",
            },
            {
                "candidate": "core_hessian_curvature_width",
                "formula": "sigma_mu^curv(D) = c_sigma * ((2 kappa_rho(D)+kappa_z(D))/3)^(-1/4)",
                "status": "best current parent-side candidate after one reference calibration",
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    detail_out = outdir / "hll_sigma_mu_parent_candidate_detail.csv"
    summary_out = outdir / "hll_sigma_mu_parent_candidate_summary.csv"
    constants_out = outdir / "hll_sigma_mu_parent_candidate_constants.csv"
    meta_out = outdir / "hll_sigma_mu_parent_candidate_run_meta.json"

    out_detail.to_csv(detail_out, index=False)
    summary.to_csv(summary_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "overlap_detail_input": str(args.overlap_detail_input),
                "summary_input": str(args.summary_input),
                "outdir": str(outdir),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "core_h": args.core_h,
                "physical_params": {
                    "a": p.a,
                    "eps": p.eps,
                    "m0": p.m0,
                    "xi": p.xi,
                },
            },
            indent=2,
        )
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

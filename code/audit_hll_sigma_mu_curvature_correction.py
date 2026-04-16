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
            "Audit the EYMH-side bridge constant c_sigma and the first quartic "
            "curvature correction for sigma_mu."
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
        "--h2",
        type=float,
        default=1.0e-3,
        help="Finite-difference step for quadratic derivatives.",
    )
    parser.add_argument(
        "--h4",
        type=float,
        default=5.0e-3,
        help="Finite-difference step for quartic derivatives.",
    )
    return parser.parse_args()


def u_scalar(D: float, rho: float, z: float, p: PhysicalParams) -> float:
    return float(u_potential(np.array([[rho]]), np.array([[z]]), D, p)[0, 0])


def d2_rho(D: float, p: PhysicalParams, h: float) -> float:
    z0 = 0.5 * D
    return (u_scalar(D, h, z0, p) - 2.0 * u_scalar(D, 0.0, z0, p) + u_scalar(D, -h, z0, p)) / (h * h)


def d2_z(D: float, p: PhysicalParams, h: float) -> float:
    z0 = 0.5 * D
    return (u_scalar(D, 0.0, z0 + h, p) - 2.0 * u_scalar(D, 0.0, z0, p) + u_scalar(D, 0.0, z0 - h, p)) / (h * h)


def d4_rho(D: float, p: PhysicalParams, h: float) -> float:
    z0 = 0.5 * D
    return (
        u_scalar(D, -2.0 * h, z0, p)
        - 4.0 * u_scalar(D, -h, z0, p)
        + 6.0 * u_scalar(D, 0.0, z0, p)
        - 4.0 * u_scalar(D, h, z0, p)
        + u_scalar(D, 2.0 * h, z0, p)
    ) / (h**4)


def d4_z(D: float, p: PhysicalParams, h: float) -> float:
    z0 = 0.5 * D
    return (
        u_scalar(D, 0.0, z0 - 2.0 * h, p)
        - 4.0 * u_scalar(D, 0.0, z0 - h, p)
        + 6.0 * u_scalar(D, 0.0, z0, p)
        - 4.0 * u_scalar(D, 0.0, z0 + h, p)
        + u_scalar(D, 0.0, z0 + 2.0 * h, p)
    ) / (h**4)


def finite_box_amplitude(D: float, sigma: float, rho_max: float, z_margin: float) -> float:
    i_rho = math.pi * sigma * sigma * (1.0 - math.exp(-(rho_max * rho_max) / (sigma * sigma)))
    i_z = 0.5 * math.sqrt(math.pi) * sigma * (
        math.erf((D + z_margin) / sigma) + math.erf(z_margin / sigma)
    )
    return math.exp(-(D * D) / (4.0 * sigma * sigma)) / max(i_rho * i_z, 1.0e-300)


def rel_sup_error(target: np.ndarray, approx: np.ndarray) -> float:
    return float(np.max(np.abs(target - approx)) / max(float(np.max(np.abs(target))), 1.0e-300))


def main() -> None:
    args = parse_args()
    p = PhysicalParams()
    detail = pd.read_csv(args.overlap_detail_input).copy().sort_values("D").reset_index(drop=True)
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])

    rows: list[dict[str, float]] = []
    for _, row in detail.iterrows():
        D = float(row["D"])
        dr = float(row["dr"])
        dz = float(row["dz"])
        rho_max = float(row["rho_max"])
        z_max = float(row["z_max"])
        z_margin = z_max - D / 2.0
        sigma_exact = 0.5 * (float(row["sigma_l"]) + float(row["sigma_r"]))

        kappa_rho = d2_rho(D, p, args.h2)
        kappa_z = d2_z(D, p, args.h2)
        kappa_iso = (2.0 * kappa_rho + kappa_z) / 3.0

        lambda_rho = d4_rho(D, p, args.h4) / 24.0
        lambda_z = d4_z(D, p, args.h4) / 24.0
        lambda_iso = (2.0 * lambda_rho + lambda_z) / 3.0
        q4_iso = lambda_iso / (kappa_iso ** 1.5)
        c_sigma_exact = sigma_exact * (kappa_iso ** 0.25)

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
            sigma_l=sigma_exact,
            sigma_r=sigma_exact,
        )
        kernel = f_l * f_r

        rows.append(
            {
                "D": D,
                "rho_max": rho_max,
                "z_max": z_max,
                "z_margin": z_margin,
                "sigma_exact": sigma_exact,
                "kappa_rho": kappa_rho,
                "kappa_z": kappa_z,
                "kappa_iso": kappa_iso,
                "lambda_rho": lambda_rho,
                "lambda_z": lambda_z,
                "lambda_iso": lambda_iso,
                "q4_iso": q4_iso,
                "c_sigma_exact": c_sigma_exact,
                "core_anisotropy_ratio": float(kappa_rho / max(kappa_z, 1.0e-300)),
                "kernel_norm_max": float(np.max(np.abs(kernel))),
            }
        )

    out_detail = pd.DataFrame(rows)
    ref_idx = (out_detail["D"] - ref_d).abs().idxmin()
    kappa_rho_ref = d2_rho(ref_d, p, args.h2)
    kappa_z_ref = d2_z(ref_d, p, args.h2)
    kappa_iso_ref = (2.0 * kappa_rho_ref + kappa_z_ref) / 3.0
    lambda_rho_ref = d4_rho(ref_d, p, args.h4) / 24.0
    lambda_z_ref = d4_z(ref_d, p, args.h4) / 24.0
    lambda_iso_ref = (2.0 * lambda_rho_ref + lambda_z_ref) / 3.0
    q4_ref = lambda_iso_ref / (kappa_iso_ref ** 1.5)
    c_sigma_ref = float(2.5 * (kappa_iso_ref ** 0.25))

    out_detail["sigma_curv0"] = c_sigma_ref * np.power(out_detail["kappa_iso"], -0.25)
    out_detail["sigma_curv0_relerr"] = np.abs(out_detail["sigma_curv0"] / out_detail["sigma_exact"] - 1.0)

    X = out_detail["sigma_curv0"] * (out_detail["q4_iso"] - q4_ref)
    y = out_detail["sigma_exact"] - out_detail["sigma_curv0"]
    b4 = float(np.dot(X, y) / max(np.dot(X, X), 1.0e-300))

    out_detail["sigma_curv1"] = out_detail["sigma_curv0"] * (1.0 + b4 * (out_detail["q4_iso"] - q4_ref))
    out_detail["sigma_curv1_relerr"] = np.abs(out_detail["sigma_curv1"] / out_detail["sigma_exact"] - 1.0)
    ref_row = out_detail.loc[ref_idx]

    kernel_defect0 = []
    kernel_defect1 = []
    for _, row in detail.iterrows():
        D = float(row["D"])
        dr = float(row["dr"])
        dz = float(row["dz"])
        rho_max = float(row["rho_max"])
        z_max = float(row["z_max"])
        z_margin = z_max - D / 2.0
        sigma_exact = 0.5 * (float(row["sigma_l"]) + float(row["sigma_r"]))
        sigma0 = float(out_detail.loc[out_detail["D"] == D, "sigma_curv0"].iloc[0])
        sigma1 = float(out_detail.loc[out_detail["D"] == D, "sigma_curv1"].iloc[0])

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
            sigma_l=sigma_exact,
            sigma_r=sigma_exact,
        )
        kernel = f_l * f_r

        amp0 = finite_box_amplitude(D, sigma0, rho_max, z_margin)
        amp1 = finite_box_amplitude(D, sigma1, rho_max, z_margin)
        approx0 = amp0 * np.exp(-(rr * rr + zz * zz) / (sigma0 * sigma0))
        approx1 = amp1 * np.exp(-(rr * rr + zz * zz) / (sigma1 * sigma1))

        kernel_defect0.append(rel_sup_error(kernel, approx0))
        kernel_defect1.append(rel_sup_error(kernel, approx1))

    out_detail["kernel_rel_sup_curv0"] = kernel_defect0
    out_detail["kernel_rel_sup_curv1"] = kernel_defect1

    summary = pd.DataFrame(
        [
            {
                "section": "bridge_constant_invariance",
                "c_sigma_ref": c_sigma_ref,
                "c_sigma_mean": float(out_detail["c_sigma_exact"].mean()),
                "c_sigma_min": float(out_detail["c_sigma_exact"].min()),
                "c_sigma_max": float(out_detail["c_sigma_exact"].max()),
                "c_sigma_rel_span": float(
                    (out_detail["c_sigma_exact"].max() - out_detail["c_sigma_exact"].min())
                    / max(out_detail["c_sigma_exact"].mean(), 1.0e-300)
                ),
            },
            {
                "section": "quartic_invariant_profile",
                "q4_ref": q4_ref,
                "q4_mean": float(out_detail["q4_iso"].mean()),
                "q4_min": float(out_detail["q4_iso"].min()),
                "q4_max": float(out_detail["q4_iso"].max()),
                "q4_rel_span": float(
                    (out_detail["q4_iso"].max() - out_detail["q4_iso"].min())
                    / max(abs(out_detail["q4_iso"].mean()), 1.0e-300)
                ),
            },
            {
                "section": "uncorrected_curvature_candidate",
                "max_sigma_curv0_relerr": float(out_detail["sigma_curv0_relerr"].max()),
                "mean_sigma_curv0_relerr": float(out_detail["sigma_curv0_relerr"].mean()),
                "max_kernel_rel_sup_curv0": float(out_detail["kernel_rel_sup_curv0"].max()),
                "mean_kernel_rel_sup_curv0": float(out_detail["kernel_rel_sup_curv0"].mean()),
            },
            {
                "section": "first_quartic_correction",
                "b4_fit": b4,
                "max_sigma_curv1_relerr": float(out_detail["sigma_curv1_relerr"].max()),
                "mean_sigma_curv1_relerr": float(out_detail["sigma_curv1_relerr"].mean()),
                "max_kernel_rel_sup_curv1": float(out_detail["kernel_rel_sup_curv1"].max()),
                "mean_kernel_rel_sup_curv1": float(out_detail["kernel_rel_sup_curv1"].mean()),
                "improvement_factor_max_kernel": float(
                    out_detail["kernel_rel_sup_curv0"].max() / max(out_detail["kernel_rel_sup_curv1"].max(), 1.0e-300)
                ),
            },
            {
                "section": "reference_constants",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "sigma_exact_ref": float(ref_row["sigma_exact"]),
                "sigma_curv0_ref": float(c_sigma_ref * (kappa_iso_ref ** -0.25)),
                "sigma_curv1_ref": float(c_sigma_ref * (kappa_iso_ref ** -0.25)),
                "kappa_iso_ref": float(kappa_iso_ref),
                "lambda_iso_ref": float(lambda_iso_ref),
                "nearest_detail_D": float(ref_row["D"]),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "d_invariant_bridge_constant",
                "formula": "c_sigma(D)=sigma_exact(D) * kappa_iso(D)^(1/4)",
                "status": "empirically almost constant on the audited D-window",
            },
            {
                "candidate": "quartic_invariant",
                "formula": "q4_iso(D)=lambda_iso(D)/kappa_iso(D)^(3/2)",
                "status": "first curvature-controlled correction coordinate",
            },
            {
                "candidate": "corrected_width",
                "formula": "sigma_mu^(curv,4)(D)=c_sigma,* kappa_iso(D)^(-1/4) [1+b4 (q4_iso(D)-q4_iso(D_*))]",
                "status": "best current EYMH-side corrected width candidate",
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    detail_out = outdir / "hll_sigma_mu_curvature_correction_detail.csv"
    summary_out = outdir / "hll_sigma_mu_curvature_correction_summary.csv"
    constants_out = outdir / "hll_sigma_mu_curvature_correction_constants.csv"
    meta_out = outdir / "hll_sigma_mu_curvature_correction_run_meta.json"

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
                "h2": args.h2,
                "h4": args.h4,
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

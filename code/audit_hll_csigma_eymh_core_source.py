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

from audit_hll_sigma_mu_parent_candidate import core_hessian_curvatures  # noqa: E402
from extract_chi_localized_2d import PhysicalParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the bridge constant c_sigma is already fixed by the "
            "local EYMH core jet, with the second center entering only as a "
            "small analytic mirror correction."
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
    parser.add_argument("--core-h", type=float, default=1.0e-3)
    return parser.parse_args()


def self_core_kappa_iso(p: PhysicalParams) -> float:
    a = float(p.a)
    eps = float(p.eps)
    m0 = float(p.m0)
    coeff = 1.0 - 6.0 * float(p.xi)

    omega0 = 1.0 + a / eps
    omega2 = -a / (2.0 * eps**3)
    lap0 = -3.0 * a / eps**3
    lap2 = 15.0 * a / (2.0 * eps**5)

    u2 = m0 * m0 * (2.0 * omega0 * omega2) + coeff * (lap2 / omega0 - lap0 * omega2 / (omega0 * omega0))
    return float(2.0 * u2)


def two_center_jet_coefficients(D: float, p: PhysicalParams) -> dict[str, float]:
    a = float(p.a)
    eps = float(p.eps)
    m0 = float(p.m0)
    coeff = 1.0 - 6.0 * float(p.xi)
    r2 = D * D + eps * eps
    r = math.sqrt(r2)

    s0 = 1.0 / r
    s1 = -D / (r**3)
    s_rho2 = -1.0 / (2.0 * r**3)
    s_z2 = (2.0 * D * D - eps * eps) / (2.0 * r**5)

    omega0 = 1.0 + a / eps + a * s0
    omega1 = a * s1
    omega_rho2 = -a / (2.0 * eps**3) + a * s_rho2
    omega_z2 = -a / (2.0 * eps**3) + a * s_z2

    lap0 = -3.0 * a / eps**3 - 3.0 * a * eps * eps / (r**5)
    lap1 = 15.0 * a * eps * eps * D / (r**7)
    lap_rho2 = 15.0 * a / (2.0 * eps**5) + 15.0 * a * eps * eps / (2.0 * r**7)
    lap_z2 = (
        15.0 * a / (2.0 * eps**5)
        + 15.0 * a * eps * eps / (2.0 * r**7)
        - 105.0 * a * eps * eps * D * D / (2.0 * r**9)
    )

    u_rho2 = m0 * m0 * (2.0 * omega0 * omega_rho2) + coeff * (
        lap_rho2 / omega0 - lap0 * omega_rho2 / (omega0 * omega0)
    )
    u_z2 = m0 * m0 * (2.0 * omega0 * omega_z2 + omega1 * omega1) + coeff * (
        lap_z2 / omega0
        - lap0 * omega_z2 / (omega0 * omega0)
        - lap1 * omega1 / (omega0 * omega0)
        + lap0 * omega1 * omega1 / (omega0**3)
    )

    kappa_rho = float(2.0 * u_rho2)
    kappa_z = float(2.0 * u_z2)
    kappa_iso = float((2.0 * kappa_rho + kappa_z) / 3.0)

    return {
        "omega0": float(omega0),
        "omega1": float(omega1),
        "omega_rho2": float(omega_rho2),
        "omega_z2": float(omega_z2),
        "lap0": float(lap0),
        "lap1": float(lap1),
        "lap_rho2": float(lap_rho2),
        "lap_z2": float(lap_z2),
        "kappa_rho_analytic": kappa_rho,
        "kappa_z_analytic": kappa_z,
        "kappa_iso_analytic": kappa_iso,
    }


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.overlap_detail_input).copy().sort_values("D").reset_index(drop=True)
    summary_row = pd.read_csv(args.summary_input).iloc[0]
    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])
    p = PhysicalParams()

    sigma_exact = 0.5 * (float(detail["sigma_l"].iloc[0]) + float(detail["sigma_r"].iloc[0]))
    kappa_self = self_core_kappa_iso(p)
    c_sigma_self = float(sigma_exact * (kappa_self ** 0.25))

    rows: list[dict[str, float]] = []
    for _, row in detail.iterrows():
        D = float(row["D"])
        sigma_mu = 0.5 * (float(row["sigma_l"]) + float(row["sigma_r"]))
        kappa_rho_fd, kappa_z_fd = core_hessian_curvatures(D, p, args.core_h)
        kappa_iso_fd = (2.0 * kappa_rho_fd + kappa_z_fd) / 3.0
        c_sigma_exact = float(sigma_mu * (kappa_iso_fd ** 0.25))

        analytic = two_center_jet_coefficients(D, p)
        kappa_rho_analytic = float(analytic["kappa_rho_analytic"])
        kappa_z_analytic = float(analytic["kappa_z_analytic"])
        kappa_iso_analytic = float(analytic["kappa_iso_analytic"])
        c_sigma_analytic = float(sigma_mu * (kappa_iso_analytic ** 0.25))

        delta_kappa_mirror = float(kappa_iso_analytic - kappa_self)
        mirror_frac = float(delta_kappa_mirror / max(kappa_self, 1.0e-300))

        rows.append(
            {
                "D": D,
                "sigma_mu": sigma_mu,
                "kappa_rho_fd": float(kappa_rho_fd),
                "kappa_z_fd": float(kappa_z_fd),
                "kappa_iso_fd": float(kappa_iso_fd),
                "kappa_rho_analytic": kappa_rho_analytic,
                "kappa_z_analytic": kappa_z_analytic,
                "kappa_iso_analytic": kappa_iso_analytic,
                "relerr_kappa_rho_analytic": float(abs(kappa_rho_analytic / max(kappa_rho_fd, 1.0e-300) - 1.0)),
                "relerr_kappa_z_analytic": float(abs(kappa_z_analytic / max(kappa_z_fd, 1.0e-300) - 1.0)),
                "relerr_kappa_iso_analytic": float(abs(kappa_iso_analytic / max(kappa_iso_fd, 1.0e-300) - 1.0)),
                "c_sigma_exact": c_sigma_exact,
                "c_sigma_analytic": c_sigma_analytic,
                "relerr_csigma_analytic": float(abs(c_sigma_analytic / max(c_sigma_exact, 1.0e-300) - 1.0)),
                "c_sigma_self": c_sigma_self,
                "relerr_csigma_self": float(abs(c_sigma_self / max(c_sigma_exact, 1.0e-300) - 1.0)),
                "delta_kappa_mirror": delta_kappa_mirror,
                "mirror_fraction": mirror_frac,
            }
        )

    out_detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    nearest = out_detail.loc[(out_detail["D"] - ref_d).abs().idxmin()]

    summary = pd.DataFrame(
        [
            {
                "section": "self_core_source",
                "sigma_mu": sigma_exact,
                "kappa_self": kappa_self,
                "c_sigma_self": c_sigma_self,
                "max_relerr_csigma_self": float(out_detail["relerr_csigma_self"].max()),
                "mean_relerr_csigma_self": float(out_detail["relerr_csigma_self"].mean()),
            },
            {
                "section": "full_two_center_analytic_jet",
                "max_relerr_kappa_rho_analytic": float(out_detail["relerr_kappa_rho_analytic"].max()),
                "max_relerr_kappa_z_analytic": float(out_detail["relerr_kappa_z_analytic"].max()),
                "max_relerr_kappa_iso_analytic": float(out_detail["relerr_kappa_iso_analytic"].max()),
                "mean_relerr_kappa_iso_analytic": float(out_detail["relerr_kappa_iso_analytic"].mean()),
                "max_relerr_csigma_analytic": float(out_detail["relerr_csigma_analytic"].max()),
                "mean_relerr_csigma_analytic": float(out_detail["relerr_csigma_analytic"].mean()),
            },
            {
                "section": "mirror_correction_scale",
                "mirror_fraction_min": float(out_detail["mirror_fraction"].min()),
                "mirror_fraction_max": float(out_detail["mirror_fraction"].max()),
                "mirror_fraction_mean": float(out_detail["mirror_fraction"].mean()),
                "max_abs_delta_kappa_mirror": float(np.max(np.abs(out_detail["delta_kappa_mirror"]))),
            },
            {
                "section": "reference_point",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "nearest_detail_D": float(nearest["D"]),
                "c_sigma_exact_ref": float(nearest["c_sigma_exact"]),
                "c_sigma_analytic_ref": float(nearest["c_sigma_analytic"]),
                "c_sigma_self": c_sigma_self,
                "mirror_fraction_ref": float(nearest["mirror_fraction"]),
                "kappa_iso_fd_ref": float(nearest["kappa_iso_fd"]),
                "kappa_iso_analytic_ref": float(nearest["kappa_iso_analytic"]),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "candidate": "isolated_plummer_core_source",
                "formula": (
                    "c_sigma^(self)=sigma_mu * kappa_self^(1/4), "
                    "kappa_self from the one-center EYMH core jet at r=0"
                ),
                "status": "D-independent dominant source candidate",
                "value": c_sigma_self,
            },
            {
                "candidate": "two_center_analytic_core_jet",
                "formula": (
                    "c_sigma^(analytic)(D)=sigma_mu * kappa_iso^(analytic)(D)^(1/4), "
                    "with kappa_iso^(analytic) from the full local two-center EYMH jet"
                ),
                "status": "analytic D-dependent refinement of the same core source",
                "value_ref": float(nearest["c_sigma_analytic"]),
            },
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_csigma_eymh_core_source_summary.csv"
    detail_out = outdir / "hll_csigma_eymh_core_source_detail.csv"
    constants_out = outdir / "hll_csigma_eymh_core_source_constants.csv"
    meta_out = outdir / "hll_csigma_eymh_core_source_run_meta.json"

    summary.to_csv(summary_out, index=False)
    out_detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "overlap_detail_input": str(args.overlap_detail_input),
                "summary_input": str(args.summary_input),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "core_h": float(args.core_h),
                "physical_params": {
                    "a": float(p.a),
                    "eps": float(p.eps),
                    "m0": float(p.m0),
                    "xi": float(p.xi),
                },
            },
            indent=2,
        )
    )

    print(summary_out)
    print(detail_out)
    print(constants_out)
    print(meta_out)


if __name__ == "__main__":
    main()

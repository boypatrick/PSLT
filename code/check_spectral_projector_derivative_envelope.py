#!/usr/bin/env python3
"""Analytic ||H'(D)|| envelope for the two-center finite-volume operator.

T2 scope: derive and audit the potential-derivative part of the operator family
from the explicit two-center conformal factor.  On a fixed finite-volume box (or
after a pullback to a fixed reference box with kinetic terms handled separately),
the localized G1 operator has the form

    K(D) = K_0 + diag(rho * U(D)),      M = diag(rho),

so the self-adjoint representative satisfies

    d/dD [M^{-1/2} K(D) M^{-1/2}] = diag(d_D U(D)).

Therefore ||H'(D)|| <= sup |d_D U| on the chosen grid, and the continuous
closed-form envelope follows from elementary bounds on the regularized source.

This script does not claim an inter-knot no-crossing theorem by itself; it
records a rigorous/sampled envelope and validates the analytic derivative
against finite differences.  The existing G1 artifacts use D-dependent z boxes,
so the fully literal G1 matrix family still needs either a fixed-box rerun or a
coordinate-pullback treatment for boundary/kinetic terms.  This is why T2 is an
envelope certificate rather than a full continuum transport proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extract_chi_localized_2d import Level, PhysicalParams, u_potential

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECTRUM = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_spectrum_Dgrid21.csv"
DEFAULT_TRANSPORT = ROOT / "output" / "spectral_transport" / "spectral_projector_transport_summary.csv"
OUTDIR = ROOT / "output" / "spectral_transport"
SUMMARY_CSV = OUTDIR / "spectral_projector_derivative_envelope_summary.csv"
DETAIL_CSV = OUTDIR / "spectral_projector_derivative_envelope_detail.csv"
DETAIL_JSON = OUTDIR / "spectral_projector_derivative_envelope_detail.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def omega_2center(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    sp = rho * rho + (z - D / 2.0) ** 2 + p.eps * p.eps
    sm = rho * rho + (z + D / 2.0) ** 2 + p.eps * p.eps
    return 1.0 + p.a * (sp ** (-0.5) + sm ** (-0.5))


def lap_omega_2center(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    sp = rho * rho + (z - D / 2.0) ** 2 + p.eps * p.eps
    sm = rho * rho + (z + D / 2.0) ** 2 + p.eps * p.eps
    return -3.0 * p.a * p.eps * p.eps * (sp ** (-2.5) + sm ** (-2.5))


def dD_omega_2center(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    qp = z - D / 2.0
    qm = z + D / 2.0
    sp = rho * rho + qp * qp + p.eps * p.eps
    sm = rho * rho + qm * qm + p.eps * p.eps
    return 0.5 * p.a * (qp * sp ** (-1.5) - qm * sm ** (-1.5))


def dD_lap_omega_2center(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    qp = z - D / 2.0
    qm = z + D / 2.0
    sp = rho * rho + qp * qp + p.eps * p.eps
    sm = rho * rho + qm * qm + p.eps * p.eps
    return -7.5 * p.a * p.eps * p.eps * (qp * sp ** (-3.5) - qm * sm ** (-3.5))


def dD_u_potential(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    om = omega_2center(rho, z, D, p)
    lap = lap_omega_2center(rho, z, D, p)
    dom = dD_omega_2center(rho, z, D, p)
    dlap = dD_lap_omega_2center(rho, z, D, p)
    c = 1.0 - 6.0 * p.xi
    return 2.0 * p.m0 * p.m0 * om * dom + c * (dlap * om - lap * dom) / (om * om)


def closed_form_envelope(p: PhysicalParams) -> dict[str, float]:
    eps = float(p.eps)
    a = float(p.a)
    c = abs(1.0 - 6.0 * float(p.xi))
    omega_min = 1.0
    omega_max = 1.0 + 2.0 * a / eps

    # sup_x x/(x^2+eps^2)^(3/2) = 2/(3 sqrt(3) eps^2)
    m3 = 2.0 / (3.0 * math.sqrt(3.0) * eps * eps)
    # sup_x x/(x^2+eps^2)^(7/2) = 6^3/(7^(7/2) eps^6)
    m7 = 216.0 / ((7.0 ** 3.5) * (eps ** 6))

    omega_D_max = a * m3
    lap_max = 6.0 * a / (eps ** 3)
    lap_D_max = 15.0 * a * eps * eps * m7
    u_D_bound = (
        2.0 * p.m0 * p.m0 * omega_max * omega_D_max
        + c * (lap_D_max * omega_max + lap_max * omega_D_max) / (omega_min * omega_min)
    )
    return {
        "omega_min_bound": omega_min,
        "omega_max_bound": omega_max,
        "sup_abs_dD_omega_bound": omega_D_max,
        "sup_abs_lap_omega_bound": lap_max,
        "sup_abs_dD_lap_omega_bound": lap_D_max,
        "closed_form_sup_abs_dD_U_bound": u_D_bound,
    }


def make_grid(D: float, rho_max: float, z_max: float, level: Level) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nr = int(round(rho_max / level.dr))
    nz = int(round(2.0 * z_max / level.dz))
    rho = (np.arange(nr) + 0.5) * level.dr
    z = -z_max + (np.arange(nz) + 0.5) * level.dz
    rr, zz = np.meshgrid(rho, z, indexing="ij")
    return rho, z, rr, zz


def parse_d_values(spectrum_csv: Path, level_name: str) -> list[float]:
    spec = pd.read_csv(spectrum_csv)
    spec = spec[spec["level"].astype(str) == str(level_name)]
    if spec.empty:
        raise RuntimeError(f"no rows for level={level_name} in {spectrum_csv}")
    return [float(x) for x in sorted(spec["D"].unique())]


def load_transport_summary(path: Path) -> dict[str, float]:
    row = pd.read_csv(path).iloc[0].to_dict()
    out: dict[str, float] = {}
    for key, val in row.items():
        try:
            out[key] = float(val)
        except Exception:
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--transport-summary", type=Path, default=DEFAULT_TRANSPORT)
    parser.add_argument("--level", default="fine")
    parser.add_argument("--rho-max", type=float, default=3.0)
    parser.add_argument("--z-margin", type=float, default=6.0)
    parser.add_argument("--fd-h", type=float, default=1.0e-4)
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--detail", type=Path, default=DETAIL_CSV)
    parser.add_argument("--detail-json", type=Path, default=DETAIL_JSON)
    args = parser.parse_args()

    spectrum_csv = args.spectrum_csv if args.spectrum_csv.is_absolute() else ROOT / args.spectrum_csv
    transport_csv = args.transport_summary if args.transport_summary.is_absolute() else ROOT / args.transport_summary
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    detail_path = args.detail if args.detail.is_absolute() else ROOT / args.detail
    json_path = args.detail_json if args.detail_json.is_absolute() else ROOT / args.detail_json

    if args.level != "fine":
        raise RuntimeError("T2 currently records the fine-grid release-aligned envelope only.")
    level = Level("fine", dr=0.06, dz=0.03)
    p = PhysicalParams()
    d_values = parse_d_values(spectrum_csv, args.level)
    d_min = min(d_values)
    d_max = max(d_values)
    fixed_z_max = d_max / 2.0 + float(args.z_margin)
    cf = closed_form_envelope(p)

    detail_rows: list[dict[str, Any]] = []
    max_grid_fixed = 0.0
    max_grid_varying = 0.0
    max_fd_err = 0.0
    max_fd_rel = 0.0

    for D in d_values:
        for policy, z_max in [("fixed_box", fixed_z_max), ("g1_varying_box", D / 2.0 + float(args.z_margin))]:
            _rho, _z, rr, zz = make_grid(float(D), float(args.rho_max), float(z_max), level)
            dU = dD_u_potential(rr, zz, float(D), p)
            grid_norm = float(np.max(np.abs(dU)))
            U_plus = u_potential(rr, zz, float(D) + float(args.fd_h), p)
            U_minus = u_potential(rr, zz, float(D) - float(args.fd_h), p)
            fd = (U_plus - U_minus) / (2.0 * float(args.fd_h))
            fd_err = float(np.max(np.abs(fd - dU)))
            fd_rel = float(fd_err / max(grid_norm, 1.0e-300))
            row = {
                "D": float(D),
                "policy": policy,
                "rho_max": float(args.rho_max),
                "z_max": float(z_max),
                "dr": float(level.dr),
                "dz": float(level.dz),
                "grid_sup_abs_dD_U": grid_norm,
                "closed_form_sup_abs_dD_U_bound": cf["closed_form_sup_abs_dD_U_bound"],
                "closed_form_over_grid_ratio": float(cf["closed_form_sup_abs_dD_U_bound"] / max(grid_norm, 1.0e-300)),
                "fd_h": float(args.fd_h),
                "fd_max_abs_error": fd_err,
                "fd_max_rel_error_vs_grid_sup": fd_rel,
                "closed_form_dominates_grid": bool(cf["closed_form_sup_abs_dD_U_bound"] >= grid_norm),
            }
            detail_rows.append(row)
            if policy == "fixed_box":
                max_grid_fixed = max(max_grid_fixed, grid_norm)
            else:
                max_grid_varying = max(max_grid_varying, grid_norm)
            max_fd_err = max(max_fd_err, fd_err)
            max_fd_rel = max(max_fd_rel, fd_rel)

    t1 = load_transport_summary(transport_csv)
    alpha = float(t1.get("contour_fraction", 0.45))
    gap1 = float(t1["min_nearest_gap_N1"])
    gap2 = float(t1["min_nearest_gap_N2"])
    gap3 = float(t1["min_nearest_gap_N3"])
    closed_bound = float(cf["closed_form_sup_abs_dD_U_bound"])

    proj_bound_grid_n1 = max_grid_fixed / (alpha * gap1)
    proj_bound_grid_n2 = max_grid_fixed / (alpha * gap2)
    proj_bound_grid_n3 = max_grid_fixed / (alpha * gap3)
    proj_bound_closed_n1 = closed_bound / (alpha * gap1)
    proj_bound_closed_n2 = closed_bound / (alpha * gap2)
    proj_bound_closed_n3 = closed_bound / (alpha * gap3)

    observed_slope_n1 = float(t1["max_abs_lambda_slope_N1"])
    observed_slope_n2 = float(t1["max_abs_lambda_slope_N2"])
    observed_slope_n3 = float(t1["max_abs_lambda_slope_N3"])

    closed_dominates_all_grid = bool(all(bool(r["closed_form_dominates_grid"]) for r in detail_rows))
    fd_pass = bool(max_fd_rel <= 1.0e-5 or max_fd_err <= 1.0e-6)
    verdict = (
        "T2_ANALYTIC_HPRIME_ENVELOPE_VERIFIED_SUPNORM_LOOSE"
        if closed_dominates_all_grid and fd_pass
        else "T2_DERIVATIVE_ENVELOPE_DIAGNOSTIC_ONLY"
    )

    summary_row = {
        "gate": "T2",
        "verdict": verdict,
        "level": args.level,
        "D_min": float(d_min),
        "D_max": float(d_max),
        "n_D": int(len(d_values)),
        "rho_max": float(args.rho_max),
        "fixed_z_max": float(fixed_z_max),
        "z_margin_for_varying_box": float(args.z_margin),
        "dr": float(level.dr),
        "dz": float(level.dz),
        "fd_h": float(args.fd_h),
        **cf,
        "max_grid_sup_abs_dD_U_fixed_box": float(max_grid_fixed),
        "max_grid_sup_abs_dD_U_g1_varying_box": float(max_grid_varying),
        "closed_form_over_fixed_grid_ratio": float(closed_bound / max(max_grid_fixed, 1.0e-300)),
        "closed_form_over_varying_grid_ratio": float(closed_bound / max(max_grid_varying, 1.0e-300)),
        "max_fd_abs_error": float(max_fd_err),
        "max_fd_rel_error_vs_grid_sup": float(max_fd_rel),
        "closed_form_dominates_all_grid_rows": closed_dominates_all_grid,
        "fd_derivative_check_pass": fd_pass,
        "contour_fraction_alpha": alpha,
        "min_gap_N1": gap1,
        "min_gap_N2": gap2,
        "min_gap_N3": gap3,
        "projector_derivative_bound_fixed_grid_N1": float(proj_bound_grid_n1),
        "projector_derivative_bound_fixed_grid_N2": float(proj_bound_grid_n2),
        "projector_derivative_bound_fixed_grid_N3": float(proj_bound_grid_n3),
        "projector_derivative_bound_closed_form_N1": float(proj_bound_closed_n1),
        "projector_derivative_bound_closed_form_N2": float(proj_bound_closed_n2),
        "projector_derivative_bound_closed_form_N3": float(proj_bound_closed_n3),
        "observed_max_abs_lambda_slope_N1": observed_slope_n1,
        "observed_max_abs_lambda_slope_N2": observed_slope_n2,
        "observed_max_abs_lambda_slope_N3": observed_slope_n3,
        "observed_slope_over_fixed_grid_bound_N1": float(observed_slope_n1 / max(max_grid_fixed, 1.0e-300)),
        "observed_slope_over_fixed_grid_bound_N2": float(observed_slope_n2 / max(max_grid_fixed, 1.0e-300)),
        "observed_slope_over_fixed_grid_bound_N3": float(observed_slope_n3 / max(max_grid_fixed, 1.0e-300)),
        "supnorm_envelope_sufficient_for_no_crossing": False,
        "reason_not_sufficient": "core singular envelope is rigorous but too loose; use T3/Davis-Kahan or weighted low-mode matrix elements for inter-knot promotion",
    }

    write_csv(summary_path, [summary_row])
    write_csv(detail_path, detail_rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(
            {
                "claim_firewall": {
                    "baseline_change": False,
                    "main_tex_change": False,
                    "continuum_no_crossing_claimed": False,
                    "fixed_box_reading": "On a fixed finite-volume box, M is D-independent and H'(D) is multiplication by d_D U.",
                    "g1_varying_box_caveat": "The literal G1 artifact uses z_max=D/2+margin; boundary/coordinate-pullback terms are not included in this T2 envelope.",
                },
                "closed_form_derivation_constants": cf,
                "summary": summary_row,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(verdict)
    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    print(f"detail_json={json_path}")
    print(f"max_grid_sup_abs_dD_U_fixed_box={max_grid_fixed:.12g}")
    print(f"closed_form_sup_abs_dD_U_bound={closed_bound:.12g}")
    print(f"max_fd_rel_error={max_fd_rel:.12g}")
    print(
        "projector_bound_fixed_grid_N=(%.12g, %.12g, %.12g)"
        % (proj_bound_grid_n1, proj_bound_grid_n2, proj_bound_grid_n3)
    )


if __name__ == "__main__":
    main()

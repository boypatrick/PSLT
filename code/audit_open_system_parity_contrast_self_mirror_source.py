#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

from extract_chi_localized_2d import (
    Level,
    PhysicalParams,
    build_generalized_operator,
    normalize_modes,
)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
DETAIL = OUTDIR / "chi_open_system_parity_contrast_self_mirror_source_detail.csv"
SUMMARY = OUTDIR / "chi_open_system_parity_contrast_self_mirror_source_summary.csv"


def _radius_sq(rho: np.ndarray, z: np.ndarray, center: float) -> np.ndarray:
    return rho * rho + (z - center) ** 2


def omega_single(rho: np.ndarray, z: np.ndarray, center: float, p: PhysicalParams) -> np.ndarray:
    r2 = _radius_sq(rho, z, center)
    return 1.0 + p.a / np.sqrt(r2 + p.eps * p.eps)


def lap_omega_single(rho: np.ndarray, z: np.ndarray, center: float, p: PhysicalParams) -> np.ndarray:
    r2 = _radius_sq(rho, z, center)
    return -3.0 * p.a * p.eps * p.eps * (r2 + p.eps * p.eps) ** (-2.5)


def u_single(rho: np.ndarray, z: np.ndarray, center: float, p: PhysicalParams) -> np.ndarray:
    om = omega_single(rho, z, center, p)
    lap = lap_omega_single(rho, z, center, p)
    return p.m0 * p.m0 * (om * om - 1.0) + (1.0 - 6.0 * p.xi) * (lap / om)


def u_two_center(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    om_plus = omega_single(rho, z, D / 2.0, p)
    om_minus = omega_single(rho, z, -D / 2.0, p)
    lap_plus = lap_omega_single(rho, z, D / 2.0, p)
    lap_minus = lap_omega_single(rho, z, -D / 2.0, p)
    om = om_plus + om_minus - 1.0
    lap = lap_plus + lap_minus
    return p.m0 * p.m0 * (om * om - 1.0) + (1.0 - 6.0 * p.xi) * (lap / om)


def u_cross_mass(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    s_plus = omega_single(rho, z, D / 2.0, p) - 1.0
    s_minus = omega_single(rho, z, -D / 2.0, p) - 1.0
    return 2.0 * p.m0 * p.m0 * s_plus * s_minus


def u_cross_deriv(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    om_plus = omega_single(rho, z, D / 2.0, p)
    om_minus = omega_single(rho, z, -D / 2.0, p)
    lap_plus = lap_omega_single(rho, z, D / 2.0, p)
    lap_minus = lap_omega_single(rho, z, -D / 2.0, p)
    om_two = om_plus + om_minus - 1.0
    coeff = 1.0 - 6.0 * p.xi
    return coeff * ((lap_plus + lap_minus) / om_two - lap_plus / om_plus - lap_minus / om_minus)


def spherical_average_generic(
    r_vals: np.ndarray,
    D: float,
    p: PhysicalParams,
    field_fn: Callable[[np.ndarray, np.ndarray, float, PhysicalParams], np.ndarray],
    n_mu: int,
) -> np.ndarray:
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    rr = r_vals[:, None]
    rho = rr * np.sqrt(np.maximum(0.0, 1.0 - mu * mu))[None, :]
    z = rr * mu[None, :]
    vv = field_fn(rho, z, D, p)
    return 0.5 * (vv @ w)


def solve_fields(D: float, p: PhysicalParams, level: Level, rho_max: float, z_margin: float, sigma: float, tol: float, maxiter: int):
    z_max = D / 2.0 + z_margin
    rho, z, rr, zz, uu_two, k_mat, m_mat = build_generalized_operator(
        D=D,
        p=p,
        rho_max=rho_max,
        z_max=z_max,
        dr=level.dr,
        dz=level.dz,
    )
    evals, evecs = eigsh(k_mat, k=2, M=m_mat, sigma=sigma, which="LM", tol=tol, maxiter=maxiter)
    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    psi = np.stack([evecs[:, 0].reshape(len(rho), len(z)), evecs[:, 1].reshape(len(rho), len(z))], axis=2)
    psi = normalize_modes(psi, rho, level.dr, level.dz)
    psi1 = psi[:, :, 0]
    psi2 = psi[:, :, 1]
    w = 2.0 * math.pi * rho[:, None] * level.dr * level.dz
    contrast_density = w * (psi1 * psi1 - psi2 * psi2)
    r_field = np.sqrt(rr * rr + zz * zz)
    r_line = np.linspace(0.0, float(r_field.max()), 1800)

    u_plus = u_single(rr, zz, D / 2.0, p)
    u_minus = u_single(rr, zz, -D / 2.0, p)
    u_cross_m = u_cross_mass(rr, zz, D, p)
    u_cross_d = u_cross_deriv(rr, zz, D, p)
    u_cross = u_cross_m + u_cross_d
    u_two_check = u_plus + u_minus + u_cross

    ubar_two = spherical_average_generic(r_line, D, p, u_two_center, n_mu=120)
    ubar_plus = spherical_average_generic(r_line, D, p, lambda rho, z, D, p: u_single(rho, z, D / 2.0, p), n_mu=120)
    ubar_minus = spherical_average_generic(r_line, D, p, lambda rho, z, D, p: u_single(rho, z, -D / 2.0, p), n_mu=120)
    ubar_cross_m = spherical_average_generic(r_line, D, p, u_cross_mass, n_mu=120)
    ubar_cross_d = spherical_average_generic(r_line, D, p, u_cross_deriv, n_mu=120)
    ubar_cross = ubar_cross_m + ubar_cross_d

    interp = lambda arr: np.interp(r_field.ravel(), r_line, arr).reshape(r_field.shape)
    delta_two = uu_two - interp(ubar_two)
    delta_plus = u_plus - interp(ubar_plus)
    delta_minus = u_minus - interp(ubar_minus)
    delta_cross_m = u_cross_m - interp(ubar_cross_m)
    delta_cross_d = u_cross_d - interp(ubar_cross_d)
    delta_cross = u_cross - interp(ubar_cross)

    return {
        "rho": rho,
        "z": z,
        "rr": rr,
        "zz": zz,
        "r_field": r_field,
        "psi1": psi1,
        "psi2": psi2,
        "contrast_density": contrast_density,
        "uu_two": uu_two,
        "u_two_check": u_two_check,
        "u_plus": u_plus,
        "u_minus": u_minus,
        "u_cross_m": u_cross_m,
        "u_cross_d": u_cross_d,
        "u_cross": u_cross,
        "delta_two": delta_two,
        "delta_plus": delta_plus,
        "delta_minus": delta_minus,
        "delta_cross_m": delta_cross_m,
        "delta_cross_d": delta_cross_d,
        "delta_cross": delta_cross,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", default="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    args = ap.parse_args()

    Ds = [float(x.strip()) for x in args.Ds.split(",") if x.strip()]
    p = PhysicalParams()
    level = Level("fine", dr=float(args.dr), dz=float(args.dz))

    detail_rows = []
    for D in Ds:
        fields = solve_fields(
            D=D,
            p=p,
            level=level,
            rho_max=float(args.rho_max),
            z_margin=float(args.z_margin),
            sigma=float(args.sigma),
            tol=float(args.tol),
            maxiter=int(args.maxiter),
        )
        contrast = fields["contrast_density"]
        i_exact = float(np.sum(contrast * fields["delta_two"]))
        i_plus = float(np.sum(contrast * fields["delta_plus"]))
        i_minus = float(np.sum(contrast * fields["delta_minus"]))
        i_cross_m = float(np.sum(contrast * fields["delta_cross_m"]))
        i_cross_d = float(np.sum(contrast * fields["delta_cross_d"]))
        i_cross = float(np.sum(contrast * fields["delta_cross"]))
        i_pair = i_plus + i_minus

        detail_rows.append(
            {
                "D": D,
                "exact": i_exact,
                "I_plus": i_plus,
                "I_minus": i_minus,
                "I_self_pair": i_pair,
                "I_cross_mass": i_cross_m,
                "I_cross_deriv": i_cross_d,
                "I_cross_total": i_cross,
                "self_pair_over_exact": i_pair / i_exact,
                "cross_mass_over_exact": i_cross_m / i_exact,
                "cross_deriv_over_exact": i_cross_d / i_exact,
                "cross_total_over_exact": i_cross / i_exact,
                "self_pair_abs_share": abs(i_pair) / max(abs(i_exact), 1e-300),
                "cross_mass_abs_share": abs(i_cross_m) / max(abs(i_exact), 1e-300),
                "cross_deriv_abs_share": abs(i_cross_d) / max(abs(i_exact), 1e-300),
                "cross_total_abs_share": abs(i_cross) / max(abs(i_exact), 1e-300),
                "self_pair_same_sign_as_exact": int(i_pair == 0.0 or np.sign(i_pair) == np.sign(i_exact)),
                "cross_total_same_sign_as_exact": int(i_cross == 0.0 or np.sign(i_cross) == np.sign(i_exact)),
                "plus_minus_sym_resid_abs": abs(i_plus - i_minus),
                "reconstruction_resid_abs": abs(i_exact - (i_pair + i_cross)),
                "u_split_resid_abs": float(np.max(np.abs(fields["uu_two"] - fields["u_two_check"]))),
                "delta_split_resid_abs": float(
                    np.max(np.abs(fields["delta_two"] - (fields["delta_plus"] + fields["delta_minus"] + fields["delta_cross"])))
                ),
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values("D").reset_index(drop=True)
    exact = detail["exact"].astype(float)
    floor_idx = int(exact.abs().idxmin())
    summary = pd.DataFrame(
        [
            {
                "max_u_split_resid_abs": float(detail["u_split_resid_abs"].max()),
                "max_delta_split_resid_abs": float(detail["delta_split_resid_abs"].max()),
                "max_plus_minus_sym_resid_abs": float(detail["plus_minus_sym_resid_abs"].max()),
                "max_reconstruction_resid_abs": float(detail["reconstruction_resid_abs"].max()),
                "exact_floor_abs": float(abs(detail.loc[floor_idx, "exact"])),
                "exact_floor_D": float(detail.loc[floor_idx, "D"]),
                "self_pair_at_floor": float(detail.loc[floor_idx, "I_self_pair"]),
                "cross_total_at_floor": float(detail.loc[floor_idx, "I_cross_total"]),
                "cross_mass_at_floor": float(detail.loc[floor_idx, "I_cross_mass"]),
                "cross_deriv_at_floor": float(detail.loc[floor_idx, "I_cross_deriv"]),
                "self_pair_over_exact_at_floor": float(detail.loc[floor_idx, "self_pair_over_exact"]),
                "cross_total_over_exact_at_floor": float(detail.loc[floor_idx, "cross_total_over_exact"]),
                "cross_mass_over_exact_at_floor": float(detail.loc[floor_idx, "cross_mass_over_exact"]),
                "cross_deriv_over_exact_at_floor": float(detail.loc[floor_idx, "cross_deriv_over_exact"]),
                "min_self_pair_abs_share": float(detail["self_pair_abs_share"].min()),
                "max_self_pair_abs_share": float(detail["self_pair_abs_share"].max()),
                "min_cross_total_abs_share": float(detail["cross_total_abs_share"].min()),
                "max_cross_total_abs_share": float(detail["cross_total_abs_share"].max()),
                "all_self_pair_same_sign_as_exact": int((detail["self_pair_same_sign_as_exact"] == 1).all()),
                "all_cross_total_same_sign_as_exact": int((detail["cross_total_same_sign_as_exact"] == 1).all()),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL, index=False)
    summary.to_csv(SUMMARY, index=False)
    print(DETAIL)
    print(SUMMARY)


if __name__ == "__main__":
    main()

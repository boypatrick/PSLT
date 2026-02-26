#!/usr/bin/env python3
"""
Microscopic open-system chi extraction (localized 2-mode + Lindblad closure).

Pipeline:
  1) Solve the same 2D localized low modes (psi1, psi2) as chi_LR extraction.
  2) Build geometric fluctuation operator deltaV = Veff - Vbar(r).
  3) Project deltaV into parity basis and rotate to localized (L/R) basis.
  4) Define microscopic couplings:
       g_z = |(V_LL - V_RR)/2|,   g_x = |V_LR|
  5) Use minimal Born-Markov rates with Lorentzian PSD:
       S(omega) = 2*tau / (1 + (omega*tau)^2),
       gamma_phi = kappa_env * g_z^2 * S(0),
       gamma_mix = kappa_env * g_x^2 * S(DeltaE).
  6) Feed (delta, gamma_phi, gamma_mix) into two-level Lindblad evolution and
     export chi_eff_micro = 2*gamma_mix*Cmax/gamma_ref.

Outputs:
  - output/chi_open_system/chi_open_system_micro_D*.csv
  - output/chi_open_system/chi_open_system_micro_ratio_band_D*.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh

from extract_chi_localized_2d import (
    Level,
    PhysicalParams,
    build_generalized_operator,
    normalize_modes,
    spherical_average_v_eff,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHI_SOURCE = ROOT / "output" / "chi_fp_2d" / "localized_chi_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
DEFAULT_OUTDIR = ROOT / "output" / "chi_open_system"


@dataclass(frozen=True)
class MicroEnvConfig:
    kappa_env: float = 1.0
    tau_scale: float = 1.0
    tau_floor: float = 1e-6
    rtol: float = 1e-8
    atol: float = 1e-10


def lindblad_metrics(delta: float, gamma_phi: float, gamma_mix: float, tmax: float, nstep: int, cfg: MicroEnvConfig):
    H = np.array([[0.0, delta / 2.0], [delta / 2.0, 0.0]], dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    L_ops = [np.sqrt(max(gamma_phi, 0.0)) * sigma_z, np.sqrt(max(gamma_mix, 0.0)) * sigma_x]

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        rho = y.reshape(2, 2)
        drho = -1j * (H @ rho - rho @ H)
        for L in L_ops:
            drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        return drho.reshape(-1)

    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex).reshape(-1)
    t_eval = np.linspace(0.0, tmax, nstep)
    sol = solve_ivp(rhs, (0.0, tmax), rho0, t_eval=t_eval, rtol=cfg.rtol, atol=cfg.atol)

    cmax = 0.0
    pmax = 0.0
    for i in range(sol.y.shape[1]):
        rho = sol.y[:, i].reshape(2, 2)
        cmax = max(cmax, float(abs(rho[0, 1])))
        pmax = max(pmax, float(np.real(rho[1, 1])))
    return cmax, pmax


def resolve_gamma_ref(dval: float, src_fine: pd.DataFrame, omega1: float) -> tuple[float, float]:
    row = src_fine[np.isclose(src_fine["D"].to_numpy(dtype=float), float(dval))]
    if not row.empty and "Gamma_ref" in row.columns and "chi_LR" in row.columns:
        gamma_ref = float(row.iloc[0]["Gamma_ref"])
        chi_lr = float(row.iloc[0]["chi_LR"])
        return max(gamma_ref, 1e-30), max(chi_lr, 1e-30)

    gamma1 = omega1 ** 9
    gamma2 = omega1 ** 13
    gamma_ref = math.sqrt(max(gamma1 * gamma2, 1e-30))
    return max(gamma_ref, 1e-30), float("nan")


def run_micro_case(
    D: float,
    level: Level,
    p: PhysicalParams,
    rho_max: float,
    z_margin: float,
    n_mu: int,
    tol: float,
    maxiter: int,
    sigma: Optional[float],
    cfg: MicroEnvConfig,
    src_fine: pd.DataFrame,
    tmax: float,
    nstep: int,
) -> Dict[str, float]:
    z_max = D / 2.0 + z_margin
    rho, z, rr, zz, uu, k_mat, m_mat = build_generalized_operator(
        D=D,
        p=p,
        rho_max=rho_max,
        z_max=z_max,
        dr=level.dr,
        dz=level.dz,
    )

    if sigma is None:
        evals, evecs = eigsh(k_mat, k=2, M=m_mat, which="SA", tol=tol, maxiter=maxiter)
    else:
        evals, evecs = eigsh(k_mat, k=2, M=m_mat, sigma=sigma, which="LM", tol=tol, maxiter=maxiter)

    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    nr, nz = len(rho), len(z)
    psi = np.stack([evecs[:, 0].reshape(nr, nz), evecs[:, 1].reshape(nr, nz)], axis=2)
    psi = normalize_modes(psi, rho, level.dr, level.dz)
    psi1 = psi[:, :, 0]
    psi2 = psi[:, :, 1]

    w_int = 2.0 * math.pi * rho[:, None] * level.dr * level.dz

    # Geometric fluctuation operator around spherical average.
    vfull = uu + p.m0 * p.m0
    r_field = np.sqrt(rr * rr + zz * zz)
    r_line = np.linspace(0.0, float(r_field.max()), 1800)
    vbar_line = spherical_average_v_eff(r_line, D, p, n_mu=n_mu)
    vbar = np.interp(r_field.ravel(), r_line, vbar_line).reshape(r_field.shape)
    delta_v = vfull - vbar

    # Parity-basis projected matrix V_ij = <psi_i|deltaV|psi_j>.
    v11 = float(np.sum(w_int * psi1 * delta_v * psi1))
    v22 = float(np.sum(w_int * psi2 * delta_v * psi2))
    v12 = float(np.sum(w_int * psi1 * delta_v * psi2))
    v_par = np.array([[v11, v12], [v12, v22]], dtype=float)

    # Rotate to localized basis: |L/R> = (|1> +/- |2>)/sqrt(2).
    u_lr = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=float) / math.sqrt(2.0)
    v_loc = u_lr.T @ v_par @ u_lr
    g_z = float(abs(0.5 * (v_loc[0, 0] - v_loc[1, 1])))
    g_x = float(abs(v_loc[0, 1]))

    e1 = float(evals[0])
    e2 = float(evals[1])
    delta = float(abs(e2 - e1))
    omega1 = float(math.sqrt(max(e1 + p.m0 * p.m0, 0.0)))
    gamma_ref, chi_lr = resolve_gamma_ref(D, src_fine=src_fine, omega1=omega1)

    # Minimal Lorentzian PSD and Born-Markov rates.
    tau_env = float(max(cfg.tau_scale / max(omega1, 1e-30), cfg.tau_floor))
    szz_0 = float(2.0 * tau_env)
    sxx_delta = float(2.0 * tau_env / (1.0 + (delta * tau_env) ** 2))
    gamma_phi = float(max(cfg.kappa_env * (g_z ** 2) * szz_0, 0.0))
    gamma_mix = float(max(cfg.kappa_env * (g_x ** 2) * sxx_delta, 0.0))

    cmax, pmax = lindblad_metrics(
        delta=delta,
        gamma_phi=gamma_phi,
        gamma_mix=gamma_mix,
        tmax=tmax,
        nstep=nstep,
        cfg=cfg,
    )
    chi_eff_micro = float(max(2.0 * gamma_mix * cmax / max(gamma_ref, 1e-30), 0.0))

    return {
        "D": float(D),
        "kappa_env": float(cfg.kappa_env),
        "tau_scale": float(cfg.tau_scale),
        "tau_floor": float(cfg.tau_floor),
        "delta": delta,  # generic key for pslt_lib interpolation path
        "delta_micro": delta,
        "omega1": omega1,
        "gamma_ref": gamma_ref,  # generic key
        "Gamma_ref": gamma_ref,  # compatibility with legacy open-system loader
        "gamma_ref_micro": gamma_ref,
        "V11_deltaV": v11,
        "V22_deltaV": v22,
        "V12_deltaV": v12,
        "VLL_deltaV": float(v_loc[0, 0]),
        "VRR_deltaV": float(v_loc[1, 1]),
        "VLR_deltaV": float(v_loc[0, 1]),
        "g_z_micro": g_z,
        "g_x_micro": g_x,
        "tau_env": tau_env,
        "Szz_0": szz_0,
        "Sxx_delta": sxx_delta,
        "gamma_phi_micro": gamma_phi,
        "gamma_mix_micro": gamma_mix,
        "Cmax_micro": cmax,
        "Pmax_micro": pmax,
        "chi_eff_micro": chi_eff_micro,
        "chi_LR_baseline": chi_lr,
        "ratio_micro_to_baseline": chi_eff_micro / max(chi_lr, 1e-30) if np.isfinite(chi_lr) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Microscopic open-system chi extraction.")
    ap.add_argument("--Ds", default="6,12,18")
    ap.add_argument("--full-scan", action="store_true", help="Use D=4..20 integer grid (overrides --Ds).")
    ap.add_argument("--chi-source", default=str(DEFAULT_CHI_SOURCE))
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--n-mu", type=int, default=120)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    ap.add_argument("--sigma", type=float, default=2.5, help="Shift-invert target; use negative value to disable.")
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--nstep", type=int, default=1200)
    ap.add_argument("--kappa-env", type=float, default=1.0)
    ap.add_argument("--tau-scale", type=float, default=1.0)
    ap.add_argument("--tau-floor", type=float, default=1e-6)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    if args.full_scan:
        d_list = [float(d) for d in range(4, 21)]
    else:
        d_list = [float(s.strip()) for s in args.Ds.split(",") if s.strip()]

    src = pd.read_csv(args.chi_source)
    if "level" in src.columns:
        src = src[src["level"] == "fine"].copy()
    src = src.sort_values("D").reset_index(drop=True)
    p = PhysicalParams()
    level = Level("fine", dr=float(args.dr), dz=float(args.dz))
    sigma = None if float(args.sigma) < 0 else float(args.sigma)
    env_cfg = MicroEnvConfig(
        kappa_env=float(args.kappa_env),
        tau_scale=float(args.tau_scale),
        tau_floor=float(args.tau_floor),
    )

    rows: List[Dict[str, float]] = []
    for dval in d_list:
        row = run_micro_case(
            D=float(dval),
            level=level,
            p=p,
            rho_max=float(args.rho_max),
            z_margin=float(args.z_margin),
            n_mu=int(args.n_mu),
            tol=float(args.tol),
            maxiter=int(args.maxiter),
            sigma=sigma,
            cfg=env_cfg,
            src_fine=src,
            tmax=float(args.tmax),
            nstep=int(args.nstep),
        )
        rows.append(row)
        print(
            f"[run] D={dval:g}, gz={row['g_z_micro']:.3e}, gx={row['g_x_micro']:.3e}, "
            f"gamma_phi={row['gamma_phi_micro']:.3e}, gamma_mix={row['gamma_mix_micro']:.3e}, "
            f"chi_eff_micro={row['chi_eff_micro']:.3e}, ratio={row['ratio_micro_to_baseline']:.3f}"
        )

    out = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)
    out_csv = outdir / f"chi_open_system_micro_D{tag}.csv"
    out.to_csv(out_csv, index=False)

    ratio = out["ratio_micro_to_baseline"].to_numpy(dtype=float)
    band = pd.DataFrame(
        [
            {
                "csv_profile": str(out_csv),
                "D_min": float(np.min(out["D"])),
                "D_max": float(np.max(out["D"])),
                "ratio_min": float(np.nanmin(ratio)),
                "ratio_max": float(np.nanmax(ratio)),
                "ratio_mean": float(np.nanmean(ratio)),
                "ratio_std": float(np.nanstd(ratio)),
                "ratio_p10": float(np.nanpercentile(ratio, 10.0)),
                "ratio_p90": float(np.nanpercentile(ratio, 90.0)),
            }
        ]
    )
    band_csv = outdir / f"chi_open_system_micro_ratio_band_D{tag}.csv"
    band.to_csv(band_csv, index=False)

    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {band_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

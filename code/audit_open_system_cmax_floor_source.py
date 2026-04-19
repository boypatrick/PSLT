#!/usr/bin/env python3
"""
Audit a closed-form underdamped source for the open-system Cmax floor.

For the minimal localized two-level Lindblad block with

    H = (delta/2) sigma_x,
    L_phi = sqrt(gamma_phi) sigma_z,
    L_mix = sqrt(gamma_mix) sigma_x,

and initial state |L><L|, the Bloch equations close on the (y,z)-plane and,
throughout the exported exact bridge, remain underdamped.  This script
verifies the exact closed-form Cmax formula and audits its monotonicity on the
exported parameter box, yielding a separated floor from
(delta_min, gamma_phi_max, gamma_mix_max).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


def read_csv_any(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(paths[0])


def cmax_closed_form(delta: float, gamma_phi: float, gamma_mix: float) -> float:
    if delta <= gamma_phi:
        return float("nan")
    Gamma = gamma_phi + 2.0 * gamma_mix
    omega = math.sqrt(max(delta * delta - gamma_phi * gamma_phi, 0.0))
    if omega <= 0.0:
        return 0.0
    theta = math.atan2(omega, Gamma)
    return float(
        delta
        / (2.0 * math.sqrt(Gamma * Gamma + omega * omega))
        * math.exp(-(Gamma / omega) * theta)
    )


def main() -> None:
    bridge = read_csv_any(
        OUTDIR / "chi_open_system_micro_bridge_map.csv",
        PAPER_DIR / "chi_open_system_micro_bridge_map.csv",
    ).copy()
    chi_sum = read_csv_any(
        OUTDIR / "chi_open_system_chi_eff_floor_source_summary.csv",
        PAPER_DIR / "chi_open_system_chi_eff_floor_source_summary.csv",
    ).iloc[0]

    delta = bridge["delta_micro"].astype(float)
    gamma_phi = bridge["gamma_phi_micro"].astype(float)
    gamma_mix = bridge["gamma_mix_micro"].astype(float)
    cmax = bridge["Cmax_micro"].astype(float)

    Gamma = gamma_phi + 2.0 * gamma_mix
    omega = np.sqrt(np.maximum(delta * delta - gamma_phi * gamma_phi, 0.0))
    c_formula = np.array(
        [cmax_closed_form(float(d), float(gp), float(gm)) for d, gp, gm in zip(delta, gamma_phi, gamma_mix)],
        dtype=float,
    )

    bridge["Gamma_closed_form"] = Gamma
    bridge["omega_closed_form"] = omega
    bridge["Cmax_closed_form"] = c_formula
    bridge["Cmax_closed_form_residual_abs"] = np.abs(cmax - c_formula)
    bridge["delta_minus_gamma_phi"] = delta - gamma_phi
    bridge["gamma_phi_over_delta"] = gamma_phi / np.maximum(delta, 1e-300)

    dmin = float(delta.min())
    dmax = float(delta.max())
    gpmin = float(gamma_phi.min())
    gpmax = float(gamma_phi.max())
    gmmin = float(gamma_mix.min())
    gmmax = float(gamma_mix.max())

    # Finite-difference monotonicity audit on the exported parameter box.
    h = 1.0e-6
    ds = np.linspace(dmin, dmax, 21)
    gps = np.linspace(gpmin, gpmax, 21)
    gms = np.linspace(gmmin, gmmax, 21)
    min_dC_dd = float("inf")
    max_dC_dgphi = -float("inf")
    max_dC_dgmix = -float("inf")
    for d in ds:
        for gp in gps:
            for gm in gms:
                if d <= gp + 1.0e-9:
                    continue
                c_dp = cmax_closed_form(d + h, gp, gm)
                c_dm = cmax_closed_form(d - h, gp, gm)
                dC_dd = (c_dp - c_dm) / (2.0 * h)

                gp_lo = max(gp - h, 0.0)
                gp_hi = min(gp + h, d - 1.0e-9)
                c_gpp = cmax_closed_form(d, gp_hi, gm)
                c_gpm = cmax_closed_form(d, gp_lo, gm)
                dC_dgphi = (c_gpp - c_gpm) / max(gp_hi - gp_lo, 1.0e-12)

                gm_lo = max(gm - h, 0.0)
                gm_hi = gm + h
                c_gmp = cmax_closed_form(d, gp, gm_hi)
                c_gmm = cmax_closed_form(d, gp, gm_lo)
                dC_dgmix = (c_gmp - c_gmm) / max(gm_hi - gm_lo, 1.0e-12)

                min_dC_dd = min(min_dC_dd, dC_dd)
                max_dC_dgphi = max(max_dC_dgphi, dC_dgphi)
                max_dC_dgmix = max(max_dC_dgmix, dC_dgmix)

    omega_min_sep = math.sqrt(max(dmin * dmin - gpmax * gpmax, 0.0))
    Gamma_max_sep = gpmax + 2.0 * gmmax
    cmax_sep_floor = cmax_closed_form(dmin, gpmax, gmmax)

    a_chi_sep_floor = float(chi_sum["A_chi_separated_floor"])
    chi_eff_floor_from_sep = a_chi_sep_floor * cmax_sep_floor

    out_detail = OUTDIR / "chi_open_system_cmax_floor_source_detail.csv"
    out_summary = OUTDIR / "chi_open_system_cmax_floor_source_summary.csv"
    bridge.to_csv(out_detail, index=False)

    summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(bridge)),
                "all_rows_underdamped": int(bool(np.all(delta > gamma_phi))),
                "min_delta_minus_gamma_phi": float((delta - gamma_phi).min()),
                "max_gamma_phi_over_delta": float((gamma_phi / np.maximum(delta, 1e-300)).max()),
                "max_abs_Cmax_closed_form_residual": float(bridge["Cmax_closed_form_residual_abs"].max()),
                "Cmax_closed_form_floor": float(np.min(c_formula)),
                "Cmax_closed_form_floor_D": float(bridge.loc[np.argmin(c_formula), "D"]),
                "Cmax_exact_floor": float(cmax.min()),
                "Cmax_exact_floor_D": float(bridge.loc[cmax.idxmin(), "D"]),
                "actual_over_closed_form_floor": float(cmax.min() / max(float(np.min(c_formula)), 1e-300)),
                "delta_min": dmin,
                "delta_min_D": float(bridge.loc[delta.idxmin(), "D"]),
                "gamma_phi_max": gpmax,
                "gamma_phi_max_D": float(bridge.loc[gamma_phi.idxmax(), "D"]),
                "gamma_mix_max": gmmax,
                "gamma_mix_max_D": float(bridge.loc[gamma_mix.idxmax(), "D"]),
                "omega_min_from_extrema": omega_min_sep,
                "Gamma_max_from_extrema": Gamma_max_sep,
                "Cmax_separated_floor": cmax_sep_floor,
                "actual_over_separated_Cmax_floor": float(cmax.min() / max(cmax_sep_floor, 1e-300)),
                "min_dC_dd_on_box": min_dC_dd,
                "max_dC_dgamma_phi_on_box": max_dC_dgphi,
                "max_dC_dgamma_mix_on_box": max_dC_dgmix,
                "A_chi_separated_floor": a_chi_sep_floor,
                "chi_eff_floor_from_separated_Achi_and_Cmax_floors": chi_eff_floor_from_sep,
                "actual_chi_eff_floor": float(chi_sum["actual_chi_eff_floor"]),
                "actual_over_separated_chi_eff_floor": float(chi_sum["actual_chi_eff_floor"] / max(chi_eff_floor_from_sep, 1e-300)),
            }
        ]
    )
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f"[saved] {out_detail}")
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Riccati/DtN static-width extraction audit.

This is the next gate after the local Robin/Siegert negative control.  The
local approximation

    m_R^{loc}(E) = i sqrt(E - U_l(R;D))

is replaced by an exterior Dirichlet-to-Neumann map computed from the Riccati
log-derivative equation

    m'(z) + m(z)^2 = U_l(z;D) - E,     m(z)=f'(z;E)/f(z;E).

For each trial E, the right outgoing exterior is initialized at a large
R_tail by one of three outgoing logarithmic derivatives (WKB, asymptotic
Coulomb series, or the exact Coulomb/Whittaker tail) and integrated inward to
the finite interface R.  The resulting m_+(E;R) is inserted into the same
half-line parity recurrence used by the finite-radius Siegert audit.

This file intentionally remains an audit gate: a physical width is promoted
only if the resulting pole is stable in R, stable in R_tail, and parity
degenerate.

Outputs:
  - output/cap_resonance_1d/static_width_exact_dtn_detail.csv
  - output/cap_resonance_1d/static_width_exact_dtn_summary.csv
  - output/cap_resonance_1d/static_width_coulomb_tail_dtn_o*_*.csv
  - output/cap_resonance_1d/static_width_whittaker_dtn_dps*_*.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root

from audit_static_width_siegert_extraction import (
    OUTDIR,
    AuditConfig,
    outgoing_q,
    relative_span,
    u_eff,
)
from extract_cap_resonance_1d import hermitian_seed_state
from extract_superrad_prefactor_1d import PhysicalParams, wkb_action_central


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExactDtnConfig:
    ell: int = 1
    n_target: int = 3
    n_half: int = 1200
    d_values: tuple[float, ...] = (6.0, 12.0, 18.0)
    r_values: tuple[float, ...] = (60.0, 80.0, 100.0, 120.0)
    r_tail_values: tuple[float, ...] = (800.0,)
    parity_values: tuple[str, ...] = ("even", "odd")
    imag_guesses: tuple[float, ...] = (-1.0e-5, -1.0e-4, -1.0e-3, -2.0e-3)
    root_xtol: float = 1.0e-9
    root_maxfev: int = 80
    dtn_rtol: float = 1.0e-7
    dtn_atol: float = 1.0e-9
    dtn_max_step: float = 5.0
    fallback_guesses: bool = False
    tail_mode: str = "wkb"
    coulomb_order: int = 6
    whittaker_dps: int = 50

    # Promotion gates.
    r_gamma_rel_span_max: float = 0.15
    tail_gamma_rel_span_max: float = 0.05
    r120_over_r80_min: float = 0.90
    r120_over_r80_max: float = 1.10
    parity_gamma_rel_split_max: float = 1.0e-5
    root_residual_abs_max: float = 1.0e-6


def finite_median(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))


def du_eff_dz(z: float, D: float, ell: int, p: PhysicalParams) -> float:
    step = max(1.0e-4, 1.0e-5 * abs(z))
    vals = u_eff(np.array([z - step, z + step]), D=D, ell=ell, p=p)
    return float((vals[1] - vals[0]) / (2.0 * step))


def outgoing_wkb_log_derivative(
    E: complex,
    z: float,
    D: float,
    ell: int,
    p: PhysicalParams,
) -> complex:
    U_z = float(u_eff(np.array([z]), D=D, ell=ell, p=p)[0])
    q = outgoing_q(E, U_z)
    if abs(q) < 1.0e-14:
        return 1j * q
    # For f ~ q^{-1/2} exp(i int q dz), m=f'/f=i q - q'/(2q)
    # and q'=-U'/(2q), hence the amplitude term is U'/(4q^2).
    return 1j * q + du_eff_dz(z, D=D, ell=ell, p=p) / (4.0 * q * q)


def coulomb_tail_coefficients(
    E: complex,
    ell: int,
    p: PhysicalParams,
    order: int,
) -> tuple[complex, list[complex], float, float]:
    """Outgoing Coulomb-tail log-derivative coefficients.

    The action-derived one-dimensional tail has

        U_l(z;D) = A/z + B_l/z^2 + O(z^-3),
        A = 4 a m0^2,  B_l = l(l+1) + 4 a^2 m0^2.

    Inserting m(z)=ik+sum c_n z^-n into m'+m^2=A/z+B/z^2-E gives the
    triangular recurrence used here.  This is an analytic normalization of the
    outgoing branch; the remaining integration from R_tail to R uses the full
    numerical U_l, so D-dependent O(z^-3) tail terms are retained there.
    """

    k = np.sqrt(E + 0j)
    if k.real < 0.0:
        k = -k
    A = 4.0 * p.a * p.m0 * p.m0
    B = ell * (ell + 1.0) + 4.0 * p.a * p.a * p.m0 * p.m0
    coeffs = [0j] * (order + 1)
    for n in range(1, order + 1):
        rhs = A if n == 1 else (B if n == 2 else 0.0)
        convolution = sum(coeffs[j] * coeffs[n - j] for j in range(1, n))
        derivative_prev = (n - 1) * coeffs[n - 1] if n > 1 else 0j
        coeffs[n] = (rhs + derivative_prev - convolution) / (2j * k)
    return k, coeffs, A, B


def outgoing_coulomb_log_derivative(
    E: complex,
    z: float,
    ell: int,
    p: PhysicalParams,
    order: int,
) -> tuple[complex, dict[str, float]]:
    k, coeffs, A, B = coulomb_tail_coefficients(E=E, ell=ell, p=p, order=order)
    m_tail = 1j * k + sum(coeffs[n] / (z**n) for n in range(1, order + 1))
    next_term_abs = np.nan
    if order >= 1:
        # One extra recurrence term gives a cheap truncation-size diagnostic.
        _, next_coeffs, _, _ = coulomb_tail_coefficients(
            E=E,
            ell=ell,
            p=p,
            order=order + 1,
        )
        next_term_abs = float(abs(next_coeffs[order + 1] / (z ** (order + 1))))
    meta = {
        "coulomb_A": float(A),
        "coulomb_B": float(B),
        "coulomb_k_real": float(k.real),
        "coulomb_k_imag": float(k.imag),
        "coulomb_next_term_abs": next_term_abs,
    }
    return m_tail, meta


def whittaker_tail_parameters(
    E: complex,
    ell: int,
    p: PhysicalParams,
) -> tuple[complex, complex, float, float, float]:
    """Parameters for the exact Coulomb/Whittaker tail.

    The analytic tail equation is

        f'' + (E - A/z - B/z^2) f = 0,
        A = 4 a m0^2,  B = ell(ell+1) + 4 a^2 m0^2.

    With x=-2 i k z and k=sqrt(E), the outgoing solution is

        f(z) = W_{kappa,mu}(x),
        kappa = -i A/(2k),  mu = sqrt(B+1/4).
    """

    k = np.sqrt(E + 0j)
    if k.real < 0.0:
        k = -k
    A = 4.0 * p.a * p.m0 * p.m0
    B = ell * (ell + 1.0) + 4.0 * p.a * p.a * p.m0 * p.m0
    mu = float(np.sqrt(B + 0.25))
    kappa = -1j * A / (2.0 * k)
    return k, kappa, mu, A, B


def outgoing_whittaker_log_derivative(
    E: complex,
    z: float,
    ell: int,
    p: PhysicalParams,
    dps: int,
) -> tuple[complex, dict[str, float]]:
    k, kappa, mu, A, B = whittaker_tail_parameters(E=E, ell=ell, p=p)
    with mp.workdps(dps):
        k_mp = mp.mpc(float(k.real), float(k.imag))
        kappa_mp = mp.mpc(float(kappa.real), float(kappa.imag))
        mu_mp = mp.mpf(mu)
        x = -2j * k_mp * mp.mpf(z)
        a_hyp = mu_mp - kappa_mp + mp.mpf("0.5")
        b_hyp = 2 * mu_mp + 1
        U0 = mp.hyperu(a_hyp, b_hyp, x)
        U1 = mp.hyperu(a_hyp + 1, b_hyp + 1, x)
        # W_{kappa,mu}(x)=exp(-x/2) x^{mu+1/2} U(a,b,x).
        dlogW_dx = -mp.mpf("0.5") + (mu_mp + mp.mpf("0.5")) / x - a_hyp * U1 / U0
        m_tail_mp = (-2j * k_mp) * dlogW_dx
        U0_abs = abs(U0)
        U1_abs = abs(U1)
    m_tail = complex(m_tail_mp)
    meta = {
        "coulomb_A": float(A),
        "coulomb_B": float(B),
        "coulomb_k_real": float(k.real),
        "coulomb_k_imag": float(k.imag),
        "coulomb_next_term_abs": np.nan,
        "whittaker_kappa_real": float(kappa.real),
        "whittaker_kappa_imag": float(kappa.imag),
        "whittaker_mu": float(mu),
        "whittaker_x_abs": float(abs(complex(-2j * k * z))),
        "whittaker_U_abs": float(U0_abs),
        "whittaker_U_shift_abs": float(U1_abs),
        "whittaker_dps": float(dps),
    }
    return m_tail, meta


def exterior_dtn(
    E: complex,
    R: float,
    R_tail: float,
    D: float,
    ell: int,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
) -> tuple[complex, dict[str, float]]:
    if R_tail <= R:
        raise ValueError(f"R_tail={R_tail} must exceed R={R}")

    if cfg.tail_mode == "wkb":
        m_tail = outgoing_wkb_log_derivative(E=E, z=R_tail, D=D, ell=ell, p=p)
        tail_meta = {
            "coulomb_A": np.nan,
            "coulomb_B": np.nan,
            "coulomb_k_real": np.nan,
            "coulomb_k_imag": np.nan,
            "coulomb_next_term_abs": np.nan,
            "whittaker_kappa_real": np.nan,
            "whittaker_kappa_imag": np.nan,
            "whittaker_mu": np.nan,
            "whittaker_x_abs": np.nan,
            "whittaker_U_abs": np.nan,
            "whittaker_U_shift_abs": np.nan,
            "whittaker_dps": np.nan,
        }
    elif cfg.tail_mode == "coulomb_series":
        m_tail, tail_meta = outgoing_coulomb_log_derivative(
            E=E,
            z=R_tail,
            ell=ell,
            p=p,
            order=cfg.coulomb_order,
        )
        tail_meta.update(
            {
                "whittaker_kappa_real": np.nan,
                "whittaker_kappa_imag": np.nan,
                "whittaker_mu": np.nan,
                "whittaker_x_abs": np.nan,
                "whittaker_U_abs": np.nan,
                "whittaker_U_shift_abs": np.nan,
                "whittaker_dps": np.nan,
            }
        )
    elif cfg.tail_mode == "whittaker":
        m_tail, tail_meta = outgoing_whittaker_log_derivative(
            E=E,
            z=R_tail,
            ell=ell,
            p=p,
            dps=cfg.whittaker_dps,
        )
    else:
        raise ValueError(f"Unknown tail_mode: {cfg.tail_mode}")

    def rhs(z: float, y: np.ndarray) -> list[float]:
        m = complex(float(y[0]), float(y[1]))
        U_z = float(u_eff(np.array([z]), D=D, ell=ell, p=p)[0])
        val = U_z - E - m * m
        return [float(val.real), float(val.imag)]

    sol = solve_ivp(
        rhs,
        t_span=(R_tail, R),
        y0=np.array([m_tail.real, m_tail.imag], dtype=float),
        method="DOP853",
        rtol=cfg.dtn_rtol,
        atol=cfg.dtn_atol,
        max_step=cfg.dtn_max_step,
    )
    m_R = complex(float(sol.y[0, -1]), float(sol.y[1, -1]))
    defect = rhs(R, np.array([m_R.real, m_R.imag], dtype=float))
    meta = {
        "dtn_success": float(bool(sol.success)),
        "dtn_nfev": float(sol.nfev),
        "dtn_status": float(sol.status),
        "dtn_residual_abs": float(abs(complex(defect[0], defect[1]))),
        "m_tail_real": float(m_tail.real),
        "m_tail_imag": float(m_tail.imag),
        "m_R_real": float(m_R.real),
        "m_R_imag": float(m_R.imag),
        "tail_mode": cfg.tail_mode,
        "coulomb_order": float(cfg.coulomb_order),
        "whittaker_dps": float(cfg.whittaker_dps),
        **tail_meta,
    }
    return m_R, meta


def ratio_residual_exact_dtn(
    E: complex,
    U_half: np.ndarray,
    h: float,
    parity: str,
    m_R: complex,
) -> complex:
    n = len(U_half) - 1
    if parity == "even":
        ratio = 1.0 + 0.5 * h * h * (U_half[0] - E)
        start = 1
    elif parity == "odd":
        ratio = 2.0 + h * h * (U_half[1] - E)
        start = 2
    else:
        raise ValueError(f"Unknown parity: {parity}")

    for i in range(start, n):
        if abs(ratio) < 1.0e-300:
            ratio = 1.0e-300 + 0j
        ratio = (2.0 + h * h * (U_half[i] - E)) - 1.0 / ratio

    return -2.0 / (h * h * ratio) + (2.0 / (h * h) - 2.0 * m_R / h + U_half[-1] - E)


def load_local_guess_map() -> dict[tuple[float, float, str], complex]:
    path = OUTDIR / "static_width_siegert_audit_detail.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[tuple[float, float, str], complex] = {}
    for _, row in df.iterrows():
        key = (float(row["D"]), float(row["R"]), str(row["parity"]))
        out[key] = complex(float(row["E_siegert_real"]), float(row["E_siegert_imag"]))
    return out


def solve_exact_dtn_case(
    D: float,
    R: float,
    R_tail: float,
    parity: str,
    cfg: ExactDtnConfig,
    p: PhysicalParams,
    local_guess_map: dict[tuple[float, float, str], complex],
) -> dict[str, float | str]:
    h = R / cfg.n_half
    z_half = np.linspace(0.0, R, cfg.n_half + 1)
    U_half = u_eff(z_half, D=D, ell=cfg.ell, p=p)

    z_full = np.linspace(-R, R, 2 * cfg.n_half + 1)
    U_full = u_eff(z_full, D=D, ell=cfg.ell, p=p)
    seed = hermitian_seed_state(
        U_full,
        z_full,
        n_target=cfg.n_target,
        n_states=max(cfg.n_target + 4, 8),
    )
    seed_E = float(seed["seed_E"])
    seed_omega = np.sqrt(p.m0 * p.m0 + seed_E)

    def residual_for(E: complex) -> tuple[complex, dict[str, float]]:
        m_R, meta = exterior_dtn(
            E=E,
            R=R,
            R_tail=R_tail,
            D=D,
            ell=cfg.ell,
            cfg=cfg,
            p=p,
        )
        val = ratio_residual_exact_dtn(E=E, U_half=U_half, h=h, parity=parity, m_R=m_R)
        return val, meta

    def fun(x: np.ndarray) -> list[float]:
        val, _ = residual_for(complex(float(x[0]), float(x[1])))
        return [float(val.real), float(val.imag)]

    guesses: list[complex] = []
    local_guess = local_guess_map.get((float(D), float(R), parity))
    if local_guess is not None:
        guesses.append(local_guess)
    if cfg.fallback_guesses or local_guess is None:
        guesses.extend(complex(seed_E, guess_im) for guess_im in cfg.imag_guesses)

    rows: list[dict[str, float]] = []
    for guess in guesses:
        try:
            sol = root(
                fun,
                np.array([guess.real, guess.imag], dtype=float),
                method="hybr",
                options={"xtol": cfg.root_xtol, "maxfev": cfg.root_maxfev},
            )
            E = complex(float(sol.x[0]), float(sol.x[1]))
            res, meta = residual_for(E)
            rows.append(
                {
                    "root_success": float(bool(sol.success)),
                    "root_nfev": float(sol.nfev),
                    "guess_E_real": float(guess.real),
                    "guess_E_imag": float(guess.imag),
                    "E_dtn_real": float(E.real),
                    "E_dtn_imag": float(E.imag),
                    "residual_abs": float(abs(res)),
                    "seed_dist": float(abs(E - seed_E)),
                    **meta,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep audit rows inspectable.
            rows.append(
                {
                    "root_success": 0.0,
                    "root_nfev": np.nan,
                    "guess_E_real": float(guess.real),
                    "guess_E_imag": float(guess.imag),
                    "E_dtn_real": np.nan,
                    "E_dtn_imag": np.nan,
                    "residual_abs": np.inf,
                    "seed_dist": np.inf,
                    "dtn_success": 0.0,
                    "dtn_nfev": np.nan,
                    "dtn_status": np.nan,
                    "dtn_residual_abs": np.inf,
                    "m_tail_real": np.nan,
                    "m_tail_imag": np.nan,
                    "m_R_real": np.nan,
                    "m_R_imag": np.nan,
                    "exception": str(exc),
                }
            )

    cand = pd.DataFrame(rows)
    good = cand[(cand["residual_abs"] <= cfg.root_residual_abs_max) & np.isfinite(cand["seed_dist"])]
    if len(good) > 0:
        best = good.sort_values(["seed_dist", "residual_abs"]).iloc[0].to_dict()
    else:
        best = cand.sort_values(["residual_abs", "seed_dist"]).iloc[0].to_dict()

    E_root = complex(float(best["E_dtn_real"]), float(best["E_dtn_imag"]))
    omega_root = np.sqrt(p.m0 * p.m0 + E_root + 0j)
    if omega_root.real < 0.0:
        omega_root = -omega_root

    gamma_width = max(0.0, -2.0 * omega_root.imag)
    gamma_energy = max(0.0, -2.0 * E_root.imag)
    gamma_narrow = gamma_energy / max(2.0 * max(omega_root.real, 0.0), 1.0e-300)
    S_seed = wkb_action_central(U_full, z_full, seed_E)
    gamma_geo = (
        float(seed_omega) * np.exp(-2.0 * S_seed)
        if np.isfinite(S_seed)
        else np.nan
    )

    return {
        "D": float(D),
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "R": float(R),
        "R_tail": float(R_tail),
        "n_half": float(cfg.n_half),
        "h": float(h),
        "parity": parity,
        "seed_E": float(seed_E),
        "seed_omega": float(seed_omega),
        "E_dtn_real": float(E_root.real),
        "E_dtn_imag": float(E_root.imag),
        "omega_dtn_real": float(omega_root.real),
        "omega_dtn_imag": float(omega_root.imag),
        "Gamma_dtn": float(gamma_width),
        "Gamma_dtn_narrow": float(gamma_narrow),
        "rel_Gamma_exact_vs_narrow": float(
            abs(gamma_width - gamma_narrow) / max(abs(gamma_width), 1.0e-300)
        ),
        "S_seed": float(S_seed) if np.isfinite(S_seed) else np.nan,
        "Gamma_geo_seed": float(gamma_geo) if np.isfinite(gamma_geo) else np.nan,
        "Gamma_dtn_over_geo": float(gamma_width / max(gamma_geo, 1.0e-300))
        if np.isfinite(gamma_geo)
        else np.nan,
        "root_candidate_count": float(len(cand)),
        **{k: v for k, v in best.items() if k not in {"E_dtn_real", "E_dtn_imag"}},
    }


def parity_split_metrics(sub: pd.DataFrame) -> tuple[float, float]:
    even = sub[sub["parity"] == "even"].set_index(["R_tail", "R"])
    odd = sub[sub["parity"] == "odd"].set_index(["R_tail", "R"])
    common = sorted(set(even.index).intersection(set(odd.index)))
    gamma_rel: list[float] = []
    e_abs: list[float] = []
    for key in common:
        ge = float(even.loc[key, "Gamma_dtn"])
        go = float(odd.loc[key, "Gamma_dtn"])
        ee = complex(float(even.loc[key, "E_dtn_real"]), float(even.loc[key, "E_dtn_imag"]))
        eo = complex(float(odd.loc[key, "E_dtn_real"]), float(odd.loc[key, "E_dtn_imag"]))
        gamma_rel.append(abs(ge - go) / max(0.5 * (abs(ge) + abs(go)), 1.0e-300))
        e_abs.append(abs(ee - eo))
    return (
        float(max(gamma_rel)) if gamma_rel else np.nan,
        float(max(e_abs)) if e_abs else np.nan,
    )


def summarize_D(sub: pd.DataFrame, cfg: ExactDtnConfig) -> dict[str, float | str]:
    by_tail_R = sub.groupby(["R_tail", "R"], sort=True)["Gamma_dtn"].mean()
    r_spans: list[float] = []
    r_ratios: list[float] = []
    for R_tail, tail_sub in by_tail_R.groupby(level=0, sort=True):
        vals = tail_sub.droplevel(0)
        r_spans.append(relative_span(vals.to_numpy(dtype=float)))
        if 80.0 in vals.index and 120.0 in vals.index and float(vals.loc[80.0]) != 0.0:
            r_ratios.append(float(vals.loc[120.0] / vals.loc[80.0]))

    tail_spans: list[float] = []
    for R, R_sub in by_tail_R.groupby(level=1, sort=True):
        vals = R_sub.droplevel(1)
        tail_spans.append(relative_span(vals.to_numpy(dtype=float)))

    parity_gamma_rel_max, parity_E_abs_max = parity_split_metrics(sub)
    residual_abs_max = float(sub["residual_abs"].max())
    dtn_residual_abs_max = float(sub["dtn_residual_abs"].max())
    r_span_max = float(max(r_spans)) if r_spans else np.nan
    tail_span_max = float(max(tail_spans)) if tail_spans else np.nan
    r_ratio_min = float(min(r_ratios)) if r_ratios else np.nan
    r_ratio_max = float(max(r_ratios)) if r_ratios else np.nan

    reasons: list[str] = []
    if residual_abs_max > cfg.root_residual_abs_max:
        reasons.append("root_residual_large")
    if parity_gamma_rel_max > cfg.parity_gamma_rel_split_max:
        reasons.append("parity_split_large")
    if r_span_max > cfg.r_gamma_rel_span_max:
        reasons.append("R_width_not_stationary")
    if tail_span_max > cfg.tail_gamma_rel_span_max:
        reasons.append("R_tail_not_stationary")
    if not (
        np.isfinite(r_ratio_min)
        and np.isfinite(r_ratio_max)
        and cfg.r120_over_r80_min <= r_ratio_min
        and r_ratio_max <= cfg.r120_over_r80_max
    ):
        reasons.append("R120_R80_ratio_outside_gate")

    pass_gate = len(reasons) == 0
    next_gate = (
        "no_static_width_promotion_without_new_exterior_stable_pole_theorem"
        if cfg.tail_mode == "whittaker"
        else "tighten_exterior_asymptotic_or_ECS_before_static_width_promotion"
    )
    return {
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "R_min": float(min(cfg.r_values)),
        "R_max": float(max(cfg.r_values)),
        "R_tail_min": float(min(cfg.r_tail_values)),
        "R_tail_max": float(max(cfg.r_tail_values)),
        "n_R": float(len(cfg.r_values)),
        "n_R_tail": float(len(cfg.r_tail_values)),
        "Gamma_dtn_min": float(sub["Gamma_dtn"].min()),
        "Gamma_dtn_max": float(sub["Gamma_dtn"].max()),
        "Gamma_dtn_candidate": finite_median(by_tail_R.to_numpy(dtype=float)),
        "Gamma_dtn_R_rel_span_max": r_span_max,
        "Gamma_dtn_Rtail_rel_span_max": tail_span_max,
        "Gamma_dtn_R120_over_R80_min": r_ratio_min,
        "Gamma_dtn_R120_over_R80_max": r_ratio_max,
        "parity_Gamma_rel_split_max": parity_gamma_rel_max,
        "parity_E_abs_split_max": parity_E_abs_max,
        "root_residual_abs_max": residual_abs_max,
        "dtn_residual_abs_max": dtn_residual_abs_max,
        "Gamma_dtn_over_geo_min": float(sub["Gamma_dtn_over_geo"].min()),
        "Gamma_dtn_over_geo_max": float(sub["Gamma_dtn_over_geo"].max()),
        "exact_dtn_pass": float(pass_gate),
        "exact_dtn_status": "physical_exact_dtn_pole" if pass_gate else "not_promoted",
        "fail_reasons": "pass" if pass_gate else ";".join(reasons),
        "recommended_next_gate": (
            "promote_static_width_after_cross_check"
            if pass_gate
            else next_gate
        ),
    }


def summarize(detail: pd.DataFrame, cfg: ExactDtnConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in detail.groupby("D", sort=True):
        row: dict[str, float | str] = {"D": float(D)}
        row.update(summarize_D(sub, cfg=cfg))
        rows.append(row)

    global_next_gate = (
        "no_static_width_promotion_without_new_exterior_stable_pole_theorem"
        if cfg.tail_mode == "whittaker"
        else "tighten_exterior_asymptotic_or_ECS_before_static_width_promotion"
    )
    global_row: dict[str, float | str] = {
        "D": "global",
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "R_min": float(min(cfg.r_values)),
        "R_max": float(max(cfg.r_values)),
        "R_tail_min": float(min(cfg.r_tail_values)),
        "R_tail_max": float(max(cfg.r_tail_values)),
        "n_R": float(len(cfg.r_values)),
        "n_R_tail": float(len(cfg.r_tail_values)),
        "Gamma_dtn_min": float(detail["Gamma_dtn"].min()),
        "Gamma_dtn_max": float(detail["Gamma_dtn"].max()),
        "Gamma_dtn_candidate": float(max(r["Gamma_dtn_candidate"] for r in rows)),
        "Gamma_dtn_R_rel_span_max": float(max(r["Gamma_dtn_R_rel_span_max"] for r in rows)),
        "Gamma_dtn_Rtail_rel_span_max": float(max(r["Gamma_dtn_Rtail_rel_span_max"] for r in rows)),
        "Gamma_dtn_R120_over_R80_min": float(min(r["Gamma_dtn_R120_over_R80_min"] for r in rows)),
        "Gamma_dtn_R120_over_R80_max": float(max(r["Gamma_dtn_R120_over_R80_max"] for r in rows)),
        "parity_Gamma_rel_split_max": float(max(r["parity_Gamma_rel_split_max"] for r in rows)),
        "parity_E_abs_split_max": float(max(r["parity_E_abs_split_max"] for r in rows)),
        "root_residual_abs_max": float(detail["residual_abs"].max()),
        "dtn_residual_abs_max": float(detail["dtn_residual_abs"].max()),
        "Gamma_dtn_over_geo_min": float(detail["Gamma_dtn_over_geo"].min()),
        "Gamma_dtn_over_geo_max": float(detail["Gamma_dtn_over_geo"].max()),
        "exact_dtn_pass": float(all(r["exact_dtn_pass"] == 1.0 for r in rows)),
        "exact_dtn_status": "physical_exact_dtn_pole_all_D"
        if all(r["exact_dtn_pass"] == 1.0 for r in rows)
        else "not_promoted_global",
        "fail_reasons": ";".join(sorted(set(str(r["fail_reasons"]) for r in rows))),
        "recommended_next_gate": global_next_gate,
    }
    rows.append(global_row)
    return pd.DataFrame(rows)


def parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Riccati/DtN static-width extraction audit.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values.")
    ap.add_argument("--R-values", default="60,80,100,120", help="Comma-separated interface radii.")
    ap.add_argument("--R-tail-values", default="800", help="Comma-separated exterior start radii.")
    ap.add_argument("--tail-validation", action="store_true", help="Use the two-tail validation grid R_tail=400,800.")
    ap.add_argument("--fallback-guesses", action="store_true", help="Try Hermitian seed imaginary guesses in addition to local-Robin guesses.")
    ap.add_argument("--tail-mode", choices=("wkb", "coulomb_series", "whittaker"), default="wkb", help="Outgoing normalization used at R_tail.")
    ap.add_argument("--coulomb-order", type=int, default=6, help="Coulomb-series order when --tail-mode=coulomb_series.")
    ap.add_argument("--whittaker-dps", type=int, default=50, help="mpmath precision for --tail-mode=whittaker.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        ExactDtnConfig(),
        d_values=parse_floats(args.Ds),
        r_values=parse_floats(args.R_values),
        r_tail_values=(400.0, 800.0) if args.tail_validation else parse_floats(args.R_tail_values),
        fallback_guesses=bool(args.fallback_guesses),
        tail_mode=args.tail_mode,
        coulomb_order=args.coulomb_order,
        whittaker_dps=args.whittaker_dps,
    )
    p = PhysicalParams()
    local_guess_map = load_local_guess_map()

    detail_rows = []
    total = (
        len(cfg.d_values)
        * len(cfg.r_tail_values)
        * len(cfg.r_values)
        * len(cfg.parity_values)
    )
    idx = 0
    for D in cfg.d_values:
        for R_tail in cfg.r_tail_values:
            for R in cfg.r_values:
                for parity in cfg.parity_values:
                    idx += 1
                    print(
                        f"[{idx}/{total}] D={D:g} R={R:g} R_tail={R_tail:g} parity={parity}",
                        flush=True,
                    )
                    row = solve_exact_dtn_case(
                        D=D,
                        R=R,
                        R_tail=R_tail,
                        parity=parity,
                        cfg=cfg,
                        p=p,
                        local_guess_map=local_guess_map,
                    )
                    detail_rows.append(row)
                    print(
                        "    "
                        f"Gamma={row['Gamma_dtn']:.8e} "
                        f"res={row['residual_abs']:.3e} "
                        f"m_R={row['m_R_real']:.3e}+{row['m_R_imag']:.3e}i",
                        flush=True,
                    )
    detail = pd.DataFrame(detail_rows)
    summary = summarize(detail, cfg=cfg)

    out_tag = "static_width_exact_dtn"
    if cfg.tail_mode == "coulomb_series":
        out_tag = f"static_width_coulomb_tail_dtn_o{cfg.coulomb_order}"
    elif cfg.tail_mode == "whittaker":
        out_tag = f"static_width_whittaker_dtn_dps{cfg.whittaker_dps}"
    detail_out = OUTDIR / f"{out_tag}_detail.csv"
    summary_out = OUTDIR / f"{out_tag}_summary.csv"
    detail.to_csv(detail_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")
    print("[static-width exact-DtN/Riccati audit summary]")
    cols = [
        "D",
        "exact_dtn_status",
        "Gamma_dtn_candidate",
        "Gamma_dtn_R_rel_span_max",
        "Gamma_dtn_Rtail_rel_span_max",
        "Gamma_dtn_R120_over_R80_min",
        "Gamma_dtn_R120_over_R80_max",
        "parity_Gamma_rel_split_max",
        "root_residual_abs_max",
        "recommended_next_gate",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()

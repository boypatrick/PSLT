#!/usr/bin/env python3
"""
Static-width complex-pole / Siegert extraction audit.

This audit keeps the static-width line narrow.  It does not promote the existing
CAP widths into physics.  Instead it writes down and tests the finite-radius
Siegert gate for the same action-derived 1D operator used by the current
geometry-width surrogate:

    H_{D,l} = -d^2/dz^2 + U(z;D) + l(l+1)/(z^2+eps^2).

On the symmetric half-line, even/odd parity modes satisfy a parity condition at
z=0 and an outgoing Robin/Siegert condition at z=R.  A genuine resonance should
produce a complex pole that is stable as R is moved outward.  The current audit
therefore checks:

  1. finite-radius outgoing roots for D={6,12,18}, l=1, N=3,
  2. parity splitting of those roots,
  3. R-stability of the extracted widths,
  4. consistency with the existing CAP negative-control diagnostics.

Outputs:
  - output/cap_resonance_1d/static_width_siegert_audit_detail.csv
  - output/cap_resonance_1d/static_width_siegert_audit_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import root

from extract_cap_resonance_1d import DEFAULT_OUTDIR, hermitian_seed_state
from extract_superrad_prefactor_1d import PhysicalParams, u_z, wkb_action_central


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = DEFAULT_OUTDIR


@dataclass(frozen=True)
class AuditConfig:
    ell: int = 1
    n_target: int = 3
    n_half: int = 1200
    d_values: tuple[float, ...] = (6.0, 12.0, 18.0)
    r_values: tuple[float, ...] = (60.0, 80.0, 100.0, 120.0)
    parity_values: tuple[str, ...] = ("even", "odd")
    imag_guesses: tuple[float, ...] = (-1.0e-5, -1.0e-4, -1.0e-3, -2.0e-3)
    root_xtol: float = 1.0e-11
    root_maxfev: int = 120


def u_eff(z: np.ndarray, D: float, ell: int, p: PhysicalParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return u_z(z, D, p) + (ell * (ell + 1)) / (z * z + p.eps * p.eps)


def relative_span(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan
    scale = max(float(np.median(np.abs(vals))), 1.0e-300)
    return float((vals.max() - vals.min()) / scale)


def outgoing_q(E: complex, U_R: float) -> complex:
    q = np.sqrt(E - U_R + 0j)
    if q.real < 0.0:
        q = -q
    return q


def ratio_residual(E: complex, U: np.ndarray, h: float, parity: str) -> complex:
    """Finite-difference half-line outgoing residual.

    The ratio recurrence avoids exponentially large transfer-matrix entries.
    With grid z_i=i h, r_i=psi_i/psi_{i-1}.  The right boundary uses the
    central-ghost outgoing condition

        (psi_{N+1}-psi_{N-1})/(2h) = i q_R(E) psi_N,

    which gives the final row residual after division by psi_N.
    """

    E = complex(E)
    n = len(U) - 1
    if parity == "even":
        # Neumann at z=0: psi_1 = psi_0 + h^2(U_0-E) psi_0 / 2.
        ratio = 1.0 + 0.5 * h * h * (U[0] - E)
        start = 1
    elif parity == "odd":
        # Dirichlet at z=0: psi_0=0, psi_1 is arbitrary.
        ratio = 2.0 + h * h * (U[1] - E)
        start = 2
    else:
        raise ValueError(f"Unknown parity: {parity}")

    for i in range(start, n):
        if abs(ratio) < 1.0e-300:
            ratio = 1.0e-300 + 0j
        ratio = (2.0 + h * h * (U[i] - E)) - 1.0 / ratio

    q = outgoing_q(E, float(U[-1]))
    return -2.0 / (h * h * ratio) + (2.0 / (h * h) - 2j * q / h + U[-1] - E)


def solve_siegert_case(
    D: float,
    R: float,
    parity: str,
    cfg: AuditConfig,
    p: PhysicalParams,
) -> dict[str, float | str]:
    h = R / cfg.n_half
    z_half = np.linspace(0.0, R, cfg.n_half + 1)
    U_half = u_eff(z_half, D=D, ell=cfg.ell, p=p)

    # Seed from the matching Hermitian box on [-R,R].
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

    def fun(x: np.ndarray) -> list[float]:
        val = ratio_residual(complex(x[0], x[1]), U=U_half, h=h, parity=parity)
        return [float(val.real), float(val.imag)]

    best = None
    for guess_im in cfg.imag_guesses:
        sol = root(
            fun,
            np.array([seed_E, guess_im], dtype=float),
            method="hybr",
            options={"xtol": cfg.root_xtol, "maxfev": cfg.root_maxfev},
        )
        E = complex(float(sol.x[0]), float(sol.x[1]))
        res_abs = abs(ratio_residual(E, U=U_half, h=h, parity=parity))
        row = {
            "root_success": float(bool(sol.success)),
            "root_nfev": float(sol.nfev),
            "guess_E_imag": float(guess_im),
            "E_siegert_real": float(E.real),
            "E_siegert_imag": float(E.imag),
            "residual_abs": float(res_abs),
        }
        if best is None or (row["residual_abs"], -row["root_success"]) < (
            best["residual_abs"],
            -best["root_success"],
        ):
            best = row

    assert best is not None
    E_root = complex(best["E_siegert_real"], best["E_siegert_imag"])
    omega_root = np.sqrt(p.m0 * p.m0 + E_root + 0j)
    if omega_root.real < 0.0:
        omega_root = -omega_root

    gamma_width = max(0.0, -2.0 * omega_root.imag)
    gamma_energy = max(0.0, -2.0 * E_root.imag)
    gamma_narrow = gamma_energy / max(2.0 * max(omega_root.real, 0.0), 1.0e-300)
    q_R = outgoing_q(E_root, float(U_half[-1]))

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
        "n_half": float(cfg.n_half),
        "h": float(h),
        "parity": parity,
        "seed_E": float(seed_E),
        "seed_omega": float(seed_omega),
        "E_siegert_real": best["E_siegert_real"],
        "E_siegert_imag": best["E_siegert_imag"],
        "omega_siegert_real": float(omega_root.real),
        "omega_siegert_imag": float(omega_root.imag),
        "Gamma_siegert": float(gamma_width),
        "Gamma_siegert_narrow": float(gamma_narrow),
        "rel_Gamma_exact_vs_narrow": float(
            abs(gamma_width - gamma_narrow) / max(abs(gamma_width), 1.0e-300)
        ),
        "q_R_real": float(q_R.real),
        "q_R_imag": float(q_R.imag),
        "U_R": float(U_half[-1]),
        "S_seed": float(S_seed) if np.isfinite(S_seed) else np.nan,
        "Gamma_geo_seed": float(gamma_geo) if np.isfinite(gamma_geo) else np.nan,
        "Gamma_siegert_over_geo": float(gamma_width / max(gamma_geo, 1.0e-300))
        if np.isfinite(gamma_geo)
        else np.nan,
        **best,
    }


def cap_negative_control_summary() -> dict[str, float | str]:
    """Read existing CAP diagnostics and compress their plateau evidence."""

    eta_path = OUTDIR / (
        "cap_resonance_scan_D6-12-18_l1_N3_eta5_5e-05_to_7p5e-05_"
        "zcap5_0p8_to_0p9_eta_summary.csv"
    )
    zcap_path = OUTDIR / (
        "cap_resonance_scan_D6-12-18_l1_N3_eta5_5e-05_to_7p5e-05_"
        "zcap5_0p8_to_0p9_zcap_summary.csv"
    )

    out: dict[str, float | str] = {
        "cap_negative_control_source": "missing",
        "cap_eta_Gamma_rel_span_median": np.nan,
        "cap_eta_Gamma_over_eta_rel_span_max": np.nan,
        "cap_zcap_Gamma_rel_span_min": np.nan,
        "cap_overlap_min": np.nan,
    }
    if eta_path.exists() and zcap_path.exists():
        eta = pd.read_csv(eta_path)
        zcap = pd.read_csv(zcap_path)
        out.update(
            {
                "cap_negative_control_source": str(eta_path.name),
                "cap_eta_Gamma_rel_span_median": float(eta["Gamma_cap_rel_span"].median()),
                "cap_eta_Gamma_over_eta_rel_span_max": float(
                    eta["Gamma_cap_over_eta_rel_span"].max()
                ),
                "cap_zcap_Gamma_rel_span_min": float(zcap["Gamma_cap_rel_span"].min()),
                "cap_overlap_min": float(
                    min(eta["candidate_overlap_min"].min(), zcap["candidate_overlap_min"].min())
                ),
            }
        )
    return out


def summarize(detail: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    cap = cap_negative_control_summary()
    rows: list[dict[str, float | str]] = []
    for D, sub in detail.groupby("D", sort=True):
        by_R = sub.groupby("R", sort=True)["Gamma_siegert"].mean()
        even = sub[sub["parity"] == "even"].set_index("R")
        odd = sub[sub["parity"] == "odd"].set_index("R")
        common_R = sorted(set(even.index).intersection(set(odd.index)))
        parity_gamma_rel = []
        parity_E_abs = []
        for R in common_R:
            ge = float(even.loc[R, "Gamma_siegert"])
            go = float(odd.loc[R, "Gamma_siegert"])
            ee = complex(
                float(even.loc[R, "E_siegert_real"]),
                float(even.loc[R, "E_siegert_imag"]),
            )
            eo = complex(
                float(odd.loc[R, "E_siegert_real"]),
                float(odd.loc[R, "E_siegert_imag"]),
            )
            parity_gamma_rel.append(abs(ge - go) / max(0.5 * (abs(ge) + abs(go)), 1e-300))
            parity_E_abs.append(abs(ee - eo))

        gamma_R80 = float(by_R.loc[80.0]) if 80.0 in by_R.index else np.nan
        gamma_R120 = float(by_R.loc[120.0]) if 120.0 in by_R.index else np.nan
        row: dict[str, float | str] = {
            "D": float(D),
            "ell": float(cfg.ell),
            "N_target": float(cfg.n_target),
            "R_min": float(min(cfg.r_values)),
            "R_max": float(max(cfg.r_values)),
            "n_R": float(len(cfg.r_values)),
            "Gamma_siegert_min": float(sub["Gamma_siegert"].min()),
            "Gamma_siegert_max": float(sub["Gamma_siegert"].max()),
            "Gamma_siegert_R_rel_span": relative_span(by_R.to_numpy(dtype=float)),
            "Gamma_siegert_R120_over_R80": float(gamma_R120 / gamma_R80)
            if np.isfinite(gamma_R80) and gamma_R80 != 0.0
            else np.nan,
            "E_real_R_rel_span": relative_span(
                sub.groupby("R", sort=True)["E_siegert_real"].mean().to_numpy(dtype=float)
            ),
            "parity_Gamma_rel_split_max": float(max(parity_gamma_rel)) if parity_gamma_rel else np.nan,
            "parity_E_abs_split_max": float(max(parity_E_abs)) if parity_E_abs else np.nan,
            "root_residual_abs_max": float(sub["residual_abs"].max()),
            "Gamma_siegert_over_geo_min": float(sub["Gamma_siegert_over_geo"].min()),
            "Gamma_siegert_over_geo_max": float(sub["Gamma_siegert_over_geo"].max()),
            "siegert_status": "no_R_stable_pole_plateau_on_current_finite_radius_audit",
            "recommended_next_gate": (
                "replace_local_robin_by_exact_Coulomb_DtN_or_ECS_and_require_R_stability"
            ),
        }
        row.update(cap)
        rows.append(row)

    global_row: dict[str, float | str] = {
        "D": "global",
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "R_min": float(min(cfg.r_values)),
        "R_max": float(max(cfg.r_values)),
        "n_R": float(len(cfg.r_values)),
        "Gamma_siegert_min": float(detail["Gamma_siegert"].min()),
        "Gamma_siegert_max": float(detail["Gamma_siegert"].max()),
        "Gamma_siegert_R_rel_span": float(
            max(r["Gamma_siegert_R_rel_span"] for r in rows)
        ),
        "Gamma_siegert_R120_over_R80": float(
            min(r["Gamma_siegert_R120_over_R80"] for r in rows)
        ),
        "E_real_R_rel_span": float(max(r["E_real_R_rel_span"] for r in rows)),
        "parity_Gamma_rel_split_max": float(
            max(r["parity_Gamma_rel_split_max"] for r in rows)
        ),
        "parity_E_abs_split_max": float(max(r["parity_E_abs_split_max"] for r in rows)),
        "root_residual_abs_max": float(detail["residual_abs"].max()),
        "Gamma_siegert_over_geo_min": float(detail["Gamma_siegert_over_geo"].min()),
        "Gamma_siegert_over_geo_max": float(detail["Gamma_siegert_over_geo"].max()),
        "siegert_status": "gate_not_closed_current_robin_roots_are_boundary_sensitive",
        "recommended_next_gate": (
            "exact_Coulomb_DtN_or_exterior_complex_scaling_before_promoting_static_width"
        ),
    }
    global_row.update(cap)
    rows.append(global_row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = AuditConfig()
    p = PhysicalParams()

    detail_rows = [
        solve_siegert_case(D=D, R=R, parity=parity, cfg=cfg, p=p)
        for D in cfg.d_values
        for R in cfg.r_values
        for parity in cfg.parity_values
    ]
    detail = pd.DataFrame(detail_rows)
    summary = summarize(detail, cfg=cfg)

    detail_out = OUTDIR / "static_width_siegert_audit_detail.csv"
    summary_out = OUTDIR / "static_width_siegert_audit_summary.csv"
    detail.to_csv(detail_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")
    print("[static-width Siegert audit summary]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

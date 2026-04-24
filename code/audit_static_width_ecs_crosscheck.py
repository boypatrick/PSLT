#!/usr/bin/env python3
"""
Exterior-complex-scaling (ECS) cross-check for the static-width gate.

This is the next controlled exterior audit after the local Robin, Riccati-DtN,
and analytic Coulomb-tail normalization tests.  It does not introduce a new
width family.  It asks whether the same action-derived one-dimensional
operator has a resonance pole that is stable under exterior complex scaling:

    Z_{theta,R0}(x) = x,                         |x| <= R0,
                    = sign(x)(R0 + e^{i theta}(|x|-R0)), |x| > R0.

For phi(x)=psi(Z(x)) and J=dZ/dx, the unweighted ECS operator is discretized in
flux form

    H_ECS phi = - J^{-1} d_x(J^{-1} d_x phi) + U_l(Z(x);D) phi.

The audit tracks the N=3, ell=1 seed by interior overlap and requires the
resulting complex pole to be stable as theta, R0, and the exterior box length L
are varied.

Outputs:
  - output/cap_resonance_1d/static_width_ecs_crosscheck_detail.csv
  - output/cap_resonance_1d/static_width_ecs_crosscheck_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from audit_static_width_siegert_extraction import OUTDIR, relative_span
from extract_cap_resonance_1d import hermitian_seed_state
from extract_superrad_prefactor_1d import PhysicalParams, wkb_action_central


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EcsConfig:
    ell: int = 1
    n_target: int = 3
    d_values: tuple[float, ...] = (6.0, 12.0, 18.0)
    theta_values: tuple[float, ...] = (0.08, 0.12, 0.16)
    r0_values: tuple[float, ...] = (50.0, 60.0)
    l_values: tuple[float, ...] = (100.0, 120.0)
    h: float = 0.15
    k_eigs: int = 12
    tol: float = 1.0e-9
    maxiter: int = 30000

    # Promotion gates.  These are intentionally comparable to the strict DtN
    # gates: an ECS pole must not move materially with theta, R0, or L.
    theta_gamma_rel_span_max: float = 0.15
    r0_gamma_rel_span_max: float = 0.15
    l_gamma_rel_span_max: float = 0.10
    theta_E_abs_span_max: float = 2.5e-4
    r0_E_abs_span_max: float = 2.5e-4
    l_E_abs_span_max: float = 2.5e-4
    candidate_overlap_min: float = 0.50
    interior_weight_min: float = 0.50
    eig_residual_abs_max: float = 1.0e-6


def parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def odd_grid_size(L: float, h: float) -> int:
    n_intervals = max(8, int(round(2.0 * L / h)))
    if n_intervals % 2 == 1:
        n_intervals += 1
    return n_intervals + 1


def omega_z_complex(z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    r1 = np.sqrt((z - D / 2.0) ** 2 + p.eps**2)
    r2 = np.sqrt((z + D / 2.0) ** 2 + p.eps**2)
    return 1.0 + p.a * (1.0 / r1 + 1.0 / r2)


def laplace_omega_z_complex(z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    r1 = np.sqrt((z - D / 2.0) ** 2 + p.eps**2)
    r2 = np.sqrt((z + D / 2.0) ** 2 + p.eps**2)
    lap1 = -3.0 * p.eps**2 / (r1**5)
    lap2 = -3.0 * p.eps**2 / (r2**5)
    return p.a * (lap1 + lap2)


def u_eff_complex(z: np.ndarray, D: float, ell: int, p: PhysicalParams) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    om = omega_z_complex(z, D=D, p=p)
    lap_om = laplace_omega_z_complex(z, D=D, p=p)
    base = p.m0**2 * (om**2 - 1.0) + (1.0 - 6.0 * p.xi) * (lap_om / om)
    centrifugal = (ell * (ell + 1.0)) / (z * z + p.eps * p.eps)
    return base + centrifugal


def ecs_coordinate(x: np.ndarray, R0: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    sign = np.sign(x)
    scale = np.exp(1j * theta)
    outside = ax > R0
    Z = x.astype(complex)
    Z[outside] = sign[outside] * (R0 + scale * (ax[outside] - R0))
    J = np.ones_like(Z, dtype=complex)
    J[outside] = scale
    return Z, J


def build_ecs_operator(
    x: np.ndarray,
    D: float,
    ell: int,
    R0: float,
    theta: float,
    p: PhysicalParams,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    Z, J = ecs_coordinate(x, R0=R0, theta=theta)
    V = u_eff_complex(Z, D=D, ell=ell, p=p)
    h = float(x[1] - x[0])

    # Flux discretization of -J^{-1} d_x(J^{-1} d_x).  The half-grid value is a
    # local average of J, which enforces the scaled derivative matching across
    # the two piecewise-constant ECS interfaces.
    J_half = 0.5 * (J[:-1] + J[1:])
    inv_J_half = 1.0 / J_half
    inv_J_i = 1.0 / J[1:-1]
    a_left = inv_J_half[:-1]
    a_right = inv_J_half[1:]

    main = inv_J_i * (a_left + a_right) / (h * h) + V[1:-1]
    upper = -inv_J_i[:-1] * a_right[:-1] / (h * h)
    lower = -inv_J_i[1:] * a_left[1:] / (h * h)
    H = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="csr")
    return H, Z, J, V


def eig_residual_norm(H: sp.csr_matrix, E: complex, psi: np.ndarray) -> float:
    defect = H @ psi - E * psi
    return float(np.linalg.norm(defect) / max(np.linalg.norm(psi), 1.0e-300))


def solve_ecs_case(
    D: float,
    theta: float,
    R0: float,
    L: float,
    cfg: EcsConfig,
    p: PhysicalParams,
) -> dict[str, float | str]:
    Nz = odd_grid_size(L=L, h=cfg.h)
    x = np.linspace(-L, L, Nz)
    h = float(x[1] - x[0])
    U_real = np.asarray(u_eff_complex(x, D=D, ell=cfg.ell, p=p).real, dtype=float)
    seed = hermitian_seed_state(
        U_real,
        x,
        n_target=cfg.n_target,
        n_states=max(cfg.n_target + 8, cfg.k_eigs + 2),
    )
    seed_E = float(seed["seed_E"])
    seed_psi = np.asarray(seed["seed_psi"], dtype=complex)
    seed_psi /= np.sqrt(np.sum(np.abs(seed_psi) ** 2) * h)

    H, Z, J, V = build_ecs_operator(
        x=x,
        D=D,
        ell=cfg.ell,
        R0=R0,
        theta=theta,
        p=p,
    )
    evals, evecs = spla.eigs(
        H,
        k=cfg.k_eigs,
        sigma=seed_E,
        which="LM",
        tol=cfg.tol,
        maxiter=cfg.maxiter,
    )

    x_i = x[1:-1]
    interior_mask = np.abs(x_i) <= R0
    rows: list[dict[str, float]] = []
    for j, E in enumerate(evals):
        vec = np.asarray(evecs[:, j], dtype=complex)
        vec /= np.sqrt(np.sum(np.abs(vec) ** 2) * h)
        overlap = abs(np.vdot(seed_psi, vec)) * h
        interior_weight = float(np.sum(np.abs(vec[interior_mask]) ** 2) * h)
        seed_dist = abs(E - seed_E)
        residual = eig_residual_norm(H, E, vec)
        omega = np.sqrt(p.m0 * p.m0 + E + 0j)
        if omega.real < 0.0:
            omega = -omega
        rows.append(
            {
                "candidate_index": float(j),
                "E_ecs_real": float(E.real),
                "E_ecs_imag": float(E.imag),
                "omega_ecs_real": float(omega.real),
                "omega_ecs_imag": float(omega.imag),
                "Gamma_ecs": float(max(0.0, -2.0 * omega.imag)),
                "Gamma_ecs_narrow": float(
                    max(0.0, -2.0 * E.imag)
                    / max(2.0 * max(float(omega.real), 0.0), 1.0e-300)
                ),
                "candidate_overlap": float(overlap),
                "candidate_interior_weight": interior_weight,
                "candidate_seed_dist": float(seed_dist),
                "eig_residual_abs": residual,
            }
        )

    cand = pd.DataFrame(rows)
    non_growing = cand[cand["E_ecs_imag"] <= 1.0e-10]
    pool = non_growing if len(non_growing) else cand
    good_res = pool[pool["eig_residual_abs"] <= cfg.eig_residual_abs_max]
    pool = good_res if len(good_res) else pool
    best = pool.sort_values(
        [
            "candidate_overlap",
            "candidate_interior_weight",
            "candidate_seed_dist",
            "eig_residual_abs",
        ],
        ascending=[False, False, True, True],
    ).iloc[0]

    E_root = complex(float(best["E_ecs_real"]), float(best["E_ecs_imag"]))
    omega_root = np.sqrt(p.m0 * p.m0 + E_root + 0j)
    if omega_root.real < 0.0:
        omega_root = -omega_root
    S_seed = wkb_action_central(U_real, x, seed_E)
    seed_omega = float(np.sqrt(p.m0 * p.m0 + seed_E))
    gamma_geo = seed_omega * np.exp(-2.0 * S_seed) if np.isfinite(S_seed) else np.nan

    return {
        "D": float(D),
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "theta": float(theta),
        "R0": float(R0),
        "L": float(L),
        "Nz": float(Nz),
        "h": float(h),
        "exterior_length": float(L - R0),
        "seed_E": seed_E,
        "seed_omega": seed_omega,
        "seed_bound": float(seed_E < 0.0),
        "S_seed": float(S_seed) if np.isfinite(S_seed) else np.nan,
        "Gamma_geo_seed": float(gamma_geo) if np.isfinite(gamma_geo) else np.nan,
        "Gamma_ecs_over_geo": float(best["Gamma_ecs"] / max(gamma_geo, 1.0e-300))
        if np.isfinite(gamma_geo)
        else np.nan,
        "Z_left_imag": float(Z[1].imag),
        "Z_right_imag": float(Z[-2].imag),
        "J_exterior_real": float(np.exp(1j * theta).real),
        "J_exterior_imag": float(np.exp(1j * theta).imag),
        "V_imag_abs_max": float(np.max(np.abs(V[1:-1].imag))),
        "candidate_count": float(len(cand)),
        **{k: float(best[k]) for k in best.index},
    }


def finite_median(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))


def complex_abs_span(sub: pd.DataFrame) -> float:
    vals = np.array(
        [complex(float(r.E_ecs_real), float(r.E_ecs_imag)) for r in sub.itertuples()],
        dtype=complex,
    )
    vals = vals[np.isfinite(vals.real) & np.isfinite(vals.imag)]
    if len(vals) < 2:
        return np.nan
    return float(np.max(np.abs(vals[:, None] - vals[None, :])))


def grouped_rel_span(df: pd.DataFrame, vary: str, fixed: list[str], value: str) -> float:
    spans = []
    for _, sub in df.groupby(fixed, sort=True):
        if sub[vary].nunique() >= 2:
            spans.append(relative_span(sub[value].to_numpy(dtype=float)))
    return float(np.nanmax(spans)) if spans else np.nan


def grouped_complex_span(df: pd.DataFrame, vary: str, fixed: list[str]) -> float:
    spans = []
    for _, sub in df.groupby(fixed, sort=True):
        if sub[vary].nunique() >= 2:
            spans.append(complex_abs_span(sub))
    return float(np.nanmax(spans)) if spans else np.nan


def summarize_D(sub: pd.DataFrame, cfg: EcsConfig) -> dict[str, float | str]:
    theta_gamma_span = grouped_rel_span(
        sub, vary="theta", fixed=["R0", "L"], value="Gamma_ecs"
    )
    r0_gamma_span = grouped_rel_span(
        sub, vary="R0", fixed=["theta", "L"], value="Gamma_ecs"
    )
    l_gamma_span = grouped_rel_span(
        sub, vary="L", fixed=["theta", "R0"], value="Gamma_ecs"
    )
    theta_E_span = grouped_complex_span(sub, vary="theta", fixed=["R0", "L"])
    r0_E_span = grouped_complex_span(sub, vary="R0", fixed=["theta", "L"])
    l_E_span = grouped_complex_span(sub, vary="L", fixed=["theta", "R0"])

    reasons: list[str] = []
    if theta_gamma_span > cfg.theta_gamma_rel_span_max:
        reasons.append("theta_width_not_stationary")
    if r0_gamma_span > cfg.r0_gamma_rel_span_max:
        reasons.append("R0_width_not_stationary")
    if l_gamma_span > cfg.l_gamma_rel_span_max:
        reasons.append("L_width_not_stationary")
    if theta_E_span > cfg.theta_E_abs_span_max:
        reasons.append("theta_E_not_stationary")
    if r0_E_span > cfg.r0_E_abs_span_max:
        reasons.append("R0_E_not_stationary")
    if l_E_span > cfg.l_E_abs_span_max:
        reasons.append("L_E_not_stationary")
    if float(sub["candidate_overlap"].min()) < cfg.candidate_overlap_min:
        reasons.append("seed_overlap_low")
    if float(sub["candidate_interior_weight"].min()) < cfg.interior_weight_min:
        reasons.append("interior_weight_low")
    if float(sub["eig_residual_abs"].max()) > cfg.eig_residual_abs_max:
        reasons.append("eig_residual_large")
    if float(sub["E_ecs_imag"].max()) > 1.0e-10:
        reasons.append("growing_candidate")

    pass_gate = len(reasons) == 0
    return {
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "theta_min": float(min(cfg.theta_values)),
        "theta_max": float(max(cfg.theta_values)),
        "R0_min": float(min(cfg.r0_values)),
        "R0_max": float(max(cfg.r0_values)),
        "L_min": float(min(cfg.l_values)),
        "L_max": float(max(cfg.l_values)),
        "n_theta": float(len(cfg.theta_values)),
        "n_R0": float(len(cfg.r0_values)),
        "n_L": float(len(cfg.l_values)),
        "Gamma_ecs_min": float(sub["Gamma_ecs"].min()),
        "Gamma_ecs_max": float(sub["Gamma_ecs"].max()),
        "Gamma_ecs_candidate": finite_median(sub["Gamma_ecs"].to_numpy(dtype=float)),
        "Gamma_ecs_theta_rel_span_max": theta_gamma_span,
        "Gamma_ecs_R0_rel_span_max": r0_gamma_span,
        "Gamma_ecs_L_rel_span_max": l_gamma_span,
        "E_ecs_theta_abs_span_max": theta_E_span,
        "E_ecs_R0_abs_span_max": r0_E_span,
        "E_ecs_L_abs_span_max": l_E_span,
        "candidate_overlap_min": float(sub["candidate_overlap"].min()),
        "candidate_interior_weight_min": float(sub["candidate_interior_weight"].min()),
        "eig_residual_abs_max": float(sub["eig_residual_abs"].max()),
        "Gamma_ecs_over_geo_min": float(sub["Gamma_ecs_over_geo"].min()),
        "Gamma_ecs_over_geo_max": float(sub["Gamma_ecs_over_geo"].max()),
        "ecs_pass": float(pass_gate),
        "ecs_status": "physical_ecs_pole" if pass_gate else "not_promoted",
        "fail_reasons": "pass" if pass_gate else ";".join(reasons),
        "recommended_next_gate": (
            "cross_confirm_with_full_Coulomb_Whittaker_DtN"
            if pass_gate
            else "full_Coulomb_Whittaker_DtN_or_refined_ECS_branch_tracking_before_promotion"
        ),
    }


def summarize(detail: pd.DataFrame, cfg: EcsConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in detail.groupby("D", sort=True):
        row: dict[str, float | str] = {"D": float(D)}
        row.update(summarize_D(sub, cfg=cfg))
        rows.append(row)

    global_reasons = sorted(
        {
            reason
            for row in rows
            for reason in str(row["fail_reasons"]).split(";")
            if reason and reason != "pass"
        }
    )
    global_row: dict[str, float | str] = {
        "D": "global",
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "theta_min": float(min(cfg.theta_values)),
        "theta_max": float(max(cfg.theta_values)),
        "R0_min": float(min(cfg.r0_values)),
        "R0_max": float(max(cfg.r0_values)),
        "L_min": float(min(cfg.l_values)),
        "L_max": float(max(cfg.l_values)),
        "n_theta": float(len(cfg.theta_values)),
        "n_R0": float(len(cfg.r0_values)),
        "n_L": float(len(cfg.l_values)),
        "Gamma_ecs_min": float(detail["Gamma_ecs"].min()),
        "Gamma_ecs_max": float(detail["Gamma_ecs"].max()),
        "Gamma_ecs_candidate": float(max(row["Gamma_ecs_candidate"] for row in rows)),
        "Gamma_ecs_theta_rel_span_max": float(
            max(row["Gamma_ecs_theta_rel_span_max"] for row in rows)
        ),
        "Gamma_ecs_R0_rel_span_max": float(max(row["Gamma_ecs_R0_rel_span_max"] for row in rows)),
        "Gamma_ecs_L_rel_span_max": float(max(row["Gamma_ecs_L_rel_span_max"] for row in rows)),
        "E_ecs_theta_abs_span_max": float(max(row["E_ecs_theta_abs_span_max"] for row in rows)),
        "E_ecs_R0_abs_span_max": float(max(row["E_ecs_R0_abs_span_max"] for row in rows)),
        "E_ecs_L_abs_span_max": float(max(row["E_ecs_L_abs_span_max"] for row in rows)),
        "candidate_overlap_min": float(detail["candidate_overlap"].min()),
        "candidate_interior_weight_min": float(detail["candidate_interior_weight"].min()),
        "eig_residual_abs_max": float(detail["eig_residual_abs"].max()),
        "Gamma_ecs_over_geo_min": float(detail["Gamma_ecs_over_geo"].min()),
        "Gamma_ecs_over_geo_max": float(detail["Gamma_ecs_over_geo"].max()),
        "ecs_pass": float(all(row["ecs_pass"] == 1.0 for row in rows)),
        "ecs_status": "physical_ecs_pole_all_D"
        if all(row["ecs_pass"] == 1.0 for row in rows)
        else "not_promoted_global",
        "fail_reasons": "pass" if not global_reasons else ";".join(global_reasons),
        "recommended_next_gate": "full_Coulomb_Whittaker_DtN_or_refined_ECS_branch_tracking_before_promotion",
    }
    rows.append(global_row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="ECS cross-check for the static-width complex pole.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values.")
    ap.add_argument("--thetas", default="0.08,0.12,0.16", help="Comma-separated ECS angles in radians.")
    ap.add_argument("--R0-values", default="50,60", help="Comma-separated ECS onset radii.")
    ap.add_argument("--L-values", default="100,120", help="Comma-separated half-box lengths.")
    ap.add_argument("--h", type=float, default=0.15, help="Target grid spacing.")
    ap.add_argument("--k-eigs", type=int, default=12, help="Eigenpairs near the Hermitian seed.")
    ap.add_argument("--tol", type=float, default=1.0e-9, help="Sparse eigensolver tolerance.")
    args = ap.parse_args()

    cfg = EcsConfig(
        d_values=parse_floats(args.Ds),
        theta_values=parse_floats(args.thetas),
        r0_values=parse_floats(args.R0_values),
        l_values=parse_floats(args.L_values),
        h=float(args.h),
        k_eigs=int(args.k_eigs),
        tol=float(args.tol),
    )
    p = PhysicalParams()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    total = len(cfg.d_values) * len(cfg.theta_values) * len(cfg.r0_values) * len(cfg.l_values)
    idx = 0
    for D in cfg.d_values:
        for theta in cfg.theta_values:
            for R0 in cfg.r0_values:
                for L in cfg.l_values:
                    idx += 1
                    print(
                        f"[{idx}/{total}] D={D:g} theta={theta:g} R0={R0:g} L={L:g}",
                        flush=True,
                    )
                    try:
                        row = solve_ecs_case(D=D, theta=theta, R0=R0, L=L, cfg=cfg, p=p)
                    except Exception as exc:  # noqa: BLE001 - keep failed audit rows inspectable.
                        row = {
                            "D": float(D),
                            "ell": float(cfg.ell),
                            "N_target": float(cfg.n_target),
                            "theta": float(theta),
                            "R0": float(R0),
                            "L": float(L),
                            "exception": str(exc),
                            "E_ecs_real": np.nan,
                            "E_ecs_imag": np.nan,
                            "omega_ecs_real": np.nan,
                            "omega_ecs_imag": np.nan,
                            "Gamma_ecs": np.nan,
                            "candidate_overlap": np.nan,
                            "candidate_interior_weight": np.nan,
                            "eig_residual_abs": np.inf,
                            "Gamma_ecs_over_geo": np.nan,
                        }
                    rows.append(row)
                    print(
                        "    "
                        f"Gamma={row.get('Gamma_ecs', np.nan):.8e} "
                        f"E={row.get('E_ecs_real', np.nan):.8e}"
                        f"{row.get('E_ecs_imag', np.nan):+.3e}i "
                        f"overlap={row.get('candidate_overlap', np.nan):.3f}",
                        flush=True,
                    )

    detail = pd.DataFrame(rows)
    summary = summarize(detail, cfg=cfg)
    detail_out = OUTDIR / "static_width_ecs_crosscheck_detail.csv"
    summary_out = OUTDIR / "static_width_ecs_crosscheck_summary.csv"
    detail.to_csv(detail_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")
    print("[static-width ECS cross-check summary]")
    cols = [
        "D",
        "ecs_status",
        "Gamma_ecs_candidate",
        "Gamma_ecs_theta_rel_span_max",
        "Gamma_ecs_R0_rel_span_max",
        "Gamma_ecs_L_rel_span_max",
        "E_ecs_theta_abs_span_max",
        "E_ecs_R0_abs_span_max",
        "E_ecs_L_abs_span_max",
        "candidate_overlap_min",
        "eig_residual_abs_max",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()

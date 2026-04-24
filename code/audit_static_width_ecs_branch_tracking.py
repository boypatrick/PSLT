#!/usr/bin/env python3
"""
Refined ECS branch-tracking audit for the static-width gate.

Proposition S4 tracks the Hermitian seed branch by overlap.  This follow-up asks
the stronger question: is there any resolved ECS eigen-branch near the N=3,
ell=1 seed that remains stable under theta, scaling onset R0, and exterior box
length L?

The audit computes all sparse eigenvalues returned near the Hermitian seed for
each ECS parameter tuple.  For every eigenvalue used as an anchor, it forms an
optimistic nearest-neighbor continuation across the whole parameter grid:

    B_a(p) = argmin_j |E_j(p) - E_a|.

If a stable resonance pole is present in the resolved window, this optimistic
envelope should find a low-dispersion branch.  Failure of even this nearest
neighbor branch search is therefore a sharper negative control than a single
overlap-tracked ECS branch.

Outputs:
  - output/cap_resonance_1d/static_width_ecs_branch_spectrum_detail.csv
  - output/cap_resonance_1d/static_width_ecs_branch_candidates.csv
  - output/cap_resonance_1d/static_width_ecs_branch_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla

from audit_static_width_ecs_crosscheck import (
    EcsConfig,
    OUTDIR,
    build_ecs_operator,
    eig_residual_norm,
    odd_grid_size,
    parse_floats,
    u_eff_complex,
)
from audit_static_width_siegert_extraction import relative_span
from extract_cap_resonance_1d import hermitian_seed_state
from extract_superrad_prefactor_1d import PhysicalParams, wkb_action_central


def complex_pairwise_span(values: Iterable[complex]) -> float:
    vals = np.asarray(list(values), dtype=complex)
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
            vals = [complex(float(r.E_real), float(r.E_imag)) for r in sub.itertuples()]
            spans.append(complex_pairwise_span(vals))
    return float(np.nanmax(spans)) if spans else np.nan


def solve_ecs_spectrum_case(
    D: float,
    theta: float,
    R0: float,
    L: float,
    cfg: EcsConfig,
    p: PhysicalParams,
) -> list[dict[str, float]]:
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
    seed_omega = float(np.sqrt(p.m0 * p.m0 + seed_E))
    S_seed = wkb_action_central(U_real, x, seed_E)
    gamma_geo = seed_omega * np.exp(-2.0 * S_seed) if np.isfinite(S_seed) else np.nan

    H, _, _, _ = build_ecs_operator(
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
        residual = eig_residual_norm(H, E, vec)
        omega = np.sqrt(p.m0 * p.m0 + E + 0j)
        if omega.real < 0.0:
            omega = -omega
        Gamma = float(max(0.0, -2.0 * omega.imag))
        rows.append(
            {
                "D": float(D),
                "ell": float(cfg.ell),
                "N_target": float(cfg.n_target),
                "theta": float(theta),
                "R0": float(R0),
                "L": float(L),
                "Nz": float(Nz),
                "h": float(h),
                "case_id": float(theta * 1.0e6 + R0 * 1.0e3 + L),
                "candidate_index": float(j),
                "seed_E": seed_E,
                "seed_omega": seed_omega,
                "S_seed": float(S_seed) if np.isfinite(S_seed) else np.nan,
                "Gamma_geo_seed": float(gamma_geo) if np.isfinite(gamma_geo) else np.nan,
                "E_real": float(E.real),
                "E_imag": float(E.imag),
                "omega_real": float(omega.real),
                "omega_imag": float(omega.imag),
                "Gamma": Gamma,
                "Gamma_narrow": float(
                    max(0.0, -2.0 * E.imag)
                    / max(2.0 * max(float(omega.real), 0.0), 1.0e-300)
                ),
                "Gamma_over_geo": float(Gamma / max(gamma_geo, 1.0e-300))
                if np.isfinite(gamma_geo)
                else np.nan,
                "seed_overlap": float(overlap),
                "interior_weight": interior_weight,
                "seed_dist": float(abs(E - seed_E)),
                "eig_residual_abs": residual,
            }
        )
    return rows


def branch_from_anchor(
    spectrum: pd.DataFrame,
    anchor: pd.Series,
    cfg: EcsConfig,
    branch_id: int,
) -> dict[str, float | str]:
    anchor_E = complex(float(anchor["E_real"]), float(anchor["E_imag"]))
    selected_rows = []
    for _, case in spectrum.groupby(["theta", "R0", "L"], sort=True):
        distances = np.abs(
            case["E_real"].to_numpy(dtype=float)
            + 1j * case["E_imag"].to_numpy(dtype=float)
            - anchor_E
        )
        best = case.iloc[int(np.argmin(distances))].copy()
        best["branch_pick_dist"] = float(np.min(distances))
        selected_rows.append(best)
    branch = pd.DataFrame(selected_rows)
    vals = [complex(float(r.E_real), float(r.E_imag)) for r in branch.itertuples()]
    E_abs_span = complex_pairwise_span(vals)
    theta_gamma_span = grouped_rel_span(branch, vary="theta", fixed=["R0", "L"], value="Gamma")
    r0_gamma_span = grouped_rel_span(branch, vary="R0", fixed=["theta", "L"], value="Gamma")
    l_gamma_span = grouped_rel_span(branch, vary="L", fixed=["theta", "R0"], value="Gamma")
    theta_E_span = grouped_complex_span(branch, vary="theta", fixed=["R0", "L"])
    r0_E_span = grouped_complex_span(branch, vary="R0", fixed=["theta", "L"])
    l_E_span = grouped_complex_span(branch, vary="L", fixed=["theta", "R0"])

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
    if float(branch["seed_overlap"].min()) < cfg.candidate_overlap_min:
        reasons.append("seed_overlap_low")
    if float(branch["eig_residual_abs"].max()) > cfg.eig_residual_abs_max:
        reasons.append("eig_residual_large")
    if float(branch["E_imag"].max()) > 1.0e-10:
        reasons.append("growing_branch")

    score_terms = [
        theta_gamma_span / max(cfg.theta_gamma_rel_span_max, 1.0e-300),
        r0_gamma_span / max(cfg.r0_gamma_rel_span_max, 1.0e-300),
        l_gamma_span / max(cfg.l_gamma_rel_span_max, 1.0e-300),
        theta_E_span / max(cfg.theta_E_abs_span_max, 1.0e-300),
        r0_E_span / max(cfg.r0_E_abs_span_max, 1.0e-300),
        l_E_span / max(cfg.l_E_abs_span_max, 1.0e-300),
    ]
    score = float(np.nanmax(score_terms))

    return {
        "D": float(anchor["D"]),
        "branch_id": float(branch_id),
        "anchor_theta": float(anchor["theta"]),
        "anchor_R0": float(anchor["R0"]),
        "anchor_L": float(anchor["L"]),
        "anchor_candidate_index": float(anchor["candidate_index"]),
        "anchor_E_real": float(anchor["E_real"]),
        "anchor_E_imag": float(anchor["E_imag"]),
        "anchor_Gamma": float(anchor["Gamma"]),
        "n_cases": float(len(branch)),
        "Gamma_min": float(branch["Gamma"].min()),
        "Gamma_max": float(branch["Gamma"].max()),
        "Gamma_candidate": float(np.median(branch["Gamma"].to_numpy(dtype=float))),
        "E_abs_span": E_abs_span,
        "Gamma_theta_rel_span_max": theta_gamma_span,
        "Gamma_R0_rel_span_max": r0_gamma_span,
        "Gamma_L_rel_span_max": l_gamma_span,
        "E_theta_abs_span_max": theta_E_span,
        "E_R0_abs_span_max": r0_E_span,
        "E_L_abs_span_max": l_E_span,
        "seed_overlap_min": float(branch["seed_overlap"].min()),
        "seed_overlap_median": float(branch["seed_overlap"].median()),
        "interior_weight_min": float(branch["interior_weight"].min()),
        "eig_residual_abs_max": float(branch["eig_residual_abs"].max()),
        "branch_pick_dist_max": float(branch["branch_pick_dist"].max()),
        "branch_stability_score": score,
        "branch_pass": float(len(reasons) == 0),
        "branch_status": "ecs_stable_branch" if len(reasons) == 0 else "not_promoted",
        "fail_reasons": "pass" if len(reasons) == 0 else ";".join(reasons),
    }


def summarize_branches(branches: pd.DataFrame, cfg: EcsConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in branches.groupby("D", sort=True):
        passing = sub[sub["branch_pass"] == 1.0]
        seed_qualified = sub[sub["seed_overlap_min"] >= cfg.candidate_overlap_min]
        best = sub.sort_values(
            ["branch_pass", "branch_stability_score", "E_abs_span", "seed_overlap_median"],
            ascending=[False, True, True, False],
        ).iloc[0]
        best_seed = (
            seed_qualified.sort_values(
                ["branch_pass", "branch_stability_score", "E_abs_span", "seed_overlap_median"],
                ascending=[False, True, True, False],
            ).iloc[0]
            if len(seed_qualified)
            else None
        )
        rows.append(
            {
                "D": float(D),
                "ell": float(cfg.ell),
                "N_target": float(cfg.n_target),
                "n_branch_candidates": float(len(sub)),
                "n_seed_qualified_branches": float(len(seed_qualified)),
                "n_passing_branches": float(len(passing)),
                "best_branch_id": float(best["branch_id"]),
                "best_branch_status": str(best["branch_status"]),
                "best_Gamma_candidate": float(best["Gamma_candidate"]),
                "best_E_abs_span": float(best["E_abs_span"]),
                "best_Gamma_theta_rel_span_max": float(best["Gamma_theta_rel_span_max"]),
                "best_Gamma_R0_rel_span_max": float(best["Gamma_R0_rel_span_max"]),
                "best_Gamma_L_rel_span_max": float(best["Gamma_L_rel_span_max"]),
                "best_E_theta_abs_span_max": float(best["E_theta_abs_span_max"]),
                "best_E_R0_abs_span_max": float(best["E_R0_abs_span_max"]),
                "best_E_L_abs_span_max": float(best["E_L_abs_span_max"]),
                "best_seed_overlap_min": float(best["seed_overlap_min"]),
                "best_seed_overlap_median": float(best["seed_overlap_median"]),
                "best_eig_residual_abs_max": float(best["eig_residual_abs_max"]),
                "best_branch_stability_score": float(best["branch_stability_score"]),
                "best_seed_branch_id": np.nan if best_seed is None else float(best_seed["branch_id"]),
                "best_seed_Gamma_candidate": np.nan
                if best_seed is None
                else float(best_seed["Gamma_candidate"]),
                "best_seed_E_abs_span": np.nan if best_seed is None else float(best_seed["E_abs_span"]),
                "best_seed_Gamma_theta_rel_span_max": np.nan
                if best_seed is None
                else float(best_seed["Gamma_theta_rel_span_max"]),
                "best_seed_Gamma_R0_rel_span_max": np.nan
                if best_seed is None
                else float(best_seed["Gamma_R0_rel_span_max"]),
                "best_seed_Gamma_L_rel_span_max": np.nan
                if best_seed is None
                else float(best_seed["Gamma_L_rel_span_max"]),
                "best_seed_E_theta_abs_span_max": np.nan
                if best_seed is None
                else float(best_seed["E_theta_abs_span_max"]),
                "best_seed_E_R0_abs_span_max": np.nan
                if best_seed is None
                else float(best_seed["E_R0_abs_span_max"]),
                "best_seed_E_L_abs_span_max": np.nan
                if best_seed is None
                else float(best_seed["E_L_abs_span_max"]),
                "best_seed_seed_overlap_min": np.nan
                if best_seed is None
                else float(best_seed["seed_overlap_min"]),
                "best_seed_branch_stability_score": np.nan
                if best_seed is None
                else float(best_seed["branch_stability_score"]),
                "ecs_branch_status": "physical_ecs_branch" if len(passing) else "not_promoted",
                "fail_reasons": "pass" if len(passing) else str(best["fail_reasons"]),
                "recommended_next_gate": (
                    "cross_confirm_with_full_Coulomb_Whittaker_DtN"
                    if len(passing)
                    else "full_Coulomb_Whittaker_DtN_required_before_static_width_promotion"
                ),
            }
        )

    global_row: dict[str, float | str] = {
        "D": "global",
        "ell": float(cfg.ell),
        "N_target": float(cfg.n_target),
        "n_branch_candidates": float(len(branches)),
        "n_seed_qualified_branches": float(
            (branches["seed_overlap_min"] >= cfg.candidate_overlap_min).sum()
        ),
        "n_passing_branches": float((branches["branch_pass"] == 1.0).sum()),
        "best_branch_id": np.nan,
        "best_branch_status": "not_promoted_global"
        if (branches["branch_pass"] == 1.0).sum() == 0
        else "physical_ecs_branch_all_D",
        "best_Gamma_candidate": float(branches["Gamma_candidate"].median()),
        "best_E_abs_span": float(branches["E_abs_span"].min()),
        "best_Gamma_theta_rel_span_max": float(branches["Gamma_theta_rel_span_max"].min()),
        "best_Gamma_R0_rel_span_max": float(branches["Gamma_R0_rel_span_max"].min()),
        "best_Gamma_L_rel_span_max": float(branches["Gamma_L_rel_span_max"].min()),
        "best_E_theta_abs_span_max": float(branches["E_theta_abs_span_max"].min()),
        "best_E_R0_abs_span_max": float(branches["E_R0_abs_span_max"].min()),
        "best_E_L_abs_span_max": float(branches["E_L_abs_span_max"].min()),
        "best_seed_overlap_min": float(branches["seed_overlap_min"].max()),
        "best_seed_overlap_median": float(branches["seed_overlap_median"].max()),
        "best_eig_residual_abs_max": float(branches["eig_residual_abs_max"].max()),
        "best_branch_stability_score": float(branches["branch_stability_score"].min()),
        "best_seed_branch_id": np.nan,
        "best_seed_Gamma_candidate": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "Gamma_candidate",
            ].median()
        ),
        "best_seed_E_abs_span": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "E_abs_span",
            ].max()
        ),
        "best_seed_Gamma_theta_rel_span_max": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "Gamma_theta_rel_span_max",
            ].max()
        ),
        "best_seed_Gamma_R0_rel_span_max": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "Gamma_R0_rel_span_max",
            ].max()
        ),
        "best_seed_Gamma_L_rel_span_max": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "Gamma_L_rel_span_max",
            ].max()
        ),
        "best_seed_E_theta_abs_span_max": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "E_theta_abs_span_max",
            ].max()
        ),
        "best_seed_E_R0_abs_span_max": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "E_R0_abs_span_max",
            ].max()
        ),
        "best_seed_E_L_abs_span_max": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "E_L_abs_span_max",
            ].max()
        ),
        "best_seed_seed_overlap_min": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "seed_overlap_min",
            ].min()
        ),
        "best_seed_branch_stability_score": float(
            branches.loc[
                branches["seed_overlap_min"] >= cfg.candidate_overlap_min,
                "branch_stability_score",
            ].max()
        ),
        "ecs_branch_status": "physical_ecs_branch_all_D"
        if (branches["branch_pass"] == 1.0).sum() > 0
        else "not_promoted_global",
        "fail_reasons": "pass"
        if (branches["branch_pass"] == 1.0).sum() > 0
        else "no_resolved_branch_passes_all_ECS_stability_gates",
        "recommended_next_gate": "full_Coulomb_Whittaker_DtN_required_before_static_width_promotion",
    }
    rows.append(global_row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refined ECS branch-tracking audit.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values.")
    ap.add_argument("--thetas", default="0.08,0.12,0.16", help="Comma-separated ECS angles in radians.")
    ap.add_argument("--R0-values", default="50,60", help="Comma-separated ECS onset radii.")
    ap.add_argument("--L-values", default="100,120", help="Comma-separated half-box lengths.")
    ap.add_argument("--h", type=float, default=0.15, help="Target grid spacing.")
    ap.add_argument("--k-eigs", type=int, default=24, help="Eigenpairs near the Hermitian seed.")
    ap.add_argument("--tol", type=float, default=1.0e-9, help="Sparse eigensolver tolerance.")
    args = ap.parse_args()

    cfg = replace(
        EcsConfig(),
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

    spectrum_rows: list[dict[str, float]] = []
    total = len(cfg.d_values) * len(cfg.theta_values) * len(cfg.r0_values) * len(cfg.l_values)
    idx = 0
    for D in cfg.d_values:
        for theta in cfg.theta_values:
            for R0 in cfg.r0_values:
                for L in cfg.l_values:
                    idx += 1
                    print(
                        f"[{idx}/{total}] spectrum D={D:g} theta={theta:g} R0={R0:g} L={L:g}",
                        flush=True,
                    )
                    rows = solve_ecs_spectrum_case(D=D, theta=theta, R0=R0, L=L, cfg=cfg, p=p)
                    spectrum_rows.extend(rows)
                    frame = pd.DataFrame(rows)
                    best_overlap = frame.sort_values("seed_overlap", ascending=False).iloc[0]
                    print(
                        "    "
                        f"n={len(rows)} best_overlap={best_overlap['seed_overlap']:.3f} "
                        f"E={best_overlap['E_real']:.8e}{best_overlap['E_imag']:+.3e}i "
                        f"Gamma={best_overlap['Gamma']:.8e}",
                        flush=True,
                    )

    spectrum = pd.DataFrame(spectrum_rows)
    branch_rows: list[dict[str, float | str]] = []
    branch_id = 0
    for D, sub in spectrum.groupby("D", sort=True):
        anchors = sub[
            (sub["E_imag"] <= 1.0e-10)
            & (sub["eig_residual_abs"] <= cfg.eig_residual_abs_max)
            & (sub["seed_overlap"] >= 0.05)
        ].copy()
        for _, anchor in anchors.iterrows():
            branch_id += 1
            branch_rows.append(branch_from_anchor(sub, anchor, cfg=cfg, branch_id=branch_id))

    branches = pd.DataFrame(branch_rows)
    summary = summarize_branches(branches, cfg=cfg)

    spectrum_out = OUTDIR / "static_width_ecs_branch_spectrum_detail.csv"
    branches_out = OUTDIR / "static_width_ecs_branch_candidates.csv"
    summary_out = OUTDIR / "static_width_ecs_branch_summary.csv"
    spectrum.to_csv(spectrum_out, index=False)
    branches.to_csv(branches_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"Wrote {spectrum_out}")
    print(f"Wrote {branches_out}")
    print(f"Wrote {summary_out}")
    print("[static-width refined ECS branch audit summary]")
    cols = [
        "D",
        "ecs_branch_status",
        "n_branch_candidates",
        "n_seed_qualified_branches",
        "n_passing_branches",
        "best_Gamma_candidate",
        "best_E_abs_span",
        "best_Gamma_theta_rel_span_max",
        "best_Gamma_R0_rel_span_max",
        "best_Gamma_L_rel_span_max",
        "best_seed_overlap_min",
        "best_seed_Gamma_candidate",
        "best_seed_Gamma_theta_rel_span_max",
        "best_seed_Gamma_R0_rel_span_max",
        "best_seed_Gamma_L_rel_span_max",
        "best_seed_seed_overlap_min",
        "best_branch_stability_score",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()

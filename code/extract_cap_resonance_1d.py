#!/usr/bin/env python3
"""
Minimal CAP resonance prototype for the 1D action-derived operator chain.

This script plugs directly into the existing on-axis effective potential used by
`extract_omega_exact_convergence.py` and `extract_superrad_prefactor_1d.py`:

    U(z;D) = m0^2 (Omega^2 - 1) + (1 - 6 xi) Omega^{-1} LapOmega
    [-d^2/dz^2 + U] psi = E psi,   E = omega^2 - m0^2

We then promote the same finite-difference chain to a non-Hermitian complex
absorbing potential (CAP) problem on a finite box:

    H_cap = -d^2/dz^2 + U_eff(z;D) - i eta W(z),

where U_eff is either the base operator U or the channel-augmented operator
U_l = U + l(l+1)/(z^2 + eps^2).

Workflow
--------
1. Solve the Hermitian box problem to obtain a seed state (E_seed, psi_seed).
2. Add a smooth CAP profile W(z) in the outer box region.
3. Solve the complex sparse eigenproblem near E_seed.
4. Select the CAP candidate by maximal overlap with psi_seed.
5. Convert the complex energy to the frequency plane:

       omega_cap = sqrt(m0^2 + E_cap),
       Gamma_cap = -2 Im(omega_cap).

6. Report the narrow-width conversion from the energy plane:

       gamma_E = -2 Im(E_cap),
       Gamma_cap^(narrow) = gamma_E / (2 omega_R).

Notes
-----
- In the current 1D chain, the lowest states N=1,2 are true bound states
  (E<0). CAP widths there should remain numerically tiny.
- The default target is therefore N=3, the first above-threshold box state,
  which provides a minimal open-channel seed for CAP testing.
- This is an infrastructure-grade prototype. A stable physical width still
  requires eta-plateau checks and, eventually, a parent-side outgoing-channel
  derivation.
- The CLI also supports small plateau scans over both eta and z_cap_frac and
  writes summary tables for each scan direction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from extract_superrad_prefactor_1d import PhysicalParams, u_z, wkb_action_central

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "cap_resonance_1d"


@dataclass(frozen=True)
class CapParams:
    zmax: float = 80.0
    Nz: int = 2001
    z_cap_frac: float = 0.75
    cap_power: float = 2.0
    n_target: int = 3
    k_eigs: int = 6
    tol: float = 1.0e-9
    maxiter: int = 20000


def parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def tag_from_floats(values: List[float], prefix: str) -> str:
    if len(values) == 1:
        return f"{prefix}{str(values[0]).replace('.', 'p')}"
    vmin = min(values)
    vmax = max(values)
    return f"{prefix}{len(values)}_{str(vmin).replace('.', 'p')}_to_{str(vmax).replace('.', 'p')}"


def build_grid(zmax: float, Nz: int) -> np.ndarray:
    return np.linspace(-zmax, zmax, Nz)


def centrifugal_term(z: np.ndarray, ell: int, p: PhysicalParams) -> np.ndarray:
    if ell <= 0:
        return np.zeros_like(z)
    return (ell * (ell + 1)) / (z**2 + p.eps**2)


def cap_profile(z: np.ndarray, z_cap_start: float, zmax: float, cap_power: float) -> np.ndarray:
    x = np.clip((np.abs(z) - z_cap_start) / max(zmax - z_cap_start, 1.0e-12), 0.0, 1.0)
    return x**cap_power


def hermitian_seed_state(U: np.ndarray, z: np.ndarray, n_target: int, n_states: int) -> Dict[str, np.ndarray]:
    dz = z[1] - z[0]
    Ui = U[1:-1]
    main = 2.0 / dz**2 + Ui
    off = -1.0 / dz**2 * np.ones(len(Ui) - 1)

    evals, evecs = la.eigh_tridiagonal(main, off, select="i", select_range=(0, n_states - 1))
    idx = n_target - 1
    psi = np.asarray(evecs[:, idx], dtype=float)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dz)
    return {
        "evals": np.asarray(evals, dtype=float),
        "seed_E": float(evals[idx]),
        "seed_psi": psi,
    }


def build_cap_operator(U: np.ndarray, z: np.ndarray, eta: float, W: np.ndarray) -> sp.csr_matrix:
    dz = z[1] - z[0]
    Ui = U[1:-1].astype(complex)
    Wi = W[1:-1].astype(complex)

    main = 2.0 / dz**2 + Ui - 1j * eta * Wi
    off = -1.0 / dz**2 * np.ones(len(Ui) - 1, dtype=complex)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")


def choose_cap_candidate(
    evals: np.ndarray,
    evecs: np.ndarray,
    seed_E: float,
    seed_psi: np.ndarray,
    dz: float,
    z_interior: np.ndarray,
    z_cap_start: float,
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    seed_psi = seed_psi / np.sqrt(np.sum(np.abs(seed_psi) ** 2) * dz)
    interior_mask = np.abs(z_interior) <= z_cap_start

    for j, val in enumerate(evals):
        vec = np.asarray(evecs[:, j], dtype=complex)
        vec /= np.sqrt(np.sum(np.abs(vec) ** 2) * dz)

        overlap = abs(np.vdot(seed_psi, vec)) * dz
        seed_dist = abs(val - seed_E)
        interior_weight = float(np.sum(np.abs(vec[interior_mask]) ** 2) * dz)

        rows.append(
            {
                "candidate_index": float(j),
                "E_real": float(val.real),
                "E_imag": float(val.imag),
                "seed_dist": float(seed_dist),
                "overlap": float(overlap),
                "interior_weight": interior_weight,
                "abs_E_imag": float(abs(val.imag)),
            }
        )

    cand = pd.DataFrame(rows)
    non_growing = cand[cand["E_imag"] <= 1.0e-10]
    pool = non_growing if len(non_growing) > 0 else cand
    best = pool.sort_values(
        ["overlap", "interior_weight", "seed_dist", "abs_E_imag"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return {k: float(best[k]) for k in best.index}


def solve_cap_case(
    D: float,
    ell: int,
    eta: float,
    cfg: CapParams,
    p: PhysicalParams,
) -> Dict[str, float]:
    z = build_grid(cfg.zmax, cfg.Nz)
    base_U = u_z(z, D, p)
    U_eff = base_U + centrifugal_term(z, ell=ell, p=p)

    n_states = max(cfg.n_target + 4, cfg.k_eigs + 2)
    seed = hermitian_seed_state(U_eff, z, n_target=cfg.n_target, n_states=n_states)
    seed_E = seed["seed_E"]
    seed_psi = seed["seed_psi"]
    seed_omega = np.lib.scimath.sqrt(p.m0**2 + seed_E)
    if seed_omega.real < 0.0:
        seed_omega = -seed_omega

    z_cap_start = cfg.z_cap_frac * cfg.zmax
    W = cap_profile(z, z_cap_start=z_cap_start, zmax=cfg.zmax, cap_power=cfg.cap_power)
    H_cap = build_cap_operator(U_eff, z, eta=eta, W=W)

    evals, evecs = spla.eigs(
        H_cap,
        k=cfg.k_eigs,
        sigma=seed_E,
        which="LM",
        tol=cfg.tol,
        maxiter=cfg.maxiter,
    )

    z_interior = z[1:-1]
    best = choose_cap_candidate(
        evals=evals,
        evecs=evecs,
        seed_E=seed_E,
        seed_psi=seed_psi,
        dz=z[1] - z[0],
        z_interior=z_interior,
        z_cap_start=z_cap_start,
    )

    E_cap = complex(best["E_real"], best["E_imag"])
    omega_cap = np.lib.scimath.sqrt(p.m0**2 + E_cap)
    if omega_cap.real < 0.0:
        omega_cap = -omega_cap

    gamma_E = max(0.0, -2.0 * E_cap.imag)
    gamma_cap = max(0.0, -2.0 * omega_cap.imag)
    gamma_cap_narrow = gamma_E / max(2.0 * max(omega_cap.real, 0.0), 1.0e-30)

    S_seed = wkb_action_central(U_eff, z, float(seed_E))
    gamma_wkb_seed = (
        float(seed_omega.real) * np.exp(-2.0 * S_seed)
        if np.isfinite(S_seed)
        else np.nan
    )
    rel_gap = abs(gamma_cap - gamma_cap_narrow) / max(abs(gamma_cap), 1.0e-30)
    rel_vs_wkb = abs(gamma_cap - gamma_wkb_seed) / max(abs(gamma_wkb_seed), 1.0e-30) if np.isfinite(gamma_wkb_seed) else np.nan

    return {
        "D": D,
        "ell": float(ell),
        "eta_cap": eta,
        "Nz": float(cfg.Nz),
        "zmax": cfg.zmax,
        "z_cap_frac": cfg.z_cap_frac,
        "z_cap_start": z_cap_start,
        "cap_power": cfg.cap_power,
        "N_target": float(cfg.n_target),
        "seed_E": float(seed_E),
        "seed_omega_real": float(seed_omega.real),
        "seed_bound": float(seed_E < 0.0),
        "E_cap_real": float(E_cap.real),
        "E_cap_imag": float(E_cap.imag),
        "omega_cap_real": float(omega_cap.real),
        "omega_cap_imag": float(omega_cap.imag),
        "Gamma_cap": float(gamma_cap),
        "Gamma_cap_narrow": float(gamma_cap_narrow),
        "Gamma_cap_over_eta": float(gamma_cap / max(eta, 1.0e-30)),
        "rel_Gamma_exact_vs_narrow": float(rel_gap),
        "S_seed": float(S_seed) if np.isfinite(S_seed) else np.nan,
        "Gamma_wkb_seed": float(gamma_wkb_seed) if np.isfinite(gamma_wkb_seed) else np.nan,
        "rel_Gamma_cap_vs_wkb": float(rel_vs_wkb) if np.isfinite(rel_vs_wkb) else np.nan,
        "candidate_overlap": best["overlap"],
        "candidate_seed_dist": best["seed_dist"],
        "candidate_interior_weight": best["interior_weight"],
    }


def rel_span(values: pd.Series) -> float:
    vals = values.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan
    scale = max(float(np.median(np.abs(vals))), 1.0e-30)
    return float((vals.max() - vals.min()) / scale)


def summarize_plateau_over_eta(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    group_cols = ["D", "ell", "N_target", "z_cap_frac"]
    for keys, sub in df.groupby(group_cols, sort=True):
        D, ell, n_target, z_cap_frac = keys
        sub = sub.sort_values("eta_cap")
        rows.append(
            {
                "D": float(D),
                "ell": float(ell),
                "N_target": float(n_target),
                "z_cap_frac": float(z_cap_frac),
                "n_eta": float(len(sub)),
                "eta_min": float(sub["eta_cap"].min()),
                "eta_max": float(sub["eta_cap"].max()),
                "Gamma_cap_min": float(sub["Gamma_cap"].min()),
                "Gamma_cap_max": float(sub["Gamma_cap"].max()),
                "Gamma_cap_rel_span": rel_span(sub["Gamma_cap"]),
                "Gamma_cap_over_eta_rel_span": rel_span(sub["Gamma_cap_over_eta"]),
                "candidate_overlap_min": float(sub["candidate_overlap"].min()),
                "candidate_overlap_max": float(sub["candidate_overlap"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["D", "z_cap_frac"]).reset_index(drop=True)


def summarize_plateau_over_zcap(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    group_cols = ["D", "ell", "N_target", "eta_cap"]
    for keys, sub in df.groupby(group_cols, sort=True):
        D, ell, n_target, eta_cap = keys
        sub = sub.sort_values("z_cap_frac")
        rows.append(
            {
                "D": float(D),
                "ell": float(ell),
                "N_target": float(n_target),
                "eta_cap": float(eta_cap),
                "n_z_cap_frac": float(len(sub)),
                "z_cap_frac_min": float(sub["z_cap_frac"].min()),
                "z_cap_frac_max": float(sub["z_cap_frac"].max()),
                "Gamma_cap_min": float(sub["Gamma_cap"].min()),
                "Gamma_cap_max": float(sub["Gamma_cap"].max()),
                "Gamma_cap_rel_span": rel_span(sub["Gamma_cap"]),
                "Gamma_cap_over_eta_rel_span": rel_span(sub["Gamma_cap_over_eta"]),
                "candidate_overlap_min": float(sub["candidate_overlap"].min()),
                "candidate_overlap_max": float(sub["candidate_overlap"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["D", "eta_cap"]).reset_index(drop=True)


def classify_plateau_regime(gamma_rel_span: float, gamma_over_eta_rel_span: float) -> str:
    if not np.isfinite(gamma_rel_span) or not np.isfinite(gamma_over_eta_rel_span):
        return "insufficient"
    if gamma_rel_span <= 0.25 and gamma_over_eta_rel_span >= 0.50:
        return "plateau_like"
    if gamma_over_eta_rel_span <= 0.10 * max(gamma_rel_span, 1.0e-12):
        return "absorber_linear_like"
    return "mixed"


def enumerate_contiguous_windows(sub: pd.DataFrame, axis_col: str, min_points: int) -> List[pd.DataFrame]:
    sub = sub.sort_values(axis_col).reset_index(drop=True)
    windows: List[pd.DataFrame] = []
    for start in range(len(sub)):
        for stop in range(start + min_points, len(sub) + 1):
            windows.append(sub.iloc[start:stop].copy())
    return windows


def extract_best_plateau_windows(
    df: pd.DataFrame,
    axis_col: str,
    fixed_cols: List[str],
    min_points: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []

    for keys, sub in df.groupby(fixed_cols, sort=True):
        if len(sub) < min_points:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)

        candidates: List[Dict[str, float | str]] = []
        for window in enumerate_contiguous_windows(sub, axis_col=axis_col, min_points=min_points):
            gamma_rel_span = rel_span(window["Gamma_cap"])
            gamma_over_eta_rel_span = rel_span(window["Gamma_cap_over_eta"])
            overlap_min = float(window["candidate_overlap"].min())
            exact_gap_max = float(window["rel_Gamma_exact_vs_narrow"].max())

            row: Dict[str, float | str] = {
                col: key for col, key in zip(fixed_cols, keys)
            }
            row.update(
                {
                    "scan_axis": axis_col,
                    "window_min": float(window[axis_col].min()),
                    "window_max": float(window[axis_col].max()),
                    "window_center": float(np.sqrt(window[axis_col].min() * window[axis_col].max()))
                    if axis_col == "eta_cap"
                    else float(0.5 * (window[axis_col].min() + window[axis_col].max())),
                    "n_points": float(len(window)),
                    "Gamma_cap_min": float(window["Gamma_cap"].min()),
                    "Gamma_cap_max": float(window["Gamma_cap"].max()),
                    "Gamma_cap_rel_span": float(gamma_rel_span),
                    "Gamma_cap_over_eta_rel_span": float(gamma_over_eta_rel_span),
                    "candidate_overlap_min": overlap_min,
                    "rel_Gamma_exact_vs_narrow_max": exact_gap_max,
                    "regime_guess": classify_plateau_regime(gamma_rel_span, gamma_over_eta_rel_span),
                }
            )
            candidates.append(row)

        cand = pd.DataFrame(candidates).sort_values(
            [
                "Gamma_cap_rel_span",
                "Gamma_cap_over_eta_rel_span",
                "candidate_overlap_min",
                "n_points",
            ],
            ascending=[True, False, False, False],
        )
        rows.append(cand.iloc[0].to_dict())

    if len(rows) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(fixed_cols).reset_index(drop=True)


def infer_axis_trend(sub: pd.DataFrame, axis_col: str) -> str:
    ordered = sub.sort_values(axis_col)
    vals = ordered["Gamma_cap"].to_numpy(dtype=float)
    if len(vals) < 2:
        return "insufficient"
    scale = max(float(np.median(np.abs(vals))), 1.0e-30)
    drift = float(vals[-1] - vals[0]) / scale
    if abs(drift) < 0.05:
        return "flat"
    return "increasing" if drift > 0.0 else "decreasing"


def recommend_refined_windows(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    axis_col: str,
    fixed_cols: List[str],
) -> pd.DataFrame:
    if len(candidates) == 0:
        return pd.DataFrame()

    rows: List[Dict[str, float | str]] = []
    for _, cand in candidates.iterrows():
        mask = np.ones(len(df), dtype=bool)
        for col in fixed_cols:
            mask &= np.isclose(df[col].to_numpy(dtype=float), float(cand[col]))
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue

        data_min = float(sub[axis_col].min())
        data_max = float(sub[axis_col].max())
        window_min = float(cand["window_min"])
        window_max = float(cand["window_max"])
        regime = str(cand["regime_guess"])
        trend = infer_axis_trend(sub, axis_col=axis_col)

        if axis_col == "eta_cap":
            spacing = "log"
            ratio = max(window_max / max(window_min, 1.0e-30), 1.5)
            if regime == "absorber_linear_like":
                rec_min = window_min / ratio
                rec_max = window_min
                note = "push to lower eta to test departure from absorber-linear scaling"
            elif regime == "plateau_like":
                rec_min = window_min
                rec_max = window_max
                note = "densify inside the plateau-like eta window"
            else:
                rec_min = window_min / np.sqrt(ratio)
                rec_max = window_max
                note = "densify around the current best eta window with a mild lower-eta extension"
            center = float(np.sqrt(rec_min * rec_max))
        else:
            spacing = "linear"
            span = max(window_max - window_min, 0.02)
            if regime == "plateau_like":
                rec_min = window_min
                rec_max = window_max
                note = "densify inside the plateau-like z_cap_frac window"
            elif trend == "decreasing":
                rec_min = max(window_max, data_max - span)
                rec_max = min(0.95, data_max + span)
                note = "push the absorber farther out because Gamma_cap decreases with z_cap_frac"
            elif trend == "increasing":
                rec_min = max(0.05, data_min - span)
                rec_max = min(window_min, data_min + span)
                note = "pull the absorber inward because Gamma_cap increases with z_cap_frac"
            else:
                rec_min = max(0.05, window_min - 0.5 * span)
                rec_max = min(0.95, window_max + 0.5 * span)
                note = "densify around the current z_cap_frac window"
            center = float(0.5 * (rec_min + rec_max))

        row: Dict[str, float | str] = {col: cand[col] for col in fixed_cols}
        row.update(
            {
                "scan_axis": axis_col,
                "regime_guess": regime,
                "trend_guess": trend,
                "data_axis_min": data_min,
                "data_axis_max": data_max,
                "source_window_min": window_min,
                "source_window_max": window_max,
                "recommended_min": float(rec_min),
                "recommended_max": float(rec_max),
                "recommended_center": center,
                "recommended_spacing": spacing,
                "recommended_points": 5.0,
                "recommendation_note": note,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(fixed_cols).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal CAP resonance prototype for the 1D action-derived operator chain.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values.")
    ap.add_argument("--etas", default="1e-4,3e-4,1e-3", help="Comma-separated CAP strengths.")
    ap.add_argument("--ell", type=int, default=1, help="Channel index ell for U_eff = U + ell(ell+1)/(z^2+eps^2).")
    ap.add_argument("--N-target", type=int, default=3, help="Target seed level from the Hermitian box spectrum.")
    ap.add_argument("--Nz", type=int, default=2001, help="Number of z-grid points.")
    ap.add_argument("--zmax", type=float, default=80.0, help="Half-box size.")
    ap.add_argument("--z-cap-frac", type=float, default=0.75, help="Fraction of zmax where CAP starts.")
    ap.add_argument("--z-cap-fracs", default=None, help="Optional comma-separated z_cap_frac scan. Overrides --z-cap-frac.")
    ap.add_argument("--cap-power", type=float, default=2.0, help="Polynomial CAP power.")
    ap.add_argument("--k-eigs", type=int, default=6, help="Number of complex eigenvalues requested near the seed.")
    ap.add_argument("--plateau-min-points", type=int, default=2, help="Minimum contiguous scan points for a plateau-candidate window.")
    ap.add_argument("--tol", type=float, default=1.0e-9, help="ARPACK tolerance.")
    ap.add_argument("--maxiter", type=int, default=20000, help="ARPACK max iterations.")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory.")
    args = ap.parse_args()

    base_cfg = CapParams(
        zmax=args.zmax,
        Nz=args.Nz,
        z_cap_frac=args.z_cap_frac,
        cap_power=args.cap_power,
        n_target=args.N_target,
        k_eigs=args.k_eigs,
        tol=args.tol,
        maxiter=args.maxiter,
    )
    p = PhysicalParams()
    d_values = parse_floats(args.Ds)
    eta_values = parse_floats(args.etas)
    z_cap_frac_values = parse_floats(args.z_cap_fracs) if args.z_cap_fracs else [args.z_cap_frac]

    rows: List[Dict[str, float]] = []
    for D in d_values:
        for z_cap_frac in z_cap_frac_values:
            case_cfg = replace(base_cfg, z_cap_frac=z_cap_frac)
            for eta in eta_values:
                rows.append(solve_cap_case(D=D, ell=args.ell, eta=eta, cfg=case_cfg, p=p))

    df = pd.DataFrame(rows).sort_values(["D", "z_cap_frac", "eta_cap"]).reset_index(drop=True)
    eta_summary = summarize_plateau_over_eta(df)
    zcap_summary = summarize_plateau_over_zcap(df)
    eta_candidates = extract_best_plateau_windows(
        df=df,
        axis_col="eta_cap",
        fixed_cols=["D", "ell", "N_target", "z_cap_frac"],
        min_points=args.plateau_min_points,
    )
    zcap_candidates = extract_best_plateau_windows(
        df=df,
        axis_col="z_cap_frac",
        fixed_cols=["D", "ell", "N_target", "eta_cap"],
        min_points=args.plateau_min_points,
    )
    eta_recommend = recommend_refined_windows(
        df=df,
        candidates=eta_candidates,
        axis_col="eta_cap",
        fixed_cols=["D", "ell", "N_target", "z_cap_frac"],
    )
    zcap_recommend = recommend_refined_windows(
        df=df,
        candidates=zcap_candidates,
        axis_col="z_cap_frac",
        fixed_cols=["D", "ell", "N_target", "eta_cap"],
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag_D = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_values)
    tag_eta = tag_from_floats(eta_values, prefix="eta")
    tag_zcap = tag_from_floats(z_cap_frac_values, prefix="zcap")
    base_name = f"cap_resonance_scan_D{tag_D}_l{args.ell}_N{args.N_target}_{tag_eta}_{tag_zcap}"
    out_csv = outdir / f"{base_name}.csv"
    out_eta = outdir / f"{base_name}_eta_summary.csv"
    out_zcap = outdir / f"{base_name}_zcap_summary.csv"
    out_eta_candidates = outdir / f"{base_name}_eta_plateau_candidates.csv"
    out_zcap_candidates = outdir / f"{base_name}_zcap_plateau_candidates.csv"
    out_eta_recommend = outdir / f"{base_name}_eta_refine_recommend.csv"
    out_zcap_recommend = outdir / f"{base_name}_zcap_refine_recommend.csv"
    df.to_csv(out_csv, index=False)
    eta_summary.to_csv(out_eta, index=False)
    zcap_summary.to_csv(out_zcap, index=False)
    eta_candidates.to_csv(out_eta_candidates, index=False)
    zcap_candidates.to_csv(out_zcap_candidates, index=False)
    eta_recommend.to_csv(out_eta_recommend, index=False)
    zcap_recommend.to_csv(out_zcap_recommend, index=False)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_eta}")
    print(f"[saved] {out_zcap}")
    print(f"[saved] {out_eta_candidates}")
    print(f"[saved] {out_zcap_candidates}")
    print(f"[saved] {out_eta_recommend}")
    print(f"[saved] {out_zcap_recommend}")
    cols = [
        "D",
        "ell",
        "z_cap_frac",
        "eta_cap",
        "seed_E",
        "seed_omega_real",
        "E_cap_real",
        "E_cap_imag",
        "omega_cap_real",
        "omega_cap_imag",
        "Gamma_cap",
        "Gamma_cap_narrow",
        "Gamma_cap_over_eta",
        "S_seed",
        "Gamma_wkb_seed",
        "candidate_overlap",
    ]
    print(df[cols].to_string(index=False))
    if len(eta_summary) > 0:
        print("\n[eta plateau summary]")
        print(eta_summary.to_string(index=False))
    if len(zcap_summary) > 0:
        print("\n[z_cap_frac plateau summary]")
        print(zcap_summary.to_string(index=False))
    if len(eta_candidates) > 0:
        print("\n[best eta-window plateau candidates]")
        print(eta_candidates.to_string(index=False))
    if len(zcap_candidates) > 0:
        print("\n[best z_cap_frac-window plateau candidates]")
        print(zcap_candidates.to_string(index=False))
    if len(eta_recommend) > 0:
        print("\n[recommended next eta scans]")
        print(eta_recommend.to_string(index=False))
    if len(zcap_recommend) > 0:
        print("\n[recommended next z_cap_frac scans]")
        print(zcap_recommend.to_string(index=False))


if __name__ == "__main__":
    main()

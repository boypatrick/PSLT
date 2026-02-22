#!/usr/bin/env python3
"""
Extract overlap-defined y_N^eff profile in 2D axisymmetry (N=1,2,3).

Definition:
  y_N^eff(D) = | 2 pi * int rho dr dz * u_N(rho,z;D) * f_L(rho,z;D) * f_R(rho,z;D) * W_frame |

where u_N are low-lying 2D eigenmodes from the same operator chain as
extract_chi_localized_2d.py, and f_L/f_R are normalized chiral profile proxies.

This version adds two stabilizers for map-level use:
  1) mode tracking across D (Hungarian assignment on lambda/parity/log-y continuity)
  2) microcanonical window averaging around each tracked branch

We report:
  - raw overlaps y_eff_raw_n (microcanonical-windowed)
  - cumulative overlaps y_eff_cum_n = sum_{k<=n} y_eff_raw_k
  - visibility profile B_n = y_eff_cum_n / y_eff_cum_3  (n=1,2,3)

Outputs:
  - output/y_eff_2d/y_eff_2d_three_channel_D*.csv
  - output/y_eff_2d/y_eff_2d_three_channel_D*_relerr.csv
  - output/y_eff_2d/y_eff_2d_three_channel_profile.csv (from full-scan fine rows)
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

from extract_chi_localized_2d import (
    Level,
    PhysicalParams,
    build_generalized_operator,
    normalize_modes,
    omega_2center,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "y_eff_2d"


@dataclass(frozen=True)
class OverlapConfig:
    sigma_l: float = 2.5
    sigma_r: float = 2.5
    frame_power: float = 0.0
    n_track: int = 3
    n_eigs: int = 8
    window_k: int = 1
    window_gap_scale: float = 1.0
    window_sigma_mult: float = 2.0
    window_floor: float = 0.05
    track_lambda_weight: float = 1.0
    track_parity_weight: float = 0.75
    track_logy_weight: float = 0.20


def chirality_profiles(
    rr: np.ndarray,
    zz: np.ndarray,
    rho: np.ndarray,
    dr: float,
    dz: float,
    d_val: float,
    sigma_l: float,
    sigma_r: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sigma_l <= 0 or sigma_r <= 0:
        raise ValueError("sigma_l and sigma_r must be > 0.")

    f_l = np.exp(-0.5 * (rr * rr + (zz + d_val / 2.0) ** 2) / (sigma_l * sigma_l))
    f_r = np.exp(-0.5 * (rr * rr + (zz - d_val / 2.0) ** 2) / (sigma_r * sigma_r))

    w = 2.0 * math.pi * rho[:, None] * dr * dz
    n_l = math.sqrt(max(float(np.sum(w * f_l * f_l)), 1e-30))
    n_r = math.sqrt(max(float(np.sum(w * f_r * f_r)), 1e-30))
    f_l /= n_l
    f_r /= n_r
    return f_l, f_r


def solve_modes(
    d_val: float,
    level: Level,
    p: PhysicalParams,
    rho_max: float,
    z_margin: float,
    tol: float,
    maxiter: int,
    sigma: Optional[float],
    n_eigs: int,
) -> Dict[str, object]:
    z_max = d_val / 2.0 + z_margin
    t0 = time.time()
    rho, z, rr, zz, _, k_mat, m_mat = build_generalized_operator(
        D=d_val,
        p=p,
        rho_max=rho_max,
        z_max=z_max,
        dr=level.dr,
        dz=level.dz,
    )
    t_build = time.time() - t0

    t1 = time.time()
    if sigma is None:
        evals, evecs = eigsh(
            k_mat,
            k=n_eigs,
            M=m_mat,
            which="SA",
            tol=tol,
            maxiter=maxiter,
        )
    else:
        evals, evecs = eigsh(
            k_mat,
            k=n_eigs,
            M=m_mat,
            sigma=sigma,
            which="LM",
            tol=tol,
            maxiter=maxiter,
        )
    t_solve = time.time() - t1

    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])

    nr, nz = len(rho), len(z)
    modes = []
    for i in range(n_eigs):
        modes.append(evecs[:, i].reshape(nr, nz))
    psi = np.stack(modes, axis=2)
    psi = normalize_modes(psi, rho, level.dr, level.dz)
    return {
        "rho": rho,
        "z": z,
        "rr": rr,
        "zz": zz,
        "psi": psi,
        "evals": evals,
        "Nr": nr,
        "Nz": nz,
        "z_max": z_max,
        "build_s": float(t_build),
        "solve_s": float(t_solve),
    }


def mode_parity_indicators(psi: np.ndarray, rho: np.ndarray, dr: float, dz: float) -> np.ndarray:
    # z -> -z corresponds to reversing axis if grid is symmetric.
    psi_flip = psi[:, ::-1, :]
    w = 2.0 * math.pi * rho[:, None] * dr * dz
    vals = []
    for k in range(psi.shape[2]):
        vals.append(float(np.sum(w * psi[:, :, k] * psi_flip[:, :, k])))
    return np.asarray(vals, dtype=float)


def mode_overlap_values(psi: np.ndarray, kernel: np.ndarray, rho: np.ndarray, dr: float, dz: float) -> np.ndarray:
    w = 2.0 * math.pi * rho[:, None] * dr * dz
    vals = []
    for k in range(psi.shape[2]):
        vals.append(abs(float(np.sum(w * psi[:, :, k] * kernel))))
    return np.asarray(vals, dtype=float)


def assign_tracked_modes(
    evals: np.ndarray,
    y_modes: np.ndarray,
    parity: np.ndarray,
    prev_lam: Optional[np.ndarray],
    prev_y: Optional[np.ndarray],
    prev_parity: Optional[np.ndarray],
    cfg: OverlapConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    n_track = cfg.n_track
    n_eigs = len(evals)
    if n_track > n_eigs:
        raise ValueError(f"n_track={n_track} cannot exceed n_eigs={n_eigs}.")

    if prev_lam is None or prev_y is None or prev_parity is None:
        idx0 = np.arange(n_track, dtype=int)
        score0 = np.ones(n_track, dtype=float)
        return idx0, score0

    if n_eigs > 1:
        dlam = np.diff(np.sort(evals))
        lam_scale = float(max(np.median(np.abs(dlam)), 1e-4))
    else:
        lam_scale = 1.0

    eps = 1e-30
    cost_lam = np.abs(prev_lam[:, None] - evals[None, :]) / lam_scale
    cost_parity = np.abs(prev_parity[:, None] - parity[None, :])
    cost_logy = np.abs(np.log(prev_y[:, None] + eps) - np.log(y_modes[None, :] + eps))
    cost = (
        cfg.track_lambda_weight * cost_lam
        + cfg.track_parity_weight * cost_parity
        + cfg.track_logy_weight * cost_logy
    )

    row_ind, col_ind = linear_sum_assignment(cost)
    idx = np.zeros(n_track, dtype=int)
    score = np.zeros(n_track, dtype=float)
    for r, c in zip(row_ind, col_ind):
        idx[r] = int(c)
        score[r] = float(np.exp(-cost[r, c]))
    return idx, score


def local_gap(evals: np.ndarray, center_idx: int) -> float:
    n = len(evals)
    if n <= 1:
        return 1.0
    gaps = []
    if center_idx > 0:
        gaps.append(abs(float(evals[center_idx] - evals[center_idx - 1])))
    if center_idx < n - 1:
        gaps.append(abs(float(evals[center_idx + 1] - evals[center_idx])))
    if not gaps:
        return 1.0
    return float(max(min(gaps), 1e-6))


def microcanonical_average(
    evals: np.ndarray,
    y_modes: np.ndarray,
    center_idx: int,
    cfg: OverlapConfig,
) -> Tuple[float, float, int]:
    gap = local_gap(evals, center_idx)
    sigma = float(max(cfg.window_floor, cfg.window_gap_scale * gap))

    idx = np.arange(len(evals), dtype=int)
    dlam = np.abs(evals - evals[center_idx])
    mask = (np.abs(idx - center_idx) <= cfg.window_k) | (dlam <= cfg.window_sigma_mult * sigma)
    if not np.any(mask):
        mask[center_idx] = True

    weights = np.exp(-0.5 * (dlam / max(sigma, 1e-30)) ** 2)
    weights = weights * mask.astype(float)
    norm = float(np.sum(weights))
    if norm <= 1e-30:
        return float(y_modes[center_idx]), sigma, int(np.sum(mask))
    weights /= norm
    y_avg = float(np.sum(weights * y_modes))
    return y_avg, sigma, int(np.sum(mask))


def run_level_scan(
    d_vals: List[float],
    level: Level,
    p: PhysicalParams,
    cfg: OverlapConfig,
    rho_max: float,
    z_margin: float,
    tol: float,
    maxiter: int,
    sigma: Optional[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    prev_lam: Optional[np.ndarray] = None
    prev_y: Optional[np.ndarray] = None
    prev_parity: Optional[np.ndarray] = None

    for d_val in sorted(float(x) for x in d_vals):
        solved = solve_modes(
            d_val=d_val,
            level=level,
            p=p,
            rho_max=rho_max,
            z_margin=z_margin,
            tol=tol,
            maxiter=maxiter,
            sigma=sigma,
            n_eigs=cfg.n_eigs,
        )

        rho = solved["rho"]
        rr = solved["rr"]
        zz = solved["zz"]
        psi = solved["psi"]
        evals = np.asarray(solved["evals"], dtype=float)

        f_l, f_r = chirality_profiles(
            rr=rr,
            zz=zz,
            rho=rho,
            dr=level.dr,
            dz=level.dz,
            d_val=d_val,
            sigma_l=cfg.sigma_l,
            sigma_r=cfg.sigma_r,
        )

        if abs(cfg.frame_power) > 0:
            frame = np.power(omega_2center(rr, zz, d_val, p), cfg.frame_power)
        else:
            frame = np.ones_like(rr)
        kernel = f_l * f_r * frame

        y_modes = mode_overlap_values(psi=psi, kernel=kernel, rho=rho, dr=level.dr, dz=level.dz)
        parity = mode_parity_indicators(psi=psi, rho=rho, dr=level.dr, dz=level.dz)

        track_idx, track_score = assign_tracked_modes(
            evals=evals,
            y_modes=y_modes,
            parity=parity,
            prev_lam=prev_lam,
            prev_y=prev_y,
            prev_parity=prev_parity,
            cfg=cfg,
        )

        y_raw = np.zeros(cfg.n_track, dtype=float)
        y_center = np.zeros(cfg.n_track, dtype=float)
        lam_track = np.zeros(cfg.n_track, dtype=float)
        parity_track = np.zeros(cfg.n_track, dtype=float)
        sigma_track = np.zeros(cfg.n_track, dtype=float)
        nwin_track = np.zeros(cfg.n_track, dtype=int)

        for i in range(cfg.n_track):
            cidx = int(track_idx[i])
            y_center[i] = float(y_modes[cidx])
            lam_track[i] = float(evals[cidx])
            parity_track[i] = float(parity[cidx])
            y_mc, sigma_mc, n_mc = microcanonical_average(
                evals=evals,
                y_modes=y_modes,
                center_idx=cidx,
                cfg=cfg,
            )
            y_raw[i] = y_mc
            sigma_track[i] = sigma_mc
            nwin_track[i] = n_mc

        y_cum = np.cumsum(y_raw)
        y3 = max(y_cum[min(2, cfg.n_track - 1)], 1e-30)
        b_vals = y_cum / y3
        b_vals = np.maximum(b_vals, 1e-12)
        b_vals = b_vals / max(b_vals[min(2, cfg.n_track - 1)], 1e-30)

        row = {
            "D": float(d_val),
            "level": level.name,
            "dr": float(level.dr),
            "dz": float(level.dz),
            "rho_max": float(rho_max),
            "z_max": float(solved["z_max"]),
            "Nr": int(solved["Nr"]),
            "Nz": int(solved["Nz"]),
            "sigma_l": float(cfg.sigma_l),
            "sigma_r": float(cfg.sigma_r),
            "frame_power": float(cfg.frame_power),
            "n_eigs": int(cfg.n_eigs),
            "window_k": int(cfg.window_k),
            "window_gap_scale": float(cfg.window_gap_scale),
            "window_sigma_mult": float(cfg.window_sigma_mult),
            "window_floor": float(cfg.window_floor),
            "lambda_1": float(lam_track[0]),
            "lambda_2": float(lam_track[1]),
            "lambda_3": float(lam_track[2]),
            "track_idx_1": int(track_idx[0]),
            "track_idx_2": int(track_idx[1]),
            "track_idx_3": int(track_idx[2]),
            "track_score_1": float(track_score[0]),
            "track_score_2": float(track_score[1]),
            "track_score_3": float(track_score[2]),
            "parity_1": float(parity_track[0]),
            "parity_2": float(parity_track[1]),
            "parity_3": float(parity_track[2]),
            "y_eff_center_1": float(y_center[0]),
            "y_eff_center_2": float(y_center[1]),
            "y_eff_center_3": float(y_center[2]),
            "y_eff_raw_1": float(y_raw[0]),
            "y_eff_raw_2": float(y_raw[1]),
            "y_eff_raw_3": float(y_raw[2]),
            "y_eff_cum_1": float(y_cum[0]),
            "y_eff_cum_2": float(y_cum[1]),
            "y_eff_cum_3": float(y_cum[2]),
            "nwin_1": int(nwin_track[0]),
            "nwin_2": int(nwin_track[1]),
            "nwin_3": int(nwin_track[2]),
            "sigma_mc_1": float(sigma_track[0]),
            "sigma_mc_2": float(sigma_track[1]),
            "sigma_mc_3": float(sigma_track[2]),
            "B1": float(b_vals[0]),
            "B2": float(b_vals[1]),
            "B3": float(b_vals[2]),
            "build_s": float(solved["build_s"]),
            "solve_s": float(solved["solve_s"]),
        }
        rows.append(row)

        prev_lam = lam_track.copy()
        prev_y = y_raw.copy()
        prev_parity = parity_track.copy()

        print(
            f"[run] D={d_val:g}, level={level.name}, dr={level.dr}, dz={level.dz}, "
            f"track_idx={track_idx.tolist()}, B=({b_vals[0]:.4e}, {b_vals[1]:.4e}, {b_vals[2]:.4e}), "
            f"y_raw=({y_raw[0]:.3e}, {y_raw[1]:.3e}, {y_raw[2]:.3e}), solve={row['solve_s']:.2f}s"
        )

    return rows


def relerr_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for d_val in sorted(df["D"].unique()):
        sub = df[df["D"] == d_val].set_index("level")
        if "fine" not in sub.index:
            continue
        ref = sub.loc["fine"]
        for lev in ["coarse", "mid", "fine"]:
            if lev not in sub.index:
                continue
            cur = sub.loc[lev]
            rows.append(
                {
                    "D": float(d_val),
                    "level": lev,
                    "rel_B1_vs_fine": abs(cur["B1"] - ref["B1"]) / max(abs(ref["B1"]), 1e-30),
                    "rel_B2_vs_fine": abs(cur["B2"] - ref["B2"]) / max(abs(ref["B2"]), 1e-30),
                    "rel_B3_vs_fine": abs(cur["B3"] - ref["B3"]) / max(abs(ref["B3"]), 1e-30),
                    "rel_yraw1_vs_fine": abs(cur["y_eff_raw_1"] - ref["y_eff_raw_1"]) / max(abs(ref["y_eff_raw_1"]), 1e-30),
                    "rel_yraw2_vs_fine": abs(cur["y_eff_raw_2"] - ref["y_eff_raw_2"]) / max(abs(ref["y_eff_raw_2"]), 1e-30),
                    "rel_yraw3_vs_fine": abs(cur["y_eff_raw_3"] - ref["y_eff_raw_3"]) / max(abs(ref["y_eff_raw_3"]), 1e-30),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["D", "level"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract overlap-defined y_eff profile in 2D.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values")
    ap.add_argument(
        "--full-scan",
        action="store_true",
        help="Use full D grid D=4,5,...,20 and export profile CSV for scan integration.",
    )
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--sigma-l", type=float, default=2.5)
    ap.add_argument("--sigma-r", type=float, default=2.5)
    ap.add_argument("--frame-power", type=float, default=0.0, help="Frame factor power: W_frame=Omega^p")
    ap.add_argument("--n-eigs", type=int, default=8, help="Number of low eigenmodes to solve for tracking/windowing.")
    ap.add_argument("--window-k", type=int, default=1, help="Index window half-width around tracked mode.")
    ap.add_argument("--window-gap-scale", type=float, default=1.0, help="Sigma scale relative to local spectral gap.")
    ap.add_argument("--window-sigma-mult", type=float, default=2.0, help="Energy-window multiplier on sigma.")
    ap.add_argument("--window-floor", type=float, default=0.05, help="Absolute lower bound for microcanonical sigma.")
    ap.add_argument("--track-lambda-weight", type=float, default=1.0, help="Tracking cost weight for lambda continuity.")
    ap.add_argument("--track-parity-weight", type=float, default=0.75, help="Tracking cost weight for parity continuity.")
    ap.add_argument("--track-logy-weight", type=float, default=0.20, help="Tracking cost weight for log-overlap continuity.")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    ap.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="Shift-invert target; use negative value to disable shift-invert.",
    )
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    if args.n_eigs < 3:
        raise ValueError("--n-eigs must be >= 3.")
    if args.window_k < 0:
        raise ValueError("--window-k must be >= 0.")

    if args.full_scan:
        d_list = [float(d) for d in range(4, 21)]
        levels = [Level("fine", dr=0.06, dz=0.03)]
    else:
        d_list = [float(s.strip()) for s in args.Ds.split(",") if s.strip()]
        levels = [
            Level("coarse", dr=0.12, dz=0.06),
            Level("mid", dr=0.08, dz=0.04),
            Level("fine", dr=0.06, dz=0.03),
        ]

    p = PhysicalParams()
    cfg = OverlapConfig(
        sigma_l=float(args.sigma_l),
        sigma_r=float(args.sigma_r),
        frame_power=float(args.frame_power),
        n_track=3,
        n_eigs=int(args.n_eigs),
        window_k=int(args.window_k),
        window_gap_scale=float(args.window_gap_scale),
        window_sigma_mult=float(args.window_sigma_mult),
        window_floor=float(args.window_floor),
        track_lambda_weight=float(args.track_lambda_weight),
        track_parity_weight=float(args.track_parity_weight),
        track_logy_weight=float(args.track_logy_weight),
    )
    sigma = None if args.sigma < 0 else float(args.sigma)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for lev in levels:
        rows.extend(
            run_level_scan(
                d_vals=d_list,
                level=lev,
                p=p,
                cfg=cfg,
                rho_max=float(args.rho_max),
                z_margin=float(args.z_margin),
                tol=float(args.tol),
                maxiter=int(args.maxiter),
                sigma=sigma,
            )
        )

    df = pd.DataFrame(rows).sort_values(["D", "level"]).reset_index(drop=True)
    rel_df = relerr_rows(df)

    tag = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)
    out_main = outdir / f"y_eff_2d_three_channel_D{tag}.csv"
    out_rel = outdir / f"y_eff_2d_three_channel_D{tag}_relerr.csv"
    df.to_csv(out_main, index=False)
    if len(rel_df) > 0:
        rel_df.to_csv(out_rel, index=False)

    if args.full_scan:
        prof_cols = [
            "D",
            "B1",
            "B2",
            "B3",
            "y_eff_raw_1",
            "y_eff_raw_2",
            "y_eff_raw_3",
            "y_eff_cum_1",
            "y_eff_cum_2",
            "y_eff_cum_3",
            "track_idx_1",
            "track_idx_2",
            "track_idx_3",
            "track_score_1",
            "track_score_2",
            "track_score_3",
            "parity_1",
            "parity_2",
            "parity_3",
            "nwin_1",
            "nwin_2",
            "nwin_3",
            "sigma_mc_1",
            "sigma_mc_2",
            "sigma_mc_3",
            "window_k",
            "window_gap_scale",
            "window_sigma_mult",
            "window_floor",
            "n_eigs",
        ]
        prof = df[df["level"] == "fine"].sort_values("D").reset_index(drop=True)[prof_cols]
        prof_tag = outdir / f"y_eff_2d_three_channel_profile_D{tag}.csv"
        prof_canonical = outdir / "y_eff_2d_three_channel_profile.csv"
        prof.to_csv(prof_tag, index=False)
        prof.to_csv(prof_canonical, index=False)
        print(f"[saved] {prof_tag}")
        print(f"[saved] {prof_canonical}")

    print(f"[saved] {out_main}")
    if out_rel.exists():
        print(f"[saved] {out_rel}")
    if len(rel_df) > 0:
        print("\n[summary] relerr vs fine")
        print(rel_df.to_string(index=False))


if __name__ == "__main__":
    main()


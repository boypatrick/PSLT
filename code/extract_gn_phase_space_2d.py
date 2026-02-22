#!/usr/bin/env python3
"""
2D axisymmetric phase-space extraction for g_N using the localized solver chain.

This script reuses the same generalized operator builder as
`extract_chi_localized_2d.py` (same geometry, same finite-difference structure)
and computes low-lying generalized eigenvalues lambda_n on (rho, z) grids.

Microcanonical definition (explicit):
  N_ps(E) = (1 / 4pi) * integral d^2x d^2p Theta(E - U - p^2)
          = 0.5 * integral rho dr dz [E - U(rho,z)]_+

For each mode n, the phase-space multiplicity candidate uses the window
[lambda_1, lambda_n]:
  g_raw,n = 1 + integral_{lambda_1}^{lambda_n} rho_ps(E') dE'
          = 1 + N_ps(lambda_n) - N_ps(lambda_1)

We export:
  - low-N profile summary for n=1,2,3 (coarse/mid/fine)
  - relative errors vs fine grid
  - full low-mode spectral table (up to n_eigs) for reproducibility
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

# Reuse the same localized extraction chain to avoid diverging operator choices.
from extract_chi_localized_2d import PhysicalParams, Level, build_generalized_operator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "gn_fp_2d"


@dataclass(frozen=True)
class SolveConfig:
    tol: float
    maxiter: int
    sigma: float | None
    n_eigs: int


def solve_low_modes(K, M, cfg: SolveConfig) -> np.ndarray:
    n_dim = K.shape[0]
    k_req = max(6, cfg.n_eigs)
    k_eff = min(k_req, max(2, n_dim - 2))
    if k_eff < 3:
        raise RuntimeError(f"Operator dimension too small for 3 low modes (dim={n_dim}).")

    if cfg.sigma is None:
        vals, _ = eigsh(
            K,
            k=k_eff,
            M=M,
            which="SA",
            tol=cfg.tol,
            maxiter=cfg.maxiter,
        )
    else:
        vals, _ = eigsh(
            K,
            k=k_eff,
            M=M,
            sigma=cfg.sigma,
            which="LM",
            tol=cfg.tol,
            maxiter=cfg.maxiter,
        )
    vals = np.sort(np.real(vals))
    return vals


def n_phase_space(E: float, U: np.ndarray, rho: np.ndarray, dr: float, dz: float) -> float:
    # N_ps(E) = 0.5 * integral rho dr dz [E - U]_+
    return float(0.5 * np.sum(rho[:, None] * np.maximum(E - U, 0.0)) * dr * dz)


def parse_d_values(args) -> List[float]:
    if args.full_scan:
        return [float(d) for d in range(4, 21)]
    return [float(s.strip()) for s in args.Ds.split(",") if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="2D phase-space extraction for g_N (localized-chain aligned).")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values.")
    ap.add_argument(
        "--full-scan",
        action="store_true",
        help="Use full D grid: D=4,5,...,20 (overrides --Ds).",
    )
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0, help="Use z_max = D/2 + z_margin.")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    ap.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="Shift-invert target (same convention as localized chi extraction). Use negative to disable.",
    )
    ap.add_argument(
        "--n-eigs",
        type=int,
        default=40,
        help="Number of low generalized eigenvalues to retain in spectral export.",
    )
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    d_list = parse_d_values(args)
    levels = [
        Level("coarse", dr=0.12, dz=0.06),
        Level("mid", dr=0.08, dz=0.04),
        Level("fine", dr=0.06, dz=0.03),
    ]
    p = PhysicalParams()
    sigma = None if args.sigma < 0 else float(args.sigma)
    cfg = SolveConfig(tol=float(args.tol), maxiter=int(args.maxiter), sigma=sigma, n_eigs=int(args.n_eigs))

    profile_rows = []
    spec_rows = []

    for D in d_list:
        for lev in levels:
            z_max = D / 2.0 + args.z_margin
            rho, z, _rr, _zz, uu, K, M = build_generalized_operator(
                D=D,
                p=p,
                rho_max=args.rho_max,
                z_max=z_max,
                dr=lev.dr,
                dz=lev.dz,
            )
            vals = solve_low_modes(K, M, cfg)

            nps_vals = np.asarray([n_phase_space(float(E), uu, rho, lev.dr, lev.dz) for E in vals], dtype=float)
            nps0 = float(nps_vals[0])
            g_raw_all = np.asarray([1.0 + max(float(nv - nps0), 0.0) for nv in nps_vals], dtype=float)
            if len(g_raw_all) < 3:
                raise RuntimeError(f"D={D}, level={lev.name}: need >=3 modes, got {len(g_raw_all)}.")

            g1, g2, g3 = (float(g_raw_all[i]) for i in range(3))
            g3_safe = max(g3, 1e-30)
            ghat = [g1 / g3_safe, g2 / g3_safe, g3 / g3_safe]
            lam = [float(vals[i]) for i in range(3)]

            profile_rows.append(
                {
                    "D": float(D),
                    "level": lev.name,
                    "dr": float(lev.dr),
                    "dz": float(lev.dz),
                    "Nr": len(rho),
                    "Nz": len(z),
                    "lambda1": lam[0],
                    "lambda2": lam[1],
                    "lambda3": lam[2],
                    "window_E_lo": lam[0],
                    "window_E_hi_n2": lam[1],
                    "window_E_hi_n3": lam[2],
                    "Nps_lambda1": float(nps_vals[0]),
                    "Nps_lambda2": float(nps_vals[1]),
                    "Nps_lambda3": float(nps_vals[2]),
                    "g1_raw": g1,
                    "g2_raw": g2,
                    "g3_raw": g3,
                    "g1_hat": float(ghat[0]),
                    "g2_hat": float(ghat[1]),
                    "g3_hat": float(ghat[2]),
                    "n_eigs_exported": len(vals),
                    "solver_sigma": np.nan if sigma is None else float(sigma),
                }
            )

            for i, ev in enumerate(vals, start=1):
                g_raw = float(g_raw_all[i - 1])
                spec_rows.append(
                    {
                        "D": float(D),
                        "level": lev.name,
                        "dr": float(lev.dr),
                        "dz": float(lev.dz),
                        "mode_n": i,
                        "lambda_n": float(ev),
                        "window_E_lo": float(vals[0]),
                        "window_E_hi": float(ev),
                        "Nps_lambda_n": float(nps_vals[i - 1]),
                        "g_raw_n": g_raw,
                        "g_hat_to_g3": float(g_raw / g3_safe),
                    }
                )

            print(
                f"[run] D={D:g}, level={lev.name}, "
                f"lambda1..3=({lam[0]:.6f}, {lam[1]:.6f}, {lam[2]:.6f}), "
                f"g_hat(1,2,3)=({ghat[0]:.6f}, {ghat[1]:.6f}, {ghat[2]:.6f})"
            )

    df = pd.DataFrame(profile_rows).sort_values(["D", "level"]).reset_index(drop=True)
    spec_df = pd.DataFrame(spec_rows).sort_values(["D", "level", "mode_n"]).reset_index(drop=True)

    rel_rows = []
    for D in sorted(df["D"].unique()):
        sub = df[df["D"] == D].set_index("level")
        if "fine" not in sub.index:
            continue
        ref = sub.loc["fine"]
        for lev in ["coarse", "mid", "fine"]:
            if lev not in sub.index:
                continue
            cur = sub.loc[lev]
            rel_rows.append(
                {
                    "D": float(D),
                    "level": lev,
                    "rel_g1_hat_vs_fine": abs(cur["g1_hat"] - ref["g1_hat"]) / max(abs(ref["g1_hat"]), 1e-30),
                    "rel_g2_hat_vs_fine": abs(cur["g2_hat"] - ref["g2_hat"]) / max(abs(ref["g2_hat"]), 1e-30),
                    "rel_g3_hat_vs_fine": abs(cur["g3_hat"] - ref["g3_hat"]) / max(abs(ref["g3_hat"]), 1e-30),
                    "max_rel_ghat_vs_fine": max(
                        abs(cur["g1_hat"] - ref["g1_hat"]) / max(abs(ref["g1_hat"]), 1e-30),
                        abs(cur["g2_hat"] - ref["g2_hat"]) / max(abs(ref["g2_hat"]), 1e-30),
                        abs(cur["g3_hat"] - ref["g3_hat"]) / max(abs(ref["g3_hat"]), 1e-30),
                    ),
                }
            )
    rel_df = pd.DataFrame(rel_rows).sort_values(["D", "level"]).reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)
    out_main = outdir / f"gn_phase_space_2d_D{tag}.csv"
    out_rel = outdir / f"gn_phase_space_2d_D{tag}_relerr.csv"
    out_spec = outdir / f"gn_phase_space_2d_spectrum_D{tag}.csv"
    df.to_csv(out_main, index=False)
    rel_df.to_csv(out_rel, index=False)
    spec_df.to_csv(out_spec, index=False)

    print("\n[summary] rel errors vs fine")
    if len(rel_df) > 0:
        print(rel_df.to_string(index=False))
    print(f"\n[done] wrote {out_main}")
    print(f"[done] wrote {out_rel}")
    print(f"[done] wrote {out_spec}")


if __name__ == "__main__":
    main()

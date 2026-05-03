#!/usr/bin/env python3
"""
V1 protected Spin^c moment/polynomial anchor-overlap diagnostic.

This optional diagnostic does not modify the PSLT baseline.  It tests whether
the protected O(2) polynomial flag can be embedded into the same finite-volume
solver space used by the G1 low-N spectral-rank certificate.

Canonical anchor used here:
  e_k(x) = taper(rho,z;D) * x^k,  k=0,1,2,
  x = z / z_max,
  taper = cos(pi*rho/(2*rho_max)) * cos(pi*z/(2*z_max)).

The taper is fixed before seeing the overlap result.  It is only a
Dirichlet-compatible finite-box lift of the degree flag, not a new scan family
or a tuned production object.

Outputs:
  - output/spinc_bridge/spinc_anchor_overlap_detail.csv
  - output/spinc_bridge/spinc_anchor_overlap_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
DEFAULT_OUTDIR = ROOT / "output" / "spinc_bridge"
DEFAULT_GN_DETAIL = ROOT / "output" / "gn_fp_2d" / "gn_lowN_microstate_count_detail.csv"


@dataclass(frozen=True)
class AnchorConfig:
    rho_max: float = 3.0
    z_margin: float = 6.0
    dr: float = 0.06
    dz: float = 0.03
    sigma: float = 2.5
    tol: float = 1e-8
    maxiter: int = 30000
    n_modes: int = 3
    sigma_min_tol: float = 1e-6
    strong_tail_tol: float = 0.50


def deterministic_v0(n_dim: int) -> np.ndarray:
    v0 = np.linspace(1.0, 2.0, int(n_dim), dtype=float)
    v0 /= max(float(np.linalg.norm(v0)), 1e-30)
    return v0


def parse_d_values(args: argparse.Namespace) -> list[float]:
    if args.Ds:
        return [float(s.strip()) for s in args.Ds.split(",") if s.strip()]
    if args.gn_detail.exists():
        df = pd.read_csv(args.gn_detail)
        return [float(x) for x in sorted(df["D"].astype(float).unique())]
    return [4.0 + 0.8 * i for i in range(21)]


def weight_matrix(rho: np.ndarray, dr: float, dz: float, shape: tuple[int, int]) -> np.ndarray:
    return 2.0 * math.pi * rho[:, None] * dr * dz * np.ones(shape, dtype=float)


def weighted_inner(f: np.ndarray, g: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * f * g))


def weighted_qr_ordered(basis: list[np.ndarray], w: np.ndarray, tol: float = 1e-14) -> list[np.ndarray]:
    q_basis: list[np.ndarray] = []
    for b in basis:
        q = np.array(b, dtype=float, copy=True)
        for prev in q_basis:
            q -= weighted_inner(prev, q, w) * prev
        nrm2 = weighted_inner(q, q, w)
        if nrm2 <= tol:
            raise RuntimeError(f"Weighted Gram-Schmidt failed; norm^2={nrm2:.3e}.")
        q_basis.append(q / math.sqrt(nrm2))
    return q_basis


def polynomial_anchor_basis(
    rho: np.ndarray,
    z: np.ndarray,
    rr: np.ndarray,
    zz: np.ndarray,
    rho_max: float,
    z_max: float,
) -> list[np.ndarray]:
    x = zz / float(z_max)
    radial_taper = np.cos(0.5 * math.pi * rr / float(rho_max))
    axial_taper = np.cos(0.5 * math.pi * zz / float(z_max))
    taper = radial_taper * axial_taper
    # Tiny roundoff near the finite-box boundary should not flip the taper sign.
    taper = np.where(taper < 0.0, 0.0, taper)
    return [taper, taper * x, taper * x * x]


def solve_modes(D: float, cfg: AnchorConfig):
    p = PhysicalParams()
    level = Level("fine", dr=cfg.dr, dz=cfg.dz)
    z_max = D / 2.0 + cfg.z_margin
    rho, z, rr, zz, _uu, k_mat, m_mat = build_generalized_operator(
        D=D,
        p=p,
        rho_max=cfg.rho_max,
        z_max=z_max,
        dr=level.dr,
        dz=level.dz,
    )
    v0 = deterministic_v0(k_mat.shape[0])
    evals, evecs = eigsh(
        k_mat,
        k=max(3, int(cfg.n_modes)),
        M=m_mat,
        sigma=float(cfg.sigma),
        which="LM",
        tol=float(cfg.tol),
        maxiter=int(cfg.maxiter),
        v0=v0,
    )
    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    nr, nz = len(rho), len(z)
    psi = np.stack([evecs[:, i].reshape(nr, nz) for i in range(3)], axis=2)
    psi = normalize_modes(psi, rho, level.dr, level.dz)
    return rho, z, rr, zz, z_max, evals[:3], psi


def run_case(D: float, cfg: AnchorConfig) -> dict[str, Any]:
    rho, _z, rr, zz, z_max, evals, psi = solve_modes(D, cfg)
    w = weight_matrix(rho, cfg.dr, cfg.dz, rr.shape)
    raw_basis = polynomial_anchor_basis(rho, _z, rr, zz, cfg.rho_max, z_max)
    q_basis = weighted_qr_ordered(raw_basis, w)

    A = np.zeros((3, 3), dtype=float)
    for n in range(3):
        for k in range(3):
            A[n, k] = weighted_inner(psi[:, :, n], q_basis[k], w)

    svals = np.linalg.svd(A, compute_uv=False)
    col_proj = np.sum(A * A, axis=0)
    tail = np.sqrt(np.maximum(1.0 - col_proj, 0.0))
    det_abs = abs(float(np.linalg.det(A)))
    cond = float(svals[0] / max(svals[-1], 1e-300))
    rank = int(np.linalg.matrix_rank(A, tol=cfg.sigma_min_tol))
    invertible_pass = bool(svals[-1] > cfg.sigma_min_tol and rank == 3)
    strong_tail_pass = bool(float(np.max(tail)) <= cfg.strong_tail_tol)

    return {
        "D": float(D),
        "lambda1": float(evals[0]),
        "lambda2": float(evals[1]),
        "lambda3": float(evals[2]),
        "gap12": float(evals[1] - evals[0]),
        "gap23": float(evals[2] - evals[1]),
        "A00": float(A[0, 0]),
        "A01": float(A[0, 1]),
        "A02": float(A[0, 2]),
        "A10": float(A[1, 0]),
        "A11": float(A[1, 1]),
        "A12": float(A[1, 2]),
        "A20": float(A[2, 0]),
        "A21": float(A[2, 1]),
        "A22": float(A[2, 2]),
        "sigma1": float(svals[0]),
        "sigma2": float(svals[1]),
        "sigma3": float(svals[2]),
        "sigma_min": float(svals[-1]),
        "det_abs": det_abs,
        "cond": cond,
        "rank": rank,
        "tail_basis0": float(tail[0]),
        "tail_basis1": float(tail[1]),
        "tail_basis2": float(tail[2]),
        "tail_max": float(np.max(tail)),
        "proj_min": float(np.min(col_proj)),
        "proj_max": float(np.max(col_proj)),
        "invertible_pass": invertible_pass,
        "strong_tail_pass": strong_tail_pass,
        "row_pass": bool(invertible_pass and strong_tail_pass),
    }


def write_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="V1 Spin^c polynomial anchor-overlap diagnostic.")
    ap.add_argument("--Ds", default="", help="Comma-separated D values. Default: audited G1 D grid.")
    ap.add_argument("--gn-detail", type=Path, default=DEFAULT_GN_DETAIL)
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    ap.add_argument("--sigma-min-tol", type=float, default=1e-6)
    ap.add_argument(
        "--strict-invertible",
        action="store_true",
        help="Exit nonzero if the anchor-overlap matrix is not invertible on every audited D.",
    )
    ap.add_argument(
        "--strong-tail-tol",
        type=float,
        default=0.50,
        help="Tail threshold for strong promotion. Invertibility is reported separately.",
    )
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    cfg = AnchorConfig(
        rho_max=float(args.rho_max),
        z_margin=float(args.z_margin),
        dr=float(args.dr),
        dz=float(args.dz),
        sigma=float(args.sigma),
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        sigma_min_tol=float(args.sigma_min_tol),
        strong_tail_tol=float(args.strong_tail_tol),
    )
    d_values = parse_d_values(args)

    rows: list[dict[str, Any]] = []
    for D in d_values:
        print(f"[run] D={D:g}")
        row = run_case(float(D), cfg)
        rows.append(row)
        print(
            "      "
            f"sigma_min={row['sigma_min']:.6e}, "
            f"tail_max={row['tail_max']:.6e}, "
            f"rank={row['rank']}, "
            f"invertible={row['invertible_pass']}, "
            f"strong_tail={row['strong_tail_pass']}"
        )

    detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    all_invertible = bool(detail["invertible_pass"].all())
    all_strong_tail = bool(detail["strong_tail_pass"].all())
    all_row_pass = bool(detail["row_pass"].all())
    status = (
        "PROMOTABLE_ANCHOR_INTERTWINER"
        if all_row_pass
        else (
            "INVERTIBLE_BUT_LARGE_TAIL_DIAGNOSTIC"
            if all_invertible
            else "ANCHOR_OVERLAP_NOT_INVERTIBLE"
        )
    )
    summary = {
        "anchor": "tapered_polynomial_degree_flag",
        "n_D": int(len(detail)),
        "D_min": float(detail["D"].min()),
        "D_max": float(detail["D"].max()),
        "sigma_min_floor": float(detail["sigma_min"].min()),
        "sigma_min_tol": float(cfg.sigma_min_tol),
        "det_abs_min": float(detail["det_abs"].min()),
        "cond_max": float(detail["cond"].max()),
        "tail_max": float(detail["tail_max"].max()),
        "strong_tail_tol": float(cfg.strong_tail_tol),
        "proj_min": float(detail["proj_min"].min()),
        "min_gap12": float(detail["gap12"].min()),
        "min_gap23": float(detail["gap23"].min()),
        "all_invertible_pass": all_invertible,
        "all_strong_tail_pass": all_strong_tail,
        "all_row_pass": all_row_pass,
        "status": status,
        "baseline_changed": False,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    detail_path = args.outdir / "spinc_anchor_overlap_detail.csv"
    summary_path = args.outdir / "spinc_anchor_overlap_summary.csv"
    detail.to_csv(detail_path, index=False)
    write_summary(summary_path, summary)

    print(f"\n[done] wrote {detail_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[status] {status}")

    if not all_invertible and args.strict_invertible:
        raise SystemExit("Anchor-overlap invertibility gate failed; see detail CSV.")


if __name__ == "__main__":
    main()

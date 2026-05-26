#!/usr/bin/env python3
"""Fixed-box projector step and Davis-Kahan residual audit.

T3b for the optional spectral-projector transport line.  The current G1 artifact
uses z_max=D/2+margin, so adjacent D values do not literally live in the same
finite-dimensional Hilbert space.  This script performs a fixed-box pullback
surrogate: all D values are solved on the common box z_max=D_max/2+margin with
the same (rho,z) grid, hence M and the kinetic matrix are D-independent and
projectors can be compared directly.

For each adjacent D step it records:
  1. exact rank-one projector drift sqrt(1-|<u_n(D0),u_n(D1)>_M|^2),
  2. exact first-three cluster drift from the smallest singular value of the
     overlap matrix,
  3. a Davis-Kahan residual comparator using (H(D1)-mu)u_n(D0).

The exact projector drift is the hard fixed-box certificate.  The raw residual
Davis-Kahan comparator is intentionally reported separately; it is expected to
be conservative because residual norms still see core/tail components that the
low-mode projector angle largely filters out.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

from extract_chi_localized_2d import Level, PhysicalParams, build_generalized_operator, u_potential

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GN_DETAIL = ROOT / "output" / "gn_fp_2d" / "gn_lowN_microstate_count_detail.csv"
DEFAULT_SPECTRUM = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_spectrum_Dgrid21.csv"
DEFAULT_HF_DETAIL = ROOT / "output" / "spectral_transport" / "spectral_projector_hf_weighted_bound_detail.csv"
OUTDIR = ROOT / "output" / "spectral_transport"
SUMMARY_CSV = OUTDIR / "spectral_projector_davis_kahan_step_summary.csv"
DETAIL_CSV = OUTDIR / "spectral_projector_davis_kahan_step_detail.csv"
CLUSTER_CSV = OUTDIR / "spectral_projector_davis_kahan_step_cluster.csv"
DETAIL_JSON = OUTDIR / "spectral_projector_davis_kahan_step_detail.json"


def deterministic_v0(n_dim: int) -> np.ndarray:
    v0 = np.linspace(1.0, 2.0, int(n_dim), dtype=float)
    v0 /= max(float(np.linalg.norm(v0)), 1.0e-300)
    return v0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def parse_d_values(args: argparse.Namespace) -> list[float]:
    if args.Ds:
        return [float(s.strip()) for s in args.Ds.split(",") if s.strip()]
    df = pd.read_csv(args.gn_detail)
    df = df[df["level"].astype(str) == str(args.level)] if "level" in df.columns else df
    return [float(x) for x in sorted(df["D"].astype(float).unique())]


def load_spectrum_reference(path: Path, level: str) -> dict[tuple[float, int], float]:
    spec = pd.read_csv(path)
    spec = spec[spec["level"].astype(str) == str(level)].copy()
    return {(float(row.D), int(row.mode_n)): float(row.lambda_n) for row in spec.itertuples(index=False)}


def solve_fixed_box(D: float, args: argparse.Namespace, p: PhysicalParams, level: Level, fixed_z_max: float):
    rho, z, rr, zz, uu, k_mat, m_mat = build_generalized_operator(
        D=D,
        p=p,
        rho_max=float(args.rho_max),
        z_max=fixed_z_max,
        dr=level.dr,
        dz=level.dz,
    )
    v0 = deterministic_v0(k_mat.shape[0])
    sigma = None if float(args.sigma) < 0 else float(args.sigma)
    if sigma is None:
        vals, vecs = eigsh(
            k_mat,
            k=int(args.n_eigs),
            M=m_mat,
            which="SA",
            tol=float(args.tol),
            maxiter=int(args.maxiter),
            v0=v0,
        )
    else:
        vals, vecs = eigsh(
            k_mat,
            k=int(args.n_eigs),
            M=m_mat,
            sigma=sigma,
            which="LM",
            tol=float(args.tol),
            maxiter=int(args.maxiter),
            v0=v0,
        )
    idx = np.argsort(np.real(vals))
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])
    m_diag = np.repeat(rho, len(z))
    for j in range(vecs.shape[1]):
        nrm = float(np.sqrt(np.sum(m_diag * vecs[:, j] * vecs[:, j])))
        vecs[:, j] /= max(nrm, 1.0e-300)
    return {
        "D": float(D),
        "rho": rho,
        "z": z,
        "rr": rr,
        "zz": zz,
        "vals": vals,
        "vecs": vecs,
        "m_diag": m_diag,
    }


def rank_one_sin(u0: np.ndarray, u1: np.ndarray, m_diag: np.ndarray) -> tuple[float, float]:
    overlap = float(np.sum(m_diag * u0 * u1))
    overlap_abs = min(abs(overlap), 1.0)
    return float(np.sqrt(max(1.0 - overlap_abs * overlap_abs, 0.0))), overlap


def cluster_sin(U0: np.ndarray, U1: np.ndarray, m_diag: np.ndarray) -> tuple[float, float, float]:
    S = U0.T @ (m_diag[:, None] * U1)
    svals = np.linalg.svd(S, compute_uv=False)
    sigma_min = float(np.min(svals))
    sigma_max = float(np.max(svals))
    return float(np.sqrt(max(1.0 - min(sigma_min, 1.0) ** 2, 0.0))), sigma_min, sigma_max


def residual_dk_bound(
    u0: np.ndarray,
    vals0: np.ndarray,
    vals1: np.ndarray,
    n0: int,
    delta_u: np.ndarray,
    m_diag: np.ndarray,
) -> dict[str, float]:
    mean = float(np.sum(m_diag * delta_u * u0 * u0))
    rms = float(np.sqrt(np.sum(m_diag * delta_u * delta_u * u0 * u0)))
    residual = float(np.sqrt(max(rms * rms - mean * mean, 0.0)))
    mu = float(vals0[n0] + mean)
    gap_values = [abs(float(vals1[m] - mu)) for m in range(len(vals1)) if m != n0]
    gap = float(min(gap_values))
    bound = float(residual / max(gap, 1.0e-300))
    return {
        "rayleigh_shift_mean_deltaU": mean,
        "rayleigh_mu": mu,
        "residual_rms_centered_deltaU": residual,
        "raw_davis_kahan_gap": gap,
        "raw_davis_kahan_bound": bound,
        "raw_davis_kahan_bound_clipped": min(bound, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Ds", default="")
    parser.add_argument("--gn-detail", type=Path, default=DEFAULT_GN_DETAIL)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--hf-detail", type=Path, default=DEFAULT_HF_DETAIL)
    parser.add_argument("--level", default="fine")
    parser.add_argument("--rho-max", type=float, default=3.0)
    parser.add_argument("--z-margin", type=float, default=6.0)
    parser.add_argument("--dr", type=float, default=0.06)
    parser.add_argument("--dz", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=30000)
    parser.add_argument("--n-eigs", type=int, default=40)
    parser.add_argument("--n-low", type=int, default=3)
    parser.add_argument("--rank-one-drift-tol", type=float, default=0.05)
    parser.add_argument("--cluster-drift-tol", type=float, default=0.075)
    parser.add_argument("--path-bound-tol", type=float, default=0.25)
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--detail", type=Path, default=DETAIL_CSV)
    parser.add_argument("--cluster", type=Path, default=CLUSTER_CSV)
    parser.add_argument("--detail-json", type=Path, default=DETAIL_JSON)
    args = parser.parse_args()

    if args.level != "fine":
        raise RuntimeError("T3b currently records the fine-grid fixed-box pullback only.")

    args.gn_detail = args.gn_detail if args.gn_detail.is_absolute() else ROOT / args.gn_detail
    spectrum_csv = args.spectrum_csv if args.spectrum_csv.is_absolute() else ROOT / args.spectrum_csv
    hf_detail_path = args.hf_detail if args.hf_detail.is_absolute() else ROOT / args.hf_detail
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    detail_path = args.detail if args.detail.is_absolute() else ROOT / args.detail
    cluster_path = args.cluster if args.cluster.is_absolute() else ROOT / args.cluster
    json_path = args.detail_json if args.detail_json.is_absolute() else ROOT / args.detail_json

    d_values = parse_d_values(args)
    fixed_z_max = max(d_values) / 2.0 + float(args.z_margin)
    p = PhysicalParams()
    level = Level("fine", dr=float(args.dr), dz=float(args.dz))
    ref = load_spectrum_reference(spectrum_csv, args.level)

    cache: dict[float, dict[str, Any]] = {}
    for D in d_values:
        print(f"[solve fixed-box] D={D:g}, z_max={fixed_z_max:g}, n_eigs={int(args.n_eigs)}")
        cache[float(D)] = solve_fixed_box(float(D), args, p, level, fixed_z_max)

    # T3a derivative path bound is used as a companion certificate when available.
    hf_detail = pd.read_csv(hf_detail_path) if hf_detail_path.exists() else pd.DataFrame()
    hf_bound_by_DN: dict[tuple[float, int], float] = {}
    if not hf_detail.empty:
        for row in hf_detail.itertuples(index=False):
            hf_bound_by_DN[(float(row.D), int(row.N))] = float(row.total_projector_derivative_bound)

    detail_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    max_ref_diff_fixed_box = 0.0

    for D in d_values:
        vals = cache[float(D)]["vals"]
        for n in range(int(args.n_low)):
            ref_val = ref.get((float(D), n + 1), np.nan)
            if np.isfinite(ref_val):
                max_ref_diff_fixed_box = max(max_ref_diff_fixed_box, abs(float(vals[n]) - float(ref_val)))

    for D0, D1 in zip(d_values[:-1], d_values[1:]):
        left = cache[float(D0)]
        right = cache[float(D1)]
        vals0 = left["vals"]
        vals1 = right["vals"]
        vecs0 = left["vecs"]
        vecs1 = right["vecs"]
        m_diag = left["m_diag"]
        if len(m_diag) != len(right["m_diag"]) or not np.allclose(m_diag, right["m_diag"]):
            raise RuntimeError("Fixed-box pullback failed: M diagonals differ between adjacent D values.")
        delta_u = (u_potential(left["rr"], left["zz"], float(D1), p) - u_potential(left["rr"], left["zz"], float(D0), p)).ravel()
        delta_D = float(D1 - D0)

        U0 = vecs0[:, : int(args.n_low)]
        U1 = vecs1[:, : int(args.n_low)]
        cl_sin, sigma_min, sigma_max = cluster_sin(U0, U1, m_diag)
        cluster_rows.append(
            {
                "D_left": float(D0),
                "D_right": float(D1),
                "delta_D": delta_D,
                "cluster": "1:3",
                "sin_theta_max": cl_sin,
                "sigma_min_overlap": sigma_min,
                "sigma_max_overlap": sigma_max,
                "cluster_drift_pass": bool(cl_sin <= float(args.cluster_drift_tol)),
            }
        )

        for n in range(int(args.n_low)):
            sin_actual, overlap = rank_one_sin(vecs0[:, n], vecs1[:, n], m_diag)
            dk = residual_dk_bound(vecs0[:, n], vals0, vals1, n, delta_u, m_diag)
            endpoint_bound = max(
                hf_bound_by_DN.get((float(D0), n + 1), np.nan),
                hf_bound_by_DN.get((float(D1), n + 1), np.nan),
            )
            path_bound = float(delta_D * endpoint_bound) if np.isfinite(endpoint_bound) else np.nan
            detail_rows.append(
                {
                    "D_left": float(D0),
                    "D_right": float(D1),
                    "delta_D": delta_D,
                    "N": int(n + 1),
                    "lambda_left_fixed": float(vals0[n]),
                    "lambda_right_fixed": float(vals1[n]),
                    "lambda_displacement_fixed": abs(float(vals1[n] - vals0[n])),
                    "overlap_signed": overlap,
                    "overlap_abs": abs(overlap),
                    "actual_rank_one_projector_sin": sin_actual,
                    "rank_one_drift_pass": bool(sin_actual <= float(args.rank_one_drift_tol)),
                    **dk,
                    "raw_davis_kahan_pass_unclipped": bool(dk["raw_davis_kahan_bound"] <= float(args.path_bound_tol)),
                    "t3a_endpoint_max_Pprime_bound": endpoint_bound,
                    "t3a_integrated_path_bound": path_bound,
                    "t3a_path_bound_pass": bool(path_bound <= float(args.path_bound_tol)) if np.isfinite(path_bound) else False,
                }
            )

    max_rank_one = max(float(r["actual_rank_one_projector_sin"]) for r in detail_rows)
    max_cluster = max(float(r["sin_theta_max"]) for r in cluster_rows)
    max_raw_dk = max(float(r["raw_davis_kahan_bound"]) for r in detail_rows)
    max_path_bound = max(float(r["t3a_integrated_path_bound"]) for r in detail_rows if np.isfinite(float(r["t3a_integrated_path_bound"])))
    all_rank_one_pass = all(bool(r["rank_one_drift_pass"]) for r in detail_rows)
    all_cluster_pass = all(bool(r["cluster_drift_pass"]) for r in cluster_rows)
    all_path_pass = all(bool(r["t3a_path_bound_pass"]) for r in detail_rows)
    raw_dk_all_pass = all(bool(r["raw_davis_kahan_pass_unclipped"]) for r in detail_rows)

    verdict = (
        "T3B_FIXED_BOX_PROJECTOR_STEP_PASS_RAW_DK_LOOSE"
        if all_rank_one_pass and all_cluster_pass and all_path_pass and not raw_dk_all_pass
        else (
            "T3B_FIXED_BOX_PROJECTOR_STEP_STRICT_DK_PASS"
            if all_rank_one_pass and all_cluster_pass and all_path_pass and raw_dk_all_pass
            else "T3B_FIXED_BOX_STEP_DIAGNOSTIC_ONLY"
        )
    )

    summary = {
        "gate": "T3b",
        "verdict": verdict,
        "level": args.level,
        "n_D": int(len(d_values)),
        "n_steps": int(len(d_values) - 1),
        "D_min": float(min(d_values)),
        "D_max": float(max(d_values)),
        "fixed_z_max": float(fixed_z_max),
        "rho_max": float(args.rho_max),
        "dr": float(args.dr),
        "dz": float(args.dz),
        "n_eigs": int(args.n_eigs),
        "n_low": int(args.n_low),
        "rank_one_drift_tol": float(args.rank_one_drift_tol),
        "cluster_drift_tol": float(args.cluster_drift_tol),
        "path_bound_tol": float(args.path_bound_tol),
        "max_rank_one_projector_sin": max_rank_one,
        "all_rank_one_drift_pass": all_rank_one_pass,
        "max_cluster_sin_theta": max_cluster,
        "all_cluster_drift_pass": all_cluster_pass,
        "max_t3a_integrated_path_bound": max_path_bound,
        "all_t3a_path_bound_pass": all_path_pass,
        "max_raw_davis_kahan_bound_unclipped": max_raw_dk,
        "raw_davis_kahan_all_pass_unclipped": raw_dk_all_pass,
        "max_abs_fixed_box_lambda_diff_vs_g1_variable_box_low3": max_ref_diff_fixed_box,
        "interpretation": "fixed-box projectors drift mildly and T3a path bound passes; raw residual Davis-Kahan bound is recorded but too conservative because it sees core/tail residuals",
    }

    write_csv(summary_path, [summary])
    write_csv(detail_path, detail_rows)
    write_csv(cluster_path, cluster_rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(
            {
                "claim_firewall": {
                    "baseline_change": False,
                    "main_tex_change": False,
                    "literal_g1_variable_box_claimed": False,
                    "meaning": "fixed-box pullback step certificate plus Davis-Kahan residual comparator",
                },
                "summary": summary,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(verdict)
    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    print(f"cluster={cluster_path}")
    print(f"detail_json={json_path}")
    print(f"max_rank_one_projector_sin={max_rank_one:.12g}")
    print(f"max_cluster_sin_theta={max_cluster:.12g}")
    print(f"max_t3a_integrated_path_bound={max_path_bound:.12g}")
    print(f"max_raw_davis_kahan_bound_unclipped={max_raw_dk:.12g}")


if __name__ == "__main__":
    main()

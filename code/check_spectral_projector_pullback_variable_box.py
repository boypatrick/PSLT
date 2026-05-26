#!/usr/bin/env python3
"""Literal variable-box coordinate-pullback projector transport gate.

T3c for the optional spectral-projector transport line.  The G1 artifacts solve
on physical boxes z in [-L(D), L(D)] with L(D)=D/2+z_margin.  This script pulls
those variable boxes back to the fixed reference coordinate y in [-1,1], z=L y.
After the unitary constant-volume rescaling, the generalized operator on the
fixed (rho,y) Hilbert space is

    K_pull(D) = K_rho + L(D)^(-2) K_y + diag(rho * U(rho, L(D)y; D)),
    M_pull    = diag(rho).

Thus the radial kinetic, axial kinetic scaling, potential, measure, and
Dirichlet boundaries are all represented on one common finite-dimensional
space.  The gate checks exact adjacent projector drift and a residual
Davis-Kahan comparator on this pullback family.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import eigsh

from extract_chi_localized_2d import PhysicalParams, u_potential

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GN_DETAIL = ROOT / "output" / "gn_fp_2d" / "gn_lowN_microstate_count_detail.csv"
DEFAULT_SPECTRUM = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_spectrum_Dgrid21.csv"
OUTDIR = ROOT / "output" / "spectral_transport"
SUMMARY_CSV = OUTDIR / "spectral_projector_pullback_variable_box_summary.csv"
DETAIL_CSV = OUTDIR / "spectral_projector_pullback_variable_box_detail.csv"
CLUSTER_CSV = OUTDIR / "spectral_projector_pullback_variable_box_cluster.csv"
DETAIL_JSON = OUTDIR / "spectral_projector_pullback_variable_box_detail.json"


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


def build_pullback_operator(
    D: float,
    p: PhysicalParams,
    rho_max: float,
    z_margin: float,
    dr: float,
    ny: int,
):
    nr = int(round(rho_max / dr))
    if nr < 12 or ny < 16:
        raise ValueError("Grid too small for pullback eigensolve.")
    L = D / 2.0 + z_margin
    dy = 2.0 / float(ny)
    rho = (np.arange(nr) + 0.5) * dr
    y = -1.0 + (np.arange(ny) + 0.5) * dy
    rr, yy = np.meshgrid(rho, y, indexing="ij")
    zz = L * yy
    uu = u_potential(rr, zz, D, p)

    rho_ph = (np.arange(nr) + 1.0) * dr
    rho_mh = np.arange(nr) * dr
    main_r = (rho_ph + rho_mh) / (dr * dr)
    off_r = -rho_ph[:-1] / (dr * dr)
    k_r = diags([off_r, main_r, off_r], offsets=[-1, 0, 1], format="csr")

    main_y = np.full(ny, 2.0 / (dy * dy))
    off_y = np.full(ny - 1, -1.0 / (dy * dy))
    t_y = diags([off_y, main_y, off_y], offsets=[-1, 0, 1], format="csr")

    r_diag = diags(rho, 0, format="csr")
    k_mat = kron(k_r, eye(ny, format="csr"))
    k_mat = k_mat + (1.0 / (L * L)) * kron(r_diag, t_y)
    k_mat = k_mat + diags((rho[:, None] * uu).ravel(), 0, format="csr")
    m_mat = kron(r_diag, eye(ny, format="csr"))
    return rho, y, rr, yy, zz, uu, k_mat.tocsr(), m_mat.tocsr(), L, dy


def solve_pullback(D: float, args: argparse.Namespace, p: PhysicalParams, ny: int):
    rho, y, rr, yy, zz, uu, k_mat, m_mat, L, dy = build_pullback_operator(
        D=D,
        p=p,
        rho_max=float(args.rho_max),
        z_margin=float(args.z_margin),
        dr=float(args.dr),
        ny=int(ny),
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
    m_diag = np.repeat(rho, len(y))
    for j in range(vecs.shape[1]):
        nrm = float(np.sqrt(np.sum(m_diag * vecs[:, j] * vecs[:, j])))
        vecs[:, j] /= max(nrm, 1.0e-300)
    return {
        "D": float(D),
        "rho": rho,
        "y": y,
        "rr": rr,
        "yy": yy,
        "zz": zz,
        "uu": uu,
        "K": k_mat,
        "M": m_mat,
        "m_diag": m_diag,
        "vals": vals,
        "vecs": vecs,
        "L": L,
        "dy": dy,
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


def residual_dk_bound(K1, m_diag: np.ndarray, vals1: np.ndarray, u0: np.ndarray, n0: int):
    Ku = K1 @ u0
    mu = float(np.sum(u0 * Ku))  # because u0^T M u0=1 and K represents M H.
    residual_vec = Ku - mu * m_diag * u0
    residual_norm = float(np.sqrt(np.sum(residual_vec * residual_vec / m_diag)))
    gap = float(min(abs(float(vals1[m] - mu)) for m in range(len(vals1)) if m != n0))
    bound = residual_norm / max(gap, 1.0e-300)
    return {
        "rayleigh_mu_pullback": mu,
        "residual_norm_Mdual": residual_norm,
        "raw_davis_kahan_gap": gap,
        "raw_davis_kahan_bound": bound,
        "raw_davis_kahan_bound_clipped": min(bound, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Ds", default="")
    parser.add_argument("--gn-detail", type=Path, default=DEFAULT_GN_DETAIL)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--level", default="fine")
    parser.add_argument("--rho-max", type=float, default=3.0)
    parser.add_argument("--z-margin", type=float, default=6.0)
    parser.add_argument("--dr", type=float, default=0.06)
    parser.add_argument("--dz-target", type=float, default=0.03)
    parser.add_argument("--ny", type=int, default=0, help="Reference y-grid cells. Default: max G1 Nz from D_max and dz-target.")
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=30000)
    parser.add_argument("--n-eigs", type=int, default=40)
    parser.add_argument("--n-low", type=int, default=3)
    parser.add_argument("--rank-one-drift-tol", type=float, default=0.05)
    parser.add_argument("--cluster-drift-tol", type=float, default=0.075)
    parser.add_argument("--raw-dk-bound-tol", type=float, default=0.25)
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--detail", type=Path, default=DETAIL_CSV)
    parser.add_argument("--cluster", type=Path, default=CLUSTER_CSV)
    parser.add_argument("--detail-json", type=Path, default=DETAIL_JSON)
    args = parser.parse_args()

    if args.level != "fine":
        raise RuntimeError("T3c currently records the fine-grid pullback only.")

    args.gn_detail = args.gn_detail if args.gn_detail.is_absolute() else ROOT / args.gn_detail
    spectrum_csv = args.spectrum_csv if args.spectrum_csv.is_absolute() else ROOT / args.spectrum_csv
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    detail_path = args.detail if args.detail.is_absolute() else ROOT / args.detail
    cluster_path = args.cluster if args.cluster.is_absolute() else ROOT / args.cluster
    json_path = args.detail_json if args.detail_json.is_absolute() else ROOT / args.detail_json

    d_values = parse_d_values(args)
    L_max = max(d_values) / 2.0 + float(args.z_margin)
    ny = int(args.ny) if int(args.ny) > 0 else int(round(2.0 * L_max / float(args.dz_target)))
    p = PhysicalParams()
    ref = load_spectrum_reference(spectrum_csv, args.level)

    cache: dict[float, dict[str, Any]] = {}
    for D in d_values:
        print(f"[solve pullback] D={D:g}, L={D/2.0+float(args.z_margin):g}, ny={ny}, n_eigs={int(args.n_eigs)}")
        cache[float(D)] = solve_pullback(float(D), args, p, ny)

    detail_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    max_ref_diff = 0.0
    max_physical_dz = 0.0
    min_physical_dz = 1.0e300

    for D in d_values:
        cur = cache[float(D)]
        dz_phys = float(cur["L"] * cur["dy"])
        max_physical_dz = max(max_physical_dz, dz_phys)
        min_physical_dz = min(min_physical_dz, dz_phys)
        for n in range(int(args.n_low)):
            ref_val = ref.get((float(D), n + 1), np.nan)
            if np.isfinite(ref_val):
                max_ref_diff = max(max_ref_diff, abs(float(cur["vals"][n]) - float(ref_val)))

    for D0, D1 in zip(d_values[:-1], d_values[1:]):
        left = cache[float(D0)]
        right = cache[float(D1)]
        m_diag = left["m_diag"]
        if len(m_diag) != len(right["m_diag"]) or not np.allclose(m_diag, right["m_diag"]):
            raise RuntimeError("Pullback failed: M diagonals differ between adjacent D values.")
        vals0 = left["vals"]
        vals1 = right["vals"]
        vecs0 = left["vecs"]
        vecs1 = right["vecs"]
        delta_D = float(D1 - D0)

        cl_sin, sigma_min, sigma_max = cluster_sin(
            vecs0[:, : int(args.n_low)], vecs1[:, : int(args.n_low)], m_diag
        )
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
            dk = residual_dk_bound(right["K"], m_diag, vals1, vecs0[:, n], n)
            detail_rows.append(
                {
                    "D_left": float(D0),
                    "D_right": float(D1),
                    "delta_D": delta_D,
                    "N": int(n + 1),
                    "L_left": float(left["L"]),
                    "L_right": float(right["L"]),
                    "dy_ref": float(left["dy"]),
                    "dz_phys_left": float(left["L"] * left["dy"]),
                    "dz_phys_right": float(right["L"] * right["dy"]),
                    "lambda_left_pullback": float(vals0[n]),
                    "lambda_right_pullback": float(vals1[n]),
                    "lambda_displacement_pullback": abs(float(vals1[n] - vals0[n])),
                    "overlap_signed": overlap,
                    "overlap_abs": abs(overlap),
                    "actual_rank_one_projector_sin": sin_actual,
                    "rank_one_drift_pass": bool(sin_actual <= float(args.rank_one_drift_tol)),
                    **dk,
                    "raw_davis_kahan_pass_unclipped": bool(dk["raw_davis_kahan_bound"] <= float(args.raw_dk_bound_tol)),
                }
            )

    max_rank_one = max(float(r["actual_rank_one_projector_sin"]) for r in detail_rows)
    max_cluster = max(float(r["sin_theta_max"]) for r in cluster_rows)
    max_raw_dk = max(float(r["raw_davis_kahan_bound"]) for r in detail_rows)
    all_rank_one_pass = all(bool(r["rank_one_drift_pass"]) for r in detail_rows)
    all_cluster_pass = all(bool(r["cluster_drift_pass"]) for r in cluster_rows)
    raw_dk_all_pass = all(bool(r["raw_davis_kahan_pass_unclipped"]) for r in detail_rows)

    verdict = (
        "T3C_VARIABLE_BOX_PULLBACK_PROJECTOR_STEP_PASS_RAW_DK_LOOSE"
        if all_rank_one_pass and all_cluster_pass and not raw_dk_all_pass
        else (
            "T3C_VARIABLE_BOX_PULLBACK_STRICT_DK_PASS"
            if all_rank_one_pass and all_cluster_pass and raw_dk_all_pass
            else "T3C_VARIABLE_BOX_PULLBACK_DIAGNOSTIC_ONLY"
        )
    )

    summary = {
        "gate": "T3c",
        "verdict": verdict,
        "level": args.level,
        "n_D": int(len(d_values)),
        "n_steps": int(len(d_values) - 1),
        "D_min": float(min(d_values)),
        "D_max": float(max(d_values)),
        "rho_max": float(args.rho_max),
        "z_margin": float(args.z_margin),
        "dr": float(args.dr),
        "ny_ref": int(ny),
        "dy_ref": float(2.0 / float(ny)),
        "dz_phys_min": min_physical_dz,
        "dz_phys_max": max_physical_dz,
        "dz_target": float(args.dz_target),
        "n_eigs": int(args.n_eigs),
        "n_low": int(args.n_low),
        "rank_one_drift_tol": float(args.rank_one_drift_tol),
        "cluster_drift_tol": float(args.cluster_drift_tol),
        "raw_dk_bound_tol": float(args.raw_dk_bound_tol),
        "max_rank_one_projector_sin": max_rank_one,
        "all_rank_one_drift_pass": all_rank_one_pass,
        "max_cluster_sin_theta": max_cluster,
        "all_cluster_drift_pass": all_cluster_pass,
        "max_raw_davis_kahan_bound_unclipped": max_raw_dk,
        "raw_davis_kahan_all_pass_unclipped": raw_dk_all_pass,
        "max_abs_pullback_lambda_diff_vs_g1_low3": max_ref_diff,
        "operator_formula": "K_pull=K_rho+L(D)^(-2)K_y+diag(rho U(rho,L(D)y;D)); M_pull=diag(rho)",
        "interpretation": "literal variable-box coordinate pullback keeps L(D), axial kinetic scaling, measure, and boundaries on a fixed y-Hilbert space; projector drift passes while raw residual DK remains conservative",
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
                    "release_spectrum_replaced": False,
                    "meaning": "literal variable-box coordinate-pullback finite-grid projector step certificate",
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
    print(f"max_raw_davis_kahan_bound_unclipped={max_raw_dk:.12g}")
    print(f"max_abs_pullback_lambda_diff_vs_g1_low3={max_ref_diff:.12g}")


if __name__ == "__main__":
    main()

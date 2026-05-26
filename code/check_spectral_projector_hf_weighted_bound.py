#!/usr/bin/env python3
"""Weighted Hellmann-Feynman matrix-element bound for low-mode transport.

This is T3a for the optional spectral-projector transport line.  T2 proved a
valid but very loose sup-norm envelope for ||H'(D)||.  Here we use the actual
low-mode weights to measure the matrix elements

    A_mn(D) = <u_m(D), H'(D) u_n(D)>

for the same generalized finite-volume operator used by G1.  In the fixed-box
operator reading, H'(D) is multiplication by d_D U(D); in generalized variables
with M=diag(rho),

    A_mn = u_m^T diag(rho d_D U) u_n,

with eigenvectors M-normalized.  For a simple eigenprojector,

    ||P_n'(D)|| <= (
        sum_{m!=n} |A_mn|^2 / |lambda_m-lambda_n|^2
    )^{1/2},

and the uncomputed high-mode tail is bounded by the weighted variance
<dU^2>_n - sum_{computed m} |A_mn|^2 divided by a tail spectral gap floor.

This script is still non-baseline and does not alter paper/main.tex.
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

from extract_chi_localized_2d import Level, PhysicalParams, build_generalized_operator
from check_spectral_projector_derivative_envelope import dD_u_potential

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GN_DETAIL = ROOT / "output" / "gn_fp_2d" / "gn_lowN_microstate_count_detail.csv"
DEFAULT_SPECTRUM = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_spectrum_Dgrid21.csv"
DEFAULT_TRANSPORT = ROOT / "output" / "spectral_transport" / "spectral_projector_transport_summary.csv"
OUTDIR = ROOT / "output" / "spectral_transport"
SUMMARY_CSV = OUTDIR / "spectral_projector_hf_weighted_bound_summary.csv"
DETAIL_CSV = OUTDIR / "spectral_projector_hf_weighted_bound_detail.csv"
MATRIX_CSV = OUTDIR / "spectral_projector_hf_weighted_bound_matrix.csv"
DETAIL_JSON = OUTDIR / "spectral_projector_hf_weighted_bound_detail.json"


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


def load_transport_summary(path: Path) -> dict[str, float]:
    row = pd.read_csv(path).iloc[0].to_dict()
    out: dict[str, float] = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def load_spectrum_reference(path: Path, level: str) -> dict[tuple[float, int], float]:
    spec = pd.read_csv(path)
    spec = spec[spec["level"].astype(str) == str(level)].copy()
    return {(float(row.D), int(row.mode_n)): float(row.lambda_n) for row in spec.itertuples(index=False)}


def solve_modes(D: float, args: argparse.Namespace, p: PhysicalParams, level: Level):
    z_max = D / 2.0 + float(args.z_margin)
    rho, z, rr, zz, _uu, k_mat, m_mat = build_generalized_operator(
        D=D,
        p=p,
        rho_max=float(args.rho_max),
        z_max=z_max,
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

    # M is exactly diag(rho) repeated along z in the generalized problem.
    nz = len(z)
    m_diag = np.repeat(rho, nz)
    for j in range(vecs.shape[1]):
        nrm = float(np.sqrt(np.sum(m_diag * vecs[:, j] * vecs[:, j])))
        vecs[:, j] /= max(nrm, 1.0e-300)
    return rho, z, rr, zz, vals, vecs, m_diag


def run_case(D: float, args: argparse.Namespace, p: PhysicalParams, level: Level, ref: dict[tuple[float, int], float]):
    rho, z, rr, zz, vals, vecs, m_diag = solve_modes(D, args, p, level)
    dU = dD_u_potential(rr, zz, D, p).ravel()
    weighted_dU = m_diag * dU
    weighted_dU2 = m_diag * dU * dU

    # Matrix elements of H'=multiplication by dU in the M-normalized eigenbasis.
    A = vecs.T @ (weighted_dU[:, None] * vecs)
    A2 = np.sum(weighted_dU2[:, None] * vecs * vecs, axis=0)
    k = len(vals)
    # Reserve the last computed Ritz value as a conservative spectral floor for
    # the unexpanded tail.  Therefore only modes 1..k-1 are subtracted from the
    # variance identity; mode k and all uncomputed modes remain in the tail.
    included_basis_power = np.sum(A[: max(k - 1, 1), :] * A[: max(k - 1, 1), :], axis=0)
    residual_power = np.maximum(A2 - included_basis_power, 0.0)

    detail_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    n_low = int(args.n_low)
    for n0 in range(n_low):
        lam_n = float(vals[n0])
        included_sq = 0.0
        nearest_gap = np.inf
        max_offdiag_abs = 0.0
        for m0 in range(k - 1):  # reserve the last Ritz value as a conservative tail-gap floor
            if m0 == n0:
                continue
            gap = abs(float(vals[m0] - vals[n0]))
            nearest_gap = min(nearest_gap, gap)
            val = abs(float(A[m0, n0]))
            max_offdiag_abs = max(max_offdiag_abs, val)
            included_sq += (val / max(gap, 1.0e-300)) ** 2
        included_bound = float(np.sqrt(included_sq))
        tail_gap = abs(float(vals[-1] - vals[n0]))
        tail_residual = float(np.sqrt(max(residual_power[n0], 0.0)))
        tail_bound = tail_residual / max(tail_gap, 1.0e-300)
        total_bound = included_bound + tail_bound
        hf_diag = float(A[n0, n0])
        fd_slope_ref = np.nan
        # Ref slopes are handled at summary level; per-row stores only HF diagonal.
        lam_ref = ref.get((float(D), n0 + 1), np.nan)
        detail_rows.append(
            {
                "D": float(D),
                "N": int(n0 + 1),
                "lambda_N": lam_n,
                "lambda_ref": float(lam_ref) if np.isfinite(lam_ref) else np.nan,
                "lambda_abs_diff_vs_ref": abs(lam_n - float(lam_ref)) if np.isfinite(lam_ref) else np.nan,
                "hf_lambda_prime_diag": hf_diag,
                "hf_abs_diag": abs(hf_diag),
                "weighted_rms_dD_U": float(np.sqrt(max(A2[n0], 0.0))),
                "weighted_std_dD_U": float(np.sqrt(max(A2[n0] - hf_diag * hf_diag, 0.0))),
                "included_basis_coupling_power": float(included_basis_power[n0]),
                "tail_residual_coupling": tail_residual,
                "tail_gap_floor_reserved_last_mode": tail_gap,
                "nearest_gap_within_computed": float(nearest_gap),
                "max_offdiag_abs_A_mn": max_offdiag_abs,
                "included_projector_derivative_bound": included_bound,
                "tail_projector_derivative_bound": tail_bound,
                "total_projector_derivative_bound": total_bound,
                "n_eigs": int(k),
                "reserved_tail_mode_index": int(k),
                "fd_slope_ref_placeholder": fd_slope_ref,
            }
        )

    for i in range(min(k, int(args.matrix_export_modes))):
        for j in range(min(k, int(args.matrix_export_modes))):
            matrix_rows.append(
                {
                    "D": float(D),
                    "i": int(i + 1),
                    "j": int(j + 1),
                    "lambda_i": float(vals[i]),
                    "lambda_j": float(vals[j]),
                    "A_ij": float(A[i, j]),
                    "abs_A_ij": abs(float(A[i, j])),
                    "gap_abs": abs(float(vals[i] - vals[j])),
                }
            )
    return detail_rows, matrix_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Ds", default="")
    parser.add_argument("--gn-detail", type=Path, default=DEFAULT_GN_DETAIL)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--transport-summary", type=Path, default=DEFAULT_TRANSPORT)
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
    parser.add_argument("--matrix-export-modes", type=int, default=8)
    parser.add_argument("--max-total-projector-bound-tol", type=float, default=0.25)
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--detail", type=Path, default=DETAIL_CSV)
    parser.add_argument("--matrix", type=Path, default=MATRIX_CSV)
    parser.add_argument("--detail-json", type=Path, default=DETAIL_JSON)
    args = parser.parse_args()

    if args.level != "fine":
        raise RuntimeError("T3a currently records the fine-grid release-aligned bound only.")
    if int(args.n_eigs) < int(args.n_low) + 2:
        raise RuntimeError("Need at least n_low+2 Ritz values to reserve a tail gap floor.")

    gn_detail = args.gn_detail if args.gn_detail.is_absolute() else ROOT / args.gn_detail
    spectrum_csv = args.spectrum_csv if args.spectrum_csv.is_absolute() else ROOT / args.spectrum_csv
    transport_csv = args.transport_summary if args.transport_summary.is_absolute() else ROOT / args.transport_summary
    args.gn_detail = gn_detail

    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    detail_path = args.detail if args.detail.is_absolute() else ROOT / args.detail
    matrix_path = args.matrix if args.matrix.is_absolute() else ROOT / args.matrix
    json_path = args.detail_json if args.detail_json.is_absolute() else ROOT / args.detail_json

    level = Level("fine", dr=float(args.dr), dz=float(args.dz))
    p = PhysicalParams()
    d_values = parse_d_values(args)
    ref = load_spectrum_reference(spectrum_csv, args.level)
    transport = load_transport_summary(transport_csv)

    all_detail: list[dict[str, Any]] = []
    all_matrix: list[dict[str, Any]] = []
    for D in d_values:
        print(f"[run] D={D:g}, n_eigs={int(args.n_eigs)}")
        detail_rows, matrix_rows = run_case(float(D), args, p, level, ref)
        all_detail.extend(detail_rows)
        all_matrix.extend(matrix_rows)
        worst = max(r["total_projector_derivative_bound"] for r in detail_rows)
        diag = ", ".join(f"N{r['N']} diag={r['hf_lambda_prime_diag']:.4e}" for r in detail_rows)
        print(f"      max_total_Pprime_bound={worst:.6e}; {diag}")

    detail = pd.DataFrame(all_detail).sort_values(["D", "N"]).reset_index(drop=True)
    matrix = pd.DataFrame(all_matrix).sort_values(["D", "i", "j"]).reset_index(drop=True)

    # Compare HF diagonal derivatives with finite-difference slopes from T1 in scale, not pointwise.
    max_total = float(detail["total_projector_derivative_bound"].max())
    max_included = float(detail["included_projector_derivative_bound"].max())
    max_tail = float(detail["tail_projector_derivative_bound"].max())
    max_weighted_rms = float(detail["weighted_rms_dD_U"].max())
    max_weighted_std = float(detail["weighted_std_dD_U"].max())
    max_hf_diag = float(detail["hf_abs_diag"].max())
    max_lambda_diff = float(detail["lambda_abs_diff_vs_ref"].max())
    all_lambda_match = bool(max_lambda_diff <= 5.0e-8)
    pass_projector_bound = bool(max_total <= float(args.max_total_projector_bound_tol))
    verdict = (
        "T3A_WEIGHTED_HF_PROJECTOR_BOUND_PASS"
        if all_lambda_match and pass_projector_bound
        else "T3A_WEIGHTED_HF_BOUND_DIAGNOSTIC_ONLY"
    )

    summary = {
        "gate": "T3a",
        "verdict": verdict,
        "level": args.level,
        "n_D": int(len(d_values)),
        "D_min": float(min(d_values)),
        "D_max": float(max(d_values)),
        "rho_max": float(args.rho_max),
        "z_margin": float(args.z_margin),
        "dr": float(args.dr),
        "dz": float(args.dz),
        "sigma": float(args.sigma),
        "n_eigs": int(args.n_eigs),
        "n_low": int(args.n_low),
        "max_lambda_abs_diff_vs_g1_spectrum": max_lambda_diff,
        "all_lambda_match_g1_spectrum": all_lambda_match,
        "max_hf_abs_diag_lambda_prime": max_hf_diag,
        "max_weighted_rms_dD_U": max_weighted_rms,
        "max_weighted_std_dD_U": max_weighted_std,
        "max_included_projector_derivative_bound": max_included,
        "max_tail_projector_derivative_bound": max_tail,
        "max_total_projector_derivative_bound": max_total,
        "max_total_projector_bound_tol": float(args.max_total_projector_bound_tol),
        "pass_projector_bound_tol": pass_projector_bound,
        "t1_max_fd_slope_N1": float(transport["max_abs_lambda_slope_N1"]),
        "t1_max_fd_slope_N2": float(transport["max_abs_lambda_slope_N2"]),
        "t1_max_fd_slope_N3": float(transport["max_abs_lambda_slope_N3"]),
        "t2_fixed_grid_supnorm": float(pd.read_csv(ROOT / "output" / "spectral_transport" / "spectral_projector_derivative_envelope_summary.csv").iloc[0]["max_grid_sup_abs_dD_U_fixed_box"]),
        "hf_diag_over_t2_supnorm": float(max_hf_diag / max(float(pd.read_csv(ROOT / "output" / "spectral_transport" / "spectral_projector_derivative_envelope_summary.csv").iloc[0]["max_grid_sup_abs_dD_U_fixed_box"]), 1.0e-300)),
        "interpretation": "weighted low-mode matrix elements avoid the Plummer-core sup-norm spike; still a finite-grid certificate unless paired with inter-knot operator control",
    }

    write_csv(summary_path, [summary])
    write_csv(detail_path, detail.to_dict(orient="records"))
    write_csv(matrix_path, matrix.to_dict(orient="records"))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(
            {
                "claim_firewall": {
                    "baseline_change": False,
                    "main_tex_change": False,
                    "continuum_no_crossing_claimed": False,
                    "meaning": "finite-grid weighted Hellmann-Feynman/projector derivative bound for the current G1 operator chain",
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
    print(f"matrix={matrix_path}")
    print(f"detail_json={json_path}")
    print(f"max_total_projector_derivative_bound={max_total:.12g}")
    print(f"max_hf_abs_diag_lambda_prime={max_hf_diag:.12g}")
    print(f"max_weighted_rms_dD_U={max_weighted_rms:.12g}")


if __name__ == "__main__":
    main()

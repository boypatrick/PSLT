#!/usr/bin/env python3
"""
V2 protected Spin^c nodal/Sturm ordering diagnostic.

This optional diagnostic checks whether the first three finite-volume spectral
modes admit the physically readable ordering expected from the protected
degree flag:

  O(2) degree basis: 1, zeta, zeta^2
  expected parity:   even, odd, even
  expected nodes:    0, 1, 2

The script reuses the G1 finite-volume operator convention: compute a broad
shift-invert Ritz window, sort the returned eigenvalues, and test the lowest
three modes.  It does not modify the submission baseline.

Outputs:
  - output/spinc_bridge/spinc_nodal_sturm_detail.csv
  - output/spinc_bridge/spinc_nodal_sturm_summary.csv
"""

from __future__ import annotations

import argparse
import csv
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
EXPECTED_PARITY = ["even", "odd", "even"]
EXPECTED_NODES = [0, 1, 2]


@dataclass(frozen=True)
class NodalConfig:
    rho_max: float = 3.0
    z_margin: float = 6.0
    dr: float = 0.06
    dz: float = 0.03
    sigma: float = 2.5
    tol: float = 1e-8
    maxiter: int = 30000
    n_eigs: int = 40
    parity_ratio_tol: float = 1e-3
    parity_abs_guard: float = 2e-3
    node_threshold_frac: float = 1e-3


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


def solve_modes(D: float, cfg: NodalConfig):
    p = PhysicalParams()
    level = Level("fine", dr=cfg.dr, dz=cfg.dz)
    z_max = D / 2.0 + cfg.z_margin
    rho, z, _rr, zz, _uu, k_mat, m_mat = build_generalized_operator(
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
        k=max(6, int(cfg.n_eigs)),
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
    return rho, z, zz, evals[:3], psi


def weighted_norm(arr: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.sum(w * arr * arr)))


def count_sign_changes(profile: np.ndarray, threshold_frac: float) -> tuple[int, float, int]:
    prof = np.asarray(profile, dtype=float)
    max_abs = float(np.max(np.abs(prof)))
    if max_abs <= 0.0:
        return 0, 0.0, 0
    keep = np.abs(prof) > float(threshold_frac) * max_abs
    signs = np.sign(prof[keep])
    if len(signs) <= 1:
        return 0, max_abs, int(len(signs))
    return int(np.sum(signs[1:] * signs[:-1] < 0)), max_abs, int(len(signs))


def run_case(D: float, cfg: NodalConfig) -> list[dict[str, Any]]:
    rho, _z, _zz, evals, psi = solve_modes(D, cfg)
    w = 2.0 * np.pi * rho[:, None] * cfg.dr * cfg.dz * np.ones_like(_zz)
    radial_weight_no_dz = 2.0 * np.pi * rho[:, None] * cfg.dr

    rows: list[dict[str, Any]] = []
    for i in range(3):
        mode = i + 1
        u = psi[:, :, i]
        u_reflected = u[:, ::-1]
        even_resid = weighted_norm(u - u_reflected, w)
        odd_resid = weighted_norm(u + u_reflected, w)
        observed_parity = "even" if even_resid <= odd_resid else "odd"
        expected_parity = EXPECTED_PARITY[i]
        expected_resid = even_resid if expected_parity == "even" else odd_resid
        wrong_resid = odd_resid if expected_parity == "even" else even_resid
        parity_ratio = expected_resid / max(wrong_resid, 1e-300)

        axial_profile = np.sum(radial_weight_no_dz * u, axis=0)
        axial_nodes, axial_max, axial_kept = count_sign_changes(axial_profile, cfg.node_threshold_frac)

        radial_energy = np.sum(w * u * u, axis=1)
        dominant_rho_index = int(np.argmax(radial_energy))
        dominant_line = u[dominant_rho_index, :]
        line_nodes, line_max, line_kept = count_sign_changes(dominant_line, cfg.node_threshold_frac)

        expected_nodes = EXPECTED_NODES[i]
        parity_pass = (
            observed_parity == expected_parity
            and parity_ratio <= cfg.parity_ratio_tol
            and expected_resid <= cfg.parity_abs_guard
        )
        axial_nodes_pass = axial_nodes == expected_nodes
        line_nodes_pass = line_nodes == expected_nodes

        rows.append(
            {
                "D": float(D),
                "mode_n": mode,
                "lambda_n": float(evals[i]),
                "expected_parity": expected_parity,
                "observed_parity": observed_parity,
                "even_resid": float(even_resid),
                "odd_resid": float(odd_resid),
                "expected_parity_resid": float(expected_resid),
                "wrong_parity_resid": float(wrong_resid),
                "parity_ratio": float(parity_ratio),
                "expected_nodes": expected_nodes,
                "axial_nodes": int(axial_nodes),
                "dominant_line_nodes": int(line_nodes),
                "axial_profile_max_abs": float(axial_max),
                "dominant_line_max_abs": float(line_max),
                "axial_kept_points": int(axial_kept),
                "dominant_line_kept_points": int(line_kept),
                "dominant_rho": float(rho[dominant_rho_index]),
                "parity_pass": bool(parity_pass),
                "axial_nodes_pass": bool(axial_nodes_pass),
                "dominant_line_nodes_pass": bool(line_nodes_pass),
                "row_pass": bool(parity_pass and axial_nodes_pass and line_nodes_pass),
            }
        )
    return rows


def write_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="V2 Spin^c nodal/Sturm ordering diagnostic.")
    ap.add_argument("--Ds", default="", help="Comma-separated D values. Default: audited G1 D grid.")
    ap.add_argument("--gn-detail", type=Path, default=DEFAULT_GN_DETAIL)
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    ap.add_argument("--n-eigs", type=int, default=40)
    ap.add_argument("--parity-ratio-tol", type=float, default=1e-3)
    ap.add_argument("--parity-abs-guard", type=float, default=2e-3)
    ap.add_argument("--node-threshold-frac", type=float, default=1e-3)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    cfg = NodalConfig(
        rho_max=float(args.rho_max),
        z_margin=float(args.z_margin),
        dr=float(args.dr),
        dz=float(args.dz),
        sigma=float(args.sigma),
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        n_eigs=int(args.n_eigs),
        parity_ratio_tol=float(args.parity_ratio_tol),
        parity_abs_guard=float(args.parity_abs_guard),
        node_threshold_frac=float(args.node_threshold_frac),
    )

    detail_rows: list[dict[str, Any]] = []
    for D in parse_d_values(args):
        print(f"[run] D={D:g}")
        rows = run_case(float(D), cfg)
        detail_rows.extend(rows)
        pattern = ",".join(f"{r['observed_parity']}:{r['axial_nodes']}/{r['dominant_line_nodes']}" for r in rows)
        print(f"      {pattern}; pass={all(r['row_pass'] for r in rows)}")

    detail = pd.DataFrame(detail_rows).sort_values(["D", "mode_n"]).reset_index(drop=True)
    all_pass = bool(detail["row_pass"].all())
    status = "NODAL_STURM_ORDER_CONFIRMED" if all_pass else "NODAL_STURM_ORDER_MISMATCH"
    summary = {
        "n_D": int(detail["D"].nunique()),
        "n_rows": int(len(detail)),
        "D_min": float(detail["D"].min()),
        "D_max": float(detail["D"].max()),
        "expected_parity_pattern": "even,odd,even",
        "expected_node_pattern": "0,1,2",
        "max_expected_parity_resid": float(detail["expected_parity_resid"].max()),
        "min_wrong_parity_resid": float(detail["wrong_parity_resid"].min()),
        "max_parity_ratio": float(detail["parity_ratio"].max()),
        "parity_ratio_tol": float(cfg.parity_ratio_tol),
        "parity_abs_guard": float(cfg.parity_abs_guard),
        "max_abs_axial_node_error": int((detail["axial_nodes"] - detail["expected_nodes"]).abs().max()),
        "max_abs_dominant_line_node_error": int(
            (detail["dominant_line_nodes"] - detail["expected_nodes"]).abs().max()
        ),
        "all_parity_pass": bool(detail["parity_pass"].all()),
        "all_axial_nodes_pass": bool(detail["axial_nodes_pass"].all()),
        "all_dominant_line_nodes_pass": bool(detail["dominant_line_nodes_pass"].all()),
        "all_row_pass": all_pass,
        "status": status,
        "baseline_changed": False,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    detail_path = args.outdir / "spinc_nodal_sturm_detail.csv"
    summary_path = args.outdir / "spinc_nodal_sturm_summary.csv"
    detail.to_csv(detail_path, index=False)
    write_summary(summary_path, summary)
    print(f"\n[done] wrote {detail_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[status] {status}")

    if not all_pass:
        raise SystemExit("Nodal/Sturm ordering gate failed; see detail CSV.")


if __name__ == "__main__":
    main()

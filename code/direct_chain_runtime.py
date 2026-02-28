#!/usr/bin/env python3
"""
Runtime builder for full-direct action-derived scan profiles.

This module promotes the extraction path
    V_eff -> localized modes -> chi_LR(D), A_l(D)
into the scan runtime chain by building the active D-grid profiles on demand.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from extract_chi_localized_2d import Level as ChiLevel
from extract_chi_localized_2d import PhysicalParams as ChiParams
from extract_chi_localized_2d import run_case as run_chi_case
from extract_superrad_prefactor_1d import Level as SuperradLevel
from extract_superrad_prefactor_1d import PhysicalParams as SuperradParams
from extract_superrad_prefactor_1d import solve_case as run_superrad_case


def _read_d_column(path: Path, value_col: str) -> np.ndarray | None:
    if not path.exists():
        return None
    vals = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_raw = row.get("D", "")
            v_raw = row.get(value_col, "")
            if d_raw in {"", None} or v_raw in {"", None}:
                continue
            level = str(row.get("level", "")).strip().lower()
            if level and level != "fine":
                continue
            vals.append(float(d_raw))
    if len(vals) < 2:
        return None
    arr = np.array(sorted(vals), dtype=float)
    return arr


def _matches_scan_grid(path: Path, d_scan: np.ndarray, value_col: str, tol: float = 1e-8) -> bool:
    d = _read_d_column(path, value_col=value_col)
    if d is None:
        return False
    if len(d) != len(d_scan):
        return False
    return bool(np.allclose(d, np.asarray(d_scan, dtype=float), rtol=0.0, atol=tol))


def _extract_runtime_chi(
    d_scan: np.ndarray,
    out_csv: Path,
    rho_max: float,
    z_margin: float,
    n_mu: int,
    tol: float,
    maxiter: int,
    sigma: float,
) -> None:
    level = ChiLevel("fine", dr=0.06, dz=0.03)
    params = ChiParams()
    rows = []
    for i, d in enumerate(d_scan):
        if i % 10 == 0 or i == len(d_scan) - 1:
            print(f"[runtime direct chi] {i+1}/{len(d_scan)} D={float(d):.6f}")
        row = run_chi_case(
            D=float(d),
            level=level,
            p=params,
            rho_max=float(rho_max),
            z_margin=float(z_margin),
            n_mu=int(n_mu),
            tol=float(tol),
            maxiter=int(maxiter),
            sigma=None if float(sigma) < 0 else float(sigma),
        )
        row["grid_index"] = int(i)
        rows.append(row)
    pd.DataFrame(rows).sort_values("D").reset_index(drop=True).to_csv(out_csv, index=False)


def _extract_runtime_superrad(
    d_scan: np.ndarray,
    out_csv: Path,
    zmax: float,
    d_ref: float,
    n_ref: int,
) -> None:
    level = SuperradLevel("fine", Nz=8001)
    params = SuperradParams()
    ref = run_superrad_case(D=float(d_ref), level=level, zmax=float(zmax), p=params, n_ref=int(n_ref))
    a1_ref = float(ref["A1_fp"])
    a2_ref = float(ref["A2_fp"])

    rows = []
    for i, d in enumerate(d_scan):
        if i % 10 == 0 or i == len(d_scan) - 1:
            print(f"[runtime direct A_l] {i+1}/{len(d_scan)} D={float(d):.6f}")
        row = run_superrad_case(D=float(d), level=level, zmax=float(zmax), p=params, n_ref=int(n_ref))
        row["grid_index"] = int(i)
        row["A1_profile"] = float(row["A1_fp"]) / max(a1_ref, 1e-300)
        row["A2_profile"] = float(row["A2_fp"]) / max(a2_ref, 1e-300)
        row["D_ref_profile"] = float(d_ref)
        rows.append(row)
    pd.DataFrame(rows).sort_values("D").reset_index(drop=True).to_csv(out_csv, index=False)


def ensure_runtime_full_direct_profiles(
    root: Path,
    d_scan: np.ndarray,
    force: bool = False,
    chi_rho_max: float = 3.0,
    chi_z_margin: float = 6.0,
    chi_n_mu: int = 120,
    chi_tol: float = 1e-8,
    chi_maxiter: int = 30000,
    chi_sigma: float = 2.5,
    superrad_zmax: float = 80.0,
    superrad_ref_d: float = 12.0,
    superrad_n_ref: int = 2,
) -> Tuple[Path, Path]:
    """
    Ensure strict full_direct profile CSVs exist for the active scan D-grid.

    Returns:
      (chi_csv_path, superrad_csv_path)
    """
    d_scan = np.asarray(d_scan, dtype=float)
    outdir = root / "output" / "kinetic_action_chain"
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(d_scan)
    chi_csv = outdir / f"localized_direct_profiles_chi_Dgrid{n}.csv"
    superrad_csv = outdir / f"localized_direct_profiles_superrad_Dgrid{n}.csv"

    chi_ok = (not force) and _matches_scan_grid(chi_csv, d_scan, value_col="chi_LR")
    sup_ok = (not force) and _matches_scan_grid(superrad_csv, d_scan, value_col="A1_profile")

    if not chi_ok:
        print(f"[runtime direct] rebuilding chi profile: {chi_csv}")
        _extract_runtime_chi(
            d_scan=d_scan,
            out_csv=chi_csv,
            rho_max=float(chi_rho_max),
            z_margin=float(chi_z_margin),
            n_mu=int(chi_n_mu),
            tol=float(chi_tol),
            maxiter=int(chi_maxiter),
            sigma=float(chi_sigma),
        )
    else:
        print(f"[runtime direct] reuse chi profile: {chi_csv}")

    if not sup_ok:
        print(f"[runtime direct] rebuilding superrad profile: {superrad_csv}")
        _extract_runtime_superrad(
            d_scan=d_scan,
            out_csv=superrad_csv,
            zmax=float(superrad_zmax),
            d_ref=float(superrad_ref_d),
            n_ref=int(superrad_n_ref),
        )
    else:
        print(f"[runtime direct] reuse superrad profile: {superrad_csv}")

    return chi_csv, superrad_csv


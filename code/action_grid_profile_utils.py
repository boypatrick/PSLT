#!/usr/bin/env python3
"""
Utilities for selecting action-derived chi/superradiant profiles for scan grids.

Goal:
  Prefer exact grid-aligned action-derived profiles (no interpolation on scan D-grid),
  and gracefully fall back to legacy sparse-knot profiles when unavailable.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def scan_d_values(d_min: float, d_max: float, d_num: int) -> np.ndarray:
    return np.linspace(float(d_min), float(d_max), int(d_num))


def _read_chi_rows(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("D", "") in {"", None}:
                continue
            level = row.get("level", "").strip().lower()
            if level and level != "fine":
                continue
            chi = row.get("chi_LR", "")
            if chi in {"", None}:
                continue
            rows.append((float(row["D"]), float(chi)))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda t: t[0])
    d = np.array([x for x, _ in rows], dtype=float)
    chi = np.array([y for _, y in rows], dtype=float)
    return d, chi


def _read_superrad_rows(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("D", "") in {"", None}:
                continue
            level = row.get("level", "").strip().lower()
            if level and level != "fine":
                continue
            a1 = row.get("A1_profile", "")
            a2 = row.get("A2_profile", "")
            if a1 in {"", None} or a2 in {"", None}:
                continue
            rows.append((float(row["D"]), float(a1), float(a2)))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda t: t[0])
    d = np.array([r[0] for r in rows], dtype=float)
    a1 = np.array([r[1] for r in rows], dtype=float)
    a2 = np.array([r[2] for r in rows], dtype=float)
    return {"D": d, "A1": a1, "A2": a2}


def _matches_scan_grid(d_profile: np.ndarray, d_scan: np.ndarray, tol: float = 1e-8) -> bool:
    if len(d_profile) != len(d_scan):
        return False
    return bool(np.allclose(d_profile, d_scan, rtol=0.0, atol=tol))


def select_chi_profile(root: Path, d_scan: np.ndarray) -> Dict[str, object]:
    """
    Return:
      {
        "path": Path,
        "d": np.ndarray,
        "chi": np.ndarray,
        "mode": "localized_grid" | "localized_interp",
      }
    """
    chi_dir = root / "output" / "chi_fp_2d"
    kin_dir = root / "output" / "kinetic_action_chain"
    n = len(d_scan)
    candidates = [
        kin_dir / f"localized_direct_profiles_chi_Dgrid{n}.csv",
        chi_dir / f"localized_chi_Dgrid{n}_fine.csv",
        chi_dir / "localized_chi_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv",
        chi_dir / "localized_chi_D6-12-18.csv",
    ]
    candidates.extend(sorted(kin_dir.glob("localized_direct_profiles_chi_Dgrid*.csv")))
    candidates.extend(sorted(chi_dir.glob("localized_chi_D*.csv")))

    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        parsed = _read_chi_rows(path)
        if parsed is None:
            continue
        d, chi = parsed
        mode = "localized_grid" if _matches_scan_grid(d, d_scan) else "localized_interp"
        return {"path": path, "d": d, "chi": chi, "mode": mode}

    raise FileNotFoundError("No readable localized chi profile CSV found in output/chi_fp_2d.")


def select_superrad_profile(root: Path, d_scan: np.ndarray) -> Dict[str, object]:
    """
    Return:
      {
        "path": Path,
        "d": np.ndarray,
        "A1": np.ndarray,
        "A2": np.ndarray,
        "mode": "action_grid" | "action_profile",
      }
    """
    sup_dir = root / "output" / "superrad_fp_1d"
    kin_dir = root / "output" / "kinetic_action_chain"
    n = len(d_scan)
    candidates = [
        kin_dir / f"localized_direct_profiles_superrad_Dgrid{n}.csv",
        sup_dir / f"superrad_prefactor_Dgrid{n}_fine.csv",
        sup_dir / "superrad_prefactor_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv",
    ]
    candidates.extend(sorted(kin_dir.glob("localized_direct_profiles_superrad_Dgrid*.csv")))
    candidates.extend(sorted(sup_dir.glob("superrad_prefactor_D*.csv")))

    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        parsed = _read_superrad_rows(path)
        if parsed is None:
            continue
        mode = "action_grid" if _matches_scan_grid(parsed["D"], d_scan) else "action_profile"
        return {
            "path": path,
            "d": parsed["D"],
            "A1": parsed["A1"],
            "A2": parsed["A2"],
            "mode": mode,
        }

    raise FileNotFoundError("No readable superradiant profile CSV found in output/superrad_fp_1d.")

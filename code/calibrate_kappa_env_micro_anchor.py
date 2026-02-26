#!/usr/bin/env python3
"""
Anchor-based calibration for open_system_micro kappa_env.

Design goal:
  - avoid full-map fitting;
  - calibrate kappa_env from one fixed D anchor (default: D_ref=12).

Target definition (default):
  ratio_target(D_ref) := ratio_proxy_to_baseline from geometry-informed
  open-system profile at the same D_ref.

Model:
  ratio_micro(kappa; D_ref)
    = [2 * gamma_mix(kappa) * Cmax(kappa) / gamma_ref] / chi_LR
  with gamma_phi, gamma_mix scaled linearly from an existing micro CSV:
    gamma_* (kappa) = gamma_* (kappa_src) * (kappa / kappa_src).

Outputs:
  - output/chi_open_system/kappa_env_anchor_calibration.csv
  - output/chi_open_system/kappa_env_anchor_calibration.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


@dataclass(frozen=True)
class AnchorInputs:
    D_ref: float
    delta: float
    gamma_phi_src: float
    gamma_mix_src: float
    gamma_ref: float
    chi_lr: float
    kappa_src: float
    ratio_target: float
    target_source_csv: str
    micro_source_csv: str


def _pick_largest_csv(pattern: str) -> Path:
    cands = sorted(OUTDIR.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No files matched: {OUTDIR / pattern}")
    best: Optional[Path] = None
    best_n = -1
    for p in cands:
        try:
            n = len(pd.read_csv(p))
        except Exception:
            continue
        if n > best_n:
            best_n = n
            best = p
    if best is None:
        raise RuntimeError(f"Could not parse any CSV for pattern: {pattern}")
    return best


def _infer_kappa_src(df: pd.DataFrame) -> float:
    if "kappa_env" in df.columns and np.isfinite(df["kappa_env"].to_numpy(dtype=float)).any():
        return float(np.nanmedian(df["kappa_env"].to_numpy(dtype=float)))
    # Fallback: infer from gamma_mix relation.
    num = df["gamma_mix_micro"].to_numpy(dtype=float)
    den = np.maximum(
        df["g_x_micro"].to_numpy(dtype=float) ** 2 * df["Sxx_delta"].to_numpy(dtype=float),
        1e-30,
    )
    return float(np.nanmedian(num / den))


def _require_anchor_row(df: pd.DataFrame, d_ref: float, name: str) -> pd.Series:
    arr = df["D"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(arr - float(d_ref))))
    if abs(float(arr[idx]) - float(d_ref)) > 1e-9:
        raise ValueError(f"{name}: D_ref={d_ref} not found exactly in D grid.")
    return df.iloc[idx]


def _build_inputs(micro_csv: Path, geom_csv: Path, d_ref: float) -> AnchorInputs:
    df_micro = pd.read_csv(micro_csv)
    df_geom = pd.read_csv(geom_csv)
    r_micro = _require_anchor_row(df_micro, d_ref, "micro_csv")
    r_geom = _require_anchor_row(df_geom, d_ref, "geom_csv")
    kappa_src = _infer_kappa_src(df_micro)
    return AnchorInputs(
        D_ref=float(d_ref),
        delta=float(r_micro["delta_micro"] if "delta_micro" in r_micro else r_micro["delta"]),
        gamma_phi_src=float(r_micro["gamma_phi_micro"]),
        gamma_mix_src=float(r_micro["gamma_mix_micro"]),
        gamma_ref=float(r_micro["gamma_ref_micro"] if "gamma_ref_micro" in r_micro else r_micro["Gamma_ref"]),
        chi_lr=float(r_micro["chi_LR_baseline"]),
        kappa_src=float(kappa_src),
        ratio_target=float(r_geom["ratio_proxy_to_baseline"]),
        target_source_csv=str(geom_csv),
        micro_source_csv=str(micro_csv),
    )


def _lindblad_cmax(delta: float, gamma_phi: float, gamma_mix: float, tmax: float, nstep: int, rtol: float, atol: float) -> float:
    H = np.array([[0.0, delta / 2.0], [delta / 2.0, 0.0]], dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    L_ops = [np.sqrt(max(gamma_phi, 0.0)) * sigma_z, np.sqrt(max(gamma_mix, 0.0)) * sigma_x]

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        rho = y.reshape(2, 2)
        drho = -1j * (H @ rho - rho @ H)
        for L in L_ops:
            drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        return drho.reshape(-1)

    y0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex).reshape(-1)
    t_eval = np.linspace(0.0, tmax, nstep)
    sol = solve_ivp(rhs, (0.0, tmax), y0, t_eval=t_eval, rtol=rtol, atol=atol)
    cmax = 0.0
    for i in range(sol.y.shape[1]):
        rho = sol.y[:, i].reshape(2, 2)
        cmax = max(cmax, float(abs(rho[0, 1])))
    return cmax


def _ratio_at_kappa(kappa: float, inp: AnchorInputs, tmax: float, nstep: int, rtol: float, atol: float) -> float:
    scale = float(kappa / max(inp.kappa_src, 1e-30))
    gamma_phi = float(inp.gamma_phi_src * scale)
    gamma_mix = float(inp.gamma_mix_src * scale)
    cmax = _lindblad_cmax(
        delta=inp.delta,
        gamma_phi=gamma_phi,
        gamma_mix=gamma_mix,
        tmax=tmax,
        nstep=nstep,
        rtol=rtol,
        atol=atol,
    )
    chi_eff = float(max(2.0 * gamma_mix * cmax / max(inp.gamma_ref, 1e-30), 0.0))
    return float(chi_eff / max(inp.chi_lr, 1e-30))


def calibrate_anchor_kappa(
    inp: AnchorInputs,
    kappa_min: float,
    kappa_max: float,
    n_coarse: int,
    tmax: float,
    nstep: int,
    rtol: float,
    atol: float,
) -> Dict[str, float]:
    if not (kappa_min > 0 and kappa_max > kappa_min):
        raise ValueError("Require 0 < kappa_min < kappa_max.")
    k_coarse = np.geomspace(kappa_min, kappa_max, max(int(n_coarse), 5))
    rows: List[Tuple[float, float]] = []
    for k in k_coarse:
        rows.append((float(k), _ratio_at_kappa(float(k), inp, tmax, nstep, rtol, atol)))

    # Pick best coarse first.
    k_best, r_best = min(rows, key=lambda kv: abs(kv[1] - inp.ratio_target))

    # Optional local refine around best coarse by log-neighborhood.
    logk = np.log10(k_best)
    k_lo = max(kappa_min, 10 ** (logk - 0.25))
    k_hi = min(kappa_max, 10 ** (logk + 0.25))
    k_ref = np.geomspace(k_lo, k_hi, 17)
    ref_rows: List[Tuple[float, float]] = []
    for k in k_ref:
        ref_rows.append((float(k), _ratio_at_kappa(float(k), inp, tmax, nstep, rtol, atol)))
    k_best, r_best = min(ref_rows, key=lambda kv: abs(kv[1] - inp.ratio_target))

    return {
        "kappa_env_calibrated": float(k_best),
        "ratio_anchor_target": float(inp.ratio_target),
        "ratio_anchor_pred": float(r_best),
        "ratio_anchor_abs_err": float(abs(r_best - inp.ratio_target)),
        "D_ref": float(inp.D_ref),
        "kappa_src_micro_csv": float(inp.kappa_src),
        "micro_source_csv": inp.micro_source_csv,
        "target_source_csv": inp.target_source_csv,
        "target_definition": "single-point anchor: ratio_proxy_to_baseline at D_ref from geometry profile",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate kappa_env via single-point anchor (no full-map fit).")
    ap.add_argument("--micro-csv", default="", help="Path to chi_open_system_micro_D*.csv (default: auto-pick largest).")
    ap.add_argument("--geom-csv", default="", help="Path to chi_open_system_geometry_D*.csv (default: auto-pick largest).")
    ap.add_argument("--D-ref", type=float, default=12.0)
    ap.add_argument("--kappa-min", type=float, default=1e4)
    ap.add_argument("--kappa-max", type=float, default=2e6)
    ap.add_argument("--n-coarse", type=int, default=13)
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--nstep", type=int, default=300)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)
    args = ap.parse_args()

    micro_csv = Path(args.micro_csv) if args.micro_csv else _pick_largest_csv("chi_open_system_micro_D*.csv")
    geom_csv = Path(args.geom_csv) if args.geom_csv else _pick_largest_csv("chi_open_system_geometry_D*.csv")

    inp = _build_inputs(micro_csv=micro_csv, geom_csv=geom_csv, d_ref=float(args.D_ref))
    out = calibrate_anchor_kappa(
        inp=inp,
        kappa_min=float(args.kappa_min),
        kappa_max=float(args.kappa_max),
        n_coarse=int(args.n_coarse),
        tmax=float(args.tmax),
        nstep=int(args.nstep),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / "kappa_env_anchor_calibration.csv"
    out_json = OUTDIR / "kappa_env_anchor_calibration.json"
    pd.DataFrame([out]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(out, indent=2))

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


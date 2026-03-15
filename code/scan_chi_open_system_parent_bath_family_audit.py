#!/usr/bin/env python3
"""
Natural-family audit for the projected parent bath block.

We test the smallest multiplicative deformation family around the canonical
projected parent bath statement

  gamma_phi = kappa_env * K_sys,phi * K_spec,phi
  gamma_mix = kappa_env * K_sys,mix * K_spec,mix

by allowing a shared normalization shift and uniform system/spectral warps:

  gamma_a^gen
    = kappa_env * exp(m)
      * K_sys,a * K_spec,a
      * (K_sys,a / K_sys,a^ref)^u
      * (K_spec,a / K_spec,a^ref)^v

for a in {phi, mix}.  The canonical parent family is the point (m,u,v)=(0,0,0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


def q(v: np.ndarray | pd.Series, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


def _parse_d_list(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


def _geom_mean(v: np.ndarray) -> float:
    vv = np.maximum(np.asarray(v, dtype=float), 1e-30)
    return float(np.exp(np.mean(np.log(vv))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Natural-family audit for projected parent bath block.")
    ap.add_argument(
        "--factor-map",
        type=Path,
        default=OUTDIR / "chi_open_system_bath_factorization_map.csv",
    )
    ap.add_argument(
        "--calib-csv",
        type=Path,
        default=OUTDIR / "kappa_env_anchor_calibration.csv",
    )
    ap.add_argument("--m-grid", default="-0.10,-0.05,0.0,0.05,0.10")
    ap.add_argument("--u-grid", default="-0.25,-0.125,0.0,0.125,0.25")
    ap.add_argument("--v-grid", default="-0.25,-0.125,0.0,0.125,0.25")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.factor_map.exists():
        raise FileNotFoundError(args.factor_map)
    if not args.calib_csv.exists():
        raise FileNotFoundError(args.calib_csv)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.factor_map).copy().sort_values("D").reset_index(drop=True)
    calib = pd.read_csv(args.calib_csv).iloc[0]
    if df.empty:
        raise RuntimeError(f"Empty factor map: {args.factor_map}")

    for col in [
        "D",
        "kappa_env",
        "bath_system_kernel_phi",
        "bath_system_kernel_mix",
        "bath_shape_phi",
        "bath_shape_mix",
        "gamma_phi_micro",
        "gamma_mix_micro",
        "bath_phi_identifiable",
    ]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    anchor_set = {round(v, 8) for v in _parse_d_list(str(calib["D_anchor_list"]))}
    if not anchor_set:
        raise RuntimeError("Calibration CSV has no anchor D list.")
    anchor_mask = np.asarray([round(float(d), 8) in anchor_set for d in df["D"]], dtype=bool)
    phi_identifiable = df["bath_phi_identifiable"].to_numpy(dtype=int) == 1
    if not np.any(anchor_mask):
        raise RuntimeError("No anchor points overlap factor map.")

    kappa = np.maximum(df["kappa_env"].to_numpy(dtype=float), 1e-30)
    sys_phi = np.maximum(df["bath_system_kernel_phi"].to_numpy(dtype=float), 1e-30)
    sys_mix = np.maximum(df["bath_system_kernel_mix"].to_numpy(dtype=float), 1e-30)
    spec_phi = np.maximum(df["bath_shape_phi"].to_numpy(dtype=float), 1e-30)
    spec_mix = np.maximum(df["bath_shape_mix"].to_numpy(dtype=float), 1e-30)
    gamma_phi = np.maximum(df["gamma_phi_micro"].to_numpy(dtype=float), 1e-30)
    gamma_mix = np.maximum(df["gamma_mix_micro"].to_numpy(dtype=float), 1e-30)

    sys_phi_ref = _geom_mean(sys_phi[anchor_mask])
    sys_mix_ref = _geom_mean(sys_mix[anchor_mask])
    spec_phi_ref = _geom_mean(spec_phi[anchor_mask])
    spec_mix_ref = _geom_mean(spec_mix[anchor_mask])

    base_phi = kappa * sys_phi * spec_phi
    base_mix = kappa * sys_mix * spec_mix

    m_grid = [float(x) for x in _parse_d_list(args.m_grid)]
    u_grid = [float(x) for x in _parse_d_list(args.u_grid)]
    v_grid = [float(x) for x in _parse_d_list(args.v_grid)]

    scan_rows: List[Dict[str, float]] = []
    best_row: Dict[str, float] | None = None
    best_maps: Dict[str, np.ndarray] | None = None

    for m in m_grid:
        for u in u_grid:
            for v in v_grid:
                warp_phi = np.exp(m) * np.power(sys_phi / sys_phi_ref, u) * np.power(spec_phi / spec_phi_ref, v)
                warp_mix = np.exp(m) * np.power(sys_mix / sys_mix_ref, u) * np.power(spec_mix / spec_mix_ref, v)
                gamma_phi_gen = base_phi * warp_phi
                gamma_mix_gen = base_mix * warp_mix

                trace_gen = gamma_phi_gen + gamma_mix_gen
                trace_obs = gamma_phi + gamma_mix
                det_gen = gamma_phi_gen * gamma_mix_gen
                det_obs = gamma_phi * gamma_mix
                log_rate_obs = np.log(gamma_mix / gamma_phi)
                log_rate_gen = np.log(gamma_mix_gen / gamma_phi_gen)

                block11_rel = np.abs(gamma_phi_gen / gamma_phi - 1.0)
                block22_rel = np.abs(gamma_mix_gen / gamma_mix - 1.0)
                trace_rel = np.abs(trace_gen / trace_obs - 1.0)
                det_rel = np.abs(det_gen / det_obs - 1.0)
                aniso_abs = np.abs(log_rate_gen - log_rate_obs)

                block11_ident = block11_rel[phi_identifiable] if np.any(phi_identifiable) else np.asarray([], dtype=float)
                det_ident = det_rel[phi_identifiable] if np.any(phi_identifiable) else np.asarray([], dtype=float)
                aniso_ident = aniso_abs[phi_identifiable] if np.any(phi_identifiable) else np.asarray([], dtype=float)

                row = {
                    "m": float(m),
                    "u": float(u),
                    "v": float(v),
                    "p95_block11_rel_residual_identifiable": q(block11_ident, 0.95) if block11_ident.size else float("nan"),
                    "p95_block22_rel_residual": q(block22_rel, 0.95),
                    "p95_trace_rel_residual": q(trace_rel, 0.95),
                    "p95_det_rel_residual_identifiable": q(det_ident, 0.95) if det_ident.size else float("nan"),
                    "p95_anisotropy_abs_residual_identifiable": q(aniso_ident, 0.95) if aniso_ident.size else float("nan"),
                    "max_block11_rel_residual_identifiable": float(np.max(block11_ident)) if block11_ident.size else float("nan"),
                    "max_block22_rel_residual": float(np.max(block22_rel)),
                    "max_trace_rel_residual": float(np.max(trace_rel)),
                    "max_det_rel_residual_identifiable": float(np.max(det_ident)) if det_ident.size else float("nan"),
                    "max_anisotropy_abs_residual_identifiable": float(np.max(aniso_ident)) if aniso_ident.size else float("nan"),
                }
                row["objective"] = (
                    row["p95_block11_rel_residual_identifiable"]
                    + row["p95_block22_rel_residual"]
                    + row["p95_trace_rel_residual"]
                    + row["p95_det_rel_residual_identifiable"]
                    + row["p95_anisotropy_abs_residual_identifiable"]
                )
                scan_rows.append(row)

                if best_row is None or row["objective"] < best_row["objective"]:
                    best_row = row
                    best_maps = {
                        "gamma_phi_gen": gamma_phi_gen,
                        "gamma_mix_gen": gamma_mix_gen,
                        "trace_gen": trace_gen,
                        "det_gen": det_gen,
                        "log_rate_gen": log_rate_gen,
                    }

    if best_row is None or best_maps is None:
        raise RuntimeError("Family scan produced no rows.")

    scan_df = pd.DataFrame(scan_rows).sort_values(["objective", "max_trace_rel_residual"]).reset_index(drop=True)

    canonical = scan_df[(scan_df["m"] == 0.0) & (scan_df["u"] == 0.0) & (scan_df["v"] == 0.0)]
    if canonical.empty:
        raise RuntimeError("Canonical point missing from scan.")
    canonical = canonical.iloc[0].to_dict()

    runner_up_df = scan_df[~((scan_df["m"] == 0.0) & (scan_df["u"] == 0.0) & (scan_df["v"] == 0.0))]
    runner_up = runner_up_df.iloc[0].to_dict() if not runner_up_df.empty else None

    summary = {
        "factor_map_csv": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "n_points": int(len(df)),
        "n_anchor": int(np.sum(anchor_mask)),
        "n_phi_identifiable": int(np.sum(phi_identifiable)),
        "best_m": float(best_row["m"]),
        "best_u": float(best_row["u"]),
        "best_v": float(best_row["v"]),
        "canonical_objective": float(canonical["objective"]),
        "canonical_p95_block11_rel_residual_identifiable": float(canonical["p95_block11_rel_residual_identifiable"]),
        "canonical_p95_block22_rel_residual": float(canonical["p95_block22_rel_residual"]),
        "canonical_p95_trace_rel_residual": float(canonical["p95_trace_rel_residual"]),
        "canonical_p95_det_rel_residual_identifiable": float(canonical["p95_det_rel_residual_identifiable"]),
        "canonical_p95_anisotropy_abs_residual_identifiable": float(
            canonical["p95_anisotropy_abs_residual_identifiable"]
        ),
        "canonical_max_block11_rel_residual_identifiable": float(canonical["max_block11_rel_residual_identifiable"]),
        "canonical_max_block22_rel_residual": float(canonical["max_block22_rel_residual"]),
        "canonical_max_trace_rel_residual": float(canonical["max_trace_rel_residual"]),
        "canonical_max_det_rel_residual_identifiable": float(canonical["max_det_rel_residual_identifiable"]),
        "canonical_max_anisotropy_abs_residual_identifiable": float(
            canonical["max_anisotropy_abs_residual_identifiable"]
        ),
    }
    if runner_up is not None:
        summary.update(
            {
                "runner_up_m": float(runner_up["m"]),
                "runner_up_u": float(runner_up["u"]),
                "runner_up_v": float(runner_up["v"]),
                "runner_up_objective": float(runner_up["objective"]),
                "runner_up_max_trace_rel_residual": float(runner_up["max_trace_rel_residual"]),
                "runner_up_max_det_rel_residual_identifiable": float(runner_up["max_det_rel_residual_identifiable"]),
                "runner_up_max_anisotropy_abs_residual_identifiable": float(
                    runner_up["max_anisotropy_abs_residual_identifiable"]
                ),
                "selection_gap_objective": float(runner_up["objective"] - canonical["objective"]),
            }
        )

    map_df = df[["D"]].copy()
    map_df["gamma_phi_obs"] = gamma_phi
    map_df["gamma_mix_obs"] = gamma_mix
    map_df["gamma_phi_best_family"] = best_maps["gamma_phi_gen"]
    map_df["gamma_mix_best_family"] = best_maps["gamma_mix_gen"]
    map_df["trace_obs"] = gamma_phi + gamma_mix
    map_df["trace_best_family"] = best_maps["trace_gen"]
    map_df["det_obs"] = gamma_phi * gamma_mix
    map_df["det_best_family"] = best_maps["det_gen"]
    map_df["log_rate_obs"] = np.log(gamma_mix / gamma_phi)
    map_df["log_rate_best_family"] = best_maps["log_rate_gen"]
    map_df["bath_phi_identifiable"] = phi_identifiable.astype(int)

    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""
    scan_csv = OUTDIR / f"chi_open_system_parent_bath_family_scan{suffix}.csv"
    summary_csv = OUTDIR / f"chi_open_system_parent_bath_family_summary{suffix}.csv"
    map_csv = OUTDIR / f"chi_open_system_parent_bath_family_map{suffix}.csv"
    png = OUTDIR / f"chi_open_system_parent_bath_family{suffix}.png"
    meta_json = OUTDIR / f"chi_open_system_parent_bath_family_run_meta{suffix}.json"

    scan_df.to_csv(scan_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    map_df.to_csv(map_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax0, ax1 = axes
    for m in sorted(scan_df["m"].unique()):
        sub = scan_df[(scan_df["m"] == m) & (scan_df["v"] == 0.0)].sort_values("u")
        ax0.plot(sub["u"], sub["objective"], marker="o", label=f"m={m:g}, v=0")
    ax0.set_title("Objective across system-warp slices")
    ax0.set_xlabel("u")
    ax0.set_ylabel("objective")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=8)

    for m in sorted(scan_df["m"].unique()):
        sub = scan_df[(scan_df["m"] == m) & (scan_df["u"] == 0.0)].sort_values("v")
        ax1.plot(sub["v"], sub["objective"], marker="o", label=f"m={m:g}, u=0")
    ax1.set_title("Objective across spectral-warp slices")
    ax1.set_xlabel("v")
    ax1.set_ylabel("objective")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)
    fig.suptitle("Open-system parent bath family audit", y=1.02)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    meta = {
        "factor_map_csv": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "statement": {
            "canonical_family": "gamma_a = kappa_env * K_sys,a * K_spec,a",
            "deformation": "exp(m) * (K_sys,a/K_sys,a^ref)^u * (K_spec,a/K_spec,a^ref)^v",
        },
        "grids": {"m": m_grid, "u": u_grid, "v": v_grid},
        "references": {
            "sys_phi_ref": sys_phi_ref,
            "sys_mix_ref": sys_mix_ref,
            "spec_phi_ref": spec_phi_ref,
            "spec_mix_ref": spec_mix_ref,
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    for src in [scan_csv, summary_csv, map_csv, png, meta_json]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[saved] {scan_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {map_csv}")
    print(f"[saved] {png}")
    print(f"[saved] {meta_json}")


if __name__ == "__main__":
    main()

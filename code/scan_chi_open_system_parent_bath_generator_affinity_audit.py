#!/usr/bin/env python3
"""
Generator-affinity audit for the projected parent bath action.

This script checks whether the projected parent bath action naturally uses
affine generators in the canonical log coordinates

  L_sys  = log(K_sys / K_sys^ref)
  L_spec = log(K_spec / K_spec^ref)

by adding the first non-affine quadratic generator terms:

  G_a = L_sys,a + L_spec,a
      + q_ss L_sys,a^2
      + q_bb L_spec,a^2
      + q_sb L_sys,a L_spec,a

and testing whether the canonical affine point
  (q_ss, q_bb, q_sb) = (0, 0, 0)
is uniquely selected.
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


def _parse_grid(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


def _geom_mean(v: np.ndarray) -> float:
    vv = np.maximum(np.asarray(v, dtype=float), 1e-300)
    return float(np.exp(np.mean(np.log(vv))))


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.asarray(x, dtype=float), -700.0, 700.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generator-affinity audit for projected parent bath action.")
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
    ap.add_argument("--q-ss-grid", default="-0.125,-0.0625,0.0,0.0625,0.125")
    ap.add_argument("--q-bb-grid", default="-0.125,-0.0625,0.0,0.0625,0.125")
    ap.add_argument("--q-sb-grid", default="-0.125,-0.0625,0.0,0.0625,0.125")
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

    required = [
        "D",
        "kappa_env",
        "bath_system_kernel_phi",
        "bath_system_kernel_mix",
        "bath_shape_phi",
        "bath_shape_mix",
        "gamma_phi_micro",
        "gamma_mix_micro",
        "bath_phi_identifiable",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    anchor_set = {round(float(v), 8) for v in str(calib["D_anchor_list"]).split(",") if str(v).strip()}
    if not anchor_set:
        raise RuntimeError("Calibration CSV has no anchor D list.")
    anchor_mask = np.asarray([round(float(d), 8) in anchor_set for d in df["D"]], dtype=bool)
    phi_identifiable = df["bath_phi_identifiable"].to_numpy(dtype=int) == 1

    kappa = np.maximum(df["kappa_env"].to_numpy(dtype=float), 1e-300)
    sys_phi = np.maximum(df["bath_system_kernel_phi"].to_numpy(dtype=float), 1e-300)
    sys_mix = np.maximum(df["bath_system_kernel_mix"].to_numpy(dtype=float), 1e-300)
    spec_phi = np.maximum(df["bath_shape_phi"].to_numpy(dtype=float), 1e-300)
    spec_mix = np.maximum(df["bath_shape_mix"].to_numpy(dtype=float), 1e-300)
    gamma_phi = np.maximum(df["gamma_phi_micro"].to_numpy(dtype=float), 1e-300)
    gamma_mix = np.maximum(df["gamma_mix_micro"].to_numpy(dtype=float), 1e-300)

    sys_phi_ref = _geom_mean(sys_phi[anchor_mask])
    sys_mix_ref = _geom_mean(sys_mix[anchor_mask])
    spec_phi_ref = _geom_mean(spec_phi[anchor_mask])
    spec_mix_ref = _geom_mean(spec_mix[anchor_mask])

    lsys_phi = np.log(sys_phi / sys_phi_ref)
    lsys_mix = np.log(sys_mix / sys_mix_ref)
    lspec_phi = np.log(spec_phi / spec_phi_ref)
    lspec_mix = np.log(spec_mix / spec_mix_ref)

    q_ss_grid = _parse_grid(args.q_ss_grid)
    q_bb_grid = _parse_grid(args.q_bb_grid)
    q_sb_grid = _parse_grid(args.q_sb_grid)

    scan_rows: List[Dict[str, float]] = []
    best_row: Dict[str, float] | None = None
    best_maps: Dict[str, np.ndarray] | None = None

    for q_ss in q_ss_grid:
        for q_bb in q_bb_grid:
            for q_sb in q_sb_grid:
                gen_phi = (
                    lsys_phi
                    + lspec_phi
                    + float(q_ss) * np.square(lsys_phi)
                    + float(q_bb) * np.square(lspec_phi)
                    + float(q_sb) * lsys_phi * lspec_phi
                )
                gen_mix = (
                    lsys_mix
                    + lspec_mix
                    + float(q_ss) * np.square(lsys_mix)
                    + float(q_bb) * np.square(lspec_mix)
                    + float(q_sb) * lsys_mix * lspec_mix
                )

                gamma_phi_gen = kappa * sys_phi_ref * spec_phi_ref * _safe_exp(gen_phi)
                gamma_mix_gen = kappa * sys_mix_ref * spec_mix_ref * _safe_exp(gen_mix)

                trace_gen = gamma_phi_gen + gamma_mix_gen
                trace_obs = gamma_phi + gamma_mix
                det_gen = gamma_phi_gen * gamma_mix_gen
                det_obs = gamma_phi * gamma_mix
                log_rate_gen = np.log(np.maximum(gamma_mix_gen, 1e-300) / np.maximum(gamma_phi_gen, 1e-300))
                log_rate_obs = np.log(gamma_mix / gamma_phi)

                block11_rel = np.abs(gamma_phi_gen / gamma_phi - 1.0)
                block22_rel = np.abs(gamma_mix_gen / gamma_mix - 1.0)
                trace_rel = np.abs(trace_gen / trace_obs - 1.0)
                det_rel = np.abs(det_gen / det_obs - 1.0)
                aniso_abs = np.abs(log_rate_gen - log_rate_obs)

                block11_ident = block11_rel[phi_identifiable] if np.any(phi_identifiable) else np.asarray([], dtype=float)
                det_ident = det_rel[phi_identifiable] if np.any(phi_identifiable) else np.asarray([], dtype=float)
                aniso_ident = aniso_abs[phi_identifiable] if np.any(phi_identifiable) else np.asarray([], dtype=float)

                row = {
                    "q_ss": float(q_ss),
                    "q_bb": float(q_bb),
                    "q_sb": float(q_sb),
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
        raise RuntimeError("No scan rows produced.")

    scan_df = pd.DataFrame(scan_rows).sort_values(["objective", "max_trace_rel_residual"]).reset_index(drop=True)
    canonical = scan_df[(scan_df["q_ss"] == 0.0) & (scan_df["q_bb"] == 0.0) & (scan_df["q_sb"] == 0.0)]
    if canonical.empty:
        raise RuntimeError("Canonical point (0,0,0) missing from scan.")
    canonical = canonical.iloc[0].to_dict()

    runner_up_df = scan_df[~((scan_df["q_ss"] == 0.0) & (scan_df["q_bb"] == 0.0) & (scan_df["q_sb"] == 0.0))]
    runner_up = runner_up_df.iloc[0].to_dict() if not runner_up_df.empty else None

    summary = {
        "factor_map_csv": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "n_points": int(len(df)),
        "n_anchor": int(np.sum(anchor_mask)),
        "n_phi_identifiable": int(np.sum(phi_identifiable)),
        "best_q_ss": float(best_row["q_ss"]),
        "best_q_bb": float(best_row["q_bb"]),
        "best_q_sb": float(best_row["q_sb"]),
        "canonical_objective": float(canonical["objective"]),
        "canonical_p95_block11_rel_residual_identifiable": float(canonical["p95_block11_rel_residual_identifiable"]),
        "canonical_p95_block22_rel_residual": float(canonical["p95_block22_rel_residual"]),
        "canonical_p95_trace_rel_residual": float(canonical["p95_trace_rel_residual"]),
        "canonical_p95_det_rel_residual_identifiable": float(canonical["p95_det_rel_residual_identifiable"]),
        "canonical_p95_anisotropy_abs_residual_identifiable": float(canonical["p95_anisotropy_abs_residual_identifiable"]),
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
                "runner_up_q_ss": float(runner_up["q_ss"]),
                "runner_up_q_bb": float(runner_up["q_bb"]),
                "runner_up_q_sb": float(runner_up["q_sb"]),
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
    map_df["gamma_phi_best_generator_family"] = best_maps["gamma_phi_gen"]
    map_df["gamma_mix_best_generator_family"] = best_maps["gamma_mix_gen"]
    map_df["trace_obs"] = gamma_phi + gamma_mix
    map_df["trace_best_generator_family"] = best_maps["trace_gen"]
    map_df["det_obs"] = gamma_phi * gamma_mix
    map_df["det_best_generator_family"] = best_maps["det_gen"]
    map_df["log_rate_obs"] = np.log(gamma_mix / gamma_phi)
    map_df["log_rate_best_generator_family"] = best_maps["log_rate_gen"]
    map_df["bath_phi_identifiable"] = phi_identifiable.astype(int)

    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""
    scan_csv = OUTDIR / f"chi_open_system_parent_bath_generator_affinity_scan{suffix}.csv"
    summary_csv = OUTDIR / f"chi_open_system_parent_bath_generator_affinity_summary{suffix}.csv"
    map_csv = OUTDIR / f"chi_open_system_parent_bath_generator_affinity_map{suffix}.csv"
    png = OUTDIR / f"chi_open_system_parent_bath_generator_affinity{suffix}.png"
    meta_json = OUTDIR / f"chi_open_system_parent_bath_generator_affinity_run_meta{suffix}.json"

    scan_df.to_csv(scan_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    map_df.to_csv(map_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax0, ax1 = axes
    for qss in sorted(scan_df["q_ss"].unique()):
        sub = scan_df[(scan_df["q_ss"] == qss) & (scan_df["q_sb"] == 0.0)].sort_values("q_bb")
        ax0.plot(sub["q_bb"], sub["objective"], marker="o", label=f"q_ss={qss:g}, q_sb=0")
    ax0.set_title("Objective across bath self-curvature")
    ax0.set_xlabel("q_bb")
    ax0.set_ylabel("objective")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=8)

    for qbb in sorted(scan_df["q_bb"].unique()):
        sub = scan_df[(scan_df["q_bb"] == qbb) & (scan_df["q_ss"] == 0.0)].sort_values("q_sb")
        ax1.plot(sub["q_sb"], sub["objective"], marker="o", label=f"q_bb={qbb:g}, q_ss=0")
    ax1.set_title("Objective across mixed curvature")
    ax1.set_xlabel("q_sb")
    ax1.set_ylabel("objective")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)
    fig.suptitle("Open-system parent bath generator-affinity audit", y=1.02)
    fig.savefig(png, dpi=200)
    plt.close(fig)

    meta = {
        "factor_map_csv": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "statement": {
            "canonical_generator": "G = L_sys + L_spec",
            "non_affine_family": "G -> G + q_ss L_sys^2 + q_bb L_spec^2 + q_sb L_sys L_spec",
        },
        "grids": {"q_ss": q_ss_grid, "q_bb": q_bb_grid, "q_sb": q_sb_grid},
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

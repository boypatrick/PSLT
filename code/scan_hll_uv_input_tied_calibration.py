#!/usr/bin/env python3
"""
Invariant-level calibration audit for the input-tied UV finite-match witness.

This script treats the canonical input-tied UV comparator as a structured
diagonal threshold witness and exports:

  1. A per-cell map joining the refreshed constant baseline and the canonical
     input-tied run.
  2. Summary statistics and simple correlations for the diagonal invariant
     I_diag = shell_spread * (1 + coeff_cv).
  3. A compact four-panel figure visualizing I_diag, kappa_diag_eff,
     relative match deformation, and map-level mu drift against the refreshed
     constant baseline.

The script is intentionally read-only with respect to the model definition:
it consumes existing canonical artifacts and produces calibration diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPER_DIR = ROOT / "paper"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calibrate the input-tied UV threshold witness")
    ap.add_argument(
        "--input-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_input_tied_D21E21.csv",
    )
    ap.add_argument(
        "--constant-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_constant_refresh_D21E21.csv",
    )
    ap.add_argument(
        "--scale-summary",
        type=Path,
        default=OUTDIR / "hll_uv_input_tied_vs_constant_refresh_scale_summary_D21E21.csv",
    )
    ap.add_argument("--tag", type=str, default="D21E21")
    ap.add_argument("--skip-paper-copy", action="store_true")
    return ap.parse_args()


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def plot_calibration(
    out_png: Path,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    i_diag: np.ndarray,
    kappa_diag_eff: np.ndarray,
    rel_match_shift: np.ndarray,
    abs_mu_drift: np.ndarray,
) -> None:
    extent = [float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    panels = [
        (axes[0, 0], i_diag, r"$I_{\rm diag}=\mathrm{shell\_spread}(1+\mathrm{coeff\_cv})$"),
        (axes[0, 1], kappa_diag_eff, r"$\kappa_{\rm diag}^{\rm eff}$"),
        (axes[1, 0], rel_match_shift, r"$|\Delta C_{\mu\mu}^{\rm match}|/|C_{\mu\mu}^{\rm tree}|$"),
        (axes[1, 1], abs_mu_drift, r"$|\Delta \mu_{\mu\mu}|$ vs refreshed constant"),
    ]
    for ax, arr, title in panels:
        vmin = float(np.percentile(arr, 5.0))
        vmax = float(np.percentile(arr, 95.0))
        if vmax <= vmin:
            vmin = float(np.min(arr))
            vmax = float(np.max(arr)) + 1e-12
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("D")
        ax.set_ylabel("eta")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("value")
    fig.suptitle("Input-tied UV threshold calibration", fontsize=13)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    inp = pd.read_csv(args.input_map)
    const = pd.read_csv(args.constant_map)
    scale = pd.read_csv(args.scale_summary)

    merged = inp.merge(
        const[["D", "eta", "mu_mumu_uv_rge", "C_uv_mumu"]],
        on=["D", "eta"],
        suffixes=("_input", "_const"),
    )
    merged["I_diag"] = merged["shell_spread"] * (1.0 + merged["coeff_cv"])
    merged["abs_mu_drift_vs_const"] = (merged["mu_mumu_uv_rge_input"] - merged["mu_mumu_uv_rge_const"]).abs()
    merged["rel_match_shift_diag"] = merged["abs_delta_C_match_mumu"] / merged["C_uv_mumu_input"].abs().clip(lower=1e-30)

    d_vals = np.sort(merged["D"].unique())
    eta_vals = np.sort(merged["eta"].unique())
    pivot = lambda col: merged.pivot(index="eta", columns="D", values=col).reindex(index=eta_vals, columns=d_vals).to_numpy()

    map_csv = OUTDIR / f"hll_uv_input_tied_invariant_calibration_map_{args.tag}.csv"
    summary_csv = OUTDIR / f"hll_uv_input_tied_invariant_calibration_summary_{args.tag}.csv"
    meta_json = OUTDIR / f"hll_uv_input_tied_invariant_calibration_run_meta_{args.tag}.json"
    fig_png = OUTDIR / f"hll_uv_input_tied_invariant_calibration_{args.tag}.png"

    merged.to_csv(map_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "n_points": int(len(merged)),
                "diag_scale_canonical": 1.0,
                "offdiag_scale_canonical": 0.0,
                "I_diag_p50": float(merged["I_diag"].quantile(0.5)),
                "I_diag_p90": float(merged["I_diag"].quantile(0.9)),
                "I_diag_max": float(merged["I_diag"].max()),
                "kappa_diag_eff_p50": float(merged["kappa_diag_eff"].quantile(0.5)),
                "kappa_diag_eff_p90": float(merged["kappa_diag_eff"].quantile(0.9)),
                "kappa_diag_eff_max": float(merged["kappa_diag_eff"].max()),
                "rel_match_shift_diag_p50": float(merged["rel_match_shift_diag"].quantile(0.5)),
                "rel_match_shift_diag_p90": float(merged["rel_match_shift_diag"].quantile(0.9)),
                "rel_match_shift_diag_max": float(merged["rel_match_shift_diag"].max()),
                "abs_mu_drift_vs_const_mean": float(merged["abs_mu_drift_vs_const"].mean()),
                "abs_mu_drift_vs_const_p95": float(merged["abs_mu_drift_vs_const"].quantile(0.95)),
                "abs_mu_drift_vs_const_max": float(merged["abs_mu_drift_vs_const"].max()),
                "corr_I_diag__kappa_diag_eff": corrcoef(
                    merged["I_diag"].to_numpy(),
                    merged["kappa_diag_eff"].to_numpy(),
                ),
                "corr_I_diag__rel_match_shift": corrcoef(
                    merged["I_diag"].to_numpy(),
                    merged["rel_match_shift_diag"].to_numpy(),
                ),
                "corr_kappa_diag_eff__abs_mu_drift": corrcoef(
                    merged["kappa_diag_eff"].to_numpy(),
                    merged["abs_mu_drift_vs_const"].to_numpy(),
                ),
                "offdiag_mix_max": float(merged["offdiag_mix"].max()),
                "kappa_offdiag_eff_max": float(merged["kappa_offdiag_eff"].max()),
                "scale_scan_best_small_deformation_diag": float(
                    scale.sort_values(["max_abs_mapdiff", "p95_abs_mapdiff"]).iloc[0]["kappa_diag_eff_p50"]
                ),
            }
        ]
    )
    summary.to_csv(summary_csv, index=False)

    plot_calibration(
        fig_png,
        d_vals=d_vals,
        eta_vals=eta_vals,
        i_diag=pivot("I_diag"),
        kappa_diag_eff=pivot("kappa_diag_eff"),
        rel_match_shift=pivot("rel_match_shift_diag"),
        abs_mu_drift=pivot("abs_mu_drift_vs_const"),
    )

    meta = {
        "input_map": str(args.input_map),
        "constant_map": str(args.constant_map),
        "scale_summary": str(args.scale_summary),
        "tag": str(args.tag),
        "n_points": int(len(merged)),
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    if not args.skip_paper_copy:
        for src in [map_csv, summary_csv, meta_json, fig_png]:
            (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary.to_string(index=False))
    print(f"[saved] {map_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {fig_png}")
    print(f"[saved] {meta_json}")


if __name__ == "__main__":
    main()

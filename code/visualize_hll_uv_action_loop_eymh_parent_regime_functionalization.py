#!/usr/bin/env python3
"""
Visualize low-D / high-D regime functionalization for EYMH Lemma 2 witnesses.

This figure is diagnostic rather than a promotion audit.  It shows how the
currently best low-D and high-D witnesses differ, where the unresolved middle-D
transition band sits, and how each regime tracks the canonical target xi and
quartic curvature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPERDIR = ROOT / "paper"

LOW_CUT = 9.6
HIGH_SPLIT = 16.8
LOW_COEFF = 0.231819
HIGH_COEFF = 0.857050


def q(v: np.ndarray | pd.Series, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize EYMH low/high-D regime functionalization.")
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    args = ap.parse_args()

    df = pd.read_csv(args.parented_map)

    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    k_part = coeff_l1 / coeff_l2
    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    g_schur = 1.0 + chi_tree
    a_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )
    xi_target = np.maximum((k_part - 1.0) * (g_schur - 1.0), 0.0) / np.maximum(k_part * g_schur, 1e-30)

    D = np.asarray(df["D"], dtype=float)
    diag = np.maximum(a_ref, 1e-30)
    hk_abs_over_diag = np.maximum(np.asarray(df["hk_abs_offdiag"], dtype=float), 0.0) / diag
    action_abs_over_diag = np.maximum(np.asarray(df["action_abs_offdiag"], dtype=float), 0.0) / diag

    xi_low = np.maximum(LOW_COEFF * hk_abs_over_diag, 0.0)
    xi_high = np.maximum(HIGH_COEFF * action_abs_over_diag, 0.0)

    quartic_target = 0.25 * (a_ref**2) * (xi_target**2)
    quartic_low = 0.25 * (a_ref**2) * (xi_low**2)
    quartic_high = 0.25 * (a_ref**2) * (xi_high**2)

    profile_rows = []
    for dval in sorted(df["D"].unique()):
        mask = np.asarray(df["D"] == dval)
        profile_rows.append(
            {
                "D": float(dval),
                "region": (
                    "low" if float(dval) <= LOW_CUT else ("high" if float(dval) > HIGH_SPLIT else "transition")
                ),
                "xi_target_med": float(np.median(xi_target[mask])),
                "xi_low_med": float(np.median(xi_low[mask])),
                "xi_high_med": float(np.median(xi_high[mask])),
                "quartic_target_med": float(np.median(quartic_target[mask])),
                "quartic_low_med": float(np.median(quartic_low[mask])),
                "quartic_high_med": float(np.median(quartic_high[mask])),
                "slice_p95_xi_resid_low": q(np.abs(xi_low[mask] - xi_target[mask]), 0.95),
                "slice_p95_xi_resid_high": q(np.abs(xi_high[mask] - xi_target[mask]), 0.95),
                "slice_p95_quartic_resid_low": q(np.abs(quartic_low[mask] - quartic_target[mask]), 0.95),
                "slice_p95_quartic_resid_high": q(np.abs(quartic_high[mask] - quartic_target[mask]), 0.95),
            }
        )
    profiles = pd.DataFrame(profile_rows)

    summary = pd.DataFrame(
        [
            {
                "regime": "low",
                "D_condition": f"D <= {LOW_CUT}",
                "functionalization": "xi_low = 0.231819 * hk_abs_offdiag / diag",
                "corr_xi": float(np.corrcoef(xi_low[D <= LOW_CUT], xi_target[D <= LOW_CUT])[0, 1]),
                "corr_quartic": float(np.corrcoef(quartic_low[D <= LOW_CUT], quartic_target[D <= LOW_CUT])[0, 1]),
                "p95_abs_xi_resid": q(np.abs(xi_low[D <= LOW_CUT] - xi_target[D <= LOW_CUT]), 0.95),
                "p95_abs_quartic_resid": q(np.abs(quartic_low[D <= LOW_CUT] - quartic_target[D <= LOW_CUT]), 0.95),
            },
            {
                "regime": "high",
                "D_condition": f"D > {HIGH_SPLIT}",
                "functionalization": "xi_high = 0.857050 * action_abs_offdiag / diag",
                "corr_xi": float(np.corrcoef(xi_high[D > HIGH_SPLIT], xi_target[D > HIGH_SPLIT])[0, 1]),
                "corr_quartic": float(np.corrcoef(quartic_high[D > HIGH_SPLIT], quartic_target[D > HIGH_SPLIT])[0, 1]),
                "p95_abs_xi_resid": q(np.abs(xi_high[D > HIGH_SPLIT] - xi_target[D > HIGH_SPLIT]), 0.95),
                "p95_abs_quartic_resid": q(np.abs(quartic_high[D > HIGH_SPLIT] - quartic_target[D > HIGH_SPLIT]), 0.95),
            },
            {
                "regime": "transition",
                "D_condition": f"{LOW_CUT} < D <= {HIGH_SPLIT}",
                "functionalization": "No stable single-regime witness yet",
                "corr_xi": np.nan,
                "corr_quartic": np.nan,
                "p95_abs_xi_resid": np.nan,
                "p95_abs_quartic_resid": np.nan,
            },
        ]
    )

    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_regime_functionalization_summary_{args.tag}.csv"
    out_profiles = OUTDIR / f"hll_uv_action_loop_eymh_parent_regime_functionalization_profiles_{args.tag}.csv"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_regime_functionalization_run_meta_{args.tag}.json"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_regime_functionalization_{args.tag}.png"

    summary.to_csv(out_summary, index=False)
    profiles.to_csv(out_profiles, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "low_cut_D": LOW_CUT,
                "high_split_D": HIGH_SPLIT,
                "low_functionalization": "0.231819 * hk_abs_offdiag / diag",
                "high_functionalization": "0.857050 * action_abs_offdiag / diag",
                "note": "Low-D and high-D regime visualization built from current best regime-specific curvature witnesses.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    prof = profiles.sort_values("D")
    dvals = prof["D"].to_numpy()
    for ax in axes.ravel():
        ax.axvspan(dvals.min(), LOW_CUT, color="#f6f0c4", alpha=0.55)
        ax.axvspan(LOW_CUT, HIGH_SPLIT, color="#e8edf5", alpha=0.55)
        ax.axvspan(HIGH_SPLIT, dvals.max(), color="#f9dfd4", alpha=0.55)
        ax.axvline(LOW_CUT, color="#6b7280", linestyle="--", linewidth=1)
        ax.axvline(HIGH_SPLIT, color="#6b7280", linestyle="--", linewidth=1)

    axes[0, 0].plot(dvals, prof["xi_target_med"], marker="o", label="target $\\xi$")
    axes[0, 0].plot(dvals, prof["xi_low_med"], marker="s", label="low-D functionalization")
    axes[0, 0].plot(dvals, prof["xi_high_med"], marker="^", label="high-D functionalization")
    axes[0, 0].set_title("Target vs regime functionalizations on $\\xi(D)$")
    axes[0, 0].set_xlabel("D")
    axes[0, 0].set_ylabel("median $\\xi$")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(dvals, prof["slice_p95_xi_resid_low"], marker="s", label="low-D witness residual")
    axes[0, 1].plot(dvals, prof["slice_p95_xi_resid_high"], marker="^", label="high-D witness residual")
    axes[0, 1].set_title("Per-D $\\xi$ residual by regime witness")
    axes[0, 1].set_xlabel("D")
    axes[0, 1].set_ylabel("slice p95 abs residual")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(dvals, prof["quartic_target_med"], marker="o", label="target quartic")
    axes[1, 0].plot(dvals, prof["quartic_low_med"], marker="s", label="low-D functionalization")
    axes[1, 0].plot(dvals, prof["quartic_high_med"], marker="^", label="high-D functionalization")
    axes[1, 0].set_title("Target vs regime functionalizations on quartic witness")
    axes[1, 0].set_xlabel("D")
    axes[1, 0].set_ylabel("median quartic")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].scatter(xi_target[D <= LOW_CUT], xi_low[D <= LOW_CUT], s=18, alpha=0.7, label="low-D points")
    axes[1, 1].scatter(xi_target[D > HIGH_SPLIT], xi_high[D > HIGH_SPLIT], s=18, alpha=0.7, label="high-D points")
    lo = min(float(xi_target.min()), float(xi_low[D <= LOW_CUT].min()), float(xi_high[D > HIGH_SPLIT].min()))
    hi = max(float(xi_target.max()), float(xi_low[D <= LOW_CUT].max()), float(xi_high[D > HIGH_SPLIT].max()))
    axes[1, 1].plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    axes[1, 1].set_title("Regime-specific witness vs target $\\xi$")
    axes[1, 1].set_xlabel("target $\\xi$")
    axes[1, 1].set_ylabel("functionalized witness")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("EYMH low-D / high-D regime functionalization", fontsize=14)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_summary, out_profiles, out_meta, out_fig):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

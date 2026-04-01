#!/usr/bin/env python3
"""
Visualize the dense-D EYMH parented target map.

Builds xi_target and quartic target from a denser D scan of the
action_loop_eymh_parented UV-to-EFT map, then renders a heatmap and several
eta-slice profiles to inspect D-structure directly.
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


ETA_SLICE_TARGETS = [0.2, 1.15, 2.1, 3.05, 4.0]
REGIME_MARKS = [9.6, 16.8]


def nearest_available(values: np.ndarray, target: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(target)))])


def pivot_metric(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = df.pivot(index="eta", columns="D", values=col).sort_index().sort_index(axis=1)
    return p.columns.to_numpy(float), p.index.to_numpy(float), p.to_numpy(float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize dense-D EYMH parented target structure.")
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_uv_action_loop_eymh_parented_D81E21_targetdense.csv",
    )
    ap.add_argument("--tag", default="D81E21_targetdense")
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
    quartic_target = 0.25 * (a_ref**2) * (xi_target**2)

    plot_df = df[["D", "eta"]].copy()
    plot_df["xi_target"] = xi_target
    plot_df["quartic_target"] = quartic_target

    dvals, etavals, xi_grid = pivot_metric(plot_df, "xi_target")
    _, _, quartic_grid = pivot_metric(plot_df, "quartic_target")

    profile_rows = []
    chosen_etas = [nearest_available(etavals, target) for target in ETA_SLICE_TARGETS]
    chosen_etas = list(dict.fromkeys(chosen_etas))
    for eta in chosen_etas:
        sub = plot_df[np.isclose(plot_df["eta"], eta)].sort_values("D")
        for row in sub.itertuples():
            profile_rows.append(
                {
                    "eta": float(eta),
                    "D": float(row.D),
                    "xi_target": float(row.xi_target),
                    "quartic_target": float(row.quartic_target),
                }
            )
    profiles = pd.DataFrame(profile_rows)

    summary = pd.DataFrame(
        [
            {
                "n_points": int(len(plot_df)),
                "n_D": int(len(dvals)),
                "n_eta": int(len(etavals)),
                "D_min": float(dvals.min()),
                "D_max": float(dvals.max()),
                "eta_min": float(etavals.min()),
                "eta_max": float(etavals.max()),
                "xi_target_max": float(np.max(xi_grid)),
                "quartic_target_max": float(np.max(quartic_grid)),
                "selected_eta_slices": ", ".join(f"{x:.2f}" for x in chosen_etas),
            }
        ]
    )

    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_dense_target_summary_{args.tag}.csv"
    out_profiles = OUTDIR / f"hll_uv_action_loop_eymh_parent_dense_target_profiles_{args.tag}.csv"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_dense_target_run_meta_{args.tag}.json"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_dense_target_{args.tag}.png"

    summary.to_csv(out_summary, index=False)
    profiles.to_csv(out_profiles, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "selected_eta_slices": chosen_etas,
                "regime_marks": REGIME_MARKS,
                "note": "Dense-D EYMH parented target visualization for xi_target and quartic target.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    im0 = axes[0, 0].imshow(
        xi_grid,
        aspect="auto",
        origin="lower",
        extent=[float(dvals.min()), float(dvals.max()), float(etavals.min()), float(etavals.max())],
        cmap="viridis",
    )
    axes[0, 0].set_title(r"Dense target heatmap: $\xi_{\rm target}(D,\eta)$")
    axes[0, 0].set_xlabel("D")
    axes[0, 0].set_ylabel(r"$\eta$")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.85)
    for mark in REGIME_MARKS:
        axes[0, 0].axvline(mark, color="white", linestyle="--", linewidth=1)

    im1 = axes[0, 1].imshow(
        quartic_grid,
        aspect="auto",
        origin="lower",
        extent=[float(dvals.min()), float(dvals.max()), float(etavals.min()), float(etavals.max())],
        cmap="magma",
    )
    axes[0, 1].set_title(r"Dense target heatmap: quartic target")
    axes[0, 1].set_xlabel("D")
    axes[0, 1].set_ylabel(r"$\eta$")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.85)
    for mark in REGIME_MARKS:
        axes[0, 1].axvline(mark, color="white", linestyle="--", linewidth=1)

    for eta in chosen_etas:
        sub = profiles[np.isclose(profiles["eta"], eta)]
        axes[1, 0].plot(sub["D"], sub["xi_target"], marker="o", linewidth=1.2, markersize=3, label=fr"$\eta={eta:.2f}$")
        axes[1, 1].plot(sub["D"], sub["quartic_target"], marker="o", linewidth=1.2, markersize=3, label=fr"$\eta={eta:.2f}$")

    axes[1, 0].set_title(r"Dense target slices: $\xi_{\rm target}(D)$")
    axes[1, 0].set_xlabel("D")
    axes[1, 0].set_ylabel(r"$\xi_{\rm target}$")
    axes[1, 1].set_title("Dense target slices: quartic target")
    axes[1, 1].set_xlabel("D")
    axes[1, 1].set_ylabel("quartic target")
    for ax in (axes[1, 0], axes[1, 1]):
        for mark in REGIME_MARKS:
            ax.axvline(mark, color="#6b7280", linestyle="--", linewidth=1)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle("EYMH dense-D target scan", fontsize=14)
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

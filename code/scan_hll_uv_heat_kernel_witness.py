#!/usr/bin/env python3
"""
Audit the conformal-background heat-kernel witness used for next-step
UV-to-EFT loop normalization.

This script samples the current H->ll UV matching witness on a scan grid and
records the heat-kernel quantities now threaded through the finite-match API.
At this phase the witness is diagnostic only: it checks that the new loop-side
invariants are numerically stable, eta-independent at fixed D, and suitable for
use as a parent-action-side normalization source in the next matching mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from scan_hll_uv_to_eft_matching import BASELINE, make_baseline_kinetics  # noqa: E402


OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPER_DIR = ROOT / "paper"
def sanitize_tag(tag: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(tag).strip())


def make_suffix(tag: str) -> str:
    clean = sanitize_tag(tag)
    return f"_{clean}" if clean else ""

def main() -> None:
    ap = argparse.ArgumentParser(description="Audit heat-kernel loop witness on the UV matching grid")
    ap.add_argument("--d-min", type=float, default=float(BASELINE["D_min"]))
    ap.add_argument("--d-max", type=float, default=float(BASELINE["D_max"]))
    ap.add_argument("--d-num", type=int, default=int(BASELINE["D_num"]))
    ap.add_argument("--eta-min", type=float, default=float(BASELINE["eta_min"]))
    ap.add_argument("--eta-max", type=float, default=float(BASELINE["eta_max"]))
    ap.add_argument("--eta-num", type=int, default=int(BASELINE["eta_num"]))
    ap.add_argument("--tag", type=str, default="uv_heat_kernel_witness_D21E21")
    ap.add_argument("--skip-paper-copy", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    kin = make_baseline_kinetics(
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        chain_mode="full_direct",
        uv_blend=float(BASELINE["hll_uv_blend"]),
        uv_m2_power=float(BASELINE["hll_uv_m2_power"]),
        uv_match_kappa_diag=float(BASELINE["hll_uv_match_kappa_diag"]),
        uv_match_kappa_offdiag=float(BASELINE["hll_uv_match_kappa_offdiag"]),
        uv_match_mode="action_absolute",
        uv_match_input_diag_scale=0.0,
        uv_match_input_offdiag_scale=0.0,
        uv_rge_mu_low=float(BASELINE["hll_uv_rge_mu_low"]),
        uv_rge_gamma_diag=float(BASELINE["hll_uv_rge_gamma_diag"]),
        uv_rge_gamma_offdiag=float(BASELINE["hll_uv_rge_gamma_offdiag"]),
        uv_rge_log_clip=float(BASELINE["hll_uv_rge_log_clip"]),
    )
    d_vals = np.linspace(args.d_min, args.d_max, args.d_num)
    eta_vals = np.linspace(args.eta_min, args.eta_max, args.eta_num)

    rows: list[dict[str, float]] = []
    for D in d_vals:
        for eta in eta_vals:
            wit = kin.hll_uv_matching_witness(D=float(D), eta=float(eta), t_coh=float(BASELINE["t_coh"]), N_max=int(BASELINE["hll_observable_nmax"]))
            rows.append(
                {
                    "D": float(D),
                    "eta": float(eta),
                    "hk_omega_mid": float(np.asarray(wit["hk_omega_mid"], dtype=float)[0]),
                    "hk_R_mid": float(np.asarray(wit["hk_R_mid"], dtype=float)[0]),
                    "hk_X_mid": float(np.asarray(wit["hk_X_mid"], dtype=float)[0]),
                    "hk_a1_well": float(np.asarray(wit["hk_a1_well"], dtype=float)[0]),
                    "hk_a2_well": float(np.asarray(wit["hk_a2_well"], dtype=float)[0]),
                    "hk_a2_barrier": float(np.asarray(wit["hk_a2_barrier"], dtype=float)[0]),
                    "hk_diag_density": float(np.asarray(wit["hk_diag_density"], dtype=float)[0]),
                    "hk_barrier_density": float(np.asarray(wit["hk_barrier_density"], dtype=float)[0]),
                    "hk_abs_diag": float(np.asarray(wit["hk_abs_diag"], dtype=float)[0]),
                    "hk_abs_offdiag": float(np.asarray(wit["hk_abs_offdiag"], dtype=float)[0]),
                    "hk_barrier_ratio": float(np.asarray(wit["hk_barrier_ratio"], dtype=float)[0]),
                    "hk_a1_flat": float(np.asarray(wit["hk_a1_flat"], dtype=float)[0]),
                    "hk_a2_flat": float(np.asarray(wit["hk_a2_flat"], dtype=float)[0]),
                    "hk_a1_well_geom": float(np.asarray(wit["hk_a1_well_geom"], dtype=float)[0]),
                    "hk_a2_well_geom": float(np.asarray(wit["hk_a2_well_geom"], dtype=float)[0]),
                    "hk_a2_barrier_geom": float(np.asarray(wit["hk_a2_barrier_geom"], dtype=float)[0]),
                    "hk_diag_density_geom": float(np.asarray(wit["hk_diag_density_geom"], dtype=float)[0]),
                    "hk_barrier_density_geom": float(np.asarray(wit["hk_barrier_density_geom"], dtype=float)[0]),
                    "hk_abs_diag_geom": float(np.asarray(wit["hk_abs_diag_geom"], dtype=float)[0]),
                    "hk_abs_offdiag_geom": float(np.asarray(wit["hk_abs_offdiag_geom"], dtype=float)[0]),
                    "hk_barrier_ratio_geom": float(np.asarray(wit["hk_barrier_ratio_geom"], dtype=float)[0]),
                    "hk_R_well_abs": float(np.asarray(wit["hk_R_well_abs"], dtype=float)[0]),
                    "hk_R_barrier_abs": float(np.asarray(wit["hk_R_barrier_abs"], dtype=float)[0]),
                    "hk_X_well_abs": float(np.asarray(wit["hk_X_well_abs"], dtype=float)[0]),
                    "hk_X_barrier_abs": float(np.asarray(wit["hk_X_barrier_abs"], dtype=float)[0]),
                    "hk_gradX_barrier": float(np.asarray(wit["hk_gradX_barrier"], dtype=float)[0]),
                    "hk_curv_contrast_log": float(np.asarray(wit["hk_curv_contrast_log"], dtype=float)[0]),
                    "hk_curv_access": float(np.asarray(wit["hk_curv_access"], dtype=float)[0]),
                    "hk_barrier_stiffness_log": float(np.asarray(wit["hk_barrier_stiffness_log"], dtype=float)[0]),
                    "action_abs_diag": float(np.asarray(wit["action_abs_diag"], dtype=float)[0]),
                }
            )

    suffix = make_suffix(args.tag)
    out_map = OUTDIR / f"hll_uv_heat_kernel_witness_map{suffix}.csv"
    out_summary = OUTDIR / f"hll_uv_heat_kernel_witness_summary{suffix}.csv"
    out_fig = OUTDIR / f"hll_uv_heat_kernel_witness{suffix}.png"
    out_meta = OUTDIR / f"hll_uv_heat_kernel_witness_run_meta{suffix}.json"

    with out_map.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    arr = np.array(rows, dtype=object)
    map_dtype = [(k, float) for k in rows[0].keys()]
    map_arr = np.array([tuple(r[k] for k in rows[0].keys()) for r in rows], dtype=map_dtype)
    eta_spread = []
    for D in d_vals:
        mask = np.isclose(map_arr["D"], D)
        eta_spread.append(float(np.max(map_arr["hk_diag_density"][mask]) - np.min(map_arr["hk_diag_density"][mask])))

    summary = {
        "n_points": float(len(rows)),
        "hk_abs_diag_p50": float(np.percentile(map_arr["hk_abs_diag"], 50.0)),
        "hk_abs_diag_p90": float(np.percentile(map_arr["hk_abs_diag"], 90.0)),
        "hk_abs_diag_max": float(np.max(map_arr["hk_abs_diag"])),
        "hk_abs_diag_geom_p50": float(np.percentile(map_arr["hk_abs_diag_geom"], 50.0)),
        "hk_abs_diag_geom_p90": float(np.percentile(map_arr["hk_abs_diag_geom"], 90.0)),
        "hk_abs_diag_geom_max": float(np.max(map_arr["hk_abs_diag_geom"])),
        "hk_curv_access_p50": float(np.percentile(map_arr["hk_curv_access"], 50.0)),
        "hk_curv_access_p90": float(np.percentile(map_arr["hk_curv_access"], 90.0)),
        "hk_curv_access_max": float(np.max(map_arr["hk_curv_access"])),
        "hk_curv_contrast_log_p50": float(np.percentile(map_arr["hk_curv_contrast_log"], 50.0)),
        "hk_curv_contrast_log_p90": float(np.percentile(map_arr["hk_curv_contrast_log"], 90.0)),
        "hk_curv_contrast_log_max": float(np.max(map_arr["hk_curv_contrast_log"])),
        "hk_barrier_stiffness_log_p50": float(np.percentile(map_arr["hk_barrier_stiffness_log"], 50.0)),
        "hk_barrier_stiffness_log_p90": float(np.percentile(map_arr["hk_barrier_stiffness_log"], 90.0)),
        "hk_barrier_stiffness_log_max": float(np.max(map_arr["hk_barrier_stiffness_log"])),
        "hk_barrier_ratio_p50": float(np.percentile(map_arr["hk_barrier_ratio"], 50.0)),
        "hk_barrier_ratio_p90": float(np.percentile(map_arr["hk_barrier_ratio"], 90.0)),
        "hk_barrier_ratio_max": float(np.max(map_arr["hk_barrier_ratio"])),
        "hk_barrier_ratio_geom_p50": float(np.percentile(map_arr["hk_barrier_ratio_geom"], 50.0)),
        "hk_barrier_ratio_geom_p90": float(np.percentile(map_arr["hk_barrier_ratio_geom"], 90.0)),
        "hk_barrier_ratio_geom_max": float(np.max(map_arr["hk_barrier_ratio_geom"])),
        "max_eta_spread_hk_diag_density": float(max(eta_spread)),
        "max_eta_spread_hk_diag_density_geom": float(
            max(
                float(np.max(map_arr["hk_diag_density_geom"][np.isclose(map_arr["D"], D)]) - np.min(map_arr["hk_diag_density_geom"][np.isclose(map_arr["D"], D)]))
                for D in d_vals
            )
        ),
        "corr_hk_abs_diag_action_abs_diag": float(np.corrcoef(map_arr["hk_abs_diag"], map_arr["action_abs_diag"])[0, 1]),
        "corr_hk_abs_diag_geom_action_abs_diag": float(np.corrcoef(map_arr["hk_abs_diag_geom"], map_arr["action_abs_diag"])[0, 1]),
        "corr_hk_curv_access_action_abs_diag": float(np.corrcoef(map_arr["hk_curv_access"], map_arr["action_abs_diag"])[0, 1]),
        "corr_hk_curv_contrast_log_action_abs_diag": float(np.corrcoef(map_arr["hk_curv_contrast_log"], map_arr["action_abs_diag"])[0, 1]),
        "corr_hk_barrier_stiffness_log_action_abs_diag": float(np.corrcoef(map_arr["hk_barrier_stiffness_log"], map_arr["action_abs_diag"])[0, 1]),
    }
    with out_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    uniq_d = np.asarray(d_vals, dtype=float)
    hk_diag_d = np.array([map_arr["hk_abs_diag"][np.isclose(map_arr["D"], D)][0] for D in uniq_d], dtype=float)
    hk_barrier_d = np.array([map_arr["hk_barrier_ratio"][np.isclose(map_arr["D"], D)][0] for D in uniq_d], dtype=float)
    hk_diag_geom_d = np.array([map_arr["hk_abs_diag_geom"][np.isclose(map_arr["D"], D)][0] for D in uniq_d], dtype=float)
    hk_curv_access_d = np.array([map_arr["hk_curv_access"][np.isclose(map_arr["D"], D)][0] for D in uniq_d], dtype=float)
    action_abs_d = np.array([map_arr["action_abs_diag"][np.isclose(map_arr["D"], D)][0] for D in uniq_d], dtype=float)

    fig, axes = plt.subplots(1, 5, figsize=(20.0, 3.8), constrained_layout=True)
    axes[0].plot(uniq_d, hk_diag_d, marker="o")
    axes[0].set_title("hk_abs_diag(D)")
    axes[0].set_xlabel("D")
    axes[1].plot(uniq_d, hk_barrier_d, marker="o")
    axes[1].set_title("hk_barrier_ratio(D)")
    axes[1].set_xlabel("D")
    axes[2].plot(uniq_d, hk_diag_geom_d, marker="o")
    axes[2].set_title("hk_abs_diag_geom(D)")
    axes[2].set_xlabel("D")
    axes[3].plot(uniq_d, hk_curv_access_d, marker="o")
    axes[3].set_title("hk_curv_access(D)")
    axes[3].set_xlabel("D")
    axes[4].plot(uniq_d, action_abs_d, marker="o")
    axes[4].set_title("action_abs_diag(D)")
    axes[4].set_xlabel("D")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Conformal heat-kernel witness on UV matching grid", fontsize=13)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    out_meta.write_text(
        json.dumps(
            {
                "d_min": float(args.d_min),
                "d_max": float(args.d_max),
                "d_num": int(args.d_num),
                "eta_min": float(args.eta_min),
                "eta_max": float(args.eta_max),
                "eta_num": int(args.eta_num),
                "tag": args.tag,
                "baseline_match_mode": str(BASELINE["hll_uv_match_mode"]),
            },
            indent=2,
        )
    )

    if not args.skip_paper_copy:
        for src in [out_map, out_summary, out_fig, out_meta]:
            (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

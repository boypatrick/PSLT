#!/usr/bin/env python3
"""
Operator-basis audit for the PSLT UV->EFT H->ll closure.

This script exposes the current UV-tree -> finite one-loop -> LL-RG chain as an
explicit layer-resolved operator basis:

  C_tree = sum_N c_N * B_N
         = sum_N [P_N^(kin)/M_N^2] * (g_N g_N^T),

then verifies that:
  - the basis reconstruction exactly rebuilds C_tree,
  - the finite one-loop and LL-RG blockwise updates rebuild C_match / C_IR,
  - the reconstructed C_IR reproduces mu_mumu from the native observable path.

Outputs:
  - output/hll_uv_matching/hll_uv_operator_basis_map*.csv
  - output/hll_uv_matching/hll_uv_operator_basis_summary*.csv
  - output/hll_uv_matching/hll_uv_operator_basis_maps*.png
  - output/hll_uv_matching/hll_uv_operator_basis_run_meta*.json
and copies them to paper/ unless --skip-paper-copy is set.
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

from eft_wilson_matching import reconstruct_from_layer_basis, total_width_ratio  # noqa: E402
from scan_hll_uv_to_eft_matching import BASELINE, make_baseline_kinetics, make_suffix, snap_ref_d_for_full_direct  # noqa: E402


OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPER_DIR = ROOT / "paper"


def write_map_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, row: dict[str, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def plot_maps(
    out_png: Path,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    c_tree_mumu: np.ndarray,
    delta_match_mumu: np.ndarray,
    delta_rge_mumu: np.ndarray,
    mu_recon_resid: np.ndarray,
) -> None:
    extent = [float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    panels = [
        (axes[0, 0], c_tree_mumu, r"$C_{\mu\mu}^{\rm tree}$"),
        (axes[0, 1], np.abs(delta_match_mumu), r"$|\Delta C_{\mu\mu}^{\rm match}|$"),
        (axes[1, 0], np.abs(delta_rge_mumu), r"$|\Delta C_{\mu\mu}^{\rm RGE}|$"),
        (axes[1, 1], mu_recon_resid, r"$|\Delta \mu_{\mu\mu}^{\rm recon}|$"),
    ]

    for ax, arr, title in panels:
        p05, p95 = np.percentile(arr, [5.0, 95.0])
        if p95 <= p05:
            p05 = float(np.min(arr))
            p95 = float(np.max(arr)) + 1e-12
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=float(p05),
            vmax=float(p95),
        )
        ax.set_title(title)
        ax.set_xlabel("D")
        ax.set_ylabel("eta")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("value")

    fig.suptitle("UV operator-basis audit (tree -> finite match -> LL-RG)", fontsize=13)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Operator-basis audit for PSLT UV->EFT matching")
    ap.add_argument(
        "--chain-mode",
        choices=[
            "auto",
            "full_direct",
            "full_direct_runtime",
            "cell_direct_runtime",
            "cell_direct_runtime_release",
            "cell_direct_runtime_extreme",
        ],
        default="full_direct",
    )
    ap.add_argument("--d-min", type=float, default=float(BASELINE["D_min"]))
    ap.add_argument("--d-max", type=float, default=float(BASELINE["D_max"]))
    ap.add_argument("--d-num", type=int, default=int(BASELINE["D_num"]))
    ap.add_argument("--eta-min", type=float, default=float(BASELINE["eta_min"]))
    ap.add_argument("--eta-max", type=float, default=float(BASELINE["eta_max"]))
    ap.add_argument("--eta-num", type=int, default=int(BASELINE["eta_num"]))
    ap.add_argument("--ref-d", type=float, default=float(BASELINE["ref_D"]))
    ap.add_argument("--ref-eta", type=float, default=float(BASELINE["ref_eta"]))
    ap.add_argument("--mu-obs", type=float, default=1.4)
    ap.add_argument("--sigma-obs", type=float, default=0.4)
    ap.add_argument("--uv-blend", type=float, default=float(BASELINE["hll_uv_blend"]))
    ap.add_argument("--uv-m2-power", type=float, default=float(BASELINE["hll_uv_m2_power"]))
    ap.add_argument("--uv-match-kappa-diag", type=float, default=float(BASELINE["hll_uv_match_kappa_diag"]))
    ap.add_argument("--uv-match-kappa-offdiag", type=float, default=float(BASELINE["hll_uv_match_kappa_offdiag"]))
    ap.add_argument("--uv-match-mode", choices=["constant", "input_tied", "action_normalized", "action_absolute", "action_loop_contrast", "action_loop_absolute"], default=str(BASELINE["hll_uv_match_mode"]))
    ap.add_argument("--uv-match-input-diag-scale", type=float, default=float(BASELINE["hll_uv_match_input_diag_scale"]))
    ap.add_argument("--uv-match-input-offdiag-scale", type=float, default=float(BASELINE["hll_uv_match_input_offdiag_scale"]))
    ap.add_argument("--uv-rge-mu-low", type=float, default=float(BASELINE["hll_uv_rge_mu_low"]))
    ap.add_argument("--uv-rge-gamma-diag", type=float, default=float(BASELINE["hll_uv_rge_gamma_diag"]))
    ap.add_argument("--uv-rge-gamma-offdiag", type=float, default=float(BASELINE["hll_uv_rge_gamma_offdiag"]))
    ap.add_argument("--uv-rge-log-clip", type=float, default=float(BASELINE["hll_uv_rge_log_clip"]))
    ap.add_argument("--runtime-direct-force", action="store_true")
    ap.add_argument("--runtime-direct-no-cache", action="store_true")
    ap.add_argument("--runtime-direct-chi-rho-max", type=float, default=3.0)
    ap.add_argument("--runtime-direct-chi-z-margin", type=float, default=6.0)
    ap.add_argument("--runtime-direct-chi-n-mu", type=int, default=120)
    ap.add_argument("--runtime-direct-chi-tol", type=float, default=1e-8)
    ap.add_argument("--runtime-direct-chi-maxiter", type=int, default=30000)
    ap.add_argument("--runtime-direct-chi-sigma", type=float, default=2.5)
    ap.add_argument("--runtime-direct-superrad-zmax", type=float, default=80.0)
    ap.add_argument("--runtime-direct-superrad-ref-d", type=float, default=12.0)
    ap.add_argument("--runtime-direct-superrad-n-ref", type=int, default=2)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--skip-paper-copy", action="store_true")
    ap.add_argument("--skip-plot", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.d_num < 2 or args.eta_num < 2:
        raise ValueError("d-num and eta-num must be >= 2")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    kin = make_baseline_kinetics(
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        chain_mode=str(args.chain_mode),
        uv_blend=float(args.uv_blend),
        uv_m2_power=float(args.uv_m2_power),
        uv_match_kappa_diag=float(args.uv_match_kappa_diag),
        uv_match_kappa_offdiag=float(args.uv_match_kappa_offdiag),
        uv_match_mode=str(args.uv_match_mode),
        uv_match_input_diag_scale=float(args.uv_match_input_diag_scale),
        uv_match_input_offdiag_scale=float(args.uv_match_input_offdiag_scale),
        uv_rge_mu_low=float(args.uv_rge_mu_low),
        uv_rge_gamma_diag=float(args.uv_rge_gamma_diag),
        uv_rge_gamma_offdiag=float(args.uv_rge_gamma_offdiag),
        uv_rge_log_clip=float(args.uv_rge_log_clip),
        runtime_direct_force=bool(args.runtime_direct_force),
        runtime_direct_no_cache=bool(args.runtime_direct_no_cache),
        runtime_direct_chi_rho_max=float(args.runtime_direct_chi_rho_max),
        runtime_direct_chi_z_margin=float(args.runtime_direct_chi_z_margin),
        runtime_direct_chi_n_mu=int(args.runtime_direct_chi_n_mu),
        runtime_direct_chi_tol=float(args.runtime_direct_chi_tol),
        runtime_direct_chi_maxiter=int(args.runtime_direct_chi_maxiter),
        runtime_direct_chi_sigma=float(args.runtime_direct_chi_sigma),
        runtime_direct_superrad_zmax=float(args.runtime_direct_superrad_zmax),
        runtime_direct_superrad_ref_d=float(args.runtime_direct_superrad_ref_d),
        runtime_direct_superrad_n_ref=int(args.runtime_direct_superrad_n_ref),
    )

    d_vals = np.linspace(float(args.d_min), float(args.d_max), int(args.d_num))
    eta_vals = np.linspace(float(args.eta_min), float(args.eta_max), int(args.eta_num))
    ref_d_eff, snapped_ref_d = snap_ref_d_for_full_direct(str(args.chain_mode), float(args.ref_d), d_vals)

    t_coh = float(BASELINE["t_coh"])
    nmax = int(BASELINE["hll_observable_nmax"])
    match_cfg = kin._hll_match_config()

    ref_wit = kin.hll_uv_matching_witness(D=float(ref_d_eff), eta=float(args.ref_eta), t_coh=t_coh, N_max=nmax)
    ref_c_tree_diag = np.diag(np.asarray(ref_wit["c_tree"], dtype=float))
    ref_c_ir_diag = np.diag(np.asarray(ref_wit["c_ir"], dtype=float))

    c_tree_mumu = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    delta_match_mumu = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    delta_rge_mumu = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    mu_recon_resid = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    rows: list[dict[str, float]] = []

    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            wit = kin.hll_uv_matching_witness(D=float(d), eta=float(eta), t_coh=t_coh, N_max=nmax)
            c_tree = np.asarray(wit["c_tree"], dtype=float)
            c_match = np.asarray(wit["c_match"], dtype=float)
            c_ir = np.asarray(wit["c_ir"], dtype=float)
            coeff = np.asarray(wit["coefficients"], dtype=float)
            basis = np.asarray(wit["basis_matrices"], dtype=float)

            c_tree_recon = reconstruct_from_layer_basis(coefficients=coeff, basis_matrices=basis)
            c_match_recon = c_tree + np.asarray(wit["delta_match_diag"], dtype=float) + np.asarray(wit["delta_match_offdiag"], dtype=float)
            c_ir_recon = c_match + np.asarray(wit["delta_rge_diag"], dtype=float) + np.asarray(wit["delta_rge_offdiag"], dtype=float)

            c_tree_diag = np.diag(c_tree)
            c_ir_diag = np.diag(c_ir)
            mu_tree_recon = float((c_tree_diag[1] / max(ref_c_tree_diag[1], 1e-30)) ** 2)
            width_ir_recon = total_width_ratio(c_diag=c_ir_diag, c_diag_ref=ref_c_ir_diag, cfg=match_cfg)
            mu_ir_recon = float((c_ir_diag[1] / max(ref_c_ir_diag[1], 1e-30)) ** 2 / max(width_ir_recon, 1e-30))
            mu_ir_native = kin.hll_mu_pred(
                layer_n=2,
                D=float(d),
                eta=float(eta),
                t_coh=t_coh,
                ref_D=float(ref_d_eff),
                ref_eta=float(args.ref_eta),
                observable_mode="eft_wilson_uv_rge",
                N_max=nmax,
            )

            tree_term_mumu = [float(coeff[n] * basis[1, 1, n]) for n in range(3)]
            basis_offdiag_norm = [float(np.linalg.norm(basis[:, :, n] - np.diag(np.diag(basis[:, :, n])), ord="fro")) for n in range(3)]

            c_tree_mumu[i, j] = float(c_tree_diag[1])
            delta_match_mumu[i, j] = float(np.asarray(wit["delta_match_diag"], dtype=float)[1, 1])
            delta_rge_mumu[i, j] = float(np.asarray(wit["delta_rge_diag"], dtype=float)[1, 1])
            mu_recon_resid[i, j] = abs(float(mu_ir_native) - mu_ir_recon)

            rows.append(
                {
                    "D": float(d),
                    "eta": float(eta),
                    "coeff_1": float(coeff[0]),
                    "coeff_2": float(coeff[1]),
                    "coeff_3": float(coeff[2]),
                    "basis_1_mumu": float(basis[1, 1, 0]),
                    "basis_2_mumu": float(basis[1, 1, 1]),
                    "basis_3_mumu": float(basis[1, 1, 2]),
                    "basis_1_offdiag_frob": basis_offdiag_norm[0],
                    "basis_2_offdiag_frob": basis_offdiag_norm[1],
                    "basis_3_offdiag_frob": basis_offdiag_norm[2],
                    "tree_term_1_mumu": tree_term_mumu[0],
                    "tree_term_2_mumu": tree_term_mumu[1],
                    "tree_term_3_mumu": tree_term_mumu[2],
                    "C_tree_mumu": float(c_tree_diag[1]),
                    "C_match_mumu": float(np.diag(c_match)[1]),
                    "C_ir_mumu": float(c_ir_diag[1]),
                    "delta_C_match_diag_mumu": float(np.asarray(wit["delta_match_diag"], dtype=float)[1, 1]),
                    "delta_C_rge_diag_mumu": float(np.asarray(wit["delta_rge_diag"], dtype=float)[1, 1]),
                    "tree_rebuild_residual": float(np.max(np.abs(c_tree - c_tree_recon))),
                    "match_rebuild_residual": float(np.max(np.abs(c_match - c_match_recon))),
                    "ir_rebuild_residual": float(np.max(np.abs(c_ir - c_ir_recon))),
                    "mu_match": float(np.asarray(wit["mu_match"], dtype=float)[0]),
                    "mu_low": float(np.asarray(wit["mu_low"], dtype=float)[0]),
                    "log_ratio": float(np.asarray(wit["log_ratio"], dtype=float)[0]),
                    "finite_fac_diag": float(np.asarray(wit["finite_fac_diag"], dtype=float)[0]),
                    "finite_fac_offdiag": float(np.asarray(wit["finite_fac_offdiag"], dtype=float)[0]),
                    "shell_spread": float(np.asarray(wit["shell_spread"], dtype=float)[0]),
                    "coeff_cv": float(np.asarray(wit["coeff_cv"], dtype=float)[0]),
                    "offdiag_mix": float(np.asarray(wit["offdiag_mix"], dtype=float)[0]),
                    "gap_cv": float(np.asarray(wit["gap_cv"], dtype=float)[0]),
                    "gap_asym": float(np.asarray(wit["gap_asym"], dtype=float)[0]),
                    "g_col_norm_cv": float(np.asarray(wit["g_col_norm_cv"], dtype=float)[0]),
                    "c_tree_diag_cv": float(np.asarray(wit["c_tree_diag_cv"], dtype=float)[0]),
                    "pkin_entropy": float(np.asarray(wit["pkin_entropy"], dtype=float)[0]),
                    "action_norm_diag": float(np.asarray(wit["action_norm_diag"], dtype=float)[0]),
                    "action_norm_offdiag": float(np.asarray(wit["action_norm_offdiag"], dtype=float)[0]),
                    "coeff_l1": float(np.asarray(wit["coeff_l1"], dtype=float)[0]),
                    "coeff_l2": float(np.asarray(wit["coeff_l2"], dtype=float)[0]),
                    "coeff_align": float(np.asarray(wit["coeff_align"], dtype=float)[0]),
                    "action_abs_diag": float(np.asarray(wit["action_abs_diag"], dtype=float)[0]),
                    "action_abs_offdiag": float(np.asarray(wit["action_abs_offdiag"], dtype=float)[0]),
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
                    "hk_loop_prefactor_diag": float(np.asarray(wit["hk_loop_prefactor_diag"], dtype=float)[0]),
                    "hk_loop_prefactor_offdiag": float(np.asarray(wit["hk_loop_prefactor_offdiag"], dtype=float)[0]),
                    "mu_mumu_tree_recon": mu_tree_recon,
                    "mu_mumu_ir_recon": mu_ir_recon,
                    "mu_mumu_ir_native": float(mu_ir_native),
                    "abs_delta_mu_mumu_ir_recon_vs_native": abs(float(mu_ir_native) - mu_ir_recon),
                    "chi2_uv_rge": float(((float(mu_ir_native) - float(args.mu_obs)) / float(args.sigma_obs)) ** 2),
                }
            )

    arr_tree = np.asarray([r["tree_rebuild_residual"] for r in rows], dtype=float)
    arr_match = np.asarray([r["match_rebuild_residual"] for r in rows], dtype=float)
    arr_ir = np.asarray([r["ir_rebuild_residual"] for r in rows], dtype=float)
    arr_mu_recon = np.asarray([r["abs_delta_mu_mumu_ir_recon_vs_native"] for r in rows], dtype=float)
    arr_log = np.asarray([r["log_ratio"] for r in rows], dtype=float)
    arr_chi2 = np.asarray([r["chi2_uv_rge"] for r in rows], dtype=float)

    summary = {
        "n_points": float(len(rows)),
        "max_tree_rebuild_residual": float(np.max(arr_tree)),
        "max_match_rebuild_residual": float(np.max(arr_match)),
        "max_ir_rebuild_residual": float(np.max(arr_ir)),
        "mean_abs_delta_mu_mumu_ir_recon_vs_native": float(np.mean(arr_mu_recon)),
        "p95_abs_delta_mu_mumu_ir_recon_vs_native": float(np.percentile(arr_mu_recon, 95.0)),
        "max_abs_delta_mu_mumu_ir_recon_vs_native": float(np.max(arr_mu_recon)),
        "mean_log_ratio": float(np.mean(arr_log)),
        "p05_log_ratio": float(np.percentile(arr_log, 5.0)),
        "p95_log_ratio": float(np.percentile(arr_log, 95.0)),
        "f_chi2_le_4_uv_rge": float(np.mean(arr_chi2 <= 4.0)),
        "ref_D": float(ref_d_eff),
        "ref_eta": float(args.ref_eta),
        "ref_D_snapped_to_grid": bool(snapped_ref_d),
        "chain_mode": str(args.chain_mode),
        "uv_match_kappa_diag": float(args.uv_match_kappa_diag),
        "uv_match_kappa_offdiag": float(args.uv_match_kappa_offdiag),
        "uv_match_mode": str(args.uv_match_mode),
        "uv_match_input_diag_scale": float(args.uv_match_input_diag_scale),
        "uv_match_input_offdiag_scale": float(args.uv_match_input_offdiag_scale),
        "uv_rge_gamma_diag": float(args.uv_rge_gamma_diag),
        "uv_rge_gamma_offdiag": float(args.uv_rge_gamma_offdiag),
    }

    suffix = make_suffix(str(args.tag))
    out_map = OUTDIR / f"hll_uv_operator_basis_map{suffix}.csv"
    out_summary = OUTDIR / f"hll_uv_operator_basis_summary{suffix}.csv"
    out_fig = OUTDIR / f"hll_uv_operator_basis_maps{suffix}.png"
    out_meta = OUTDIR / f"hll_uv_operator_basis_run_meta{suffix or '_baseline'}.json"

    write_map_csv(out_map, rows)
    write_summary_csv(out_summary, summary)
    if not bool(args.skip_plot):
        plot_maps(
            out_png=out_fig,
            d_vals=d_vals,
            eta_vals=eta_vals,
            c_tree_mumu=c_tree_mumu,
            delta_match_mumu=delta_match_mumu,
            delta_rge_mumu=delta_rge_mumu,
            mu_recon_resid=mu_recon_resid,
        )

    run_meta = {
        "tag": str(args.tag),
        "chain_mode": str(args.chain_mode),
        "summary_file": str(out_summary),
        "map_file": str(out_map),
        "figure_file": str(out_fig),
        "ref_D_input": float(args.ref_d),
        "ref_D_effective": float(ref_d_eff),
        "ref_D_snapped_to_grid": bool(snapped_ref_d),
        "uv_match_mode": str(args.uv_match_mode),
        "uv_match_input_diag_scale": float(args.uv_match_input_diag_scale),
        "uv_match_input_offdiag_scale": float(args.uv_match_input_offdiag_scale),
        "d_min": float(args.d_min),
        "d_max": float(args.d_max),
        "d_num": int(args.d_num),
        "eta_min": float(args.eta_min),
        "eta_max": float(args.eta_max),
        "eta_num": int(args.eta_num),
    }
    out_meta.write_text(json.dumps(run_meta, indent=2))

    if not args.skip_paper_copy:
        (PAPER_DIR / out_map.name).write_text(out_map.read_text())
        (PAPER_DIR / out_summary.name).write_text(out_summary.read_text())
        if out_fig.exists():
            (PAPER_DIR / out_fig.name).write_bytes(out_fig.read_bytes())
        (PAPER_DIR / out_meta.name).write_text(out_meta.read_text())

    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    if out_fig.exists():
        print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")
    if not args.skip_paper_copy:
        print(f"[saved] {PAPER_DIR / out_map.name}")
        print(f"[saved] {PAPER_DIR / out_summary.name}")
        if out_fig.exists():
            print(f"[saved] {PAPER_DIR / out_fig.name}")
        print(f"[saved] {PAPER_DIR / out_meta.name}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

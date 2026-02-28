#!/usr/bin/env python3
"""
Grid-level UV-to-EFT matching audit for PSLT H->ll observables.

For each (D, eta) scan point, this script computes:
  - UV-tree Wilson matrix C_{eH}^{tree}(mu_match)
  - finite one-loop matched matrix C_{eH}^{match}(mu_match)
  - LL-RG evolved Wilson matrix C_{eH}(mu_low)
  - running metadata (mu_match, log(mu_match/mu_low))
  - induced drift in mu_mumu between uv_tree and uv_rge modes

Outputs:
  - output/hll_uv_matching/hll_uv_to_eft_map*.csv
  - output/hll_uv_matching/hll_uv_to_eft_summary*.csv
  - output/hll_uv_matching/hll_uv_to_eft_maps*.png
  - output/hll_uv_matching/hll_uv_to_eft_run_meta*.json
and copies summary/figure/meta to paper/ unless --skip-paper-copy is set.

Chain profile selection:
  - --chain-mode auto        (default): prefer grid-aligned profiles and allow
                              interpolation fallback when needed.
  - --chain-mode full_direct: require exact localized-direct D-grid profiles
                              and strict grid lookup (no interpolation fallback).
  - --chain-mode full_direct_runtime:
                              build/rebuild active D-grid localized-direct
                              profiles at runtime, then run strict full_direct.
  - --chain-mode cell_direct_runtime:
                              no profile object; evaluate g_N(D), chi_LR(D),
                              and A_l(D) by direct solvers inside scan cells.
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

from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile  # noqa: E402
from pslt_lib import PSLTKinetics, PSLTParameters  # noqa: E402
from direct_chain_runtime import ensure_runtime_full_direct_profiles  # noqa: E402


OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


BASELINE = {
    "c_eff": 0.5,
    "nu": 5.0,
    "kappa_g": 0.03,
    "g_mode": "fp_2d_full",
    "g_fp_norm_mode": "phase_space",
    "g_fp_full_window_blend": 0.8,
    "g_fp_full_tail_beta": 1.1,
    "g_fp_full_tail_shell_power": 0.0,
    "g_fp_full_tail_clip_min": 1e-3,
    "g_fp_full_tail_clip_max": 0.95,
    "chi_legacy": 0.2,
    "chi_mode": "localized_grid",
    "A1": 1.0,
    "A2": 1.0,
    "gamma_mode": "action_grid",
    "p_B": 0.30,
    "b_mode": "overlap_2d",
    "t_coh": 1.0,
    "hll_observable_nmax": 20,
    "hll_uv_blend": 0.00,
    "hll_uv_m2_power": 1.00,
    "hll_uv_match_kappa_diag": 0.0,
    "hll_uv_match_kappa_offdiag": 0.0,
    "hll_uv_rge_mu_low": 1.0,
    "hll_uv_rge_gamma_diag": 2.0,
    "hll_uv_rge_gamma_offdiag": 1.0,
    "hll_uv_rge_log_clip": 6.0,
    "D_min": 4.0,
    "D_max": 20.0,
    "D_num": 60,
    "eta_min": 0.2,
    "eta_max": 4.0,
    "eta_num": 60,
    "ref_D": 10.0,
    "ref_eta": 1.0,
}


def sanitize_tag(tag: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(tag).strip())


def make_suffix(tag: str) -> str:
    clean = sanitize_tag(tag)
    return f"_{clean}" if clean else ""


def snap_ref_d_for_full_direct(chain_mode: str, ref_d: float, d_vals: np.ndarray) -> tuple[float, bool]:
    if str(chain_mode) not in {"full_direct", "full_direct_runtime"}:
        return float(ref_d), False
    if len(d_vals) == 0:
        return float(ref_d), False
    arr = np.asarray(d_vals, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(ref_d))))
    snapped = float(arr[idx])
    changed = not np.isclose(snapped, float(ref_d), rtol=0.0, atol=1e-10)
    return snapped, changed


def make_baseline_kinetics(
    d_min: float,
    d_max: float,
    d_num: int,
    chain_mode: str,
    uv_blend: float,
    uv_m2_power: float,
    uv_match_kappa_diag: float,
    uv_match_kappa_offdiag: float,
    uv_rge_mu_low: float,
    uv_rge_gamma_diag: float,
    uv_rge_gamma_offdiag: float,
    uv_rge_log_clip: float,
    runtime_direct_force: bool,
    runtime_direct_no_cache: bool,
    runtime_direct_chi_rho_max: float,
    runtime_direct_chi_z_margin: float,
    runtime_direct_chi_n_mu: int,
    runtime_direct_chi_tol: float,
    runtime_direct_chi_maxiter: int,
    runtime_direct_chi_sigma: float,
    runtime_direct_superrad_zmax: float,
    runtime_direct_superrad_ref_d: float,
    runtime_direct_superrad_n_ref: int,
) -> PSLTKinetics:
    d_scan = scan_d_values(d_min, d_max, d_num)
    chain_mode_eff = str(chain_mode).strip().lower()
    selection_mode = "full_direct" if chain_mode_eff in {"full_direct", "full_direct_runtime"} else "auto"
    g_mode = str(BASELINE["g_mode"])
    chi_prof = None
    superrad_prof = None
    chi_mode = "localized_grid"
    gamma_mode = "action_grid"
    g_source = "gn_profile_csv"
    chi_source = "runtime_cell_solver"
    gamma_source = "runtime_cell_solver"

    if chain_mode_eff == "full_direct_runtime":
        ensure_runtime_full_direct_profiles(
            root=ROOT,
            d_scan=d_scan,
            force=bool(runtime_direct_force),
            chi_rho_max=float(runtime_direct_chi_rho_max),
            chi_z_margin=float(runtime_direct_chi_z_margin),
            chi_n_mu=int(runtime_direct_chi_n_mu),
            chi_tol=float(runtime_direct_chi_tol),
            chi_maxiter=int(runtime_direct_chi_maxiter),
            chi_sigma=float(runtime_direct_chi_sigma),
            superrad_zmax=float(runtime_direct_superrad_zmax),
            superrad_ref_d=float(runtime_direct_superrad_ref_d),
            superrad_n_ref=int(runtime_direct_superrad_n_ref),
        )
        chi_prof = select_chi_profile(ROOT, d_scan, selection_mode=selection_mode)
        superrad_prof = select_superrad_profile(ROOT, d_scan, selection_mode=selection_mode)
        chi_mode = str(chi_prof["mode"])
        gamma_mode = str(superrad_prof["mode"])
        chi_source = str(chi_prof["path"])
        gamma_source = str(superrad_prof["path"])
    elif chain_mode_eff == "cell_direct_runtime":
        g_mode = "fp_2d_full_runtime_direct"
        chi_mode = "localized_runtime_direct"
        gamma_mode = "action_runtime_direct"
        g_source = "runtime_cell_solver"
    else:
        chi_prof = select_chi_profile(ROOT, d_scan, selection_mode=selection_mode)
        superrad_prof = select_superrad_profile(ROOT, d_scan, selection_mode=selection_mode)
        chi_mode = str(chi_prof["mode"])
        gamma_mode = str(superrad_prof["mode"])
        chi_source = str(chi_prof["path"])
        gamma_source = str(superrad_prof["path"])

    params = PSLTParameters(
        c_eff=BASELINE["c_eff"],
        nu=BASELINE["nu"],
        kappa_g=BASELINE["kappa_g"],
        g_mode=g_mode,
        g_fp_norm_mode=BASELINE["g_fp_norm_mode"],
        g_fp_full_window_blend=BASELINE["g_fp_full_window_blend"],
        g_fp_full_tail_beta=BASELINE["g_fp_full_tail_beta"],
        g_fp_full_tail_shell_power=BASELINE["g_fp_full_tail_shell_power"],
        g_fp_full_tail_clip_min=BASELINE["g_fp_full_tail_clip_min"],
        g_fp_full_tail_clip_max=BASELINE["g_fp_full_tail_clip_max"],
        runtime_direct_g_rho_max=float(runtime_direct_chi_rho_max),
        runtime_direct_g_z_margin=float(runtime_direct_chi_z_margin),
        runtime_direct_g_n_eigs=40,
        runtime_direct_g_tol=float(runtime_direct_chi_tol),
        runtime_direct_g_maxiter=int(runtime_direct_chi_maxiter),
        runtime_direct_g_sigma=float(runtime_direct_chi_sigma),
        chi=BASELINE["chi_legacy"],
        chi_mode=str(chi_mode),
        chi_lr_D=tuple(float(x) for x in (np.asarray(chi_prof["d"], dtype=float) if chi_prof is not None else np.array([6.0, 12.0, 18.0], dtype=float))),
        chi_lr_vals=tuple(float(x) for x in (np.asarray(chi_prof["chi"], dtype=float) if chi_prof is not None else np.array([4.01827e-4, 2.21414e-4, 2.13187e-4], dtype=float))),
        A1=BASELINE["A1"],
        A2=BASELINE["A2"],
        gamma_mode=str(gamma_mode),
        gamma_superrad_csv=str(superrad_prof["path"]) if superrad_prof is not None else None,
        runtime_direct_use_cache=not bool(runtime_direct_no_cache),
        runtime_direct_chi_rho_max=float(runtime_direct_chi_rho_max),
        runtime_direct_chi_z_margin=float(runtime_direct_chi_z_margin),
        runtime_direct_chi_n_mu=int(runtime_direct_chi_n_mu),
        runtime_direct_chi_tol=float(runtime_direct_chi_tol),
        runtime_direct_chi_maxiter=int(runtime_direct_chi_maxiter),
        runtime_direct_chi_sigma=float(runtime_direct_chi_sigma),
        runtime_direct_superrad_zmax=float(runtime_direct_superrad_zmax),
        runtime_direct_superrad_ref_d=float(runtime_direct_superrad_ref_d),
        runtime_direct_superrad_n_ref=int(runtime_direct_superrad_n_ref),
        b_mode=BASELINE["b_mode"],
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode="eft_wilson_uv_rge",
        hll_observable_nmax=int(BASELINE["hll_observable_nmax"]),
        hll_uv_blend=float(uv_blend),
        hll_uv_m2_power=float(uv_m2_power),
        hll_uv_match_kappa_diag=float(uv_match_kappa_diag),
        hll_uv_match_kappa_offdiag=float(uv_match_kappa_offdiag),
        hll_uv_rge_mu_low=float(uv_rge_mu_low),
        hll_uv_rge_gamma_diag=float(uv_rge_gamma_diag),
        hll_uv_rge_gamma_offdiag=float(uv_rge_gamma_offdiag),
        hll_uv_rge_log_clip=float(uv_rge_log_clip),
    )

    print(
        "[baseline]",
        f"chain_mode={chain_mode_eff},",
        f"selection_mode={selection_mode},",
        f"g_mode={params.g_mode},",
        f"g_source={g_source},",
        f"chi_mode={params.chi_mode},",
        f"chi_source={chi_source},",
        f"gamma_mode={params.gamma_mode},",
        f"gamma_source={gamma_source}",
        f"runtime_direct_use_cache={params.runtime_direct_use_cache}",
    )
    return PSLTKinetics(params)


def write_map_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise RuntimeError("no map rows")
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
    ceh_uv_mumu: np.ndarray,
    ceh_ir_mumu: np.ndarray,
    abs_delta_c: np.ndarray,
    abs_delta_mu: np.ndarray,
) -> None:
    extent = [float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)

    panels = [
        (axes[0, 0], ceh_uv_mumu, r"$C_{\mu\mu}^{UV}$"),
        (axes[0, 1], ceh_ir_mumu, r"$C_{\mu\mu}^{IR}$"),
        (axes[1, 0], abs_delta_c, r"$|\Delta C_{\mu\mu}|$"),
        (axes[1, 1], abs_delta_mu, r"$|\Delta \mu_{\mu\mu}|$"),
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

    fig.suptitle("UV-to-EFT matching maps (UV tree vs LL-RG IR)", fontsize=13)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="UV-to-EFT matching audit on PSLT scan grid")
    ap.add_argument("--chain-mode", choices=["auto", "full_direct", "full_direct_runtime", "cell_direct_runtime"], default="full_direct")
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
    if snapped_ref_d:
        print(
            "[info] chain_mode in {full_direct,full_direct_runtime} snapped ref_D to grid:",
            f"{float(args.ref_d):.6g} -> {ref_d_eff:.6g}",
        )

    t_coh = float(BASELINE["t_coh"])
    nmax = int(BASELINE["hll_observable_nmax"])

    ceh_uv_mumu = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    ceh_ir_mumu = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    abs_delta_c = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    abs_delta_mu = np.zeros((len(eta_vals), len(d_vals)), dtype=float)

    rows: list[dict[str, float]] = []

    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            c_uv = kin.compute_ceh_uv(float(d), float(eta), t_coh, N_max=nmax)
            c_match, _match_meta = kin.hll_wilson_matrix_uv_match_with_meta(float(d), float(eta), t_coh, N_max=nmax)
            c_ir, meta = kin.hll_wilson_matrix_uv_rge_with_meta(float(d), float(eta), t_coh, N_max=nmax)

            mu_uv = kin.hll_mu_pred(
                layer_n=2,
                D=float(d),
                eta=float(eta),
                t_coh=t_coh,
                ref_D=float(ref_d_eff),
                ref_eta=float(args.ref_eta),
                observable_mode="eft_wilson_uv_tree",
                N_max=nmax,
            )
            mu_ir = kin.hll_mu_pred(
                layer_n=2,
                D=float(d),
                eta=float(eta),
                t_coh=t_coh,
                ref_D=float(ref_d_eff),
                ref_eta=float(args.ref_eta),
                observable_mode="eft_wilson_uv_rge",
                N_max=nmax,
            )

            c_uv_diag = np.diag(c_uv)
            c_match_diag = np.diag(c_match)
            c_ir_diag = np.diag(c_ir)
            d_c_match = float(c_match_diag[1] - c_uv_diag[1])
            d_c_rge = float(c_ir_diag[1] - c_match_diag[1])
            d_c = float(c_ir_diag[1] - c_uv_diag[1])
            d_mu = float(mu_ir - mu_uv)

            ceh_uv_mumu[i, j] = float(c_uv_diag[1])
            ceh_ir_mumu[i, j] = float(c_ir_diag[1])
            abs_delta_c[i, j] = abs(d_c)
            abs_delta_mu[i, j] = abs(d_mu)

            rows.append(
                {
                    "D": float(d),
                    "eta": float(eta),
                    "C_uv_ee": float(c_uv_diag[0]),
                    "C_uv_mumu": float(c_uv_diag[1]),
                    "C_uv_tautau": float(c_uv_diag[2]),
                    "C_match_ee": float(c_match_diag[0]),
                    "C_match_mumu": float(c_match_diag[1]),
                    "C_match_tautau": float(c_match_diag[2]),
                    "C_ir_ee": float(c_ir_diag[0]),
                    "C_ir_mumu": float(c_ir_diag[1]),
                    "C_ir_tautau": float(c_ir_diag[2]),
                    "delta_C_match_mumu": d_c_match,
                    "abs_delta_C_match_mumu": abs(d_c_match),
                    "delta_C_rge_mumu": d_c_rge,
                    "abs_delta_C_rge_mumu": abs(d_c_rge),
                    "delta_C_mumu": d_c,
                    "abs_delta_C_mumu": abs(d_c),
                    "rel_delta_C_mumu": float(d_c / max(abs(c_uv_diag[1]), 1e-30)),
                    "mu_match": float(meta["mu_match"]),
                    "mu_low": float(meta["mu_low"]),
                    "log_ratio": float(meta["log_ratio"]),
                    "finite_fac_diag": float(meta["finite_fac_diag"]),
                    "finite_fac_offdiag": float(meta["finite_fac_offdiag"]),
                    "kappa_diag": float(meta["kappa_diag"]),
                    "kappa_offdiag": float(meta["kappa_offdiag"]),
                    "mu_mumu_uv_tree": float(mu_uv),
                    "mu_mumu_uv_rge": float(mu_ir),
                    "delta_mu_mumu": d_mu,
                    "abs_delta_mu_mumu": abs(d_mu),
                    "chi2_uv_tree": float(((mu_uv - float(args.mu_obs)) / float(args.sigma_obs)) ** 2),
                    "chi2_uv_rge": float(((mu_ir - float(args.mu_obs)) / float(args.sigma_obs)) ** 2),
                }
            )

    arr_abs_dc = np.asarray([r["abs_delta_C_mumu"] for r in rows], dtype=float)
    arr_abs_dc_match = np.asarray([r["abs_delta_C_match_mumu"] for r in rows], dtype=float)
    arr_abs_dc_rge = np.asarray([r["abs_delta_C_rge_mumu"] for r in rows], dtype=float)
    arr_abs_dm = np.asarray([r["abs_delta_mu_mumu"] for r in rows], dtype=float)
    arr_log = np.asarray([r["log_ratio"] for r in rows], dtype=float)
    arr_chi2_uv = np.asarray([r["chi2_uv_tree"] for r in rows], dtype=float)
    arr_chi2_ir = np.asarray([r["chi2_uv_rge"] for r in rows], dtype=float)

    summary = {
        "n_points": float(len(rows)),
        "mean_abs_delta_C_match_mumu": float(np.mean(arr_abs_dc_match)),
        "p95_abs_delta_C_match_mumu": float(np.percentile(arr_abs_dc_match, 95.0)),
        "max_abs_delta_C_match_mumu": float(np.max(arr_abs_dc_match)),
        "mean_abs_delta_C_rge_mumu": float(np.mean(arr_abs_dc_rge)),
        "p95_abs_delta_C_rge_mumu": float(np.percentile(arr_abs_dc_rge, 95.0)),
        "max_abs_delta_C_rge_mumu": float(np.max(arr_abs_dc_rge)),
        "mean_abs_delta_C_mumu": float(np.mean(arr_abs_dc)),
        "p95_abs_delta_C_mumu": float(np.percentile(arr_abs_dc, 95.0)),
        "max_abs_delta_C_mumu": float(np.max(arr_abs_dc)),
        "mean_abs_delta_mu_mumu": float(np.mean(arr_abs_dm)),
        "p95_abs_delta_mu_mumu": float(np.percentile(arr_abs_dm, 95.0)),
        "max_abs_delta_mu_mumu": float(np.max(arr_abs_dm)),
        "mean_log_ratio": float(np.mean(arr_log)),
        "p05_log_ratio": float(np.percentile(arr_log, 5.0)),
        "p95_log_ratio": float(np.percentile(arr_log, 95.0)),
        "f_chi2_le_4_uv_tree": float(np.mean(arr_chi2_uv <= 4.0)),
        "f_chi2_le_4_uv_rge": float(np.mean(arr_chi2_ir <= 4.0)),
        "delta_f_chi2_le_4_uv_rge_minus_uv_tree": float(np.mean(arr_chi2_ir <= 4.0) - np.mean(arr_chi2_uv <= 4.0)),
        "best_chi2_uv_tree": float(np.min(arr_chi2_uv)),
        "best_chi2_uv_rge": float(np.min(arr_chi2_ir)),
        "ref_D": float(ref_d_eff),
        "ref_eta": float(args.ref_eta),
        "mu_obs": float(args.mu_obs),
        "sigma_obs": float(args.sigma_obs),
        "uv_blend": float(args.uv_blend),
        "uv_m2_power": float(args.uv_m2_power),
        "uv_match_kappa_diag": float(args.uv_match_kappa_diag),
        "uv_match_kappa_offdiag": float(args.uv_match_kappa_offdiag),
        "uv_rge_mu_low": float(args.uv_rge_mu_low),
        "uv_rge_gamma_diag": float(args.uv_rge_gamma_diag),
        "uv_rge_gamma_offdiag": float(args.uv_rge_gamma_offdiag),
        "uv_rge_log_clip": float(args.uv_rge_log_clip),
        "chain_mode": str(args.chain_mode),
    }

    suffix = make_suffix(str(args.tag))
    out_map = OUTDIR / f"hll_uv_to_eft_map{suffix}.csv"
    out_summary = OUTDIR / f"hll_uv_to_eft_summary{suffix}.csv"
    out_fig = OUTDIR / f"hll_uv_to_eft_maps{suffix}.png"
    out_meta = OUTDIR / f"hll_uv_to_eft_run_meta{suffix or '_baseline'}.json"

    write_map_csv(out_map, rows)
    write_summary_csv(out_summary, summary)
    plot_maps(
        out_png=out_fig,
        d_vals=d_vals,
        eta_vals=eta_vals,
        ceh_uv_mumu=ceh_uv_mumu,
        ceh_ir_mumu=ceh_ir_mumu,
        abs_delta_c=abs_delta_c,
        abs_delta_mu=abs_delta_mu,
    )

    run_meta = {
        "tag": str(args.tag),
        "chain_mode": str(args.chain_mode),
        "runtime_direct_force": bool(args.runtime_direct_force),
        "runtime_direct_no_cache": bool(args.runtime_direct_no_cache),
        "runtime_direct_chi_rho_max": float(args.runtime_direct_chi_rho_max),
        "runtime_direct_chi_z_margin": float(args.runtime_direct_chi_z_margin),
        "runtime_direct_chi_n_mu": int(args.runtime_direct_chi_n_mu),
        "runtime_direct_chi_tol": float(args.runtime_direct_chi_tol),
        "runtime_direct_chi_maxiter": int(args.runtime_direct_chi_maxiter),
        "runtime_direct_chi_sigma": float(args.runtime_direct_chi_sigma),
        "runtime_direct_superrad_zmax": float(args.runtime_direct_superrad_zmax),
        "runtime_direct_superrad_ref_d": float(args.runtime_direct_superrad_ref_d),
        "runtime_direct_superrad_n_ref": int(args.runtime_direct_superrad_n_ref),
        "ref_D_input": float(args.ref_d),
        "ref_D_effective": float(ref_d_eff),
        "ref_D_snapped_to_grid": bool(snapped_ref_d),
        "d_min": float(args.d_min),
        "d_max": float(args.d_max),
        "d_num": int(args.d_num),
        "eta_min": float(args.eta_min),
        "eta_max": float(args.eta_max),
        "eta_num": int(args.eta_num),
        "uv_match_kappa_diag": float(args.uv_match_kappa_diag),
        "uv_match_kappa_offdiag": float(args.uv_match_kappa_offdiag),
        "summary_file": str(out_summary),
        "map_file": str(out_map),
        "figure_file": str(out_fig),
    }
    out_meta.write_text(json.dumps(run_meta, indent=2))

    if not args.skip_paper_copy:
        (PAPER_DIR / out_map.name).write_text(out_map.read_text())
        (PAPER_DIR / out_summary.name).write_text(out_summary.read_text())
        (PAPER_DIR / out_fig.name).write_bytes(out_fig.read_bytes())
        (PAPER_DIR / out_meta.name).write_text(out_meta.read_text())

    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")
    if not args.skip_paper_copy:
        print(f"[saved] {PAPER_DIR / out_map.name}")
        print(f"[saved] {PAPER_DIR / out_summary.name}")
        print(f"[saved] {PAPER_DIR / out_fig.name}")
        print(f"[saved] {PAPER_DIR / out_meta.name}")
    print("[summary]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

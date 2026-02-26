#!/usr/bin/env python3
"""
End-to-end localized direct-chain bias audit on a small map surface.

Goal:
  Quantify map-level transfer bias between
    (A) current scan baseline profile chain
    (B) direct localized extraction chain on sampled D-grid
  where both chi_LR(D) and superradiant A_l(D) are extracted directly.

At each sampled D:
  - chi_LR(D): from 2D localized solver (extract_chi_localized_2d.run_case)
  - A1/A2(D): from 1D superrad extractor (extract_superrad_prefactor_1d.solve_case)

Then evaluate map metrics on (D_i, eta_j):
  winner, R3, mu_mumu, chi2_mumu
and report bias upper bounds.

Outputs:
  - output/kinetic_action_chain/localized_direct_surface_points_Dgrid{N}_Egrid{M}.csv
  - output/kinetic_action_chain/localized_direct_surface_summary_Dgrid{N}_Egrid{M}.csv
  - output/kinetic_action_chain/localized_direct_profiles_chi_Dgrid{N}.csv
  - output/kinetic_action_chain/localized_direct_profiles_superrad_Dgrid{N}.csv
  - output/kinetic_action_chain/localized_direct_surface_bias_Dgrid{N}_Egrid{M}.png
  - paper/localized_direct_surface_summary_Dgrid{N}_Egrid{M}.csv
  - paper/localized_direct_surface_bias_Dgrid{N}_Egrid{M}.png
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from action_grid_profile_utils import scan_d_values
from extract_chi_localized_2d import Level as ChiLevel
from extract_chi_localized_2d import PhysicalParams as ChiParams
from extract_chi_localized_2d import run_case as run_chi_case
from extract_superrad_prefactor_1d import Level as SuperradLevel
from extract_superrad_prefactor_1d import PhysicalParams as SuperradParams
from extract_superrad_prefactor_1d import solve_case as run_superrad_case
from hll_observable import HLLChannelPredictor, HLLObservableConfig
from pslt_lib import PSLTKinetics, PSLTParameters


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"
BASELINE_CHI_CSV = ROOT / "output" / "chi_fp_2d" / "localized_chi_Dgrid60_fine.csv"
BASELINE_SUPERRAD_CSV = ROOT / "output" / "superrad_fp_1d" / "superrad_prefactor_Dgrid60_fine.csv"


@dataclass(frozen=True)
class BaselineConfig:
    c_eff: float = 0.5
    nu: float = 5.0
    kappa_g: float = 0.03
    g_mode: str = "fp_2d_full"
    g_fp_norm_mode: str = "phase_space"
    g_fp_full_window_blend: float = 0.8
    g_fp_full_tail_beta: float = 1.1
    g_fp_full_tail_shell_power: float = 0.0
    g_fp_full_tail_clip_min: float = 1e-3
    g_fp_full_tail_clip_max: float = 0.95
    b_mode: str = "overlap_2d"
    p_B: float = 0.30
    t_coh: float = 1.0
    n_max: int = 20
    hll_observable_mode: str = "eft_wilson_uv_rge"
    hll_observable_nmax: int = 20
    hmumu_ref_D: float = 10.0
    hmumu_ref_eta: float = 1.0
    hmumu_mu_obs: float = 1.4
    hmumu_sigma_obs: float = 0.4


CFG = BaselineConfig()


def _load_baseline_chi(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline chi profile not found: {path}")
    rows: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            level = str(row.get("level", "")).strip().lower()
            if level and level != "fine":
                continue
            d = row.get("D", "")
            chi = row.get("chi_LR", "")
            if d in {"", None} or chi in {"", None}:
                continue
            rows.append((float(d), float(chi)))
    if len(rows) < 2:
        raise RuntimeError(f"Baseline chi profile has insufficient rows: {path}")
    rows.sort(key=lambda t: t[0])
    d = np.array([x for x, _ in rows], dtype=float)
    chi = np.array([y for _, y in rows], dtype=float)
    return d, chi


def extract_direct_chi_profile(
    d_vals: np.ndarray,
    rho_max: float,
    z_margin: float,
    n_mu: int,
    tol: float,
    maxiter: int,
    sigma: float,
) -> pd.DataFrame:
    level = ChiLevel("fine", dr=0.06, dz=0.03)
    p = ChiParams()
    rows: List[Dict[str, float]] = []
    for i, d in enumerate(d_vals):
        if i % 5 == 0 or i == len(d_vals) - 1:
            print(f"[direct chi] {i+1}/{len(d_vals)} D={d:.6f}")
        row = run_chi_case(
            D=float(d),
            level=level,
            p=p,
            rho_max=float(rho_max),
            z_margin=float(z_margin),
            n_mu=int(n_mu),
            tol=float(tol),
            maxiter=int(maxiter),
            sigma=None if sigma < 0 else float(sigma),
        )
        row["grid_index"] = int(i)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def extract_direct_superrad_profile(
    d_vals: np.ndarray,
    zmax: float,
    d_ref: float,
    n_ref: int,
) -> pd.DataFrame:
    level = SuperradLevel("fine", Nz=8001)
    p = SuperradParams()
    ref = run_superrad_case(D=float(d_ref), level=level, zmax=float(zmax), p=p, n_ref=int(n_ref))
    a1_ref = float(ref["A1_fp"])
    a2_ref = float(ref["A2_fp"])

    rows: List[Dict[str, float]] = []
    for i, d in enumerate(d_vals):
        if i % 5 == 0 or i == len(d_vals) - 1:
            print(f"[direct A_l] {i+1}/{len(d_vals)} D={d:.6f}")
        row = run_superrad_case(D=float(d), level=level, zmax=float(zmax), p=p, n_ref=int(n_ref))
        row["grid_index"] = int(i)
        row["A1_profile"] = float(row["A1_fp"]) / max(a1_ref, 1e-300)
        row["A2_profile"] = float(row["A2_fp"]) / max(a2_ref, 1e-300)
        row["D_ref_profile"] = float(d_ref)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def make_kinetics_baseline() -> PSLTKinetics:
    d_chi, chi_vals = _load_baseline_chi(BASELINE_CHI_CSV)
    if not BASELINE_SUPERRAD_CSV.exists():
        raise FileNotFoundError(f"Baseline superrad profile not found: {BASELINE_SUPERRAD_CSV}")
    params = PSLTParameters(
        c_eff=CFG.c_eff,
        nu=CFG.nu,
        kappa_g=CFG.kappa_g,
        g_mode=CFG.g_mode,
        g_fp_norm_mode=CFG.g_fp_norm_mode,
        g_fp_full_window_blend=CFG.g_fp_full_window_blend,
        g_fp_full_tail_beta=CFG.g_fp_full_tail_beta,
        g_fp_full_tail_shell_power=CFG.g_fp_full_tail_shell_power,
        g_fp_full_tail_clip_min=CFG.g_fp_full_tail_clip_min,
        g_fp_full_tail_clip_max=CFG.g_fp_full_tail_clip_max,
        chi=0.2,
        chi_mode="localized_interp",
        chi_lr_D=tuple(float(x) for x in d_chi),
        chi_lr_vals=tuple(float(x) for x in chi_vals),
        A1=1.0,
        A2=1.0,
        gamma_mode="action_profile",
        gamma_superrad_csv=str(BASELINE_SUPERRAD_CSV),
        b_mode=CFG.b_mode,
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=CFG.p_B,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode=CFG.hll_observable_mode,
        hll_observable_nmax=CFG.hll_observable_nmax,
    )
    return PSLTKinetics(params)


def stable_probabilities(
    kin: PSLTKinetics,
    d_val: float,
    eta_val: float,
    t_coh: float,
    n_max: int,
    gamma_t_floor: float,
    prob_total_floor: float,
) -> Tuple[np.ndarray, bool]:
    ns = np.arange(1, int(n_max) + 1, dtype=int)
    t_eff = float(kin.t_coh_effective(float(d_val), float(t_coh)))
    weights = np.zeros(len(ns), dtype=float)

    for idx, n in enumerate(ns):
        gamma = max(float(kin.calculate_gamma_N(int(n), float(d_val), float(eta_val))), 0.0)
        g_n = max(float(kin.g_N_effective(int(n), float(d_val))), 0.0)
        b_n = max(float(kin.B_N(int(n), float(d_val))), float(kin.params.b_overlap_floor))
        x = max(gamma * t_eff, 0.0)
        one_minus_exp = -math.expm1(-x) if x > 0.0 else 0.0
        if one_minus_exp <= 0.0 and gamma > 0.0:
            one_minus_exp = max(x, float(gamma_t_floor))
        weights[idx] = max(b_n * g_n * one_minus_exp, 0.0)

    total = float(np.sum(weights))
    if total > float(prob_total_floor):
        return (weights / total, False)

    # Underflow fallback: keep ranking from static b_n * g_n shape if all kinetic terms collapsed.
    fallback = np.zeros(len(ns), dtype=float)
    for idx, n in enumerate(ns):
        g_n = max(float(kin.g_N_effective(int(n), float(d_val))), 0.0)
        b_n = max(float(kin.B_N(int(n), float(d_val))), float(kin.params.b_overlap_floor))
        fallback[idx] = max(b_n * g_n, 0.0)
    fsum = float(np.sum(fallback))
    if fsum > 0.0:
        return (fallback / fsum, True)
    return (np.full(len(ns), 1.0 / len(ns), dtype=float), True)


def make_kinetics_direct(
    d_vals: np.ndarray,
    chi_df: pd.DataFrame,
    superrad_csv: Path,
) -> PSLTKinetics:
    d = np.asarray(chi_df["D"], dtype=float)
    chi = np.asarray(chi_df["chi_LR"], dtype=float)
    if len(d) != len(d_vals):
        raise RuntimeError("Direct chi profile length does not match requested D grid.")

    params = PSLTParameters(
        c_eff=CFG.c_eff,
        nu=CFG.nu,
        kappa_g=CFG.kappa_g,
        g_mode=CFG.g_mode,
        g_fp_norm_mode=CFG.g_fp_norm_mode,
        g_fp_full_window_blend=CFG.g_fp_full_window_blend,
        g_fp_full_tail_beta=CFG.g_fp_full_tail_beta,
        g_fp_full_tail_shell_power=CFG.g_fp_full_tail_shell_power,
        g_fp_full_tail_clip_min=CFG.g_fp_full_tail_clip_min,
        g_fp_full_tail_clip_max=CFG.g_fp_full_tail_clip_max,
        chi=0.2,
        chi_mode="localized_grid",
        chi_lr_D=tuple(float(x) for x in d),
        chi_lr_vals=tuple(float(x) for x in chi),
        A1=1.0,
        A2=1.0,
        gamma_mode="action_grid",
        gamma_superrad_csv=str(superrad_csv),
        b_mode=CFG.b_mode,
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=CFG.p_B,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode=CFG.hll_observable_mode,
        hll_observable_nmax=CFG.hll_observable_nmax,
    )
    return PSLTKinetics(params)


def evaluate_map(
    kin: PSLTKinetics,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    hll_cfg: HLLObservableConfig,
    gamma_t_floor: float,
    prob_total_floor: float,
) -> Dict[str, np.ndarray]:
    winner = np.zeros((len(eta_vals), len(d_vals)), dtype=int)
    r3 = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    mu = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    chi2 = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    underflow = np.zeros((len(eta_vals), len(d_vals)), dtype=int)

    pred = HLLChannelPredictor(kin, layer_n=2, cfg=hll_cfg)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            p_n, used_fallback = stable_probabilities(
                kin=kin,
                d_val=float(d),
                eta_val=float(eta),
                t_coh=float(CFG.t_coh),
                n_max=int(CFG.n_max),
                gamma_t_floor=float(gamma_t_floor),
                prob_total_floor=float(prob_total_floor),
            )
            winner[i, j] = int(np.argmax(p_n) + 1)
            r3[i, j] = float(np.sum(p_n[:3]))
            underflow[i, j] = int(used_fallback)
            mu_ij = float(pred.mu_pred(float(d), float(eta)))
            mu[i, j] = mu_ij
            chi2[i, j] = float(((mu_ij - CFG.hmumu_mu_obs) / CFG.hmumu_sigma_obs) ** 2)

    return {"winner": winner, "R3": r3, "mu_mumu": mu, "chi2_mumu": chi2, "underflow_fallback": underflow}


def save_point_table(
    out_csv: Path,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    base: Dict[str, np.ndarray],
    direct: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, float | int]] = []
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            w_b = int(base["winner"][i, j])
            w_d = int(direct["winner"][i, j])
            r3_b = float(base["R3"][i, j])
            r3_d = float(direct["R3"][i, j])
            mu_b = float(base["mu_mumu"][i, j])
            mu_d = float(direct["mu_mumu"][i, j])
            c2_b = float(base["chi2_mumu"][i, j])
            c2_d = float(direct["chi2_mumu"][i, j])
            rows.append(
                {
                    "D": float(d),
                    "eta": float(eta),
                    "winner_baseline": w_b,
                    "winner_direct": w_d,
                    "winner_mismatch": int(w_b != w_d),
                    "R3_baseline": r3_b,
                    "R3_direct": r3_d,
                    "delta_R3": r3_d - r3_b,
                    "abs_delta_R3": abs(r3_d - r3_b),
                    "mu_mumu_baseline": mu_b,
                    "mu_mumu_direct": mu_d,
                    "delta_mu_mumu": mu_d - mu_b,
                    "abs_delta_mu_mumu": abs(mu_d - mu_b),
                    "chi2_mumu_baseline": c2_b,
                    "chi2_mumu_direct": c2_d,
                    "delta_chi2_mumu": c2_d - c2_b,
                    "abs_delta_chi2_mumu": abs(c2_d - c2_b),
                    "underflow_fallback_baseline": int(base["underflow_fallback"][i, j]),
                    "underflow_fallback_direct": int(direct["underflow_fallback"][i, j]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def summarize_bias(df: pd.DataFrame, d_num: int, eta_num: int) -> pd.DataFrame:
    def q(col: str, p: float) -> float:
        return float(np.quantile(np.asarray(df[col], dtype=float), p))

    f_r3_base = float(np.mean(np.asarray(df["R3_baseline"], dtype=float) >= 0.90))
    f_r3_direct = float(np.mean(np.asarray(df["R3_direct"], dtype=float) >= 0.90))
    f_chi2_base = float(np.mean(np.asarray(df["chi2_mumu_baseline"], dtype=float) <= 4.0))
    f_chi2_direct = float(np.mean(np.asarray(df["chi2_mumu_direct"], dtype=float) <= 4.0))
    mask_accept = (np.asarray(df["chi2_mumu_baseline"], dtype=float) <= 4.0) | (
        np.asarray(df["chi2_mumu_direct"], dtype=float) <= 4.0
    )
    if np.any(mask_accept):
        max_mu_accept = float(np.max(np.asarray(df.loc[mask_accept, "abs_delta_mu_mumu"], dtype=float)))
    else:
        max_mu_accept = float("nan")

    out = pd.DataFrame(
        [
            {
                "n_points": int(len(df)),
                "d_num": int(d_num),
                "eta_num": int(eta_num),
                "frac_winner_mismatch": float(np.mean(np.asarray(df["winner_mismatch"], dtype=float))),
                "n_winner_mismatch": int(np.sum(np.asarray(df["winner_mismatch"], dtype=int))),
                "max_abs_delta_R3": float(np.max(np.asarray(df["abs_delta_R3"], dtype=float))),
                "p99_abs_delta_R3": q("abs_delta_R3", 0.99),
                "mean_abs_delta_R3": float(np.mean(np.asarray(df["abs_delta_R3"], dtype=float))),
                "max_abs_delta_mu_mumu": float(np.max(np.asarray(df["abs_delta_mu_mumu"], dtype=float))),
                "p99_abs_delta_mu_mumu": q("abs_delta_mu_mumu", 0.99),
                "mean_abs_delta_mu_mumu": float(np.mean(np.asarray(df["abs_delta_mu_mumu"], dtype=float))),
                "max_abs_delta_chi2_mumu": float(np.max(np.asarray(df["abs_delta_chi2_mumu"], dtype=float))),
                "p99_abs_delta_chi2_mumu": q("abs_delta_chi2_mumu", 0.99),
                "f_R3_gt_0p90_baseline": f_r3_base,
                "f_R3_gt_0p90_direct": f_r3_direct,
                "delta_f_R3_gt_0p90": f_r3_direct - f_r3_base,
                "f_chi2_mumu_le_4_baseline": f_chi2_base,
                "f_chi2_mumu_le_4_direct": f_chi2_direct,
                "delta_f_chi2_mumu_le_4": f_chi2_direct - f_chi2_base,
                "best_chi2_mumu_baseline": float(np.min(np.asarray(df["chi2_mumu_baseline"], dtype=float))),
                "best_chi2_mumu_direct": float(np.min(np.asarray(df["chi2_mumu_direct"], dtype=float))),
                "frac_underflow_fallback_baseline": float(
                    np.mean(np.asarray(df["underflow_fallback_baseline"], dtype=float))
                ),
                "frac_underflow_fallback_direct": float(
                    np.mean(np.asarray(df["underflow_fallback_direct"], dtype=float))
                ),
                "max_abs_delta_mu_mumu_accept_band": max_mu_accept,
                "bias_upper_bound_R3_abs": float(np.max(np.asarray(df["abs_delta_R3"], dtype=float))),
                "bias_upper_bound_mu_abs": float(np.max(np.asarray(df["abs_delta_mu_mumu"], dtype=float))),
            }
        ]
    )
    return out


def plot_bias_heatmaps(
    out_png: Path,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    base: Dict[str, np.ndarray],
    direct: Dict[str, np.ndarray],
) -> None:
    abs_dr3 = np.abs(direct["R3"] - base["R3"])
    abs_dmu = np.abs(direct["mu_mumu"] - base["mu_mumu"])
    mismatch = (direct["winner"] != base["winner"]).astype(float)

    extent = [float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4), constrained_layout=True)

    im0 = axes[0].imshow(abs_dr3, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[0].set_title(r"$|\Delta \mathcal{R}_3|$")
    axes[0].set_xlabel("D")
    axes[0].set_ylabel(r"$\eta$")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    im1 = axes[1].imshow(abs_dmu, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[1].set_title(r"$|\Delta \mu_{\mu\mu}|$")
    axes[1].set_xlabel("D")
    axes[1].set_ylabel(r"$\eta$")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    im2 = axes[2].imshow(mismatch, origin="lower", aspect="auto", extent=extent, cmap="gray_r", vmin=0.0, vmax=1.0)
    axes[2].set_title("Winner mismatch (0/1)")
    axes[2].set_xlabel("D")
    axes[2].set_ylabel(r"$\eta$")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)

    fig.suptitle("Localized Direct Chain Bias Audit on Small Surface", fontsize=12.5)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Localized direct-chain small-surface bias audit.")
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=20.0)
    ap.add_argument("--d-num", type=int, default=11, help="Small sampled D-grid size.")
    ap.add_argument("--eta-min", type=float, default=0.2)
    ap.add_argument("--eta-max", type=float, default=4.0)
    ap.add_argument("--eta-num", type=int, default=21, help="Small sampled eta-grid size.")
    ap.add_argument("--chi-rho-max", type=float, default=3.0)
    ap.add_argument("--chi-z-margin", type=float, default=6.0)
    ap.add_argument("--chi-n-mu", type=int, default=120)
    ap.add_argument("--chi-tol", type=float, default=1e-8)
    ap.add_argument("--chi-maxiter", type=int, default=30000)
    ap.add_argument("--chi-sigma", type=float, default=2.5)
    ap.add_argument("--superrad-zmax", type=float, default=80.0)
    ap.add_argument("--superrad-d-ref", type=float, default=12.0)
    ap.add_argument("--superrad-n-ref", type=int, default=2)
    ap.add_argument("--gamma-t-floor", type=float, default=1e-18, help="Floor for Gamma*t in underflow guard.")
    ap.add_argument(
        "--prob-total-floor",
        type=float,
        default=1e-280,
        help="If total layer weight <= floor, use fallback normalization.",
    )
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_vals = scan_d_values(args.d_min, args.d_max, args.d_num)
    eta_vals = np.linspace(float(args.eta_min), float(args.eta_max), int(args.eta_num))
    tag = f"Dgrid{len(d_vals)}_Egrid{len(eta_vals)}"

    print("[grid]", f"D:[{args.d_min},{args.d_max}] N={len(d_vals)}", f"eta:[{args.eta_min},{args.eta_max}] N={len(eta_vals)}")

    chi_df = extract_direct_chi_profile(
        d_vals=d_vals,
        rho_max=float(args.chi_rho_max),
        z_margin=float(args.chi_z_margin),
        n_mu=int(args.chi_n_mu),
        tol=float(args.chi_tol),
        maxiter=int(args.chi_maxiter),
        sigma=float(args.chi_sigma),
    )
    superrad_df = extract_direct_superrad_profile(
        d_vals=d_vals,
        zmax=float(args.superrad_zmax),
        d_ref=float(args.superrad_d_ref),
        n_ref=int(args.superrad_n_ref),
    )

    chi_csv = OUTDIR / f"localized_direct_profiles_chi_Dgrid{len(d_vals)}.csv"
    sup_csv = OUTDIR / f"localized_direct_profiles_superrad_Dgrid{len(d_vals)}.csv"
    chi_df.to_csv(chi_csv, index=False)
    superrad_df.to_csv(sup_csv, index=False)
    print(f"[saved] {chi_csv}")
    print(f"[saved] {sup_csv}")

    print(f"[baseline chi] {BASELINE_CHI_CSV}")
    print(f"[baseline superrad] {BASELINE_SUPERRAD_CSV}")
    kin_baseline = make_kinetics_baseline()
    kin_direct = make_kinetics_direct(d_vals=d_vals, chi_df=chi_df, superrad_csv=sup_csv)

    hll_cfg = HLLObservableConfig(
        mode=str(CFG.hll_observable_mode),
        t_coh=float(CFG.t_coh),
        ref_D=float(CFG.hmumu_ref_D),
        ref_eta=float(CFG.hmumu_ref_eta),
        n_max=int(CFG.hll_observable_nmax),
    )

    base_map = evaluate_map(
        kin_baseline,
        d_vals=d_vals,
        eta_vals=eta_vals,
        hll_cfg=hll_cfg,
        gamma_t_floor=float(args.gamma_t_floor),
        prob_total_floor=float(args.prob_total_floor),
    )
    direct_map = evaluate_map(
        kin_direct,
        d_vals=d_vals,
        eta_vals=eta_vals,
        hll_cfg=hll_cfg,
        gamma_t_floor=float(args.gamma_t_floor),
        prob_total_floor=float(args.prob_total_floor),
    )

    points_csv = OUTDIR / f"localized_direct_surface_points_{tag}.csv"
    df_points = save_point_table(points_csv, d_vals=d_vals, eta_vals=eta_vals, base=base_map, direct=direct_map)
    summary_df = summarize_bias(df_points, d_num=len(d_vals), eta_num=len(eta_vals))
    summary_csv = OUTDIR / f"localized_direct_surface_summary_{tag}.csv"
    summary_df.to_csv(summary_csv, index=False)

    fig_png = OUTDIR / f"localized_direct_surface_bias_{tag}.png"
    plot_bias_heatmaps(fig_png, d_vals=d_vals, eta_vals=eta_vals, base=base_map, direct=direct_map)

    paper_summary = PAPER_DIR / summary_csv.name
    paper_fig = PAPER_DIR / fig_png.name
    paper_summary.write_text(summary_csv.read_text())
    paper_fig.write_bytes(fig_png.read_bytes())

    print(f"[saved] {points_csv}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {fig_png}")
    print(f"[saved] {paper_summary}")
    print(f"[saved] {paper_fig}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

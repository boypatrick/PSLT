#!/usr/bin/env python3
"""
Bath-side factorization audit for the microscopic open-system chi witness.

This script takes the existing bridge map and rewrites the open-system chain
into explicit blocks:

  system block         : g_z^2, g_x^2
  bath-shape block     : Szz(0), Sxx(DeltaE)
  bath normalization   : kappa_env
  response block       : Cmax

The rates obey

  gamma_phi = kappa_env * g_z^2 * Szz(0)
  gamma_mix = kappa_env * g_x^2 * Sxx(DeltaE)

and the observable obeys

  chi_eff = kappa_env * [2 g_x^2 Sxx(DeltaE) / gamma_ref] * Cmax.

The audit exports map-level factor columns and a compact summary that shows
whether kappa_env behaves as a pure amplitude / normalization block or is
implicitly compensating for bath-shape distortions.
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
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


def q(v: np.ndarray | pd.Series, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


def corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Bath-side factorization audit for open_system_micro.")
    ap.add_argument(
        "--bridge-map",
        type=Path,
        default=OUTDIR / "chi_open_system_micro_bridge_map.csv",
        help="Bridge-map CSV from scan_chi_open_system_micro_bridge.py",
    )
    ap.add_argument(
        "--calib-csv",
        type=Path,
        default=OUTDIR / "kappa_env_anchor_calibration.csv",
    )
    ap.add_argument(
        "--holdout-csv",
        type=Path,
        default=OUTDIR / "kappa_env_anchor_holdout.csv",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.bridge_map.exists():
        raise FileNotFoundError(args.bridge_map)

    df = pd.read_csv(args.bridge_map).copy()
    if df.empty:
        raise RuntimeError(f"Empty bridge map: {args.bridge_map}")

    tag = args.tag.strip() or ""
    suffix = f"_{tag}" if tag else ""

    for col in [
        "D",
        "kappa_env",
        "g_z_micro",
        "g_x_micro",
        "Szz_0",
        "Sxx_delta",
        "gamma_phi_micro",
        "gamma_mix_micro",
        "Cmax_micro",
        "gamma_ref",
        "chi_eff_micro",
        "ratio_micro_to_baseline",
    ]:
        if col not in df.columns:
            raise KeyError(f"Required column missing in bridge map: {col}")

    gz2 = np.square(np.maximum(df["g_z_micro"].to_numpy(dtype=float), 0.0))
    gx2 = np.square(np.maximum(df["g_x_micro"].to_numpy(dtype=float), 0.0))
    szz = np.maximum(df["Szz_0"].to_numpy(dtype=float), 0.0)
    sxx = np.maximum(df["Sxx_delta"].to_numpy(dtype=float), 0.0)
    gamma_phi = np.maximum(df["gamma_phi_micro"].to_numpy(dtype=float), 0.0)
    gamma_mix = np.maximum(df["gamma_mix_micro"].to_numpy(dtype=float), 0.0)
    kappa = np.maximum(df["kappa_env"].to_numpy(dtype=float), 1e-30)
    gamma_ref = np.maximum(df["gamma_ref"].to_numpy(dtype=float), 1e-30)
    cmax = np.maximum(df["Cmax_micro"].to_numpy(dtype=float), 0.0)
    chi_eff = np.maximum(df["chi_eff_micro"].to_numpy(dtype=float), 0.0)

    system_kernel_phi = gz2
    system_kernel_mix = gx2
    bath_shape_phi = szz
    bath_shape_mix = sxx
    bath_shape_suppression = sxx / np.maximum(szz, 1e-30)
    system_coupling_ratio = gx2 / np.maximum(gz2, 1e-30)

    rate_kernel_phi = system_kernel_phi * bath_shape_phi
    rate_kernel_mix = system_kernel_mix * bath_shape_mix

    bath_norm_from_phi = gamma_phi / np.maximum(rate_kernel_phi, 1e-30)
    bath_norm_from_mix = gamma_mix / np.maximum(rate_kernel_mix, 1e-30)
    bath_norm_geom_mean = np.sqrt(np.maximum(bath_norm_from_phi * bath_norm_from_mix, 0.0))

    rate_ratio_pred = system_coupling_ratio * bath_shape_suppression
    rate_ratio_obs = gamma_mix / np.maximum(gamma_phi, 1e-30)
    rate_ratio_log_residual_abs = np.abs(
        np.log(np.maximum(rate_ratio_obs, 1e-30) / np.maximum(rate_ratio_pred, 1e-30))
    )

    # When the sigma_z channel is effectively shut off, gamma_phi no longer gives
    # a stable absolute-normalization estimate even if the forward factorization
    # itself remains exact. We therefore report phi-derived normalization on the
    # subset where the diagonal channel is not negligible relative to the mixing
    # channel.
    phi_identifiable = (rate_kernel_phi / np.maximum(rate_kernel_mix, 1e-30)) > 1e-6

    chi_linear_prefactor = 2.0 * rate_kernel_mix / np.maximum(gamma_ref, 1e-30)
    chi_response_gain = cmax
    chi_eff_factor_recon = kappa * chi_linear_prefactor * chi_response_gain
    bath_norm_from_obs = chi_eff / np.maximum(chi_linear_prefactor * chi_response_gain, 1e-30)

    df["bath_system_kernel_phi"] = system_kernel_phi
    df["bath_system_kernel_mix"] = system_kernel_mix
    df["bath_shape_phi"] = bath_shape_phi
    df["bath_shape_mix"] = bath_shape_mix
    df["bath_shape_suppression"] = bath_shape_suppression
    df["bath_system_coupling_ratio"] = system_coupling_ratio
    df["bath_rate_kernel_phi"] = rate_kernel_phi
    df["bath_rate_kernel_mix"] = rate_kernel_mix
    df["bath_norm_from_phi"] = bath_norm_from_phi
    df["bath_norm_from_mix"] = bath_norm_from_mix
    df["bath_norm_geom_mean"] = bath_norm_geom_mean
    df["bath_norm_phi_residual_abs"] = np.abs(bath_norm_from_phi - kappa)
    df["bath_norm_mix_residual_abs"] = np.abs(bath_norm_from_mix - kappa)
    df["bath_norm_channel_consistency_abs"] = np.abs(bath_norm_from_phi - bath_norm_from_mix)
    df["bath_phi_identifiable"] = phi_identifiable.astype(int)
    df["bath_rate_ratio_pred"] = rate_ratio_pred
    df["bath_rate_ratio_obs"] = rate_ratio_obs
    df["bath_rate_ratio_residual_abs"] = np.abs(rate_ratio_obs - rate_ratio_pred)
    df["bath_rate_ratio_log_residual_abs"] = rate_ratio_log_residual_abs
    df["chi_linear_prefactor"] = chi_linear_prefactor
    df["chi_response_gain"] = chi_response_gain
    df["chi_eff_factor_recon"] = chi_eff_factor_recon
    df["chi_eff_factor_residual_abs"] = np.abs(chi_eff - chi_eff_factor_recon)
    df["bath_norm_from_obs"] = bath_norm_from_obs
    df["bath_norm_obs_residual_abs"] = np.abs(bath_norm_from_obs - kappa)

    log_sys = np.abs(np.log(np.maximum(system_coupling_ratio, 1e-30)))
    log_bath = np.abs(np.log(np.maximum(bath_shape_suppression, 1e-30)))
    denom = np.maximum(log_sys + log_bath, 1e-30)
    df["bath_ratio_system_share"] = log_sys / denom
    df["bath_ratio_shape_share"] = log_bath / denom

    calib = pd.read_csv(args.calib_csv).iloc[0] if args.calib_csv.exists() else None
    holdout = pd.read_csv(args.holdout_csv).iloc[0] if args.holdout_csv.exists() else None

    log_system_ratio = np.log(np.maximum(system_coupling_ratio, 1e-30))
    log_bath_supp = np.log(np.maximum(bath_shape_suppression, 1e-30))
    log_rate_ratio = np.log(np.maximum(rate_ratio_obs, 1e-30))

    phi_norm_resid_ident = (
        df.loc[phi_identifiable, "bath_norm_phi_residual_abs"].to_numpy(dtype=float)
        if np.any(phi_identifiable)
        else np.asarray([], dtype=float)
    )
    phi_channel_consistency_ident = (
        df.loc[phi_identifiable, "bath_norm_channel_consistency_abs"].to_numpy(dtype=float)
        if np.any(phi_identifiable)
        else np.asarray([], dtype=float)
    )
    mix_obs_consistency = np.abs(bath_norm_from_mix - bath_norm_from_obs)

    summary = {
        "bridge_map_csv": str(args.bridge_map),
        "n_points": int(len(df)),
        "kappa_env_p50": q(kappa, 0.5),
        "kappa_env_p90": q(kappa, 0.9),
        "max_abs_gamma_phi_factor_residual": float(np.max(np.abs(gamma_phi - kappa * rate_kernel_phi))),
        "max_abs_gamma_mix_factor_residual": float(np.max(np.abs(gamma_mix - kappa * rate_kernel_mix))),
        "n_phi_identifiable": int(np.sum(phi_identifiable)),
        "max_abs_bath_norm_phi_residual_identifiable": float(np.max(phi_norm_resid_ident))
        if phi_norm_resid_ident.size
        else float("nan"),
        "max_abs_bath_norm_mix_residual": float(df["bath_norm_mix_residual_abs"].max()),
        "max_abs_bath_norm_obs_residual": float(df["bath_norm_obs_residual_abs"].max()),
        "max_abs_bath_norm_channel_consistency": float(df["bath_norm_channel_consistency_abs"].max()),
        "max_abs_bath_norm_channel_consistency_identifiable": float(np.max(phi_channel_consistency_ident))
        if phi_channel_consistency_ident.size
        else float("nan"),
        "max_abs_bath_norm_mix_obs_consistency": float(np.max(mix_obs_consistency)),
        "max_abs_rate_ratio_log_residual": float(df["bath_rate_ratio_log_residual_abs"].max()),
        "p95_abs_rate_ratio_log_residual": q(df["bath_rate_ratio_log_residual_abs"], 0.95),
        "max_abs_chi_factor_residual": float(df["chi_eff_factor_residual_abs"].max()),
        "system_coupling_ratio_p50": q(system_coupling_ratio, 0.5),
        "system_coupling_ratio_p90": q(system_coupling_ratio, 0.9),
        "bath_shape_suppression_p50": q(bath_shape_suppression, 0.5),
        "bath_shape_suppression_p90": q(bath_shape_suppression, 0.9),
        "rate_ratio_obs_p50": q(rate_ratio_obs, 0.5),
        "rate_ratio_obs_p90": q(rate_ratio_obs, 0.9),
        "chi_response_gain_p50": q(chi_response_gain, 0.5),
        "chi_response_gain_p90": q(chi_response_gain, 0.9),
        "bath_ratio_system_share_p50": q(df["bath_ratio_system_share"], 0.5),
        "bath_ratio_shape_share_p50": q(df["bath_ratio_shape_share"], 0.5),
        "corr(log_system_coupling_ratio, log_rate_ratio_obs)": corr(log_system_ratio, log_rate_ratio),
        "corr(log_bath_shape_suppression, log_rate_ratio_obs)": corr(log_bath_supp, log_rate_ratio),
        "corr(chi_linear_prefactor, chi_eff_micro)": corr(chi_linear_prefactor, chi_eff),
        "corr(chi_response_gain, chi_eff_micro)": corr(chi_response_gain, chi_eff),
    }

    if calib is not None:
        summary["kappa_env_calibrated"] = float(calib["kappa_env_calibrated"])
        summary["anchor_rmse"] = float(calib["anchor_rmse"])
        summary["holdout_rmse"] = float(calib["holdout_rmse"])
        summary["holdout_max_abs_err"] = float(calib["holdout_max_abs_err"])
        summary["anchor_mode"] = str(calib["anchor_mode"]) if "anchor_mode" in calib.index else ""
        summary["n_anchor"] = int(calib["n_anchor"]) if "n_anchor" in calib.index else 0
        summary["n_holdout"] = int(calib["n_holdout"]) if "n_holdout" in calib.index else 0
    if holdout is not None:
        summary["holdout_rmse_csv"] = float(holdout["holdout_rmse"])
        summary["holdout_max_abs_err_csv"] = float(holdout["holdout_max_abs_err"])

    out_map = OUTDIR / f"chi_open_system_bath_factorization_map{suffix}.csv"
    out_summary = OUTDIR / f"chi_open_system_bath_factorization_summary{suffix}.csv"
    out_png = OUTDIR / f"chi_open_system_bath_factorization{suffix}.png"
    out_meta = OUTDIR / f"chi_open_system_bath_factorization_run_meta{suffix}.json"

    keep_cols = [
        "D",
        "delta_micro",
        "kappa_env",
        "g_z_micro",
        "g_x_micro",
        "tau_env",
        "Szz_0",
        "Sxx_delta",
        "gamma_phi_micro",
        "gamma_mix_micro",
        "gamma_ref",
        "Cmax_micro",
        "chi_eff_micro",
        "ratio_micro_to_baseline",
        "bath_system_kernel_phi",
        "bath_system_kernel_mix",
        "bath_shape_phi",
        "bath_shape_mix",
        "bath_shape_suppression",
        "bath_system_coupling_ratio",
        "bath_rate_kernel_phi",
        "bath_rate_kernel_mix",
        "bath_norm_from_phi",
        "bath_norm_from_mix",
        "bath_norm_geom_mean",
        "bath_norm_phi_residual_abs",
        "bath_norm_mix_residual_abs",
        "bath_norm_channel_consistency_abs",
        "bath_phi_identifiable",
        "bath_rate_ratio_pred",
        "bath_rate_ratio_obs",
        "bath_rate_ratio_residual_abs",
        "bath_rate_ratio_log_residual_abs",
        "chi_linear_prefactor",
        "chi_response_gain",
        "chi_eff_factor_recon",
        "chi_eff_factor_residual_abs",
        "bath_norm_from_obs",
        "bath_norm_obs_residual_abs",
        "bath_ratio_system_share",
        "bath_ratio_shape_share",
    ]
    df[keep_cols].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "bridge_map": str(args.bridge_map),
                "calib_csv": str(args.calib_csv),
                "holdout_csv": str(args.holdout_csv),
                "tag": tag,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(df["D"], df["bath_system_kernel_phi"], marker="o", label=r"$g_z^2$")
    ax.plot(df["D"], df["bath_system_kernel_mix"], marker="s", label=r"$g_x^2$")
    ax.set_title("System block")
    ax.set_xlabel("D")
    ax.set_ylabel("kernel")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(df["D"], df["bath_shape_phi"], marker="o", label=r"$S_{zz}(0)$")
    ax.plot(df["D"], df["bath_shape_mix"], marker="s", label=r"$S_{xx}(\Delta E)$")
    ax.set_title("Bath-shape block")
    ax.set_xlabel("D")
    ax.set_ylabel("PSD factor")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(df["D"], df["bath_norm_from_phi"], marker="o", label=r"$\kappa_\phi$")
    ax.plot(df["D"], df["bath_norm_from_mix"], marker="s", label=r"$\kappa_{\rm mix}$")
    ax.plot(df["D"], df["bath_norm_from_obs"], marker="^", label=r"$\kappa_{\chi}$")
    ax.set_title("Bath-normalization block")
    ax.set_xlabel("D")
    ax.set_ylabel(r"inferred $\kappa_{\rm env}$")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(df["D"], df["bath_rate_ratio_obs"], marker="o", label=r"$\gamma_{\rm mix}/\gamma_\phi$")
    ax.plot(df["D"], df["bath_rate_ratio_pred"], marker="s", ls="--", label="system x shape")
    ax2 = ax.twinx()
    ax2.plot(df["D"], df["chi_eff_micro"], color="#d62728", marker="^", label=r"$\chi_{\rm eff}$")
    ax.set_title("Ratio / observable closure")
    ax.set_xlabel("D")
    ax.set_ylabel("rate ratio")
    ax2.set_ylabel(r"$\chi_{\rm eff}$")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")

    fig.suptitle("Microscopic open-system bath factorization audit")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    for src in [out_map, out_summary, out_png, out_meta]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

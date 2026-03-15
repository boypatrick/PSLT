#!/usr/bin/env python3
"""
Projected parent-bath statement audit for open_system_micro.

This script rewrites the microscopic open-system chain as a minimal projected
system-bath block:

  K_sys   = diag(g_z^2, g_x^2)
  K_spec  = diag(Szz(0), Sxx(DeltaE))
  K_bath  = kappa_env * sqrt(K_sys) K_spec sqrt(K_sys)

so that

  gamma_phi = (K_bath)_{11}
  gamma_mix = (K_bath)_{22}
  chi_eff   = [2 Cmax / gamma_ref] * (K_bath)_{22}.

It also checks the additive anisotropy statement

  log(gamma_mix/gamma_phi) = log(g_x^2/g_z^2) + log(Sxx/Szz)

on the phi-identifiable subset.
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
    ap = argparse.ArgumentParser(description="Projected parent-bath statement audit for open_system_micro.")
    ap.add_argument(
        "--factor-map",
        type=Path,
        default=OUTDIR / "chi_open_system_bath_factorization_map.csv",
        help="Map from scan_chi_open_system_bath_factorization.py",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.factor_map.exists():
        raise FileNotFoundError(args.factor_map)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.factor_map).copy()
    if df.empty:
        raise RuntimeError(f"Empty factorization map: {args.factor_map}")

    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""

    required = [
        "D",
        "kappa_env",
        "bath_system_kernel_phi",
        "bath_system_kernel_mix",
        "bath_shape_phi",
        "bath_shape_mix",
        "gamma_phi_micro",
        "gamma_mix_micro",
        "gamma_ref",
        "chi_response_gain",
        "chi_eff_micro",
        "bath_phi_identifiable",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    kappa = np.maximum(df["kappa_env"].to_numpy(dtype=float), 1e-30)
    sys_phi = np.maximum(df["bath_system_kernel_phi"].to_numpy(dtype=float), 0.0)
    sys_mix = np.maximum(df["bath_system_kernel_mix"].to_numpy(dtype=float), 0.0)
    spec_phi = np.maximum(df["bath_shape_phi"].to_numpy(dtype=float), 0.0)
    spec_mix = np.maximum(df["bath_shape_mix"].to_numpy(dtype=float), 0.0)
    gamma_phi = np.maximum(df["gamma_phi_micro"].to_numpy(dtype=float), 0.0)
    gamma_mix = np.maximum(df["gamma_mix_micro"].to_numpy(dtype=float), 0.0)
    gamma_ref = np.maximum(df["gamma_ref"].to_numpy(dtype=float), 1e-30)
    cmax = np.maximum(df["chi_response_gain"].to_numpy(dtype=float), 0.0)
    chi_eff = np.maximum(df["chi_eff_micro"].to_numpy(dtype=float), 0.0)
    phi_identifiable = df["bath_phi_identifiable"].to_numpy(dtype=int) == 1

    kbath_11 = kappa * sys_phi * spec_phi
    kbath_22 = kappa * sys_mix * spec_mix
    kbath_trace = kbath_11 + kbath_22
    kbath_det = kbath_11 * kbath_22

    gamma_trace = gamma_phi + gamma_mix
    gamma_det = gamma_phi * gamma_mix
    chi_eff_recon = (2.0 * cmax / gamma_ref) * kbath_22

    log_sys_aniso = np.log(np.maximum(sys_mix / np.maximum(sys_phi, 1e-30), 1e-30))
    log_bath_aniso = np.log(np.maximum(spec_mix / np.maximum(spec_phi, 1e-30), 1e-30))
    log_rate_aniso = np.log(np.maximum(gamma_mix / np.maximum(gamma_phi, 1e-30), 1e-30))
    log_rate_aniso_recon = log_sys_aniso + log_bath_aniso

    df["parent_bath_block_11"] = kbath_11
    df["parent_bath_block_22"] = kbath_22
    df["parent_bath_trace"] = kbath_trace
    df["parent_bath_det"] = kbath_det
    df["parent_gamma_trace"] = gamma_trace
    df["parent_gamma_det"] = gamma_det
    df["parent_bath_trace_residual_abs"] = np.abs(kbath_trace - gamma_trace)
    df["parent_bath_det_residual_abs"] = np.abs(kbath_det - gamma_det)
    df["parent_bath_chi_recon"] = chi_eff_recon
    df["parent_bath_chi_residual_abs"] = np.abs(chi_eff_recon - chi_eff)
    df["parent_log_sys_anisotropy"] = log_sys_aniso
    df["parent_log_bath_anisotropy"] = log_bath_aniso
    df["parent_log_rate_anisotropy"] = log_rate_aniso
    df["parent_log_rate_anisotropy_recon"] = log_rate_aniso_recon
    df["parent_log_rate_anisotropy_residual_abs"] = np.abs(log_rate_aniso - log_rate_aniso_recon)
    df["parent_trace_mix_fraction"] = kbath_22 / np.maximum(kbath_trace, 1e-30)

    ident_log_resid = (
        df.loc[phi_identifiable, "parent_log_rate_anisotropy_residual_abs"].to_numpy(dtype=float)
        if np.any(phi_identifiable)
        else np.asarray([], dtype=float)
    )

    summary = {
        "factor_map_csv": str(args.factor_map),
        "n_points": int(len(df)),
        "n_phi_identifiable": int(np.sum(phi_identifiable)),
        "kappa_env_p50": q(kappa, 0.5),
        "parent_trace_mix_fraction_p50": q(df["parent_trace_mix_fraction"], 0.5),
        "parent_trace_mix_fraction_p90": q(df["parent_trace_mix_fraction"], 0.9),
        "max_abs_parent_block11_residual": float(np.max(np.abs(kbath_11 - gamma_phi))),
        "max_abs_parent_block22_residual": float(np.max(np.abs(kbath_22 - gamma_mix))),
        "max_abs_parent_trace_residual": float(df["parent_bath_trace_residual_abs"].max()),
        "max_abs_parent_det_residual": float(df["parent_bath_det_residual_abs"].max()),
        "max_abs_parent_chi_residual": float(df["parent_bath_chi_residual_abs"].max()),
        "max_abs_parent_log_rate_anisotropy_residual_identifiable": float(np.max(ident_log_resid))
        if ident_log_resid.size
        else float("nan"),
        "p95_abs_parent_log_rate_anisotropy_residual_identifiable": q(ident_log_resid, 0.95)
        if ident_log_resid.size
        else float("nan"),
        "corr(log_sys_anisotropy, log_rate_anisotropy)_identifiable": corr(
            df.loc[phi_identifiable, "parent_log_sys_anisotropy"],
            df.loc[phi_identifiable, "parent_log_rate_anisotropy"],
        ),
        "corr(log_bath_anisotropy, log_rate_anisotropy)_identifiable": corr(
            df.loc[phi_identifiable, "parent_log_bath_anisotropy"],
            df.loc[phi_identifiable, "parent_log_rate_anisotropy"],
        ),
        "corr(parent_trace, gamma_trace)": corr(df["parent_bath_trace"], df["parent_gamma_trace"]),
        "corr(parent_det, gamma_det)": corr(df["parent_bath_det"], df["parent_gamma_det"]),
        "corr(parent_block22, chi_eff_micro)": corr(df["parent_bath_block_22"], chi_eff),
        "corr(parent_bath_chi_recon, chi_eff_micro)": corr(df["parent_bath_chi_recon"], chi_eff),
    }

    out_map = OUTDIR / f"chi_open_system_parent_bath_statement_map{suffix}.csv"
    out_summary = OUTDIR / f"chi_open_system_parent_bath_statement_summary{suffix}.csv"
    out_png = OUTDIR / f"chi_open_system_parent_bath_statement{suffix}.png"
    out_meta = OUTDIR / f"chi_open_system_parent_bath_statement_run_meta{suffix}.json"

    df.to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(df["D"], gamma_phi, marker="o", label=r"$\gamma_\phi$")
    ax.plot(df["D"], kbath_11, marker="x", linestyle="--", label="parent block (11)")
    ax.plot(df["D"], gamma_mix, marker="s", label=r"$\gamma_{\rm mix}$")
    ax.plot(df["D"], kbath_22, marker="+", linestyle="--", label="parent block (22)")
    ax.set_title("Projected parent-bath diagonal block")
    ax.set_xlabel("D")
    ax.set_ylabel("rate")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(df["D"], df["parent_gamma_trace"], marker="o", label="gamma trace")
    ax.plot(df["D"], df["parent_bath_trace"], marker="x", linestyle="--", label="parent trace")
    ax.plot(df["D"], df["parent_gamma_det"], marker="s", label="gamma determinant")
    ax.plot(df["D"], df["parent_bath_det"], marker="+", linestyle="--", label="parent determinant")
    ax.set_title("Trace / determinant closure")
    ax.set_xlabel("D")
    ax.set_ylabel("value")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(df["D"], df["parent_log_sys_anisotropy"], marker="o", label="system anisotropy")
    ax.plot(df["D"], df["parent_log_bath_anisotropy"], marker="s", label="bath anisotropy")
    ax.plot(df["D"], df["parent_log_rate_anisotropy"], marker="^", label="rate anisotropy")
    ax.plot(df["D"], df["parent_log_rate_anisotropy_recon"], linestyle="--", color="k", label="system+bath")
    ax.set_title("Additive anisotropy statement")
    ax.set_xlabel("D")
    ax.set_ylabel("log ratio")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.plot(df["D"], chi_eff, marker="o", label=r"$\chi_{\rm eff}^{\rm micro}$")
    ax.plot(df["D"], chi_eff_recon, marker="x", linestyle="--", label="parent-bath recon")
    ax2 = ax.twinx()
    ax2.plot(df["D"], df["parent_bath_chi_residual_abs"], color="#d62728", linestyle=":", label="abs residual")
    ax.set_title("Observable closure")
    ax.set_xlabel("D")
    ax.set_ylabel("chi")
    ax2.set_ylabel("abs residual")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="best")

    fig.suptitle("Open-system parent bath statement", y=0.99)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    meta = {
        "factor_map_csv": str(args.factor_map),
        "statement": {
            "K_sys": "diag(g_z^2, g_x^2)",
            "K_spec": "diag(Szz_0, Sxx_delta)",
            "K_bath": "kappa_env * sqrt(K_sys) K_spec sqrt(K_sys)",
            "observable": "chi_eff = (2 Cmax / gamma_ref) * (K_bath)22",
        },
        "notes": [
            "Anisotropy statement is evaluated on the phi-identifiable subset only.",
            "The parent bath block is diagonal in the localized sigma_z / sigma_x basis used by the micro extraction.",
        ],
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    for src in [out_map, out_summary, out_png, out_meta]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

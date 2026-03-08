#!/usr/bin/env python3
"""
Bridge audit for the microscopic open-system chi witness.

This script does not regenerate the localized modes. Instead, it consumes the
canonical `chi_open_system_micro_D*.csv` export and reconstructs the full
system/bath/coupling chain:

  system     : delta_micro
  couplings  : g_z_micro, g_x_micro
  bath       : tau_env, Szz_0, Sxx_delta
  rates      : gamma_phi_micro, gamma_mix_micro
  observable : chi_eff_micro

It also checks that `pslt_lib.py` reproduces the same chi(D) values when the
same CSV profile is loaded through `chi_mode="open_system_micro"`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from pslt_lib import PSLTParameters, PSLTKinetics


OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge audit for microscopic open-system chi.")
    ap.add_argument("--micro-csv", default="", help="Path to chi_open_system_micro_D*.csv (default: auto-pick largest).")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--skip-paper-copy", action="store_true")
    return ap.parse_args()


def pick_micro_csv(path_arg: str) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    cands = sorted(OUTDIR.glob("chi_open_system_micro_D*.csv"))
    if not cands:
        raise FileNotFoundError("No chi_open_system_micro_D*.csv found in output/chi_open_system.")
    best: Optional[Path] = None
    best_n = -1
    for p in cands:
        try:
            n = len(pd.read_csv(p))
        except Exception:
            continue
        if n > best_n:
            best_n = n
            best = p
    if best is None:
        raise RuntimeError("Could not parse any chi_open_system_micro CSV.")
    return best


def make_kinetics(micro_csv: Path, nstep: int) -> PSLTKinetics:
    params = PSLTParameters(
        chi_mode="open_system_micro",
        chi_open_csv=str(micro_csv),
        chi_open_phi_scale=1.0,
        chi_open_mix_scale=1.0,
        chi_open_tmax=200.0,
        chi_open_nstep=int(nstep),
        chi_open_rtol=1e-8,
        chi_open_atol=1e-10,
    )
    return PSLTKinetics(params)


def plot_bridge(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(df["D"], df["delta_micro"], marker="o", label=r"$\Delta E$")
    ax.set_title("System splitting")
    ax.set_xlabel("D")
    ax.set_ylabel("value")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(df["D"], df["g_z_micro"], marker="o", label=r"$g_z$")
    ax.plot(df["D"], df["g_x_micro"], marker="s", label=r"$g_x$")
    ax.set_title("Localized-basis couplings")
    ax.set_xlabel("D")
    ax.set_ylabel("value")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(df["D"], df["gamma_phi_micro"], marker="o", label=r"$\gamma_\phi$")
    ax.plot(df["D"], df["gamma_mix_micro"], marker="s", label=r"$\gamma_{\rm mix}$")
    ax.set_title("Born-Markov rates")
    ax.set_xlabel("D")
    ax.set_ylabel("value")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(df["D"], df["ratio_micro_to_baseline"], marker="o", label=r"$\chi_{\rm open}^{\rm micro}/\chi_{LR}$")
    ax2 = ax.twinx()
    ax2.plot(df["D"], df["chi_eff_loader_residual_abs"], color="#d62728", ls="--", label="loader residual")
    ax.set_title("Observable bridge")
    ax.set_xlabel("D")
    ax.set_ylabel("ratio")
    ax2.set_ylabel("abs residual")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")

    fig.suptitle("Microscopic open-system bridge audit")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    micro_csv = pick_micro_csv(args.micro_csv)
    df = pd.read_csv(micro_csv).copy()
    if df.empty:
        raise RuntimeError(f"Empty micro CSV: {micro_csv}")

    nstep = 1200 if "Cmax_micro" in df.columns else 400
    kin = make_kinetics(micro_csv, nstep=nstep)

    # Reconstruct the bridge from explicit formulas.
    df["gamma_phi_recon"] = df["kappa_env"] * (df["g_z_micro"] ** 2) * df["Szz_0"]
    df["gamma_mix_recon"] = df["kappa_env"] * (df["g_x_micro"] ** 2) * df["Sxx_delta"]
    df["chi_eff_formula_recon"] = 2.0 * df["gamma_mix_micro"] * df["Cmax_micro"] / df["gamma_ref"].clip(lower=1e-30)
    df["gamma_phi_recon_residual_abs"] = (df["gamma_phi_micro"] - df["gamma_phi_recon"]).abs()
    df["gamma_mix_recon_residual_abs"] = (df["gamma_mix_micro"] - df["gamma_mix_recon"]).abs()
    df["chi_eff_formula_residual_abs"] = (df["chi_eff_micro"] - df["chi_eff_formula_recon"]).abs()

    loader_vals = []
    for d in df["D"].to_numpy(dtype=float):
        loader_vals.append(float(kin.chi_effective(float(d))))
    df["chi_eff_loader_recon"] = np.asarray(loader_vals, dtype=float)
    df["chi_eff_loader_residual_abs"] = (df["chi_eff_micro"] - df["chi_eff_loader_recon"]).abs()

    out_map = outdir / "chi_open_system_micro_bridge_map.csv"
    out_summary = outdir / "chi_open_system_micro_bridge_summary.csv"
    out_png = outdir / "chi_open_system_micro_bridge.png"
    out_meta = outdir / "chi_open_system_micro_bridge_run_meta.json"

    df.to_csv(out_map, index=False)
    plot_bridge(df, out_png)

    summary = pd.DataFrame(
        [
            {
                "micro_csv": str(micro_csv),
                "n_points": int(len(df)),
                "system_basis": "localized_LR_two_level",
                "bath_model": "gaussian_markov_lorentzian",
                "coupling_ops": "sigma_z,sigma_x",
                "kappa_env": float(df["kappa_env"].iloc[0]),
                "delta_p50": float(df["delta_micro"].quantile(0.5)),
                "delta_p90": float(df["delta_micro"].quantile(0.9)),
                "g_z_p50": float(df["g_z_micro"].quantile(0.5)),
                "g_x_p50": float(df["g_x_micro"].quantile(0.5)),
                "tau_env_p50": float(df["tau_env"].quantile(0.5)),
                "gamma_phi_p50": float(df["gamma_phi_micro"].quantile(0.5)),
                "gamma_mix_p50": float(df["gamma_mix_micro"].quantile(0.5)),
                "ratio_min": float(df["ratio_micro_to_baseline"].min()),
                "ratio_max": float(df["ratio_micro_to_baseline"].max()),
                "ratio_mean": float(df["ratio_micro_to_baseline"].mean()),
                "max_abs_gamma_phi_recon_residual": float(df["gamma_phi_recon_residual_abs"].max()),
                "max_abs_gamma_mix_recon_residual": float(df["gamma_mix_recon_residual_abs"].max()),
                "max_abs_chi_eff_formula_residual": float(df["chi_eff_formula_residual_abs"].max()),
                "max_abs_chi_eff_loader_residual": float(df["chi_eff_loader_residual_abs"].max()),
            }
        ]
    )
    summary.to_csv(out_summary, index=False)

    meta = {
        "micro_csv": str(micro_csv),
        "nstep_loader": int(nstep),
        "notes": [
            "Bridge reconstructs gamma_phi and gamma_mix from kappa_env, localized couplings, and Lorentzian PSD.",
            "Loader residual validates the pslt_lib open_system_micro path at CSV knot points.",
        ],
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    if not args.skip_paper_copy:
        for src in [out_map, out_summary, out_png, out_meta]:
            (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary.to_string(index=False))
    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

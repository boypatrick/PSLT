#!/usr/bin/env python3
"""
Projected bath potential / effective-action statement audit.

This script upgrades the cocycle statement to a positive scalar-potential
statement. After dividing out kappa_env, the normalized parent bath block
defines anchored scalar potentials

  Phi_a(D) = log(B_a(D) / B_a^ref),
  B_a(D)   = gamma_a(D) / kappa_env(D),

with the canonical decomposition

  Phi_a(D) = L_sys,a(D) + L_spec,a(D),

where the gauge is fixed by anchor geometric means.  We then reconstruct the
same potential in two independent ways:

  1. anchor-average recovery from pairwise cocycle increments
  2. cumulative recovery from nearest-neighbor chain increments on the sorted D grid

If both agree with the canonical decomposition, the projected bath action is no
longer just an affine cocycle: it is an exact single-valued anchored potential
in the canonical log variables.
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


def _geom_mean(v: np.ndarray) -> float:
    vv = np.maximum(np.asarray(v, dtype=float), 1e-300)
    return float(np.exp(np.mean(np.log(vv))))


def _anchor_average_recovery(obs: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    anchor_vals = obs[anchor_mask]
    log_anchor = np.log(np.maximum(anchor_vals, 1e-300))
    mean_anchor = float(np.mean(log_anchor))
    return np.log(np.maximum(obs, 1e-300)) - mean_anchor


def _chain_recovery(obs: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    log_obs = np.log(np.maximum(obs, 1e-300))
    inc = np.zeros_like(log_obs)
    inc[1:] = np.diff(log_obs)
    rec = np.cumsum(inc)
    gauge_shift = float(np.mean(rec[anchor_mask]))
    return rec - gauge_shift


def main() -> None:
    ap = argparse.ArgumentParser(description="Projected bath potential / effective-action statement audit.")
    ap.add_argument(
        "--factor-map",
        type=Path,
        default=OUTDIR / "chi_open_system_bath_factorization_map.csv",
    )
    ap.add_argument(
        "--calib-csv",
        type=Path,
        default=OUTDIR / "kappa_env_anchor_calibration.csv",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.factor_map.exists():
        raise FileNotFoundError(args.factor_map)
    if not args.calib_csv.exists():
        raise FileNotFoundError(args.calib_csv)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.factor_map).copy().sort_values("D").reset_index(drop=True)
    calib = pd.read_csv(args.calib_csv).iloc[0]
    if df.empty:
        raise RuntimeError(f"Empty factor map: {args.factor_map}")

    required = [
        "D",
        "kappa_env",
        "bath_system_kernel_phi",
        "bath_system_kernel_mix",
        "bath_shape_phi",
        "bath_shape_mix",
        "gamma_phi_micro",
        "gamma_mix_micro",
        "bath_phi_identifiable",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    anchor_set = {
        round(float(v), 8)
        for v in str(calib["D_anchor_list"]).split(",")
        if str(v).strip()
    }
    if not anchor_set:
        raise RuntimeError("Calibration CSV has no anchor D list.")
    anchor_mask = np.asarray([round(float(d), 8) in anchor_set for d in df["D"]], dtype=bool)
    phi_identifiable = df["bath_phi_identifiable"].to_numpy(dtype=int) == 1

    kappa = np.maximum(df["kappa_env"].to_numpy(dtype=float), 1e-300)
    sys_phi = np.maximum(df["bath_system_kernel_phi"].to_numpy(dtype=float), 1e-300)
    sys_mix = np.maximum(df["bath_system_kernel_mix"].to_numpy(dtype=float), 1e-300)
    spec_phi = np.maximum(df["bath_shape_phi"].to_numpy(dtype=float), 1e-300)
    spec_mix = np.maximum(df["bath_shape_mix"].to_numpy(dtype=float), 1e-300)
    gamma_phi = np.maximum(df["gamma_phi_micro"].to_numpy(dtype=float), 1e-300)
    gamma_mix = np.maximum(df["gamma_mix_micro"].to_numpy(dtype=float), 1e-300)

    parent_phi = gamma_phi / kappa
    parent_mix = gamma_mix / kappa

    sys_phi_ref = _geom_mean(sys_phi[anchor_mask])
    sys_mix_ref = _geom_mean(sys_mix[anchor_mask])
    spec_phi_ref = _geom_mean(spec_phi[anchor_mask])
    spec_mix_ref = _geom_mean(spec_mix[anchor_mask])
    parent_phi_ref = _geom_mean(parent_phi[anchor_mask])
    parent_mix_ref = _geom_mean(parent_mix[anchor_mask])

    phi_obs = np.log(parent_phi / parent_phi_ref)
    mix_obs = np.log(parent_mix / parent_mix_ref)
    phi_recon = np.log(sys_phi / sys_phi_ref) + np.log(spec_phi / spec_phi_ref)
    mix_recon = np.log(sys_mix / sys_mix_ref) + np.log(spec_mix / spec_mix_ref)

    phi_anchor_rec = _anchor_average_recovery(parent_phi, anchor_mask)
    mix_anchor_rec = _anchor_average_recovery(parent_mix, anchor_mask)
    phi_chain_rec = _chain_recovery(parent_phi, anchor_mask)
    mix_chain_rec = _chain_recovery(parent_mix, anchor_mask)

    df["parent_bath_potential_phi_obs"] = phi_obs
    df["parent_bath_potential_mix_obs"] = mix_obs
    df["parent_bath_potential_phi_recon"] = phi_recon
    df["parent_bath_potential_mix_recon"] = mix_recon
    df["parent_bath_potential_phi_anchor_recovery"] = phi_anchor_rec
    df["parent_bath_potential_mix_anchor_recovery"] = mix_anchor_rec
    df["parent_bath_potential_phi_chain_recovery"] = phi_chain_rec
    df["parent_bath_potential_mix_chain_recovery"] = mix_chain_rec
    df["parent_bath_potential_phi_residual_abs"] = np.abs(phi_obs - phi_recon)
    df["parent_bath_potential_mix_residual_abs"] = np.abs(mix_obs - mix_recon)
    df["parent_bath_potential_phi_anchor_residual_abs"] = np.abs(phi_obs - phi_anchor_rec)
    df["parent_bath_potential_mix_anchor_residual_abs"] = np.abs(mix_obs - mix_anchor_rec)
    df["parent_bath_potential_phi_chain_residual_abs"] = np.abs(phi_obs - phi_chain_rec)
    df["parent_bath_potential_mix_chain_residual_abs"] = np.abs(mix_obs - mix_chain_rec)
    df["parent_bath_action_phi_obs"] = -phi_obs
    df["parent_bath_action_mix_obs"] = -mix_obs

    ident_phi = df.loc[phi_identifiable]

    summary = {
        "factor_map_csv": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "n_points": int(len(df)),
        "n_anchor": int(np.sum(anchor_mask)),
        "n_phi_identifiable": int(np.sum(phi_identifiable)),
        "max_abs_potential_residual_phi": float(df["parent_bath_potential_phi_residual_abs"].max()),
        "max_abs_potential_residual_mix": float(df["parent_bath_potential_mix_residual_abs"].max()),
        "max_abs_anchor_recovery_residual_phi": float(df["parent_bath_potential_phi_anchor_residual_abs"].max()),
        "max_abs_anchor_recovery_residual_mix": float(df["parent_bath_potential_mix_anchor_residual_abs"].max()),
        "max_abs_chain_recovery_residual_phi": float(df["parent_bath_potential_phi_chain_residual_abs"].max()),
        "max_abs_chain_recovery_residual_mix": float(df["parent_bath_potential_mix_chain_residual_abs"].max()),
        "p95_abs_potential_residual_phi_identifiable": q(ident_phi["parent_bath_potential_phi_residual_abs"], 0.95)
        if not ident_phi.empty
        else float("nan"),
        "p95_abs_anchor_recovery_residual_phi_identifiable": q(ident_phi["parent_bath_potential_phi_anchor_residual_abs"], 0.95)
        if not ident_phi.empty
        else float("nan"),
        "p95_abs_chain_recovery_residual_phi_identifiable": q(ident_phi["parent_bath_potential_phi_chain_residual_abs"], 0.95)
        if not ident_phi.empty
        else float("nan"),
        "p95_abs_potential_residual_mix": q(df["parent_bath_potential_mix_residual_abs"], 0.95),
        "p95_abs_anchor_recovery_residual_mix": q(df["parent_bath_potential_mix_anchor_residual_abs"], 0.95),
        "p95_abs_chain_recovery_residual_mix": q(df["parent_bath_potential_mix_chain_residual_abs"], 0.95),
        "corr(phi_obs, phi_recon)_identifiable": corr(
            ident_phi["parent_bath_potential_phi_obs"],
            ident_phi["parent_bath_potential_phi_recon"],
        ) if not ident_phi.empty else float("nan"),
        "corr(mix_obs, mix_recon)": corr(
            df["parent_bath_potential_mix_obs"],
            df["parent_bath_potential_mix_recon"],
        ),
        "corr(phi_obs, phi_anchor_rec)_identifiable": corr(
            ident_phi["parent_bath_potential_phi_obs"],
            ident_phi["parent_bath_potential_phi_anchor_recovery"],
        ) if not ident_phi.empty else float("nan"),
        "corr(mix_obs, mix_anchor_rec)": corr(
            df["parent_bath_potential_mix_obs"],
            df["parent_bath_potential_mix_anchor_recovery"],
        ),
        "corr(phi_obs, phi_chain_rec)_identifiable": corr(
            ident_phi["parent_bath_potential_phi_obs"],
            ident_phi["parent_bath_potential_phi_chain_recovery"],
        ) if not ident_phi.empty else float("nan"),
        "corr(mix_obs, mix_chain_rec)": corr(
            df["parent_bath_potential_mix_obs"],
            df["parent_bath_potential_mix_chain_recovery"],
        ),
    }
    summary["canonical_objective"] = (
        (summary["p95_abs_potential_residual_phi_identifiable"] if np.isfinite(summary["p95_abs_potential_residual_phi_identifiable"]) else 0.0)
        + summary["p95_abs_potential_residual_mix"]
        + (summary["p95_abs_anchor_recovery_residual_phi_identifiable"] if np.isfinite(summary["p95_abs_anchor_recovery_residual_phi_identifiable"]) else 0.0)
        + summary["p95_abs_anchor_recovery_residual_mix"]
        + (summary["p95_abs_chain_recovery_residual_phi_identifiable"] if np.isfinite(summary["p95_abs_chain_recovery_residual_phi_identifiable"]) else 0.0)
        + summary["p95_abs_chain_recovery_residual_mix"]
    )

    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""

    out_map = OUTDIR / f"chi_open_system_parent_bath_potential_map{suffix}.csv"
    out_summary = OUTDIR / f"chi_open_system_parent_bath_potential_summary{suffix}.csv"
    out_png = OUTDIR / f"chi_open_system_parent_bath_potential{suffix}.png"
    out_meta = OUTDIR / f"chi_open_system_parent_bath_potential_run_meta{suffix}.json"

    df.to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(df["D"], df["parent_bath_potential_mix_obs"], marker="o", label="mix potential")
    ax.plot(df["D"], df["parent_bath_potential_mix_recon"], linestyle="--", label="mix recon")
    ax.plot(df["D"], df["parent_bath_potential_mix_anchor_recovery"], linestyle=":", label="mix anchor-rec")
    ax.plot(df["D"], df["parent_bath_potential_mix_chain_recovery"], linestyle="-.", label="mix chain-rec")
    ax.set_title("Mix potential recovery")
    ax.set_xlabel("D")
    ax.set_ylabel("potential")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    if not ident_phi.empty:
        ax.plot(ident_phi["D"], ident_phi["parent_bath_potential_phi_obs"], marker="o", label="phi potential")
        ax.plot(ident_phi["D"], ident_phi["parent_bath_potential_phi_recon"], linestyle="--", label="phi recon")
        ax.plot(ident_phi["D"], ident_phi["parent_bath_potential_phi_anchor_recovery"], linestyle=":", label="phi anchor-rec")
        ax.plot(ident_phi["D"], ident_phi["parent_bath_potential_phi_chain_recovery"], linestyle="-.", label="phi chain-rec")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No identifiable phi subset", ha="center", va="center")
    ax.set_title("Phi potential recovery (identifiable subset)")
    ax.set_xlabel("D")
    ax.set_ylabel("potential")

    ax = axes[1, 0]
    ax.semilogy(df["D"], np.maximum(df["parent_bath_potential_mix_residual_abs"], 1e-18), marker="o", label="mix recon")
    ax.semilogy(df["D"], np.maximum(df["parent_bath_potential_mix_anchor_residual_abs"], 1e-18), marker="s", label="mix anchor")
    ax.semilogy(df["D"], np.maximum(df["parent_bath_potential_mix_chain_residual_abs"], 1e-18), marker="^", label="mix chain")
    if not ident_phi.empty:
        ax.semilogy(ident_phi["D"], np.maximum(ident_phi["parent_bath_potential_phi_residual_abs"], 1e-18), linestyle="--", label="phi recon")
        ax.semilogy(ident_phi["D"], np.maximum(ident_phi["parent_bath_potential_phi_anchor_residual_abs"], 1e-18), linestyle=":", label="phi anchor")
        ax.semilogy(ident_phi["D"], np.maximum(ident_phi["parent_bath_potential_phi_chain_residual_abs"], 1e-18), linestyle="-.", label="phi chain")
    ax.set_title("Potential residuals")
    ax.set_xlabel("D")
    ax.set_ylabel("absolute residual")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.plot(df["D"], df["parent_bath_action_mix_obs"], marker="o", label=r"$\Delta\Gamma_{\rm mix}$")
    ax.plot(df["D"], df["parent_bath_action_phi_obs"], marker="s", label=r"$\Delta\Gamma_{\phi}$")
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax.set_title("Anchored parent bath action")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$-\Phi$")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Projected parent bath potential / effective-action statement", fontsize=13)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "factor_map": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "statement": {
            "B_a": "gamma_a / kappa_env",
            "Phi_a": "log(B_a / B_a_ref)",
            "Phi_decomposition": "L_sys,a + L_spec,a",
            "DeltaGamma_a": "-Phi_a",
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    for src in [out_map, out_summary, out_png, out_meta]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parent-source-model audit for the EYMH source-informed UV comparator.

This audit rewrites the two dominant source factors in parent-action language:

1. coeff_participation_access
   -> effective loop-trace participation number and entropy of the two-layer
      shell weights.
2. tree_diag_compressibility
   -> shell-background-normalized diagonal susceptibility and its pressure
      fraction inside the tree-level diagonal block.
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


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPERDIR = ROOT / "paper"


def corr(a: pd.Series, b: pd.Series) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def q(v: pd.Series, p: float) -> float:
    return float(v.quantile(p))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit parent-source-model witnesses for EYMH source-informed matching.")
    ap.add_argument(
        "--source-informed-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_source_informed_D21E21_fix.csv",
    )
    ap.add_argument(
        "--constant-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_constant_refresh_D21E21.csv",
    )
    ap.add_argument("--tag", default="D21E21")
    args = ap.parse_args()

    src = pd.read_csv(args.source_informed_map)
    const = pd.read_csv(args.constant_map)[["D", "eta", "mu_mumu_uv_rge"]].rename(
        columns={"mu_mumu_uv_rge": "mu_mumu_uv_rge_const"}
    )
    df = src.merge(const, on=["D", "eta"], how="left")

    coeff_l1 = np.asarray(df["coeff_l1"], dtype=float)
    coeff_l2 = np.asarray(df["coeff_l2"], dtype=float)
    coeff_sum = coeff_l1 + coeff_l2
    coeff_sum_safe = np.where(coeff_sum > 0.0, coeff_sum, 1.0)
    p1 = coeff_l1 / coeff_sum_safe
    p2 = coeff_l2 / coeff_sum_safe

    neff_trace = 1.0 / np.clip(p1**2 + p2**2, 1e-30, None)
    participation_entropy = -(p1 * np.log(np.clip(p1, 1e-30, None)) + p2 * np.log(np.clip(p2, 1e-30, None)))
    participation_entropy_norm = participation_entropy / np.log(2.0)
    coeff_participation_exact = np.full(len(df), np.inf, dtype=float)
    valid = (coeff_l1 > 1e-30) & (coeff_l2 > 1e-30)
    both_zero = (coeff_l1 <= 1e-30) & (coeff_l2 <= 1e-30)
    coeff_participation_exact[valid] = (coeff_l1[valid] / coeff_l2[valid]) ** 2
    coeff_participation_access_recon = np.zeros(len(df), dtype=float)
    coeff_participation_access_recon[valid] = np.power(np.maximum(coeff_participation_exact[valid], 1e-30), -0.25)
    coeff_participation_access_recon[both_zero] = 1.0
    coeff_participation_residual = coeff_participation_access_recon - np.asarray(
        df["coeff_participation_access"], dtype=float
    )

    gap_cv = np.asarray(df["gap_cv"], dtype=float)
    c_tree_diag_cv = np.asarray(df["c_tree_diag_cv"], dtype=float)
    tree_diag_susceptibility_recon = c_tree_diag_cv / (1.0 + gap_cv)
    tree_diag_compressibility_recon = (1.0 + tree_diag_susceptibility_recon) ** (-0.5)
    tree_diag_residual = tree_diag_compressibility_recon - np.asarray(df["tree_diag_compressibility"], dtype=float)
    tree_diag_pressure_fraction = tree_diag_susceptibility_recon / (1.0 + tree_diag_susceptibility_recon)
    shell_background_load = gap_cv / (1.0 + gap_cv)

    df["coeff_p1"] = p1
    df["coeff_p2"] = p2
    df["loop_trace_neff"] = neff_trace
    df["loop_trace_entropy_norm"] = participation_entropy_norm
    df["coeff_participation_exact"] = coeff_participation_exact
    df["coeff_participation_access_recon"] = coeff_participation_access_recon
    df["coeff_participation_residual"] = coeff_participation_residual
    df["tree_diag_susceptibility_recon"] = tree_diag_susceptibility_recon
    df["tree_diag_compressibility_recon"] = tree_diag_compressibility_recon
    df["tree_diag_residual"] = tree_diag_residual
    df["tree_diag_pressure_fraction"] = tree_diag_pressure_fraction
    df["shell_background_load"] = shell_background_load
    df["abs_mu_diff_vs_const"] = np.abs(
        np.asarray(df["mu_mumu_uv_rge"], dtype=float) - np.asarray(df["mu_mumu_uv_rge_const"], dtype=float)
    )

    summary = {
        "n_points": int(len(df)),
        "max_abs_coeff_participation_recon_residual": float(np.max(np.abs(coeff_participation_residual))),
        "max_abs_tree_diag_recon_residual": float(np.max(np.abs(tree_diag_residual))),
        "coeff_participation_exact_p50": q(df["coeff_participation_exact"].replace(np.inf, np.nan).dropna(), 0.5),
        "coeff_participation_exact_p90": q(df["coeff_participation_exact"].replace(np.inf, np.nan).dropna(), 0.9),
        "loop_trace_neff_p50": q(df["loop_trace_neff"], 0.5),
        "loop_trace_neff_p90": q(df["loop_trace_neff"], 0.9),
        "loop_trace_entropy_norm_p50": q(df["loop_trace_entropy_norm"], 0.5),
        "loop_trace_entropy_norm_p90": q(df["loop_trace_entropy_norm"], 0.9),
        "coeff_participation_access_p50": q(df["coeff_participation_access"], 0.5),
        "coeff_participation_access_p90": q(df["coeff_participation_access"], 0.9),
        "tree_diag_susceptibility_p50": q(df["tree_diag_susceptibility"], 0.5),
        "tree_diag_susceptibility_p90": q(df["tree_diag_susceptibility"], 0.9),
        "tree_diag_compressibility_p50": q(df["tree_diag_compressibility"], 0.5),
        "tree_diag_compressibility_p90": q(df["tree_diag_compressibility"], 0.9),
        "tree_diag_pressure_fraction_p50": q(df["tree_diag_pressure_fraction"], 0.5),
        "tree_diag_pressure_fraction_p90": q(df["tree_diag_pressure_fraction"], 0.9),
        "corr(coeff_participation_exact, coeff_participation_access)": corr(
            df["coeff_participation_exact"].replace(np.inf, np.nan).fillna(0.0), df["coeff_participation_access"]
        ),
        "corr(loop_trace_neff, coeff_participation_access)": corr(df["loop_trace_neff"], df["coeff_participation_access"]),
        "corr(loop_trace_entropy_norm, coeff_participation_access)": corr(
            df["loop_trace_entropy_norm"], df["coeff_participation_access"]
        ),
        "corr(coeff_participation_access, eymh_source_prefactor_diag)": corr(
            df["coeff_participation_access"], df["eymh_source_prefactor_diag"]
        ),
        "corr(tree_diag_susceptibility, eymh_source_prefactor_diag)": corr(
            df["tree_diag_susceptibility"], df["eymh_source_prefactor_diag"]
        ),
        "corr(tree_diag_compressibility, eymh_source_prefactor_diag)": corr(
            df["tree_diag_compressibility"], df["eymh_source_prefactor_diag"]
        ),
        "corr(tree_diag_pressure_fraction, eymh_source_prefactor_diag)": corr(
            df["tree_diag_pressure_fraction"], df["eymh_source_prefactor_diag"]
        ),
        "corr(loop_trace_neff, abs_mu_diff_vs_const)": corr(df["loop_trace_neff"], df["abs_mu_diff_vs_const"]),
        "corr(loop_trace_entropy_norm, abs_mu_diff_vs_const)": corr(
            df["loop_trace_entropy_norm"], df["abs_mu_diff_vs_const"]
        ),
        "corr(tree_diag_pressure_fraction, abs_mu_diff_vs_const)": corr(
            df["tree_diag_pressure_fraction"], df["abs_mu_diff_vs_const"]
        ),
        "corr(tree_diag_compressibility, abs_mu_diff_vs_const)": corr(
            df["tree_diag_compressibility"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_parent_source_model_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_source_model_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_source_model_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_source_model_run_meta_{args.tag}.json"

    df[
        [
            "D",
            "eta",
            "coeff_l1",
            "coeff_l2",
            "coeff_p1",
            "coeff_p2",
            "coeff_participation_exact",
            "loop_trace_neff",
            "loop_trace_entropy_norm",
            "coeff_participation_access",
            "coeff_participation_access_recon",
            "coeff_participation_residual",
            "gap_cv",
            "c_tree_diag_cv",
            "tree_diag_susceptibility",
            "tree_diag_susceptibility_recon",
            "tree_diag_compressibility",
            "tree_diag_compressibility_recon",
            "tree_diag_residual",
            "tree_diag_pressure_fraction",
            "shell_background_load",
            "eymh_source_prefactor_diag",
            "mu_mumu_uv_rge",
            "mu_mumu_uv_rge_const",
            "abs_mu_diff_vs_const",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "source_informed_map": str(args.source_informed_map),
                "constant_map": str(args.constant_map),
                "n_points": len(df),
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    axes[0].scatter(df["coeff_participation_exact"].replace(np.inf, np.nan), df["coeff_participation_access"], s=12, alpha=0.7)
    axes[0].set_xlabel("coeff_participation_exact")
    axes[0].set_ylabel("coeff_participation_access")
    axes[0].set_title("Canonical participation witness")

    axes[1].scatter(df["tree_diag_susceptibility"], df["tree_diag_compressibility"], s=12, alpha=0.7)
    axes[1].set_xlabel("tree_diag_susceptibility")
    axes[1].set_ylabel("tree_diag_compressibility")
    axes[1].set_title("Diagonal susceptibility/compressibility")

    axes[2].scatter(df["tree_diag_pressure_fraction"], df["eymh_source_prefactor_diag"], s=12, alpha=0.7)
    axes[2].set_xlabel("tree_diag_pressure_fraction")
    axes[2].set_ylabel("eymh_source_prefactor_diag")
    axes[2].set_title("Pressure vs source prefactor")
    fig.suptitle("EYMH parent-source model audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_map, out_summary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

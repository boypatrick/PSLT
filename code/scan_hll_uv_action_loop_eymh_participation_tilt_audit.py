#!/usr/bin/env python3
"""
Projected coefficient-norm tilt audit for the parented EYMH UV comparator.

This audit pushes the parent-action participation story one step further by
rewriting the exact norm-ratio coherence as a projected free-energy tilt of the
coefficient block

    c_N = P_N^kin / M_N^2.

Using the projected norm pair (||c||_1, ||c||_2), define

    DeltaF_norm = log(||c||_1 / ||c||_2),
    A_part^tilt = exp(-DeltaF_norm / 2) = sqrt(||c||_2 / ||c||_1).

The canonical parented prefactor then factorizes as

    Pi_parent = hk_loop_local_prefactor_diag
                * shell_access
                * A_part^tilt
                * tree_diag_compressibility_parented.

So the participation side can be read as an exact projected coefficient-norm
free-energy tilt, rather than only a norm-ratio or two-mode probability proxy.
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
    ap = argparse.ArgumentParser(
        description="Audit projected coefficient-norm free-energy tilt for the parented EYMH UV comparator."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument(
        "--constant-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_constant_refresh_D21E21.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    args = ap.parse_args()

    parented = pd.read_csv(args.parented_map)
    const = pd.read_csv(args.constant_map)[["D", "eta", "mu_mumu_uv_rge"]].rename(
        columns={"mu_mumu_uv_rge": "mu_mumu_uv_rge_const"}
    )
    df = parented.merge(const, on=["D", "eta"], how="left")

    coeff_l1 = np.asarray(df["coeff_l1"], dtype=float)
    coeff_l2 = np.asarray(df["coeff_l2"], dtype=float)
    coeff_l1_safe = np.maximum(coeff_l1, 1e-30)
    coeff_l2_safe = np.maximum(coeff_l2, 1e-30)

    coeff_norm_gap_log = np.log(coeff_l1_safe / coeff_l2_safe)
    coeff_tilt_coherence = np.exp(-0.5 * coeff_norm_gap_log)
    coeff_access_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float)
    coeff_resid = coeff_tilt_coherence - coeff_access_ref

    shell_spread = np.asarray(df["shell_spread"], dtype=float)
    shell_access = np.sqrt(np.maximum(shell_spread, 1e-30) / (1.0 + np.maximum(shell_spread, 1e-30)))
    tree_comp = np.asarray(df["tree_diag_compressibility_parented"], dtype=float)
    hk_local = np.asarray(df["hk_loop_local_prefactor_diag"], dtype=float)
    prefactor_exact = hk_local * shell_access * coeff_tilt_coherence * tree_comp
    prefactor_ref = np.asarray(df["eymh_parented_prefactor_diag"], dtype=float)
    prefactor_resid = prefactor_exact - prefactor_ref

    parented_response = shell_access * coeff_tilt_coherence * tree_comp
    mu_diff_vs_const = np.abs(
        np.asarray(df["mu_mumu_uv_rge"], dtype=float) - np.asarray(df["mu_mumu_uv_rge_const"], dtype=float)
    )

    df["coeff_norm_gap_log"] = coeff_norm_gap_log
    df["coeff_tilt_coherence"] = coeff_tilt_coherence
    df["coeff_participation_access_parented_tilt"] = coeff_tilt_coherence
    df["coeff_participation_access_parented_tilt_residual"] = coeff_resid
    df["shell_access_parented"] = shell_access
    df["parented_response_factor"] = parented_response
    df["eymh_parented_prefactor_diag_tilt_recon"] = prefactor_exact
    df["eymh_parented_prefactor_diag_tilt_residual"] = prefactor_resid
    df["abs_mu_diff_vs_const"] = mu_diff_vs_const

    summary = {
        "n_points": int(len(df)),
        "max_abs_coeff_participation_tilt_residual": float(np.max(np.abs(coeff_resid))),
        "max_abs_eymh_parented_prefactor_tilt_residual": float(np.max(np.abs(prefactor_resid))),
        "coeff_norm_gap_log_p50": q(df["coeff_norm_gap_log"], 0.5),
        "coeff_norm_gap_log_p90": q(df["coeff_norm_gap_log"], 0.9),
        "coeff_tilt_coherence_p50": q(df["coeff_tilt_coherence"], 0.5),
        "coeff_tilt_coherence_p90": q(df["coeff_tilt_coherence"], 0.9),
        "shell_access_parented_p50": q(df["shell_access_parented"], 0.5),
        "shell_access_parented_p90": q(df["shell_access_parented"], 0.9),
        "parented_response_factor_p50": q(df["parented_response_factor"], 0.5),
        "parented_response_factor_p90": q(df["parented_response_factor"], 0.9),
        "corr(coeff_norm_gap_log, coeff_participation_access_parented)": corr(
            df["coeff_norm_gap_log"], df["coeff_participation_access_parented"]
        ),
        "corr(coeff_tilt_coherence, coeff_participation_access_parented)": corr(
            df["coeff_tilt_coherence"], df["coeff_participation_access_parented"]
        ),
        "corr(coeff_norm_gap_log, eymh_parented_prefactor_diag)": corr(
            df["coeff_norm_gap_log"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(coeff_tilt_coherence, eymh_parented_prefactor_diag)": corr(
            df["coeff_tilt_coherence"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(parented_response_factor, eymh_parented_prefactor_diag)": corr(
            df["parented_response_factor"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(coeff_norm_gap_log, abs_mu_diff_vs_const)": corr(df["coeff_norm_gap_log"], df["abs_mu_diff_vs_const"]),
        "corr(parented_response_factor, abs_mu_diff_vs_const)": corr(
            df["parented_response_factor"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_participation_tilt_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_participation_tilt_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_participation_tilt_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_participation_tilt_audit_run_meta_{args.tag}.json"

    df[
        [
            "D",
            "eta",
            "coeff_l1",
            "coeff_l2",
            "coeff_norm_gap_log",
            "coeff_tilt_coherence",
            "coeff_participation_access_parented",
            "coeff_participation_access_parented_tilt",
            "coeff_participation_access_parented_tilt_residual",
            "shell_spread",
            "shell_access_parented",
            "tree_diag_compressibility_parented",
            "hk_loop_local_prefactor_diag",
            "parented_response_factor",
            "eymh_parented_prefactor_diag",
            "eymh_parented_prefactor_diag_tilt_recon",
            "eymh_parented_prefactor_diag_tilt_residual",
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
                "parented_map": str(args.parented_map),
                "constant_map": str(args.constant_map),
                "n_points": len(df),
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    axes[0].scatter(df["coeff_norm_gap_log"], df["coeff_participation_access_parented"], s=12, alpha=0.7)
    axes[0].set_xlabel("DeltaF_norm = log(||c||_1 / ||c||_2)")
    axes[0].set_ylabel("coeff_participation_access_parented")
    axes[0].set_title("Coherence vs norm tilt")

    axes[1].scatter(df["coeff_tilt_coherence"], df["eymh_parented_prefactor_diag"], s=12, alpha=0.7)
    axes[1].set_xlabel("A_part^tilt")
    axes[1].set_ylabel("eymh_parented_prefactor_diag")
    axes[1].set_title("Tilt coherence vs prefactor")

    axes[2].scatter(df["parented_response_factor"], df["eymh_parented_prefactor_diag"], s=12, alpha=0.7)
    axes[2].set_xlabel("shell_access * A_part^tilt * tree_comp")
    axes[2].set_ylabel("eymh_parented_prefactor_diag")
    axes[2].set_title("Projected response vs prefactor")
    fig.suptitle("EYMH coefficient-norm tilt audit", fontsize=13)
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

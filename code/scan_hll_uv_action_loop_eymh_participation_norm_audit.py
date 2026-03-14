#!/usr/bin/env python3
"""
Exact projected coefficient-norm participation audit for the parented EYMH UV comparator.

This audit pushes the participation story one step closer to parent-action
language. Instead of reconstructing the participation block through the derived
two-mode probabilities (p1, p2), we rewrite it directly in terms of the
projected coefficient vector norms

    c_N = P_N^kin / M_N^2,
    ||c||_1 = coeff_l1,
    ||c||_2 = coeff_l2.

The canonical parented participation/coherence factor is then exactly

    A_part^norm = (||c||_2^2 / ||c||_1^2)^(1/4) = sqrt(||c||_2 / ||c||_1),

which is a norm-ratio participation/coherence response of the projected
parent-action coefficient block.
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
        description="Audit exact coefficient-norm participation closure for the parented EYMH UV comparator."
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

    coeff_norm_q2 = (coeff_l2_safe**2) / (coeff_l1_safe**2)
    coeff_norm_neff = 1.0 / np.maximum(coeff_norm_q2, 1e-30)
    coeff_norm_coherence = np.power(np.maximum(coeff_norm_q2, 1e-30), 0.25)
    coeff_access_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float)
    coeff_resid = coeff_norm_coherence - coeff_access_ref

    mu_diff_vs_const = np.abs(
        np.asarray(df["mu_mumu_uv_rge"], dtype=float) - np.asarray(df["mu_mumu_uv_rge_const"], dtype=float)
    )

    df["coeff_norm_q2"] = coeff_norm_q2
    df["coeff_norm_neff"] = coeff_norm_neff
    df["coeff_norm_coherence"] = coeff_norm_coherence
    df["coeff_participation_access_parented_norm"] = coeff_norm_coherence
    df["coeff_participation_access_parented_norm_residual"] = coeff_resid
    df["abs_mu_diff_vs_const"] = mu_diff_vs_const

    summary = {
        "n_points": int(len(df)),
        "max_abs_coeff_participation_norm_residual": float(np.max(np.abs(coeff_resid))),
        "coeff_norm_q2_p50": q(df["coeff_norm_q2"], 0.5),
        "coeff_norm_q2_p90": q(df["coeff_norm_q2"], 0.9),
        "coeff_norm_neff_p50": q(df["coeff_norm_neff"], 0.5),
        "coeff_norm_neff_p90": q(df["coeff_norm_neff"], 0.9),
        "coeff_norm_coherence_p50": q(df["coeff_norm_coherence"], 0.5),
        "coeff_norm_coherence_p90": q(df["coeff_norm_coherence"], 0.9),
        "coeff_participation_access_parented_p50": q(df["coeff_participation_access_parented"], 0.5),
        "coeff_participation_access_parented_p90": q(df["coeff_participation_access_parented"], 0.9),
        "corr(coeff_norm_q2, coeff_participation_access_parented)": corr(
            df["coeff_norm_q2"], df["coeff_participation_access_parented"]
        ),
        "corr(coeff_norm_neff, coeff_participation_access_parented)": corr(
            df["coeff_norm_neff"], df["coeff_participation_access_parented"]
        ),
        "corr(coeff_norm_coherence, coeff_participation_access_parented)": corr(
            df["coeff_norm_coherence"], df["coeff_participation_access_parented"]
        ),
        "corr(coeff_norm_coherence, eymh_parented_prefactor_diag)": corr(
            df["coeff_norm_coherence"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(coeff_norm_q2, eymh_parented_prefactor_diag)": corr(
            df["coeff_norm_q2"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(coeff_norm_neff, eymh_parented_prefactor_diag)": corr(
            df["coeff_norm_neff"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(coeff_norm_coherence, abs_mu_diff_vs_const)": corr(
            df["coeff_norm_coherence"], df["abs_mu_diff_vs_const"]
        ),
        "corr(coeff_norm_q2, abs_mu_diff_vs_const)": corr(
            df["coeff_norm_q2"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_participation_norm_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_participation_norm_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_participation_norm_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_participation_norm_audit_run_meta_{args.tag}.json"

    df[
        [
            "D",
            "eta",
            "coeff_l1",
            "coeff_l2",
            "coeff_norm_q2",
            "coeff_norm_neff",
            "coeff_norm_coherence",
            "coeff_participation_access_parented",
            "coeff_participation_access_parented_norm",
            "coeff_participation_access_parented_norm_residual",
            "eymh_parented_prefactor_diag",
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
    axes[0].scatter(df["coeff_norm_q2"], df["coeff_participation_access_parented"], s=12, alpha=0.7)
    axes[0].set_xlabel("Q2 = ||c||_2^2 / ||c||_1^2")
    axes[0].set_ylabel("coeff_participation_access_parented")
    axes[0].set_title("Participation vs norm ratio")

    axes[1].scatter(df["coeff_norm_neff"], df["coeff_participation_access_parented"], s=12, alpha=0.7)
    axes[1].set_xlabel("N_eff^norm = ||c||_1^2 / ||c||_2^2")
    axes[1].set_ylabel("coeff_participation_access_parented")
    axes[1].set_title("Participation vs norm-based N_eff")

    axes[2].scatter(df["coeff_norm_coherence"], df["eymh_parented_prefactor_diag"], s=12, alpha=0.7)
    axes[2].set_xlabel("A_part^norm")
    axes[2].set_ylabel("eymh_parented_prefactor_diag")
    axes[2].set_title("Norm coherence vs parented prefactor")
    fig.suptitle("EYMH exact coefficient-norm participation audit", fontsize=13)
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

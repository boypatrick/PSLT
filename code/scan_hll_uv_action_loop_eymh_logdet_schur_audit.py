#!/usr/bin/env python3
"""
Projected log-det / Schur-kernel audit for the parented EYMH UV comparator.

This audit recasts the canonical parented response action in the most direct
mother-action language available in the current pipeline:

1. Participation block:
     K_part = ||c||_1 / ||c||_2
     S_part = log det K_part = log(||c||_1 / ||c||_2)

2. Tree block:
     K_bg   = 1 + gap_cv
     K_full = 1 + gap_cv + c_tree_diag_cv
     G_schur = det(K_full) / det(K_bg) = K_full / K_bg = 1 + chi_tree
     S_schur = log G_schur

Then the full projected response action is

    S_resp = S_part + S_schur = log det(K_part) + log G_schur

and the canonical parented prefactor closes as

    Pi_parent
      = Pi_hk,local
        * A_shell
        * exp(-S_resp / 2).

This is stronger than the previous response-action wording because the two
surviving source factors are now explicitly identified as a projected log-det
piece and a shell-normalized Schur-kernel piece.
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


def corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def q(v: pd.Series | np.ndarray, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit projected log-det / Schur-kernel closure of the parented EYMH UV comparator."
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

    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    k_part = coeff_l1 / coeff_l2
    s_part = np.log(k_part)

    k_bg = 1.0 + np.maximum(np.asarray(df["gap_cv"], dtype=float), 0.0)
    k_full = k_bg + np.maximum(np.asarray(df["c_tree_diag_cv"], dtype=float), 0.0)
    g_schur = k_full / np.maximum(k_bg, 1e-30)
    s_schur = np.log(g_schur)

    s_resp = s_part + s_schur
    a_resp = np.exp(-0.5 * s_resp)

    coeff_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float)
    tree_ref = np.asarray(df["tree_diag_compressibility_parented"], dtype=float)
    response_ref = coeff_ref * tree_ref
    response_resid = a_resp - response_ref

    shell_spread = np.maximum(np.asarray(df["shell_spread"], dtype=float), 1e-30)
    shell_access = np.sqrt(shell_spread / (1.0 + shell_spread))
    hk_local = np.asarray(df["hk_loop_local_prefactor_diag"], dtype=float)
    prefactor_ref = np.asarray(df["eymh_parented_prefactor_diag"], dtype=float)
    prefactor_recon = hk_local * shell_access * a_resp
    prefactor_resid = prefactor_recon - prefactor_ref

    mu_diff_vs_const = np.abs(
        np.asarray(df["mu_mumu_uv_rge"], dtype=float) - np.asarray(df["mu_mumu_uv_rge_const"], dtype=float)
    )

    part_share = s_part / np.maximum(s_resp, 1e-30)
    schur_share = s_schur / np.maximum(s_resp, 1e-30)

    df["projected_logdet_part_kernel"] = k_part
    df["projected_logdet_part_action"] = s_part
    df["schur_bg_kernel"] = k_bg
    df["schur_full_kernel"] = k_full
    df["projected_schur_gain"] = g_schur
    df["projected_schur_action"] = s_schur
    df["projected_logdet_schur_response_action"] = s_resp
    df["projected_logdet_schur_response_weight"] = a_resp
    df["projected_logdet_schur_response_weight_residual"] = response_resid
    df["projected_logdet_schur_prefactor_recon"] = prefactor_recon
    df["projected_logdet_schur_prefactor_residual"] = prefactor_resid
    df["projected_logdet_part_share"] = part_share
    df["projected_schur_share"] = schur_share
    df["shell_access_parented"] = shell_access
    df["abs_mu_diff_vs_const"] = mu_diff_vs_const

    summary = {
        "n_points": int(len(df)),
        "max_abs_logdet_schur_weight_residual": float(np.max(np.abs(response_resid))),
        "max_abs_logdet_schur_prefactor_residual": float(np.max(np.abs(prefactor_resid))),
        "projected_logdet_part_action_p50": q(df["projected_logdet_part_action"], 0.5),
        "projected_logdet_part_action_p90": q(df["projected_logdet_part_action"], 0.9),
        "projected_schur_action_p50": q(df["projected_schur_action"], 0.5),
        "projected_schur_action_p90": q(df["projected_schur_action"], 0.9),
        "projected_logdet_schur_response_action_p50": q(df["projected_logdet_schur_response_action"], 0.5),
        "projected_logdet_schur_response_action_p90": q(df["projected_logdet_schur_response_action"], 0.9),
        "projected_logdet_part_share_p50": q(df["projected_logdet_part_share"], 0.5),
        "projected_logdet_part_share_p90": q(df["projected_logdet_part_share"], 0.9),
        "projected_schur_share_p50": q(df["projected_schur_share"], 0.5),
        "projected_schur_share_p90": q(df["projected_schur_share"], 0.9),
        "corr(projected_logdet_part_action, response_action)": corr(
            df["projected_logdet_part_action"], df["projected_logdet_schur_response_action"]
        ),
        "corr(projected_schur_action, response_action)": corr(
            df["projected_schur_action"], df["projected_logdet_schur_response_action"]
        ),
        "corr(response_action, eymh_parented_prefactor_diag)": corr(
            df["projected_logdet_schur_response_action"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(response_weight, eymh_parented_prefactor_diag)": corr(
            df["projected_logdet_schur_response_weight"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(response_action, abs_mu_diff_vs_const)": corr(
            df["projected_logdet_schur_response_action"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_logdet_schur_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_logdet_schur_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_logdet_schur_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_logdet_schur_audit_run_meta_{args.tag}.json"

    df[
        [
            "D",
            "eta",
            "coeff_l1",
            "coeff_l2",
            "projected_logdet_part_kernel",
            "projected_logdet_part_action",
            "tree_diag_susceptibility_parented",
            "schur_bg_kernel",
            "schur_full_kernel",
            "projected_schur_gain",
            "projected_schur_action",
            "projected_logdet_schur_response_action",
            "projected_logdet_schur_response_weight",
            "projected_logdet_schur_response_weight_residual",
            "shell_spread",
            "shell_access_parented",
            "hk_loop_local_prefactor_diag",
            "eymh_parented_prefactor_diag",
            "projected_logdet_schur_prefactor_recon",
            "projected_logdet_schur_prefactor_residual",
            "projected_logdet_part_share",
            "projected_schur_share",
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
    axes[0].scatter(df["projected_logdet_part_action"], df["projected_schur_action"], s=12, alpha=0.7)
    axes[0].set_xlabel("S_part = log det K_part")
    axes[0].set_ylabel("S_schur = log G_schur")
    axes[0].set_title("Projected log-det vs Schur")

    axes[1].scatter(
        df["projected_logdet_schur_response_action"], df["eymh_parented_prefactor_diag"], s=12, alpha=0.7
    )
    axes[1].set_xlabel("S_resp = S_part + S_schur")
    axes[1].set_ylabel("eymh_parented_prefactor_diag")
    axes[1].set_title("Response action vs prefactor")

    axes[2].scatter(df["projected_logdet_part_share"], df["projected_schur_share"], s=12, alpha=0.7)
    axes[2].set_xlabel("log-det share")
    axes[2].set_ylabel("Schur share")
    axes[2].set_title("Projected action shares")

    fig.suptitle("EYMH projected log-det / Schur audit", fontsize=13)
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

#!/usr/bin/env python3
"""
Projected response-action audit for the parented EYMH UV comparator.

This audit fuses the two surviving parent-action response blocks into a single
projected response action:

  DeltaF_norm = log(||c||_1 / ||c||_2),
  S_tree      = log(1 + chi_tree),
  S_resp      = DeltaF_norm + S_tree.

The canonical parented prefactor is then

  Pi_parent
    = Pi_hk,local
      * A_shell
      * exp(-S_resp / 2)
    = Pi_hk,local
      * A_shell
      * A_part^tilt
      * tree_diag_compressibility_parented.

Equivalently, if we define the projected response matrix

  K_resp = diag(exp(DeltaF_norm), 1 + chi_tree),

then

  det(K_resp) = exp(S_resp),
  A_resp = det(K_resp)^(-1/2).

This is the strongest parent-action reading so far: the participation tilt and
the shell-background-normalized tree-diagonal compressibility enter additively
in a single projected response action.
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
        description="Audit the projected response-action closure behind the parented EYMH UV comparator."
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
    delta_f_norm = np.log(coeff_l1 / coeff_l2)
    a_part_tilt = np.exp(-0.5 * delta_f_norm)

    chi_tree = np.asarray(df["tree_diag_susceptibility_parented"], dtype=float)
    s_tree = np.log1p(np.maximum(chi_tree, 0.0))
    tree_comp = np.asarray(df["tree_diag_compressibility_parented"], dtype=float)

    shell_spread = np.maximum(np.asarray(df["shell_spread"], dtype=float), 1e-30)
    shell_access = np.sqrt(shell_spread / (1.0 + shell_spread))
    hk_local = np.asarray(df["hk_loop_local_prefactor_diag"], dtype=float)
    prefactor_ref = np.asarray(df["eymh_parented_prefactor_diag"], dtype=float)

    response_action = delta_f_norm + s_tree
    response_det = np.exp(response_action)
    response_weight = np.exp(-0.5 * response_action)
    response_weight_ref = a_part_tilt * tree_comp
    response_weight_resid = response_weight - response_weight_ref

    prefactor_recon = hk_local * shell_access * response_weight
    prefactor_resid = prefactor_recon - prefactor_ref

    mu_diff_vs_const = np.abs(
        np.asarray(df["mu_mumu_uv_rge"], dtype=float) - np.asarray(df["mu_mumu_uv_rge_const"], dtype=float)
    )

    coeff_share = delta_f_norm / np.maximum(response_action, 1e-30)
    tree_share = s_tree / np.maximum(response_action, 1e-30)

    df["delta_f_norm"] = delta_f_norm
    df["s_tree_parented"] = s_tree
    df["projected_response_action"] = response_action
    df["projected_response_det"] = response_det
    df["projected_response_weight"] = response_weight
    df["projected_response_weight_ref"] = response_weight_ref
    df["projected_response_weight_residual"] = response_weight_resid
    df["projected_response_prefactor_recon"] = prefactor_recon
    df["projected_response_prefactor_residual"] = prefactor_resid
    df["coeff_action_share"] = coeff_share
    df["tree_action_share"] = tree_share
    df["shell_access_parented"] = shell_access
    df["abs_mu_diff_vs_const"] = mu_diff_vs_const

    summary = {
        "n_points": int(len(df)),
        "max_abs_response_weight_residual": float(np.max(np.abs(response_weight_resid))),
        "max_abs_response_prefactor_residual": float(np.max(np.abs(prefactor_resid))),
        "delta_f_norm_p50": q(df["delta_f_norm"], 0.5),
        "delta_f_norm_p90": q(df["delta_f_norm"], 0.9),
        "s_tree_parented_p50": q(df["s_tree_parented"], 0.5),
        "s_tree_parented_p90": q(df["s_tree_parented"], 0.9),
        "projected_response_action_p50": q(df["projected_response_action"], 0.5),
        "projected_response_action_p90": q(df["projected_response_action"], 0.9),
        "projected_response_weight_p50": q(df["projected_response_weight"], 0.5),
        "projected_response_weight_p90": q(df["projected_response_weight"], 0.9),
        "coeff_action_share_p50": q(df["coeff_action_share"], 0.5),
        "coeff_action_share_p90": q(df["coeff_action_share"], 0.9),
        "tree_action_share_p50": q(df["tree_action_share"], 0.5),
        "tree_action_share_p90": q(df["tree_action_share"], 0.9),
        "corr(delta_f_norm, projected_response_action)": corr(
            df["delta_f_norm"], df["projected_response_action"]
        ),
        "corr(s_tree_parented, projected_response_action)": corr(
            df["s_tree_parented"], df["projected_response_action"]
        ),
        "corr(projected_response_weight, eymh_parented_prefactor_diag)": corr(
            df["projected_response_weight"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(projected_response_action, eymh_parented_prefactor_diag)": corr(
            df["projected_response_action"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(projected_response_action, abs_mu_diff_vs_const)": corr(
            df["projected_response_action"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_response_action_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_response_action_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_response_action_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_response_action_audit_run_meta_{args.tag}.json"

    df[
        [
            "D",
            "eta",
            "coeff_l1",
            "coeff_l2",
            "delta_f_norm",
            "coeff_participation_access_parented",
            "tree_diag_susceptibility_parented",
            "s_tree_parented",
            "tree_diag_compressibility_parented",
            "projected_response_action",
            "projected_response_det",
            "projected_response_weight",
            "projected_response_weight_ref",
            "projected_response_weight_residual",
            "shell_spread",
            "shell_access_parented",
            "hk_loop_local_prefactor_diag",
            "eymh_parented_prefactor_diag",
            "projected_response_prefactor_recon",
            "projected_response_prefactor_residual",
            "coeff_action_share",
            "tree_action_share",
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
    axes[0].scatter(df["delta_f_norm"], df["s_tree_parented"], s=12, alpha=0.7)
    axes[0].set_xlabel("DeltaF_norm")
    axes[0].set_ylabel("S_tree = log(1 + chi_tree)")
    axes[0].set_title("Participation vs tree action")

    axes[1].scatter(df["projected_response_action"], df["eymh_parented_prefactor_diag"], s=12, alpha=0.7)
    axes[1].set_xlabel("S_resp = DeltaF_norm + S_tree")
    axes[1].set_ylabel("eymh_parented_prefactor_diag")
    axes[1].set_title("Response action vs prefactor")

    axes[2].scatter(df["coeff_action_share"], df["tree_action_share"], s=12, alpha=0.7)
    axes[2].set_xlabel("participation share")
    axes[2].set_ylabel("tree share")
    axes[2].set_title("Action-share balance")

    fig.suptitle("EYMH projected response-action audit", fontsize=13)
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

#!/usr/bin/env python3
"""
Audit the parent-action-side source factors behind the canonical
`action_loop_eymh_absolute` prefactor.

We refine the previous decomposition

  eymh_loop_prefactor_diag
    = hk_loop_local_prefactor_diag
    * shell_access
    * align_access
    * diag_dispersion_screen

by resolving the last two pieces into more primitive source factors:

  align_access = coeff_participation_access
               = N_eff^(-1/4),
  N_eff = (coeff_l1 / coeff_l2)^2

  diag_dispersion_screen
    = shell_gap_screen * tree_diag_screen
    = (1 + gap_cv)^(-1/2)
      * sqrt((1+gap_cv)/(1+gap_cv+c_tree_diag_cv))

This gives an exact five-factor reconstruction:

  eymh_loop_prefactor_diag
    = hk_loop_local_prefactor_diag
    * shell_access
    * coeff_participation_access
    * shell_gap_screen
    * tree_diag_screen
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
PAPER_DIR = ROOT / "paper"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit source factors behind EYMH absolute loop prefactor.")
    ap.add_argument(
        "--eymh-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_absolute_D21E21.csv",
    )
    ap.add_argument(
        "--constant-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_constant_refresh_D21E21.csv",
    )
    ap.add_argument("--tag", type=str, default="D21E21")
    return ap.parse_args()


def arr_summary(arr: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
        }
    return {
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    eymh = pd.read_csv(args.eymh_map)
    const = pd.read_csv(args.constant_map)[["D", "eta", "mu_mumu_uv_rge"]].rename(
        columns={"mu_mumu_uv_rge": "mu_mumu_uv_rge_constant"}
    )
    df = eymh.merge(const, on=["D", "eta"])

    floor = 1e-30
    df["shell_access"] = np.sqrt(np.maximum(df["shell_spread"], floor) / (1.0 + np.maximum(df["shell_spread"], floor)))
    l1 = np.maximum(df["coeff_l1"].to_numpy(dtype=float), 0.0)
    l2 = np.maximum(df["coeff_l2"].to_numpy(dtype=float), 0.0)
    coeff_participation = np.full(len(df), np.inf, dtype=float)
    valid = (l1 > floor) & (l2 > floor)
    coeff_participation[valid] = (l1[valid] / l2[valid]) ** 2
    coeff_participation_access = np.zeros(len(df), dtype=float)
    coeff_participation_access[valid] = np.power(np.maximum(coeff_participation[valid], floor), -0.25)
    df["coeff_participation"] = coeff_participation
    df["coeff_participation_access"] = coeff_participation_access
    df["align_access"] = np.sqrt(np.maximum(df["coeff_align"], floor))
    df["shell_gap_screen"] = 1.0 / np.sqrt(1.0 + np.maximum(df["gap_cv"], 0.0))
    df["tree_diag_screen"] = np.sqrt(
        (1.0 + np.maximum(df["gap_cv"], 0.0))
        / (1.0 + np.maximum(df["gap_cv"], 0.0) + np.maximum(df["c_tree_diag_cv"], 0.0))
    )
    df["diag_dispersion_screen"] = 1.0 / np.sqrt(
        1.0 + np.maximum(df["gap_cv"], 0.0) + np.maximum(df["c_tree_diag_cv"], 0.0)
    )
    df["eymh_prefactor_source_recon"] = (
        df["hk_loop_local_prefactor_diag"]
        * df["shell_access"]
        * df["coeff_participation_access"]
        * df["shell_gap_screen"]
        * df["tree_diag_screen"]
    )
    df["eymh_source_residual"] = df["eymh_loop_prefactor_diag"] - df["eymh_prefactor_source_recon"]
    df["abs_mu_diff_vs_const"] = (df["mu_mumu_uv_rge"] - df["mu_mumu_uv_rge_constant"]).abs()

    summary = {
        "n_points": int(len(df)),
        **arr_summary(df["coeff_participation"].to_numpy(), "coeff_participation"),
        **arr_summary(df["coeff_participation_access"].to_numpy(), "coeff_participation_access"),
        **arr_summary(df["shell_gap_screen"].to_numpy(), "shell_gap_screen"),
        **arr_summary(df["tree_diag_screen"].to_numpy(), "tree_diag_screen"),
        **arr_summary(df["eymh_loop_prefactor_diag"].to_numpy(), "eymh_loop_prefactor_diag"),
        **arr_summary(df["abs_mu_diff_vs_const"].to_numpy(), "abs_mu_diff_vs_const"),
        "max_abs_source_recon_residual": float(np.max(np.abs(df["eymh_source_residual"]))),
        "corr(coeff_participation_access, align_access)": corr(df["coeff_participation_access"], df["align_access"]),
        "corr(coeff_participation_access, eymh_loop_prefactor_diag)": corr(df["coeff_participation_access"], df["eymh_loop_prefactor_diag"]),
        "corr(shell_gap_screen, eymh_loop_prefactor_diag)": corr(df["shell_gap_screen"], df["eymh_loop_prefactor_diag"]),
        "corr(tree_diag_screen, eymh_loop_prefactor_diag)": corr(df["tree_diag_screen"], df["eymh_loop_prefactor_diag"]),
        "corr(coeff_participation_access, abs_mu_diff_vs_const)": corr(df["coeff_participation_access"], df["abs_mu_diff_vs_const"]),
        "corr(shell_gap_screen, abs_mu_diff_vs_const)": corr(df["shell_gap_screen"], df["abs_mu_diff_vs_const"]),
        "corr(tree_diag_screen, abs_mu_diff_vs_const)": corr(df["tree_diag_screen"], df["abs_mu_diff_vs_const"]),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_source_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_source_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_source_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_source_audit_run_meta_{args.tag}.json"

    keep = [
        "D", "eta",
        "hk_loop_local_prefactor_diag",
        "shell_access",
        "coeff_l1", "coeff_l2", "coeff_align",
        "coeff_participation", "coeff_participation_access",
        "gap_cv", "c_tree_diag_cv",
        "shell_gap_screen", "tree_diag_screen", "diag_dispersion_screen",
        "eymh_loop_prefactor_diag", "eymh_prefactor_source_recon", "eymh_source_residual",
        "abs_mu_diff_vs_const",
    ]
    df[keep].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].scatter(df["coeff_participation_access"], df["align_access"], s=12, alpha=0.7)
    axes[0].set_xlabel("coeff_participation_access")
    axes[0].set_ylabel("align_access")
    axes[0].set_title("Participation access vs align_access")

    axes[1].scatter(df["shell_gap_screen"], df["tree_diag_screen"], s=12, alpha=0.7)
    axes[1].set_xlabel("shell_gap_screen")
    axes[1].set_ylabel("tree_diag_screen")
    axes[1].set_title("Dispersion-screen source factors")

    axes[2].scatter(df["coeff_participation_access"], df["eymh_loop_prefactor_diag"], s=12, alpha=0.7)
    axes[2].set_xlabel("coeff_participation_access")
    axes[2].set_ylabel("eymh_loop_prefactor_diag")
    axes[2].set_title("Alignment source vs EYMH prefactor")

    axes[3].hist(df["eymh_source_residual"], bins=30, color="#4C78A8", alpha=0.85)
    axes[3].set_xlabel("source reconstruction residual")
    axes[3].set_ylabel("count")
    axes[3].set_title("Source-level reconstruction residual")

    fig.suptitle("EYMH absolute prefactor source audit", fontsize=13)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    out_meta.write_text(json.dumps({
        "eymh_map": str(args.eymh_map),
        "constant_map": str(args.constant_map),
        "tag": args.tag,
        "n_points": int(len(df)),
    }, indent=2))

    for src in [out_map, out_summary, out_fig, out_meta]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Decompose the action_loop_eymh_absolute prefactor into its constituent factors.

This audit is comparator-only. It verifies that the EYMH-side absolute loop
prefactor exported in the canonical UV map is internally consistent with the
constructed factors

  eymh_loop_prefactor_diag
    = hk_loop_local_prefactor_diag
    * shell_access
    * align_access
    * diag_dispersion_screen

and quantifies which factors drive its variation across the D21xE21 map.
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
    ap = argparse.ArgumentParser(description="Decompose EYMH absolute loop prefactor on the canonical D21xE21 map.")
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
    ap.add_argument(
        "--loop-absolute-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_absolute_D21E21.csv",
    )
    ap.add_argument("--tag", type=str, default="D21E21")
    ap.add_argument("--skip-paper-copy", action="store_true")
    return ap.parse_args()


def arr_summary(arr: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float).reshape(-1)
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
    loop_abs = pd.read_csv(args.loop_absolute_map)[["D", "eta", "mu_mumu_uv_rge"]].rename(
        columns={"mu_mumu_uv_rge": "mu_mumu_uv_rge_action_loop_absolute"}
    )
    df = eymh.merge(const, on=["D", "eta"]).merge(loop_abs, on=["D", "eta"])

    floor = 1e-30
    df["shell_access"] = np.sqrt(np.maximum(df["shell_spread"], floor) / (1.0 + np.maximum(df["shell_spread"], floor)))
    df["align_access"] = np.sqrt(np.maximum(df["coeff_align"], floor))
    df["diag_dispersion_screen"] = 1.0 / np.sqrt(1.0 + np.maximum(df["gap_cv"], 0.0) + np.maximum(df["c_tree_diag_cv"], 0.0))
    df["eymh_prefactor_recon"] = (
        df["hk_loop_local_prefactor_diag"]
        * df["shell_access"]
        * df["align_access"]
        * df["diag_dispersion_screen"]
    )
    df["eymh_prefactor_residual"] = df["eymh_loop_prefactor_diag"] - df["eymh_prefactor_recon"]
    df["abs_mu_diff_vs_const"] = (df["mu_mumu_uv_rge"] - df["mu_mumu_uv_rge_constant"]).abs()
    df["abs_mu_diff_vs_loop_abs"] = (df["mu_mumu_uv_rge"] - df["mu_mumu_uv_rge_action_loop_absolute"]).abs()

    # Log-space additive decomposition where finite.
    for col in [
        "hk_loop_local_prefactor_diag",
        "shell_access",
        "align_access",
        "diag_dispersion_screen",
        "eymh_loop_prefactor_diag",
    ]:
        df[f"log10_{col}"] = np.log10(np.maximum(df[col], floor))
    df["log10_prefactor_residual"] = (
        df["log10_eymh_loop_prefactor_diag"]
        - df["log10_hk_loop_local_prefactor_diag"]
        - df["log10_shell_access"]
        - df["log10_align_access"]
        - df["log10_diag_dispersion_screen"]
    )

    summary = {
        "n_points": int(len(df)),
        **arr_summary(df["eymh_loop_prefactor_diag"].to_numpy(), "eymh_loop_prefactor_diag"),
        **arr_summary(df["hk_loop_local_prefactor_diag"].to_numpy(), "hk_loop_local_prefactor_diag"),
        **arr_summary(df["shell_access"].to_numpy(), "shell_access"),
        **arr_summary(df["align_access"].to_numpy(), "align_access"),
        **arr_summary(df["diag_dispersion_screen"].to_numpy(), "diag_dispersion_screen"),
        **arr_summary(df["abs_mu_diff_vs_const"].to_numpy(), "abs_mu_diff_vs_const"),
        **arr_summary(df["abs_mu_diff_vs_loop_abs"].to_numpy(), "abs_mu_diff_vs_loop_abs"),
        "max_abs_prefactor_recon_residual": float(np.max(np.abs(df["eymh_prefactor_residual"]))),
        "max_abs_log10_prefactor_residual": float(np.max(np.abs(df["log10_prefactor_residual"]))),
        "corr(hk_loop_local_prefactor_diag, eymh_loop_prefactor_diag)": corr(
            df["hk_loop_local_prefactor_diag"], df["eymh_loop_prefactor_diag"]
        ),
        "corr(shell_access, eymh_loop_prefactor_diag)": corr(df["shell_access"], df["eymh_loop_prefactor_diag"]),
        "corr(align_access, eymh_loop_prefactor_diag)": corr(df["align_access"], df["eymh_loop_prefactor_diag"]),
        "corr(diag_dispersion_screen, eymh_loop_prefactor_diag)": corr(
            df["diag_dispersion_screen"], df["eymh_loop_prefactor_diag"]
        ),
        "corr(eymh_loop_prefactor_diag, abs_mu_diff_vs_const)": corr(
            df["eymh_loop_prefactor_diag"], df["abs_mu_diff_vs_const"]
        ),
        "corr(hk_loop_local_prefactor_diag, abs_mu_diff_vs_const)": corr(
            df["hk_loop_local_prefactor_diag"], df["abs_mu_diff_vs_const"]
        ),
        "corr(shell_access, abs_mu_diff_vs_const)": corr(df["shell_access"], df["abs_mu_diff_vs_const"]),
        "corr(align_access, abs_mu_diff_vs_const)": corr(df["align_access"], df["abs_mu_diff_vs_const"]),
        "corr(diag_dispersion_screen, abs_mu_diff_vs_const)": corr(
            df["diag_dispersion_screen"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_prefactor_decomposition_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_prefactor_decomposition_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_prefactor_decomposition_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_prefactor_decomposition_run_meta_{args.tag}.json"

    keep = [
        "D",
        "eta",
        "eymh_loop_prefactor_diag",
        "hk_loop_local_prefactor_diag",
        "shell_access",
        "align_access",
        "diag_dispersion_screen",
        "eymh_prefactor_recon",
        "eymh_prefactor_residual",
        "log10_hk_loop_local_prefactor_diag",
        "log10_shell_access",
        "log10_align_access",
        "log10_diag_dispersion_screen",
        "log10_eymh_loop_prefactor_diag",
        "log10_prefactor_residual",
        "abs_mu_diff_vs_const",
        "abs_mu_diff_vs_loop_abs",
        "shell_spread",
        "coeff_align",
        "gap_cv",
        "c_tree_diag_cv",
    ]
    df[keep].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].scatter(df["hk_loop_local_prefactor_diag"], df["eymh_loop_prefactor_diag"], s=12, alpha=0.7)
    axes[0].set_xlabel("hk_loop_local_prefactor_diag")
    axes[0].set_ylabel("eymh_loop_prefactor_diag")
    axes[0].set_title("Local loop piece vs EYMH prefactor")

    axes[1].scatter(df["shell_access"], df["eymh_loop_prefactor_diag"], s=12, alpha=0.7)
    axes[1].set_xlabel("shell_access")
    axes[1].set_ylabel("eymh_loop_prefactor_diag")
    axes[1].set_title("Shell access vs EYMH prefactor")

    axes[2].scatter(df["eymh_loop_prefactor_diag"], df["abs_mu_diff_vs_const"], s=12, alpha=0.7)
    axes[2].set_xlabel("eymh_loop_prefactor_diag")
    axes[2].set_ylabel("|Δμ| vs constant")
    axes[2].set_title("Prefactor vs map deformation")

    axes[3].hist(df["log10_prefactor_residual"], bins=30, color="#4C78A8", alpha=0.85)
    axes[3].set_xlabel("log10 reconstruction residual")
    axes[3].set_ylabel("count")
    axes[3].set_title("Log-space prefactor reconstruction residual")

    fig.suptitle("EYMH absolute loop-prefactor decomposition", fontsize=13)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    out_meta.write_text(
        json.dumps(
            {
                "eymh_map": str(args.eymh_map),
                "constant_map": str(args.constant_map),
                "loop_absolute_map": str(args.loop_absolute_map),
                "tag": args.tag,
                "n_points": int(len(df)),
            },
            indent=2,
        )
    )

    for src in [out_map, out_summary, out_fig, out_meta]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Audit the parent-action susceptibility/compressibility witness behind the
tree-diagonal screening factor in the canonical `action_loop_eymh_absolute`
prefactor.

We rewrite

  tree_diag_screen
    = sqrt((1 + gap_cv) / (1 + gap_cv + c_tree_diag_cv))

as

  tree_diag_screen = (1 + chi_tree_diag)^(-1/2),
  chi_tree_diag = c_tree_diag_cv / (1 + gap_cv),

which can be read as a shell-background-normalized diagonal susceptibility.
The corresponding compressibility witness is exactly `tree_diag_screen`.
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
    ap = argparse.ArgumentParser(
        description="Audit tree-diagonal susceptibility/compressibility witness for EYMH absolute prefactor."
    )
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
    gap_headroom = 1.0 / (1.0 + np.maximum(df["gap_cv"].to_numpy(dtype=float), 0.0))
    tree_diag_susceptibility = np.maximum(df["c_tree_diag_cv"].to_numpy(dtype=float), 0.0) * gap_headroom
    tree_diag_compressibility = np.power(1.0 + tree_diag_susceptibility, -0.5)
    tree_diag_screen = tree_diag_compressibility.copy()
    diag_dispersion_screen = 1.0 / np.sqrt(
        1.0
        + np.maximum(df["gap_cv"].to_numpy(dtype=float), 0.0)
        + np.maximum(df["c_tree_diag_cv"].to_numpy(dtype=float), 0.0)
    )
    tree_diag_pressure_fraction = np.zeros(len(df), dtype=float)
    denom = (
        1.0
        + np.maximum(df["gap_cv"].to_numpy(dtype=float), 0.0)
        + np.maximum(df["c_tree_diag_cv"].to_numpy(dtype=float), 0.0)
    )
    valid = denom > floor
    tree_diag_pressure_fraction[valid] = (
        np.maximum(df["c_tree_diag_cv"].to_numpy(dtype=float)[valid], 0.0) / denom[valid]
    )

    df["tree_diag_gap_headroom"] = gap_headroom
    df["tree_diag_susceptibility"] = tree_diag_susceptibility
    df["tree_diag_compressibility"] = tree_diag_compressibility
    df["tree_diag_screen"] = tree_diag_screen
    df["diag_dispersion_screen"] = diag_dispersion_screen
    df["tree_diag_pressure_fraction"] = tree_diag_pressure_fraction
    df["tree_diag_recon_residual"] = df["tree_diag_screen"] - df["tree_diag_compressibility"]
    df["abs_mu_diff_vs_const"] = (df["mu_mumu_uv_rge"] - df["mu_mumu_uv_rge_constant"]).abs()

    summary = {
        "n_points": int(len(df)),
        **arr_summary(df["tree_diag_gap_headroom"].to_numpy(), "tree_diag_gap_headroom"),
        **arr_summary(df["tree_diag_susceptibility"].to_numpy(), "tree_diag_susceptibility"),
        **arr_summary(df["tree_diag_compressibility"].to_numpy(), "tree_diag_compressibility"),
        **arr_summary(df["tree_diag_pressure_fraction"].to_numpy(), "tree_diag_pressure_fraction"),
        **arr_summary(df["abs_mu_diff_vs_const"].to_numpy(), "abs_mu_diff_vs_const"),
        "max_abs_tree_diag_recon_residual": float(np.max(np.abs(df["tree_diag_recon_residual"]))),
        "corr(tree_diag_susceptibility, tree_diag_screen)": corr(
            df["tree_diag_susceptibility"], df["tree_diag_screen"]
        ),
        "corr(tree_diag_compressibility, tree_diag_screen)": corr(
            df["tree_diag_compressibility"], df["tree_diag_screen"]
        ),
        "corr(tree_diag_susceptibility, eymh_loop_prefactor_diag)": corr(
            df["tree_diag_susceptibility"], df["eymh_loop_prefactor_diag"]
        ),
        "corr(tree_diag_compressibility, eymh_loop_prefactor_diag)": corr(
            df["tree_diag_compressibility"], df["eymh_loop_prefactor_diag"]
        ),
        "corr(tree_diag_pressure_fraction, eymh_loop_prefactor_diag)": corr(
            df["tree_diag_pressure_fraction"], df["eymh_loop_prefactor_diag"]
        ),
        "corr(tree_diag_susceptibility, abs_mu_diff_vs_const)": corr(
            df["tree_diag_susceptibility"], df["abs_mu_diff_vs_const"]
        ),
        "corr(tree_diag_compressibility, abs_mu_diff_vs_const)": corr(
            df["tree_diag_compressibility"], df["abs_mu_diff_vs_const"]
        ),
        "corr(tree_diag_pressure_fraction, abs_mu_diff_vs_const)": corr(
            df["tree_diag_pressure_fraction"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_tree_diag_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_tree_diag_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_tree_diag_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_tree_diag_audit_run_meta_{args.tag}.json"

    keep = [
        "D",
        "eta",
        "gap_cv",
        "c_tree_diag_cv",
        "tree_diag_gap_headroom",
        "tree_diag_susceptibility",
        "tree_diag_compressibility",
        "tree_diag_pressure_fraction",
        "tree_diag_screen",
        "diag_dispersion_screen",
        "eymh_loop_prefactor_diag",
        "tree_diag_recon_residual",
        "abs_mu_diff_vs_const",
    ]
    df[keep].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].scatter(df["tree_diag_susceptibility"], df["tree_diag_screen"], s=12, alpha=0.7)
    axes[0].set_xlabel("tree_diag_susceptibility")
    axes[0].set_ylabel("tree_diag_screen")
    axes[0].set_title("Susceptibility vs tree_diag_screen")

    axes[1].scatter(df["tree_diag_compressibility"], df["tree_diag_screen"], s=12, alpha=0.7)
    axes[1].set_xlabel("tree_diag_compressibility")
    axes[1].set_ylabel("tree_diag_screen")
    axes[1].set_title("Compressibility witness")

    axes[2].scatter(df["tree_diag_pressure_fraction"], df["eymh_loop_prefactor_diag"], s=12, alpha=0.7)
    axes[2].set_xlabel("tree_diag_pressure_fraction")
    axes[2].set_ylabel("eymh_loop_prefactor_diag")
    axes[2].set_title("Tree-diag pressure vs EYMH prefactor")

    axes[3].hist(df["tree_diag_recon_residual"], bins=30, color="#4C78A8", alpha=0.85)
    axes[3].set_xlabel("tree_diag reconstruction residual")
    axes[3].set_ylabel("count")
    axes[3].set_title("Compressibility reconstruction residual")

    fig.suptitle("EYMH tree-diagonal susceptibility/compressibility audit", fontsize=13)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    out_meta.write_text(
        json.dumps(
            {
                "eymh_map": str(args.eymh_map),
                "constant_map": str(args.constant_map),
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

#!/usr/bin/env python3
"""
Exact projected parent-kernel selection statement for the canonical EYMH block.

This audit pushes the projected effective-action story one step further.
Instead of only comparing the direct mismatch functional

    J(alpha,beta,lambda) = < (A(alpha,beta,lambda) - A_ref)^2 >,

to local surrogates, we rewrite the same quantity as an exact projected
parent-kernel excess functional.

For the minimal deformed kernel family

    K11 = exp(alpha S_part),
    K22 = exp(beta  S_schur),
    K12 = lambda * sqrt((K11 - 1)(K22 - 1)),

the canonical parented block sits at (alpha,beta,lambda)=(1,1,0) with

    A_ref = exp[-(S_part + S_schur)/2].

Defining the exact projected kernel excess

    Delta S_kernel
      = (alpha - 1) S_part
      + (beta  - 1) S_schur
      + log(1 - lambda^2 xi_cross),

where xi_cross = ((K_part - 1)(G_schur - 1)) / (K_part G_schur),
the deformed response weight is exactly

    A(alpha,beta,lambda) = A_ref * exp[-Delta S_kernel / 2].

Therefore the exact mismatch functional can be written identically as

    J_parent = < A_ref^2 (exp[-Delta S_kernel / 2] - 1)^2 >.

This script verifies that exact identity and summarizes how the canonical
selection arises from linear log-det / Schur deviations plus a positive,
even mixing penalty.
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


def kernel_weight(
    s_part: np.ndarray,
    s_schur: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
) -> np.ndarray:
    k11 = np.exp(float(alpha) * s_part)
    k22 = np.exp(float(beta) * s_schur)
    k12 = float(lam) * np.sqrt(np.maximum((k11 - 1.0) * (k22 - 1.0), 0.0))
    det = np.maximum(k11 * k22 - k12**2, 1e-30)
    return np.exp(-0.5 * np.log(det))


def deformed_cross_xi(
    s_part: np.ndarray,
    s_schur: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    k11 = np.exp(float(alpha) * s_part)
    k22 = np.exp(float(beta) * s_schur)
    return np.maximum((k11 - 1.0) * (k22 - 1.0), 0.0) / np.maximum(k11 * k22, 1e-30)


def parse_grid(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit the exact projected parent-kernel excess functional behind the canonical EYMH logdet/Schur block."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--alpha-grid", default="0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2")
    ap.add_argument("--beta-grid", default="0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2")
    ap.add_argument("--lambda-grid", default="-0.6,-0.4,-0.2,-0.1,0.0,0.1,0.2,0.4,0.6")
    args = ap.parse_args()

    alphas = parse_grid(args.alpha_grid)
    betas = parse_grid(args.beta_grid)
    lambdas = parse_grid(args.lambda_grid)

    df = pd.read_csv(args.parented_map)
    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    k_part = coeff_l1 / coeff_l2
    s_part = np.log(k_part)

    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    g_schur = 1.0 + chi_tree
    s_schur = np.log(g_schur)

    a_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )
    xi_cross = np.maximum((k_part - 1.0) * (g_schur - 1.0), 0.0) / np.maximum(k_part * g_schur, 1e-30)

    rows: list[dict[str, float]] = []
    for alpha in alphas:
        for beta in betas:
            for lam in lambdas:
                w_direct = kernel_weight(s_part, s_schur, alpha, beta, lam)
                delta_linear = (alpha - 1.0) * s_part + (beta - 1.0) * s_schur
                xi_ab = deformed_cross_xi(s_part, s_schur, alpha, beta)
                delta_mix = np.log(np.maximum(1.0 - (lam**2) * xi_ab, 1e-30))
                delta_total = delta_linear + delta_mix
                w_stmt = a_ref * np.exp(-0.5 * delta_total)
                j_direct = float(np.mean((w_direct - a_ref) ** 2))
                j_stmt = float(np.mean((a_ref * (np.exp(-0.5 * delta_total) - 1.0)) ** 2))
                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "lambda": lam,
                        "j_direct": j_direct,
                        "j_parent_statement": j_stmt,
                        "abs_objective_identity_residual": abs(j_direct - j_stmt),
                        "max_abs_weight_identity_residual": float(np.max(np.abs(w_direct - w_stmt))),
                        "mean_delta_linear": float(np.mean(delta_linear)),
                        "mean_delta_mix": float(np.mean(delta_mix)),
                        "mean_delta_total": float(np.mean(delta_total)),
                        "p90_delta_mix": q(delta_mix, 0.9),
                        "mean_abs_delta_linear": float(np.mean(np.abs(delta_linear))),
                        "mean_abs_delta_mix": float(np.mean(np.abs(delta_mix))),
                        "mean_xi_cross_deformed": float(np.mean(xi_ab)),
                    }
                )

    scan = pd.DataFrame(rows)
    canonical = scan[
        (np.isclose(scan["alpha"], 1.0)) & (np.isclose(scan["beta"], 1.0)) & (np.isclose(scan["lambda"], 0.0))
    ].iloc[0]
    best = scan.sort_values(["j_parent_statement", "abs_objective_identity_residual"]).iloc[0]
    runner_up = scan.sort_values(["j_parent_statement", "abs_objective_identity_residual"]).iloc[1]
    lam_slice = scan[(scan["alpha"] == 1.0) & (scan["beta"] == 1.0)].sort_values("lambda")

    # Canonical map-level quantities for the statement itself.
    df["parent_kernel_k_part"] = k_part
    df["parent_kernel_g_schur"] = g_schur
    df["parent_kernel_s_part"] = s_part
    df["parent_kernel_s_schur"] = s_schur
    df["parent_kernel_response_action"] = s_part + s_schur
    df["parent_kernel_response_weight"] = a_ref
    df["parent_kernel_cross_xi"] = xi_cross

    coeff_share = s_part / np.maximum(s_part + s_schur, 1e-30)
    tree_share = s_schur / np.maximum(s_part + s_schur, 1e-30)
    df["parent_kernel_coeff_share"] = coeff_share
    df["parent_kernel_tree_share"] = tree_share

    summary = {
        "n_scan_points": int(len(scan)),
        "best_alpha": float(best["alpha"]),
        "best_beta": float(best["beta"]),
        "best_lambda": float(best["lambda"]),
        "canonical_objective": float(canonical["j_parent_statement"]),
        "canonical_identity_residual": float(canonical["abs_objective_identity_residual"]),
        "max_abs_objective_identity_residual": float(scan["abs_objective_identity_residual"].max()),
        "p95_abs_objective_identity_residual": q(scan["abs_objective_identity_residual"], 0.95),
        "max_abs_weight_identity_residual": float(scan["max_abs_weight_identity_residual"].max()),
        "runner_up_alpha": float(runner_up["alpha"]),
        "runner_up_beta": float(runner_up["beta"]),
        "runner_up_lambda": float(runner_up["lambda"]),
        "selection_gap_parent_objective": float(runner_up["j_parent_statement"] - canonical["j_parent_statement"]),
        "lambda_slice_max_parent_objective": float(lam_slice["j_parent_statement"].max()),
        "lambda_slice_p95_parent_objective": q(lam_slice["j_parent_statement"], 0.95),
        "canonical_coeff_share_p50": q(coeff_share, 0.5),
        "canonical_tree_share_p50": q(tree_share, 0.5),
        "canonical_cross_xi_p50": q(xi_cross, 0.5),
        "canonical_cross_xi_p90": q(xi_cross, 0.9),
        "corr(j_direct, j_parent_statement)": corr(scan["j_direct"], scan["j_parent_statement"]),
        "corr(mean_delta_total, j_parent_statement)": corr(scan["mean_delta_total"], scan["j_parent_statement"]),
        "corr(mean_delta_mix, j_parent_statement)": corr(scan["mean_delta_mix"], scan["j_parent_statement"]),
    }

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_parent_kernel_statement_scan_{args.tag}.csv"
    out_map = OUTDIR / f"hll_uv_action_loop_eymh_parent_kernel_statement_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_kernel_statement_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_kernel_statement_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_kernel_statement_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    df[
        [
            "D",
            "eta",
            "coeff_l1",
            "coeff_l2",
            "parent_kernel_k_part",
            "parent_kernel_g_schur",
            "parent_kernel_s_part",
            "parent_kernel_s_schur",
            "parent_kernel_response_action",
            "parent_kernel_response_weight",
            "parent_kernel_cross_xi",
            "parent_kernel_coeff_share",
            "parent_kernel_tree_share",
            "eymh_parented_prefactor_diag",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "alpha_grid": alphas,
                "beta_grid": betas,
                "lambda_grid": lambdas,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].scatter(scan["j_direct"], scan["j_parent_statement"], s=18, alpha=0.7)
    maxv = float(max(scan["j_direct"].max(), scan["j_parent_statement"].max(), 1e-12))
    axes[0].plot([0.0, maxv], [0.0, maxv], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("direct mismatch J")
    axes[0].set_ylabel("parent-kernel statement objective")
    axes[0].set_title("Exact objective identity")

    axes[1].plot(lam_slice["lambda"], lam_slice["mean_delta_mix"], marker="o")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("mean mixing penalty")
    axes[1].set_title("Even positive mixing penalty")

    axes[2].scatter(df["parent_kernel_coeff_share"], df["parent_kernel_tree_share"], s=12, alpha=0.7)
    axes[2].set_xlabel("coefficient share")
    axes[2].set_ylabel("tree/Schur share")
    axes[2].set_title("Canonical action shares")

    fig.suptitle("EYMH projected parent-kernel statement audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_scan, out_map, out_summary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

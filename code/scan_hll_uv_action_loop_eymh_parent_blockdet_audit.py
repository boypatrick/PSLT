#!/usr/bin/env python3
"""
Projected parent block-determinant / Schur-complement derivation audit.

This script turns the current EYMH parented kernel statement into an explicit
background-normalized parent-block determinant identity.

For a deformed projected parent block

    K_parent =
        [[K11, lambda * C_parent],
         [lambda * C_parent, K22]],

with

    K11 = exp(alpha S_part),
    K_bg = 1 + gap_cv,
    G_beta = exp(beta S_schur),
    K22 = K_bg * G_beta,
    C_parent = sqrt((K11 - 1) (K22 - K_bg)),

the parent block determinant obeys

    det(K_parent) / K_bg = K11 * G_beta * (1 - lambda^2 xi_cross),

and the normalized Schur complement of the lower block is

    G_schur,eff = (K22 - K12^2 / K11) / K_bg
                = G_beta * (1 - lambda^2 xi_cross).

Therefore the projected response action is exactly

    S_parent = log K11 + log G_schur,eff,

which explains why the canonical parented prefactor is read as a
logdet(participation) + Schur(tree/background) kernel.
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


def parse_grid(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit the projected parent block-determinant / Schur-complement derivation behind the EYMH parented kernel."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--alpha-grid", default="0.9,0.95,1.0,1.05,1.1")
    ap.add_argument("--beta-grid", default="0.9,0.95,1.0,1.05,1.1")
    ap.add_argument("--lambda-grid", default="-0.4,-0.2,-0.1,0.0,0.1,0.2,0.4")
    args = ap.parse_args()

    alphas = parse_grid(args.alpha_grid)
    betas = parse_grid(args.beta_grid)
    lambdas = parse_grid(args.lambda_grid)

    df = pd.read_csv(args.parented_map)
    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    k_part = coeff_l1 / coeff_l2
    s_part = np.log(k_part)

    k_bg = 1.0 + np.maximum(np.asarray(df["gap_cv"], dtype=float), 0.0)
    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    g_schur = 1.0 + chi_tree
    s_schur = np.log(g_schur)

    a_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )
    s_ref = -2.0 * np.log(np.maximum(a_ref, 1e-30))

    # Canonical parent block at (alpha,beta,lambda)=(1,1,0)
    k22_canonical = k_bg * g_schur
    c_parent_canonical = np.sqrt(np.maximum((k_part - 1.0) * (k22_canonical - k_bg), 0.0))
    det_parent_canonical = k_part * k22_canonical
    schur_norm_canonical = k22_canonical / np.maximum(k_bg, 1e-30)
    s_parent_canonical = np.log(np.maximum(det_parent_canonical / np.maximum(k_bg, 1e-30), 1e-30))

    df["parent_blockdet_k11"] = k_part
    df["parent_blockdet_k_bg"] = k_bg
    df["parent_blockdet_k22"] = k22_canonical
    df["parent_blockdet_c_parent"] = c_parent_canonical
    df["parent_blockdet_det_ratio"] = det_parent_canonical / np.maximum(k_bg, 1e-30)
    df["parent_blockdet_schur_norm"] = schur_norm_canonical
    df["parent_blockdet_action"] = s_parent_canonical
    df["parent_blockdet_action_residual"] = s_parent_canonical - s_ref
    df["parent_blockdet_coeff_share"] = s_part / np.maximum(s_ref, 1e-30)
    df["parent_blockdet_tree_share"] = s_schur / np.maximum(s_ref, 1e-30)

    rows: list[dict[str, float]] = []
    for alpha in alphas:
        for beta in betas:
            for lam in lambdas:
                k11 = np.exp(float(alpha) * s_part)
                g_beta = np.exp(float(beta) * s_schur)
                k22 = k_bg * g_beta
                c_parent = np.sqrt(np.maximum((k11 - 1.0) * (k22 - k_bg), 0.0))
                k12 = float(lam) * c_parent

                det_parent = np.maximum(k11 * k22 - k12**2, 1e-30)
                det_ratio_direct = det_parent / np.maximum(k_bg, 1e-30)

                xi_cross = np.maximum((k11 - 1.0) * (g_beta - 1.0), 0.0) / np.maximum(k11 * g_beta, 1e-30)
                det_ratio_split = k11 * g_beta * np.maximum(1.0 - (lam**2) * xi_cross, 1e-30)

                schur_norm_direct = (k22 - (k12**2) / np.maximum(k11, 1e-30)) / np.maximum(k_bg, 1e-30)
                schur_norm_split = g_beta * np.maximum(1.0 - (lam**2) * xi_cross, 1e-30)

                s_direct = np.log(np.maximum(det_ratio_direct, 1e-30))
                s_split = np.log(np.maximum(k11, 1e-30)) + np.log(np.maximum(schur_norm_split, 1e-30))

                w_direct = np.exp(-0.5 * s_direct)
                w_split = np.exp(-0.5 * s_split)
                j_direct = float(np.mean((w_direct - a_ref) ** 2))
                j_blockdet = float(np.mean((w_split - a_ref) ** 2))

                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "lambda": lam,
                        "j_direct": j_direct,
                        "j_blockdet": j_blockdet,
                        "max_abs_det_ratio_residual": float(np.max(np.abs(det_ratio_direct - det_ratio_split))),
                        "max_abs_schur_norm_residual": float(np.max(np.abs(schur_norm_direct - schur_norm_split))),
                        "max_abs_action_factor_residual": float(np.max(np.abs(s_direct - s_split))),
                        "max_abs_weight_factor_residual": float(np.max(np.abs(w_direct - w_split))),
                        "mean_logdet_part": float(np.mean(np.log(np.maximum(k11, 1e-30)))),
                        "mean_log_schur_eff": float(np.mean(np.log(np.maximum(schur_norm_split, 1e-30)))),
                        "mean_even_penalty": float(np.mean(np.log(np.maximum(1.0 - (lam**2) * xi_cross, 1e-30)))),
                        "mean_xi_cross": float(np.mean(xi_cross)),
                    }
                )

    scan = pd.DataFrame(rows)
    canonical = scan[
        (np.isclose(scan["alpha"], 1.0)) & (np.isclose(scan["beta"], 1.0)) & (np.isclose(scan["lambda"], 0.0))
    ].iloc[0]

    summary = {
        "n_scan_points": int(len(scan)),
        "canonical_objective": float(canonical["j_blockdet"]),
        "max_abs_canonical_action_residual": float(np.max(np.abs(df["parent_blockdet_action_residual"]))),
        "p95_abs_canonical_action_residual": q(np.abs(df["parent_blockdet_action_residual"]), 0.95),
        "corr(parent_blockdet_action, response_action)": corr(df["parent_blockdet_action"], s_ref),
        "canonical_coeff_share_p50": q(df["parent_blockdet_coeff_share"], 0.5),
        "canonical_tree_share_p50": q(df["parent_blockdet_tree_share"], 0.5),
        "lambda0_max_abs_det_ratio_residual": float(canonical["max_abs_det_ratio_residual"]),
        "lambda0_max_abs_schur_norm_residual": float(canonical["max_abs_schur_norm_residual"]),
        "lambda0_max_abs_action_factor_residual": float(canonical["max_abs_action_factor_residual"]),
        "scan_max_abs_det_ratio_residual": float(scan["max_abs_det_ratio_residual"].max()),
        "scan_max_abs_schur_norm_residual": float(scan["max_abs_schur_norm_residual"].max()),
        "scan_max_abs_action_factor_residual": float(scan["max_abs_action_factor_residual"].max()),
        "scan_max_abs_weight_factor_residual": float(scan["max_abs_weight_factor_residual"].max()),
        "corr(j_direct, j_blockdet)": corr(scan["j_direct"], scan["j_blockdet"]),
        "selection_gap_blockdet_objective": float(
            scan.sort_values(["j_blockdet", "max_abs_action_factor_residual"]).iloc[1]["j_blockdet"]
            - canonical["j_blockdet"]
        ),
    }

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_parent_blockdet_scan_{args.tag}.csv"
    out_map = OUTDIR / f"hll_uv_action_loop_eymh_parent_blockdet_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_blockdet_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_blockdet_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_blockdet_audit_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    df[
        [
            "D",
            "eta",
            "parent_blockdet_k11",
            "parent_blockdet_k_bg",
            "parent_blockdet_k22",
            "parent_blockdet_c_parent",
            "parent_blockdet_det_ratio",
            "parent_blockdet_schur_norm",
            "parent_blockdet_action",
            "parent_blockdet_action_residual",
            "parent_blockdet_coeff_share",
            "parent_blockdet_tree_share",
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

    axes[0].scatter(scan["j_direct"], scan["j_blockdet"], s=18, alpha=0.7)
    maxv = float(max(scan["j_direct"].max(), scan["j_blockdet"].max(), 1e-12))
    axes[0].plot([0.0, maxv], [0.0, maxv], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("direct objective")
    axes[0].set_ylabel("blockdet objective")
    axes[0].set_title("Exact objective identity")

    axes[1].scatter(df["parent_blockdet_coeff_share"], df["parent_blockdet_tree_share"], s=12, alpha=0.7)
    axes[1].set_xlabel("participation share")
    axes[1].set_ylabel("tree/Schur share")
    axes[1].set_title("Canonical action shares")

    lam_slice = scan[(scan["alpha"] == 1.0) & (scan["beta"] == 1.0)].sort_values("lambda")
    axes[2].plot(lam_slice["lambda"], lam_slice["mean_even_penalty"], marker="o")
    axes[2].set_xlabel("lambda")
    axes[2].set_ylabel("mean even penalty")
    axes[2].set_title("Schur mixing penalty")

    fig.suptitle("EYMH projected parent block-determinant / Schur audit", fontsize=13)
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

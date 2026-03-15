#!/usr/bin/env python3
"""
Ratio-warp minimality audit for the projected EYMH parent mixing block.

We probe the first nontrivial extension beyond the minimal symmetric-excess
family by allowing the canonical geometric-mean parent block

    C_parent = sqrt(E_part E_tree)

to be multiplied by a smooth ratio-dependent warp,

    C_warp = kappa * C_parent * exp(delta * L + nu * L^2),
    L = 1/2 log(E_part / E_tree),

where

    E_part = K11 - 1,
    E_tree = K22 - K_bg.

If the projected parent-action selection is truly minimal, the exact closure of
the block-determinant / Schur factorization should force the unwarped point

    (kappa, delta, nu) = (1, 0, 0).

This audit verifies that expectation on the canonical D21xE21 fix map.
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
        description="Audit whether ratio-warp deformations of the EYMH parent mixing block preserve exact projected closure."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--kappa-grid", default="0.9,1.0,1.1")
    ap.add_argument("--delta-grid", default="-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3")
    ap.add_argument("--nu-grid", default="-0.1,-0.05,0.0,0.05,0.1")
    ap.add_argument("--alpha-grid", default="0.9,1.0,1.1")
    ap.add_argument("--beta-grid", default="0.9,1.0,1.1")
    ap.add_argument("--lambda-grid", default="-0.2,-0.1,0.0,0.1,0.2")
    args = ap.parse_args()

    kappa_grid = parse_grid(args.kappa_grid)
    delta_grid = parse_grid(args.delta_grid)
    nu_grid = parse_grid(args.nu_grid)
    alpha_grid = parse_grid(args.alpha_grid)
    beta_grid = parse_grid(args.beta_grid)
    lambda_grid = parse_grid(args.lambda_grid)

    df = pd.read_csv(args.parented_map)
    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    s_part_base = np.log(coeff_l1 / coeff_l2)
    k_bg = 1.0 + np.maximum(np.asarray(df["gap_cv"], dtype=float), 0.0)
    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    s_schur_base = np.log1p(chi_tree)
    a_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )

    rows: list[dict[str, float]] = []
    for kappa in kappa_grid:
        for delta in delta_grid:
            for nu in nu_grid:
                max_abs_det_ratio_residual = 0.0
                max_abs_schur_residual = 0.0
                max_abs_weight_residual = 0.0
                xi_target_values = []
                xi_warp_values = []
                j_values = []

                for alpha in alpha_grid:
                    for beta in beta_grid:
                        s_part = float(alpha) * s_part_base
                        s_schur = float(beta) * s_schur_base
                        k11 = np.exp(s_part)
                        g_beta = np.exp(s_schur)
                        k22 = k_bg * g_beta

                        e_part = np.maximum(k11 - 1.0, 1e-30)
                        e_tree = np.maximum(k22 - k_bg, 1e-30)
                        lratio = 0.5 * np.log(e_part / e_tree)
                        c_parent = np.sqrt(e_part * e_tree)
                        c_warp = float(kappa) * c_parent * np.exp(float(delta) * lratio + float(nu) * (lratio**2))

                        xi_target = (c_parent**2) / np.maximum(k_bg * k11 * g_beta, 1e-30)
                        xi_warp = (c_warp**2) / np.maximum(k_bg * k11 * g_beta, 1e-30)

                        for lam in lambda_grid:
                            k12_target = float(lam) * c_parent
                            det_ratio_target = (k11 * k22 - k12_target**2) / np.maximum(k_bg, 1e-30)
                            schur_target = (k22 - (k12_target**2) / np.maximum(k11, 1e-30)) / np.maximum(k_bg, 1e-30)
                            w_target = np.exp(-0.5 * np.log(np.maximum(det_ratio_target, 1e-30)))

                            k12_warp = float(lam) * c_warp
                            det_ratio_warp = (k11 * k22 - k12_warp**2) / np.maximum(k_bg, 1e-30)
                            schur_warp = (k22 - (k12_warp**2) / np.maximum(k11, 1e-30)) / np.maximum(k_bg, 1e-30)
                            w_warp = np.exp(-0.5 * np.log(np.maximum(det_ratio_warp, 1e-30)))

                            max_abs_det_ratio_residual = max(
                                max_abs_det_ratio_residual,
                                float(np.max(np.abs(det_ratio_warp - det_ratio_target))),
                            )
                            max_abs_schur_residual = max(
                                max_abs_schur_residual,
                                float(np.max(np.abs(schur_warp - schur_target))),
                            )
                            max_abs_weight_residual = max(
                                max_abs_weight_residual,
                                float(np.max(np.abs(w_warp - w_target))),
                            )
                            j_values.append(float(np.mean((w_warp - a_ref) ** 2)))

                        xi_target_values.append(xi_target)
                        xi_warp_values.append(xi_warp)

                xi_target_concat = np.concatenate(xi_target_values)
                xi_warp_concat = np.concatenate(xi_warp_values)
                rows.append(
                    {
                        "kappa": kappa,
                        "delta": delta,
                        "nu": nu,
                        "mean_objective": float(np.mean(j_values)),
                        "max_objective": float(np.max(j_values)),
                        "max_abs_det_ratio_residual": max_abs_det_ratio_residual,
                        "max_abs_schur_residual": max_abs_schur_residual,
                        "max_abs_weight_residual": max_abs_weight_residual,
                        "mean_abs_xi_residual": float(np.mean(np.abs(xi_warp_concat - xi_target_concat))),
                        "max_abs_xi_residual": float(np.max(np.abs(xi_warp_concat - xi_target_concat))),
                        "corr(xi_warp, xi_target)": corr(xi_warp_concat, xi_target_concat),
                    }
                )

    scan = pd.DataFrame(rows).sort_values(
        ["max_abs_det_ratio_residual", "max_abs_schur_residual", "max_abs_weight_residual", "mean_objective"]
    )
    best = scan.iloc[0]
    canonical = scan[
        np.isclose(scan["kappa"], 1.0) & np.isclose(scan["delta"], 0.0) & np.isclose(scan["nu"], 0.0)
    ].iloc[0]
    runner_up = scan.iloc[1]

    # Canonical map-level warp variables.
    k_part = np.exp(s_part_base)
    g_schur = np.exp(s_schur_base)
    k22 = k_bg * g_schur
    e_part = np.maximum(k_part - 1.0, 1e-30)
    e_tree = np.maximum(k22 - k_bg, 1e-30)
    lratio = 0.5 * np.log(e_part / e_tree)
    c_parent = np.sqrt(e_part * e_tree)
    xi_target = (c_parent**2) / np.maximum(k_bg * k_part * g_schur, 1e-30)

    df["parent_ratio_e_part"] = e_part
    df["parent_ratio_e_tree"] = e_tree
    df["parent_ratio_log_ratio"] = lratio
    df["parent_ratio_geomean_mix"] = c_parent
    df["parent_ratio_xi_target"] = xi_target

    summary = {
        "n_scan_points": int(len(scan)),
        "best_kappa": float(best["kappa"]),
        "best_delta": float(best["delta"]),
        "best_nu": float(best["nu"]),
        "canonical_max_abs_det_ratio_residual": float(canonical["max_abs_det_ratio_residual"]),
        "canonical_max_abs_schur_residual": float(canonical["max_abs_schur_residual"]),
        "canonical_max_abs_weight_residual": float(canonical["max_abs_weight_residual"]),
        "canonical_max_abs_xi_residual": float(canonical["max_abs_xi_residual"]),
        "canonical_corr_xi": float(canonical["corr(xi_warp, xi_target)"]),
        "runner_up_kappa": float(runner_up["kappa"]),
        "runner_up_delta": float(runner_up["delta"]),
        "runner_up_nu": float(runner_up["nu"]),
        "runner_up_max_abs_det_ratio_residual": float(runner_up["max_abs_det_ratio_residual"]),
        "runner_up_max_abs_schur_residual": float(runner_up["max_abs_schur_residual"]),
        "runner_up_max_abs_weight_residual": float(runner_up["max_abs_weight_residual"]),
        "runner_up_max_abs_xi_residual": float(runner_up["max_abs_xi_residual"]),
        "selection_gap_det_ratio_residual": float(
            runner_up["max_abs_det_ratio_residual"] - canonical["max_abs_det_ratio_residual"]
        ),
        "selection_gap_schur_residual": float(
            runner_up["max_abs_schur_residual"] - canonical["max_abs_schur_residual"]
        ),
        "selection_gap_weight_residual": float(
            runner_up["max_abs_weight_residual"] - canonical["max_abs_weight_residual"]
        ),
        "selection_gap_xi_residual": float(runner_up["max_abs_xi_residual"] - canonical["max_abs_xi_residual"]),
        "canonical_log_ratio_p50": q(lratio, 0.5),
        "canonical_log_ratio_p90": q(np.abs(lratio), 0.9),
    }

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_parent_ratio_warp_scan_{args.tag}.csv"
    out_map = OUTDIR / f"hll_uv_action_loop_eymh_parent_ratio_warp_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_ratio_warp_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_ratio_warp_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_ratio_warp_audit_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    df[
        [
            "D",
            "eta",
            "parent_ratio_e_part",
            "parent_ratio_e_tree",
            "parent_ratio_log_ratio",
            "parent_ratio_geomean_mix",
            "parent_ratio_xi_target",
            "eymh_parented_prefactor_diag",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "kappa_grid": kappa_grid,
                "delta_grid": delta_grid,
                "nu_grid": nu_grid,
                "alpha_grid": alpha_grid,
                "beta_grid": beta_grid,
                "lambda_grid": lambda_grid,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    heat = scan.pivot_table(index="delta", columns="nu", values="max_abs_det_ratio_residual", aggfunc="min")
    im = axes[0].imshow(
        heat.values,
        origin="lower",
        aspect="auto",
        extent=[heat.columns.min(), heat.columns.max(), heat.index.min(), heat.index.max()],
    )
    axes[0].set_xlabel("nu")
    axes[0].set_ylabel("delta")
    axes[0].set_title("min max|det residual| over kappa")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    kslice = scan[np.isclose(scan["delta"], 0.0) & np.isclose(scan["nu"], 0.0)].sort_values("kappa")
    axes[1].plot(kslice["kappa"], kslice["max_abs_weight_residual"], marker="o")
    axes[1].set_xlabel("kappa")
    axes[1].set_ylabel("max |weight residual|")
    axes[1].set_title("Unwarped slice")

    axes[2].hist(df["parent_ratio_log_ratio"], bins=25, alpha=0.8)
    axes[2].set_xlabel("0.5 log(E_part / E_tree)")
    axes[2].set_ylabel("count")
    axes[2].set_title("Canonical ratio-warp variable")

    fig.suptitle("EYMH parent ratio-warp minimality audit", fontsize=13)
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

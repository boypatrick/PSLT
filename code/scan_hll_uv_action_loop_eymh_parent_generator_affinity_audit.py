#!/usr/bin/env python3
"""
Local log-generator affinity audit for the projected EYMH parent mixing block.

After the symmetry/normalization audit fixes the canonical parent family at

    (m, s, a) = (0, 1, 0),

the remaining proof obligation is whether the projected parent action naturally
selects the *multiplicative excess class* itself, i.e. whether the logarithm of
the mixed block is affine in the local excess coordinates

    L_sum  = log(E_part E_tree),
    L_diff = log(E_part / E_tree).

This script probes the first non-affine local extension:

    log C_gen
      = 1/2 L_sum
      + q_ss L_sum^2
      + q_dd L_diff^2
      + q_sd L_sum L_diff.

The canonical multiplicative class corresponds to

    (q_ss, q_dd, q_sd) = (0, 0, 0).

If the projected parent action really lands in the local multiplicative excess
class, this point should be the unique exact member of the nearby quadratic
log-generator family.
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
        description="Audit whether the EYMH projected parent action selects a locally affine log-generator for the mixed block."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--qss-grid", default="-0.025,-0.0125,0.0,0.0125,0.025")
    ap.add_argument("--qdd-grid", default="-0.025,-0.0125,0.0,0.0125,0.025")
    ap.add_argument("--qsd-grid", default="-0.025,-0.0125,0.0,0.0125,0.025")
    ap.add_argument("--alpha-grid", default="0.9,1.0,1.1")
    ap.add_argument("--beta-grid", default="0.9,1.0,1.1")
    ap.add_argument("--lambda-grid", default="-0.2,-0.1,0.0,0.1,0.2")
    args = ap.parse_args()

    qss_grid = parse_grid(args.qss_grid)
    qdd_grid = parse_grid(args.qdd_grid)
    qsd_grid = parse_grid(args.qsd_grid)
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
    for qss in qss_grid:
        for qdd in qdd_grid:
            for qsd in qsd_grid:
                max_abs_det_ratio_residual = 0.0
                max_abs_schur_residual = 0.0
                max_abs_weight_residual = 0.0
                xi_target_values = []
                xi_gen_values = []
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
                        l_sum = np.log(e_part * e_tree)
                        l_diff = np.log(e_part / e_tree)

                        c_parent = np.sqrt(e_part * e_tree)
                        log_c_gen = 0.5 * l_sum + float(qss) * (l_sum**2) + float(qdd) * (l_diff**2) + float(qsd) * (
                            l_sum * l_diff
                        )
                        log_c_gen = np.clip(log_c_gen, -80.0, 80.0)
                        c_gen = np.exp(log_c_gen)

                        xi_target = (c_parent**2) / np.maximum(k_bg * k11 * g_beta, 1e-30)
                        xi_gen = (c_gen**2) / np.maximum(k_bg * k11 * g_beta, 1e-30)

                        for lam in lambda_grid:
                            k12_target = float(lam) * c_parent
                            det_ratio_target = (k11 * k22 - k12_target**2) / np.maximum(k_bg, 1e-30)
                            schur_target = (k22 - (k12_target**2) / np.maximum(k11, 1e-30)) / np.maximum(k_bg, 1e-30)
                            w_target = np.exp(-0.5 * np.log(np.maximum(det_ratio_target, 1e-30)))

                            k12_gen = float(lam) * c_gen
                            det_ratio_gen = (k11 * k22 - k12_gen**2) / np.maximum(k_bg, 1e-30)
                            schur_gen = (k22 - (k12_gen**2) / np.maximum(k11, 1e-30)) / np.maximum(k_bg, 1e-30)
                            w_gen = np.exp(-0.5 * np.log(np.maximum(det_ratio_gen, 1e-30)))

                            max_abs_det_ratio_residual = max(
                                max_abs_det_ratio_residual,
                                float(np.max(np.abs(det_ratio_gen - det_ratio_target))),
                            )
                            max_abs_schur_residual = max(
                                max_abs_schur_residual,
                                float(np.max(np.abs(schur_gen - schur_target))),
                            )
                            max_abs_weight_residual = max(
                                max_abs_weight_residual,
                                float(np.max(np.abs(w_gen - w_target))),
                            )
                            j_values.append(float(np.mean((w_gen - a_ref) ** 2)))

                        xi_target_values.append(xi_target)
                        xi_gen_values.append(xi_gen)

                xi_target_concat = np.concatenate(xi_target_values)
                xi_gen_concat = np.concatenate(xi_gen_values)
                rows.append(
                    {
                        "q_ss": qss,
                        "q_dd": qdd,
                        "q_sd": qsd,
                        "mean_objective": float(np.mean(j_values)),
                        "max_objective": float(np.max(j_values)),
                        "max_abs_det_ratio_residual": max_abs_det_ratio_residual,
                        "max_abs_schur_residual": max_abs_schur_residual,
                        "max_abs_weight_residual": max_abs_weight_residual,
                        "mean_abs_xi_residual": float(np.mean(np.abs(xi_gen_concat - xi_target_concat))),
                        "max_abs_xi_residual": float(np.max(np.abs(xi_gen_concat - xi_target_concat))),
                        "corr(xi_gen, xi_target)": corr(xi_gen_concat, xi_target_concat),
                    }
                )

    scan = pd.DataFrame(rows)
    finite_mask = np.isfinite(
        scan[
            [
                "mean_objective",
                "max_objective",
                "max_abs_det_ratio_residual",
                "max_abs_schur_residual",
                "max_abs_weight_residual",
                "mean_abs_xi_residual",
                "max_abs_xi_residual",
            ]
        ]
    ).all(axis=1)
    scan = scan.loc[finite_mask].sort_values(
        ["max_abs_det_ratio_residual", "max_abs_schur_residual", "max_abs_weight_residual", "mean_objective"]
    )
    best = scan.iloc[0]
    canonical = scan[
        np.isclose(scan["q_ss"], 0.0) & np.isclose(scan["q_dd"], 0.0) & np.isclose(scan["q_sd"], 0.0)
    ].iloc[0]
    runner_up = scan.iloc[1]

    k_part = np.exp(s_part_base)
    g_schur = np.exp(s_schur_base)
    k22 = k_bg * g_schur
    e_part = np.maximum(k_part - 1.0, 1e-30)
    e_tree = np.maximum(k22 - k_bg, 1e-30)
    l_sum = np.log(e_part * e_tree)
    l_diff = np.log(e_part / e_tree)
    c_parent = np.sqrt(e_part * e_tree)
    xi_target = (c_parent**2) / np.maximum(k_bg * k_part * g_schur, 1e-30)

    df["parent_affine_e_part"] = e_part
    df["parent_affine_e_tree"] = e_tree
    df["parent_affine_log_sum"] = l_sum
    df["parent_affine_log_diff"] = l_diff
    df["parent_affine_log_canonical"] = 0.5 * l_sum
    df["parent_affine_c_parent"] = c_parent
    df["parent_affine_xi_target"] = xi_target

    summary = {
        "n_scan_points": int(len(scan)),
        "best_q_ss": float(best["q_ss"]),
        "best_q_dd": float(best["q_dd"]),
        "best_q_sd": float(best["q_sd"]),
        "canonical_max_abs_det_ratio_residual": float(canonical["max_abs_det_ratio_residual"]),
        "canonical_max_abs_schur_residual": float(canonical["max_abs_schur_residual"]),
        "canonical_max_abs_weight_residual": float(canonical["max_abs_weight_residual"]),
        "canonical_max_abs_xi_residual": float(canonical["max_abs_xi_residual"]),
        "canonical_corr_xi": float(canonical["corr(xi_gen, xi_target)"]),
        "runner_up_q_ss": float(runner_up["q_ss"]),
        "runner_up_q_dd": float(runner_up["q_dd"]),
        "runner_up_q_sd": float(runner_up["q_sd"]),
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
        "canonical_log_sum_p50": q(l_sum, 0.5),
        "canonical_abs_log_diff_p90": q(np.abs(l_diff), 0.9),
    }

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_parent_generator_affinity_scan_{args.tag}.csv"
    out_map = OUTDIR / f"hll_uv_action_loop_eymh_parent_generator_affinity_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_generator_affinity_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_generator_affinity_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_generator_affinity_audit_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    df[
        [
            "D",
            "eta",
            "parent_affine_e_part",
            "parent_affine_e_tree",
            "parent_affine_log_sum",
            "parent_affine_log_diff",
            "parent_affine_log_canonical",
            "parent_affine_c_parent",
            "parent_affine_xi_target",
            "eymh_parented_prefactor_diag",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "qss_grid": qss_grid,
                "qdd_grid": qdd_grid,
                "qsd_grid": qsd_grid,
                "alpha_grid": alpha_grid,
                "beta_grid": beta_grid,
                "lambda_grid": lambda_grid,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].scatter(scan["q_ss"], scan["max_abs_det_ratio_residual"], s=20, alpha=0.75)
    axes[0].axvline(0.0, linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("q_ss")
    axes[0].set_ylabel("max |Δ(det/K_bg)|")
    axes[0].set_title("Symmetric log-curvature penalty")

    axes[1].scatter(scan["q_dd"], scan["max_abs_xi_residual"], s=20, alpha=0.75)
    axes[1].axvline(0.0, linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("q_dd")
    axes[1].set_ylabel("max |Δxi|")
    axes[1].set_title("Asymmetry-curvature penalty")

    axes[2].scatter(scan["q_sd"], scan["max_abs_weight_residual"], s=20, alpha=0.75)
    axes[2].axvline(0.0, linestyle="--", color="black", linewidth=1)
    axes[2].set_xlabel("q_sd")
    axes[2].set_ylabel("max |ΔA|")
    axes[2].set_title("Mixed log-curvature penalty")

    fig.suptitle("EYMH projected parent generator-affinity audit", fontsize=13)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    for src in [out_scan, out_map, out_summary, out_fig, out_meta]:
        dst = PAPERDIR / src.name
        dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Normal-coordinate audit for the EYMH projected parent excess variables.

After fixing the parent-action fixed points

    E_part = K11 - 1,
    E_tree = K22 - K_bg,

and showing that the same tangent-normalized Box-Cox family still uniquely
selects the linear excess coordinates, the remaining proof obligation is more
positive than negative:

why are these coordinates the *natural* local variables of the projected parent
action rather than merely the survivor of successive exclusion tests?

This script reframes the question in normal-coordinate language. We probe the
first nonlinear local jet extension that preserves the same fixed points and the
same unit tangent normalization:

    E_part^(zeta_p) = E_part + zeta_p E_part^2,
    E_tree^(zeta_t) = E_tree + zeta_t E_tree^2 / K_bg.

The canonical excess coordinates correspond to

    (zeta_p, zeta_t) = (0, 0),

which is exactly the zero-second-jet, zero-curvature normal-coordinate choice.
If the projected parent action naturally uses the linear excess variables, this
point should again be the unique exact choice.
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
        description="Audit zero-curvature normal coordinates for the EYMH projected parent action."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--zeta-part-grid", default="-1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0")
    ap.add_argument("--zeta-tree-grid", default="-1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0")
    ap.add_argument("--alpha-grid", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--beta-grid", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--lambda-grid", default="-0.2,-0.1,0.0,0.1,0.2")
    args = ap.parse_args()

    zeta_part_grid = parse_grid(args.zeta_part_grid)
    zeta_tree_grid = parse_grid(args.zeta_tree_grid)
    alpha_grid = parse_grid(args.alpha_grid)
    beta_grid = parse_grid(args.beta_grid)
    lambda_grid = parse_grid(args.lambda_grid)
    alpha_positive = sorted([x for x in alpha_grid if x > 0.0])
    beta_positive = sorted([x for x in beta_grid if x > 0.0])
    alpha_onset = alpha_positive[0] if alpha_positive else None
    beta_onset = beta_positive[0] if beta_positive else None

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
    for zeta_part in zeta_part_grid:
        for zeta_tree in zeta_tree_grid:
            max_abs_det_ratio_residual = 0.0
            max_abs_schur_residual = 0.0
            max_abs_weight_residual = 0.0
            xi_target_values = []
            xi_gen_values = []
            j_values = []
            part_anchor_leak = 0.0
            tree_anchor_leak = 0.0
            part_onset_residual = 0.0
            tree_onset_residual = 0.0

            for alpha in alpha_grid:
                for beta in beta_grid:
                    s_part = float(alpha) * s_part_base
                    s_schur = float(beta) * s_schur_base
                    k11 = np.exp(s_part)
                    g_beta = np.exp(s_schur)
                    k22 = k_bg * g_beta

                    e_part_target = np.maximum(k11 - 1.0, 0.0)
                    e_tree_target = np.maximum(k22 - k_bg, 0.0)
                    c_target = np.sqrt(e_part_target * e_tree_target)

                    e_part_gen = np.maximum(e_part_target + float(zeta_part) * e_part_target**2, 0.0)
                    e_tree_gen = np.maximum(
                        e_tree_target + float(zeta_tree) * (e_tree_target**2) / np.maximum(k_bg, 1e-30),
                        0.0,
                    )
                    c_gen = np.sqrt(e_part_gen * e_tree_gen)

                    xi_target = (c_target**2) / np.maximum(k_bg * k11 * g_beta, 1e-30)
                    xi_gen = (c_gen**2) / np.maximum(k_bg * k11 * g_beta, 1e-30)

                    if np.isclose(alpha, 0.0):
                        part_anchor_leak = max(part_anchor_leak, float(np.max(np.abs(c_gen))))
                    if np.isclose(beta, 0.0):
                        tree_anchor_leak = max(tree_anchor_leak, float(np.max(np.abs(c_gen))))
                    if alpha_onset is not None and np.isclose(alpha, alpha_onset):
                        part_onset_residual = max(part_onset_residual, float(np.max(np.abs(c_gen - c_target))))
                    if beta_onset is not None and np.isclose(beta, beta_onset):
                        tree_onset_residual = max(tree_onset_residual, float(np.max(np.abs(c_gen - c_target))))

                    for lam in lambda_grid:
                        k12_target = float(lam) * c_target
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
                    "zeta_part": zeta_part,
                    "zeta_tree": zeta_tree,
                    "mean_objective": float(np.mean(j_values)),
                    "max_objective": float(np.max(j_values)),
                    "max_abs_det_ratio_residual": max_abs_det_ratio_residual,
                    "max_abs_schur_residual": max_abs_schur_residual,
                    "max_abs_weight_residual": max_abs_weight_residual,
                    "mean_abs_xi_residual": float(np.mean(np.abs(xi_gen_concat - xi_target_concat))),
                    "max_abs_xi_residual": float(np.max(np.abs(xi_gen_concat - xi_target_concat))),
                    "corr(xi_gen, xi_target)": corr(xi_gen_concat, xi_target_concat),
                    "part_anchor_leak": part_anchor_leak,
                    "tree_anchor_leak": tree_anchor_leak,
                    "part_onset_residual": part_onset_residual,
                    "tree_onset_residual": tree_onset_residual,
                }
            )

    scan = pd.DataFrame(rows).sort_values(
        [
            "part_anchor_leak",
            "tree_anchor_leak",
            "part_onset_residual",
            "tree_onset_residual",
            "max_abs_det_ratio_residual",
            "max_abs_schur_residual",
            "max_abs_weight_residual",
            "mean_objective",
        ]
    )
    best = scan.iloc[0]
    canonical = scan[
        np.isclose(scan["zeta_part"], 0.0) & np.isclose(scan["zeta_tree"], 0.0)
    ].iloc[0]
    runner_up = scan.iloc[1]

    k_part = np.exp(s_part_base)
    g_schur = np.exp(s_schur_base)
    k22 = k_bg * g_schur
    e_part_target = np.maximum(k_part - 1.0, 0.0)
    e_tree_target = np.maximum(k22 - k_bg, 0.0)
    c_target = np.sqrt(e_part_target * e_tree_target)
    xi_target = (c_target**2) / np.maximum(k_bg * k_part * g_schur, 1e-30)

    df["parent_normal_e_part"] = e_part_target
    df["parent_normal_e_tree"] = e_tree_target
    df["parent_normal_c_parent"] = c_target
    df["parent_normal_xi_target"] = xi_target
    df["parent_normal_part_curvature"] = np.zeros_like(e_part_target)
    df["parent_normal_tree_curvature"] = np.zeros_like(e_tree_target)

    summary = {
        "n_scan_points": int(len(scan)),
        "best_zeta_part": float(best["zeta_part"]),
        "best_zeta_tree": float(best["zeta_tree"]),
        "canonical_max_abs_det_ratio_residual": float(canonical["max_abs_det_ratio_residual"]),
        "canonical_max_abs_schur_residual": float(canonical["max_abs_schur_residual"]),
        "canonical_max_abs_weight_residual": float(canonical["max_abs_weight_residual"]),
        "canonical_max_abs_xi_residual": float(canonical["max_abs_xi_residual"]),
        "canonical_part_anchor_leak": float(canonical["part_anchor_leak"]),
        "canonical_tree_anchor_leak": float(canonical["tree_anchor_leak"]),
        "runner_up_zeta_part": float(runner_up["zeta_part"]),
        "runner_up_zeta_tree": float(runner_up["zeta_tree"]),
        "runner_up_max_abs_det_ratio_residual": float(runner_up["max_abs_det_ratio_residual"]),
        "runner_up_max_abs_schur_residual": float(runner_up["max_abs_schur_residual"]),
        "runner_up_max_abs_weight_residual": float(runner_up["max_abs_weight_residual"]),
        "runner_up_max_abs_xi_residual": float(runner_up["max_abs_xi_residual"]),
        "runner_up_part_anchor_leak": float(runner_up["part_anchor_leak"]),
        "runner_up_tree_anchor_leak": float(runner_up["tree_anchor_leak"]),
        "canonical_part_onset_residual": float(canonical["part_onset_residual"]),
        "canonical_tree_onset_residual": float(canonical["tree_onset_residual"]),
        "runner_up_part_onset_residual": float(runner_up["part_onset_residual"]),
        "runner_up_tree_onset_residual": float(runner_up["tree_onset_residual"]),
        "selection_gap_det_ratio_residual": float(
            runner_up["max_abs_det_ratio_residual"] - canonical["max_abs_det_ratio_residual"]
        ),
        "selection_gap_schur_residual": float(
            runner_up["max_abs_schur_residual"] - canonical["max_abs_schur_residual"]
        ),
        "selection_gap_weight_residual": float(
            runner_up["max_abs_weight_residual"] - canonical["max_abs_weight_residual"]
        ),
        "selection_gap_xi_residual": float(
            runner_up["max_abs_xi_residual"] - canonical["max_abs_xi_residual"]
        ),
        "selection_gap_part_onset_residual": float(
            runner_up["part_onset_residual"] - canonical["part_onset_residual"]
        ),
        "selection_gap_tree_onset_residual": float(
            runner_up["tree_onset_residual"] - canonical["tree_onset_residual"]
        ),
        "canonical_second_jet_part": 0.0,
        "canonical_second_jet_tree": 0.0,
        "runner_up_second_jet_part": float(2.0 * runner_up["zeta_part"]),
        "runner_up_second_jet_tree_p50": float(np.median(2.0 * runner_up["zeta_tree"] / np.maximum(k_bg, 1e-30))),
        "alpha_onset": float(alpha_onset) if alpha_onset is not None else float("nan"),
        "beta_onset": float(beta_onset) if beta_onset is not None else float("nan"),
    }

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_normal_coordinate_scan_{args.tag}.csv"
    out_map = OUTDIR / f"hll_uv_action_loop_eymh_normal_coordinate_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_normal_coordinate_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_normal_coordinate_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_normal_coordinate_audit_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    df[
        [
            "D",
            "eta",
            "parent_normal_e_part",
            "parent_normal_e_tree",
            "parent_normal_c_parent",
            "parent_normal_xi_target",
            "parent_normal_part_curvature",
            "parent_normal_tree_curvature",
            "eymh_parented_prefactor_diag",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "zeta_part_grid": zeta_part_grid,
                "zeta_tree_grid": zeta_tree_grid,
                "alpha_grid": alpha_grid,
                "beta_grid": beta_grid,
                "lambda_grid": lambda_grid,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].scatter(scan["zeta_part"], scan["part_onset_residual"], s=20, alpha=0.75)
    axes[0].axvline(0.0, linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("zeta_part")
    axes[0].set_ylabel("part-onset residual")
    axes[0].set_title("Participation normal-coordinate jet")

    axes[1].scatter(scan["zeta_tree"], scan["tree_onset_residual"], s=20, alpha=0.75)
    axes[1].axvline(0.0, linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("zeta_tree")
    axes[1].set_ylabel("tree-onset residual")
    axes[1].set_title("Tree/background normal-coordinate jet")

    axes[2].scatter(scan["max_abs_det_ratio_residual"], scan["max_abs_weight_residual"], s=20, alpha=0.75)
    axes[2].set_xlabel("max det-ratio residual")
    axes[2].set_ylabel("max weight residual")
    axes[2].set_title("Closure degradation")

    fig.suptitle("EYMH projected parent normal-coordinate audit", fontsize=13)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    for src in [out_scan, out_map, out_summary, out_fig, out_meta]:
        dst = PAPERDIR / src.name
        dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()

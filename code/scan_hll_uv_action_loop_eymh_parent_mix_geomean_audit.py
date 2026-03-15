#!/usr/bin/env python3
"""
Geometric-mean uniqueness audit for the projected EYMH parent mixing block.

We test the smallest symmetric excess family for the off-diagonal parent block:

    C_gen = kappa * (K11 - 1)^u * (K22 - K_bg)^v

with

    K11 = exp(alpha S_part),
    K22 = K_bg * exp(beta S_schur),
    K_bg = 1 + gap_cv.

The canonical parent block used in the EYMH derivation corresponds to

    (u, v, kappa) = (1/2, 1/2, 1),

namely the geometric mean of the participation and tree/background excesses:

    C_parent = sqrt((K11 - 1) (K22 - K_bg)).

This audit shows that, in a neighborhood of the canonical kernel, that point is
the unique member of the symmetric excess family that preserves the exact
background-normalized determinant / Schur factorization behind the parented
response action.
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
        description="Audit the uniqueness of the geometric-mean parent mixing block in the EYMH projected kernel."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--u-grid", default="0.25,0.375,0.5,0.625,0.75")
    ap.add_argument("--v-grid", default="0.25,0.375,0.5,0.625,0.75")
    ap.add_argument("--kappa-grid", default="0.8,0.9,1.0,1.1,1.2")
    ap.add_argument("--alpha-grid", default="0.9,1.0,1.1")
    ap.add_argument("--beta-grid", default="0.9,1.0,1.1")
    ap.add_argument("--lambda-grid", default="-0.2,-0.1,0.0,0.1,0.2")
    args = ap.parse_args()

    u_grid = parse_grid(args.u_grid)
    v_grid = parse_grid(args.v_grid)
    kappa_grid = parse_grid(args.kappa_grid)
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
    for u in u_grid:
        for v in v_grid:
            for kappa in kappa_grid:
                max_abs_det_ratio_residual = 0.0
                max_abs_schur_residual = 0.0
                max_abs_weight_residual = 0.0
                j_values = []
                xi_target_values = []
                xi_gen_values = []

                for alpha in alpha_grid:
                    for beta in beta_grid:
                        s_part = float(alpha) * s_part_base
                        s_schur = float(beta) * s_schur_base
                        k11 = np.exp(s_part)
                        g_beta = np.exp(s_schur)
                        k22 = k_bg * g_beta

                        target_c = np.sqrt(np.maximum((k11 - 1.0) * (k22 - k_bg), 0.0))
                        xi_target = np.maximum((k11 - 1.0) * (g_beta - 1.0), 0.0) / np.maximum(
                            k11 * g_beta, 1e-30
                        )

                        c_gen = float(kappa) * np.power(np.maximum(k11 - 1.0, 0.0), float(u)) * np.power(
                            np.maximum(k22 - k_bg, 0.0), float(v)
                        )

                        for lam in lambda_grid:
                            k12_target = float(lam) * target_c
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

                        xi_gen = (c_gen**2) / np.maximum(k_bg * k11 * g_beta, 1e-30)
                        xi_target_values.append(xi_target)
                        xi_gen_values.append(xi_gen)

                xi_target_concat = np.concatenate(xi_target_values)
                xi_gen_concat = np.concatenate(xi_gen_values)
                rows.append(
                    {
                        "u": u,
                        "v": v,
                        "kappa": kappa,
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

    scan = pd.DataFrame(rows).sort_values(
        ["max_abs_det_ratio_residual", "max_abs_schur_residual", "max_abs_weight_residual", "mean_objective"]
    )
    best = scan.iloc[0]
    canonical = scan[
        np.isclose(scan["u"], 0.5) & np.isclose(scan["v"], 0.5) & np.isclose(scan["kappa"], 1.0)
    ].iloc[0]
    runner_up = scan.iloc[1]

    # Canonical map-level geometric-mean structure.
    k_part = np.exp(s_part_base)
    g_schur = np.exp(s_schur_base)
    k22 = k_bg * g_schur
    excess_part = np.maximum(k_part - 1.0, 0.0)
    excess_tree = np.maximum(k22 - k_bg, 0.0)
    c_parent = np.sqrt(excess_part * excess_tree)
    xi_target = (c_parent**2) / np.maximum(k_bg * k_part * g_schur, 1e-30)

    df["parent_mix_excess_part"] = excess_part
    df["parent_mix_excess_tree"] = excess_tree
    df["parent_mix_geomean"] = c_parent
    df["parent_mix_xi_target"] = xi_target
    df["parent_mix_norm_ratio"] = np.sqrt(np.maximum(excess_tree, 0.0) / np.maximum(excess_part, 1e-30))

    summary = {
        "n_scan_points": int(len(scan)),
        "best_u": float(best["u"]),
        "best_v": float(best["v"]),
        "best_kappa": float(best["kappa"]),
        "canonical_max_abs_det_ratio_residual": float(canonical["max_abs_det_ratio_residual"]),
        "canonical_max_abs_schur_residual": float(canonical["max_abs_schur_residual"]),
        "canonical_max_abs_weight_residual": float(canonical["max_abs_weight_residual"]),
        "canonical_max_abs_xi_residual": float(canonical["max_abs_xi_residual"]),
        "canonical_corr_xi": float(canonical["corr(xi_gen, xi_target)"]),
        "runner_up_u": float(runner_up["u"]),
        "runner_up_v": float(runner_up["v"]),
        "runner_up_kappa": float(runner_up["kappa"]),
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
        "canonical_excess_part_p50": q(excess_part, 0.5),
        "canonical_excess_tree_p50": q(excess_tree, 0.5),
        "canonical_xi_target_p50": q(xi_target, 0.5),
    }

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_parent_mix_geomean_scan_{args.tag}.csv"
    out_map = OUTDIR / f"hll_uv_action_loop_eymh_parent_mix_geomean_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_mix_geomean_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_mix_geomean_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_mix_geomean_audit_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    df[
        [
            "D",
            "eta",
            "parent_mix_excess_part",
            "parent_mix_excess_tree",
            "parent_mix_geomean",
            "parent_mix_xi_target",
            "parent_mix_norm_ratio",
            "eymh_parented_prefactor_diag",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "u_grid": u_grid,
                "v_grid": v_grid,
                "kappa_grid": kappa_grid,
                "alpha_grid": alpha_grid,
                "beta_grid": beta_grid,
                "lambda_grid": lambda_grid,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    heat = scan.pivot_table(index="u", columns="v", values="max_abs_det_ratio_residual", aggfunc="min")
    im = axes[0].imshow(
        heat.values,
        origin="lower",
        aspect="auto",
        extent=[heat.columns.min(), heat.columns.max(), heat.index.min(), heat.index.max()],
    )
    axes[0].set_xlabel("v")
    axes[0].set_ylabel("u")
    axes[0].set_title("min max|det residual| over kappa")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    kslice = scan[np.isclose(scan["u"], 0.5) & np.isclose(scan["v"], 0.5)].sort_values("kappa")
    axes[1].plot(kslice["kappa"], kslice["max_abs_weight_residual"], marker="o")
    axes[1].set_xlabel("kappa")
    axes[1].set_ylabel("max |weight residual|")
    axes[1].set_title("Geometric-mean slice")

    axes[2].scatter(df["parent_mix_excess_part"], df["parent_mix_excess_tree"], s=12, alpha=0.7)
    axes[2].set_xlabel("participation excess")
    axes[2].set_ylabel("tree/background excess")
    axes[2].set_title("Canonical excess sectors")

    fig.suptitle("EYMH parent mixing geometric-mean audit", fontsize=13)
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

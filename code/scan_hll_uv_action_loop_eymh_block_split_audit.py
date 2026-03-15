#!/usr/bin/env python3
"""
Projected fluctuation-operator block-split audit for the canonical EYMH kernel.

This audit isolates the structural split that underlies the canonical parented
EYMH normalization block. On the canonical D21xE21 fix map we define:

1. Participation block:
     K_part = ||c||_1 / ||c||_2

2. Background/tree block:
     K_bg   = 1 + gap_cv
     K_full = 1 + gap_cv + c_tree_diag_cv
     G_schur = K_full / K_bg

3. Mixed block scale:
     C_mix = sqrt((K_part - 1) (G_schur - 1))
     xi_cross = C_mix^2 / (K_part G_schur)

For the deformed family

    K_sel = [[K_part, lambda C_mix],
             [lambda C_mix, G_schur]],

the determinant splits exactly as

    det K_sel = K_part * G_schur * (1 - lambda^2 xi_cross),

so the projected response action decomposes into

    S_kernel = logdet(K_part) + log(G_schur) + log(1 - lambda^2 xi_cross).

This is the cleanest block-level mother-action statement available in the
current pipeline: the canonical kernel is block-diagonal in the participation
and shell-background/tree sectors, while explicit mixing appears only through
an even determinant penalty.
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
        description="Audit the participation/tree/mixed block split of the canonical projected EYMH fluctuation kernel."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    ap.add_argument("--lambda-grid", default="-0.6,-0.4,-0.2,-0.1,0.0,0.1,0.2,0.4,0.6")
    args = ap.parse_args()

    lambdas = parse_grid(args.lambda_grid)
    df = pd.read_csv(args.parented_map)

    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    k_part = coeff_l1 / coeff_l2
    s_part = np.log(k_part)

    gap_cv = np.maximum(np.asarray(df["gap_cv"], dtype=float), 0.0)
    c_tree_diag_cv = np.maximum(np.asarray(df["c_tree_diag_cv"], dtype=float), 0.0)
    k_bg = 1.0 + gap_cv
    k_full = k_bg + c_tree_diag_cv
    g_schur = k_full / np.maximum(k_bg, 1e-30)
    s_schur = np.log(g_schur)

    c_mix = np.sqrt(np.maximum((k_part - 1.0) * (g_schur - 1.0), 0.0))
    xi_cross = np.maximum(c_mix**2 / np.maximum(k_part * g_schur, 1e-30), 0.0)

    response_weight = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )
    response_action = -2.0 * np.log(np.maximum(response_weight, 1e-30))
    split_action = s_part + s_schur
    split_residual = split_action - response_action

    df["block_split_k_part"] = k_part
    df["block_split_k_bg"] = k_bg
    df["block_split_k_full"] = k_full
    df["block_split_g_schur"] = g_schur
    df["block_split_c_mix"] = c_mix
    df["block_split_xi_cross"] = xi_cross
    df["block_split_s_part"] = s_part
    df["block_split_s_schur"] = s_schur
    df["block_split_response_action"] = response_action
    df["block_split_split_action"] = split_action
    df["block_split_action_residual"] = split_residual
    df["block_split_coeff_share"] = s_part / np.maximum(split_action, 1e-30)
    df["block_split_tree_share"] = s_schur / np.maximum(split_action, 1e-30)

    scan_rows: list[dict[str, float]] = []
    for lam in lambdas:
        det_direct = np.maximum(k_part * g_schur - (lam * c_mix) ** 2, 1e-30)
        det_split = np.maximum(k_part * g_schur * (1.0 - (lam**2) * xi_cross), 1e-30)
        s_direct = np.log(det_direct)
        s_split = s_part + s_schur + np.log(np.maximum(1.0 - (lam**2) * xi_cross, 1e-30))
        scan_rows.append(
            {
                "lambda": lam,
                "max_abs_det_split_residual": float(np.max(np.abs(det_direct - det_split))),
                "max_abs_action_split_residual": float(np.max(np.abs(s_direct - s_split))),
                "mean_abs_det_split_residual": float(np.mean(np.abs(det_direct - det_split))),
                "mean_abs_action_split_residual": float(np.mean(np.abs(s_direct - s_split))),
                "mean_mixing_penalty": float(np.mean(np.log(np.maximum(1.0 - (lam**2) * xi_cross, 1e-30)))),
                "p90_abs_mixing_penalty": q(np.abs(np.log(np.maximum(1.0 - (lam**2) * xi_cross, 1e-30))), 0.9),
            }
        )
    scan = pd.DataFrame(scan_rows)
    canonical = scan[np.isclose(scan["lambda"], 0.0)].iloc[0]

    summary = {
        "n_points": int(len(df)),
        "n_lambda_scan_points": int(len(scan)),
        "max_abs_canonical_action_split_residual": float(np.max(np.abs(split_residual))),
        "p95_abs_canonical_action_split_residual": q(np.abs(split_residual), 0.95),
        "corr(split_action, response_action)": corr(split_action, response_action),
        "canonical_coeff_share_p50": q(df["block_split_coeff_share"], 0.5),
        "canonical_tree_share_p50": q(df["block_split_tree_share"], 0.5),
        "canonical_xi_cross_p50": q(xi_cross, 0.5),
        "canonical_xi_cross_p90": q(xi_cross, 0.9),
        "lambda0_max_abs_det_split_residual": float(canonical["max_abs_det_split_residual"]),
        "lambda0_max_abs_action_split_residual": float(canonical["max_abs_action_split_residual"]),
        "scan_max_abs_det_split_residual": float(scan["max_abs_det_split_residual"].max()),
        "scan_max_abs_action_split_residual": float(scan["max_abs_action_split_residual"].max()),
        "scan_p95_abs_action_split_residual": q(scan["max_abs_action_split_residual"], 0.95),
    }

    out_scan = OUTDIR / f"hll_uv_action_loop_eymh_block_split_scan_{args.tag}.csv"
    out_map = OUTDIR / f"hll_uv_action_loop_eymh_block_split_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_block_split_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_block_split_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_block_split_audit_run_meta_{args.tag}.json"

    scan.to_csv(out_scan, index=False)
    df[
        [
            "D",
            "eta",
            "block_split_k_part",
            "block_split_k_bg",
            "block_split_k_full",
            "block_split_g_schur",
            "block_split_c_mix",
            "block_split_xi_cross",
            "block_split_s_part",
            "block_split_s_schur",
            "block_split_response_action",
            "block_split_split_action",
            "block_split_action_residual",
            "block_split_coeff_share",
            "block_split_tree_share",
            "eymh_parented_prefactor_diag",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "lambda_grid": lambdas,
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].scatter(df["block_split_s_part"], df["block_split_s_schur"], s=12, alpha=0.7)
    axes[0].set_xlabel("participation block action")
    axes[0].set_ylabel("Schur/tree block action")
    axes[0].set_title("Projected block actions")

    axes[1].scatter(df["block_split_c_mix"], df["block_split_xi_cross"], s=12, alpha=0.7)
    axes[1].set_xlabel("mixed block scale")
    axes[1].set_ylabel("normalized cross ratio")
    axes[1].set_title("Mixed block enters via determinant")

    axes[2].plot(scan["lambda"], scan["mean_mixing_penalty"], marker="o")
    axes[2].set_xlabel("lambda")
    axes[2].set_ylabel("mean log(1-lambda^2 xi_cross)")
    axes[2].set_title("Even mixing penalty")

    fig.suptitle("EYMH projected fluctuation-operator block split audit", fontsize=13)
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

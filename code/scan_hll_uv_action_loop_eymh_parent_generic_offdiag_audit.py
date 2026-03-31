#!/usr/bin/env python3
"""
Generic off-diagonal consistency audit for the projected EYMH parent kernel.

This audit is designed to avoid the circularity that appears if one starts from
the already-geometric-mean parent block determinant.  Instead, it treats the
off-diagonal sector only through its square,

    Q(E_part, E_tree) := K12^2,

and infers that quantity directly from:

1. the diagonal sectors fixed by the projected split blocks, and
2. the exact response action read from the parented direct map.

On the localized canonical map we therefore define

    det_ratio_ref = exp(S_resp),
    Q_inferred    = K11 * K22 - K_bg * det_ratio_ref,

with

    E_part = K11 - 1,
    E_tree = K22 - K_bg.

If the mixed entry is genuinely fixed by determinant matching, then on the
local positive cone we should recover

    Q_inferred = E_part * E_tree,

without having inserted the geometric mean into the audit itself.
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Infer the generic EYMH off-diagonal square from the response determinant law without assuming the geometric mean."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument("--tag", default="D21E21_fix")
    args = ap.parse_args()

    df = pd.read_csv(args.parented_map)

    coeff_l1 = np.maximum(np.asarray(df["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(df["coeff_l2"], dtype=float), 1e-30)
    k11 = coeff_l1 / coeff_l2

    k_bg = 1.0 + np.maximum(np.asarray(df["gap_cv"], dtype=float), 0.0)
    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    g_schur = 1.0 + chi_tree
    k22 = k_bg * g_schur

    a_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )
    s_resp = -2.0 * np.log(np.maximum(a_ref, 1e-30))
    det_ratio_ref = np.exp(s_resp)

    e_part = np.maximum(k11 - 1.0, 0.0)
    e_tree = np.maximum(k22 - k_bg, 0.0)
    q_geomean = e_part * e_tree
    q_inferred = k11 * k22 - k_bg * det_ratio_ref
    q_residual = q_inferred - q_geomean

    positive_cone = (e_part >= -1e-12) & (e_tree >= -1e-12)
    positive_q = q_inferred >= -1e-12
    rho_inferred = q_inferred / np.maximum(q_geomean, 1e-30)
    rho_residual = rho_inferred - 1.0

    schur_from_q = (k22 - np.maximum(q_inferred, 0.0) / np.maximum(k11, 1e-30)) / np.maximum(k_bg, 1e-30)
    schur_from_det = det_ratio_ref / np.maximum(k11, 1e-30)

    df["generic_offdiag_k11"] = k11
    df["generic_offdiag_k_bg"] = k_bg
    df["generic_offdiag_k22"] = k22
    df["generic_offdiag_e_part"] = e_part
    df["generic_offdiag_e_tree"] = e_tree
    df["generic_offdiag_det_ratio_ref"] = det_ratio_ref
    df["generic_offdiag_q_inferred"] = q_inferred
    df["generic_offdiag_q_geomean"] = q_geomean
    df["generic_offdiag_q_residual"] = q_residual
    df["generic_offdiag_rho_inferred"] = rho_inferred
    df["generic_offdiag_rho_residual"] = rho_residual
    df["generic_offdiag_schur_from_q"] = schur_from_q
    df["generic_offdiag_schur_from_det"] = schur_from_det
    df["generic_offdiag_positive_cone"] = positive_cone.astype(int)
    df["generic_offdiag_positive_q"] = positive_q.astype(int)

    summary = {
        "n_points": int(len(df)),
        "positive_cone_fraction": float(np.mean(positive_cone)),
        "positive_q_fraction": float(np.mean(positive_q)),
        "max_abs_q_residual": float(np.max(np.abs(q_residual))),
        "p95_abs_q_residual": q(np.abs(q_residual), 0.95),
        "max_abs_rho_residual": float(np.max(np.abs(rho_residual))),
        "p95_abs_rho_residual": q(np.abs(rho_residual), 0.95),
        "median_rho_inferred": q(rho_inferred, 0.5),
        "p10_rho_inferred": q(rho_inferred, 0.10),
        "p90_rho_inferred": q(rho_inferred, 0.90),
        "corr(q_inferred, q_geomean)": corr(q_inferred, q_geomean),
        "max_abs_schur_identity_residual": float(np.max(np.abs(schur_from_q - schur_from_det))),
        "p95_abs_schur_identity_residual": q(np.abs(schur_from_q - schur_from_det), 0.95),
        "max_abs_det_identity_residual": float(np.max(np.abs((k11 * k22 - q_inferred) / np.maximum(k_bg, 1e-30) - det_ratio_ref))),
        "max_negative_q_magnitude": float(np.max(np.maximum(-q_inferred, 0.0))),
    }

    d_summary = (
        df.groupby("D", dropna=False)
        .agg(
            generic_offdiag_q_residual_p95=("generic_offdiag_q_residual", lambda s: float(pd.Series(np.abs(s)).quantile(0.95))),
            generic_offdiag_rho_inferred_p50=("generic_offdiag_rho_inferred", "median"),
            generic_offdiag_rho_inferred_p95=("generic_offdiag_rho_inferred", lambda s: float(pd.Series(s).quantile(0.95))),
            generic_offdiag_positive_q_fraction=("generic_offdiag_positive_q", "mean"),
        )
        .reset_index()
    )

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_parent_generic_offdiag_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_generic_offdiag_audit_summary_{args.tag}.csv"
    out_dsummary = OUTDIR / f"hll_uv_action_loop_eymh_parent_generic_offdiag_audit_dsummary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_generic_offdiag_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_generic_offdiag_audit_run_meta_{args.tag}.json"

    df[
        [
            "D",
            "eta",
            "generic_offdiag_k11",
            "generic_offdiag_k_bg",
            "generic_offdiag_k22",
            "generic_offdiag_e_part",
            "generic_offdiag_e_tree",
            "generic_offdiag_det_ratio_ref",
            "generic_offdiag_q_inferred",
            "generic_offdiag_q_geomean",
            "generic_offdiag_q_residual",
            "generic_offdiag_rho_inferred",
            "generic_offdiag_rho_residual",
            "generic_offdiag_schur_from_q",
            "generic_offdiag_schur_from_det",
            "generic_offdiag_positive_cone",
            "generic_offdiag_positive_q",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    d_summary.to_csv(out_dsummary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "design_note": "Infer Q=K12^2 from direct response determinant law without assuming geometric-mean mixing.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].scatter(q_geomean, q_inferred, s=14, alpha=0.7)
    max_q = float(max(np.max(q_geomean), np.max(q_inferred), 1e-30))
    axes[0].plot([0.0, max_q], [0.0, max_q], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel(r"$E_{\rm part}E_{\rm tree}$")
    axes[0].set_ylabel(r"$Q_{\rm inferred}$")
    axes[0].set_title("Generic off-diagonal square")

    axes[1].hist(rho_residual, bins=30)
    axes[1].set_xlabel(r"$\rho_{\rm inferred}-1$")
    axes[1].set_ylabel("count")
    axes[1].set_title("Normalized square residual")

    axes[2].plot(d_summary["D"], d_summary["generic_offdiag_rho_inferred_p50"], marker="o", label="median rho")
    axes[2].fill_between(
        d_summary["D"],
        d_summary["generic_offdiag_rho_inferred_p50"],
        d_summary["generic_offdiag_rho_inferred_p95"],
        alpha=0.2,
        label="up to p95 rho",
    )
    axes[2].axhline(1.0, linestyle="--", color="black", linewidth=1)
    axes[2].set_xlabel("D")
    axes[2].set_ylabel(r"$\rho_{\rm inferred}$")
    axes[2].set_title("D-profile of inferred ratio")
    axes[2].legend(fontsize=8)

    fig.suptitle("EYMH generic off-diagonal determinant consistency audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_map, out_summary, out_dsummary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

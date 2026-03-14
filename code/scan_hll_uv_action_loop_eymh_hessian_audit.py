#!/usr/bin/env python3
"""
Projected-Hessian / Schur-complement audit for the parented EYMH UV comparator.

This audit rewrites the two surviving parent-action source factors in response
language:

1. coeff_participation_access_parented
   -> projected two-mode Hessian participation response built from
      (Tr H)^2 / Tr(H^2).
2. tree_diag_compressibility_parented
   -> Schur-complement / compressibility response of the shell-background-
      normalized diagonal block.

The goal is not to introduce a new baseline, but to check whether the current
canonical parented comparator can already be read as a projected one-loop
response quantity.
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


def corr(a: pd.Series, b: pd.Series) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def q(v: pd.Series, p: float) -> float:
    return float(v.quantile(p))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit projected-Hessian / Schur-complement rewrites of the parented EYMH UV comparator."
    )
    ap.add_argument(
        "--parented-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv",
    )
    ap.add_argument(
        "--constant-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_constant_refresh_D21E21.csv",
    )
    ap.add_argument("--tag", default="D21E21")
    args = ap.parse_args()

    parented = pd.read_csv(args.parented_map)
    const = pd.read_csv(args.constant_map)[["D", "eta", "mu_mumu_uv_rge"]].rename(
        columns={"mu_mumu_uv_rge": "mu_mumu_uv_rge_const"}
    )
    df = parented.merge(const, on=["D", "eta"], how="left")

    p1 = np.asarray(df["loop_trace_p1"], dtype=float)
    p2 = np.asarray(df["loop_trace_p2"], dtype=float)
    tr_h = p1 + p2
    fro2_h = p1**2 + p2**2
    det_h = p1 * p2
    tr_h_safe = np.maximum(tr_h, 1e-30)
    fro2_safe = np.maximum(fro2_h, 1e-30)
    neff_h = (tr_h_safe**2) / fro2_safe
    part_susc = np.maximum(neff_h - 1.0, 0.0)
    part_response = np.power(1.0 + part_susc, -0.25)

    gap_cv = np.asarray(df["gap_cv"], dtype=float)
    c_tree_diag_cv = np.asarray(df["c_tree_diag_cv"], dtype=float)
    base_diag = 1.0 + gap_cv
    load_diag = c_tree_diag_cv
    schur_ratio = base_diag / np.maximum(base_diag + load_diag, 1e-30)
    schur_response = np.sqrt(np.maximum(schur_ratio, 0.0))
    schur_pressure = load_diag / np.maximum(base_diag + load_diag, 1e-30)

    coeff_access_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float)
    tree_comp_ref = np.asarray(df["tree_diag_compressibility_parented"], dtype=float)
    prefactor_ref = np.asarray(df["eymh_parented_prefactor_diag"], dtype=float)
    mu_diff_vs_const = np.abs(
        np.asarray(df["mu_mumu_uv_rge"], dtype=float) - np.asarray(df["mu_mumu_uv_rge_const"], dtype=float)
    )

    coeff_resid = part_response - coeff_access_ref
    tree_resid = schur_response - tree_comp_ref

    df["h_part_trace"] = tr_h
    df["h_part_fro2"] = fro2_h
    df["h_part_det"] = det_h
    df["h_part_neff"] = neff_h
    df["h_part_susceptibility"] = part_susc
    df["h_part_response"] = part_response
    df["coeff_participation_parented_recon"] = part_response
    df["coeff_participation_parented_residual"] = coeff_resid

    df["h_tree_base"] = base_diag
    df["h_tree_load"] = load_diag
    df["h_tree_schur_ratio"] = schur_ratio
    df["h_tree_pressure_fraction"] = schur_pressure
    df["tree_diag_compressibility_parented_recon"] = schur_response
    df["tree_diag_compressibility_parented_residual"] = tree_resid
    df["abs_mu_diff_vs_const"] = mu_diff_vs_const

    summary = {
        "n_points": int(len(df)),
        "max_abs_coeff_participation_parented_recon_residual": float(np.max(np.abs(coeff_resid))),
        "max_abs_tree_diag_parented_recon_residual": float(np.max(np.abs(tree_resid))),
        "h_part_neff_p50": q(df["h_part_neff"], 0.5),
        "h_part_neff_p90": q(df["h_part_neff"], 0.9),
        "h_part_susceptibility_p50": q(df["h_part_susceptibility"], 0.5),
        "h_part_susceptibility_p90": q(df["h_part_susceptibility"], 0.9),
        "coeff_participation_access_parented_p50": q(df["coeff_participation_access_parented"], 0.5),
        "coeff_participation_access_parented_p90": q(df["coeff_participation_access_parented"], 0.9),
        "h_tree_pressure_fraction_p50": q(df["h_tree_pressure_fraction"], 0.5),
        "h_tree_pressure_fraction_p90": q(df["h_tree_pressure_fraction"], 0.9),
        "tree_diag_compressibility_parented_p50": q(df["tree_diag_compressibility_parented"], 0.5),
        "tree_diag_compressibility_parented_p90": q(df["tree_diag_compressibility_parented"], 0.9),
        "corr(h_part_neff, coeff_participation_access_parented)": corr(
            df["h_part_neff"], df["coeff_participation_access_parented"]
        ),
        "corr(h_part_susceptibility, coeff_participation_access_parented)": corr(
            df["h_part_susceptibility"], df["coeff_participation_access_parented"]
        ),
        "corr(h_part_response, coeff_participation_access_parented)": corr(
            df["h_part_response"], df["coeff_participation_access_parented"]
        ),
        "corr(h_tree_pressure_fraction, tree_diag_compressibility_parented)": corr(
            df["h_tree_pressure_fraction"], df["tree_diag_compressibility_parented"]
        ),
        "corr(h_tree_pressure_fraction, eymh_parented_prefactor_diag)": corr(
            df["h_tree_pressure_fraction"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(tree_diag_compressibility_parented, eymh_parented_prefactor_diag)": corr(
            df["tree_diag_compressibility_parented"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(coeff_participation_access_parented, eymh_parented_prefactor_diag)": corr(
            df["coeff_participation_access_parented"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(h_part_response, abs_mu_diff_vs_const)": corr(df["h_part_response"], df["abs_mu_diff_vs_const"]),
        "corr(h_tree_pressure_fraction, abs_mu_diff_vs_const)": corr(
            df["h_tree_pressure_fraction"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_hessian_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_hessian_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_hessian_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_hessian_audit_run_meta_{args.tag}.json"

    df[
        [
            "D",
            "eta",
            "loop_trace_p1",
            "loop_trace_p2",
            "h_part_trace",
            "h_part_fro2",
            "h_part_det",
            "h_part_neff",
            "h_part_susceptibility",
            "coeff_participation_access_parented",
            "coeff_participation_parented_recon",
            "coeff_participation_parented_residual",
            "gap_cv",
            "c_tree_diag_cv",
            "h_tree_base",
            "h_tree_load",
            "h_tree_schur_ratio",
            "h_tree_pressure_fraction",
            "tree_diag_compressibility_parented",
            "tree_diag_compressibility_parented_recon",
            "tree_diag_compressibility_parented_residual",
            "eymh_parented_prefactor_diag",
            "mu_mumu_uv_rge",
            "mu_mumu_uv_rge_const",
            "abs_mu_diff_vs_const",
        ]
    ].to_csv(out_map, index=False)
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "constant_map": str(args.constant_map),
                "n_points": len(df),
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    axes[0].scatter(df["h_part_neff"], df["coeff_participation_access_parented"], s=12, alpha=0.7)
    axes[0].set_xlabel("projected Hessian N_eff")
    axes[0].set_ylabel("coeff_participation_access_parented")
    axes[0].set_title("Participation response")

    axes[1].scatter(df["h_tree_pressure_fraction"], df["tree_diag_compressibility_parented"], s=12, alpha=0.7)
    axes[1].set_xlabel("Schur pressure fraction")
    axes[1].set_ylabel("tree_diag_compressibility_parented")
    axes[1].set_title("Compressibility response")

    axes[2].scatter(df["h_tree_pressure_fraction"], df["eymh_parented_prefactor_diag"], s=12, alpha=0.7)
    axes[2].set_xlabel("Schur pressure fraction")
    axes[2].set_ylabel("eymh_parented_prefactor_diag")
    axes[2].set_title("Pressure vs parented prefactor")
    fig.suptitle("EYMH projected-Hessian / Schur audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_map, out_summary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

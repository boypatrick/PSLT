#!/usr/bin/env python3
"""
Exact two-mode participation audit for the parented EYMH UV comparator.

This audit upgrades the projected-Hessian participation story one step further:
for a two-mode loop-trace block with probabilities (p1, p2), the effective
participation number

    N_eff = 1 / (p1^2 + p2^2)

determines the imbalance

    d = |p1 - p2| = sqrt(2 / N_eff - 1)

and therefore the canonical participation/coherence access exactly,

    coeff_participation_access_parented = sqrt((1 - d) / (1 + d)).

So unlike the earlier Hessian-response proxy, the two-mode projected response
can be closed exactly once the parented loop-trace block is reduced to N_eff.
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
        description="Audit exact two-mode participation closure for the parented EYMH UV comparator."
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
    ap.add_argument("--tag", default="D21E21_fix")
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
    tr_h_safe = np.maximum(tr_h, 1e-30)
    fro2_safe = np.maximum(fro2_h, 1e-30)

    neff_h = (tr_h_safe**2) / fro2_safe
    imbalance = np.sqrt(np.maximum(2.0 / np.maximum(neff_h, 1e-30) - 1.0, 0.0))
    ratio_low_high = (1.0 - imbalance) / np.maximum(1.0 + imbalance, 1e-30)
    coeff_access_exact = np.sqrt(np.maximum(ratio_low_high, 0.0))

    coeff_access_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float)
    coeff_resid = coeff_access_exact - coeff_access_ref

    mu_diff_vs_const = np.abs(
        np.asarray(df["mu_mumu_uv_rge"], dtype=float) - np.asarray(df["mu_mumu_uv_rge_const"], dtype=float)
    )

    df["h_part_neff"] = neff_h
    df["h_part_imbalance"] = imbalance
    df["h_part_ratio_low_high"] = ratio_low_high
    df["coeff_participation_access_parented_exact"] = coeff_access_exact
    df["coeff_participation_access_parented_exact_residual"] = coeff_resid
    df["abs_mu_diff_vs_const"] = mu_diff_vs_const

    summary = {
        "n_points": int(len(df)),
        "max_abs_coeff_participation_exact_residual": float(np.max(np.abs(coeff_resid))),
        "h_part_neff_p50": q(df["h_part_neff"], 0.5),
        "h_part_neff_p90": q(df["h_part_neff"], 0.9),
        "h_part_imbalance_p50": q(df["h_part_imbalance"], 0.5),
        "h_part_imbalance_p90": q(df["h_part_imbalance"], 0.9),
        "h_part_ratio_low_high_p50": q(df["h_part_ratio_low_high"], 0.5),
        "h_part_ratio_low_high_p90": q(df["h_part_ratio_low_high"], 0.9),
        "coeff_participation_access_parented_p50": q(df["coeff_participation_access_parented"], 0.5),
        "coeff_participation_access_parented_p90": q(df["coeff_participation_access_parented"], 0.9),
        "corr(h_part_neff, coeff_participation_access_parented)": corr(
            df["h_part_neff"], df["coeff_participation_access_parented"]
        ),
        "corr(h_part_imbalance, coeff_participation_access_parented)": corr(
            df["h_part_imbalance"], df["coeff_participation_access_parented"]
        ),
        "corr(h_part_ratio_low_high, coeff_participation_access_parented)": corr(
            df["h_part_ratio_low_high"], df["coeff_participation_access_parented"]
        ),
        "corr(coeff_participation_access_parented, eymh_parented_prefactor_diag)": corr(
            df["coeff_participation_access_parented"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(h_part_imbalance, eymh_parented_prefactor_diag)": corr(
            df["h_part_imbalance"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(h_part_ratio_low_high, eymh_parented_prefactor_diag)": corr(
            df["h_part_ratio_low_high"], df["eymh_parented_prefactor_diag"]
        ),
        "corr(h_part_imbalance, abs_mu_diff_vs_const)": corr(df["h_part_imbalance"], df["abs_mu_diff_vs_const"]),
        "corr(h_part_ratio_low_high, abs_mu_diff_vs_const)": corr(
            df["h_part_ratio_low_high"], df["abs_mu_diff_vs_const"]
        ),
    }

    out_map = OUTDIR / f"hll_uv_action_loop_eymh_participation_exact_audit_map_{args.tag}.csv"
    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_participation_exact_audit_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_participation_exact_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_participation_exact_audit_run_meta_{args.tag}.json"

    df[
        [
            "D",
            "eta",
            "loop_trace_p1",
            "loop_trace_p2",
            "h_part_neff",
            "h_part_imbalance",
            "h_part_ratio_low_high",
            "coeff_participation_access_parented",
            "coeff_participation_access_parented_exact",
            "coeff_participation_access_parented_exact_residual",
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
    axes[0].set_xlabel("projected N_eff")
    axes[0].set_ylabel("coeff_participation_access_parented")
    axes[0].set_title("Participation vs N_eff")

    axes[1].scatter(df["h_part_imbalance"], df["coeff_participation_access_parented"], s=12, alpha=0.7)
    axes[1].set_xlabel("projected imbalance")
    axes[1].set_ylabel("coeff_participation_access_parented")
    axes[1].set_title("Participation vs imbalance")

    axes[2].scatter(df["h_part_ratio_low_high"], df["eymh_parented_prefactor_diag"], s=12, alpha=0.7)
    axes[2].set_xlabel("low/high ratio from N_eff")
    axes[2].set_ylabel("eymh_parented_prefactor_diag")
    axes[2].set_title("Participation ratio vs prefactor")
    fig.suptitle("EYMH exact two-mode participation audit", fontsize=13)
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

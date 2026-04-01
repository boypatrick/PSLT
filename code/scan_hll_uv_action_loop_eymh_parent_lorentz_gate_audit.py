#!/usr/bin/env python3
"""
Lorentz-gated curvature audit for EYMH Lemma 2 mixed-entry witnesses.

This audit tests whether a Lorentz-like D reparameterization, D_eff = D / gamma,
helps global D convergence when we connect the empirically stable low-D witness
(hk_abs_offdiag / diag) to the empirically dominant high-D witness
(action_abs_offdiag / diag) through a soft logistic gate.
"""

from __future__ import annotations

import argparse
import json
import math
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


def gamma(beta: float) -> float:
    return 1.0 / math.sqrt(max(1.0 - beta * beta, 1e-12))


def sigmoid_gate(x: np.ndarray, center: float, width: float) -> np.ndarray:
    w = max(float(width), 1e-9)
    z = np.clip((np.asarray(x, dtype=float) - float(center)) / w, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_nonnegative_weighted_ls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(w[:, None] * X, w * y, rcond=None)[0]
    return np.maximum(beta, 0.0)


def score_family(xi_proxy: np.ndarray, xi_target: np.ndarray, a_ref: np.ndarray) -> dict[str, float]:
    quartic_proxy = 0.25 * (a_ref**2) * (xi_proxy**2)
    quartic_target = 0.25 * (a_ref**2) * (xi_target**2)
    xi_resid = xi_proxy - xi_target
    quartic_resid = quartic_proxy - quartic_target
    return {
        "corr_xi": corr(xi_proxy, xi_target),
        "corr_quartic": corr(quartic_proxy, quartic_target),
        "p95_abs_xi_resid": q(np.abs(xi_resid), 0.95),
        "p95_abs_quartic_resid": q(np.abs(quartic_resid), 0.95),
        "max_abs_xi_resid": float(np.max(np.abs(xi_resid))),
        "max_abs_quartic_resid": float(np.max(np.abs(quartic_resid))),
        "abs_c4_gap": abs(float(np.mean(quartic_proxy)) - float(np.mean(quartic_target))),
        "rank_score": q(np.abs(quartic_resid), 0.95)
        + 0.25 * q(np.abs(xi_resid), 0.95)
        + abs(float(np.mean(quartic_proxy)) - float(np.mean(quartic_target))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Lorentz-gated low/high-D witness transitions for EYMH Lemma 2.")
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
    k_part = coeff_l1 / coeff_l2
    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    g_schur = 1.0 + chi_tree
    a_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )
    xi_target = np.maximum((k_part - 1.0) * (g_schur - 1.0), 0.0) / np.maximum(k_part * g_schur, 1e-30)

    D = np.asarray(df["D"], dtype=float)
    diag = np.maximum(a_ref, 1e-30)
    p_low = np.maximum(np.asarray(df["hk_abs_offdiag"], dtype=float), 0.0) / diag
    p_high = np.maximum(np.asarray(df["action_abs_offdiag"], dtype=float), 0.0) / diag

    betas = [0.0, 0.2, 0.4, 0.6]
    centers = [float(x) for x in sorted(df["D"].unique())]
    widths = [0.4, 0.8, 1.2, 1.6, 2.4, 3.2]
    d_values = [float(x) for x in sorted(df["D"].unique())]
    d_masks = {dval: np.asarray(df["D"] == dval) for dval in d_values}

    rows: list[dict[str, float | str]] = []
    drows: list[dict[str, float | str]] = []
    beta_best_rows: list[dict[str, float | str]] = []

    for beta in betas:
        g = gamma(beta)
        d_eff = D / g
        for center in centers:
            for width in widths:
                gate = sigmoid_gate(d_eff, center, width)
                X = np.column_stack([(1.0 - gate) * p_low, gate * p_high])
                beta_fit = fit_nonnegative_weighted_ls(X, xi_target, a_ref)
                xi_proxy = np.maximum(X @ beta_fit, 0.0)
                metrics = score_family(xi_proxy, xi_target, a_ref)
                quartic_proxy = 0.25 * (a_ref**2) * (xi_proxy**2)
                quartic_target = 0.25 * (a_ref**2) * (xi_target**2)
                d_slice_quartic = [
                    q(np.abs(quartic_proxy[mask] - quartic_target[mask]), 0.95) for mask in d_masks.values()
                ]
                d_slice_xi = [q(np.abs(xi_proxy[mask] - xi_target[mask]), 0.95) for mask in d_masks.values()]
                d_star = float(center * g)
                row = {
                    "family_name": "lorentz_gate_hkabs_to_actionabs",
                    "beta": float(beta),
                    "gamma": float(g),
                    "center_D_eff": float(center),
                    "width_D_eff": float(width),
                    "implied_center_D": d_star,
                    "coeff_low": float(beta_fit[0]),
                    "coeff_high": float(beta_fit[1]),
                    "mean_D_slice_p95_abs_quartic_resid": float(np.mean(d_slice_quartic)),
                    "max_D_slice_p95_abs_quartic_resid": float(np.max(d_slice_quartic)),
                    "mean_D_slice_p95_abs_xi_resid": float(np.mean(d_slice_xi)),
                    "max_D_slice_p95_abs_xi_resid": float(np.max(d_slice_xi)),
                    **metrics,
                }
                rows.append(row)

        beta_df = pd.DataFrame([r for r in rows if float(r["beta"]) == float(beta)]).sort_values(
            ["rank_score", "p95_abs_quartic_resid", "p95_abs_xi_resid"],
            ascending=[True, True, True],
        )
        best = beta_df.iloc[0].to_dict()
        beta_best_rows.append(best)

        gate = sigmoid_gate(d_eff, float(best["center_D_eff"]), float(best["width_D_eff"]))
        X = np.column_stack([(1.0 - gate) * p_low, gate * p_high])
        xi_proxy = np.maximum(
            X
            @ np.asarray([
                float(best["coeff_low"]),
                float(best["coeff_high"]),
            ]),
            0.0,
        )
        quartic_proxy = 0.25 * (a_ref**2) * (xi_proxy**2)
        quartic_target = 0.25 * (a_ref**2) * (xi_target**2)
        for dval in sorted(df["D"].unique()):
            mask = np.asarray(df["D"] == dval)
            drows.append(
                {
                    "beta": float(beta),
                    "gamma": float(g),
                    "best_center_D_eff": float(best["center_D_eff"]),
                    "best_width_D_eff": float(best["width_D_eff"]),
                    "best_implied_center_D": float(best["implied_center_D"]),
                    "D": float(dval),
                    "slice_corr_quartic": corr(quartic_proxy[mask], quartic_target[mask]),
                    "slice_p95_abs_xi_resid": q(np.abs(xi_proxy[mask] - xi_target[mask]), 0.95),
                    "slice_p95_abs_quartic_resid": q(np.abs(quartic_proxy[mask] - quartic_target[mask]), 0.95),
                    "slice_mean_quartic_gap": abs(float(np.mean(quartic_proxy[mask])) - float(np.mean(quartic_target[mask]))),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["rank_score", "p95_abs_quartic_resid", "p95_abs_xi_resid"],
        ascending=[True, True, True],
    )
    beta_best = pd.DataFrame(beta_best_rows).sort_values("beta")
    dsummary = pd.DataFrame(drows).sort_values(["beta", "D"])

    winner = summary.iloc[0].to_dict()
    beta0 = beta_best[beta_best["beta"] == 0.0].iloc[0].to_dict()
    best_nonzero = beta_best[beta_best["beta"] > 0.0].sort_values("rank_score").iloc[0].to_dict()
    best_balanced = beta_best.sort_values(
        [
            "mean_D_slice_p95_abs_quartic_resid",
            "max_D_slice_p95_abs_quartic_resid",
            "mean_D_slice_p95_abs_xi_resid",
            "rank_score",
        ],
        ascending=[True, True, True, True],
    ).iloc[0].to_dict()

    gate_decision = {
        "family_name": "lorentz_gate_hkabs_to_actionabs",
        "best_overall_beta": float(winner["beta"]),
        "best_overall_rank_score": float(winner["rank_score"]),
        "best_balanced_beta": float(best_balanced["beta"]),
        "best_balanced_mean_D_slice_p95_abs_quartic_resid": float(best_balanced["mean_D_slice_p95_abs_quartic_resid"]),
        "best_balanced_max_D_slice_p95_abs_quartic_resid": float(best_balanced["max_D_slice_p95_abs_quartic_resid"]),
        "beta0_rank_score": float(beta0["rank_score"]),
        "beta0_mean_D_slice_p95_abs_quartic_resid": float(beta0["mean_D_slice_p95_abs_quartic_resid"]),
        "beta0_max_D_slice_p95_abs_quartic_resid": float(beta0["max_D_slice_p95_abs_quartic_resid"]),
        "best_nonzero_beta": float(best_nonzero["beta"]),
        "best_nonzero_rank_score": float(best_nonzero["rank_score"]),
        "best_nonzero_mean_D_slice_p95_abs_quartic_resid": float(best_nonzero["mean_D_slice_p95_abs_quartic_resid"]),
        "best_nonzero_max_D_slice_p95_abs_quartic_resid": float(best_nonzero["max_D_slice_p95_abs_quartic_resid"]),
        "delta_rank_nonzero_minus_beta0": float(best_nonzero["rank_score"] - beta0["rank_score"]),
        "delta_quartic_nonzero_minus_beta0": float(best_nonzero["p95_abs_quartic_resid"] - beta0["p95_abs_quartic_resid"]),
        "delta_xi_nonzero_minus_beta0": float(best_nonzero["p95_abs_xi_resid"] - beta0["p95_abs_xi_resid"]),
        "delta_mean_D_slice_quartic_nonzero_minus_beta0": float(
            best_nonzero["mean_D_slice_p95_abs_quartic_resid"] - beta0["mean_D_slice_p95_abs_quartic_resid"]
        ),
        "delta_max_D_slice_quartic_nonzero_minus_beta0": float(
            best_nonzero["max_D_slice_p95_abs_quartic_resid"] - beta0["max_D_slice_p95_abs_quartic_resid"]
        ),
        "beta0_implied_center_D": float(beta0["implied_center_D"]),
        "best_nonzero_implied_center_D": float(best_nonzero["implied_center_D"]),
        "recommendation": (
            "lorentz_gate_reasonable_but_not_better_for_full_D_convergence"
            if float(best_nonzero["rank_score"]) < float(beta0["rank_score"])
            and float(best_nonzero["mean_D_slice_p95_abs_quartic_resid"]) > float(beta0["mean_D_slice_p95_abs_quartic_resid"])
            else (
                "beta_zero_preferred"
                if float(best_nonzero["rank_score"]) >= float(beta0["rank_score"])
                else "lorentz_gate_preferred"
            )
        ),
        "interpretation": (
            "Lorentz-like D_eff improves the pooled full-map score slightly, but it worsens mean and worst D-slice quartic residuals; in this EYMH mixed-entry audit it behaves like a reasonable edge-gating ansatz, not a true full-D convergence improvement."
            if float(best_nonzero["rank_score"]) < float(beta0["rank_score"])
            and float(best_nonzero["mean_D_slice_p95_abs_quartic_resid"]) > float(beta0["mean_D_slice_p95_abs_quartic_resid"])
            else (
                "Lorentz-like D_eff is numerically reasonable as a smooth regime gate, but it does not improve full-map D convergence over beta=0 in the current EYMH mixed-entry audit."
                if float(best_nonzero["rank_score"]) >= float(beta0["rank_score"])
                else "Lorentz-like D_eff improves full-map D convergence in the current EYMH mixed-entry audit."
            )
        ),
    }

    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_lorentz_gate_audit_summary_{args.tag}.csv"
    out_beta_best = OUTDIR / f"hll_uv_action_loop_eymh_parent_lorentz_gate_audit_best_by_beta_{args.tag}.csv"
    out_dsummary = OUTDIR / f"hll_uv_action_loop_eymh_parent_lorentz_gate_audit_dsummary_{args.tag}.csv"
    out_decision = OUTDIR / f"hll_uv_action_loop_eymh_parent_lorentz_gate_audit_decision_{args.tag}.json"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_lorentz_gate_audit_{args.tag}.png"

    summary.to_csv(out_summary, index=False)
    beta_best.to_csv(out_beta_best, index=False)
    dsummary.to_csv(out_dsummary, index=False)
    out_decision.write_text(json.dumps(gate_decision, indent=2, sort_keys=True))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].plot(beta_best["beta"], beta_best["rank_score"], marker="o")
    axes[0].set_title("Best rank score by Lorentz beta")
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("rank score")

    axes[1].plot(beta_best["beta"], beta_best["implied_center_D"], marker="o", label="implied center D")
    axes[1].plot(beta_best["beta"], beta_best["center_D_eff"], marker="s", label="center D_eff")
    axes[1].set_title("Best gate center by beta")
    axes[1].set_xlabel("beta")
    axes[1].legend(fontsize=8)

    for beta in betas:
        sub = dsummary[dsummary["beta"] == beta]
        axes[2].plot(sub["D"], sub["slice_p95_abs_quartic_resid"], marker="o", label=f"beta={beta:g}")
    axes[2].set_title("D-slice quartic residual at beta-specific best fit")
    axes[2].set_xlabel("D")
    axes[2].set_ylabel("slice p95 quartic residual")
    axes[2].legend(fontsize=8)

    fig.suptitle("EYMH Lorentz-gated curvature audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_summary, out_beta_best, out_dsummary, out_decision, out_fig):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_beta_best)
    print(beta_best.to_string(index=False))
    print(json.dumps(gate_decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
D-split piecewise curvature audit for shortlisted EYMH mixed-entry proxies.

This audit scans piecewise low-D / high-D split points and refits the current
shortlisted blended proxy families separately on both sides of the split.  The
goal is to test whether the weak holdout behavior of the global blends is
better described as a two-regime effect along the D-axis.
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


def fit_nonnegative_weighted_ls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(w[:, None] * X, w * y, rcond=None)[0]
    return np.maximum(beta, 0.0)


def score_piecewise(
    xi_proxy: np.ndarray,
    xi_target: np.ndarray,
    a_ref: np.ndarray,
) -> dict[str, float]:
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
        "rank_score": q(np.abs(quartic_resid), 0.95) + 0.25 * q(np.abs(xi_resid), 0.95) + abs(
            float(np.mean(quartic_proxy)) - float(np.mean(quartic_target))
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scan D-axis split points for piecewise blended EYMH curvature proxies."
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
    k_part = coeff_l1 / coeff_l2
    chi_tree = np.maximum(np.asarray(df["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    g_schur = 1.0 + chi_tree
    a_ref = np.asarray(df["coeff_participation_access_parented"], dtype=float) * np.asarray(
        df["tree_diag_compressibility_parented"], dtype=float
    )
    xi_target = np.maximum((k_part - 1.0) * (g_schur - 1.0), 0.0) / np.maximum(k_part * g_schur, 1e-30)

    diag = np.maximum(a_ref, 1e-30)
    proxy_cols = {
        "hk_curv_screen_over_diag": np.maximum(np.asarray(df["hk_curv_screen_offdiag"], dtype=float), 0.0) / diag,
        "hk_abs_over_diag": np.maximum(np.asarray(df["hk_abs_offdiag"], dtype=float), 0.0) / diag,
        "action_abs_over_diag": np.maximum(np.asarray(df["action_abs_offdiag"], dtype=float), 0.0) / diag,
    }

    families = {
        "best_blend_curv_action": ["hk_curv_screen_over_diag", "action_abs_over_diag"],
        "runnerup_blend_abs_action": ["hk_abs_over_diag", "action_abs_over_diag"],
    }

    # Global baselines for comparison.
    global_baselines: dict[str, dict[str, float]] = {}
    for family_name, comps in families.items():
        X = np.column_stack([proxy_cols[c] for c in comps])
        beta = fit_nonnegative_weighted_ls(X, xi_target, a_ref)
        xi_proxy = np.maximum(X @ beta, 0.0)
        base = score_piecewise(xi_proxy, xi_target, a_ref)
        base["beta_1"] = float(beta[0])
        base["beta_2"] = float(beta[1])
        global_baselines[family_name] = base

    d_values = np.sort(df["D"].unique())
    thresholds = [float(d) for d in d_values[4:-4]]

    rows: list[dict[str, float | str]] = []
    detail_rows: list[dict[str, float | str]] = []

    for family_name, comps in families.items():
        X = np.column_stack([proxy_cols[c] for c in comps])
        base = global_baselines[family_name]
        for split_d in thresholds:
            low_mask = np.asarray(df["D"] <= split_d)
            high_mask = np.asarray(df["D"] > split_d)
            if low_mask.sum() == 0 or high_mask.sum() == 0:
                continue

            beta_low = fit_nonnegative_weighted_ls(X[low_mask], xi_target[low_mask], a_ref[low_mask])
            beta_high = fit_nonnegative_weighted_ls(X[high_mask], xi_target[high_mask], a_ref[high_mask])

            xi_piece = np.zeros_like(xi_target)
            xi_piece[low_mask] = np.maximum(X[low_mask] @ beta_low, 0.0)
            xi_piece[high_mask] = np.maximum(X[high_mask] @ beta_high, 0.0)

            full_metrics = score_piecewise(xi_piece, xi_target, a_ref)
            low_metrics = score_piecewise(xi_piece[low_mask], xi_target[low_mask], a_ref[low_mask])
            high_metrics = score_piecewise(xi_piece[high_mask], xi_target[high_mask], a_ref[high_mask])

            rows.append(
                {
                    "family_name": family_name,
                    "split_D": split_d,
                    "n_low": int(low_mask.sum()),
                    "n_high": int(high_mask.sum()),
                    "beta_low_1": float(beta_low[0]),
                    "beta_low_2": float(beta_low[1]),
                    "beta_high_1": float(beta_high[0]),
                    "beta_high_2": float(beta_high[1]),
                    "beta_jump_l1": float(np.abs(beta_low - beta_high).sum()),
                    "improvement_vs_global_rank_score": float(base["rank_score"] - full_metrics["rank_score"]),
                    "improvement_vs_global_p95_quartic": float(base["p95_abs_quartic_resid"] - full_metrics["p95_abs_quartic_resid"]),
                    "full_corr_xi": full_metrics["corr_xi"],
                    "full_corr_quartic": full_metrics["corr_quartic"],
                    "full_p95_abs_xi_resid": full_metrics["p95_abs_xi_resid"],
                    "full_p95_abs_quartic_resid": full_metrics["p95_abs_quartic_resid"],
                    "full_rank_score": full_metrics["rank_score"],
                    "low_corr_quartic": low_metrics["corr_quartic"],
                    "low_p95_abs_quartic_resid": low_metrics["p95_abs_quartic_resid"],
                    "low_rank_score": low_metrics["rank_score"],
                    "high_corr_quartic": high_metrics["corr_quartic"],
                    "high_p95_abs_quartic_resid": high_metrics["p95_abs_quartic_resid"],
                    "high_rank_score": high_metrics["rank_score"],
                }
            )

        fam_df = pd.DataFrame([r for r in rows if r["family_name"] == family_name]).sort_values(
            ["full_rank_score", "full_p95_abs_quartic_resid", "beta_jump_l1"],
            ascending=[True, True, True],
        )
        if not fam_df.empty:
            best = fam_df.iloc[0].to_dict()
            best["global_rank_score"] = float(base["rank_score"])
            best["global_p95_abs_quartic_resid"] = float(base["p95_abs_quartic_resid"])
            detail_rows.append(best)

    summary = pd.DataFrame(rows).sort_values(
        ["full_rank_score", "full_p95_abs_quartic_resid", "beta_jump_l1"],
        ascending=[True, True, True],
    )
    detail = pd.DataFrame(detail_rows).sort_values(
        ["full_rank_score", "full_p95_abs_quartic_resid", "beta_jump_l1"],
        ascending=[True, True, True],
    )

    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_dsplit_piecewise_curvature_audit_summary_{args.tag}.csv"
    out_detail = OUTDIR / f"hll_uv_action_loop_eymh_parent_dsplit_piecewise_curvature_audit_detail_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_dsplit_piecewise_curvature_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_dsplit_piecewise_curvature_audit_run_meta_{args.tag}.json"

    summary.to_csv(out_summary, index=False)
    detail.to_csv(out_detail, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "families": families,
                "thresholds": thresholds,
                "note": "Piecewise D-split audit for shortlisted EYMH curvature blends.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for family_name, fam_df in summary.groupby("family_name"):
        fam_df = fam_df.sort_values("split_D")
        axes[0].plot(fam_df["split_D"], fam_df["full_rank_score"], marker="o", label=family_name)
        axes[1].plot(fam_df["split_D"], fam_df["improvement_vs_global_p95_quartic"], marker="o", label=family_name)
    axes[0].set_xlabel("split D")
    axes[0].set_ylabel("piecewise full rank score")
    axes[0].set_title("Piecewise fit quality")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("split D")
    axes[1].set_ylabel("improvement vs global p95 quartic")
    axes[1].set_title("Quartic residual improvement")
    axes[1].axhline(0.0, linestyle="--", color="black", linewidth=1)
    fig.suptitle("EYMH D-split piecewise curvature audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_summary, out_detail, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_detail)
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()

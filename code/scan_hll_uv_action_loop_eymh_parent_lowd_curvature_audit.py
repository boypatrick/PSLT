#!/usr/bin/env python3
"""
Low-D specific curvature audit for EYMH Lemma 2 proxy families.

This audit focuses on the low-D regime identified by the piecewise D-split
check as the main source of instability for the global blended curvature
families.  It re-ranks single proxies and short two-proxy blends on low-D
subsets only.
"""

from __future__ import annotations

import argparse
import itertools
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
        "rank_score": q(np.abs(quartic_resid), 0.95) + 0.25 * q(np.abs(xi_resid), 0.95) + abs(
            float(np.mean(quartic_proxy)) - float(np.mean(quartic_target))
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit low-D-specific EYMH curvature proxies and blends.")
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
    tree = np.maximum(np.asarray(df["tree_diag_compressibility_parented"], dtype=float), 1e-30)

    # Candidate families: keep the compact set that already showed signal,
    # plus one action_norm family in case low-D prefers the screened version.
    single_proxies = {
        "hk_curv_screen_offdiag:over_diag": np.maximum(np.asarray(df["hk_curv_screen_offdiag"], dtype=float), 0.0)
        / diag,
        "hk_abs_offdiag:over_diag": np.maximum(np.asarray(df["hk_abs_offdiag"], dtype=float), 0.0) / diag,
        "action_abs_offdiag:over_diag": np.maximum(np.asarray(df["action_abs_offdiag"], dtype=float), 0.0) / diag,
        "action_abs_offdiag:over_tree": np.maximum(np.asarray(df["action_abs_offdiag"], dtype=float), 0.0) / tree,
        "action_norm_offdiag:over_diag": np.maximum(np.asarray(df["action_norm_offdiag"], dtype=float), 0.0) / diag,
    }

    blend_pairs = [
        ("hk_curv_screen_offdiag:over_diag", "action_abs_offdiag:over_diag"),
        ("hk_abs_offdiag:over_diag", "action_abs_offdiag:over_diag"),
        ("hk_abs_offdiag:over_diag", "action_norm_offdiag:over_diag"),
        ("hk_curv_screen_offdiag:over_diag", "action_norm_offdiag:over_diag"),
    ]

    cuts = [8.0, 8.8, 9.6]
    rows: list[dict[str, float | str]] = []
    best_rows: list[dict[str, float | str]] = []

    for cut in cuts:
        mask = np.asarray(df["D"] <= cut)
        xi_sub = xi_target[mask]
        a_sub = a_ref[mask]

        # single
        for label, proxy_full in single_proxies.items():
            proxy = proxy_full[mask]
            denom = float(np.sum((a_sub**2) * proxy * proxy))
            coeff = 0.0 if denom <= 1e-30 else float(np.sum((a_sub**2) * proxy * xi_sub) / denom)
            xi_proxy = np.maximum(coeff * proxy, 0.0)
            metric = score_family(xi_proxy, xi_sub, a_sub)
            rows.append(
                {
                    "cut_D": cut,
                    "family_label": f"single::{label}",
                    "n_components": 1,
                    "component_1": label,
                    "component_2": "",
                    "coeff_1": coeff,
                    "coeff_2": np.nan,
                    **metric,
                }
            )

        # blends
        for c1, c2 in blend_pairs:
            X = np.column_stack([single_proxies[c1][mask], single_proxies[c2][mask]])
            beta = np.linalg.lstsq(a_sub[:, None] * X, a_sub * xi_sub, rcond=None)[0]
            beta = np.maximum(beta, 0.0)
            xi_proxy = np.maximum(X @ beta, 0.0)
            metric = score_family(xi_proxy, xi_sub, a_sub)
            rows.append(
                {
                    "cut_D": cut,
                    "family_label": f"blend::{c1}+{c2}",
                    "n_components": 2,
                    "component_1": c1,
                    "component_2": c2,
                    "coeff_1": float(beta[0]),
                    "coeff_2": float(beta[1]),
                    **metric,
                }
            )

        cut_df = pd.DataFrame([r for r in rows if r["cut_D"] == cut]).sort_values(
            ["rank_score", "p95_abs_quartic_resid", "p95_abs_xi_resid"],
            ascending=[True, True, True],
        )
        best_rows.append(cut_df.iloc[0].to_dict())

    summary = pd.DataFrame(rows).sort_values(
        ["cut_D", "rank_score", "p95_abs_quartic_resid", "p95_abs_xi_resid"],
        ascending=[True, True, True, True],
    )
    best = pd.DataFrame(best_rows).sort_values("cut_D")

    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_lowd_curvature_audit_summary_{args.tag}.csv"
    out_best = OUTDIR / f"hll_uv_action_loop_eymh_parent_lowd_curvature_audit_best_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_lowd_curvature_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_lowd_curvature_audit_run_meta_{args.tag}.json"

    summary.to_csv(out_summary, index=False)
    best.to_csv(out_best, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "cuts": cuts,
                "single_proxies": list(single_proxies),
                "blend_pairs": blend_pairs,
                "note": "Low-D-only curvature audit over selected single and blended proxy families.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for cut in cuts:
        sub = summary[summary["cut_D"] == cut].head(5)
        axes[0].plot(sub["rank_score"].to_numpy(), marker="o", label=f"D<={cut:g}")
    axes[0].set_title("Top-5 rank scores per low-D cut")
    axes[0].set_xlabel("ranked family index")
    axes[0].set_ylabel("rank score")
    axes[0].legend(fontsize=8)

    axes[1].plot(best["cut_D"], best["p95_abs_quartic_resid"], marker="o", label="best family")
    axes[1].plot(best["cut_D"], best["p95_abs_xi_resid"], marker="s", label="best xi residual")
    axes[1].set_title("Best low-D family by cut")
    axes[1].set_xlabel("low-D cut")
    axes[1].legend(fontsize=8)

    fig.suptitle("EYMH low-D curvature audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_summary, out_best, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_best)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Second-pass shortlist audit for EYMH generic off-diagonal curvature proxies.

This audit takes the strongest first-pass curvature witnesses and compares
single-proxy and two-proxy blended families against the canonical even-lambda
quartic target.  The output is still a design-stage proxy comparison, but it is
focused enough to support the next Lemma 2 proof-strengthening decision.
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


def score_family(
    *,
    label: str,
    xi_proxy: np.ndarray,
    xi_target: np.ndarray,
    a_ref: np.ndarray,
) -> dict[str, float | str]:
    quartic_proxy = 0.25 * (a_ref**2) * (xi_proxy**2)
    quartic_target = 0.25 * (a_ref**2) * (xi_target**2)
    xi_resid = xi_proxy - xi_target
    quartic_resid = quartic_proxy - quartic_target
    c4_proxy = float(np.mean(quartic_proxy))
    c4_target = float(np.mean(quartic_target))
    return {
        "family_label": label,
        "c4_target": c4_target,
        "c4_proxy": c4_proxy,
        "abs_c4_gap": abs(c4_proxy - c4_target),
        "rel_c4_gap": abs(c4_proxy - c4_target) / max(abs(c4_target), 1e-30),
        "corr_xi": corr(xi_proxy, xi_target),
        "corr_quartic": corr(quartic_proxy, quartic_target),
        "p95_abs_xi_resid": q(np.abs(xi_resid), 0.95),
        "p95_abs_quartic_resid": q(np.abs(quartic_resid), 0.95),
        "max_abs_xi_resid": float(np.max(np.abs(xi_resid))),
        "max_abs_quartic_resid": float(np.max(np.abs(quartic_resid))),
        "mean_xi_proxy": float(np.mean(xi_proxy)),
        "mean_xi_target": float(np.mean(xi_target)),
        "rank_score": (
            q(np.abs(quartic_resid), 0.95)
            + 0.25 * q(np.abs(xi_resid), 0.95)
            + abs(c4_proxy - c4_target)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a focused second-pass shortlist audit for EYMH curvature-based off-diagonal proxies."
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

    proxies = {
        "hk_curv_screen_offdiag:over_diag": np.maximum(np.asarray(df["hk_curv_screen_offdiag"], dtype=float), 0.0)
        / diag,
        "hk_abs_offdiag:over_diag": np.maximum(np.asarray(df["hk_abs_offdiag"], dtype=float), 0.0) / diag,
        "action_abs_offdiag:over_diag": np.maximum(np.asarray(df["action_abs_offdiag"], dtype=float), 0.0)
        / diag,
    }

    rows: list[dict[str, float | str]] = []
    d_rows: list[dict[str, float | str]] = []

    # Single-family fits
    for label, proxy in proxies.items():
        denom = float(np.sum((a_ref**2) * proxy * proxy))
        scale = 0.0 if denom <= 1e-30 else float(np.sum((a_ref**2) * proxy * xi_target) / denom)
        xi_proxy = np.maximum(scale * proxy, 0.0)
        row = score_family(label=f"single::{label}", xi_proxy=xi_proxy, xi_target=xi_target, a_ref=a_ref)
        row["n_components"] = 1
        row["coeff_1"] = scale
        row["coeff_2"] = np.nan
        row["component_1"] = label
        row["component_2"] = ""
        rows.append(row)

        quartic_proxy = 0.25 * (a_ref**2) * (xi_proxy**2)
        quartic_target = 0.25 * (a_ref**2) * (xi_target**2)
        per_d = (
            pd.DataFrame(
                {"D": df["D"], "abs_quartic_resid": np.abs(quartic_proxy - quartic_target)}
            )
            .groupby("D", dropna=False)
            .agg(p95_abs_quartic_resid=("abs_quartic_resid", lambda s: float(pd.Series(s).quantile(0.95))))
            .reset_index()
        )
        worst_idx = int(per_d["p95_abs_quartic_resid"].idxmax())
        d_rows.append(
            {
                "family_label": f"single::{label}",
                "worst_D_by_quartic_p95": float(per_d.iloc[worst_idx]["D"]),
                "worst_D_quartic_p95": float(per_d.iloc[worst_idx]["p95_abs_quartic_resid"]),
                "mean_D_quartic_p95": float(per_d["p95_abs_quartic_resid"].mean()),
            }
        )

    # Two-proxy blends
    labels = list(proxies)
    for l1, l2 in itertools.combinations(labels, 2):
        X = np.column_stack([proxies[l1], proxies[l2]])
        beta = np.linalg.lstsq(a_ref[:, None] * X, a_ref * xi_target, rcond=None)[0]
        xi_proxy = np.maximum(X @ beta, 0.0)
        row = score_family(label=f"blend::{l1}+{l2}", xi_proxy=xi_proxy, xi_target=xi_target, a_ref=a_ref)
        row["n_components"] = 2
        row["coeff_1"] = float(beta[0])
        row["coeff_2"] = float(beta[1])
        row["component_1"] = l1
        row["component_2"] = l2
        rows.append(row)

        quartic_proxy = 0.25 * (a_ref**2) * (xi_proxy**2)
        quartic_target = 0.25 * (a_ref**2) * (xi_target**2)
        per_d = (
            pd.DataFrame(
                {"D": df["D"], "abs_quartic_resid": np.abs(quartic_proxy - quartic_target)}
            )
            .groupby("D", dropna=False)
            .agg(p95_abs_quartic_resid=("abs_quartic_resid", lambda s: float(pd.Series(s).quantile(0.95))))
            .reset_index()
        )
        worst_idx = int(per_d["p95_abs_quartic_resid"].idxmax())
        d_rows.append(
            {
                "family_label": f"blend::{l1}+{l2}",
                "worst_D_by_quartic_p95": float(per_d.iloc[worst_idx]["D"]),
                "worst_D_quartic_p95": float(per_d.iloc[worst_idx]["p95_abs_quartic_resid"]),
                "mean_D_quartic_p95": float(per_d["p95_abs_quartic_resid"].mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["rank_score", "abs_c4_gap", "p95_abs_quartic_resid", "p95_abs_xi_resid"],
        ascending=[True, True, True, True],
    )
    d_summary = pd.DataFrame(d_rows)
    detail = summary.head(6).merge(d_summary, on="family_label", how="left")

    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_shortlist_audit_summary_{args.tag}.csv"
    out_detail = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_shortlist_audit_detail_{args.tag}.csv"
    out_dsummary = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_shortlist_audit_dsummary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_shortlist_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_shortlist_audit_run_meta_{args.tag}.json"

    summary.to_csv(out_summary, index=False)
    detail.to_csv(out_detail, index=False)
    d_summary.to_csv(out_dsummary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "shortlisted_components": list(proxies),
                "note": "Second-pass shortlist audit over single and two-proxy blended curvature families.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    top = summary.head(6).copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    axes[0].barh(top["family_label"], top["rank_score"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("rank score")
    axes[0].set_title("Shortlist ranking")

    axes[1].barh(top["family_label"], top["corr_quartic"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("corr(quartic_proxy, quartic_target)")
    axes[1].set_title("Quartic alignment")

    axes[2].barh(top["family_label"], top["p95_abs_quartic_resid"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("p95 |quartic residual|")
    axes[2].set_title("Residual stability")

    fig.suptitle("EYMH curvature shortlist and blended-proxy audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_summary, out_detail, out_dsummary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

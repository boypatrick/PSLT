#!/usr/bin/env python3
"""
Blocked holdout audit for shortlisted EYMH curvature blends.

There is currently only one exported canonical EYMH parented map in the repo,
so this audit treats holdout robustness as a within-map blocked validation
problem rather than a cross-grid comparison.  Blend coefficients are fit on
training subsets and then evaluated on held-out D / eta blocks.
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


def fit_weighted_nonnegative_ls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(w[:, None] * X, w * y, rcond=None)[0]
    return np.maximum(beta, 0.0)


def evaluate(
    *,
    X: np.ndarray,
    beta: np.ndarray,
    xi_target: np.ndarray,
    a_ref: np.ndarray,
) -> dict[str, float]:
    xi_proxy = np.maximum(X @ beta, 0.0)
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
        "c4_proxy": float(np.mean(quartic_proxy)),
        "c4_target": float(np.mean(quartic_target)),
        "abs_c4_gap": abs(float(np.mean(quartic_proxy)) - float(np.mean(quartic_target))),
        "rank_score": q(np.abs(quartic_resid), 0.95) + 0.25 * q(np.abs(xi_resid), 0.95),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run within-map blocked holdout validation for shortlisted EYMH curvature blends."
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
        "hk_curv_screen_over_diag": np.maximum(np.asarray(df["hk_curv_screen_offdiag"], dtype=float), 0.0) / diag,
        "hk_abs_over_diag": np.maximum(np.asarray(df["hk_abs_offdiag"], dtype=float), 0.0) / diag,
        "action_abs_over_diag": np.maximum(np.asarray(df["action_abs_offdiag"], dtype=float), 0.0) / diag,
    }

    blend_families = {
        "best_blend_curv_action": ["hk_curv_screen_over_diag", "action_abs_over_diag"],
        "runnerup_blend_abs_action": ["hk_abs_over_diag", "action_abs_over_diag"],
    }

    unique_d = np.sort(df["D"].unique())
    unique_eta = np.sort(df["eta"].unique())
    d_mid = float(np.median(unique_d))
    eta_mid = float(np.median(unique_eta))

    holdouts = {
        "D_low": np.asarray(df["D"] <= d_mid),
        "D_high": np.asarray(df["D"] > d_mid),
        "D_even_index": np.asarray(df["D"].map({d: (i % 2 == 0) for i, d in enumerate(unique_d)}), dtype=bool),
        "D_odd_index": np.asarray(df["D"].map({d: (i % 2 == 1) for i, d in enumerate(unique_d)}), dtype=bool),
        "eta_low": np.asarray(df["eta"] <= eta_mid),
        "eta_high": np.asarray(df["eta"] > eta_mid),
        "eta_even_index": np.asarray(
            df["eta"].map({e: (i % 2 == 0) for i, e in enumerate(unique_eta)}), dtype=bool
        ),
        "eta_odd_index": np.asarray(
            df["eta"].map({e: (i % 2 == 1) for i, e in enumerate(unique_eta)}), dtype=bool
        ),
    }

    rows: list[dict[str, float | str]] = []
    coef_rows: list[dict[str, float | str]] = []

    for family_name, comps in blend_families.items():
        X_full = np.column_stack([proxies[c] for c in comps])
        beta_full = fit_weighted_nonnegative_ls(X_full, xi_target, a_ref)
        full_metrics = evaluate(X=X_full, beta=beta_full, xi_target=xi_target, a_ref=a_ref)
        rows.append(
            {
                "family_name": family_name,
                "split": "full",
                "subset": "all",
                "n_points": int(len(df)),
                **full_metrics,
            }
        )
        coef_rows.append(
            {
                "family_name": family_name,
                "split": "full",
                "subset": "all",
                "component_1": comps[0],
                "component_2": comps[1],
                "beta_1": float(beta_full[0]),
                "beta_2": float(beta_full[1]),
            }
        )

        for split_name, holdout_mask in holdouts.items():
            train_mask = ~holdout_mask
            if int(np.sum(holdout_mask)) == 0 or int(np.sum(train_mask)) == 0:
                continue

            beta_train = fit_weighted_nonnegative_ls(X_full[train_mask], xi_target[train_mask], a_ref[train_mask])
            train_metrics = evaluate(
                X=X_full[train_mask], beta=beta_train, xi_target=xi_target[train_mask], a_ref=a_ref[train_mask]
            )
            holdout_metrics = evaluate(
                X=X_full[holdout_mask],
                beta=beta_train,
                xi_target=xi_target[holdout_mask],
                a_ref=a_ref[holdout_mask],
            )

            rows.append(
                {
                    "family_name": family_name,
                    "split": split_name,
                    "subset": "train",
                    "n_points": int(np.sum(train_mask)),
                    **train_metrics,
                }
            )
            rows.append(
                {
                    "family_name": family_name,
                    "split": split_name,
                    "subset": "holdout",
                    "n_points": int(np.sum(holdout_mask)),
                    **holdout_metrics,
                }
            )
            coef_rows.append(
                {
                    "family_name": family_name,
                    "split": split_name,
                    "subset": "train_fit",
                    "component_1": comps[0],
                    "component_2": comps[1],
                    "beta_1": float(beta_train[0]),
                    "beta_2": float(beta_train[1]),
                }
            )

    summary = pd.DataFrame(rows)
    coef_summary = pd.DataFrame(coef_rows)

    holdout_only = summary[summary["subset"] == "holdout"].copy()
    gate_rows = []
    for family_name, sub in holdout_only.groupby("family_name"):
        full_row = summary[(summary["family_name"] == family_name) & (summary["split"] == "full")].iloc[0]
        gate_rows.append(
            {
                "family_name": family_name,
                "holdout_p95_abs_quartic_resid_max": float(sub["p95_abs_quartic_resid"].max()),
                "holdout_p95_abs_quartic_resid_mean": float(sub["p95_abs_quartic_resid"].mean()),
                "holdout_p95_abs_xi_resid_max": float(sub["p95_abs_xi_resid"].max()),
                "holdout_corr_quartic_min": float(sub["corr_quartic"].min()),
                "holdout_corr_xi_min": float(sub["corr_xi"].min()),
                "full_rank_score": float(full_row["rank_score"]),
                "holdout_rank_score_mean": float(sub["rank_score"].mean()),
                "beta_1_range": float(
                    coef_summary[coef_summary["family_name"] == family_name]["beta_1"].max()
                    - coef_summary[coef_summary["family_name"] == family_name]["beta_1"].min()
                ),
                "beta_2_range": float(
                    coef_summary[coef_summary["family_name"] == family_name]["beta_2"].max()
                    - coef_summary[coef_summary["family_name"] == family_name]["beta_2"].min()
                ),
            }
        )
    gates = pd.DataFrame(gate_rows).sort_values("holdout_rank_score_mean")

    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_blend_holdout_audit_summary_{args.tag}.csv"
    out_coef = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_blend_holdout_audit_coeffs_{args.tag}.csv"
    out_gates = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_blend_holdout_audit_gates_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_blend_holdout_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_blend_holdout_audit_run_meta_{args.tag}.json"

    summary.to_csv(out_summary, index=False)
    coef_summary.to_csv(out_coef, index=False)
    gates.to_csv(out_gates, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "note": "Within-map blocked holdout audit because only one canonical EYMH parented map is currently available.",
                "blend_families": blend_families,
                "holdout_splits": list(holdouts),
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    holdout_plot = holdout_only.copy()
    labels = [f"{r.family_name}\n{r.split}" for r in holdout_plot.itertuples()]

    axes[0].barh(labels, holdout_plot["p95_abs_quartic_resid"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("holdout p95 |quartic residual|")
    axes[0].set_title("Holdout quartic stability")

    axes[1].barh(labels, holdout_plot["corr_quartic"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("holdout corr(quartic)")
    axes[1].set_title("Holdout quartic alignment")

    coef_plot = coef_summary[coef_summary["subset"].isin(["all", "train_fit"])].copy()
    coef_plot["label"] = coef_plot["family_name"] + "\n" + coef_plot["split"]
    axes[2].scatter(coef_plot["beta_1"], coef_plot["beta_2"])
    for row in coef_plot.itertuples():
        axes[2].annotate(row.label, (row.beta_1, row.beta_2), fontsize=7)
    axes[2].set_xlabel("beta_1")
    axes[2].set_ylabel("beta_2")
    axes[2].set_title("Coefficient stability")

    fig.suptitle("EYMH blended curvature holdout audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_summary, out_coef, out_gates, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_gates)
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()

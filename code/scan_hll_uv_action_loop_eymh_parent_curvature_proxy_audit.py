#!/usr/bin/env python3
"""
Curvature-based proxy audit for the generic EYMH off-diagonal sector.

This audit is intentionally weaker than a theorem.  It does not attempt to
prove the mixed parent-kernel entry from the determinant law alone.  Instead it
screens independently exported off-diagonal witnesses against the canonical
even-lambda quartic target in order to identify which proxy family is most
promising for a later non-circular Lemma 2 strengthening.
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


def safe_name(label: str) -> str:
    out = []
    for ch in label:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank off-diagonal proxy families against the canonical EYMH lambda-curvature target."
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

    # Canonical quartic target from the current even-lambda EYMH chain.
    xi_target = np.maximum((k_part - 1.0) * (g_schur - 1.0), 0.0) / np.maximum(k_part * g_schur, 1e-30)
    quartic_target = 0.25 * (a_ref**2) * (xi_target**2)
    c4_target = float(np.mean(quartic_target))

    part_access = np.maximum(np.asarray(df["coeff_participation_access_parented"], dtype=float), 1e-30)
    tree_comp = np.maximum(np.asarray(df["tree_diag_compressibility_parented"], dtype=float), 1e-30)
    diag_weight = np.maximum(a_ref, 1e-30)
    root_diag_weight = np.sqrt(diag_weight)

    proxy_builders: list[tuple[str, np.ndarray]] = []
    raw_proxies = [
        "action_abs_offdiag",
        "action_norm_offdiag",
        "hk_mass_access_offdiag",
        "hk_loop_prefactor_offdiag",
        "hk_curv_screen_offdiag",
        "hk_abs_offdiag",
        "hk_abs_offdiag_geom",
        "hk_loop_local_prefactor_offdiag",
        "eymh_parented_prefactor_offdiag",
    ]

    for col in raw_proxies:
        if col not in df.columns:
            continue
        raw = np.maximum(np.asarray(df[col], dtype=float), 0.0)
        proxy_builders.extend(
            [
                (f"{col}:raw", raw),
                (f"{col}:over_diag", raw / diag_weight),
                (f"{col}:over_sqrt_diag", raw / root_diag_weight),
                (f"{col}:over_part", raw / part_access),
                (f"{col}:over_tree", raw / tree_comp),
            ]
        )

    rows: list[dict[str, float | str]] = []
    d_rows: list[dict[str, float | str]] = []
    eta_rows: list[dict[str, float | str]] = []

    w = a_ref**2
    for label, base_proxy in proxy_builders:
        proxy = np.maximum(base_proxy, 0.0)
        if np.allclose(proxy, proxy[0]):
            continue

        denom = float(np.sum(w * proxy * proxy))
        scale = 0.0 if denom <= 1e-30 else float(np.sum(w * proxy * xi_target) / denom)
        xi_proxy = np.maximum(scale * proxy, 0.0)
        quartic_proxy = 0.25 * (a_ref**2) * (xi_proxy**2)
        c4_proxy = float(np.mean(quartic_proxy))

        xi_resid = xi_proxy - xi_target
        quartic_resid = quartic_proxy - quartic_target

        row = {
            "proxy_label": label,
            "proxy_family": label.split(":", 1)[0],
            "proxy_normalization": label.split(":", 1)[1],
            "scale_to_xi_target": scale,
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
        rows.append(row)

        per_d = (
            pd.DataFrame(
                {
                    "D": df["D"],
                    "abs_xi_resid": np.abs(xi_resid),
                    "abs_quartic_resid": np.abs(quartic_resid),
                }
            )
            .groupby("D", dropna=False)
            .agg(
                p95_abs_xi_resid=("abs_xi_resid", lambda s: float(pd.Series(s).quantile(0.95))),
                p95_abs_quartic_resid=("abs_quartic_resid", lambda s: float(pd.Series(s).quantile(0.95))),
            )
            .reset_index()
        )
        worst_d_idx = int(per_d["p95_abs_quartic_resid"].idxmax())
        d_rows.append(
            {
                "proxy_label": label,
                "worst_D_by_quartic_p95": float(per_d.iloc[worst_d_idx]["D"]),
                "worst_D_quartic_p95": float(per_d.iloc[worst_d_idx]["p95_abs_quartic_resid"]),
                "mean_D_quartic_p95": float(per_d["p95_abs_quartic_resid"].mean()),
            }
        )

        per_eta = (
            pd.DataFrame(
                {
                    "eta": df["eta"],
                    "abs_quartic_resid": np.abs(quartic_resid),
                }
            )
            .groupby("eta", dropna=False)
            .agg(p95_abs_quartic_resid=("abs_quartic_resid", lambda s: float(pd.Series(s).quantile(0.95))))
            .reset_index()
        )
        worst_eta_idx = int(per_eta["p95_abs_quartic_resid"].idxmax())
        eta_rows.append(
            {
                "proxy_label": label,
                "worst_eta_by_quartic_p95": float(per_eta.iloc[worst_eta_idx]["eta"]),
                "worst_eta_quartic_p95": float(per_eta.iloc[worst_eta_idx]["p95_abs_quartic_resid"]),
                "mean_eta_quartic_p95": float(per_eta["p95_abs_quartic_resid"].mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["rank_score", "abs_c4_gap", "p95_abs_quartic_resid", "p95_abs_xi_resid"],
        ascending=[True, True, True, True],
    )
    d_summary = pd.DataFrame(d_rows)
    eta_summary = pd.DataFrame(eta_rows)

    top_labels = set(summary.head(6)["proxy_label"].tolist())
    detail = summary[summary["proxy_label"].isin(top_labels)].copy()
    detail = detail.merge(d_summary, on="proxy_label", how="left").merge(eta_summary, on="proxy_label", how="left")

    out_summary = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_proxy_audit_summary_{args.tag}.csv"
    out_detail = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_proxy_audit_detail_{args.tag}.csv"
    out_dsummary = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_proxy_audit_dsummary_{args.tag}.csv"
    out_etasummary = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_proxy_audit_etasummary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_proxy_audit_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_parent_curvature_proxy_audit_run_meta_{args.tag}.json"

    summary.to_csv(out_summary, index=False)
    detail.to_csv(out_detail, index=False)
    d_summary.to_csv(out_dsummary, index=False)
    eta_summary.to_csv(out_etasummary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "parented_map": str(args.parented_map),
                "target": "canonical lambda-even quartic density",
                "note": "Proxy design audit only; not a non-circular theorem proof.",
                "n_proxy_families": int(len(summary)),
            },
            indent=2,
            sort_keys=True,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    top = summary.head(8).copy()

    axes[0].barh(top["proxy_label"], top["corr_xi"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("corr(xi_proxy, xi_target)")
    axes[0].set_title("Top proxy correlations")

    axes[1].barh(top["proxy_label"], top["abs_c4_gap"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"$|C_4^{proxy} - C_4^{target}|$")
    axes[1].set_title("Quartic-coefficient gap")

    axes[2].barh(top["proxy_label"], top["p95_abs_quartic_resid"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("p95 |quartic residual|")
    axes[2].set_title("Quartic residual stability")

    fig.suptitle("EYMH curvature-based generic off-diagonal proxy audit", fontsize=13)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    for path in (out_summary, out_detail, out_dsummary, out_etasummary, out_fig, out_meta):
        target = PAPERDIR / path.name
        if path.suffix.lower() == ".png":
            target.write_bytes(path.read_bytes())
        else:
            target.write_text(path.read_text())

    print(out_summary)
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

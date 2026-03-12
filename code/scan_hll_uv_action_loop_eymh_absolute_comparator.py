#!/usr/bin/env python3
"""
Compare UV matching modes including action_loop_eymh_absolute on a common D21xE21 grid.

This is comparator-only. It checks whether the EYMH-side absolute loop-prefactor
witness remains a small deformation relative to the current
action_absolute / action_loop_contrast / action_loop_absolute hierarchy.
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
PAPER_DIR = ROOT / "paper"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare constant/action_absolute/action_loop_contrast/action_loop_absolute/action_loop_eymh_absolute UV matching modes"
    )
    ap.add_argument("--constant-map", type=Path, default=OUTDIR / "hll_uv_to_eft_map_uv_constant_refresh_D21E21.csv")
    ap.add_argument("--absolute-map", type=Path, default=OUTDIR / "hll_uv_to_eft_map_uv_action_absolute_D21E21.csv")
    ap.add_argument("--contrast-map", type=Path, default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_contrast_D21E21.csv")
    ap.add_argument("--loop-absolute-map", type=Path, default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_absolute_D21E21.csv")
    ap.add_argument("--eymh-map", type=Path, default=OUTDIR / "hll_uv_to_eft_map_uv_action_loop_eymh_absolute_D21E21.csv")
    ap.add_argument("--constant-basis-summary", type=Path, default=OUTDIR / "hll_uv_operator_basis_summary.csv")
    ap.add_argument("--absolute-basis-summary", type=Path, default=OUTDIR / "hll_uv_operator_basis_summary_uv_action_absolute_D21E21.csv")
    ap.add_argument("--contrast-basis-summary", type=Path, default=OUTDIR / "hll_uv_operator_basis_summary_uv_action_loop_contrast_D21E21.csv")
    ap.add_argument("--loop-absolute-basis-summary", type=Path, default=OUTDIR / "hll_uv_operator_basis_summary_uv_action_loop_absolute_D21E21.csv")
    ap.add_argument("--eymh-basis-summary", type=Path, default=OUTDIR / "hll_uv_operator_basis_summary_uv_action_loop_eymh_absolute_D21E21.csv")
    ap.add_argument("--tag", type=str, default="D21E21")
    ap.add_argument("--skip-paper-copy", action="store_true")
    return ap.parse_args()


def load_map(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    keep = [
        "D",
        "eta",
        "mu_mumu_uv_rge",
        "abs_delta_mu_mumu",
        "chi2_uv_rge",
        "kappa_diag_eff",
        "kappa_offdiag_eff",
        "action_abs_diag",
        "hk_curv_access",
        "hk_curv_contrast_log",
        "hk_loop_prefactor_diag",
        "eymh_loop_prefactor_diag",
        "eymh_loop_prefactor_offdiag",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].rename(columns={c: f"{c}_{label}" for c in keep if c not in {"D", "eta"}})


def summarize_mode(df: pd.DataFrame, label: str, basis_summary_path: Path) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "mode": label,
        "n_points": int(len(df)),
        "mean_abs_delta_mu_mumu": float(df[f"abs_delta_mu_mumu_{label}"].mean()),
        "p95_abs_delta_mu_mumu": float(df[f"abs_delta_mu_mumu_{label}"].quantile(0.95)),
        "max_abs_delta_mu_mumu": float(df[f"abs_delta_mu_mumu_{label}"].max()),
        "f_chi2_le_4_uv_rge": float((df[f"chi2_uv_rge_{label}"] <= 4.0).mean()),
        "best_chi2_uv_rge": float(df[f"chi2_uv_rge_{label}"].min()),
    }
    if basis_summary_path.exists():
        bs = pd.read_csv(basis_summary_path).iloc[0]
        for c in [
            "max_tree_rebuild_residual",
            "max_match_rebuild_residual",
            "max_ir_rebuild_residual",
            "max_abs_delta_mu_mumu_ir_recon_vs_native",
        ]:
            if c in bs.index:
                row[c] = float(bs[c])
    for c in [
        "kappa_diag_eff",
        "action_abs_diag",
        "hk_curv_access",
        "hk_loop_prefactor_diag",
        "eymh_loop_prefactor_diag",
    ]:
        col = f"{c}_{label}"
        if col in df.columns:
            row[f"{c}_p50"] = float(df[col].quantile(0.5))
            row[f"{c}_p90"] = float(df[col].quantile(0.9))
    return row


def summarize_pair(df: pd.DataFrame, a: str, b: str) -> dict[str, float | str]:
    delta = (df[f"mu_mumu_uv_rge_{a}"] - df[f"mu_mumu_uv_rge_{b}"]).abs()
    mismatch = ((df[f"chi2_uv_rge_{a}"] <= 4.0) != (df[f"chi2_uv_rge_{b}"] <= 4.0)).astype(float)
    return {
        "pair": f"{a}_vs_{b}",
        "mean_abs_mapdiff": float(delta.mean()),
        "p95_abs_mapdiff": float(delta.quantile(0.95)),
        "max_abs_mapdiff": float(delta.max()),
        "acceptance_mismatch_fraction": float(mismatch.mean()),
    }


def plot_pairwise(out_png: Path, merged: pd.DataFrame) -> None:
    d_vals = np.sort(merged["D"].unique())
    eta_vals = np.sort(merged["eta"].unique())
    extent = [float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())]

    def arr_from_series(series: pd.Series) -> np.ndarray:
        return (
            pd.DataFrame({"D": merged["D"], "eta": merged["eta"], "val": np.asarray(series, dtype=float)})
            .pivot(index="eta", columns="D", values="val")
            .reindex(index=eta_vals, columns=d_vals)
            .to_numpy()
        )

    panels = [
        ("const vs abs", arr_from_series((merged["mu_mumu_uv_rge_constant"] - merged["mu_mumu_uv_rge_action_absolute"]).abs())),
        ("abs vs loop_abs", arr_from_series((merged["mu_mumu_uv_rge_action_absolute"] - merged["mu_mumu_uv_rge_action_loop_absolute"]).abs())),
        ("loop_abs vs eymh", arr_from_series((merged["mu_mumu_uv_rge_action_loop_absolute"] - merged["mu_mumu_uv_rge_action_loop_eymh_absolute"]).abs())),
        ("const vs eymh", arr_from_series((merged["mu_mumu_uv_rge_constant"] - merged["mu_mumu_uv_rge_action_loop_eymh_absolute"]).abs())),
        ("hk_loop_prefactor_diag", arr_from_series(merged["hk_loop_prefactor_diag_action_loop_absolute"])),
        ("eymh_loop_prefactor_diag", arr_from_series(merged["eymh_loop_prefactor_diag_action_loop_eymh_absolute"])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    for ax, (title, arr) in zip(axes.ravel(), panels):
        p05, p95 = np.percentile(arr, [5.0, 95.0])
        if p95 <= p05:
            p05 = float(np.min(arr))
            p95 = float(np.max(arr)) + 1e-12
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=float(p05), vmax=float(p95))
        ax.set_title(title)
        ax.set_xlabel("D")
        ax.set_ylabel("eta")
        fig.colorbar(im, ax=ax)
    fig.suptitle("UV matching comparator including action_loop_eymh_absolute", fontsize=13)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    const = load_map(args.constant_map, "constant")
    absv = load_map(args.absolute_map, "action_absolute")
    contrast = load_map(args.contrast_map, "action_loop_contrast")
    loop_abs = load_map(args.loop_absolute_map, "action_loop_absolute")
    eymh = load_map(args.eymh_map, "action_loop_eymh_absolute")
    merged = const.merge(absv, on=["D", "eta"]).merge(contrast, on=["D", "eta"]).merge(loop_abs, on=["D", "eta"]).merge(eymh, on=["D", "eta"])

    mode_summary = pd.DataFrame(
        [
            summarize_mode(merged, "constant", args.constant_basis_summary),
            summarize_mode(merged, "action_absolute", args.absolute_basis_summary),
            summarize_mode(merged, "action_loop_contrast", args.contrast_basis_summary),
            summarize_mode(merged, "action_loop_absolute", args.loop_absolute_basis_summary),
            summarize_mode(merged, "action_loop_eymh_absolute", args.eymh_basis_summary),
        ]
    )
    pair_summary = pd.DataFrame(
        [
            summarize_pair(merged, "constant", "action_absolute"),
            summarize_pair(merged, "constant", "action_loop_contrast"),
            summarize_pair(merged, "constant", "action_loop_absolute"),
            summarize_pair(merged, "constant", "action_loop_eymh_absolute"),
            summarize_pair(merged, "action_absolute", "action_loop_contrast"),
            summarize_pair(merged, "action_loop_contrast", "action_loop_absolute"),
            summarize_pair(merged, "action_loop_absolute", "action_loop_eymh_absolute"),
        ]
    )

    out_mode = OUTDIR / f"hll_uv_action_loop_eymh_absolute_mode_summary_{args.tag}.csv"
    out_pair = OUTDIR / f"hll_uv_action_loop_eymh_absolute_pairwise_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_loop_eymh_absolute_comparator_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_loop_eymh_absolute_comparator_run_meta_{args.tag}.json"

    mode_summary.to_csv(out_mode, index=False)
    pair_summary.to_csv(out_pair, index=False)
    plot_pairwise(out_fig, merged)
    out_meta.write_text(
        json.dumps(
            {
                "constant_map": str(args.constant_map),
                "absolute_map": str(args.absolute_map),
                "contrast_map": str(args.contrast_map),
                "loop_absolute_map": str(args.loop_absolute_map),
                "eymh_map": str(args.eymh_map),
                "tag": args.tag,
                "n_points": int(len(merged)),
            },
            indent=2,
        )
    )

    if not args.skip_paper_copy:
        for src in [out_mode, out_pair, out_fig, out_meta]:
            (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(mode_summary.to_string(index=False))
    print(pair_summary.to_string(index=False))
    print(f"[saved] {out_mode}")
    print(f"[saved] {out_pair}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

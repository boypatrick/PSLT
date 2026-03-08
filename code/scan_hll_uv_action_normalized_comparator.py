#!/usr/bin/env python3
"""
Compare the three canonical UV finite-match modes on a common D21xE21 grid:

  1. constant
  2. input_tied (canonical diagonal-only witness)
  3. action_normalized (parent-action-side normalized witness)

The script exports:
  - per-mode summary table
  - pairwise map-drift summary table
  - a compact four-panel figure showing pairwise |Delta mu_mumu| maps
    and the action-normalization witness map
  - run metadata
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
    ap = argparse.ArgumentParser(description="Compare constant/input_tied/action_normalized UV matching modes")
    ap.add_argument(
        "--constant-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_constant_refresh_D21E21.csv",
    )
    ap.add_argument(
        "--input-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_input_tied_diagonly_D21E21.csv",
    )
    ap.add_argument(
        "--action-map",
        type=Path,
        default=OUTDIR / "hll_uv_to_eft_map_uv_action_normalized_D21E21.csv",
    )
    ap.add_argument(
        "--constant-basis-summary",
        type=Path,
        default=OUTDIR / "hll_uv_operator_basis_summary.csv",
    )
    ap.add_argument(
        "--input-basis-summary",
        type=Path,
        default=OUTDIR / "hll_uv_operator_basis_summary_uv_input_tied_diagonly_D21E21.csv",
    )
    ap.add_argument(
        "--action-basis-summary",
        type=Path,
        default=OUTDIR / "hll_uv_operator_basis_summary_uv_action_normalized_D21E21.csv",
    )
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
        "C_uv_mumu",
        "C_ir_mumu",
        "abs_delta_C_match_mumu",
        "chi2_uv_rge",
    ]
    extra = [
        c
        for c in [
            "action_norm_diag",
            "action_norm_offdiag",
            "gap_cv",
            "gap_asym",
            "g_col_norm_cv",
            "c_tree_diag_cv",
            "pkin_entropy",
            "shell_spread",
            "coeff_cv",
            "offdiag_mix",
            "kappa_diag_eff",
            "kappa_offdiag_eff",
        ]
        if c in df.columns
    ]
    df = df[keep + extra]
    return df.rename(columns={c: f"{c}_{label}" for c in df.columns if c not in {"D", "eta"}})


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
    action_col = f"action_norm_diag_{label}"
    if action_col in df.columns:
        row["action_norm_diag_p50"] = float(df[action_col].quantile(0.5))
        row["action_norm_diag_p90"] = float(df[action_col].quantile(0.9))
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
        f"f_chi2_le_4_{a}": float((df[f"chi2_uv_rge_{a}"] <= 4.0).mean()),
        f"f_chi2_le_4_{b}": float((df[f"chi2_uv_rge_{b}"] <= 4.0).mean()),
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
        ("constant vs input_tied", arr_from_series((merged["mu_mumu_uv_rge_constant"] - merged["mu_mumu_uv_rge_input_tied"]).abs())),
        (
            "constant vs action_normalized",
            arr_from_series((merged["mu_mumu_uv_rge_constant"] - merged["mu_mumu_uv_rge_action_normalized"]).abs()),
        ),
        (
            "input_tied vs action_normalized",
            arr_from_series((merged["mu_mumu_uv_rge_input_tied"] - merged["mu_mumu_uv_rge_action_normalized"]).abs()),
        ),
        (
            "action_norm_diag",
            arr_from_series(
                merged["action_norm_diag_action_normalized"]
                if "action_norm_diag_action_normalized" in merged.columns
                else pd.Series(np.zeros(len(merged)))
            ),
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for ax, (title, arr) in zip(axes.ravel(), panels):
        p05, p95 = np.percentile(arr, [5.0, 95.0])
        if p95 <= p05:
            p05 = float(np.min(arr))
            p95 = float(np.max(arr)) + 1e-12
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=float(p05),
            vmax=float(p95),
        )
        ax.set_title(title)
        ax.set_xlabel("D")
        ax.set_ylabel("eta")
        fig.colorbar(im, ax=ax)
    fig.suptitle("UV matching mode comparator", fontsize=13)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    const = load_map(args.constant_map, "constant")
    inp = load_map(args.input_map, "input_tied")
    act = load_map(args.action_map, "action_normalized")
    merged = const.merge(inp, on=["D", "eta"]).merge(act, on=["D", "eta"])

    mode_summary = pd.DataFrame(
        [
            summarize_mode(merged, "constant", args.constant_basis_summary),
            summarize_mode(merged, "input_tied", args.input_basis_summary),
            summarize_mode(merged, "action_normalized", args.action_basis_summary),
        ]
    )
    pair_summary = pd.DataFrame(
        [
            summarize_pair(merged, "constant", "input_tied"),
            summarize_pair(merged, "constant", "action_normalized"),
            summarize_pair(merged, "input_tied", "action_normalized"),
        ]
    )

    out_mode = OUTDIR / f"hll_uv_action_normalized_mode_summary_{args.tag}.csv"
    out_pair = OUTDIR / f"hll_uv_action_normalized_pairwise_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_action_normalized_comparator_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_action_normalized_comparator_run_meta_{args.tag}.json"

    mode_summary.to_csv(out_mode, index=False)
    pair_summary.to_csv(out_pair, index=False)
    plot_pairwise(out_fig, merged)
    out_meta.write_text(
        json.dumps(
            {
                "constant_map": str(args.constant_map),
                "input_map": str(args.input_map),
                "action_map": str(args.action_map),
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

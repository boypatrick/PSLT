#!/usr/bin/env python3
"""
Phase-2 window audit for the input-tied diagonal threshold witness.

This script promotes the diag_scale scan from a raw comparator sweep to a
bounded physical window report. It combines:

  1. Map-level drift metrics versus the refreshed constant-threshold baseline.
  2. A robust small-deformation proxy using p90 of
     |Delta C_match(mumu)| / |C_tree(mumu)|.

It exports a compact summary table and a two-panel diagnostic figure and mirrors
the canonical outputs under paper/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "hll_uv_matching"
PAPER_DIR = ROOT / "paper"

LABEL_RE = re.compile(r"diag(?P<diag>[0-9]+(?:\.[0-9]+)?)_off(?P<off>[0-9]+(?:\.[0-9]+)?)$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the diag-scale physical window summary")
    ap.add_argument(
        "--scale-summary",
        type=Path,
        default=OUTDIR / "hll_uv_input_tied_vs_constant_refresh_scale_summary_D21E21.csv",
    )
    ap.add_argument(
        "--scan-dir",
        type=Path,
        default=OUTDIR,
    )
    ap.add_argument("--tag", type=str, default="D21E21")
    ap.add_argument("--skip-paper-copy", action="store_true")
    return ap.parse_args()


def parse_label(label: str) -> tuple[float, float]:
    m = LABEL_RE.match(str(label))
    if not m:
        raise ValueError(f"Unrecognized scale label: {label}")
    return float(m.group("diag")), float(m.group("off"))


def scale_token(x: float) -> str:
    if abs(x) < 1e-12:
        return "0"
    return f"{int(round(x * 100)):03d}"


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    scale_df = pd.read_csv(args.scale_summary).copy()
    scale_df[["diag_scale_value", "offdiag_scale_value"]] = scale_df["label"].apply(
        lambda s: pd.Series(parse_label(s))
    )

    rel_p90 = []
    rel_max = []
    delta_match_p90 = []
    delta_match_max = []
    for row in scale_df.itertuples(index=False):
        suffix = f"d{scale_token(row.diag_scale_value)}_o{scale_token(row.offdiag_scale_value)}"
        map_path = args.scan_dir / f"hll_uv_to_eft_map_scan_{suffix}.csv"
        if not map_path.exists():
            raise FileNotFoundError(map_path)
        df = pd.read_csv(map_path)
        rel = df["abs_delta_C_match_mumu"] / df["C_uv_mumu"].abs().clip(lower=1e-30)
        rel_p90.append(float(rel.quantile(0.9)))
        rel_max.append(float(rel.max()))
        delta_match_p90.append(float(df["abs_delta_C_match_mumu"].quantile(0.9)))
        delta_match_max.append(float(df["abs_delta_C_match_mumu"].max()))

    scale_df["rel_match_shift_diag_p90"] = rel_p90
    scale_df["rel_match_shift_diag_max"] = rel_max
    scale_df["abs_delta_C_match_mumu_p90"] = delta_match_p90
    scale_df["abs_delta_C_match_mumu_max"] = delta_match_max

    scale_df["pass_conservative_window"] = (
        (scale_df["offdiag_scale_value"] == 0.0)
        & (scale_df["acceptance_mismatch_fraction"] == 0.0)
        & (scale_df["max_abs_mapdiff"] <= 0.01)
        & (scale_df["rel_match_shift_diag_p90"] <= 0.002)
    )
    scale_df["pass_extended_window"] = (
        (scale_df["offdiag_scale_value"] == 0.0)
        & (scale_df["acceptance_mismatch_fraction"] == 0.0)
        & (scale_df["max_abs_mapdiff"] <= 0.015)
        & (scale_df["rel_match_shift_diag_p90"] <= 0.0025)
    )

    conservative = scale_df.loc[scale_df["pass_conservative_window"]].sort_values("diag_scale_value")
    extended = scale_df.loc[scale_df["pass_extended_window"]].sort_values("diag_scale_value")
    diag_only = scale_df.loc[scale_df["offdiag_scale_value"] == 0.0].sort_values("diag_scale_value")

    summary = pd.DataFrame(
        [
            {
                "n_scan_points": int(len(scale_df)),
                "canonical_diag_scale": 1.0,
                "canonical_offdiag_scale": 0.0,
                "offdiag_scale_numerically_inactive": bool(
                    np.allclose(
                        scale_df.loc[scale_df["label"] == "diag1.00_off0", "max_abs_mapdiff"].to_numpy(),
                        scale_df.loc[scale_df["label"] == "diag1.00_off2", "max_abs_mapdiff"].to_numpy(),
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
                "conservative_diag_scale_min": float(conservative["diag_scale_value"].min()),
                "conservative_diag_scale_max": float(conservative["diag_scale_value"].max()),
                "extended_diag_scale_min": float(extended["diag_scale_value"].min()),
                "extended_diag_scale_max": float(extended["diag_scale_value"].max()),
                "canonical_max_abs_mapdiff": float(
                    scale_df.loc[scale_df["label"] == "diag1.00_off0", "max_abs_mapdiff"].iloc[0]
                ),
                "canonical_p95_abs_mapdiff": float(
                    scale_df.loc[scale_df["label"] == "diag1.00_off0", "p95_abs_mapdiff"].iloc[0]
                ),
                "canonical_rel_match_shift_diag_p90": float(
                    scale_df.loc[scale_df["label"] == "diag1.00_off0", "rel_match_shift_diag_p90"].iloc[0]
                ),
                "canonical_kappa_diag_eff_p50": float(
                    scale_df.loc[scale_df["label"] == "diag1.00_off0", "kappa_diag_eff_p50"].iloc[0]
                ),
                "corr_diag_scale__max_abs_mapdiff": corr_safe(
                    diag_only["diag_scale_value"].to_numpy(),
                    diag_only["max_abs_mapdiff"].to_numpy(),
                ),
                "corr_diag_scale__rel_match_shift_diag_p90": corr_safe(
                    diag_only["diag_scale_value"].to_numpy(),
                    diag_only["rel_match_shift_diag_p90"].to_numpy(),
                ),
            }
        ]
    )

    out_summary = OUTDIR / f"hll_uv_input_tied_diag_window_summary_{args.tag}.csv"
    out_fig = OUTDIR / f"hll_uv_input_tied_diag_window_{args.tag}.png"
    out_meta = OUTDIR / f"hll_uv_input_tied_diag_window_run_meta_{args.tag}.json"

    scale_df.to_csv(OUTDIR / f"hll_uv_input_tied_diag_window_scan_{args.tag}.csv", index=False)
    summary.to_csv(out_summary, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    x = diag_only["diag_scale_value"].to_numpy()

    axes[0].plot(x, diag_only["max_abs_mapdiff"], marker="o", color="#1f77b4", label="max |Δμ|")
    axes[0].plot(x, diag_only["p95_abs_mapdiff"], marker="s", color="#2ca02c", label="p95 |Δμ|")
    axes[0].axhline(0.01, color="#d62728", ls="--", lw=1.2, label="conservative cap")
    axes[0].axhline(0.015, color="#ff7f0e", ls=":", lw=1.2, label="extended cap")
    axes[0].set_xlabel("diag_scale")
    axes[0].set_ylabel("map drift vs constant")
    axes[0].set_title("Map deformation")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        x,
        diag_only["rel_match_shift_diag_p90"],
        marker="o",
        color="#9467bd",
        label=r"$p90(|\Delta C_{\mu\mu}^{\rm match}|/|C_{\mu\mu}^{\rm tree}|)$",
    )
    axes[1].axhline(0.002, color="#d62728", ls="--", lw=1.2, label="conservative cap")
    axes[1].axhline(0.0025, color="#ff7f0e", ls=":", lw=1.2, label="extended cap")
    axes[1].set_xlabel("diag_scale")
    axes[1].set_ylabel("relative diagonal match shift")
    axes[1].set_title("Small-deformation witness")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Input-tied diagonal threshold window")
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    meta = {
        "scale_summary": str(args.scale_summary),
        "scan_dir": str(args.scan_dir),
        "tag": args.tag,
        "conservative_rule": {
            "offdiag_scale": 0.0,
            "acceptance_mismatch_fraction": 0.0,
            "max_abs_mapdiff_max": 0.01,
            "rel_match_shift_diag_p90_max": 0.002,
        },
        "extended_rule": {
            "offdiag_scale": 0.0,
            "acceptance_mismatch_fraction": 0.0,
            "max_abs_mapdiff_max": 0.015,
            "rel_match_shift_diag_p90_max": 0.0025,
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    if not args.skip_paper_copy:
        for src in [out_summary, out_fig, out_meta, OUTDIR / f"hll_uv_input_tied_diag_window_scan_{args.tag}.csv"]:
            (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(scale_df.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

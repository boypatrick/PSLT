#!/usr/bin/env python3
"""
Build a release-tuned visibility instability map.

The goal is to identify where the promoted profile-anchored runtime-direct
visibility branch still relies most strongly on profile anchoring. We compare
current canonical full-release maps against the promoted
`cell_direct_runtime_release_tuned` branch and combine:

  - pointwise |Delta mu_mumu|
  - relative |Delta mu_mumu|
  - acceptance-boundary proximity
  - local gradient / curvature of the release-tuned mu_mumu map

The D60 summary is augmented with the existing B-module attribution table so the
current worst point can be tied back to `B_N`, `y_raw`, `m2`, and width-ratio
inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
SIGDIR = ROOT / "output" / "hll_signal_strength"

SURFACES = {
    "D21E41": {
        "full_map": SIGDIR / "hll_signal_strength_map_full_direct_map_full_release_D21E41.csv",
        "rt_map": SIGDIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_tuned_D21E41.csv",
    },
    "D60E21": {
        "full_map": SIGDIR / "hll_signal_strength_map_full_direct_map_full_release_D60E21.csv",
        "rt_map": SIGDIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_tuned_D60E21.csv",
    },
}

B_DIAG = OUTDIR / "full_direct_b_module_diagnostics_D60E21.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build runtime-direct visibility instability maps")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--skip-paper-copy", action="store_true")
    return ap.parse_args()


def load_surface(tag: str, full_path: Path, rt_path: Path) -> pd.DataFrame:
    full_df = pd.read_csv(full_path).rename(
        columns={
            "mu_ee": "mu_ee_full",
            "mu_mumu": "mu_mumu_full",
            "mu_tautau": "mu_tautau_full",
            "chi2_mumu": "chi2_mumu_full",
        }
    )
    rt_df = pd.read_csv(rt_path).rename(
        columns={
            "mu_ee": "mu_ee_rt",
            "mu_mumu": "mu_mumu_rt",
            "mu_tautau": "mu_tautau_rt",
            "chi2_mumu": "chi2_mumu_rt",
        }
    )
    df = full_df.merge(rt_df, on=["D", "eta"], how="inner", validate="one_to_one")
    df["surface"] = tag
    df["abs_delta_mu_mumu"] = (df["mu_mumu_rt"] - df["mu_mumu_full"]).abs()
    df["rel_delta_mu_mumu"] = df["abs_delta_mu_mumu"] / df["mu_mumu_full"].abs().clip(lower=1e-12)
    df["acceptance_full"] = df["chi2_mumu_full"] <= 4.0
    df["acceptance_rt"] = df["chi2_mumu_rt"] <= 4.0
    df["acceptance_flip"] = (df["acceptance_full"] != df["acceptance_rt"]).astype(int)
    margin = np.minimum((df["chi2_mumu_full"] - 4.0).abs(), (df["chi2_mumu_rt"] - 4.0).abs())
    df["boundary_proximity"] = np.exp(-margin / 2.0)
    return df


def add_local_geometry_risk(df: pd.DataFrame) -> pd.DataFrame:
    d_vals = np.sort(df["D"].unique())
    eta_vals = np.sort(df["eta"].unique())
    piv = df.pivot(index="eta", columns="D", values="mu_mumu_rt").sort_index().sort_index(axis=1)
    z = np.log1p(piv.to_numpy(dtype=float))

    d_eta, d_d = np.gradient(z, eta_vals, d_vals, edge_order=2)
    grad = np.sqrt(d_eta ** 2 + d_d ** 2)
    d2_eta = np.gradient(d_eta, eta_vals, axis=0, edge_order=2)
    d2_d = np.gradient(d_d, d_vals, axis=1, edge_order=2)
    curv = np.abs(d2_eta) + np.abs(d2_d)

    geom = pd.DataFrame(
        {
            "eta": np.repeat(eta_vals, len(d_vals)),
            "D": np.tile(d_vals, len(eta_vals)),
            "grad_log_mu_rt": grad.reshape(-1),
            "curv_log_mu_rt": curv.reshape(-1),
        }
    )
    return df.merge(geom, on=["D", "eta"], how="left", validate="one_to_one")


def normalized(series: pd.Series) -> pd.Series:
    scale = float(series.quantile(0.95))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(series.max()) if float(series.max()) > 0 else 1.0
    return (series / scale).clip(lower=0.0, upper=3.0)


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    delta_n = normalized(df["abs_delta_mu_mumu"])
    grad_n = normalized(df["grad_log_mu_rt"])
    curv_n = normalized(df["curv_log_mu_rt"])
    df["risk_score"] = (
        0.50 * delta_n
        + 0.25 * grad_n
        + 0.15 * curv_n
        + 0.10 * df["boundary_proximity"].clip(0.0, 1.0)
        + 1.00 * df["acceptance_flip"]
    )
    cutoff = float(df["risk_score"].quantile(0.95))
    df["high_anchor_candidate"] = df["risk_score"] >= cutoff
    return df


def summarize_surface(df: pd.DataFrame) -> Dict[str, object]:
    top = df.sort_values("risk_score", ascending=False).iloc[0]
    hotspot_mask = df["high_anchor_candidate"]
    d_hot = np.sort(df.loc[hotspot_mask, "D"].unique())
    return {
        "surface": str(df["surface"].iloc[0]),
        "n_points": int(len(df)),
        "frac_acceptance_flip": float(df["acceptance_flip"].mean()),
        "max_abs_delta_mu_mumu": float(df["abs_delta_mu_mumu"].max()),
        "p95_abs_delta_mu_mumu": float(df["abs_delta_mu_mumu"].quantile(0.95)),
        "max_grad_log_mu_rt": float(df["grad_log_mu_rt"].max()),
        "p95_grad_log_mu_rt": float(df["grad_log_mu_rt"].quantile(0.95)),
        "max_curv_log_mu_rt": float(df["curv_log_mu_rt"].max()),
        "p95_curv_log_mu_rt": float(df["curv_log_mu_rt"].quantile(0.95)),
        "risk_score_p95": float(df["risk_score"].quantile(0.95)),
        "n_high_anchor_candidates": int(hotspot_mask.sum()),
        "high_anchor_D_min": float(d_hot.min()) if len(d_hot) else np.nan,
        "high_anchor_D_max": float(d_hot.max()) if len(d_hot) else np.nan,
        "high_anchor_D_count": int(len(d_hot)),
        "worst_D": float(top["D"]),
        "worst_eta": float(top["eta"]),
        "worst_abs_delta_mu_mumu": float(top["abs_delta_mu_mumu"]),
        "worst_boundary_proximity": float(top["boundary_proximity"]),
        "worst_risk_score": float(top["risk_score"]),
    }


def attach_b_diagnostics(hotspots: pd.DataFrame, bdiag: pd.DataFrame) -> pd.DataFrame:
    out = hotspots.copy()
    obj_cols = {"scenario_label", "point_kind"}
    for col in [
        "scenario_label", "point_kind", "B1_full", "B2_full", "B3_full", "B1_cmp", "B2_cmp", "B3_cmp",
        "yraw1_full", "yraw2_full", "yraw3_full", "yraw1_cmp", "yraw2_cmp", "yraw3_cmp",
        "m2_1_full", "m2_2_full", "m2_3_full", "m2_1_cmp", "m2_2_cmp", "m2_3_cmp",
        "width_ratio_full", "width_ratio_cmp",
    ]:
        out[col] = None if col in obj_cols else np.nan
    if bdiag.empty:
        return out
    for idx, row in out.iterrows():
        if row["surface"] != "D60E21":
            continue
        match = bdiag[(np.isclose(bdiag["D"], row["D"])) & (np.isclose(bdiag["eta"], row["eta"]))]
        if match.empty:
            continue
        m = match.iloc[0]
        for col in [c for c in out.columns if c in match.columns]:
            out.at[idx, col] = m[col]
    return out




def build_d_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (surface, d_val), grp in df.groupby(["surface", "D"], sort=True):
        mask = grp["high_anchor_candidate"]
        eta_hot = np.sort(grp.loc[mask, "eta"].to_numpy(dtype=float))
        rows.append({
            "surface": surface,
            "D": float(d_val),
            "mean_abs_delta_mu_mumu": float(grp["abs_delta_mu_mumu"].mean()),
            "max_abs_delta_mu_mumu": float(grp["abs_delta_mu_mumu"].max()),
            "mean_risk_score": float(grp["risk_score"].mean()),
            "max_risk_score": float(grp["risk_score"].max()),
            "n_high_anchor_eta": int(mask.sum()),
            "high_anchor_eta_min": float(eta_hot.min()) if len(eta_hot) else np.nan,
            "high_anchor_eta_max": float(eta_hot.max()) if len(eta_hot) else np.nan,
            "mean_boundary_proximity": float(grp["boundary_proximity"].mean()),
        })
    return pd.DataFrame(rows)


def plot_surfaces(df_all: pd.DataFrame, out_png: Path) -> None:
    surfaces = ["D21E41", "D60E21"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
    for r, surface in enumerate(surfaces):
        df = df_all[df_all["surface"] == surface].copy()
        d_vals = np.sort(df["D"].unique())
        eta_vals = np.sort(df["eta"].unique())
        X, Y = np.meshgrid(d_vals, eta_vals)
        for c, (col, title) in enumerate([
            ("abs_delta_mu_mumu", r"$|\Delta \mu_{\mu\mu}|$"),
            ("risk_score", "risk score"),
            ("boundary_proximity", "boundary proximity"),
        ]):
            Z = (
                df.pivot(index="eta", columns="D", values=col)
                .sort_index()
                .sort_index(axis=1)
                .to_numpy(dtype=float)
            )
            ax = axes[r, c]
            im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
            ax.set_title(f"{surface}: {title}")
            ax.set_xlabel("D")
            ax.set_ylabel(r"$\eta$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Release-tuned visibility instability map")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []
    summaries = []
    for tag, spec in SURFACES.items():
        df = load_surface(tag, spec["full_map"], spec["rt_map"])
        df = add_local_geometry_risk(df)
        df = add_risk_score(df)
        all_frames.append(df)
        summaries.append(summarize_surface(df))

    df_all = pd.concat(all_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    hotspots = (
        df_all.sort_values(["surface", "risk_score"], ascending=[True, False])
        .groupby("surface", group_keys=False)
        .head(args.top_k)
        .copy()
    )

    if B_DIAG.exists():
        bdiag = pd.read_csv(B_DIAG)
        hotspots = attach_b_diagnostics(hotspots, bdiag)
    else:
        bdiag = pd.DataFrame()

    d_profile = build_d_profile(df_all)

    out_map = OUTDIR / "runtime_direct_visibility_instability_map.csv"
    out_summary = OUTDIR / "runtime_direct_visibility_instability_summary.csv"
    out_hotspots = OUTDIR / "runtime_direct_visibility_instability_hotspots.csv"
    out_dprofile = OUTDIR / "runtime_direct_visibility_instability_D_profile.csv"
    out_png = OUTDIR / "runtime_direct_visibility_instability_map.png"
    out_meta = OUTDIR / "runtime_direct_visibility_instability_run_meta.json"

    df_all.to_csv(out_map, index=False)
    summary_df.to_csv(out_summary, index=False)
    hotspots.to_csv(out_hotspots, index=False)
    d_profile.to_csv(out_dprofile, index=False)
    plot_surfaces(df_all, out_png)

    meta = {
        "surfaces": {
            k: {"full_map": str(v["full_map"]), "rt_map": str(v["rt_map"])} for k, v in SURFACES.items()
        },
        "b_diag_csv": str(B_DIAG) if B_DIAG.exists() else None,
        "risk_score": {
            "weights": {
                "abs_delta_mu_mumu": 0.50,
                "grad_log_mu_rt": 0.25,
                "curv_log_mu_rt": 0.15,
                "boundary_proximity": 0.10,
                "acceptance_flip": 1.00,
            },
            "normalization": "per-surface p95 clip to [0,3]",
            "high_anchor_candidate_rule": "risk_score >= per-surface p95",
        },
        "top_k": int(args.top_k),
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    if not args.skip_paper_copy:
        for src in [out_map, out_summary, out_hotspots, out_dprofile, out_png, out_meta]:
            (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary_df.to_string(index=False))
    print(hotspots[[
        "surface", "D", "eta", "abs_delta_mu_mumu", "grad_log_mu_rt", "curv_log_mu_rt", "boundary_proximity", "risk_score"
    ]].to_string(index=False))
    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(d_profile.sort_values(["surface", "max_risk_score"], ascending=[True, False]).groupby("surface").head(6).to_string(index=False))
    print(f"[saved] {out_hotspots}")
    print(f"[saved] {out_dprofile}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()

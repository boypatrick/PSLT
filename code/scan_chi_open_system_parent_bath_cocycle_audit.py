#!/usr/bin/env python3
"""
Pairwise cocycle / flatness audit for the projected parent bath action.

This script turns the canonical parent-bath statement into a positive
integrability test. After removing the overall bath amplitude kappa_env, the
projected parent bath block should define an exact additive cocycle in the
canonical log coordinates:

  B_a(D) = gamma_a(D) / kappa_env(D)
  G_a(D_j,D_i) = log(B_a(D_j) / B_a(D_i))

with

  G_a(D_j,D_i)
    = log(K_sys,a(D_j) / K_sys,a(D_i))
    + log(K_spec,a(D_j) / K_spec,a(D_i)).

If this holds pairwise and is flat on triples, then the affine log-generator
class is supported by a positive cocycle statement rather than only by
excluding nearby warp families.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


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


def _geom_mean(v: np.ndarray) -> float:
    vv = np.maximum(np.asarray(v, dtype=float), 1e-300)
    return float(np.exp(np.mean(np.log(vv))))


def _pairwise_residual(
    obs: np.ndarray,
    sys: np.ndarray,
    spec: np.ndarray,
    idxs: List[int],
) -> pd.DataFrame:
    rows = []
    for i, j in itertools.combinations(idxs, 2):
        delta_obs = float(np.log(obs[j] / obs[i]))
        delta_sys = float(np.log(sys[j] / sys[i]))
        delta_spec = float(np.log(spec[j] / spec[i]))
        rows.append(
            {
                "i": i,
                "j": j,
                "delta_obs": delta_obs,
                "delta_sys": delta_sys,
                "delta_spec": delta_spec,
                "delta_recon": delta_sys + delta_spec,
                "cocycle_residual_abs": abs(delta_obs - delta_sys - delta_spec),
            }
        )
    return pd.DataFrame(rows)


def _triangle_defect(obs: np.ndarray, sys: np.ndarray, spec: np.ndarray, idxs: List[int]) -> pd.DataFrame:
    rows = []
    for i, j, k in itertools.combinations(idxs, 3):
        obs_ij = float(np.log(obs[j] / obs[i]))
        obs_jk = float(np.log(obs[k] / obs[j]))
        obs_ik = float(np.log(obs[k] / obs[i]))
        recon_ij = float(np.log(sys[j] / sys[i]) + np.log(spec[j] / spec[i]))
        recon_jk = float(np.log(sys[k] / sys[j]) + np.log(spec[k] / spec[j]))
        recon_ik = float(np.log(sys[k] / sys[i]) + np.log(spec[k] / spec[i]))
        rows.append(
            {
                "i": i,
                "j": j,
                "k": k,
                "obs_triangle_defect_abs": abs(obs_ij + obs_jk - obs_ik),
                "recon_triangle_defect_abs": abs(recon_ij + recon_jk - recon_ik),
                "mixed_triangle_defect_abs": abs((obs_ij - recon_ij) + (obs_jk - recon_jk) - (obs_ik - recon_ik)),
            }
        )
    return pd.DataFrame(rows)


def _heatmap_from_pairs(n: int, pair_df: pd.DataFrame, value_col: str) -> np.ndarray:
    mat = np.full((n, n), np.nan, dtype=float)
    for row in pair_df.itertuples(index=False):
        mat[row.i, row.j] = getattr(row, value_col)
        mat[row.j, row.i] = getattr(row, value_col)
    np.fill_diagonal(mat, 0.0)
    return mat


def main() -> None:
    ap = argparse.ArgumentParser(description="Pairwise cocycle audit for projected parent bath action.")
    ap.add_argument(
        "--factor-map",
        type=Path,
        default=OUTDIR / "chi_open_system_bath_factorization_map.csv",
    )
    ap.add_argument(
        "--calib-csv",
        type=Path,
        default=OUTDIR / "kappa_env_anchor_calibration.csv",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.factor_map.exists():
        raise FileNotFoundError(args.factor_map)
    if not args.calib_csv.exists():
        raise FileNotFoundError(args.calib_csv)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.factor_map).copy().sort_values("D").reset_index(drop=True)
    calib = pd.read_csv(args.calib_csv).iloc[0]
    if df.empty:
        raise RuntimeError(f"Empty factorization map: {args.factor_map}")

    required = [
        "D",
        "kappa_env",
        "bath_system_kernel_phi",
        "bath_system_kernel_mix",
        "bath_shape_phi",
        "bath_shape_mix",
        "gamma_phi_micro",
        "gamma_mix_micro",
        "bath_phi_identifiable",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    anchor_set = {
        round(float(v), 8)
        for v in str(calib["D_anchor_list"]).split(",")
        if str(v).strip()
    }
    if not anchor_set:
        raise RuntimeError("Calibration CSV has no anchor D list.")
    anchor_mask = np.asarray([round(float(d), 8) in anchor_set for d in df["D"]], dtype=bool)
    phi_identifiable = df["bath_phi_identifiable"].to_numpy(dtype=int) == 1

    kappa = np.maximum(df["kappa_env"].to_numpy(dtype=float), 1e-300)
    sys_phi = np.maximum(df["bath_system_kernel_phi"].to_numpy(dtype=float), 1e-300)
    sys_mix = np.maximum(df["bath_system_kernel_mix"].to_numpy(dtype=float), 1e-300)
    spec_phi = np.maximum(df["bath_shape_phi"].to_numpy(dtype=float), 1e-300)
    spec_mix = np.maximum(df["bath_shape_mix"].to_numpy(dtype=float), 1e-300)
    gamma_phi = np.maximum(df["gamma_phi_micro"].to_numpy(dtype=float), 1e-300)
    gamma_mix = np.maximum(df["gamma_mix_micro"].to_numpy(dtype=float), 1e-300)

    parent_phi = gamma_phi / kappa
    parent_mix = gamma_mix / kappa

    sys_phi_ref = _geom_mean(sys_phi[anchor_mask])
    sys_mix_ref = _geom_mean(sys_mix[anchor_mask])
    spec_phi_ref = _geom_mean(spec_phi[anchor_mask])
    spec_mix_ref = _geom_mean(spec_mix[anchor_mask])

    gen_phi_obs = np.log(parent_phi / (sys_phi_ref * spec_phi_ref))
    gen_mix_obs = np.log(parent_mix / (sys_mix_ref * spec_mix_ref))
    gen_phi_recon = np.log(sys_phi / sys_phi_ref) + np.log(spec_phi / spec_phi_ref)
    gen_mix_recon = np.log(sys_mix / sys_mix_ref) + np.log(spec_mix / spec_mix_ref)

    df["parent_bath_generator_phi_obs"] = gen_phi_obs
    df["parent_bath_generator_phi_recon"] = gen_phi_recon
    df["parent_bath_generator_mix_obs"] = gen_mix_obs
    df["parent_bath_generator_mix_recon"] = gen_mix_recon
    df["parent_bath_generator_phi_residual_abs"] = np.abs(gen_phi_obs - gen_phi_recon)
    df["parent_bath_generator_mix_residual_abs"] = np.abs(gen_mix_obs - gen_mix_recon)

    idxs_all = list(range(len(df)))
    idxs_phi = [i for i in idxs_all if phi_identifiable[i]]

    pair_phi = _pairwise_residual(parent_phi, sys_phi, spec_phi, idxs_phi) if idxs_phi else pd.DataFrame()
    pair_mix = _pairwise_residual(parent_mix, sys_mix, spec_mix, idxs_all)
    tri_phi = _triangle_defect(parent_phi, sys_phi, spec_phi, idxs_phi) if len(idxs_phi) >= 3 else pd.DataFrame()
    tri_mix = _triangle_defect(parent_mix, sys_mix, spec_mix, idxs_all) if len(idxs_all) >= 3 else pd.DataFrame()

    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""

    pair_df = pd.concat(
        [
            pair_phi.assign(channel="phi"),
            pair_mix.assign(channel="mix"),
        ],
        ignore_index=True,
    )
    tri_df = pd.concat(
        [
            tri_phi.assign(channel="phi"),
            tri_mix.assign(channel="mix"),
        ],
        ignore_index=True,
    )

    out_map = OUTDIR / f"chi_open_system_parent_bath_cocycle_map{suffix}.csv"
    out_pairs = OUTDIR / f"chi_open_system_parent_bath_cocycle_pair_scan{suffix}.csv"
    out_triples = OUTDIR / f"chi_open_system_parent_bath_cocycle_triangle_scan{suffix}.csv"
    out_summary = OUTDIR / f"chi_open_system_parent_bath_cocycle_summary{suffix}.csv"
    out_png = OUTDIR / f"chi_open_system_parent_bath_cocycle{suffix}.png"
    out_meta = OUTDIR / f"chi_open_system_parent_bath_cocycle_run_meta{suffix}.json"

    df.to_csv(out_map, index=False)
    pair_df.to_csv(out_pairs, index=False)
    tri_df.to_csv(out_triples, index=False)

    summary = {
        "factor_map_csv": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "n_points": int(len(df)),
        "n_anchor": int(np.sum(anchor_mask)),
        "n_phi_identifiable": int(np.sum(phi_identifiable)),
        "n_pair_phi_identifiable": int(len(pair_phi)),
        "n_pair_mix": int(len(pair_mix)),
        "n_triangle_phi_identifiable": int(len(tri_phi)),
        "n_triangle_mix": int(len(tri_mix)),
        "max_abs_local_generator_residual_phi_identifiable": float(df.loc[phi_identifiable, "parent_bath_generator_phi_residual_abs"].max())
        if np.any(phi_identifiable)
        else float("nan"),
        "max_abs_local_generator_residual_mix": float(df["parent_bath_generator_mix_residual_abs"].max()),
        "p95_abs_pairwise_cocycle_residual_phi_identifiable": q(pair_phi["cocycle_residual_abs"], 0.95)
        if not pair_phi.empty
        else float("nan"),
        "max_abs_pairwise_cocycle_residual_phi_identifiable": float(pair_phi["cocycle_residual_abs"].max())
        if not pair_phi.empty
        else float("nan"),
        "p95_abs_pairwise_cocycle_residual_mix": q(pair_mix["cocycle_residual_abs"], 0.95)
        if not pair_mix.empty
        else float("nan"),
        "max_abs_pairwise_cocycle_residual_mix": float(pair_mix["cocycle_residual_abs"].max())
        if not pair_mix.empty
        else float("nan"),
        "p95_abs_triangle_defect_phi_identifiable": q(tri_phi["mixed_triangle_defect_abs"], 0.95)
        if not tri_phi.empty
        else float("nan"),
        "max_abs_triangle_defect_phi_identifiable": float(tri_phi["mixed_triangle_defect_abs"].max())
        if not tri_phi.empty
        else float("nan"),
        "p95_abs_triangle_defect_mix": q(tri_mix["mixed_triangle_defect_abs"], 0.95)
        if not tri_mix.empty
        else float("nan"),
        "max_abs_triangle_defect_mix": float(tri_mix["mixed_triangle_defect_abs"].max())
        if not tri_mix.empty
        else float("nan"),
        "corr(generator_phi_obs, generator_phi_recon)_identifiable": corr(
            df.loc[phi_identifiable, "parent_bath_generator_phi_obs"],
            df.loc[phi_identifiable, "parent_bath_generator_phi_recon"],
        ),
        "corr(generator_mix_obs, generator_mix_recon)": corr(
            df["parent_bath_generator_mix_obs"],
            df["parent_bath_generator_mix_recon"],
        ),
    }
    summary["canonical_objective"] = (
        (summary["p95_abs_pairwise_cocycle_residual_phi_identifiable"] if np.isfinite(summary["p95_abs_pairwise_cocycle_residual_phi_identifiable"]) else 0.0)
        + (summary["p95_abs_pairwise_cocycle_residual_mix"] if np.isfinite(summary["p95_abs_pairwise_cocycle_residual_mix"]) else 0.0)
        + (summary["p95_abs_triangle_defect_phi_identifiable"] if np.isfinite(summary["p95_abs_triangle_defect_phi_identifiable"]) else 0.0)
        + (summary["p95_abs_triangle_defect_mix"] if np.isfinite(summary["p95_abs_triangle_defect_mix"]) else 0.0)
    )

    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(df["D"], df["parent_bath_generator_mix_obs"], marker="o", label="mix obs")
    ax.plot(df["D"], df["parent_bath_generator_mix_recon"], linestyle="--", label="mix recon")
    if np.any(phi_identifiable):
        phi_df = df.loc[phi_identifiable]
        ax.plot(phi_df["D"], phi_df["parent_bath_generator_phi_obs"], marker="s", label="phi obs (ident)")
        ax.plot(phi_df["D"], phi_df["parent_bath_generator_phi_recon"], linestyle=":", label="phi recon (ident)")
    ax.set_title("Local canonical generator")
    ax.set_xlabel("D")
    ax.set_ylabel("generator")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    mix_mat = _heatmap_from_pairs(len(df), pair_mix, "cocycle_residual_abs")
    im = ax.imshow(mix_mat, origin="lower", aspect="auto")
    ax.set_title("Pairwise cocycle residual (mix)")
    ax.set_xlabel("j index")
    ax.set_ylabel("i index")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    if not pair_phi.empty:
        phi_pos = np.where(phi_identifiable)[0]
        phi_mat = _heatmap_from_pairs(len(df), pair_phi, "cocycle_residual_abs")
        im = ax.imshow(phi_mat, origin="lower", aspect="auto")
        ax.set_title("Pairwise cocycle residual (phi identifiable)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, "No identifiable phi pairs", ha="center", va="center")
        ax.set_title("Pairwise cocycle residual (phi)")
    ax.set_xlabel("j index")
    ax.set_ylabel("i index")

    ax = axes[1, 1]
    mix_tri = tri_mix["mixed_triangle_defect_abs"].to_numpy(dtype=float) if not tri_mix.empty else np.asarray([], dtype=float)
    phi_tri = tri_phi["mixed_triangle_defect_abs"].to_numpy(dtype=float) if not tri_phi.empty else np.asarray([], dtype=float)
    bins = 25
    if mix_tri.size:
        ax.hist(mix_tri, bins=bins, alpha=0.7, label="mix")
    if phi_tri.size:
        ax.hist(phi_tri, bins=bins, alpha=0.7, label="phi identifiable")
    ax.set_title("Triangle flatness defect")
    ax.set_xlabel("absolute defect")
    ax.set_ylabel("count")
    if mix_tri.size or phi_tri.size:
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Projected parent bath cocycle / flatness audit", fontsize=13)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "factor_map": str(args.factor_map),
        "calibration_csv": str(args.calib_csv),
        "out_map": str(out_map),
        "out_pairs": str(out_pairs),
        "out_triples": str(out_triples),
        "out_summary": str(out_summary),
        "out_png": str(out_png),
        "tag": tag,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    for p in [out_map, out_pairs, out_triples, out_summary, out_png, out_meta]:
        (PAPER_DIR / p.name).write_bytes(p.read_bytes())

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

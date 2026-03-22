#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
WIDTH_PROFILE = OUTDIR / "model_chain_full_direct_width_profile_Dgrid21_Egrid5.csv"

D_GRID = np.linspace(4.0, 20.0, 21)
ETA_GRID = np.array([0.2, 0.6, 1.0, 2.0, 4.0], dtype=float)
REF_D = 9.6
REF_ETA = 1.0
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])
ALPHA_CENTERS = (0.80, 0.8125, 0.825)
TAPERS = (0.0, 0.01, 0.02, 0.03, 0.04)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localized D-taper scan for observable width-anchor family.")
    parser.add_argument(
        "--alpha-centers",
        type=str,
        default=",".join(str(x) for x in ALPHA_CENTERS),
        help="Comma-separated base alpha values.",
    )
    parser.add_argument(
        "--tapers",
        type=str,
        default=",".join(str(x) for x in TAPERS),
        help="Comma-separated taper amplitudes.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="model_chain_observable_width_anchor_taper_audit",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode="eft_wilson_uv_rge",
        d_min=float(D_GRID.min()),
        d_max=float(D_GRID.max()),
        d_num=int(len(D_GRID)),
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
        runtime_direct_force=False,
        runtime_direct_no_cache=False,
        runtime_direct_chi_rho_max=3.0,
        runtime_direct_chi_z_margin=6.0,
        runtime_direct_chi_n_mu=120,
        runtime_direct_chi_tol=1e-8,
        runtime_direct_chi_maxiter=30000,
        runtime_direct_chi_sigma=2.5,
    )


def _build_kin(alpha_center: float, taper: float):
    return make_baseline_kinetics(
        chain_mode="cell_direct_runtime_release_fullsumanchor",
        observable_width_anchor_blend_override=float(alpha_center),
        observable_width_anchor_blend_taper_override=float(taper),
        observable_width_anchor_blend_taper_center_D_override=6.0,
        observable_width_anchor_blend_taper_sigma_D_override=0.4,
        observable_width_anchor_csv_override=str(WIDTH_PROFILE),
        **_common_kwargs(),
    )


def _mu(kin, layer: int, D: float, eta: float) -> tuple[float, float]:
    mu = float(
        kin.hll_mu_pred(
            layer,
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            observable_mode="eft_wilson_uv_rge",
            N_max=N_MAX,
        )
    )
    chi2 = float(((mu - 1.4) / 0.4) ** 2) if layer == 2 else float("nan")
    return mu, chi2


def _build_detail(alpha_centers: tuple[float, ...], tapers: tuple[float, ...]) -> pd.DataFrame:
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **_common_kwargs())
    families = {
        (float(alpha_center), float(taper)): _build_kin(float(alpha_center), float(taper))
        for alpha_center in alpha_centers
        for taper in tapers
    }
    rows: list[dict[str, float]] = []
    for D in D_GRID:
        for eta in ETA_GRID:
            base_mu, base_chi2 = _mu(kin_full, 2, float(D), float(eta))
            for (alpha_center, taper), kin in families.items():
                mu, chi2 = _mu(kin, 2, float(D), float(eta))
                rows.append(
                    {
                        "alpha_center": float(alpha_center),
                        "taper": float(taper),
                        "D": float(D),
                        "eta": float(eta),
                        "mu": mu,
                        "base_mu": base_mu,
                        "delta_mu": float(mu - base_mu),
                        "delta_log_mu": float(np.log(max(mu, 1e-30) / max(base_mu, 1e-30))),
                        "acceptance_mismatch": float((base_chi2 <= 4.0) ^ (chi2 <= 4.0)),
                    }
                )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (alpha_center, taper), sub in detail.groupby(["alpha_center", "taper"], sort=True):
        slice_by_d = {
            float(d_val): float(np.max(np.abs(grp["delta_mu"])))
            for d_val, grp in sub.groupby("D", sort=True)
        }
        worst_idx = int(np.argmax(np.abs(sub["delta_mu"])))
        worst_row = sub.iloc[worst_idx]
        rows.append(
            {
                "alpha_center": float(alpha_center),
                "taper": float(taper),
                "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu"]))),
                "p95_abs_delta_log_mu": float(np.percentile(np.abs(sub["delta_log_mu"]), 95.0)),
                "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])),
                "D4p8_max_abs_delta_mu": float(slice_by_d.get(4.8, np.nan)),
                "D5p6_max_abs_delta_mu": float(slice_by_d.get(5.6, np.nan)),
                "D6p4_max_abs_delta_mu": float(slice_by_d.get(6.4, np.nan)),
                "D7p2_max_abs_delta_mu": float(slice_by_d.get(7.2, np.nan)),
                "worst_D": float(worst_row["D"]),
                "worst_eta": float(worst_row["eta"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["alpha_center", "taper"]).reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (alpha_center, taper, d_val), grp in detail.groupby(["alpha_center", "taper", "D"], sort=True):
        rows.append(
            {
                "alpha_center": float(alpha_center),
                "taper": float(taper),
                "D": float(d_val),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["alpha_center", "taper", "D"]).reset_index(drop=True)


def _plot_heatmaps(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("p95_abs_delta_mu", "p95 |Δμ_μμ|"),
        ("max_abs_delta_mu", "max |Δμ_μμ|"),
        ("acceptance_mismatch", "acceptance mismatch"),
    ]
    alpha_centers = np.array(sorted(summary["alpha_center"].unique()), dtype=float)
    tapers = np.array(sorted(summary["taper"].unique()), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (col, title) in zip(axes, metrics):
        pivot = (
            summary.pivot(index="alpha_center", columns="taper", values=col)
            .reindex(index=alpha_centers, columns=tapers)
            .astype(float)
        )
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("taper")
        ax.set_ylabel("alpha_center")
        ax.set_xticks(np.arange(len(tapers)))
        ax.set_xticklabels([f"{x:.3f}" for x in tapers], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(alpha_centers)))
        ax.set_yticklabels([f"{x:.3f}" for x in alpha_centers])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    args = _parse_args()
    alpha_centers = tuple(float(x) for x in str(args.alpha_centers).split(",") if str(x).strip() != "")
    tapers = tuple(float(x) for x in str(args.tapers).split(",") if str(x).strip() != "")
    if len(alpha_centers) == 0 or len(tapers) == 0:
        raise ValueError("At least one alpha_center and one taper are required.")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not WIDTH_PROFILE.exists():
        raise FileNotFoundError(f"Missing width anchor profile: {WIDTH_PROFILE}")

    detail = _build_detail(alpha_centers, tapers)
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / f"{args.tag}_detail.csv"
    summary_path = OUTDIR / f"{args.tag}_summary.csv"
    slices_path = OUTDIR / f"{args.tag}_slices.csv"
    png_path = OUTDIR / f"{args.tag}.png"
    meta_path = OUTDIR / f"{args.tag}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot_heatmaps(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "width_profile": str(WIDTH_PROFILE),
                "alpha_centers": [float(a) for a in alpha_centers],
                "tapers": [float(t) for t in tapers],
                "taper_center_D": 6.0,
                "taper_sigma_D": 0.4,
                "ref_D": REF_D,
                "ref_eta": REF_ETA,
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

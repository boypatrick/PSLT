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
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observable width-anchor family audit on reduced grid.")
    parser.add_argument(
        "--alphas",
        type=str,
        default=",".join(str(x) for x in ALPHAS),
        help="Comma-separated alpha values to scan.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="model_chain_observable_width_anchor_family_audit",
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


def _build_kin(alpha: float):
    return make_baseline_kinetics(
        chain_mode="cell_direct_runtime_release_fullsumanchor",
        observable_width_anchor_blend_override=float(alpha),
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


def _build_detail(alphas: tuple[float, ...]) -> pd.DataFrame:
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **_common_kwargs())
    families = {alpha: _build_kin(alpha) for alpha in alphas}
    rows: list[dict[str, float | str]] = []
    for D in D_GRID:
        for eta in ETA_GRID:
            base_mu, base_chi2 = _mu(kin_full, 2, float(D), float(eta))
            for alpha, kin in families.items():
                mu, chi2 = _mu(kin, 2, float(D), float(eta))
                rows.append(
                    {
                        "alpha": float(alpha),
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
    for alpha, sub in detail.groupby("alpha", sort=True):
        rows.append(
            {
                "alpha": float(alpha),
                "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu"]))),
                "p95_abs_delta_log_mu": float(np.percentile(np.abs(sub["delta_log_mu"]), 95.0)),
                "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])),
            }
        )
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (alpha, d_val), grp in detail.groupby(["alpha", "D"], sort=True):
        rows.append(
            {
                "alpha": float(alpha),
                "D": float(d_val),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["alpha", "D"]).reset_index(drop=True)


def _plot_slices(slices: pd.DataFrame, out_path: Path, alphas: tuple[float, ...]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    cmap = plt.get_cmap("viridis")
    for idx, alpha in enumerate(alphas):
        color = cmap(idx / max(len(alphas) - 1, 1))
        sub = slices[slices["alpha"] == alpha]
        label = f"alpha={alpha:.2f}"
        axes[0].plot(sub["D"], sub["slice_p95_abs_delta_mu"], marker="o", color=color, label=label)
        axes[1].plot(sub["D"], sub["slice_max_abs_delta_mu"], marker="o", color=color, label=label)
        axes[2].plot(sub["D"], sub["slice_acceptance_mismatch"], marker="o", color=color, label=label)
    axes[0].set_title("p95 |Δμ_μμ|")
    axes[1].set_title("max |Δμ_μμ|")
    axes[2].set_title("acceptance mismatch")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("D")
    axes[0].set_ylabel("value")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    args = _parse_args()
    alphas = tuple(float(x) for x in str(args.alphas).split(",") if str(x).strip() != "")
    if len(alphas) == 0:
        raise ValueError("At least one alpha is required.")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not WIDTH_PROFILE.exists():
        raise FileNotFoundError(f"Missing width anchor profile: {WIDTH_PROFILE}")

    detail = _build_detail(alphas)
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
    _plot_slices(slices, png_path, alphas)
    meta_path.write_text(
        json.dumps(
            {
                "width_profile": str(WIDTH_PROFILE),
                "alphas": [float(a) for a in alphas],
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

#!/usr/bin/env python3
from __future__ import annotations

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

D_GRID60 = np.linspace(4.0, 20.0, 60)
FOCUS_TARGETS = np.array([4.8, 5.63, 5.9, 6.17, 6.44, 6.71, 7.25], dtype=float)
FOCUS_D = np.array(
    [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - target)))]) for target in FOCUS_TARGETS],
    dtype=float,
)
ETA_GRID = np.array([0.2, 0.6, 1.0, 2.0, 4.0], dtype=float)
REF_D = float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - 9.6)))])
REF_ETA = 1.0
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])

ALPHA_CENTERS = (0.80, 0.825, 0.85)
TAPERS = (0.00, 0.01, 0.02)
TAPER_CENTERS = (5.9, 6.0)
TAPER_SIGMAS = (0.40,)


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
        d_min=4.0,
        d_max=20.0,
        d_num=60,
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


def _build_candidate(alpha_center: float, taper: float, taper_center: float, taper_sigma: float):
    return make_baseline_kinetics(
        chain_mode="cell_direct_runtime_release_fullwidthrefamp_pointamp",
        observable_width_anchor_blend_override=float(alpha_center),
        observable_width_anchor_blend_taper_override=float(taper),
        observable_width_anchor_blend_taper_center_D_override=float(taper_center),
        observable_width_anchor_blend_taper_sigma_D_override=float(taper_sigma),
        **_common_kwargs(),
    )


def _mu(kin, D: float, eta: float) -> tuple[float, float]:
    mu = float(
        kin.hll_mu_pred(
            2,
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    chi2 = float(((mu - 1.4) / 0.4) ** 2)
    return mu, chi2


def _build_detail() -> pd.DataFrame:
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **_common_kwargs())
    families = {
        (float(alpha), float(taper), float(center), float(sigma)): _build_candidate(float(alpha), float(taper), float(center), float(sigma))
        for alpha in ALPHA_CENTERS
        for taper in TAPERS
        for center in TAPER_CENTERS
        for sigma in TAPER_SIGMAS
    }
    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            base_mu, base_chi2 = _mu(kin_full, float(D), float(eta))
            for (alpha, taper, center, sigma), kin in families.items():
                mu, chi2 = _mu(kin, float(D), float(eta))
                rows.append(
                    {
                        "alpha_center": float(alpha),
                        "taper": float(taper),
                        "taper_center_D": float(center),
                        "taper_sigma_D": float(sigma),
                        "D": float(D),
                        "eta": float(eta),
                        "mu": float(mu),
                        "base_mu": float(base_mu),
                        "delta_mu": float(mu - base_mu),
                        "delta_log_mu": _safe_log_ratio(mu, base_mu),
                        "acceptance_mismatch": float((base_chi2 <= 4.0) ^ (chi2 <= 4.0)),
                    }
                )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    hotspot_mask = (detail["D"] >= 5.8) & (detail["D"] <= 6.2)
    guard_mask = ((detail["D"] <= 5.2) | (detail["D"] >= 6.35)) & (detail["D"] <= 7.3)
    rows: list[dict[str, float]] = []
    for (alpha, taper, center, sigma), sub in detail.groupby(
        ["alpha_center", "taper", "taper_center_D", "taper_sigma_D"], sort=True
    ):
        row: dict[str, float] = {
            "alpha_center": float(alpha),
            "taper": float(taper),
            "taper_center_D": float(center),
            "taper_sigma_D": float(sigma),
            "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu"]), 95.0)),
            "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu"]))),
            "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])),
            "hotspot_max_abs_delta_mu": float(np.max(np.abs(sub.loc[hotspot_mask.loc[sub.index], "delta_mu"]))),
            "guard_max_abs_delta_mu": float(np.max(np.abs(sub.loc[guard_mask.loc[sub.index], "delta_mu"]))),
        }
        for d_val, grp in sub.groupby("D", sort=True):
            key = f"D{float(d_val):.3f}".replace(".", "p")
            row[f"{key}_max_abs_delta_mu"] = float(np.max(np.abs(grp["delta_mu"])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["alpha_center", "taper", "taper_center_D", "taper_sigma_D"]).reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (alpha, taper, center, sigma, d_val), grp in detail.groupby(
        ["alpha_center", "taper", "taper_center_D", "taper_sigma_D", "D"], sort=True
    ):
        rows.append(
            {
                "alpha_center": float(alpha),
                "taper": float(taper),
                "taper_center_D": float(center),
                "taper_sigma_D": float(sigma),
                "D": float(d_val),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["alpha_center", "taper", "taper_center_D", "taper_sigma_D", "D"]
    ).reset_index(drop=True)


def _plot_heatmaps(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (metric, title) in zip(
        axes,
        [
            ("hotspot_max_abs_delta_mu", "hotspot max |Δμ_μμ|"),
            ("guard_max_abs_delta_mu", "guard max |Δμ_μμ|"),
            ("max_abs_delta_mu", "global max |Δμ_μμ|"),
        ],
    ):
        best_sigma = summary.groupby("taper_sigma_D", sort=True)[metric].min().sort_values().index[0]
        sub = summary[summary["taper_sigma_D"] == best_sigma].copy()
        pivot = (
            sub.pivot_table(
                index=["alpha_center", "taper"],
                columns="taper_center_D",
                values=metric,
                aggfunc="min",
            )
            .sort_index()
            .astype(float)
        )
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{title}\n(best sigma={best_sigma:.2f})")
        ax.set_xlabel("taper_center_D")
        ax.set_ylabel("(alpha_center, taper)")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"({a:.3f},{t:.3f})" for a, t in pivot.index], fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail = _build_detail()
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot_heatmaps(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "focus_D": [float(x) for x in FOCUS_D],
                "eta_grid": [float(x) for x in ETA_GRID],
                "alpha_centers": [float(x) for x in ALPHA_CENTERS],
                "tapers": [float(x) for x in TAPERS],
                "taper_centers": [float(x) for x in TAPER_CENTERS],
                "taper_sigmas": [float(x) for x in TAPER_SIGMAS],
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

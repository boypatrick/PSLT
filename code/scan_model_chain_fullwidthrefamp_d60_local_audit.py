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
WIDTH_PROFILE = OUTDIR / "model_chain_full_direct_width_profile_Dgrid21_Egrid5.csv"
REFAMP_PROFILE = OUTDIR / "model_chain_full_direct_ref_amp_profile.csv"

D_GRID60 = np.linspace(4.0, 20.0, 60)
FOCUS_TARGETS = np.array([4.8, 5.35, 5.63, 5.9, 6.17, 6.44, 6.71, 7.25, 8.07], dtype=float)
FOCUS_D = np.array(
    [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - target)))]) for target in FOCUS_TARGETS],
    dtype=float,
)
ETA_GRID = np.linspace(0.2, 4.0, 21)
REF_D = float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - 9.6)))])
REF_ETA = 1.0
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])

ALPHA_CENTER = 0.825
WIDTH_TAPER = 0.01
WIDTH_TAPER_CENTER_D = 6.0
WIDTH_TAPER_SIGMA_D = 0.4
REFAMP_PEAK = 1.0
CENTER_GRID = (5.8, 6.0, 6.2)
SIGMA_GRID = (0.10, 0.15, 0.20)


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


def _build_candidate(center_D: float, sigma_D: float):
    return make_baseline_kinetics(
        chain_mode="cell_direct_runtime_release_fullsumanchor",
        observable_width_anchor_blend_override=float(ALPHA_CENTER),
        observable_width_anchor_blend_taper_override=float(WIDTH_TAPER),
        observable_width_anchor_blend_taper_center_D_override=float(WIDTH_TAPER_CENTER_D),
        observable_width_anchor_blend_taper_sigma_D_override=float(WIDTH_TAPER_SIGMA_D),
        observable_width_anchor_csv_override=str(WIDTH_PROFILE),
        observable_ref_amp_anchor_peak_override=float(REFAMP_PEAK),
        observable_ref_amp_anchor_center_D_override=float(center_D),
        observable_ref_amp_anchor_sigma_D_override=float(sigma_D),
        observable_ref_amp_anchor_csv_override=str(REFAMP_PROFILE),
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


def _label_d(value: float) -> str:
    return f"D{value:.3f}".replace(".", "p")


def _build_detail() -> pd.DataFrame:
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **_common_kwargs())
    families = {
        (float(center), float(sigma)): _build_candidate(float(center), float(sigma))
        for center in CENTER_GRID
        for sigma in SIGMA_GRID
    }
    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            base_mu, base_chi2 = _mu(kin_full, float(D), float(eta))
            for (center_D, sigma_D), kin in families.items():
                mu, chi2 = _mu(kin, float(D), float(eta))
                rows.append(
                    {
                        "center_D": float(center_D),
                        "sigma_D": float(sigma_D),
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
    hotspot_mask = (detail["D"] >= 5.5) & (detail["D"] <= 6.8)
    guard_mask = ((detail["D"] <= 5.2) | (detail["D"] >= 7.0)) & (detail["D"] <= 8.1)
    rows: list[dict[str, float]] = []
    for (center_D, sigma_D), sub in detail.groupby(["center_D", "sigma_D"], sort=True):
        row: dict[str, float] = {
            "center_D": float(center_D),
            "sigma_D": float(sigma_D),
            "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu"]), 95.0)),
            "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu"]))),
            "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])),
            "hotspot_max_abs_delta_mu": float(np.max(np.abs(sub.loc[hotspot_mask.loc[sub.index], "delta_mu"]))),
            "guard_max_abs_delta_mu": float(np.max(np.abs(sub.loc[guard_mask.loc[sub.index], "delta_mu"]))),
        }
        for d_val, grp in sub.groupby("D", sort=True):
            row[f"{_label_d(float(d_val))}_max_abs_delta_mu"] = float(np.max(np.abs(grp["delta_mu"])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["center_D", "sigma_D"]).reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (center_D, sigma_D, d_val), grp in detail.groupby(["center_D", "sigma_D", "D"], sort=True):
        rows.append(
            {
                "center_D": float(center_D),
                "sigma_D": float(sigma_D),
                "D": float(d_val),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["center_D", "sigma_D", "D"]).reset_index(drop=True)


def _plot_heatmaps(summary: pd.DataFrame, out_path: Path) -> None:
    centers = np.array(sorted(summary["center_D"].unique()), dtype=float)
    sigmas = np.array(sorted(summary["sigma_D"].unique()), dtype=float)
    metrics = [
        ("hotspot_max_abs_delta_mu", "hotspot max |Δμ_μμ|"),
        ("guard_max_abs_delta_mu", "guard max |Δμ_μμ|"),
        ("max_abs_delta_mu", "global max |Δμ_μμ|"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (col, title) in zip(axes, metrics):
        pivot = (
            summary.pivot(index="center_D", columns="sigma_D", values=col)
            .reindex(index=centers, columns=sigmas)
            .astype(float)
        )
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("sigma_D")
        ax.set_ylabel("center_D")
        ax.set_xticks(np.arange(len(sigmas)))
        ax.set_xticklabels([f"{x:.2f}" for x in sigmas], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(centers)))
        ax.set_yticklabels([f"{x:.2f}" for x in centers])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not WIDTH_PROFILE.exists():
        raise FileNotFoundError(f"Missing width profile: {WIDTH_PROFILE}")
    if not REFAMP_PROFILE.exists():
        raise FileNotFoundError(f"Missing ref-amp profile: {REFAMP_PROFILE}")

    detail = _build_detail()
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_d60_local_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_d60_local_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_d60_local_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_d60_local_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_d60_local_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot_heatmaps(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "focus_D": [float(x) for x in FOCUS_D],
                "eta_grid": [float(x) for x in ETA_GRID],
                "alpha_center": ALPHA_CENTER,
                "width_taper": WIDTH_TAPER,
                "width_taper_center_D": WIDTH_TAPER_CENTER_D,
                "width_taper_sigma_D": WIDTH_TAPER_SIGMA_D,
                "refamp_peak": REFAMP_PEAK,
                "center_grid": [float(x) for x in CENTER_GRID],
                "sigma_grid": [float(x) for x in SIGMA_GRID],
                "width_profile": str(WIDTH_PROFILE),
                "refamp_profile": str(REFAMP_PROFILE),
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

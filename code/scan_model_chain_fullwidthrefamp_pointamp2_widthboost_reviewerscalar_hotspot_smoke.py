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
FOCUS_TARGETS = np.array([5.627, 5.898, 6.169, 6.441, 6.712], dtype=float)
FOCUS_D = np.array(
    [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - target)))]) for target in FOCUS_TARGETS],
    dtype=float,
)
ETA_GRID = np.linspace(0.2, 4.0, 21)
REF_D = float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - 9.966101694915254)))])
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])

MODES = [
    ("full_direct", "full_direct"),
    ("cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost", "fullwidthrefamp_pointamp2_widthboost"),
    (
        "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_reviewerscalar",
        "fullwidthrefamp_pointamp2_widthboost_reviewerscalar",
    ),
]


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


def _build_kinetics() -> dict[str, object]:
    common = _common_kwargs()
    return {mode_key: make_baseline_kinetics(chain_mode=chain_mode, **common) for chain_mode, mode_key in MODES}


def _mu(kin, D: float, eta: float) -> tuple[float, float]:
    mu = float(
        kin.hll_mu_pred(
            LAYER,
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


def _build_detail(kinetics: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            base_mu, base_chi2 = _mu(kinetics["full_direct"], float(D), float(eta))
            for mode_key, kin in kinetics.items():
                mu, chi2 = _mu(kin, float(D), float(eta))
                rows.append(
                    {
                        "mode": mode_key,
                        "D": float(D),
                        "eta": float(eta),
                        "mu": float(mu),
                        "base_mu": float(base_mu),
                        "delta_mu": float(mu - base_mu),
                        "delta_log_mu": float(np.log(max(mu, 1e-30) / max(base_mu, 1e-30))),
                        "acceptance_mismatch": float((base_chi2 <= 4.0) ^ (chi2 <= 4.0)),
                    }
                )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    sub = detail[detail["mode"] != "full_direct"].copy()
    for mode_key, grp in sub.groupby("mode", sort=True):
        by_d = {
            float(d_val): float(np.max(np.abs(dsub["delta_mu"])))
            for d_val, dsub in grp.groupby("D", sort=True)
        }
        rows.append(
            {
                "mode": mode_key,
                "p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
                "D5p627_max_abs_delta_mu": float(by_d.get(5.627118644067797, np.nan)),
                "D5p898_max_abs_delta_mu": float(by_d.get(5.898305084745763, np.nan)),
                "D6p169_max_abs_delta_mu": float(by_d.get(6.169491525423729, np.nan)),
                "D6p441_max_abs_delta_mu": float(by_d.get(6.4406779661016955, np.nan)),
                "D6p712_max_abs_delta_mu": float(by_d.get(6.711864406779661, np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("mode").reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    sub = detail[detail["mode"] != "full_direct"].copy()
    for (mode_key, d_val), grp in sub.groupby(["mode", "D"], sort=True):
        rows.append(
            {
                "mode": mode_key,
                "D": float(d_val),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "D"]).reset_index(drop=True)


def _plot_slices(slices: pd.DataFrame, out_path: Path) -> None:
    palette = {
        "fullwidthrefamp_pointamp2_widthboost": "#f28e2b",
        "fullwidthrefamp_pointamp2_widthboost_reviewerscalar": "#4e79a7",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True, sharex=True)
    for mode_key in [
        "fullwidthrefamp_pointamp2_widthboost",
        "fullwidthrefamp_pointamp2_widthboost_reviewerscalar",
    ]:
        sub = slices[slices["mode"] == mode_key]
        axes[0].plot(sub["D"], sub["slice_p95_abs_delta_mu"], marker="o", color=palette[mode_key], label=mode_key)
        axes[1].plot(sub["D"], sub["slice_max_abs_delta_mu"], marker="o", color=palette[mode_key], label=mode_key)
    axes[0].set_title("focused p95 |Δμ_μμ|")
    axes[1].set_title("focused max |Δμ_μμ|")
    for ax in axes:
        ax.set_xlabel("D")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("value")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    kinetics = _build_kinetics()
    detail = _build_detail(kinetics)
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_reviewerscalar_hotspot_smoke_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_reviewerscalar_hotspot_smoke_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_reviewerscalar_hotspot_smoke_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_reviewerscalar_hotspot_smoke.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_reviewerscalar_hotspot_smoke_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot_slices(slices, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "focus_D": [float(x) for x in FOCUS_D],
                "eta_grid": [float(x) for x in ETA_GRID],
                "ref_D": REF_D,
                "ref_eta": REF_ETA,
                "modes": [mode_key for _, mode_key in MODES],
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

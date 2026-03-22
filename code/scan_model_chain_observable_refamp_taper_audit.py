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

FOCUS_D = np.array([4.0, 4.8, 5.6, 6.4, 7.2, 8.0], dtype=float)
ETA_GRID = np.array([0.2, 0.6, 1.0, 2.0, 4.0], dtype=float)
REF_D = 9.6
REF_ETA = 1.0
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])

ALPHA_CENTER = 0.825
WIDTH_TAPER = 0.01
REFAMP_CENTER_D = 5.6
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIGMAS = (0.20, 0.30, 0.40, 0.60, 0.80)


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _geom_blend(cur: float, base: float, beta: float, floor: float = 1e-30) -> float:
    return float(
        np.exp(
            (1.0 - float(beta)) * np.log(max(float(cur), floor))
            + float(beta) * np.log(max(float(base), floor))
        )
    )


def _localized_beta(beta_peak: float, D: float, sigma: float) -> float:
    return float(beta_peak * np.exp(-0.5 * ((float(D) - REFAMP_CENTER_D) / float(sigma)) ** 2))


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
        d_min=4.0,
        d_max=20.0,
        d_num=21,
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


def _build_current_kin():
    return make_baseline_kinetics(
        chain_mode="cell_direct_runtime_release_fullsumanchor",
        observable_width_anchor_blend_override=float(ALPHA_CENTER),
        observable_width_anchor_blend_taper_override=float(WIDTH_TAPER),
        observable_width_anchor_blend_taper_center_D_override=6.0,
        observable_width_anchor_blend_taper_sigma_D_override=0.4,
        observable_width_anchor_csv_override=str(WIDTH_PROFILE),
        **_common_kwargs(),
    )


def _point_payload(kin, D: float, eta: float) -> dict[str, float]:
    amp = float(
        kin.hll_channel_amplitude(
            2,
            D,
            eta,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref = float(
        kin.hll_channel_amplitude(
            2,
            REF_D,
            REF_ETA,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ratio = float(amp / max(amp_ref, 1e-30))
    partial_ratio = float(amp_ratio * amp_ratio)
    width_ratio = float(
        kin.hll_total_width_ratio_uv_rge(
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            N_max=N_MAX,
        )
    )
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
    return {
        "amp": amp,
        "amp_ref": amp_ref,
        "amp_ratio": amp_ratio,
        "partial_ratio": partial_ratio,
        "width_ratio": width_ratio,
        "mu": mu,
        "chi2": chi2,
    }


def _build_detail() -> pd.DataFrame:
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **_common_kwargs())
    kin_cur = _build_current_kin()
    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            base = _point_payload(kin_full, float(D), float(eta))
            cur = _point_payload(kin_cur, float(D), float(eta))
            for sigma in SIGMAS:
                for beta_peak in BETAS:
                    beta_eff = _localized_beta(float(beta_peak), float(D), float(sigma))
                    amp_ref_cf = _geom_blend(cur["amp_ref"], base["amp_ref"], beta_eff)
                    partial_ratio_cf = float((cur["amp"] / max(amp_ref_cf, 1e-30)) ** 2)
                    mu_cf = float(partial_ratio_cf / max(cur["width_ratio"], 1e-30))
                    chi2_cf = float(((mu_cf - 1.4) / 0.4) ** 2)
                    rows.append(
                        {
                            "beta_peak": float(beta_peak),
                            "sigma_D": float(sigma),
                            "beta_eff": float(beta_eff),
                            "D": float(D),
                            "eta": float(eta),
                            "base_mu": float(base["mu"]),
                            "cur_mu": float(cur["mu"]),
                            "mu_cf": float(mu_cf),
                            "delta_mu_cf": float(mu_cf - base["mu"]),
                            "delta_log_mu_cf": _safe_log_ratio(mu_cf, base["mu"]),
                            "acceptance_mismatch_cf": float((base["chi2"] <= 4.0) ^ (chi2_cf <= 4.0)),
                        }
                    )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (beta_peak, sigma_D), sub in detail.groupby(["beta_peak", "sigma_D"], sort=True):
        by_d = {
            float(d_val): float(np.max(np.abs(grp["delta_mu_cf"])))
            for d_val, grp in sub.groupby("D", sort=True)
        }
        rows.append(
            {
                "beta_peak": float(beta_peak),
                "sigma_D": float(sigma_D),
                "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu_cf"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu_cf"]))),
                "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch_cf"])),
                "D4p0_max_abs_delta_mu": float(by_d.get(4.0, np.nan)),
                "D4p8_max_abs_delta_mu": float(by_d.get(4.8, np.nan)),
                "D5p6_max_abs_delta_mu": float(by_d.get(5.6, np.nan)),
                "D6p4_max_abs_delta_mu": float(by_d.get(6.4, np.nan)),
                "D7p2_max_abs_delta_mu": float(by_d.get(7.2, np.nan)),
                "D8p0_max_abs_delta_mu": float(by_d.get(8.0, np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values(["beta_peak", "sigma_D"]).reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (beta_peak, sigma_D, d_val), grp in detail.groupby(["beta_peak", "sigma_D", "D"], sort=True):
        rows.append(
            {
                "beta_peak": float(beta_peak),
                "sigma_D": float(sigma_D),
                "D": float(d_val),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu_cf"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu_cf"]))),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch_cf"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["beta_peak", "sigma_D", "D"]).reset_index(drop=True)


def _plot_heatmaps(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("D5p6_max_abs_delta_mu", "D=5.6 max |Δμ_μμ|"),
        ("D6p4_max_abs_delta_mu", "D=6.4 max |Δμ_μμ|"),
        ("max_abs_delta_mu", "global max |Δμ_μμ|"),
    ]
    betas = np.array(sorted(summary["beta_peak"].unique()), dtype=float)
    sigmas = np.array(sorted(summary["sigma_D"].unique()), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (col, title) in zip(axes, metrics):
        pivot = (
            summary.pivot(index="beta_peak", columns="sigma_D", values=col)
            .reindex(index=betas, columns=sigmas)
            .astype(float)
        )
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("sigma_D")
        ax.set_ylabel("beta_peak")
        ax.set_xticks(np.arange(len(sigmas)))
        ax.set_xticklabels([f"{x:.2f}" for x in sigmas], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(betas)))
        ax.set_yticklabels([f"{x:.2f}" for x in betas])
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
    detail = _build_detail()
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_observable_refamp_taper_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_observable_refamp_taper_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_observable_refamp_taper_audit_slices.csv"
    png_path = OUTDIR / "model_chain_observable_refamp_taper_audit.png"
    meta_path = OUTDIR / "model_chain_observable_refamp_taper_audit_run_meta.json"

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
                "refamp_center_D": REFAMP_CENTER_D,
                "beta_peaks": [float(x) for x in BETAS],
                "sigmas": [float(x) for x in SIGMAS],
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

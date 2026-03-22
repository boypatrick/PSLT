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
LAYER = 2
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])

BETA_PEAKS = (0.10, 0.20, 0.30, 0.40, 0.50)
CENTER_DS = (5.9, 6.0, 6.1)
SIGMA_DS = (0.10, 0.15, 0.20)


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


def _build_kinetics():
    common = _common_kwargs()
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **common)
    kin_cur = make_baseline_kinetics(chain_mode="cell_direct_runtime_release_fullwidthrefamp_pointamp", **common)
    return kin_full, kin_cur


def _beta_eff(D: float, peak: float, center: float, sigma: float) -> float:
    gauss = np.exp(-0.5 * ((float(D) - float(center)) / max(float(sigma), 1e-9)) ** 2)
    return float(np.clip(float(peak) * float(gauss), 0.0, 1.0))


def _point_payload(kin, D: float, eta: float) -> dict[str, float]:
    amp_raw = float(
        kin.hll_channel_amplitude(
            LAYER,
            D,
            eta,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref_raw = float(
        kin.hll_channel_amplitude(
            LAYER,
            REF_D,
            REF_ETA,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref = float(
        kin._blend_observable_ref_amp(
            amp_ref_raw,
            LAYER,
            OBSERVABLE_MODE,
            float(D),
        )
    )
    amp = float(
        kin._blend_observable_point_amp(
            amp_raw,
            LAYER,
            OBSERVABLE_MODE,
            float(D),
            float(eta),
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
    return {
        "amp_raw": amp_raw,
        "amp_ref_raw": amp_ref_raw,
        "amp": amp,
        "amp_ref": amp_ref,
        "partial_ratio": partial_ratio,
        "width_ratio": width_ratio,
        "mu": mu,
        "chi2": chi2,
    }


def _build_detail() -> pd.DataFrame:
    kin_full, kin_cur = _build_kinetics()
    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            base = _point_payload(kin_full, float(D), float(eta))
            cur = _point_payload(kin_cur, float(D), float(eta))
            for peak in BETA_PEAKS:
                for center in CENTER_DS:
                    for sigma in SIGMA_DS:
                        beta_eff = _beta_eff(float(D), float(peak), float(center), float(sigma))
                        width_boost = float(
                            np.exp(
                                (1.0 - beta_eff) * np.log(max(cur["width_ratio"], 1e-30))
                                + beta_eff * np.log(max(base["width_ratio"], 1e-30))
                            )
                        )
                        mu_boost = float(cur["partial_ratio"] / max(width_boost, 1e-30))
                        chi2_boost = float(((mu_boost - 1.4) / 0.4) ** 2)
                        rows.append(
                            {
                                "beta_peak": float(peak),
                                "center_D": float(center),
                                "sigma_D": float(sigma),
                                "D": float(D),
                                "eta": float(eta),
                                "beta_eff": beta_eff,
                                "base_mu": base["mu"],
                                "cur_mu": cur["mu"],
                                "delta_mu_current": float(cur["mu"] - base["mu"]),
                                "base_width_ratio": base["width_ratio"],
                                "cur_width_ratio": cur["width_ratio"],
                                "width_ratio_boost": width_boost,
                                "mu_boost": mu_boost,
                                "delta_mu_boost": float(mu_boost - base["mu"]),
                                "delta_log_width_current": _safe_log_ratio(cur["width_ratio"], base["width_ratio"]),
                                "delta_log_width_boost": _safe_log_ratio(width_boost, base["width_ratio"]),
                                "acceptance_mismatch_boost": float((base["chi2"] <= 4.0) ^ (chi2_boost <= 4.0)),
                            }
                        )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    hotspot_mask = (detail["D"] >= 5.85) & (detail["D"] <= 6.20)
    guard_mask = ((detail["D"] <= 5.2) | (detail["D"] >= 6.35)) & (detail["D"] <= 7.3)
    rows: list[dict[str, float]] = []
    for (peak, center, sigma), sub in detail.groupby(["beta_peak", "center_D", "sigma_D"], sort=True):
        row: dict[str, float] = {
            "beta_peak": float(peak),
            "center_D": float(center),
            "sigma_D": float(sigma),
            "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu_boost"]), 95.0)),
            "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu_boost"]))),
            "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch_boost"])),
            "hotspot_max_abs_delta_mu": float(np.max(np.abs(sub.loc[hotspot_mask.loc[sub.index], "delta_mu_boost"]))),
            "guard_max_abs_delta_mu": float(np.max(np.abs(sub.loc[guard_mask.loc[sub.index], "delta_mu_boost"]))),
            "max_beta_eff": float(np.max(sub["beta_eff"])),
        }
        for d_val, grp in sub.groupby("D", sort=True):
            key = f"D{float(d_val):.3f}".replace(".", "p")
            row[f"{key}_max_abs_delta_mu"] = float(np.max(np.abs(grp["delta_mu_boost"])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["beta_peak", "center_D", "sigma_D"]).reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (peak, center, sigma, d_val), grp in detail.groupby(["beta_peak", "center_D", "sigma_D", "D"], sort=True):
        rows.append(
            {
                "beta_peak": float(peak),
                "center_D": float(center),
                "sigma_D": float(sigma),
                "D": float(d_val),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu_boost"]))),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu_boost"]), 95.0)),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch_boost"])),
                "slice_max_beta_eff": float(np.max(grp["beta_eff"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["beta_peak", "center_D", "sigma_D", "D"]).reset_index(drop=True)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, metric in zip(
        axes,
        ["hotspot_max_abs_delta_mu", "guard_max_abs_delta_mu", "max_abs_delta_mu"],
    ):
        pivot = (
            summary[summary["sigma_D"] == 0.15]
            .pivot_table(index="beta_peak", columns="center_D", values=metric, aggfunc="min")
            .sort_index()
        )
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{metric}\n(sigma=0.15)")
        ax.set_xlabel("center_D")
        ax.set_ylabel("beta_peak")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{x:.2f}" for x in pivot.index])
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

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_boost_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_boost_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_boost_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_boost_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_d60_width_boost_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "focus_D": [float(x) for x in FOCUS_D],
                "eta_grid": [float(x) for x in ETA_GRID],
                "beta_peaks": [float(x) for x in BETA_PEAKS],
                "center_D": [float(x) for x in CENTER_DS],
                "sigma_D": [float(x) for x in SIGMA_DS],
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

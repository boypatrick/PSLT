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

FOCUS_D = np.array(
    [
        4.0,
        4.8,
        5.6,
        6.4,
        7.2,
        8.0,
    ],
    dtype=float,
)
ETA_GRID = np.linspace(0.2, 4.0, 41)
REF_D = 9.6
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])
CURRENT_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost"


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


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


def _build_kinetics():
    common = _common_kwargs()
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **common)
    kin_cur = make_baseline_kinetics(chain_mode=CURRENT_MODE, **common)
    return kin_full, kin_cur


def _effective_payload(kin, D: float, eta: float) -> dict[str, float]:
    raw_amp_ref = float(
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
            amp_ref=raw_amp_ref,
            layer_n=int(LAYER),
            observable_mode=str(OBSERVABLE_MODE),
            D=float(D),
        )
    )
    raw_amp = float(
        kin.hll_channel_amplitude(
            LAYER,
            D,
            eta,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp = float(
        kin._blend_observable_point_amp(
            amp=raw_amp,
            layer_n=int(LAYER),
            observable_mode=str(OBSERVABLE_MODE),
            D=float(D),
            eta=float(eta),
        )
    )
    ratio = float(amp / max(amp_ref, 1e-30))
    partial_ratio = float(ratio * ratio)
    partial_ratio = float(
        kin._blend_observable_partial_ratio(
            partial_ratio=partial_ratio,
            layer_n=int(LAYER),
            observable_mode=str(OBSERVABLE_MODE),
            D=float(D),
            eta=float(eta),
        )
    )
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
    mu = float(partial_ratio / max(width_ratio, 1e-30))
    chi2 = float(((mu - 1.4) / 0.4) ** 2)
    return {
        "raw_amp": raw_amp,
        "raw_amp_ref": raw_amp_ref,
        "amp": amp,
        "amp_ref": amp_ref,
        "partial_ratio": partial_ratio,
        "width_ratio": width_ratio,
        "mu": mu,
        "chi2": chi2,
        "width_alpha": float(kin._observable_width_anchor_effective_alpha(float(D))),
        "refamp_beta": float(kin._observable_ref_amp_anchor_effective_beta(float(D))),
        "pointamp_beta": float(kin._observable_point_amp_anchor_effective_beta(float(D))),
        "partial_beta": float(kin._observable_partial_anchor_effective_beta(float(D))),
    }


def _build_detail() -> pd.DataFrame:
    kin_full, kin_cur = _build_kinetics()
    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            base = _effective_payload(kin_full, float(D), float(eta))
            cur = _effective_payload(kin_cur, float(D), float(eta))
            delta_log_partial = _safe_log_ratio(cur["partial_ratio"], base["partial_ratio"])
            delta_log_width = _safe_log_ratio(cur["width_ratio"], base["width_ratio"])
            delta_log_mu = _safe_log_ratio(cur["mu"], base["mu"])
            delta_log_amp_point = _safe_log_ratio(cur["amp"], base["amp"])
            delta_log_amp_ref = _safe_log_ratio(cur["amp_ref"], base["amp_ref"])

            mu_width_anchor = float(cur["partial_ratio"] / max(base["width_ratio"], 1e-30))
            mu_partial_anchor = float(base["partial_ratio"] / max(cur["width_ratio"], 1e-30))
            partial_refamp_anchor = float((cur["amp"] / max(base["amp_ref"], 1e-30)) ** 2)
            partial_pointamp_anchor = float((base["amp"] / max(cur["amp_ref"], 1e-30)) ** 2)
            mu_refamp_anchor = float(partial_refamp_anchor / max(cur["width_ratio"], 1e-30))
            mu_pointamp_anchor = float(partial_pointamp_anchor / max(cur["width_ratio"], 1e-30))
            mu_width_refamp_anchor = float(partial_refamp_anchor / max(base["width_ratio"], 1e-30))
            mu_width_pointamp_anchor = float(partial_pointamp_anchor / max(base["width_ratio"], 1e-30))

            rows.append(
                {
                    "D": float(D),
                    "eta": float(eta),
                    "base_partial_ratio": base["partial_ratio"],
                    "base_width_ratio": base["width_ratio"],
                    "base_mu": base["mu"],
                    "base_chi2": base["chi2"],
                    "cur_partial_ratio": cur["partial_ratio"],
                    "cur_width_ratio": cur["width_ratio"],
                    "cur_mu": cur["mu"],
                    "cur_chi2": cur["chi2"],
                    "cur_width_alpha": cur["width_alpha"],
                    "cur_refamp_beta": cur["refamp_beta"],
                    "cur_pointamp_beta": cur["pointamp_beta"],
                    "cur_partial_beta": cur["partial_beta"],
                    "delta_mu": float(cur["mu"] - base["mu"]),
                    "delta_log_mu": delta_log_mu,
                    "delta_log_partial": delta_log_partial,
                    "delta_log_width": delta_log_width,
                    "delta_log_amp_point": delta_log_amp_point,
                    "delta_log_amp_ref": delta_log_amp_ref,
                    "path_closure_residual": float(delta_log_mu - (delta_log_partial - delta_log_width)),
                    "width_component": float(abs(delta_log_width)),
                    "partial_component": float(abs(delta_log_partial)),
                    "amp_point_component": float(abs(2.0 * delta_log_amp_point)),
                    "amp_ref_component": float(abs(-2.0 * delta_log_amp_ref)),
                    "mu_width_anchor": mu_width_anchor,
                    "mu_partial_anchor": mu_partial_anchor,
                    "mu_refamp_anchor": mu_refamp_anchor,
                    "mu_pointamp_anchor": mu_pointamp_anchor,
                    "mu_width_refamp_anchor": mu_width_refamp_anchor,
                    "mu_width_pointamp_anchor": mu_width_pointamp_anchor,
                    "delta_mu_width_anchor": float(mu_width_anchor - base["mu"]),
                    "delta_mu_partial_anchor": float(mu_partial_anchor - base["mu"]),
                    "delta_mu_refamp_anchor": float(mu_refamp_anchor - base["mu"]),
                    "delta_mu_pointamp_anchor": float(mu_pointamp_anchor - base["mu"]),
                    "delta_mu_width_refamp_anchor": float(mu_width_refamp_anchor - base["mu"]),
                    "delta_mu_width_pointamp_anchor": float(mu_width_pointamp_anchor - base["mu"]),
                    "acceptance_mismatch": float((base["chi2"] <= 4.0) ^ (cur["chi2"] <= 4.0)),
                }
            )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for d_val, grp in detail.groupby("D", sort=True):
        width_abs = np.abs(grp["delta_log_width"]).to_numpy(dtype=float)
        partial_abs = np.abs(grp["delta_log_partial"]).to_numpy(dtype=float)
        point_abs = np.abs(grp["amp_point_component"]).to_numpy(dtype=float)
        ref_abs = np.abs(grp["amp_ref_component"]).to_numpy(dtype=float)
        rows.append(
            {
                "D": float(d_val),
                "p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "p95_abs_delta_log_partial": float(np.percentile(partial_abs, 95.0)),
                "p95_abs_delta_log_width": float(np.percentile(width_abs, 95.0)),
                "p95_abs_amp_point_component": float(np.percentile(point_abs, 95.0)),
                "p95_abs_amp_ref_component": float(np.percentile(ref_abs, 95.0)),
                "dominant_fraction_width": float(np.mean(width_abs > partial_abs)),
                "dominant_fraction_partial": float(np.mean(partial_abs >= width_abs)),
                "dominant_fraction_pointamp": float(np.mean((point_abs >= ref_abs) & (partial_abs >= width_abs))),
                "dominant_fraction_refamp": float(np.mean((ref_abs > point_abs) & (partial_abs >= width_abs))),
                "mean_width_alpha": float(np.mean(grp["cur_width_alpha"])),
                "mean_refamp_beta": float(np.mean(grp["cur_refamp_beta"])),
                "mean_pointamp_beta": float(np.mean(grp["cur_pointamp_beta"])),
                "mean_partial_beta": float(np.mean(grp["cur_partial_beta"])),
                "acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
                "p95_abs_delta_mu_width_anchor": float(np.percentile(np.abs(grp["delta_mu_width_anchor"]), 95.0)),
                "p95_abs_delta_mu_partial_anchor": float(np.percentile(np.abs(grp["delta_mu_partial_anchor"]), 95.0)),
                "p95_abs_delta_mu_refamp_anchor": float(np.percentile(np.abs(grp["delta_mu_refamp_anchor"]), 95.0)),
                "p95_abs_delta_mu_pointamp_anchor": float(np.percentile(np.abs(grp["delta_mu_pointamp_anchor"]), 95.0)),
                "p95_abs_delta_mu_width_refamp_anchor": float(np.percentile(np.abs(grp["delta_mu_width_refamp_anchor"]), 95.0)),
                "p95_abs_delta_mu_width_pointamp_anchor": float(np.percentile(np.abs(grp["delta_mu_width_pointamp_anchor"]), 95.0)),
                "max_abs_path_closure_residual": float(np.max(np.abs(grp["path_closure_residual"]))),
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for d_val, grp in detail.groupby("D", sort=True):
        worst = grp.iloc[int(np.argmax(np.abs(grp["delta_mu"])))]
        rows.append(
            {
                "D": float(d_val),
                "worst_eta": float(worst["eta"]),
                "worst_delta_mu": float(worst["delta_mu"]),
                "worst_delta_log_partial": float(worst["delta_log_partial"]),
                "worst_delta_log_width": float(worst["delta_log_width"]),
                "worst_amp_point_component": float(worst["amp_point_component"]),
                "worst_amp_ref_component": float(worst["amp_ref_component"]),
                "worst_delta_mu_width_anchor": float(worst["delta_mu_width_anchor"]),
                "worst_delta_mu_partial_anchor": float(worst["delta_mu_partial_anchor"]),
                "worst_delta_mu_refamp_anchor": float(worst["delta_mu_refamp_anchor"]),
                "worst_delta_mu_pointamp_anchor": float(worst["delta_mu_pointamp_anchor"]),
                "worst_delta_mu_width_refamp_anchor": float(worst["delta_mu_width_refamp_anchor"]),
                "worst_delta_mu_width_pointamp_anchor": float(worst["delta_mu_width_pointamp_anchor"]),
                "worst_width_alpha": float(worst["cur_width_alpha"]),
                "worst_refamp_beta": float(worst["cur_refamp_beta"]),
                "worst_pointamp_beta": float(worst["cur_pointamp_beta"]),
                "worst_partial_beta": float(worst["cur_partial_beta"]),
                "worst_acceptance_mismatch": float(worst["acceptance_mismatch"]),
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    axes[0].plot(summary["D"], summary["p95_abs_delta_log_partial"], marker="o", label="partial")
    axes[0].plot(summary["D"], summary["p95_abs_delta_log_width"], marker="o", label="width")
    axes[0].plot(summary["D"], summary["p95_abs_amp_point_component"], marker="o", label="point amp")
    axes[0].plot(summary["D"], summary["p95_abs_amp_ref_component"], marker="o", label="ref amp")
    axes[0].set_title("D21 p95 path magnitudes")
    axes[0].set_xlabel("D")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(summary["D"], summary["p95_abs_delta_mu"], marker="o", label="current")
    axes[1].plot(summary["D"], summary["p95_abs_delta_mu_width_anchor"], marker="o", label="width anchor")
    axes[1].plot(summary["D"], summary["p95_abs_delta_mu_partial_anchor"], marker="o", label="partial anchor")
    axes[1].plot(summary["D"], summary["p95_abs_delta_mu_refamp_anchor"], marker="o", label="ref-amp anchor")
    axes[1].plot(summary["D"], summary["p95_abs_delta_mu_pointamp_anchor"], marker="o", label="point-amp anchor")
    axes[1].set_title("D21 p95 |Δμ| under anchors")
    axes[1].set_xlabel("D")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(summary["D"], summary["mean_width_alpha"], marker="o", label="width alpha")
    axes[2].plot(summary["D"], summary["mean_refamp_beta"], marker="o", label="ref beta")
    axes[2].plot(summary["D"], summary["mean_pointamp_beta"], marker="o", label="point beta")
    axes[2].plot(summary["D"], summary["mean_partial_beta"], marker="o", label="partial beta")
    axes[2].set_title("active family weights")
    axes[2].set_xlabel("D")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

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

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_d21_reviewer_hotspot_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_d21_reviewer_hotspot_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_d21_reviewer_hotspot_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_d21_reviewer_hotspot_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp2_widthboost_d21_reviewer_hotspot_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "current_mode": CURRENT_MODE,
                "focus_D": [float(x) for x in FOCUS_D],
                "eta_grid": [float(x) for x in ETA_GRID],
                "ref_D": REF_D,
                "ref_eta": REF_ETA,
                "layer": LAYER,
                "observable_mode": OBSERVABLE_MODE,
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

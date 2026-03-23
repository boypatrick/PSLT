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
        5.627118644067797,
        5.898305084745763,
        6.169491525423729,
        6.4406779661016955,
        6.711864406779661,
    ],
    dtype=float,
)
ETA_GRID = np.linspace(0.2, 4.0, 21)
REF_D = 9.694915254237289
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])

CENTER_D = 5.898305084745763
SIGMA_D = 0.20
WIDTH_PEAKS = (0.0, 0.25, 0.5, 0.75, 1.0)
REFAMP_PEAKS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _geom_blend(cur: float, base: float, beta: float, floor: float = 1e-30) -> float:
    return float(
        np.exp(
            (1.0 - float(beta)) * np.log(max(float(cur), floor))
            + float(beta) * np.log(max(float(base), floor))
        )
    )


def _localized_beta(peak: float, D: float) -> float:
    return float(np.clip(float(peak) * np.exp(-0.5 * ((float(D) - CENTER_D) / SIGMA_D) ** 2), 0.0, 1.0))


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
    kin_cur = make_baseline_kinetics(
        chain_mode="cell_direct_runtime_release_fullwidthrefamp_pointamp_widthboost",
        **common,
    )
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
    partial_ratio = float((amp / max(amp_ref, 1e-30)) ** 2)
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
            base = _effective_payload(kin_full, float(D), float(eta))
            cur = _effective_payload(kin_cur, float(D), float(eta))
            for width_peak in WIDTH_PEAKS:
                for ref_peak in REFAMP_PEAKS:
                    width_beta = _localized_beta(float(width_peak), float(D))
                    ref_beta = _localized_beta(float(ref_peak), float(D))
                    width_joint = _geom_blend(cur["width_ratio"], base["width_ratio"], width_beta)
                    amp_ref_joint = _geom_blend(cur["amp_ref"], base["amp_ref"], ref_beta)
                    partial_joint = float((cur["amp"] / max(amp_ref_joint, 1e-30)) ** 2)
                    mu_joint = float(partial_joint / max(width_joint, 1e-30))
                    chi2_joint = float(((mu_joint - 1.4) / 0.4) ** 2)
                    rows.append(
                        {
                            "width_peak": float(width_peak),
                            "refamp_peak": float(ref_peak),
                            "D": float(D),
                            "eta": float(eta),
                            "width_beta_eff": float(width_beta),
                            "refamp_beta_eff": float(ref_beta),
                            "base_mu": float(base["mu"]),
                            "cur_mu": float(cur["mu"]),
                            "mu_joint": float(mu_joint),
                            "delta_mu_joint": float(mu_joint - base["mu"]),
                            "delta_log_mu_joint": _safe_log_ratio(mu_joint, base["mu"]),
                            "delta_log_width_joint": _safe_log_ratio(width_joint, base["width_ratio"]),
                            "delta_log_refamp_joint": _safe_log_ratio(amp_ref_joint, base["amp_ref"]),
                            "acceptance_mismatch_joint": float((base["chi2"] <= 4.0) ^ (chi2_joint <= 4.0)),
                        }
                    )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (width_peak, refamp_peak), sub in detail.groupby(["width_peak", "refamp_peak"], sort=True):
        by_d = {
            float(d_val): float(np.max(np.abs(grp["delta_mu_joint"])))
            for d_val, grp in sub.groupby("D", sort=True)
        }
        rows.append(
            {
                "width_peak": float(width_peak),
                "refamp_peak": float(refamp_peak),
                "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu_joint"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu_joint"]))),
                "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch_joint"])),
                "D5p627_max_abs_delta_mu": float(by_d.get(5.627118644067797, np.nan)),
                "D5p898_max_abs_delta_mu": float(by_d.get(5.898305084745763, np.nan)),
                "D6p169_max_abs_delta_mu": float(by_d.get(6.169491525423729, np.nan)),
                "D6p441_max_abs_delta_mu": float(by_d.get(6.4406779661016955, np.nan)),
                "D6p712_max_abs_delta_mu": float(by_d.get(6.711864406779661, np.nan)),
                "joint_objective": float(
                    max(
                        by_d.get(5.898305084745763, 0.0),
                        by_d.get(6.169491525423729, 0.0),
                    )
                ),
                "guard_objective": float(
                    max(
                        by_d.get(5.627118644067797, 0.0),
                        by_d.get(6.4406779661016955, 0.0),
                        by_d.get(6.711864406779661, 0.0),
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["joint_objective", "guard_objective", "width_peak", "refamp_peak"]).reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for (width_peak, refamp_peak, d_val), grp in detail.groupby(["width_peak", "refamp_peak", "D"], sort=True):
        rows.append(
            {
                "width_peak": float(width_peak),
                "refamp_peak": float(refamp_peak),
                "D": float(d_val),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu_joint"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu_joint"]))),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch_joint"])),
                "slice_max_width_beta_eff": float(np.max(grp["width_beta_eff"])),
                "slice_max_refamp_beta_eff": float(np.max(grp["refamp_beta_eff"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["width_peak", "refamp_peak", "D"]).reset_index(drop=True)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    width_vals = np.array(sorted(summary["width_peak"].unique()), dtype=float)
    ref_vals = np.array(sorted(summary["refamp_peak"].unique()), dtype=float)
    metrics = [
        ("D5p898_max_abs_delta_mu", "D≈5.898 max |Δμ|"),
        ("joint_objective", "joint objective"),
        ("guard_objective", "guard objective"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (col, title) in zip(axes, metrics):
        pivot = (
            summary.pivot(index="width_peak", columns="refamp_peak", values=col)
            .reindex(index=width_vals, columns=ref_vals)
            .astype(float)
        )
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("refamp_peak")
        ax.set_ylabel("width_peak")
        ax.set_xticks(np.arange(len(ref_vals)))
        ax.set_xticklabels([f"{x:.2f}" for x in ref_vals])
        ax.set_yticks(np.arange(len(width_vals)))
        ax.set_yticklabels([f"{x:.2f}" for x in width_vals])
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

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d5898_joint_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d5898_joint_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d5898_joint_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d5898_joint_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_pointamp_widthboost_d5898_joint_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "focus_D": [float(x) for x in FOCUS_D],
                "eta_grid": [float(x) for x in ETA_GRID],
                "ref_D": REF_D,
                "ref_eta": REF_ETA,
                "layer": LAYER,
                "observable_mode": OBSERVABLE_MODE,
                "center_D": CENTER_D,
                "sigma_D": SIGMA_D,
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

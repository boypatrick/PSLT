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
FOCUS_TARGETS = np.array([5.9, 6.17, 6.44], dtype=float)
FOCUS_D = np.array(
    [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - target)))]) for target in FOCUS_TARGETS],
    dtype=float,
)
ETA_GRID = np.linspace(0.2, 4.0, 21)
REF_D = float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - 9.6)))])
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])


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
    kin_cur = make_baseline_kinetics(chain_mode="cell_direct_runtime_release_fullwidthrefamp", **common)
    return kin_full, kin_cur


def _point_payload(kin, D: float, eta: float) -> dict[str, float]:
    amp = float(
        kin.hll_channel_amplitude(
            LAYER,
            D,
            eta,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref = float(
        kin.hll_channel_amplitude(
            LAYER,
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
        "amp": amp,
        "amp_ref": amp_ref,
        "amp_ratio": amp_ratio,
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
            delta_log_amp_point = _safe_log_ratio(cur["amp"], base["amp"])
            delta_log_amp_ref = _safe_log_ratio(cur["amp_ref"], base["amp_ref"])
            delta_log_partial = _safe_log_ratio(cur["partial_ratio"], base["partial_ratio"])
            delta_log_width = _safe_log_ratio(cur["width_ratio"], base["width_ratio"])
            delta_log_mu = _safe_log_ratio(cur["mu"], base["mu"])
            mu_width_anchor = float(cur["partial_ratio"] / max(base["width_ratio"], 1e-30))
            mu_partial_anchor = float(base["partial_ratio"] / max(cur["width_ratio"], 1e-30))
            partial_refamp_anchor = float((cur["amp"] / max(base["amp_ref"], 1e-30)) ** 2)
            partial_pointamp_anchor = float((base["amp"] / max(cur["amp_ref"], 1e-30)) ** 2)
            mu_refamp_anchor = float(partial_refamp_anchor / max(cur["width_ratio"], 1e-30))
            mu_pointamp_anchor = float(partial_pointamp_anchor / max(cur["width_ratio"], 1e-30))
            mu_width_refamp_anchor = float(partial_refamp_anchor / max(base["width_ratio"], 1e-30))
            rows.append(
                {
                    "D": float(D),
                    "eta": float(eta),
                    "base_amp": base["amp"],
                    "base_amp_ref": base["amp_ref"],
                    "base_partial_ratio": base["partial_ratio"],
                    "base_width_ratio": base["width_ratio"],
                    "base_mu": base["mu"],
                    "base_chi2": base["chi2"],
                    "cur_amp": cur["amp"],
                    "cur_amp_ref": cur["amp_ref"],
                    "cur_partial_ratio": cur["partial_ratio"],
                    "cur_width_ratio": cur["width_ratio"],
                    "cur_mu": cur["mu"],
                    "cur_chi2": cur["chi2"],
                    "delta_mu": float(cur["mu"] - base["mu"]),
                    "delta_log_mu": delta_log_mu,
                    "delta_log_amp_point": delta_log_amp_point,
                    "delta_log_amp_ref": delta_log_amp_ref,
                    "delta_log_partial": delta_log_partial,
                    "delta_log_width": delta_log_width,
                    "path_closure_residual": float(delta_log_mu - (delta_log_partial - delta_log_width)),
                    "amp_point_component": float(abs(2.0 * delta_log_amp_point)),
                    "amp_ref_component": float(abs(-2.0 * delta_log_amp_ref)),
                    "partial_component": float(abs(delta_log_partial)),
                    "width_component": float(abs(delta_log_width)),
                    "mu_width_anchor": mu_width_anchor,
                    "mu_partial_anchor": mu_partial_anchor,
                    "mu_refamp_anchor": mu_refamp_anchor,
                    "mu_pointamp_anchor": mu_pointamp_anchor,
                    "mu_width_refamp_anchor": mu_width_refamp_anchor,
                    "delta_mu_width_anchor": float(mu_width_anchor - base["mu"]),
                    "delta_mu_partial_anchor": float(mu_partial_anchor - base["mu"]),
                    "delta_mu_refamp_anchor": float(mu_refamp_anchor - base["mu"]),
                    "delta_mu_pointamp_anchor": float(mu_pointamp_anchor - base["mu"]),
                    "delta_mu_width_refamp_anchor": float(mu_width_refamp_anchor - base["mu"]),
                    "acceptance_mismatch": float((base["chi2"] <= 4.0) ^ (cur["chi2"] <= 4.0)),
                    "acceptance_mismatch_width_anchor": float((base["chi2"] <= 4.0) ^ (((mu_width_anchor - 1.4) / 0.4) ** 2 <= 4.0)),
                    "acceptance_mismatch_partial_anchor": float((base["chi2"] <= 4.0) ^ (((mu_partial_anchor - 1.4) / 0.4) ** 2 <= 4.0)),
                    "acceptance_mismatch_refamp_anchor": float((base["chi2"] <= 4.0) ^ (((mu_refamp_anchor - 1.4) / 0.4) ** 2 <= 4.0)),
                    "acceptance_mismatch_width_refamp_anchor": float((base["chi2"] <= 4.0) ^ (((mu_width_refamp_anchor - 1.4) / 0.4) ** 2 <= 4.0)),
                }
            )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for d_val, grp in detail.groupby("D", sort=True):
        rows.append(
            {
                "D": float(d_val),
                "p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "p95_abs_delta_log_partial": float(np.percentile(np.abs(grp["delta_log_partial"]), 95.0)),
                "p95_abs_delta_log_width": float(np.percentile(np.abs(grp["delta_log_width"]), 95.0)),
                "p95_abs_amp_point_component": float(np.percentile(np.abs(grp["amp_point_component"]), 95.0)),
                "p95_abs_amp_ref_component": float(np.percentile(np.abs(grp["amp_ref_component"]), 95.0)),
                "dominant_fraction_width": float(np.mean(np.abs(grp["delta_log_width"]) > np.abs(grp["delta_log_partial"]))),
                "dominant_fraction_partial": float(np.mean(np.abs(grp["delta_log_partial"]) > np.abs(grp["delta_log_width"]))),
                "dominant_fraction_refamp": float(
                    np.mean(
                        (np.abs(grp["amp_ref_component"]) > np.abs(grp["amp_point_component"]))
                        & (np.abs(grp["delta_log_partial"]) >= np.abs(grp["delta_log_width"]))
                    )
                ),
                "dominant_fraction_pointamp": float(
                    np.mean(
                        (np.abs(grp["amp_point_component"]) >= np.abs(grp["amp_ref_component"]))
                        & (np.abs(grp["delta_log_partial"]) >= np.abs(grp["delta_log_width"]))
                    )
                ),
                "acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
                "p95_abs_delta_mu_width_anchor": float(np.percentile(np.abs(grp["delta_mu_width_anchor"]), 95.0)),
                "p95_abs_delta_mu_partial_anchor": float(np.percentile(np.abs(grp["delta_mu_partial_anchor"]), 95.0)),
                "p95_abs_delta_mu_refamp_anchor": float(np.percentile(np.abs(grp["delta_mu_refamp_anchor"]), 95.0)),
                "p95_abs_delta_mu_pointamp_anchor": float(np.percentile(np.abs(grp["delta_mu_pointamp_anchor"]), 95.0)),
                "p95_abs_delta_mu_width_refamp_anchor": float(np.percentile(np.abs(grp["delta_mu_width_refamp_anchor"]), 95.0)),
                "max_abs_delta_mu_width_anchor": float(np.max(np.abs(grp["delta_mu_width_anchor"]))),
                "max_abs_delta_mu_partial_anchor": float(np.max(np.abs(grp["delta_mu_partial_anchor"]))),
                "max_abs_delta_mu_refamp_anchor": float(np.max(np.abs(grp["delta_mu_refamp_anchor"]))),
                "max_abs_delta_mu_pointamp_anchor": float(np.max(np.abs(grp["delta_mu_pointamp_anchor"]))),
                "max_abs_delta_mu_width_refamp_anchor": float(np.max(np.abs(grp["delta_mu_width_refamp_anchor"]))),
                "acceptance_mismatch_width_anchor": float(np.mean(grp["acceptance_mismatch_width_anchor"])),
                "acceptance_mismatch_partial_anchor": float(np.mean(grp["acceptance_mismatch_partial_anchor"])),
                "acceptance_mismatch_refamp_anchor": float(np.mean(grp["acceptance_mismatch_refamp_anchor"])),
                "acceptance_mismatch_width_refamp_anchor": float(np.mean(grp["acceptance_mismatch_width_refamp_anchor"])),
                "p95_abs_path_closure_residual": float(np.percentile(np.abs(grp["path_closure_residual"]), 95.0)),
                "max_abs_path_closure_residual": float(np.max(np.abs(grp["path_closure_residual"]))),
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for d_val, grp in detail.groupby("D", sort=True):
        idx = int(np.argmax(np.abs(grp["delta_mu"])))
        worst = grp.iloc[idx]
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
                "worst_delta_mu_refamp_anchor": float(worst["delta_mu_refamp_anchor"]),
                "worst_delta_mu_pointamp_anchor": float(worst["delta_mu_pointamp_anchor"]),
                "worst_delta_mu_width_refamp_anchor": float(worst["delta_mu_width_refamp_anchor"]),
                "worst_acceptance_mismatch": float(worst["acceptance_mismatch"]),
            }
        )
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    axes[0].plot(summary["D"], summary["p95_abs_delta_log_partial"], marker="o", label="partial")
    axes[0].plot(summary["D"], summary["p95_abs_delta_log_width"], marker="o", label="width")
    axes[0].plot(summary["D"], summary["p95_abs_amp_point_component"], marker="o", label="point amp")
    axes[0].plot(summary["D"], summary["p95_abs_amp_ref_component"], marker="o", label="ref amp")
    axes[0].set_title("p95 path magnitudes")
    axes[0].set_xlabel("D")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(summary["D"], summary["p95_abs_delta_mu"], marker="o", label="current")
    axes[1].plot(summary["D"], summary["p95_abs_delta_mu_width_anchor"], marker="o", label="width anchor")
    axes[1].plot(summary["D"], summary["p95_abs_delta_mu_refamp_anchor"], marker="o", label="ref-amp anchor")
    axes[1].plot(summary["D"], summary["p95_abs_delta_mu_width_refamp_anchor"], marker="o", label="width+ref-amp")
    axes[1].set_title("p95 |Δμ| under anchors")
    axes[1].set_xlabel("D")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    width_dom = summary["dominant_fraction_width"].to_numpy(dtype=float)
    partial_dom = summary["dominant_fraction_partial"].to_numpy(dtype=float)
    ref_dom = summary["dominant_fraction_refamp"].to_numpy(dtype=float)
    point_dom = summary["dominant_fraction_pointamp"].to_numpy(dtype=float)
    x = np.arange(len(summary))
    axes[2].bar(x, width_dom, label="width")
    axes[2].bar(x, partial_dom, bottom=width_dom * 0.0, alpha=0.35, label="partial")
    axes[2].plot(x, ref_dom, marker="o", label="ref-amp dom", color="#e15759")
    axes[2].plot(x, point_dom, marker="o", label="point-amp dom", color="#59a14f")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"{d:.3f}" for d in summary["D"]], rotation=45, ha="right")
    axes[2].set_title("dominance fractions")
    axes[2].set_xlabel("D")
    axes[2].set_ylim(0.0, 1.05)
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

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_d60_hotspot_path_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_d60_hotspot_path_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_d60_hotspot_path_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_d60_hotspot_path_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_d60_hotspot_path_audit_run_meta.json"

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
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

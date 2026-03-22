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

D_GRID = np.linspace(4.0, 20.0, 21)
ETA_GRID = np.array([0.2, 0.6, 1.0, 2.0, 4.0], dtype=float)
LAYERS = (1, 2, 3)
REF_D = 9.6
REF_ETA = 1.0
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])

MODES = [
    ("full_direct", "full_direct"),
    ("cell_direct_runtime", "cell_direct_runtime"),
    ("cell_direct_runtime_release_tuned", "release_tuned"),
    ("cell_direct_runtime_release_sumanchor", "release_sumanchor"),
    ("cell_direct_runtime_release_eftsumanchor", "release_eftsumanchor"),
    ("cell_direct_runtime_release_fullsumanchor", "release_fullsumanchor"),
    ("cell_direct_runtime_release_fullwidthrefamp", "release_fullwidthrefamp"),
    ("cell_direct_runtime_release_tailm2", "tailm2_detlin"),
]
COMPARE_MODES = [mode_key for _, mode_key in MODES if mode_key != "full_direct"]


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
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


def _build_kinetics() -> dict[str, object]:
    common = _common_kwargs()
    kinetics: dict[str, object] = {}
    for chain_mode, mode_key in MODES:
        kinetics[mode_key] = make_baseline_kinetics(chain_mode=chain_mode, **common)
    return kinetics


def _width_ratio(kin, D: float, eta: float) -> float:
    return float(
        kin.hll_total_width_ratio_uv_rge(
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            N_max=N_MAX,
        )
    )


def _point_payload(kin, layer: int, D: float, eta: float) -> dict[str, float]:
    amp = float(
        kin.hll_channel_amplitude(
            layer,
            D,
            eta,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref = float(
        kin.hll_channel_amplitude(
            layer,
            REF_D,
            REF_ETA,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ratio = float(amp / max(amp_ref, 1e-30))
    partial_ratio = float(amp_ratio * amp_ratio)
    width_ratio = _width_ratio(kin, D, eta)
    mu = float(
        kin.hll_mu_pred(
            layer,
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    mu_rebuild = float(partial_ratio / max(width_ratio, 1e-30))
    chi2 = float(((mu - 1.4) / 0.4) ** 2) if layer == 2 else float("nan")
    return {
        "amp": amp,
        "amp_ref": amp_ref,
        "amp_ratio": amp_ratio,
        "partial_ratio": partial_ratio,
        "width_ratio": width_ratio,
        "mu": mu,
        "mu_rebuild": mu_rebuild,
        "mu_identity_residual": float(mu - mu_rebuild),
        "chi2": chi2,
    }


def _build_detail(kinetics: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for D in D_GRID:
        for eta in ETA_GRID:
            point: dict[int, dict[str, dict[str, float]]] = {}
            for layer in LAYERS:
                point[layer] = {}
                for mode_key, kin in kinetics.items():
                    point[layer][mode_key] = _point_payload(kin, layer, float(D), float(eta))
            for layer in LAYERS:
                base = point[layer]["full_direct"]
                for mode_key in COMPARE_MODES:
                    cur = point[layer][mode_key]
                    delta_log_amp_point = _safe_log_ratio(cur["amp"], base["amp"])
                    delta_log_amp_ref = _safe_log_ratio(cur["amp_ref"], base["amp_ref"])
                    delta_log_amp_ratio = _safe_log_ratio(cur["amp_ratio"], base["amp_ratio"])
                    delta_log_partial = _safe_log_ratio(cur["partial_ratio"], base["partial_ratio"])
                    delta_log_width = _safe_log_ratio(cur["width_ratio"], base["width_ratio"])
                    delta_log_mu = _safe_log_ratio(cur["mu"], base["mu"])
                    closure_residual = float(delta_log_mu - (delta_log_partial - delta_log_width))
                    dominant_partial = float(abs(delta_log_partial) > abs(delta_log_width))
                    dominant_width = float(abs(delta_log_width) > abs(delta_log_partial))
                    rows.append(
                        {
                            "mode": mode_key,
                            "layer": layer,
                            "D": float(D),
                            "eta": float(eta),
                            "amp": cur["amp"],
                            "amp_ref": cur["amp_ref"],
                            "amp_ratio": cur["amp_ratio"],
                            "partial_ratio": cur["partial_ratio"],
                            "width_ratio": cur["width_ratio"],
                            "mu": cur["mu"],
                            "chi2": cur["chi2"],
                            "mu_identity_residual": cur["mu_identity_residual"],
                            "delta_log_amp_point": delta_log_amp_point,
                            "delta_log_amp_ref": delta_log_amp_ref,
                            "delta_log_amp_ratio": delta_log_amp_ratio,
                            "delta_log_partial": delta_log_partial,
                            "delta_log_width": delta_log_width,
                            "delta_log_mu": delta_log_mu,
                            "delta_mu": float(cur["mu"] - base["mu"]),
                            "path_closure_residual": closure_residual,
                            "dominant_partial": dominant_partial,
                            "dominant_width": dominant_width,
                            "acceptance_mismatch": float(
                                ((base["chi2"] <= 4.0) ^ (cur["chi2"] <= 4.0)) if layer == 2 else 0.0
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for layer in LAYERS:
        for mode_key in COMPARE_MODES:
            sub = detail[(detail["layer"] == layer) & (detail["mode"] == mode_key)].copy()
            rows.append(
                {
                    "layer": layer,
                    "mode": mode_key,
                    "p95_abs_delta_log_amp_point": float(np.percentile(np.abs(sub["delta_log_amp_point"]), 95.0)),
                    "p95_abs_delta_log_amp_ref": float(np.percentile(np.abs(sub["delta_log_amp_ref"]), 95.0)),
                    "p95_abs_delta_log_partial": float(np.percentile(np.abs(sub["delta_log_partial"]), 95.0)),
                    "p95_abs_delta_log_width": float(np.percentile(np.abs(sub["delta_log_width"]), 95.0)),
                    "p95_abs_delta_log_mu": float(np.percentile(np.abs(sub["delta_log_mu"]), 95.0)),
                    "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu"]), 95.0)),
                    "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu"]))),
                    "p95_abs_path_closure_residual": float(np.percentile(np.abs(sub["path_closure_residual"]), 95.0)),
                    "max_abs_path_closure_residual": float(np.max(np.abs(sub["path_closure_residual"]))),
                    "max_abs_mu_identity_residual": float(np.max(np.abs(sub["mu_identity_residual"]))),
                    "dominant_fraction_partial": float(np.mean(sub["dominant_partial"])),
                    "dominant_fraction_width": float(np.mean(sub["dominant_width"])),
                    "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])) if layer == 2 else 0.0,
                    "corr_abs_mu_vs_abs_delta_log_partial": float(np.corrcoef(np.abs(sub["delta_mu"]), np.abs(sub["delta_log_partial"]))[0, 1]) if len(sub) > 1 else float("nan"),
                    "corr_abs_mu_vs_abs_delta_log_width": float(np.corrcoef(np.abs(sub["delta_mu"]), np.abs(sub["delta_log_width"]))[0, 1]) if len(sub) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    sub = detail[detail["layer"] == 2].copy()
    for (mode_key, d_val), grp in sub.groupby(["mode", "D"], sort=True):
        rows.append(
            {
                "mode": mode_key,
                "D": float(d_val),
                "slice_p95_abs_delta_log_partial": float(np.percentile(np.abs(grp["delta_log_partial"]), 95.0)),
                "slice_p95_abs_delta_log_width": float(np.percentile(np.abs(grp["delta_log_width"]), 95.0)),
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "slice_dominant_fraction_partial": float(np.mean(grp["dominant_partial"])),
                "slice_dominant_fraction_width": float(np.mean(grp["dominant_width"])),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "D"]).reset_index(drop=True)


def _plot_slices(slices: pd.DataFrame, out_path: Path) -> None:
    palette = {
        "cell_direct_runtime": "#4e79a7",
        "release_tuned": "#f28e2b",
        "release_sumanchor": "#59a14f",
        "release_eftsumanchor": "#76b7b2",
        "release_fullsumanchor": "#9c755f",
        "release_fullwidthrefamp": "#af7aa1",
        "tailm2_detlin": "#e15759",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for mode_key in COMPARE_MODES:
        sub = slices[slices["mode"] == mode_key]
        axes[0].plot(sub["D"], sub["slice_p95_abs_delta_log_partial"], marker="o", color=palette[mode_key], label=mode_key)
        axes[1].plot(sub["D"], sub["slice_p95_abs_delta_log_width"], marker="o", color=palette[mode_key], label=mode_key)
        axes[2].plot(sub["D"], sub["slice_p95_abs_delta_mu"], marker="o", color=palette[mode_key], label=mode_key)
    axes[0].set_title("Layer-2 p95 |Δlog partial|")
    axes[1].set_title("Layer-2 p95 |Δlog width|")
    axes[2].set_title("Layer-2 p95 |Δμ_μμ|")
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
    OUTDIR.mkdir(parents=True, exist_ok=True)
    kinetics = _build_kinetics()
    detail = _build_detail(kinetics)
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_observable_path_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_observable_path_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_observable_path_audit_slices.csv"
    png_path = OUTDIR / "model_chain_observable_path_audit.png"
    meta_path = OUTDIR / "model_chain_observable_path_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot_slices(slices, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "d_grid": [float(x) for x in D_GRID],
                "eta_grid": [float(x) for x in ETA_GRID],
                "layers": list(LAYERS),
                "ref_D": REF_D,
                "ref_eta": REF_ETA,
                "observable_mode": OBSERVABLE_MODE,
                "n_max": N_MAX,
                "modes": COMPARE_MODES,
            },
            indent=2,
        )
    )
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

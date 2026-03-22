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
    ("cell_direct_runtime_release_fullsumanchor", "release_fullsumanchor"),
    ("cell_direct_runtime_release_fullwidthrefamp", "release_fullwidthrefamp"),
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
    return {mode_key: make_baseline_kinetics(chain_mode=chain_mode, **common) for chain_mode, mode_key in MODES}


def _point_payload(kin, layer: int, D: float, eta: float) -> dict[str, float]:
    b_val = float(kin.B_N(layer, D))
    w_val = float(kin.layer_weight(layer, D, eta, T_COH))
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
    chi2 = float(((mu - 1.4) / 0.4) ** 2) if layer == 2 else float("nan")
    return {
        "B": b_val,
        "W": w_val,
        "amp": amp,
        "amp_ref": amp_ref,
        "partial_ratio": partial_ratio,
        "width_ratio": width_ratio,
        "mu": mu,
        "chi2": chi2,
    }


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.sum(vec))
    if denom <= 0.0:
        return np.zeros_like(vec)
    return np.asarray(vec / denom, dtype=float)


def _build_detail(kinetics: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for D in D_GRID:
        for eta in ETA_GRID:
            point: dict[int, dict[str, dict[str, float]]] = {}
            for layer in LAYERS:
                point[layer] = {}
                for mode_key, kin in kinetics.items():
                    point[layer][mode_key] = _point_payload(kin, layer, float(D), float(eta))

            base_b_vec = np.asarray([point[layer]["full_direct"]["B"] for layer in LAYERS], dtype=float)
            base_b_share = _normalize(base_b_vec)
            for mode_key in COMPARE_MODES:
                cur_b_vec = np.asarray([point[layer][mode_key]["B"] for layer in LAYERS], dtype=float)
                cur_b_share = _normalize(cur_b_vec)
                for layer in LAYERS:
                    base = point[layer]["full_direct"]
                    cur = point[layer][mode_key]
                    rows.append(
                        {
                            "mode": mode_key,
                            "layer": layer,
                            "D": float(D),
                            "eta": float(eta),
                            "B": float(cur["B"]),
                            "W": float(cur["W"]),
                            "amp": float(cur["amp"]),
                            "amp_ref": float(cur["amp_ref"]),
                            "partial_ratio": float(cur["partial_ratio"]),
                            "width_ratio": float(cur["width_ratio"]),
                            "mu": float(cur["mu"]),
                            "chi2": float(cur["chi2"]),
                            "log_B_ratio": _safe_log_ratio(cur["B"], base["B"]),
                            "log_B_sum_ratio": _safe_log_ratio(np.sum(cur_b_vec), np.sum(base_b_vec)),
                            "share_l1": float(np.sum(np.abs(cur_b_share - base_b_share))),
                            "delta_log_partial": _safe_log_ratio(cur["partial_ratio"], base["partial_ratio"]),
                            "delta_log_width": _safe_log_ratio(cur["width_ratio"], base["width_ratio"]),
                            "delta_log_mu": _safe_log_ratio(cur["mu"], base["mu"]),
                            "delta_mu": float(cur["mu"] - base["mu"]),
                            "acceptance_mismatch": float(((base["chi2"] <= 4.0) ^ (cur["chi2"] <= 4.0)) if layer == 2 else 0.0),
                        }
                    )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    sub = detail[detail["layer"] == 2].copy()
    for mode_key in COMPARE_MODES:
        grp = sub[sub["mode"] == mode_key]
        rows.append(
            {
                "mode": mode_key,
                "p95_abs_log_B_sum_ratio": float(np.percentile(np.abs(grp["log_B_sum_ratio"]), 95.0)),
                "p95_share_l1": float(np.percentile(grp["share_l1"], 95.0)),
                "p95_abs_delta_log_partial": float(np.percentile(np.abs(grp["delta_log_partial"]), 95.0)),
                "p95_abs_delta_log_width": float(np.percentile(np.abs(grp["delta_log_width"]), 95.0)),
                "p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
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
                "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                "slice_p95_abs_delta_log_width": float(np.percentile(np.abs(grp["delta_log_width"]), 95.0)),
                "slice_p95_abs_delta_log_partial": float(np.percentile(np.abs(grp["delta_log_partial"]), 95.0)),
                "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "D"]).reset_index(drop=True)


def _plot_slices(slices: pd.DataFrame, out_path: Path) -> None:
    palette = {
        "release_fullsumanchor": "#9c755f",
        "release_fullwidthrefamp": "#af7aa1",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, constrained_layout=True)
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
    axes[2].legend(frameon=False)
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

    detail_path = OUTDIR / "model_chain_fullwidthrefamp_reduced_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_fullwidthrefamp_reduced_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_fullwidthrefamp_reduced_audit_slices.csv"
    png_path = OUTDIR / "model_chain_fullwidthrefamp_reduced_audit.png"
    meta_path = OUTDIR / "model_chain_fullwidthrefamp_reduced_audit_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot_slices(slices, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "modes": [mode_key for _, mode_key in MODES],
                "observable_mode": OBSERVABLE_MODE,
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

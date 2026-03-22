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
BASELINE_DETAIL = OUTDIR / "model_chain_decomposition_baseline_detail.csv"

D_GRID = np.linspace(4.0, 20.0, 21)
ETA_GRID = np.array([0.2, 0.6, 1.0, 2.0, 4.0], dtype=float)
LAYERS = (1, 2, 3)
REF_D = 9.6
REF_ETA = 1.0
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])
OBSERVABLE_MODE = "eft_wilson_uv_rge"

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


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.sum(vec))
    if denom <= 0.0:
        return np.zeros_like(vec)
    return np.asarray(vec / denom, dtype=float)


def _layer_argmax(vec: np.ndarray) -> int:
    return int(np.argmax(np.asarray(vec, dtype=float))) + 1


def _load_delta_mu_lookup() -> dict[tuple[str, float, float], tuple[float, float]]:
    if not BASELINE_DETAIL.exists():
        raise FileNotFoundError(f"Missing baseline detail file: {BASELINE_DETAIL}")
    detail = pd.read_csv(BASELINE_DETAIL)
    sub = detail[detail["layer"] == 2].copy()
    lookup: dict[tuple[str, float, float], tuple[float, float]] = {}
    for row in sub.itertuples(index=False):
        key = (str(row.mode), round(float(row.D), 10), round(float(row.eta), 10))
        lookup[key] = (float(row.delta_mu), float(row.acceptance_mismatch))
    return lookup


def _point_vectors(kin, D: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    b_vec = np.asarray([float(kin.B_N(layer, D)) for layer in LAYERS], dtype=float)
    w_vec = np.asarray([float(kin.layer_weight(layer, D, eta, T_COH)) for layer in LAYERS], dtype=float)
    return b_vec, w_vec


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for mode_key in COMPARE_MODES:
        sub = detail[detail["mode"] == mode_key].copy()
        corr_norm = float(np.corrcoef(np.abs(sub["delta_mu2"]), np.abs(sub["log_B_sum_ratio"]))[0, 1]) if len(sub) > 1 else float("nan")
        corr_b2 = float(np.corrcoef(np.abs(sub["delta_mu2"]), np.abs(sub["log_B2_ratio"]))[0, 1]) if len(sub) > 1 else float("nan")
        corr_share = float(np.corrcoef(np.abs(sub["delta_mu2"]), sub["share_l1"])[0, 1]) if len(sub) > 1 else float("nan")
        rows.append(
            {
                "mode": mode_key,
                "p95_abs_log_B_sum_ratio": float(np.percentile(np.abs(sub["log_B_sum_ratio"]), 95.0)),
                "max_abs_log_B_sum_ratio": float(np.max(np.abs(sub["log_B_sum_ratio"]))),
                "p95_abs_log_B2_ratio": float(np.percentile(np.abs(sub["log_B2_ratio"]), 95.0)),
                "max_abs_log_B2_ratio": float(np.max(np.abs(sub["log_B2_ratio"]))),
                "p95_share_l1": float(np.percentile(sub["share_l1"], 95.0)),
                "max_share_l1": float(np.max(sub["share_l1"])),
                "argmax_B_mismatch_fraction": float(np.mean(sub["argmax_B_mismatch"])),
                "argmax_W_mismatch_fraction": float(np.mean(sub["argmax_W_mismatch"])),
                "acceptance_mismatch_fraction": float(np.mean(sub["acceptance_mismatch"])),
                "p95_abs_delta_mu2": float(np.percentile(np.abs(sub["delta_mu2"]), 95.0)),
                "max_abs_delta_mu2": float(np.max(np.abs(sub["delta_mu2"]))),
                "corr_abs_mu_vs_abs_log_B_sum_ratio": corr_norm,
                "corr_abs_mu_vs_abs_log_B2_ratio": corr_b2,
                "corr_abs_mu_vs_share_l1": corr_share,
            }
        )
    return pd.DataFrame(rows)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (mode_key, d_val), grp in detail.groupby(["mode", "D"], sort=True):
        rows.append(
            {
                "mode": mode_key,
                "D": float(d_val),
                "slice_p95_abs_log_B_sum_ratio": float(np.percentile(np.abs(grp["log_B_sum_ratio"]), 95.0)),
                "slice_p95_abs_log_B2_ratio": float(np.percentile(np.abs(grp["log_B2_ratio"]), 95.0)),
                "slice_p95_share_l1": float(np.percentile(grp["share_l1"], 95.0)),
                "slice_argmax_W_mismatch_fraction": float(np.mean(grp["argmax_W_mismatch"])),
                "slice_p95_abs_delta_mu2": float(np.percentile(np.abs(grp["delta_mu2"]), 95.0)),
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
        axes[0].plot(sub["D"], sub["slice_p95_abs_log_B2_ratio"], marker="o", color=palette[mode_key], label=mode_key)
        axes[1].plot(sub["D"], sub["slice_p95_share_l1"], marker="o", color=palette[mode_key], label=mode_key)
        axes[2].plot(sub["D"], sub["slice_p95_abs_delta_mu2"], marker="o", color=palette[mode_key], label=mode_key)
    axes[0].set_title("Layer-2 p95 |log B2 ratio|")
    axes[1].set_title("B-share competition L1")
    axes[2].set_title("Layer-2 p95 |Δμ_μμ|")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("D")
    axes[0].set_ylabel("value")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    delta_mu_lookup = _load_delta_mu_lookup()
    kinetics = _build_kinetics()

    rows: list[dict[str, float | str | int]] = []
    for D in D_GRID:
        for eta in ETA_GRID:
            b_vectors: dict[str, np.ndarray] = {}
            w_vectors: dict[str, np.ndarray] = {}
            for mode_key, kin in kinetics.items():
                b_vec, w_vec = _point_vectors(kin, float(D), float(eta))
                b_vectors[mode_key] = b_vec
                w_vectors[mode_key] = w_vec
            base_b = b_vectors["full_direct"]
            base_w = w_vectors["full_direct"]
            base_b_share = _normalize(base_b)
            base_w_share = _normalize(base_w)
            base_argmax_b = _layer_argmax(base_b)
            base_argmax_w = _layer_argmax(base_w)
            for mode_key in COMPARE_MODES:
                cur_b = b_vectors[mode_key]
                cur_w = w_vectors[mode_key]
                cur_b_share = _normalize(cur_b)
                cur_w_share = _normalize(cur_w)
                delta_mu2, acceptance_mismatch = delta_mu_lookup[(mode_key, round(float(D), 10), round(float(eta), 10))]
                rows.append(
                    {
                        "mode": mode_key,
                        "D": float(D),
                        "eta": float(eta),
                        "B1": float(cur_b[0]),
                        "B2": float(cur_b[1]),
                        "B3": float(cur_b[2]),
                        "W1": float(cur_w[0]),
                        "W2": float(cur_w[1]),
                        "W3": float(cur_w[2]),
                        "log_B_sum_ratio": _safe_log_ratio(float(np.sum(cur_b)), float(np.sum(base_b))),
                        "log_B2_ratio": _safe_log_ratio(float(cur_b[1]), float(base_b[1])),
                        "share_l1": float(np.sum(np.abs(cur_b_share - base_b_share))),
                        "w_share_l1": float(np.sum(np.abs(cur_w_share - base_w_share))),
                        "argmax_B_mismatch": float(_layer_argmax(cur_b) != base_argmax_b),
                        "argmax_W_mismatch": float(_layer_argmax(cur_w) != base_argmax_w),
                        "delta_mu2": float(delta_mu2),
                        "acceptance_mismatch": float(acceptance_mismatch),
                    }
                )

    detail = pd.DataFrame(rows)
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / "model_chain_bn_unification_audit_detail.csv"
    summary_path = OUTDIR / "model_chain_bn_unification_audit_summary.csv"
    slices_path = OUTDIR / "model_chain_bn_unification_audit_slices.csv"
    png_path = OUTDIR / "model_chain_bn_unification_audit.png"
    meta_path = OUTDIR / "model_chain_bn_unification_audit_run_meta.json"

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

#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eft_wilson_matching import UVTreeMatchConfig, wilson_matrix_uv_tree
from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
HLL_OUTDIR = ROOT / "output" / "hll_signal_strength"

LOW_D = [4.0, 4.8, 5.6, 6.4]
ETA_GRID = np.linspace(0.2, 4.0, 41)
REF_D = 9.6
REF_ETA = 1.0
T_COH = float(PAPER_BASELINE["t_coh"])
FULL_TAG = "chain_mode_full_direct_D21E41"
MODE_MAPS = {
    "release": "hll_signal_strength_map_full_direct_map_cell_direct_runtime_release_D21E41.csv",
    "bandcombo": "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_bandcombo_D21E41.csv",
    "bandm2": "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_bandm2_D21E41.csv",
}


def _grid_key(d: float, eta: float) -> tuple[float, float]:
    return (round(float(d), 10), round(float(eta), 10))


def _safe_log(x: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(x), floor)))


def _counterfactual_c22(
    g_uv: np.ndarray,
    p_kin: np.ndarray,
    m2: np.ndarray,
    kin,
) -> float:
    cfg = UVTreeMatchConfig(
        m2_floor=float(kin.params.hll_uv_m2_floor),
        coupling_floor=float(kin.params.hll_uv_coupling_floor),
    )
    c_uv = wilson_matrix_uv_tree(g_uv=g_uv, p_kin=p_kin, m2=m2, cfg=cfg)
    return float(max(c_uv[1, 1], kin.params.b_overlap_floor))


def _point_metrics(kin, D: float, eta: float) -> dict[str, object]:
    g_uv = np.asarray(kin._hll_g_uv_matrix(D), dtype=float)
    p_kin = np.asarray(
        kin._hll_pkin_vector(D, eta, T_COH, N_max=kin.params.hll_observable_nmax), dtype=float
    )
    m2 = np.asarray(kin._hll_m2_vector(D), dtype=float)

    uv_cfg = UVTreeMatchConfig(
        m2_floor=float(kin.params.hll_uv_m2_floor),
        coupling_floor=float(kin.params.hll_uv_coupling_floor),
    )
    c_uv = wilson_matrix_uv_tree(g_uv=g_uv, p_kin=p_kin, m2=m2, cfg=uv_cfg)
    c_match, _ = kin.hll_wilson_matrix_uv_match_with_meta(
        D=D, eta=eta, t_coh=T_COH, N_max=kin.params.hll_observable_nmax
    )
    c_low, meta = kin.hll_wilson_matrix_uv_rge_with_meta(
        D=D, eta=eta, t_coh=T_COH, N_max=kin.params.hll_observable_nmax
    )
    width_ratio = float(
        kin.hll_total_width_ratio_uv_rge(
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            N_max=kin.params.hll_observable_nmax,
        )
    )
    mu_mumu = float(
        kin.hll_mu_pred(
            2,
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            observable_mode="eft_wilson_uv_rge",
            N_max=kin.params.hll_observable_nmax,
        )
    )

    c_uv_22 = float(max(c_uv[1, 1], kin.params.b_overlap_floor))
    c_match_22 = float(max(c_match[1, 1], kin.params.b_overlap_floor))
    c_low_22 = float(max(c_low[1, 1], kin.params.b_overlap_floor))
    finite_diag_factor = float(max(c_match_22 / max(c_uv_22, kin.params.b_overlap_floor), kin.params.b_overlap_floor))
    rge_diag_factor = float(max(c_low_22 / max(c_match_22, kin.params.b_overlap_floor), kin.params.b_overlap_floor))

    return {
        "g_uv": g_uv,
        "p_kin": p_kin,
        "m2": m2,
        "c_uv_22": c_uv_22,
        "c_match_22": c_match_22,
        "c_low_22": c_low_22,
        "finite_diag_factor": finite_diag_factor,
        "rge_diag_factor": rge_diag_factor,
        "width_ratio": width_ratio,
        "mu_mumu": mu_mumu,
        "log_ratio": float(meta["log_ratio"]),
        "finite_fac_diag": float(meta["finite_fac_diag"]),
    }


def _read_mode_map(mode: str) -> pd.DataFrame:
    if mode == "full":
        path = HLL_OUTDIR / f"hll_signal_strength_map_{FULL_TAG}.csv"
    else:
        path = HLL_OUTDIR / MODE_MAPS[mode]
    if not path.exists():
        raise FileNotFoundError(f"Missing map CSV: {path}")
    return pd.read_csv(path)


def _mode_rows(label: str, kin, kin_full, full_map: pd.DataFrame, mode_map: pd.DataFrame) -> list[dict[str, float]]:
    ref_full = _point_metrics(kin_full, REF_D, REF_ETA)
    ref_mode = _point_metrics(kin, REF_D, REF_ETA)
    merged = full_map.merge(mode_map, on=["D", "eta"], suffixes=("_full", "_mode"))
    merged["acceptance_mismatch"] = (
        (merged["chi2_mumu_full"] <= 4.0) ^ (merged["chi2_mumu_mode"] <= 4.0)
    ).astype(float)
    accept_lookup = {
        _grid_key(r.D, r.eta): (
            float(r.mu_mumu_full),
            float(r.mu_mumu_mode),
            float(r.chi2_mumu_full),
            float(r.chi2_mumu_mode),
            float(r.acceptance_mismatch),
        )
        for r in merged.itertuples(index=False)
    }

    rows: list[dict[str, float]] = []
    for D in LOW_D:
        for eta in ETA_GRID:
            point_full = _point_metrics(kin_full, float(D), float(eta))
            point_mode = _point_metrics(kin, float(D), float(eta))
            mu_full, mu_mode_map, chi2_full, chi2_mode, mismatch = accept_lookup[_grid_key(D, eta)]

            log_uv_ratio_full = _safe_log(point_full["c_uv_22"]) - _safe_log(ref_full["c_uv_22"])
            log_uv_ratio_mode = _safe_log(point_mode["c_uv_22"]) - _safe_log(ref_mode["c_uv_22"])
            delta_log_uv_ratio = float(log_uv_ratio_mode - log_uv_ratio_full)

            log_fin_ratio_full = _safe_log(point_full["finite_diag_factor"]) - _safe_log(ref_full["finite_diag_factor"])
            log_fin_ratio_mode = _safe_log(point_mode["finite_diag_factor"]) - _safe_log(ref_mode["finite_diag_factor"])
            delta_log_fin_ratio = float(log_fin_ratio_mode - log_fin_ratio_full)

            log_rge_ratio_full = _safe_log(point_full["rge_diag_factor"]) - _safe_log(ref_full["rge_diag_factor"])
            log_rge_ratio_mode = _safe_log(point_mode["rge_diag_factor"]) - _safe_log(ref_mode["rge_diag_factor"])
            delta_log_rge_ratio = float(log_rge_ratio_mode - log_rge_ratio_full)

            delta_log_width_term = -(
                _safe_log(point_mode["width_ratio"]) - _safe_log(point_full["width_ratio"])
            )
            delta_log_mu = _safe_log(point_mode["mu_mumu"]) - _safe_log(point_full["mu_mumu"])
            identity_residual = float(
                delta_log_mu
                - (
                    2.0 * delta_log_uv_ratio
                    + 2.0 * delta_log_fin_ratio
                    + 2.0 * delta_log_rge_ratio
                    + delta_log_width_term
                )
            )

            cf_g_pt = _counterfactual_c22(
                g_uv=np.asarray(point_mode["g_uv"], dtype=float),
                p_kin=np.asarray(point_full["p_kin"], dtype=float),
                m2=np.asarray(point_full["m2"], dtype=float),
                kin=kin_full,
            )
            cf_g_ref = _counterfactual_c22(
                g_uv=np.asarray(ref_mode["g_uv"], dtype=float),
                p_kin=np.asarray(ref_full["p_kin"], dtype=float),
                m2=np.asarray(ref_full["m2"], dtype=float),
                kin=kin_full,
            )
            cf_p_pt = _counterfactual_c22(
                g_uv=np.asarray(point_full["g_uv"], dtype=float),
                p_kin=np.asarray(point_mode["p_kin"], dtype=float),
                m2=np.asarray(point_full["m2"], dtype=float),
                kin=kin_full,
            )
            cf_p_ref = _counterfactual_c22(
                g_uv=np.asarray(ref_full["g_uv"], dtype=float),
                p_kin=np.asarray(ref_mode["p_kin"], dtype=float),
                m2=np.asarray(ref_full["m2"], dtype=float),
                kin=kin_full,
            )
            cf_m_pt = _counterfactual_c22(
                g_uv=np.asarray(point_full["g_uv"], dtype=float),
                p_kin=np.asarray(point_full["p_kin"], dtype=float),
                m2=np.asarray(point_mode["m2"], dtype=float),
                kin=kin_full,
            )
            cf_m_ref = _counterfactual_c22(
                g_uv=np.asarray(ref_full["g_uv"], dtype=float),
                p_kin=np.asarray(ref_full["p_kin"], dtype=float),
                m2=np.asarray(ref_mode["m2"], dtype=float),
                kin=kin_full,
            )

            delta_log_uv_g_only = float((_safe_log(cf_g_pt) - _safe_log(cf_g_ref)) - log_uv_ratio_full)
            delta_log_uv_p_only = float((_safe_log(cf_p_pt) - _safe_log(cf_p_ref)) - log_uv_ratio_full)
            delta_log_uv_m2_only = float((_safe_log(cf_m_pt) - _safe_log(cf_m_ref)) - log_uv_ratio_full)

            comp_scores = {
                "width": abs(delta_log_width_term),
                "uv": abs(2.0 * delta_log_uv_ratio),
                "finite": abs(2.0 * delta_log_fin_ratio),
                "rge": abs(2.0 * delta_log_rge_ratio),
            }
            dominant_component = max(comp_scores, key=comp_scores.get)

            rows.append(
                {
                    "mode": label,
                    "D": float(D),
                    "eta": float(eta),
                    "mu_full": float(mu_full),
                    "mu_mode_map": float(mu_mode_map),
                    "mu_mode_path": float(point_mode["mu_mumu"]),
                    "chi2_full": float(chi2_full),
                    "chi2_mode": float(chi2_mode),
                    "acceptance_mismatch": float(mismatch),
                    "width_full": float(point_full["width_ratio"]),
                    "width_mode": float(point_mode["width_ratio"]),
                    "c_uv22_full": float(point_full["c_uv_22"]),
                    "c_uv22_mode": float(point_mode["c_uv_22"]),
                    "delta_log_mu": float(delta_log_mu),
                    "delta_log_uv_ratio": float(delta_log_uv_ratio),
                    "delta_log_fin_ratio": float(delta_log_fin_ratio),
                    "delta_log_rge_ratio": float(delta_log_rge_ratio),
                    "delta_log_width_term": float(delta_log_width_term),
                    "identity_residual": float(identity_residual),
                    "delta_log_uv_g_only": float(delta_log_uv_g_only),
                    "delta_log_uv_p_only": float(delta_log_uv_p_only),
                    "delta_log_uv_m2_only": float(delta_log_uv_m2_only),
                    "dominant_component": dominant_component,
                    "abs_width_component": float(comp_scores["width"]),
                    "abs_uv_component": float(comp_scores["uv"]),
                    "abs_fin_component": float(comp_scores["finite"]),
                    "abs_rge_component": float(comp_scores["rge"]),
                }
            )
    return rows


def _summary_rows(df: pd.DataFrame) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for mode, sub in df.groupby("mode", sort=True):
        rows.append(
            {
                "mode": str(mode),
                "p95_abs_delta_log_mu": float(np.percentile(np.abs(sub["delta_log_mu"]), 95.0)),
                "p95_abs_width_component": float(np.percentile(sub["abs_width_component"], 95.0)),
                "p95_abs_uv_component": float(np.percentile(sub["abs_uv_component"], 95.0)),
                "p95_abs_fin_component": float(np.percentile(np.abs(2.0 * sub["delta_log_fin_ratio"]), 95.0)),
                "p95_abs_rge_component": float(np.percentile(np.abs(2.0 * sub["delta_log_rge_ratio"]), 95.0)),
                "p95_abs_uv_g_only": float(np.percentile(np.abs(2.0 * sub["delta_log_uv_g_only"]), 95.0)),
                "p95_abs_uv_p_only": float(np.percentile(np.abs(2.0 * sub["delta_log_uv_p_only"]), 95.0)),
                "p95_abs_uv_m2_only": float(np.percentile(np.abs(2.0 * sub["delta_log_uv_m2_only"]), 95.0)),
                "max_abs_identity_residual": float(np.max(np.abs(sub["identity_residual"]))),
                "dominant_fraction_width": float(np.mean(sub["dominant_component"] == "width")),
                "dominant_fraction_uv": float(np.mean(sub["dominant_component"] == "uv")),
                "dominant_fraction_finite": float(np.mean(sub["dominant_component"] == "finite")),
                "dominant_fraction_rge": float(np.mean(sub["dominant_component"] == "rge")),
                "max_abs_mu_delta": float(np.max(np.abs(sub["mu_mode_map"] - sub["mu_full"]))),
                "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])),
            }
        )
    return rows


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    common = dict(
        observable_mode="eft_wilson_uv_rge",
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )
    kin_full = make_baseline_kinetics(chain_mode="full_direct", **common)
    mode_kin = {
        "release": make_baseline_kinetics(chain_mode="cell_direct_runtime_release", **common),
        "bandcombo": make_baseline_kinetics(chain_mode="cell_direct_runtime_release_bandcombo", **common),
        "bandm2": make_baseline_kinetics(chain_mode="cell_direct_runtime_release_bandm2", **common),
    }

    full_map = _read_mode_map("full")
    mode_maps = {mode: _read_mode_map(mode) for mode in mode_kin}

    rows = []
    for mode, kin in mode_kin.items():
        rows.extend(_mode_rows(mode, kin, kin_full, full_map, mode_maps[mode]))

    detail_df = pd.DataFrame(rows).sort_values(["mode", "D", "eta"]).reset_index(drop=True)
    summary_df = pd.DataFrame(_summary_rows(detail_df))

    slice_rows = []
    for (mode, D), sub in detail_df.groupby(["mode", "D"], sort=True):
        worst = sub.iloc[np.argmax(np.abs(sub["mu_mode_map"] - sub["mu_full"]).to_numpy())]
        slice_rows.append(
            {
                "mode": str(mode),
                "D": float(D),
                "slice_p95_abs_delta_log_mu": float(np.percentile(np.abs(sub["delta_log_mu"]), 95.0)),
                "slice_p95_abs_width_component": float(np.percentile(sub["abs_width_component"], 95.0)),
                "slice_p95_abs_uv_component": float(np.percentile(sub["abs_uv_component"], 95.0)),
                "slice_p95_abs_finite_component": float(np.percentile(np.abs(2.0 * sub["delta_log_fin_ratio"]), 95.0)),
                "slice_p95_abs_rge_component": float(np.percentile(np.abs(2.0 * sub["delta_log_rge_ratio"]), 95.0)),
                "slice_dominant_fraction_width": float(np.mean(sub["dominant_component"] == "width")),
                "slice_dominant_fraction_uv": float(np.mean(sub["dominant_component"] == "uv")),
                "slice_acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])),
                "slice_max_abs_mu_delta": float(np.max(np.abs(sub["mu_mode_map"] - sub["mu_full"]))),
                "worst_eta": float(worst["eta"]),
                "worst_mu_full": float(worst["mu_full"]),
                "worst_mu_mode": float(worst["mu_mode_map"]),
                "worst_width_component": float(worst["abs_width_component"]),
                "worst_uv_component": float(worst["abs_uv_component"]),
                "worst_uv_m2_only": float(abs(2.0 * worst["delta_log_uv_m2_only"])),
                "worst_dominant_component": str(worst["dominant_component"]),
            }
        )
    slice_df = pd.DataFrame(slice_rows).sort_values(["mode", "D"]).reset_index(drop=True)

    out_detail = OUTDIR / "runtime_direct_lowd_component_audit_detail.csv"
    out_slices = OUTDIR / "runtime_direct_lowd_component_audit_slices.csv"
    out_summary = OUTDIR / "runtime_direct_lowd_component_audit_summary.csv"
    out_fig = OUTDIR / "runtime_direct_lowd_component_audit.png"
    out_meta = OUTDIR / "runtime_direct_lowd_component_audit_run_meta.json"

    detail_df.to_csv(out_detail, index=False)
    slice_df.to_csv(out_slices, index=False)
    summary_df.to_csv(out_summary, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")
    plot_modes = [("release", "release vs full-direct"), ("bandm2", "bandm2 vs full-direct")]
    for col, (mode, title) in enumerate(plot_modes):
        sub = detail_df[detail_df["mode"] == mode].copy()
        for D in LOW_D:
            dsub = sub[sub["D"] == D].copy()
            axes[0, col].plot(dsub["eta"], np.abs(dsub["mu_mode_map"] - dsub["mu_full"]), label=f"D={D:.1f}")
            axes[1, col].plot(dsub["eta"], dsub["abs_width_component"], linestyle="--", label=f"D={D:.1f} width")
            axes[1, col].plot(dsub["eta"], dsub["abs_uv_component"], linestyle="-", label=f"D={D:.1f} uv")
            axes[2, col].plot(dsub["eta"], np.abs(2.0 * dsub["delta_log_uv_m2_only"]), linestyle="-", label=f"D={D:.1f} m2")
            axes[2, col].plot(dsub["eta"], np.abs(2.0 * dsub["delta_log_uv_g_only"]), linestyle="--", label=f"D={D:.1f} g")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("|Δμ_mumu|")
        axes[1, col].set_ylabel("width / uv size")
        axes[2, col].set_ylabel("UV subcomponents")
        axes[2, col].set_xlabel("eta")
        for row in range(3):
            axes[row, col].grid(alpha=0.25)
            axes[row, col].legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    meta = {
        "low_d": LOW_D,
        "ref_D": REF_D,
        "ref_eta": REF_ETA,
        "statement": (
            "Low-D residuals are decomposed on the actual mu path: "
            "delta log mu = 2(delta log UV + delta log finite + delta log RGE) + delta log width. "
            "The audit compares release, band-limited width, and band-limited width + m2 crossing correction "
            "against full-direct on the same D21xE41 grid."
        ),
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    for src in [out_detail, out_slices, out_summary, out_fig, out_meta]:
        dst = PAPER_DIR / src.name
        if src.suffix == ".png":
            dst.write_bytes(src.read_bytes())
        else:
            dst.write_text(src.read_text())

    print(f"[saved] {out_detail}")
    print(f"[saved] {out_slices}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")
    print(summary_df.to_csv(index=False))


if __name__ == "__main__":
    main()

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
SIGDIR = ROOT / "output" / "hll_signal_strength"

FOCUS_D = [4.0, 4.8, 6.4, 7.2]
ETA_GRID = np.linspace(0.2, 4.0, 41)
REF_D = 9.6
REF_ETA = 1.0
T_COH = float(PAPER_BASELINE["t_coh"])

FULL_MAP = SIGDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv"
DETLIN_MAP = SIGDIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_tailm2_detlin_D21E41.csv"


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


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    if not FULL_MAP.exists():
        raise FileNotFoundError(f"Missing full-direct map: {FULL_MAP}")
    if not DETLIN_MAP.exists():
        raise FileNotFoundError(f"Missing detlin map: {DETLIN_MAP}")

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
    kin_detlin = make_baseline_kinetics(chain_mode="cell_direct_runtime_release_tailm2", **common)

    full_map = pd.read_csv(FULL_MAP)
    detlin_map = pd.read_csv(DETLIN_MAP)
    merged = full_map.merge(detlin_map, on=["D", "eta"], suffixes=("_full", "_detlin"))
    merged["acceptance_mismatch"] = (
        (merged["chi2_mumu_full"] <= 4.0) ^ (merged["chi2_mumu_detlin"] <= 4.0)
    ).astype(float)
    accept_lookup = {
        _grid_key(r.D, r.eta): (
            float(r.mu_mumu_full),
            float(r.mu_mumu_detlin),
            float(r.chi2_mumu_full),
            float(r.chi2_mumu_detlin),
            float(r.acceptance_mismatch),
        )
        for r in merged.itertuples(index=False)
    }

    ref_full = _point_metrics(kin_full, REF_D, REF_ETA)
    ref_detlin = _point_metrics(kin_detlin, REF_D, REF_ETA)

    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            point_full = _point_metrics(kin_full, float(D), float(eta))
            point_detlin = _point_metrics(kin_detlin, float(D), float(eta))
            mu_full, mu_detlin_map, chi2_full, chi2_detlin, mismatch = accept_lookup[_grid_key(D, eta)]

            log_uv_ratio_full = _safe_log(point_full["c_uv_22"]) - _safe_log(ref_full["c_uv_22"])
            log_uv_ratio_detlin = _safe_log(point_detlin["c_uv_22"]) - _safe_log(ref_detlin["c_uv_22"])
            delta_log_uv_ratio = float(log_uv_ratio_detlin - log_uv_ratio_full)

            log_fin_ratio_full = _safe_log(point_full["finite_diag_factor"]) - _safe_log(ref_full["finite_diag_factor"])
            log_fin_ratio_detlin = _safe_log(point_detlin["finite_diag_factor"]) - _safe_log(ref_detlin["finite_diag_factor"])
            delta_log_fin_ratio = float(log_fin_ratio_detlin - log_fin_ratio_full)

            log_rge_ratio_full = _safe_log(point_full["rge_diag_factor"]) - _safe_log(ref_full["rge_diag_factor"])
            log_rge_ratio_detlin = _safe_log(point_detlin["rge_diag_factor"]) - _safe_log(ref_detlin["rge_diag_factor"])
            delta_log_rge_ratio = float(log_rge_ratio_detlin - log_rge_ratio_full)

            delta_log_width_term = -(
                _safe_log(point_detlin["width_ratio"]) - _safe_log(point_full["width_ratio"])
            )
            delta_log_mu = _safe_log(point_detlin["mu_mumu"]) - _safe_log(point_full["mu_mumu"])
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
                g_uv=np.asarray(point_detlin["g_uv"], dtype=float),
                p_kin=np.asarray(point_full["p_kin"], dtype=float),
                m2=np.asarray(point_full["m2"], dtype=float),
                kin=kin_full,
            )
            cf_g_ref = _counterfactual_c22(
                g_uv=np.asarray(ref_detlin["g_uv"], dtype=float),
                p_kin=np.asarray(ref_full["p_kin"], dtype=float),
                m2=np.asarray(ref_full["m2"], dtype=float),
                kin=kin_full,
            )
            cf_p_pt = _counterfactual_c22(
                g_uv=np.asarray(point_full["g_uv"], dtype=float),
                p_kin=np.asarray(point_detlin["p_kin"], dtype=float),
                m2=np.asarray(point_full["m2"], dtype=float),
                kin=kin_full,
            )
            cf_p_ref = _counterfactual_c22(
                g_uv=np.asarray(ref_full["g_uv"], dtype=float),
                p_kin=np.asarray(ref_detlin["p_kin"], dtype=float),
                m2=np.asarray(ref_full["m2"], dtype=float),
                kin=kin_full,
            )
            cf_m_pt = _counterfactual_c22(
                g_uv=np.asarray(point_full["g_uv"], dtype=float),
                p_kin=np.asarray(point_full["p_kin"], dtype=float),
                m2=np.asarray(point_detlin["m2"], dtype=float),
                kin=kin_full,
            )
            cf_m_ref = _counterfactual_c22(
                g_uv=np.asarray(ref_full["g_uv"], dtype=float),
                p_kin=np.asarray(ref_full["p_kin"], dtype=float),
                m2=np.asarray(ref_detlin["m2"], dtype=float),
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
                    "D": float(D),
                    "eta": float(eta),
                    "mu_full": float(mu_full),
                    "mu_detlin_map": float(mu_detlin_map),
                    "mu_detlin_path": float(point_detlin["mu_mumu"]),
                    "chi2_full": float(chi2_full),
                    "chi2_detlin": float(chi2_detlin),
                    "acceptance_mismatch": float(mismatch),
                    "width_full": float(point_full["width_ratio"]),
                    "width_detlin": float(point_detlin["width_ratio"]),
                    "c_uv22_full": float(point_full["c_uv_22"]),
                    "c_uv22_detlin": float(point_detlin["c_uv_22"]),
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

    detail_df = pd.DataFrame(rows).sort_values(["D", "eta"]).reset_index(drop=True)
    slice_rows: list[dict[str, float]] = []
    for d_val, sub in detail_df.groupby("D", sort=True):
        slice_rows.append(
            {
                "D": float(d_val),
                "slice_max_abs_delta_mu_mumu": float(np.max(np.abs(sub["mu_detlin_map"] - sub["mu_full"]))),
                "slice_p95_abs_delta_mu_mumu": float(np.percentile(np.abs(sub["mu_detlin_map"] - sub["mu_full"]), 95.0)),
                "slice_acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])),
                "slice_p95_abs_width_component": float(np.percentile(sub["abs_width_component"], 95.0)),
                "slice_p95_abs_uv_component": float(np.percentile(sub["abs_uv_component"], 95.0)),
                "slice_p95_abs_fin_component": float(np.percentile(np.abs(2.0 * sub["delta_log_fin_ratio"]), 95.0)),
                "slice_p95_abs_rge_component": float(np.percentile(np.abs(2.0 * sub["delta_log_rge_ratio"]), 95.0)),
                "slice_p95_abs_uv_g_only": float(np.percentile(np.abs(2.0 * sub["delta_log_uv_g_only"]), 95.0)),
                "slice_p95_abs_uv_p_only": float(np.percentile(np.abs(2.0 * sub["delta_log_uv_p_only"]), 95.0)),
                "slice_p95_abs_uv_m2_only": float(np.percentile(np.abs(2.0 * sub["delta_log_uv_m2_only"]), 95.0)),
                "slice_dominant_fraction_width": float(np.mean(sub["dominant_component"] == "width")),
                "slice_dominant_fraction_uv": float(np.mean(sub["dominant_component"] == "uv")),
                "slice_dominant_fraction_finite": float(np.mean(sub["dominant_component"] == "finite")),
                "slice_dominant_fraction_rge": float(np.mean(sub["dominant_component"] == "rge")),
            }
        )
    slice_df = pd.DataFrame(slice_rows).sort_values("D").reset_index(drop=True)

    worst = detail_df.iloc[int(np.argmax(np.abs(detail_df["mu_detlin_map"] - detail_df["mu_full"])))]
    summary_df = pd.DataFrame(
        [
            {
                "focus_D": ",".join(f"{d:.1f}" for d in FOCUS_D),
                "p95_abs_delta_log_mu": float(np.percentile(np.abs(detail_df["delta_log_mu"]), 95.0)),
                "p95_abs_width_component": float(np.percentile(detail_df["abs_width_component"], 95.0)),
                "p95_abs_uv_component": float(np.percentile(detail_df["abs_uv_component"], 95.0)),
                "p95_abs_fin_component": float(np.percentile(np.abs(2.0 * detail_df["delta_log_fin_ratio"]), 95.0)),
                "p95_abs_rge_component": float(np.percentile(np.abs(2.0 * detail_df["delta_log_rge_ratio"]), 95.0)),
                "p95_abs_uv_g_only": float(np.percentile(np.abs(2.0 * detail_df["delta_log_uv_g_only"]), 95.0)),
                "p95_abs_uv_p_only": float(np.percentile(np.abs(2.0 * detail_df["delta_log_uv_p_only"]), 95.0)),
                "p95_abs_uv_m2_only": float(np.percentile(np.abs(2.0 * detail_df["delta_log_uv_m2_only"]), 95.0)),
                "max_abs_identity_residual": float(np.max(np.abs(detail_df["identity_residual"]))),
                "dominant_fraction_width": float(np.mean(detail_df["dominant_component"] == "width")),
                "dominant_fraction_uv": float(np.mean(detail_df["dominant_component"] == "uv")),
                "dominant_fraction_finite": float(np.mean(detail_df["dominant_component"] == "finite")),
                "dominant_fraction_rge": float(np.mean(detail_df["dominant_component"] == "rge")),
                "acceptance_mismatch": float(np.mean(detail_df["acceptance_mismatch"])),
                "max_abs_mu_delta": float(np.max(np.abs(detail_df["mu_detlin_map"] - detail_df["mu_full"]))),
                "worst_D": float(worst["D"]),
                "worst_eta": float(worst["eta"]),
                "worst_mu_full": float(worst["mu_full"]),
                "worst_mu_detlin": float(worst["mu_detlin_map"]),
                "worst_abs_width_component": float(worst["abs_width_component"]),
                "worst_abs_uv_component": float(worst["abs_uv_component"]),
                "worst_abs_fin_component": float(worst["abs_fin_component"]),
                "worst_abs_rge_component": float(worst["abs_rge_component"]),
            }
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for D in FOCUS_D:
        dsub = detail_df[detail_df["D"] == D]
        axes[0, 0].plot(dsub["eta"], np.abs(dsub["mu_detlin_map"] - dsub["mu_full"]), label=f"D={D:.1f}")
        axes[0, 1].plot(dsub["eta"], dsub["abs_width_component"], label=f"D={D:.1f}")
        axes[1, 0].plot(dsub["eta"], dsub["abs_uv_component"], label=f"D={D:.1f}")
        axes[1, 1].plot(dsub["eta"], np.abs(2.0 * dsub["delta_log_uv_m2_only"]), linestyle="-", label=f"D={D:.1f} m2")
        axes[1, 1].plot(dsub["eta"], np.abs(2.0 * dsub["delta_log_uv_g_only"]), linestyle="--", alpha=0.8)
    axes[0, 0].set_title("|Δμ_μμ| vs full-direct")
    axes[0, 1].set_title("Width Component")
    axes[1, 0].set_title("UV Component")
    axes[1, 1].set_title("UV Subcomponents (solid=m2, dashed=g)")
    for ax in axes.flat:
        ax.set_xlabel("eta")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Runtime-Direct detlin Component Audit: low-D / mid-band focus")

    out_detail = OUTDIR / "runtime_direct_detlin_component_audit_detail.csv"
    out_slices = OUTDIR / "runtime_direct_detlin_component_audit_slices.csv"
    out_summary = OUTDIR / "runtime_direct_detlin_component_audit_summary.csv"
    out_fig = OUTDIR / "runtime_direct_detlin_component_audit.png"
    out_meta = OUTDIR / "runtime_direct_detlin_component_audit_run_meta.json"

    detail_df.to_csv(out_detail, index=False)
    slice_df.to_csv(out_slices, index=False)
    summary_df.to_csv(out_summary, index=False)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    out_meta.write_text(
        json.dumps(
            {
                "focus_D": FOCUS_D,
                "eta_grid_size": int(len(ETA_GRID)),
                "full_map": str(FULL_MAP),
                "detlin_map": str(DETLIN_MAP),
            },
            indent=2,
        )
    )

    for path in [out_detail, out_slices, out_summary, out_fig, out_meta]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {out_detail}")
    print(f"[saved] {out_slices}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")
    print(summary_df.to_dict(orient="records")[0])


if __name__ == "__main__":
    main()

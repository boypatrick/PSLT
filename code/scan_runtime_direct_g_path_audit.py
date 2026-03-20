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

FOCUS_D = [6.4, 7.2]
ETA_GRID = np.linspace(0.2, 4.0, 41)
REF_D = 9.6
REF_ETA = 1.0
T_COH = float(PAPER_BASELINE["t_coh"])


def _safe_log(x: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(x), floor)))


def _point_metrics(kin, D: float, eta: float) -> dict[str, float]:
    g = np.asarray(kin._hll_g_uv_matrix(D), dtype=float)
    g_diag = np.maximum(np.diag(g), kin.params.hll_uv_coupling_floor)
    g_norm = float(np.linalg.norm(g_diag))
    g_shape2 = float(g_diag[1] / max(g_norm, kin.params.hll_uv_coupling_floor))
    p = np.asarray(
        kin._hll_pkin_vector(D, eta, T_COH, N_max=kin.params.hll_observable_nmax), dtype=float
    )
    m2 = np.asarray(kin._hll_m2_vector(D), dtype=float)
    return {
        "g2": float(g_diag[1]),
        "g_norm": g_norm,
        "g_shape2": g_shape2,
        "p2": float(p[1]),
        "m2_2": float(m2[1]),
        "mu_mumu": float(
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
        ),
    }


def _mode_rows(label: str, kin, kin_full) -> list[dict[str, float]]:
    ref_full = _point_metrics(kin_full, REF_D, REF_ETA)
    ref_mode = _point_metrics(kin, REF_D, REF_ETA)
    rows: list[dict[str, float]] = []

    for D in FOCUS_D:
        for eta in ETA_GRID:
            point_full = _point_metrics(kin_full, float(D), float(eta))
            point_mode = _point_metrics(kin, float(D), float(eta))

            delta_log_g2 = (
                _safe_log(point_mode["g2"])
                - _safe_log(ref_mode["g2"])
                - (_safe_log(point_full["g2"]) - _safe_log(ref_full["g2"]))
            )
            delta_log_g_norm = (
                _safe_log(point_mode["g_norm"])
                - _safe_log(ref_mode["g_norm"])
                - (_safe_log(point_full["g_norm"]) - _safe_log(ref_full["g_norm"]))
            )
            delta_log_g_shape2 = (
                _safe_log(point_mode["g_shape2"])
                - _safe_log(ref_mode["g_shape2"])
                - (_safe_log(point_full["g_shape2"]) - _safe_log(ref_full["g_shape2"]))
            )
            delta_log_p2 = (
                _safe_log(point_mode["p2"])
                - _safe_log(ref_mode["p2"])
                - (_safe_log(point_full["p2"]) - _safe_log(ref_full["p2"]))
            )
            delta_log_m2_2 = (
                _safe_log(point_mode["m2_2"])
                - _safe_log(ref_mode["m2_2"])
                - (_safe_log(point_full["m2_2"]) - _safe_log(ref_full["m2_2"]))
            )
            delta_log_uv22 = 2.0 * delta_log_g2 + delta_log_p2 - delta_log_m2_2
            identity_residual = float(delta_log_g2 - (delta_log_g_norm + delta_log_g_shape2))

            g_norm_score = abs(2.0 * delta_log_g_norm)
            g_shape_score = abs(2.0 * delta_log_g_shape2)
            m2_score = abs(delta_log_m2_2)
            dominant = max(
                {
                    "g_norm": g_norm_score,
                    "g_shape2": g_shape_score,
                    "m2_2": m2_score,
                },
                key=lambda k: {
                    "g_norm": g_norm_score,
                    "g_shape2": g_shape_score,
                    "m2_2": m2_score,
                }[k],
            )

            rows.append(
                {
                    "mode": label,
                    "D": float(D),
                    "eta": float(eta),
                    "mu_full": float(point_full["mu_mumu"]),
                    "mu_mode": float(point_mode["mu_mumu"]),
                    "delta_log_g2": float(delta_log_g2),
                    "delta_log_g_norm": float(delta_log_g_norm),
                    "delta_log_g_shape2": float(delta_log_g_shape2),
                    "delta_log_p2": float(delta_log_p2),
                    "delta_log_m2_2": float(delta_log_m2_2),
                    "delta_log_uv22": float(delta_log_uv22),
                    "identity_residual": float(identity_residual),
                    "abs_g_norm_component": float(g_norm_score),
                    "abs_g_shape2_component": float(g_shape_score),
                    "abs_m2_component": float(m2_score),
                    "dominant_component": dominant,
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
    kin_bandm2 = make_baseline_kinetics(chain_mode="cell_direct_runtime_release_bandm2", **common)
    kin_tailm2 = make_baseline_kinetics(chain_mode="cell_direct_runtime_release_tailm2", **common)

    rows = []
    rows.extend(_mode_rows("bandm2", kin_bandm2, kin_full))
    rows.extend(_mode_rows("tailm2", kin_tailm2, kin_full))
    detail_df = pd.DataFrame(rows).sort_values(["mode", "D", "eta"]).reset_index(drop=True)

    summary_rows = []
    for mode, sub in detail_df.groupby("mode", sort=True):
        summary_rows.append(
            {
                "mode": str(mode),
                "p95_abs_g_norm_component": float(np.percentile(sub["abs_g_norm_component"], 95.0)),
                "p95_abs_g_shape2_component": float(np.percentile(sub["abs_g_shape2_component"], 95.0)),
                "p95_abs_m2_component": float(np.percentile(sub["abs_m2_component"], 95.0)),
                "p95_abs_delta_log_uv22": float(np.percentile(np.abs(sub["delta_log_uv22"]), 95.0)),
                "max_abs_identity_residual": float(np.max(np.abs(sub["identity_residual"]))),
                "dominant_fraction_g_norm": float(np.mean(sub["dominant_component"] == "g_norm")),
                "dominant_fraction_g_shape2": float(np.mean(sub["dominant_component"] == "g_shape2")),
                "dominant_fraction_m2_2": float(np.mean(sub["dominant_component"] == "m2_2")),
                "max_abs_mu_delta": float(np.max(np.abs(sub["mu_mode"] - sub["mu_full"]))),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    slice_rows = []
    for (mode, D), sub in detail_df.groupby(["mode", "D"], sort=True):
        worst = sub.iloc[np.argmax(np.abs(sub["mu_mode"] - sub["mu_full"]).to_numpy())]
        slice_rows.append(
            {
                "mode": str(mode),
                "D": float(D),
                "slice_p95_abs_g_norm_component": float(np.percentile(sub["abs_g_norm_component"], 95.0)),
                "slice_p95_abs_g_shape2_component": float(np.percentile(sub["abs_g_shape2_component"], 95.0)),
                "slice_p95_abs_m2_component": float(np.percentile(sub["abs_m2_component"], 95.0)),
                "slice_dominant_fraction_g_norm": float(np.mean(sub["dominant_component"] == "g_norm")),
                "slice_dominant_fraction_g_shape2": float(np.mean(sub["dominant_component"] == "g_shape2")),
                "slice_dominant_fraction_m2_2": float(np.mean(sub["dominant_component"] == "m2_2")),
                "slice_max_abs_mu_delta": float(np.max(np.abs(sub["mu_mode"] - sub["mu_full"]))),
                "worst_eta": float(worst["eta"]),
                "worst_mu_full": float(worst["mu_full"]),
                "worst_mu_mode": float(worst["mu_mode"]),
                "worst_abs_g_norm_component": float(worst["abs_g_norm_component"]),
                "worst_abs_g_shape2_component": float(worst["abs_g_shape2_component"]),
                "worst_abs_m2_component": float(worst["abs_m2_component"]),
                "worst_dominant_component": str(worst["dominant_component"]),
            }
        )
    slice_df = pd.DataFrame(slice_rows).sort_values(["mode", "D"]).reset_index(drop=True)

    out_detail = OUTDIR / "runtime_direct_g_path_audit_detail.csv"
    out_slices = OUTDIR / "runtime_direct_g_path_audit_slices.csv"
    out_summary = OUTDIR / "runtime_direct_g_path_audit_summary.csv"
    out_fig = OUTDIR / "runtime_direct_g_path_audit.png"
    out_meta = OUTDIR / "runtime_direct_g_path_audit_run_meta.json"

    detail_df.to_csv(out_detail, index=False)
    slice_df.to_csv(out_slices, index=False)
    summary_df.to_csv(out_summary, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    plot_modes = [("bandm2", "bandm2 vs full-direct"), ("tailm2", "tailm2 vs full-direct")]
    for col, (mode, title) in enumerate(plot_modes):
        sub = detail_df[detail_df["mode"] == mode].copy()
        for D in FOCUS_D:
            dsub = sub[sub["D"] == D].copy()
            axes[0, col].plot(dsub["eta"], dsub["abs_g_norm_component"], label=f"D={D:.1f} g-norm")
            axes[0, col].plot(dsub["eta"], dsub["abs_g_shape2_component"], linestyle="--", label=f"D={D:.1f} g-shape2")
            axes[0, col].plot(dsub["eta"], dsub["abs_m2_component"], linestyle="-.", label=f"D={D:.1f} m2")
            axes[1, col].plot(dsub["eta"], np.abs(dsub["mu_mode"] - dsub["mu_full"]), label=f"D={D:.1f} |Δμ|")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("UV component size")
        axes[1, col].set_ylabel("|Δμ_mumu|")
        axes[1, col].set_xlabel("eta")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
        axes[0, col].legend(loc="upper right", fontsize=8)
        axes[1, col].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    meta = {
        "focus_D": FOCUS_D,
        "ref_D": REF_D,
        "ref_eta": REF_ETA,
        "statement": (
            "Decompose the remaining UV residual into overall g-norm drift, middle-layer shape drift, "
            "and m2_2 drift around the D~6.4-7.2 band."
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

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MPLDIR = ROOT / ".mplconfig"
MPLDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics


OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAPDIR = ROOT / "output" / "hll_signal_strength"

FULL_MAP = MAPDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv"

OUT_SCAN = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_audit_scan.csv"
OUT_SUMMARY = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_audit_summary.csv"
OUT_SLICES = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_audit_best_slices.csv"
OUT_ACTIVATION = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_audit_best_activation.csv"
OUT_PNG = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_audit.png"
OUT_META = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_audit_run_meta.json"

FOCUS_D = [4.0, 4.8, 6.4, 7.2, 8.0]
REF_D = float(PAPER_BASELINE["ref_D"])
REF_ETA = float(PAPER_BASELINE["ref_eta"])
T_COH = float(PAPER_BASELINE["t_coh"])

WIDTH_FIX = {
    "width_power_base": 0.80,
    "width_power_log_ratio_coeff": 0.15,
    "width_power_min": 0.60,
    "width_power_max": 0.85,
    "width_power_trigger_lo": 2.5,
    "width_power_trigger_hi": 5.0,
    "width_power_turnoff_lo": 10.0,
    "width_power_turnoff_hi": 20.0,
    "width_power_tail_logratio_lo": 1.5,
    "width_power_tail_logratio_hi": 2.5,
    "width_power_tail_reboost_max": 0.15,
}
BETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]
CENTER_VALUES = [0.05, 0.06, 0.07, 0.08, 0.09]
HALF_WIDTH_VALUES = [0.01, 0.02, 0.03, 0.04]


def _load_map(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    df = pd.read_csv(path)
    rows: dict[tuple[float, float], dict[str, float]] = {}
    for row in df.itertuples(index=False):
        rows[(float(row.D), float(row.eta))] = {
            "mu_mumu": float(row.mu_mumu),
            "chi2_mumu": float(row.chi2_mumu),
        }
    return rows


def _build_kinetics():
    return make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode="cell_direct_runtime_release_tailm2gnorm",
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )


def _set_width_fix(kin) -> None:
    params = kin.params
    params.hll_match_width_power_base = float(WIDTH_FIX["width_power_base"])
    params.hll_match_width_power_log_ratio_coeff = float(WIDTH_FIX["width_power_log_ratio_coeff"])
    params.hll_match_width_power_min = float(WIDTH_FIX["width_power_min"])
    params.hll_match_width_power_max = float(WIDTH_FIX["width_power_max"])
    params.hll_match_width_power_trigger_lo = float(WIDTH_FIX["width_power_trigger_lo"])
    params.hll_match_width_power_trigger_hi = float(WIDTH_FIX["width_power_trigger_hi"])
    params.hll_match_width_power_turnoff_lo = float(WIDTH_FIX["width_power_turnoff_lo"])
    params.hll_match_width_power_turnoff_hi = float(WIDTH_FIX["width_power_turnoff_hi"])
    params.hll_match_width_power_tail_logratio_lo = float(WIDTH_FIX["width_power_tail_logratio_lo"])
    params.hll_match_width_power_tail_logratio_hi = float(WIDTH_FIX["width_power_tail_logratio_hi"])
    params.hll_match_width_power_tail_reboost_max = float(WIDTH_FIX["width_power_tail_reboost_max"])
    params.hll_uv_runtime_direct_m2_cross_blend = 0.75


def _patch_bandpass_gnorm(kin, *, beta: float, center: float, half_width: float) -> None:
    def _patched(self, hll_diag: np.ndarray, direct_diag: np.ndarray, direct_b123: np.ndarray) -> float:
        beta_max = float(beta)
        if beta_max <= 0.0:
            return 0.0

        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        if half_width <= 1e-12:
            activation = 1.0 if abs(metric - center) <= 1e-12 else 0.0
        else:
            activation = float(max(0.0, 1.0 - abs(metric - center) / half_width))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(beta_max * activation * b2_gate)

    kin._runtime_direct_gnorm_blend_weight = types.MethodType(_patched, kin)


def _evaluate_candidate(
    kin,
    full_map: dict[tuple[float, float], dict[str, float]],
    mu_obs: float,
    sigma_obs: float,
    *,
    label: str,
    beta: float,
    center: float,
    half_width: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    _set_width_fix(kin)
    _patch_bandpass_gnorm(kin, beta=beta, center=center, half_width=half_width)

    slice_stats: dict[float, dict[str, float]] = {}
    point_rows: list[dict[str, float]] = []
    global_max = 0.0

    for D in FOCUS_D:
        deltas = []
        mismatch = 0
        rows = [key for key in full_map if abs(key[0] - float(D)) < 1e-9]
        rows.sort(key=lambda key: key[1])
        for _, eta in rows:
            ref = full_map[(float(D), float(eta))]
            mu = float(
                kin.hll_mu_pred(
                    2,
                    D=float(D),
                    eta=float(eta),
                    t_coh=T_COH,
                    ref_D=REF_D,
                    ref_eta=REF_ETA,
                    observable_mode="eft_wilson_uv_rge",
                    N_max=kin.params.hll_observable_nmax,
                )
            )
            chi2 = float(((mu - mu_obs) / sigma_obs) ** 2)
            delta = abs(mu - ref["mu_mumu"])
            deltas.append(delta)
            global_max = max(global_max, delta)
            mismatch += int((ref["chi2_mumu"] <= 4.0) != (chi2 <= 4.0))
            point_rows.append(
                {
                    "label": label,
                    "beta": float(beta),
                    "center": float(center),
                    "half_width": float(half_width),
                    "D": float(D),
                    "eta": float(eta),
                    "mu_candidate": float(mu),
                    "mu_full": float(ref["mu_mumu"]),
                    "chi2_candidate": float(chi2),
                    "chi2_full": float(ref["chi2_mumu"]),
                    "abs_delta_mu_mumu": float(delta),
                    "acceptance_mismatch": float((ref["chi2_mumu"] <= 4.0) != (chi2 <= 4.0)),
                }
            )

        arr = np.asarray(deltas, dtype=float)
        slice_stats[float(D)] = {
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95.0)),
            "mismatch": float(mismatch / max(len(arr), 1)),
        }

    objective = (
        0.5 * slice_stats[4.8]["max"]
        + 1.2 * slice_stats[6.4]["max"]
        + 0.25 * slice_stats[7.2]["max"]
        + 0.25 * slice_stats[8.0]["max"]
        + 5.0 * slice_stats[4.0]["mismatch"]
    )
    row = {
        "label": label,
        "beta": float(beta),
        "center": float(center),
        "half_width": float(half_width),
        "objective": float(objective),
        "global_max_abs_delta_mu_mumu": float(global_max),
        "d4p0_p95_abs_delta_mu_mumu": float(slice_stats[4.0]["p95"]),
        "d4p0_max_abs_delta_mu_mumu": float(slice_stats[4.0]["max"]),
        "d4p0_acceptance_mismatch": float(slice_stats[4.0]["mismatch"]),
        "d4p8_p95_abs_delta_mu_mumu": float(slice_stats[4.8]["p95"]),
        "d4p8_max_abs_delta_mu_mumu": float(slice_stats[4.8]["max"]),
        "d6p4_p95_abs_delta_mu_mumu": float(slice_stats[6.4]["p95"]),
        "d6p4_max_abs_delta_mu_mumu": float(slice_stats[6.4]["max"]),
        "d7p2_p95_abs_delta_mu_mumu": float(slice_stats[7.2]["p95"]),
        "d7p2_max_abs_delta_mu_mumu": float(slice_stats[7.2]["max"]),
        "d8p0_p95_abs_delta_mu_mumu": float(slice_stats[8.0]["p95"]),
        "d8p0_max_abs_delta_mu_mumu": float(slice_stats[8.0]["max"]),
    }
    return row, point_rows


def _activation_diag(kin, D: float, *, beta: float, center: float, half_width: float) -> dict[str, float]:
    direct = kin._runtime_direct_b_operator_inputs(float(D))
    floor = float(kin.params.hll_uv_coupling_floor)
    h_diag = np.maximum(np.sqrt(np.maximum(kin._hll_yraw_vector(float(D)), floor)), floor)
    d_diag = np.maximum(np.asarray(np.diag(np.asarray(direct["g_uv"], dtype=float)), dtype=float), floor)
    metric = float(max(min(np.log(h_diag[0] / d_diag[0]), np.log(h_diag[2] / d_diag[2])), 0.0))
    if half_width <= 1e-12:
        activation = 1.0 if abs(metric - center) <= 1e-12 else 0.0
    else:
        activation = float(max(0.0, 1.0 - abs(metric - center) / half_width))
    b = np.maximum(np.asarray(direct["b123"], dtype=float).reshape(3), float(kin.params.b_overlap_floor))
    b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
    weight = float(beta) * activation * b2_gate
    return {
        "D": float(D),
        "metric": float(metric),
        "center": float(center),
        "half_width": float(half_width),
        "b2_gate": float(b2_gate),
        "activation": float(activation),
        "weight": float(weight),
        "hll_diag_1": float(h_diag[0]),
        "hll_diag_2": float(h_diag[1]),
        "hll_diag_3": float(h_diag[2]),
        "direct_diag_1": float(d_diag[0]),
        "direct_diag_2": float(d_diag[1]),
        "direct_diag_3": float(d_diag[2]),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    if not FULL_MAP.exists():
        raise FileNotFoundError(f"Missing full-direct map: {FULL_MAP}")

    full_map = _load_map(FULL_MAP)
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)
    candidates = [{"label": "width_only", "beta": 0.0, "center": 0.0, "half_width": 1.0}]
    for beta in BETA_VALUES:
        for center in CENTER_VALUES:
            for half_width in HALF_WIDTH_VALUES:
                candidates.append(
                    {
                        "label": "bandpass_gnorm",
                        "beta": float(beta),
                        "center": float(center),
                        "half_width": float(half_width),
                    }
                )

    rows: list[dict[str, float]] = []
    point_rows: list[dict[str, float]] = []
    for cand in candidates:
        kin = _build_kinetics()
        row, points = _evaluate_candidate(
            kin,
            full_map,
            mu_obs,
            sigma_obs,
            label=str(cand["label"]),
            beta=float(cand["beta"]),
            center=float(cand["center"]),
            half_width=float(cand["half_width"]),
        )
        rows.append(row)
        point_rows.extend(points)

    scan_df = pd.DataFrame(rows).sort_values(
        ["objective", "d6p4_max_abs_delta_mu_mumu", "d4p8_max_abs_delta_mu_mumu", "beta", "center", "half_width"]
    ).reset_index(drop=True)
    points_df = pd.DataFrame(point_rows)

    width_only = scan_df.loc[scan_df["label"] == "width_only"].iloc[0]
    best = scan_df.iloc[0]
    best_points = points_df[
        (points_df["label"] == best["label"])
        & (np.isclose(points_df["beta"], float(best["beta"])))
        & (np.isclose(points_df["center"], float(best["center"])))
        & (np.isclose(points_df["half_width"], float(best["half_width"])))
    ].copy()
    width_points = points_df[points_df["label"] == "width_only"].copy()

    slice_rows = []
    for D in FOCUS_D:
        width_sub = width_points[width_points["D"] == float(D)]
        best_sub = best_points[best_points["D"] == float(D)]
        slice_rows.append(
            {
                "D": float(D),
                "width_only_p95_abs_delta_mu_mumu": float(np.percentile(width_sub["abs_delta_mu_mumu"], 95.0)),
                "width_only_max_abs_delta_mu_mumu": float(width_sub["abs_delta_mu_mumu"].max()),
                "width_only_acceptance_mismatch": float(width_sub["acceptance_mismatch"].mean()),
                "best_bandpass_p95_abs_delta_mu_mumu": float(np.percentile(best_sub["abs_delta_mu_mumu"], 95.0)),
                "best_bandpass_max_abs_delta_mu_mumu": float(best_sub["abs_delta_mu_mumu"].max()),
                "best_bandpass_acceptance_mismatch": float(best_sub["acceptance_mismatch"].mean()),
            }
        )
    slices_df = pd.DataFrame(slice_rows)

    kin = _build_kinetics()
    _set_width_fix(kin)
    _patch_bandpass_gnorm(
        kin,
        beta=float(best["beta"]),
        center=float(best["center"]),
        half_width=float(best["half_width"]),
    )
    activation_df = pd.DataFrame(
        [
            _activation_diag(
                kin,
                D,
                beta=float(best["beta"]),
                center=float(best["center"]),
                half_width=float(best["half_width"]),
            )
            for D in FOCUS_D
        ]
    )

    summary = {
        "scan_rows": int(len(scan_df)),
        "width_fix_width_power_base": float(WIDTH_FIX["width_power_base"]),
        "width_fix_trigger_lo": float(WIDTH_FIX["width_power_trigger_lo"]),
        "width_fix_trigger_hi": float(WIDTH_FIX["width_power_trigger_hi"]),
        "width_fix_turnoff_lo": float(WIDTH_FIX["width_power_turnoff_lo"]),
        "width_fix_turnoff_hi": float(WIDTH_FIX["width_power_turnoff_hi"]),
        "width_fix_tail_reboost_max": float(WIDTH_FIX["width_power_tail_reboost_max"]),
        "width_only_objective": float(width_only["objective"]),
        "width_only_d4p8_max_abs_delta_mu_mumu": float(width_only["d4p8_max_abs_delta_mu_mumu"]),
        "width_only_d6p4_max_abs_delta_mu_mumu": float(width_only["d6p4_max_abs_delta_mu_mumu"]),
        "width_only_d7p2_max_abs_delta_mu_mumu": float(width_only["d7p2_max_abs_delta_mu_mumu"]),
        "width_only_d8p0_max_abs_delta_mu_mumu": float(width_only["d8p0_max_abs_delta_mu_mumu"]),
        "best_label": str(best["label"]),
        "best_beta": float(best["beta"]),
        "best_center": float(best["center"]),
        "best_half_width": float(best["half_width"]),
        "best_objective": float(best["objective"]),
        "best_d4p8_max_abs_delta_mu_mumu": float(best["d4p8_max_abs_delta_mu_mumu"]),
        "best_d6p4_max_abs_delta_mu_mumu": float(best["d6p4_max_abs_delta_mu_mumu"]),
        "best_d7p2_max_abs_delta_mu_mumu": float(best["d7p2_max_abs_delta_mu_mumu"]),
        "best_d8p0_max_abs_delta_mu_mumu": float(best["d8p0_max_abs_delta_mu_mumu"]),
        "best_d4p0_acceptance_mismatch": float(best["d4p0_acceptance_mismatch"]),
        "delta_objective_vs_width_only": float(best["objective"] - width_only["objective"]),
        "delta_d6p4_max_abs_delta_mu_mumu_vs_width_only": float(
            best["d6p4_max_abs_delta_mu_mumu"] - width_only["d6p4_max_abs_delta_mu_mumu"]
        ),
        "delta_d4p8_max_abs_delta_mu_mumu_vs_width_only": float(
            best["d4p8_max_abs_delta_mu_mumu"] - width_only["d4p8_max_abs_delta_mu_mumu"]
        ),
        "best_max_bandpass_weight_focus": float(activation_df["weight"].max()),
    }
    summary_df = pd.DataFrame([summary])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    band_df = scan_df[scan_df["label"] == "bandpass_gnorm"].copy()
    sc = axes[0].scatter(
        band_df["d6p4_max_abs_delta_mu_mumu"],
        band_df["d4p8_max_abs_delta_mu_mumu"],
        c=band_df["center"],
        cmap="plasma",
        s=42,
        alpha=0.85,
    )
    axes[0].scatter(
        [float(width_only["d6p4_max_abs_delta_mu_mumu"])],
        [float(width_only["d4p8_max_abs_delta_mu_mumu"])],
        marker="*",
        s=180,
        color="tab:orange",
        label="width-only",
        zorder=4,
    )
    axes[0].scatter(
        [float(best["d6p4_max_abs_delta_mu_mumu"])],
        [float(best["d4p8_max_abs_delta_mu_mumu"])],
        marker="*",
        s=180,
        color="tab:red",
        label="best band-pass",
        zorder=5,
    )
    axes[0].set_xlabel("D=6.4 max |Δmu_mumu|")
    axes[0].set_ylabel("D=4.8 max |Δmu_mumu|")
    axes[0].set_title("band-pass gnorm tradeoff")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    cbar = fig.colorbar(sc, ax=axes[0], shrink=0.9)
    cbar.set_label("band center")

    axes[1].plot(
        slices_df["D"],
        slices_df["width_only_p95_abs_delta_mu_mumu"],
        marker="o",
        linewidth=2.0,
        label="width-only",
    )
    axes[1].plot(
        slices_df["D"],
        slices_df["best_bandpass_p95_abs_delta_mu_mumu"],
        marker="s",
        linewidth=2.0,
        label="best band-pass",
    )
    axes[1].set_xlabel("D")
    axes[1].set_ylabel("p95 |Δmu_mumu| across eta")
    axes[1].set_title("best slice profile")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")
    fig.suptitle("strict all-direct tailm2 + band-pass gnorm audit")

    scan_df.to_csv(OUT_SCAN, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    slices_df.to_csv(OUT_SLICES, index=False)
    activation_df.to_csv(OUT_ACTIVATION, index=False)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    OUT_META.write_text(
        json.dumps(
            {
                "focus_D": FOCUS_D,
                "width_fix": WIDTH_FIX,
                "beta_values": BETA_VALUES,
                "center_values": CENTER_VALUES,
                "half_width_values": HALF_WIDTH_VALUES,
            },
            indent=2,
        )
    )

    for path in [OUT_SCAN, OUT_SUMMARY, OUT_SLICES, OUT_ACTIVATION, OUT_PNG, OUT_META]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {OUT_SCAN}")
    print(f"[saved] {OUT_SUMMARY}")
    print(f"[saved] {OUT_SLICES}")
    print(f"[saved] {OUT_ACTIVATION}")
    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_META}")
    print(summary_df.to_dict(orient="records")[0])


if __name__ == "__main__":
    main()

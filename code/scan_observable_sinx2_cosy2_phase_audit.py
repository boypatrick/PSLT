#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import make_baseline_kinetics
from scan_prolate_hybrid_chain_propagation_audit import N_MAX, OBSERVABLE_MODE, T_COH
from scan_prolate_hybrid_reviewer_counterfactual_audit import _subset_rows

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
MAP_DIR = ROOT / "output" / "hll_signal_strength"
STEM = "observable_sinx2_cosy2_phase_audit"

CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost"
LAYER = 2
BETA = 0.4

D60_FULL = MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv"
D60_BASE = MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv"
D60_PARENT = MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_D60E21_runtime_20260329_refsnapfix.csv"

D21_FULL = MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv"
D21_BASE = MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_currentbase_D21E41_20260329.csv"
D21_PARENT = MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_D21E41_runtime_20260329_refsnapfix.csv"

REGIONS = {
    "D60_hotspot": {
        "full_map": D60_FULL,
        "base_map": D60_BASE,
        "parent_map": D60_PARENT,
        "ref_D": 9.966101694915254,
        "ref_eta": 1.0,
        "D_targets": [4.813559322033898, 5.627118644067797, 6.169492525423729, 6.4406779661016955, 6.711864406779661],
    },
    "D21_guard": {
        "full_map": D21_FULL,
        "base_map": D21_BASE,
        "parent_map": D21_PARENT,
        "ref_D": 9.6,
        "ref_eta": 1.0,
        "D_targets": [4.0, 6.4, 8.0],
    },
    "D21_complement": {
        "full_map": D21_FULL,
        "base_map": D21_BASE,
        "parent_map": D21_PARENT,
        "ref_D": 9.6,
        "ref_eta": 1.0,
        "D_targets": [7.2, 9.6],
    },
}

KERNEL_GRID = ("current_cos_notch", "sinx2_cosy2")
MODE_GRID = ("partial_two_lobe", "pointamp_two_lobe")
OMEGA_GRID = (1.0, 1.2, 1.3, 1.4)
M_GRID = (12.0, 14.0)
PHASE_GRID = (
    0.4353981633974483,
    0.5853981633974483,
    0.7853981633974483,
)

CENTER_D = 5.8
WIDTH_D = 0.07
BOOST = 0.4
BOOST_CENTER = 5.86
BOOST_WIDTH = 0.05


def _build_runtime():
    return make_baseline_kinetics(
        observable_mode=OBSERVABLE_MODE,
        chain_mode=CHAIN_MODE,
        d_min=4.0,
        d_max=20.0,
        d_num=60,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )


def _gamma(beta: float) -> float:
    return float(1.0 / np.sqrt(max(1.0 - float(beta) ** 2, 1.0e-12)))


def _sigmoid_gate(x: float, center: float, width: float) -> float:
    z = (float(x) - float(center)) / max(float(width), 1.0e-9)
    return float(1.0 / (1.0 + np.exp(-z)))


def _gaussian(x: float, center: float, width: float) -> float:
    z = (float(x) - float(center)) / max(float(width), 1.0e-9)
    return float(np.exp(-0.5 * z * z))


def _phase_basis(kernel: str, theta_x: float, theta_y: float) -> float:
    if kernel == "current_cos_notch":
        return float(0.5 * (1.0 - np.cos(theta_x)))
    if kernel == "sinx2_cosy2":
        raw = np.sin(theta_x * theta_x) + np.cos(theta_y * theta_y)
        return float(0.25 * (raw + 2.0))
    raise ValueError(kernel)


def _two_lobe_factor(
    D: float,
    kernel: str,
    omega: float,
    m: float,
    phase: float,
) -> dict[str, float]:
    D_eff = float(D) / _gamma(BETA)
    gate = _sigmoid_gate(D_eff, CENTER_D, WIDTH_D)
    theta_x = float(m) * (D_eff - float(CENTER_D)) + float(phase)
    theta_y = float(m) * (D_eff - float(BOOST_CENTER)) + float(phase)
    basis = _phase_basis(str(kernel), theta_x, theta_y)
    suppress = float(np.exp(-float(omega) * gate * basis))
    compensate = float(np.exp(float(BOOST) * _gaussian(D_eff, BOOST_CENTER, BOOST_WIDTH)))
    factor = float(suppress * compensate)
    return {
        "D_eff": D_eff,
        "gate": gate,
        "theta_x": theta_x,
        "theta_y": theta_y,
        "basis": basis,
        "factor": factor,
    }


def _observable_payload(kin, D: float, eta: float, ref_D: float, ref_eta: float, cache: dict) -> dict[str, float]:
    key = (round(float(D), 12), round(float(eta), 12), round(float(ref_D), 12), round(float(ref_eta), 12))
    if key in cache:
        return cache[key]
    amp_raw = float(
        kin.hll_channel_amplitude(
            LAYER,
            float(D),
            float(eta),
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref_raw = float(
        kin.hll_channel_amplitude(
            LAYER,
            float(ref_D),
            float(ref_eta),
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref_eff = float(
        kin._blend_observable_ref_amp(
            amp_ref=amp_ref_raw,
            layer_n=int(LAYER),
            observable_mode=str(OBSERVABLE_MODE),
            D=float(D),
        )
    )
    amp_eff = float(
        kin._blend_observable_point_amp(
            amp=amp_raw,
            layer_n=int(LAYER),
            observable_mode=str(OBSERVABLE_MODE),
            D=float(D),
            eta=float(eta),
        )
    )
    partial_ratio = float((amp_eff / max(amp_ref_eff, 1.0e-30)) ** 2)
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
            D=float(D),
            eta=float(eta),
            t_coh=T_COH,
            ref_D=float(ref_D),
            ref_eta=float(ref_eta),
            N_max=N_MAX,
        )
    )
    mu = float(
        kin.hll_mu_pred(
            LAYER,
            D=float(D),
            eta=float(eta),
            t_coh=T_COH,
            ref_D=float(ref_D),
            ref_eta=float(ref_eta),
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    payload = {
        "amp_eff": amp_eff,
        "amp_ref_eff": amp_ref_eff,
        "partial_ratio": partial_ratio,
        "width_ratio": width_ratio,
        "mu": mu,
    }
    cache[key] = payload
    return payload


def _counterfactual_mu(
    kin,
    D: float,
    eta: float,
    ref_D: float,
    ref_eta: float,
    kernel: str,
    mode: str,
    omega: float,
    m: float,
    phase: float,
    cache: dict,
) -> tuple[float, dict[str, float]]:
    base = _observable_payload(kin, D, eta, ref_D, ref_eta, cache)
    lobes = _two_lobe_factor(D=float(D), kernel=str(kernel), omega=float(omega), m=float(m), phase=float(phase))
    factor = float(lobes["factor"])
    if mode == "partial_two_lobe":
        partial_cf = float(base["partial_ratio"] * factor)
    elif mode == "pointamp_two_lobe":
        amp_cf = float(base["amp_eff"] * np.sqrt(max(factor, 0.0)))
        partial_cf = float((amp_cf / max(base["amp_ref_eff"], 1.0e-30)) ** 2)
        partial_cf = float(
            kin._blend_observable_partial_ratio(
                partial_ratio=partial_cf,
                layer_n=int(LAYER),
                observable_mode=str(OBSERVABLE_MODE),
                D=float(D),
                eta=float(eta),
            )
        )
    else:
        raise ValueError(mode)
    mu_cf = float(partial_cf / max(base["width_ratio"], 1.0e-30))
    return mu_cf, lobes


def _load_region_table(cfg: dict[str, object]) -> pd.DataFrame:
    full_df = _subset_rows(Path(cfg["full_map"]), cfg["D_targets"]).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
    base_df = _subset_rows(Path(cfg["base_map"]), cfg["D_targets"]).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
    parent_df = _subset_rows(Path(cfg["parent_map"]), cfg["D_targets"]).rename(columns={"mu_mumu": "mu_parent", "chi2_mumu": "chi2_parent"})
    return (
        full_df.merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        .merge(parent_df[["D", "eta", "mu_parent", "chi2_parent"]], on=["D", "eta"], how="inner")
        .sort_values(["D", "eta"])
        .reset_index(drop=True)
    )


def _evaluate_combo(combo: dict[str, float | str], kin, region_tables: dict[str, pd.DataFrame], cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_rows = []
    detail_rows = []
    for region, cfg in REGIONS.items():
        merged = region_tables[region]
        records = []
        for row in merged.itertuples(index=False):
            mu_cf, meta = _counterfactual_mu(
                kin,
                D=float(row.D),
                eta=float(row.eta),
                ref_D=float(cfg["ref_D"]),
                ref_eta=float(cfg["ref_eta"]),
                kernel=str(combo["kernel"]),
                mode=str(combo["mode"]),
                omega=float(combo["omega"]),
                m=float(combo["m"]),
                phase=float(combo["phase"]),
                cache=cache,
            )
            records.append(
                {
                    **combo,
                    "region": region,
                    "D": float(row.D),
                    "eta": float(row.eta),
                    "mu_full": float(row.mu_full),
                    "mu_base": float(row.mu_base),
                    "mu_parent": float(row.mu_parent),
                    "mu_cf": mu_cf,
                    "abs_delta_base": abs(float(row.mu_base) - float(row.mu_full)),
                    "abs_delta_parent": abs(float(row.mu_parent) - float(row.mu_full)),
                    "abs_delta_cf": abs(mu_cf - float(row.mu_full)),
                    "basis": float(meta["basis"]),
                    "factor": float(meta["factor"]),
                }
            )
        detail = pd.DataFrame(records)
        detail["abs_improvement_vs_base"] = detail["abs_delta_base"] - detail["abs_delta_cf"]
        detail["abs_improvement_vs_parent"] = detail["abs_delta_parent"] - detail["abs_delta_cf"]
        detail["improved_vs_base"] = (detail["abs_delta_cf"] < detail["abs_delta_base"]).astype(float)
        detail["improved_vs_parent"] = (detail["abs_delta_cf"] < detail["abs_delta_parent"]).astype(float)
        detail_rows.append(detail)
        region_rows.append(
            {
                **combo,
                "region": region,
                "count_points": int(len(detail)),
                "p95_abs_delta_base": float(np.percentile(detail["abs_delta_base"], 95.0)),
                "p95_abs_delta_parent": float(np.percentile(detail["abs_delta_parent"], 95.0)),
                "p95_abs_delta_cf": float(np.percentile(detail["abs_delta_cf"], 95.0)),
                "mean_abs_improvement_vs_base": float(np.mean(detail["abs_improvement_vs_base"])),
                "mean_abs_improvement_vs_parent": float(np.mean(detail["abs_improvement_vs_parent"])),
                "improved_fraction_vs_base": float(np.mean(detail["improved_vs_base"])),
                "improved_fraction_vs_parent": float(np.mean(detail["improved_vs_parent"])),
            }
        )
    return pd.DataFrame(region_rows), pd.concat(detail_rows, ignore_index=True)


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    pivot = summary.copy().sort_values(["D60_hotspot", "D21_guard", "D21_complement"]).head(18)
    if pivot.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), constrained_layout=True)
    x = np.arange(len(pivot))
    axes[0].bar(x - 0.22, pivot["D60_hotspot"], width=0.22, label="D60 reviewer")
    axes[0].bar(x, pivot["D21_guard"], width=0.22, label="D21 reviewer")
    axes[0].bar(x + 0.22, pivot["D21_complement"], width=0.22, label="D21 complement")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pivot["kernel"], rotation=90, fontsize=8)
    axes[0].set_title("Phase basis p95 |Δμ|")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(pivot["D60_hotspot"], pivot["D21_guard"], c=(pivot["kernel"] == "sinx2_cosy2").astype(float), cmap="coolwarm", s=60)
    axes[1].set_xlabel("D60 reviewer p95 |Δμ|")
    axes[1].set_ylabel("D21 reviewer p95 |Δμ|")
    axes[1].set_title("Reviewer tradeoff by phase basis")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    kin = _build_runtime()
    cache: dict = {}
    region_tables = {region: _load_region_table(cfg) for region, cfg in REGIONS.items()}

    combos = []
    combo_id = 1
    for kernel in KERNEL_GRID:
        for mode in MODE_GRID:
            for omega in OMEGA_GRID:
                for m in M_GRID:
                    for phase in PHASE_GRID:
                        combos.append(
                            {
                                "combo_id": combo_id,
                                "kernel": str(kernel),
                                "mode": str(mode),
                                "omega": float(omega),
                                "m": float(m),
                                "phase": float(phase),
                            }
                        )
                        combo_id += 1

    region_frames = []
    detail_frames = []
    for combo in combos:
        region_df, detail_df = _evaluate_combo(combo, kin, region_tables, cache)
        region_frames.append(region_df)
        detail_frames.append(detail_df)

    summary_long = pd.concat(region_frames, ignore_index=True)
    detail = pd.concat(detail_frames, ignore_index=True)
    summary = (
        summary_long.pivot_table(
            index=["combo_id", "kernel", "mode", "omega", "m", "phase"],
            columns="region",
            values="p95_abs_delta_cf",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    parent_summary = (
        summary_long.pivot_table(
            index=["combo_id", "kernel", "mode", "omega", "m", "phase"],
            columns="region",
            values="p95_abs_delta_parent",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    parent_cols = {
        "D60_hotspot": "parent_D60_hotspot",
        "D21_guard": "parent_D21_guard",
        "D21_complement": "parent_D21_complement",
    }
    summary = summary.merge(
        parent_summary.rename(columns=parent_cols),
        on=["combo_id", "kernel", "mode", "omega", "m", "phase"],
        how="left",
    )
    summary["beats_parent_reviewer"] = (
        (summary["D60_hotspot"] < summary["parent_D60_hotspot"])
        & (summary["D21_guard"] < summary["parent_D21_guard"])
    )
    summary["beats_parent_all_three"] = summary["beats_parent_reviewer"] & (summary["D21_complement"] < summary["parent_D21_complement"])
    summary["joint_objective"] = summary["D60_hotspot"] + summary["D21_guard"] + summary["D21_complement"]
    summary = summary.sort_values(["joint_objective", "D60_hotspot", "D21_guard", "D21_complement"]).reset_index(drop=True)

    summary_path = OUTDIR / f"{STEM}_summary.csv"
    detail_path = OUTDIR / f"{STEM}_detail.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "chain_mode": CHAIN_MODE,
                "notes": (
                    "Controlled phase-basis audit comparing the current cos-notch control against a normalized "
                    "sin(x^2)+cos(y^2) kernel. x tracks the main Lorentz-contracted D-phase coordinate; y tracks "
                    "the compensating-lobe D-phase coordinate. The control isolates whether the new basis adds "
                    "useful D structure or washes the notch out."
                ),
                "fixed_params": {
                    "beta": BETA,
                    "center_D": CENTER_D,
                    "width_D": WIDTH_D,
                    "boost": BOOST,
                    "boost_center": BOOST_CENTER,
                    "boost_width": BOOST_WIDTH,
                },
            },
            indent=2,
        )
    )

    for path in [summary_path, detail_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.head(12).to_string(index=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote detail:  {detail_path}")


if __name__ == "__main__":
    main()

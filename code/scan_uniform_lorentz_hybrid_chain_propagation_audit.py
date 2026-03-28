#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pslt_lib import PSLTKinetics
from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics
from scan_uniform_lorentz_reparam_gn_chi_compatibility_audit import (
    HOTSPOT_BAND,
    KEY_D,
    _load_witness,
    _solve_profiles,
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "uniform_lorentz_hybrid_chain_propagation_audit"

GN_PATH = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_Dgrid60.csv"
GN_SPEC_PATH = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_spectrum_Dgrid60.csv"
CHI_PATH = ROOT / "output" / "chi_fp_2d" / "localized_chi_Dgrid60_fine.csv"

D_REF = 12.0
ETA_GRID = np.array([0.2, 0.6, 1.0, 2.0, 4.0], dtype=float)
LAYERS = (1, 2, 3)
OBSERVABLE_MODE = "eft_wilson_uv_rge"
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])
T_COH = float(PAPER_BASELINE["t_coh"])
CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost"

FAMILIES = (
    ("runtime_mainline", "runtime"),
    ("uniform_profile", "profile"),
    ("lorentz_hybrid_profile", "profile"),
)
COMPARE_PAIRS = (
    ("uniform_vs_runtime", "uniform_profile", "runtime_mainline"),
    ("hybrid_vs_uniform", "lorentz_hybrid_profile", "uniform_profile"),
    ("hybrid_vs_runtime", "lorentz_hybrid_profile", "runtime_mainline"),
)


def _safe_log_ratio(a: float, b: float, floor: float = 1e-30) -> float:
    return float(np.log(max(float(a), floor) / max(float(b), floor)))


def _anchor_to_reference(d: np.ndarray, proxy: np.ndarray, target: np.ndarray, ref_d: float = D_REF) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    proxy = np.asarray(proxy, dtype=float)
    target = np.asarray(target, dtype=float)
    t_ref = float(np.interp(ref_d, d, target))
    p_ref = max(float(np.interp(ref_d, d, proxy)), 1e-30)
    return np.maximum(t_ref * proxy / p_ref, 1e-30)


def _build_hybrid_profiles() -> tuple[pd.DataFrame, Path, Path, Path]:
    witness = _load_witness()
    profiles = _solve_profiles(witness["D"].to_numpy(dtype=float))
    detail = witness.merge(profiles, on="D", how="inner").sort_values("D").reset_index(drop=True)

    d_vals = detail["D"].to_numpy(dtype=float)
    chi_hybrid = _anchor_to_reference(d_vals, detail["lorentz_contract_omega2"], detail["chi_LR"])
    g3_hybrid = _anchor_to_reference(d_vals, detail["lorentz_contract_omega2"], detail["g3_raw"])

    detail["chi_LR_hybrid"] = chi_hybrid
    detail["g3_raw_hybrid"] = g3_hybrid

    gn_base = pd.read_csv(GN_PATH)
    if "level" in gn_base.columns:
        gn_base = gn_base.loc[gn_base["level"].astype(str).str.lower() == "fine"].copy()
    gn_base = gn_base.merge(detail[["D", "g3_raw_hybrid"]], on="D", how="inner")
    gn_base["g3_raw"] = gn_base["g3_raw_hybrid"]
    gn_base["g1_hat"] = gn_base["g1_raw"] / np.maximum(gn_base["g3_raw"], 1e-30)
    gn_base["g2_hat"] = gn_base["g2_raw"] / np.maximum(gn_base["g3_raw"], 1e-30)
    gn_base["g3_hat"] = 1.0
    gn_base = gn_base.drop(columns=["g3_raw_hybrid"])

    scale_df = detail[["D", "g3_raw", "g3_raw_hybrid"]].copy()
    scale_df["scale3"] = scale_df["g3_raw_hybrid"] / np.maximum(scale_df["g3_raw"], 1e-30)

    spec_base = pd.read_csv(GN_SPEC_PATH)
    if "level" in spec_base.columns:
        spec_base = spec_base.loc[spec_base["level"].astype(str).str.lower() == "fine"].copy()
    spec_base = spec_base.merge(scale_df[["D", "scale3", "g3_raw_hybrid"]], on="D", how="inner")

    mode_n = spec_base["mode_n"].astype(int).to_numpy()
    g_raw = spec_base["g_raw_n"].to_numpy(dtype=float)
    scale3 = spec_base["scale3"].to_numpy(dtype=float)
    g3_h = spec_base["g3_raw_hybrid"].to_numpy(dtype=float)

    g_new = g_raw.copy()
    g_new[mode_n >= 3] = g_raw[mode_n >= 3] * scale3[mode_n >= 3]
    spec_base["g_raw_n"] = np.maximum(g_new, 1e-30)
    spec_base["g_hat_to_g3"] = spec_base["g_raw_n"] / np.maximum(g3_h, 1e-30)
    spec_base = spec_base.drop(columns=["scale3", "g3_raw_hybrid"])

    chi_profile = detail[
        [
            "D",
            "chi_LR",
            "chi_LR_hybrid",
            "chi_DeltaE",
            "lambda3",
            "Gamma_ref",
            "lorentz_contract_omega2",
        ]
    ].copy()

    chi_path = OUTDIR / f"{STEM}_chi_profile_Dgrid60.csv"
    gn_path = OUTDIR / f"{STEM}_gn_profile_Dgrid60.csv"
    spec_path = OUTDIR / f"{STEM}_gn_spectrum_Dgrid60.csv"
    chi_profile.to_csv(chi_path, index=False)
    gn_base.to_csv(gn_path, index=False)
    spec_base.to_csv(spec_path, index=False)
    return detail, chi_path, gn_path, spec_path


def _build_kinetics(chi_path: Path, gn_path: Path, spec_path: Path, hybrid_detail: pd.DataFrame) -> dict[str, PSLTKinetics]:
    runtime = make_baseline_kinetics(
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

    chi_uniform = pd.read_csv(CHI_PATH)
    if "level" in chi_uniform.columns:
        chi_uniform = chi_uniform.loc[chi_uniform["level"].astype(str).str.lower() == "fine"].copy()
    chi_uniform = chi_uniform.sort_values("D").reset_index(drop=True)

    uniform_params = replace(
        runtime.params,
        chi_mode="localized_grid",
        chi_lr_D=tuple(float(x) for x in chi_uniform["D"].to_numpy(dtype=float)),
        chi_lr_vals=tuple(float(x) for x in chi_uniform["chi_LR"].to_numpy(dtype=float)),
        g_mode="fp_2d_full",
        g_fp_2d_csv=str(GN_PATH),
        g_fp_2d_spectrum_csv=str(GN_SPEC_PATH),
    )
    hybrid_params = replace(
        runtime.params,
        chi_mode="localized_grid",
        chi_lr_D=tuple(float(x) for x in hybrid_detail["D"].to_numpy(dtype=float)),
        chi_lr_vals=tuple(float(x) for x in hybrid_detail["chi_LR_hybrid"].to_numpy(dtype=float)),
        g_mode="fp_2d_full",
        g_fp_2d_csv=str(gn_path),
        g_fp_2d_spectrum_csv=str(spec_path),
    )

    return {
        "runtime_mainline": runtime,
        "uniform_profile": PSLTKinetics(params=uniform_params),
        "lorentz_hybrid_profile": PSLTKinetics(params=hybrid_params),
    }


def _point_payload(kin: PSLTKinetics, layer: int, D: float, eta: float) -> dict[str, float]:
    _, p_full, meta = kin.get_probabilities(D, eta, T_COH, N_max=N_MAX)
    return {
        "chi_eff": float(kin.chi_effective(D)),
        "g_eff": float(kin.g_N_effective(layer, D)),
        "gamma": float(kin.calculate_gamma_N(layer, D, eta)),
        "p_kin": float(kin.layer_kinetic_probability(layer, D, eta, T_COH, N_max=N_MAX)),
        "W": float(kin.layer_weight(layer, D, eta, T_COH)),
        "P": float(p_full[layer - 1]),
        "generation_ratio": float(meta["generation_ratio"]),
        "winner": float(meta["winner"]),
    }


def _build_detail(kinetics: dict[str, PSLTKinetics], d_grid: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for D in d_grid:
        for eta in ETA_GRID:
            payloads = {
                family: {
                    layer: _point_payload(kin, layer, float(D), float(eta))
                    for layer in LAYERS
                }
                for family, kin in kinetics.items()
            }
            for pair_name, lhs_name, rhs_name in COMPARE_PAIRS:
                for layer in LAYERS:
                    lhs = payloads[lhs_name][layer]
                    rhs = payloads[rhs_name][layer]
                    rows.append(
                        {
                            "pair": pair_name,
                            "lhs_family": lhs_name,
                            "rhs_family": rhs_name,
                            "layer": layer,
                            "D": float(D),
                            "eta": float(eta),
                            "chi_eff_lhs": lhs["chi_eff"],
                            "chi_eff_rhs": rhs["chi_eff"],
                            "g_eff_lhs": lhs["g_eff"],
                            "g_eff_rhs": rhs["g_eff"],
                            "gamma_lhs": lhs["gamma"],
                            "gamma_rhs": rhs["gamma"],
                            "p_kin_lhs": lhs["p_kin"],
                            "p_kin_rhs": rhs["p_kin"],
                            "W_lhs": lhs["W"],
                            "W_rhs": rhs["W"],
                            "P_lhs": lhs["P"],
                            "P_rhs": rhs["P"],
                            "generation_ratio_lhs": lhs["generation_ratio"],
                            "generation_ratio_rhs": rhs["generation_ratio"],
                            "winner_lhs": lhs["winner"],
                            "winner_rhs": rhs["winner"],
                            "log_chi_eff_ratio": _safe_log_ratio(lhs["chi_eff"], rhs["chi_eff"]),
                            "log_g_eff_ratio": _safe_log_ratio(lhs["g_eff"], rhs["g_eff"]),
                            "log_gamma_ratio": _safe_log_ratio(lhs["gamma"], rhs["gamma"]),
                            "log_p_kin_ratio": _safe_log_ratio(lhs["p_kin"], rhs["p_kin"]),
                            "log_W_ratio": _safe_log_ratio(lhs["W"], rhs["W"]),
                            "delta_P": float(lhs["P"] - rhs["P"]),
                            "delta_generation_ratio": float(lhs["generation_ratio"] - rhs["generation_ratio"]),
                            "winner_mismatch": float(lhs["winner"] != rhs["winner"]),
                        }
                    )
    return pd.DataFrame(rows)


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    hotspot_mask = detail["D"].between(*HOTSPOT_BAND)
    for pair_name, _, _ in COMPARE_PAIRS:
        pair_df = detail.loc[detail["pair"] == pair_name].copy()
        pair_hot = pair_df.loc[hotspot_mask.loc[pair_df.index]].copy()
        for layer in LAYERS:
            sub = pair_df.loc[pair_df["layer"] == layer].copy()
            hot = pair_hot.loc[pair_hot["layer"] == layer].copy()
            rows.append(
                {
                    "pair": pair_name,
                    "layer": layer,
                    "p95_abs_log_chi_eff_ratio": float(np.percentile(np.abs(sub["log_chi_eff_ratio"]), 95.0)),
                    "p95_abs_log_g_eff_ratio": float(np.percentile(np.abs(sub["log_g_eff_ratio"]), 95.0)),
                    "p95_abs_log_gamma_ratio": float(np.percentile(np.abs(sub["log_gamma_ratio"]), 95.0)),
                    "p95_abs_log_p_kin_ratio": float(np.percentile(np.abs(sub["log_p_kin_ratio"]), 95.0)),
                    "p95_abs_log_W_ratio": float(np.percentile(np.abs(sub["log_W_ratio"]), 95.0)),
                    "p95_abs_delta_P": float(np.percentile(np.abs(sub["delta_P"]), 95.0)),
                    "max_abs_delta_P": float(np.max(np.abs(sub["delta_P"]))),
                    "p95_abs_delta_generation_ratio": float(np.percentile(np.abs(sub["delta_generation_ratio"]), 95.0)),
                    "winner_mismatch_fraction": float(np.mean(sub["winner_mismatch"])),
                    "hotspot_p95_abs_log_gamma_ratio": float(np.percentile(np.abs(hot["log_gamma_ratio"]), 95.0)),
                    "hotspot_p95_abs_log_W_ratio": float(np.percentile(np.abs(hot["log_W_ratio"]), 95.0)),
                    "hotspot_p95_abs_delta_P": float(np.percentile(np.abs(hot["delta_P"]), 95.0)),
                    "hotspot_winner_mismatch_fraction": float(np.mean(hot["winner_mismatch"])),
                }
            )
    return pd.DataFrame(rows)


def _slices(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    layer2 = detail.loc[detail["layer"] == 2].copy()
    for pair_name, grp_pair in layer2.groupby("pair", sort=False):
        for d_val, grp in grp_pair.groupby("D", sort=True):
            rows.append(
                {
                    "pair": pair_name,
                    "D": float(d_val),
                    "slice_p95_abs_log_gamma_ratio": float(np.percentile(np.abs(grp["log_gamma_ratio"]), 95.0)),
                    "slice_p95_abs_log_W_ratio": float(np.percentile(np.abs(grp["log_W_ratio"]), 95.0)),
                    "slice_p95_abs_delta_P": float(np.percentile(np.abs(grp["delta_P"]), 95.0)),
                    "slice_p95_abs_delta_generation_ratio": float(
                        np.percentile(np.abs(grp["delta_generation_ratio"]), 95.0)
                    ),
                    "slice_winner_mismatch_fraction": float(np.mean(grp["winner_mismatch"])),
                }
            )
    return pd.DataFrame(rows).sort_values(["pair", "D"]).reset_index(drop=True)


def _plot(slices: pd.DataFrame, out_png: Path) -> None:
    palette = {
        "uniform_vs_runtime": "#4e79a7",
        "hybrid_vs_uniform": "#e15759",
        "hybrid_vs_runtime": "#59a14f",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4), sharex=True)
    for pair_name, grp in slices.groupby("pair", sort=False):
        axes[0].plot(grp["D"], grp["slice_p95_abs_log_gamma_ratio"], marker="o", color=palette[pair_name], label=pair_name)
        axes[1].plot(grp["D"], grp["slice_p95_abs_log_W_ratio"], marker="o", color=palette[pair_name], label=pair_name)
        axes[2].plot(grp["D"], grp["slice_p95_abs_delta_P"], marker="o", color=palette[pair_name], label=pair_name)
        axes[3].plot(grp["D"], grp["slice_winner_mismatch_fraction"], marker="o", color=palette[pair_name], label=pair_name)
    titles = [
        "Layer-2 p95 |log Γ ratio|",
        "Layer-2 p95 |log W ratio|",
        "Layer-2 p95 |ΔP|",
        "Layer-2 winner mismatch",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axvspan(HOTSPOT_BAND[0], HOTSPOT_BAND[1], color="#f3e7c7", alpha=0.35)
        for d_key in KEY_D:
            ax.axvline(d_key, color="0.8", ls="--", lw=1)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("D")
    axes[0].set_ylabel("value")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    hybrid_detail, chi_path, gn_path, spec_path = _build_hybrid_profiles()
    d_grid = hybrid_detail["D"].to_numpy(dtype=float)
    kinetics = _build_kinetics(chi_path, gn_path, spec_path, hybrid_detail)

    detail = _build_detail(kinetics, d_grid)
    summary = _summary(detail)
    slices = _slices(detail)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(slices, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "observable_mode": OBSERVABLE_MODE,
                "ref_D": D_REF,
                "eta_grid": [float(x) for x in ETA_GRID],
                "layers": list(LAYERS),
                "n_max": N_MAX,
                "hotspot_band": list(HOTSPOT_BAND),
                "hybrid_source_profiles": {
                    "chi_profile": str(chi_path),
                    "gn_profile": str(gn_path),
                    "gn_spectrum": str(spec_path),
                },
                "notes": (
                    "Uniform Lorentz hybrid chain audit. Only chi_LR and g3_raw/higher raw-g tower are substituted "
                    "using the length_contraction beta=0.4 extraction candidate; Gamma_ref and g2_raw remain uniform."
                ),
            },
            indent=2,
        )
    )

    for path in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.to_string(index=False))
    print(f"\nWrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")


if __name__ == "__main__":
    main()

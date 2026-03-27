#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_hybrid_chain_propagation_audit import (
    COMPARE_PAIRS,
    ETA_GRID,
    HOTSPOT_BAND,
    KEY_D,
    LAYERS,
    N_MAX,
    OBSERVABLE_MODE,
    T_COH,
    _build_hybrid_profiles,
    _build_kinetics,
    _safe_log_ratio,
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "prolate_hybrid_observable_propagation_audit"

REF_D = 9.6
REF_ETA = 1.0


def _observable_payload(kin, layer: int, D: float, eta: float) -> dict[str, float]:
    amp_raw = float(
        kin.hll_channel_amplitude(
            layer,
            D,
            eta,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref_raw = float(
        kin.hll_channel_amplitude(
            layer,
            REF_D,
            REF_ETA,
            T_COH,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    amp_ref_eff = float(
        kin._blend_observable_ref_amp(
            amp_ref=amp_ref_raw,
            layer_n=int(layer),
            observable_mode=str(OBSERVABLE_MODE),
            D=float(D),
        )
    )
    amp_eff = float(
        kin._blend_observable_point_amp(
            amp=amp_raw,
            layer_n=int(layer),
            observable_mode=str(OBSERVABLE_MODE),
            D=float(D),
            eta=float(eta),
        )
    )
    amp_ratio = float(amp_eff / max(amp_ref_eff, 1e-30))
    partial_ratio = float(amp_ratio * amp_ratio)
    partial_ratio = float(
        kin._blend_observable_partial_ratio(
            partial_ratio=partial_ratio,
            layer_n=int(layer),
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
            ref_D=float(REF_D),
            ref_eta=float(REF_ETA),
            N_max=N_MAX,
        )
    )
    mu = float(
        kin.hll_mu_pred(
            layer,
            D=float(D),
            eta=float(eta),
            t_coh=T_COH,
            ref_D=float(REF_D),
            ref_eta=float(REF_ETA),
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    mu_rebuild = float(partial_ratio / max(width_ratio, 1e-30))
    chi2 = float(((mu - 1.4) / 0.4) ** 2) if layer == 2 else float("nan")
    return {
        "amp_raw": amp_raw,
        "amp_ref_raw": amp_ref_raw,
        "amp_eff": amp_eff,
        "amp_ref_eff": amp_ref_eff,
        "amp_ratio": amp_ratio,
        "partial_ratio": partial_ratio,
        "width_ratio": width_ratio,
        "mu": mu,
        "mu_rebuild": mu_rebuild,
        "mu_identity_residual": float(mu - mu_rebuild),
        "chi2": chi2,
    }


def _build_detail(kinetics: dict[str, object], d_grid: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for D in d_grid:
        for eta in ETA_GRID:
            payloads = {
                family: {
                    layer: _observable_payload(kin, layer, float(D), float(eta))
                    for layer in LAYERS
                }
                for family, kin in kinetics.items()
            }
            for pair_name, lhs_name, rhs_name in COMPARE_PAIRS:
                for layer in LAYERS:
                    lhs = payloads[lhs_name][layer]
                    rhs = payloads[rhs_name][layer]
                    delta_log_partial = _safe_log_ratio(lhs["partial_ratio"], rhs["partial_ratio"])
                    delta_log_width = _safe_log_ratio(lhs["width_ratio"], rhs["width_ratio"])
                    delta_log_mu = _safe_log_ratio(lhs["mu"], rhs["mu"])
                    rows.append(
                        {
                            "pair": pair_name,
                            "lhs_family": lhs_name,
                            "rhs_family": rhs_name,
                            "layer": layer,
                            "D": float(D),
                            "eta": float(eta),
                            "amp_eff_lhs": lhs["amp_eff"],
                            "amp_eff_rhs": rhs["amp_eff"],
                            "amp_ref_eff_lhs": lhs["amp_ref_eff"],
                            "amp_ref_eff_rhs": rhs["amp_ref_eff"],
                            "partial_ratio_lhs": lhs["partial_ratio"],
                            "partial_ratio_rhs": rhs["partial_ratio"],
                            "width_ratio_lhs": lhs["width_ratio"],
                            "width_ratio_rhs": rhs["width_ratio"],
                            "mu_lhs": lhs["mu"],
                            "mu_rhs": rhs["mu"],
                            "mu_identity_residual_lhs": lhs["mu_identity_residual"],
                            "mu_identity_residual_rhs": rhs["mu_identity_residual"],
                            "delta_log_amp_point": _safe_log_ratio(lhs["amp_eff"], rhs["amp_eff"]),
                            "delta_log_amp_ref": _safe_log_ratio(lhs["amp_ref_eff"], rhs["amp_ref_eff"]),
                            "delta_log_partial": delta_log_partial,
                            "delta_log_width": delta_log_width,
                            "delta_log_mu": delta_log_mu,
                            "delta_mu": float(lhs["mu"] - rhs["mu"]),
                            "path_closure_residual": float(delta_log_mu - (delta_log_partial - delta_log_width)),
                            "acceptance_mismatch": float(
                                ((lhs["chi2"] <= 4.0) ^ (rhs["chi2"] <= 4.0)) if layer == 2 else 0.0
                            ),
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
                    "p95_abs_delta_log_amp_point": float(np.percentile(np.abs(sub["delta_log_amp_point"]), 95.0)),
                    "p95_abs_delta_log_amp_ref": float(np.percentile(np.abs(sub["delta_log_amp_ref"]), 95.0)),
                    "p95_abs_delta_log_partial": float(np.percentile(np.abs(sub["delta_log_partial"]), 95.0)),
                    "p95_abs_delta_log_width": float(np.percentile(np.abs(sub["delta_log_width"]), 95.0)),
                    "p95_abs_delta_log_mu": float(np.percentile(np.abs(sub["delta_log_mu"]), 95.0)),
                    "p95_abs_delta_mu": float(np.percentile(np.abs(sub["delta_mu"]), 95.0)),
                    "max_abs_delta_mu": float(np.max(np.abs(sub["delta_mu"]))),
                    "p95_abs_path_closure_residual": float(np.percentile(np.abs(sub["path_closure_residual"]), 95.0)),
                    "max_abs_path_closure_residual": float(np.max(np.abs(sub["path_closure_residual"]))),
                    "max_abs_mu_identity_residual_lhs": float(np.max(np.abs(sub["mu_identity_residual_lhs"]))),
                    "max_abs_mu_identity_residual_rhs": float(np.max(np.abs(sub["mu_identity_residual_rhs"]))),
                    "acceptance_mismatch": float(np.mean(sub["acceptance_mismatch"])) if layer == 2 else 0.0,
                    "hotspot_p95_abs_delta_log_partial": float(np.percentile(np.abs(hot["delta_log_partial"]), 95.0)),
                    "hotspot_p95_abs_delta_log_width": float(np.percentile(np.abs(hot["delta_log_width"]), 95.0)),
                    "hotspot_p95_abs_delta_mu": float(np.percentile(np.abs(hot["delta_mu"]), 95.0)),
                    "hotspot_max_abs_delta_mu": float(np.max(np.abs(hot["delta_mu"]))),
                }
            )
    return pd.DataFrame(rows)


def _slices(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    sub = detail.loc[detail["layer"] == 2].copy()
    for pair_name, grp_pair in sub.groupby("pair", sort=False):
        for d_val, grp in grp_pair.groupby("D", sort=True):
            rows.append(
                {
                    "pair": pair_name,
                    "D": float(d_val),
                    "slice_p95_abs_delta_log_partial": float(np.percentile(np.abs(grp["delta_log_partial"]), 95.0)),
                    "slice_p95_abs_delta_log_width": float(np.percentile(np.abs(grp["delta_log_width"]), 95.0)),
                    "slice_p95_abs_delta_mu": float(np.percentile(np.abs(grp["delta_mu"]), 95.0)),
                    "slice_max_abs_delta_mu": float(np.max(np.abs(grp["delta_mu"]))),
                    "slice_acceptance_mismatch": float(np.mean(grp["acceptance_mismatch"])),
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
        axes[0].plot(grp["D"], grp["slice_p95_abs_delta_log_partial"], marker="o", color=palette[pair_name], label=pair_name)
        axes[1].plot(grp["D"], grp["slice_p95_abs_delta_log_width"], marker="o", color=palette[pair_name], label=pair_name)
        axes[2].plot(grp["D"], grp["slice_p95_abs_delta_mu"], marker="o", color=palette[pair_name], label=pair_name)
        axes[3].plot(grp["D"], grp["slice_acceptance_mismatch"], marker="o", color=palette[pair_name], label=pair_name)
    titles = [
        "Layer-2 p95 |Δlog partial|",
        "Layer-2 p95 |Δlog width|",
        "Layer-2 p95 |Δμ|",
        "Layer-2 acceptance mismatch",
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
                "ref_D": REF_D,
                "ref_eta": REF_ETA,
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
                    "Observable-side audit preserves the current reviewer mainline family and only substitutes "
                    "hybrid prolate extraction into chi_LR and raw g profiles, while keeping Gamma_ref on uniform."
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

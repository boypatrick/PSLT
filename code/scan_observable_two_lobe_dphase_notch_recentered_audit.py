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
from scan_prolate_hybrid_reviewer_counterfactual_audit import REGIONS, _subset_rows
from scan_prolate_hybrid_chain_propagation_audit import N_MAX, OBSERVABLE_MODE, T_COH

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "observable_two_lobe_dphase_notch_recentered_audit"

CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost"
LAYER = 2
BETA = 0.4

MODE_GRID = ("partial_two_lobe", "pointamp_two_lobe")
OMEGA_GRID = (1.0, 1.1, 1.2, 1.3)
M_GRID = (12.0, 14.0)
PHASE_GRID = tuple(float(x) for x in (0.4353981633974483, 0.5853981633974483, 0.7853981633974483, 0.9853981633974482))
CENTER_GRID = (5.8, 5.9)
WIDTH_GRID = (0.05, 0.07)
BOOST_GRID = (0.1, 0.2, 0.4, 0.6, 0.8)
BOOST_CENTER_GRID = (5.82, 5.86, 5.90, 5.92)
BOOST_WIDTH_GRID = (0.05, 0.08, 0.10, 0.12)


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
    return float(1.0 / np.sqrt(max(1.0 - float(beta) ** 2, 1e-12)))


def _sigmoid_gate(x: float, center: float, width: float) -> float:
    z = (float(x) - float(center)) / max(float(width), 1e-9)
    return float(1.0 / (1.0 + np.exp(-z)))


def _gaussian(x: float, center: float, width: float) -> float:
    z = (float(x) - float(center)) / max(float(width), 1e-9)
    return float(np.exp(-0.5 * z * z))


def _two_lobe_factor(
    D: float,
    omega: float,
    m: float,
    phase: float,
    center_D: float,
    width_D: float,
    boost: float,
    boost_center: float,
    boost_width: float,
) -> dict[str, float]:
    D_eff = float(D) / _gamma(BETA)
    gate = _sigmoid_gate(D_eff, center_D, width_D)
    notch = 0.5 * (1.0 - np.cos(float(m) * (D_eff - float(center_D)) + float(phase)))
    suppress = float(np.exp(-float(omega) * gate * notch))
    compensate = float(np.exp(float(boost) * _gaussian(D_eff, boost_center, boost_width)))
    factor = suppress * compensate
    return {
        "D_eff": D_eff,
        "gate": gate,
        "notch": float(notch),
        "suppress": suppress,
        "compensate": compensate,
        "factor": float(factor),
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
    partial_ratio = float((amp_eff / max(amp_ref_eff, 1e-30)) ** 2)
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
    mode: str,
    omega: float,
    m: float,
    phase: float,
    center_D: float,
    width_D: float,
    boost: float,
    boost_center: float,
    boost_width: float,
    cache: dict,
) -> tuple[float, dict[str, float]]:
    base = _observable_payload(kin, D, eta, ref_D, ref_eta, cache)
    lobes = _two_lobe_factor(
        D=float(D),
        omega=float(omega),
        m=float(m),
        phase=float(phase),
        center_D=float(center_D),
        width_D=float(width_D),
        boost=float(boost),
        boost_center=float(boost_center),
        boost_width=float(boost_width),
    )
    factor = lobes["factor"]
    if mode == "partial_two_lobe":
        partial_cf = float(base["partial_ratio"] * factor)
    elif mode == "pointamp_two_lobe":
        amp_cf = float(base["amp_eff"] * np.sqrt(max(factor, 0.0)))
        partial_cf = float((amp_cf / max(base["amp_ref_eff"], 1e-30)) ** 2)
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
        raise ValueError(f"unknown mode: {mode}")
    mu_cf = float(partial_cf / max(base["width_ratio"], 1e-30))
    return mu_cf, lobes


def _region_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for region, cfg in REGIONS.items():
        full_df = _subset_rows(cfg["full_map"], cfg["D_targets"]).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
        base_df = _subset_rows(cfg["base_map"], cfg["D_targets"]).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
        tables[region] = full_df.merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
    return tables


def _evaluate_combo(combo: dict[str, float | str], kin, region_tables: dict[str, pd.DataFrame], cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_rows = []
    slice_rows = []
    for region, cfg in REGIONS.items():
        merged = region_tables[region]
        detail = []
        for row in merged.itertuples(index=False):
            mu_cf, lobes = _counterfactual_mu(
                kin,
                D=float(row.D),
                eta=float(row.eta),
                ref_D=float(cfg["ref_D"]),
                ref_eta=float(cfg["ref_eta"]),
                mode=str(combo["mode"]),
                omega=float(combo["omega"]),
                m=float(combo["m"]),
                phase=float(combo["phase"]),
                center_D=float(combo["center_D"]),
                width_D=float(combo["width_D"]),
                boost=float(combo["boost"]),
                boost_center=float(combo["boost_center"]),
                boost_width=float(combo["boost_width"]),
                cache=cache,
            )
            delta_base = float(row.mu_base - row.mu_full)
            delta_cf = float(mu_cf - row.mu_full)
            detail.append(
                {
                    **combo,
                    "region": region,
                    "D": float(row.D),
                    "eta": float(row.eta),
                    "D_eff": lobes["D_eff"],
                    "gate": lobes["gate"],
                    "notch": lobes["notch"],
                    "suppress": lobes["suppress"],
                    "compensate": lobes["compensate"],
                    "factor": lobes["factor"],
                    "mu_full": float(row.mu_full),
                    "mu_base": float(row.mu_base),
                    "mu_cf": float(mu_cf),
                    "abs_delta_base": abs(delta_base),
                    "abs_delta_cf": abs(delta_cf),
                    "abs_improvement": abs(delta_base) - abs(delta_cf),
                    "cf_shift_vs_base": float(mu_cf - row.mu_base),
                    "improved": float(abs(delta_cf) < abs(delta_base)),
                    "worsened": float(abs(delta_cf) > abs(delta_base)),
                    "acceptance_flip_vs_base": float(((row.chi2_base <= 4.0) ^ ((((mu_cf - 1.4) / 0.4) ** 2) <= 4.0))),
                }
            )
        region_detail = pd.DataFrame(detail)
        region_rows.append(
            {
                **combo,
                "region": region,
                "p95_abs_delta_base": float(np.percentile(region_detail["abs_delta_base"], 95.0)),
                "p95_abs_delta_cf": float(np.percentile(region_detail["abs_delta_cf"], 95.0)),
                "max_abs_delta_base": float(np.max(region_detail["abs_delta_base"])),
                "max_abs_delta_cf": float(np.max(region_detail["abs_delta_cf"])),
                "mean_abs_improvement": float(np.mean(region_detail["abs_improvement"])),
                "improved_fraction": float(np.mean(region_detail["improved"])),
                "worsened_fraction": float(np.mean(region_detail["worsened"])),
                "p95_abs_cf_shift_vs_base": float(np.percentile(np.abs(region_detail["cf_shift_vs_base"]), 95.0)),
                "acceptance_flip_fraction": float(np.mean(region_detail["acceptance_flip_vs_base"])),
            }
        )
        for d_val, grp in region_detail.groupby("D", sort=True):
            slice_rows.append(
                {
                    **combo,
                    "region": region,
                    "D": float(d_val),
                    "slice_p95_abs_delta_base": float(np.percentile(grp["abs_delta_base"], 95.0)),
                    "slice_p95_abs_delta_cf": float(np.percentile(grp["abs_delta_cf"], 95.0)),
                    "slice_mean_abs_improvement": float(np.mean(grp["abs_improvement"])),
                    "slice_improved_fraction": float(np.mean(grp["improved"])),
                    "slice_worsened_fraction": float(np.mean(grp["worsened"])),
                }
            )
    return pd.DataFrame(region_rows), pd.DataFrame(slice_rows)


def _wide_summary(region_summary: pd.DataFrame) -> pd.DataFrame:
    d60 = region_summary.loc[region_summary["region"] == "D60_hotspot"].copy().set_index("combo_id")
    d21 = region_summary.loc[region_summary["region"] == "D21_guard"].copy().set_index("combo_id")
    rows = []
    for combo_id in d60.index:
        r60 = d60.loc[combo_id]
        r21 = d21.loc[combo_id]
        rows.append(
            {
                "combo_id": int(combo_id),
                "mode": str(r60["mode"]),
                "omega": float(r60["omega"]),
                "m": float(r60["m"]),
                "phase": float(r60["phase"]),
                "center_D": float(r60["center_D"]),
                "width_D": float(r60["width_D"]),
                "boost": float(r60["boost"]),
                "boost_center": float(r60["boost_center"]),
                "boost_width": float(r60["boost_width"]),
                "D60_p95_base": float(r60["p95_abs_delta_base"]),
                "D60_p95_cf": float(r60["p95_abs_delta_cf"]),
                "D21_p95_base": float(r21["p95_abs_delta_base"]),
                "D21_p95_cf": float(r21["p95_abs_delta_cf"]),
                "D60_mean_abs_improvement": float(r60["mean_abs_improvement"]),
                "D21_mean_abs_improvement": float(r21["mean_abs_improvement"]),
                "D60_improved_fraction": float(r60["improved_fraction"]),
                "D21_improved_fraction": float(r21["improved_fraction"]),
                "D21_acceptance_flip_fraction": float(r21["acceptance_flip_fraction"]),
                "beats_mainline_both": float(
                    (float(r60["p95_abs_delta_cf"]) < float(r60["p95_abs_delta_base"]))
                    and (float(r21["p95_abs_delta_cf"]) < float(r21["p95_abs_delta_base"]))
                ),
                "joint_objective": float(
                    float(r60["p95_abs_delta_cf"])
                    + 8.0 * max(float(r21["p95_abs_delta_cf"]) - float(r21["p95_abs_delta_base"]), 0.0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["joint_objective", "D60_p95_cf", "D21_p95_cf"]).reset_index(drop=True)


def _plot(summary_wide: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    modes = {"partial_two_lobe": "#e15759", "pointamp_two_lobe": "#4e79a7"}
    for mode, grp in summary_wide.groupby("mode", sort=False):
        axes[0].scatter(grp["D60_p95_cf"], grp["D21_p95_cf"], s=48, color=modes[mode], label=mode)
        axes[1].scatter(grp["D60_mean_abs_improvement"], grp["D21_mean_abs_improvement"], s=48, color=modes[mode], label=mode)
    axes[0].set_title("p95 |Δμ|: D60 vs D21")
    axes[0].set_xlabel("D60 hotspot p95 |Δμ|")
    axes[0].set_ylabel("D21 guard p95 |Δμ|")
    axes[1].set_title("Mean improvement vs base")
    axes[1].set_xlabel("D60 mean improvement")
    axes[1].set_ylabel("D21 mean improvement")
    axes[2].plot(summary_wide.index, summary_wide["joint_objective"], marker="o", lw=1.5)
    axes[2].set_title("Ranked joint objective")
    axes[2].set_xlabel("sorted combo rank")
    axes[2].set_ylabel("objective")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    kin = _build_runtime()
    region_tables = _region_tables()
    cache: dict = {}

    combos = []
    all_region_rows = []
    all_slice_rows = []
    total = (
        len(MODE_GRID)
        * len(OMEGA_GRID)
        * len(M_GRID)
        * len(PHASE_GRID)
        * len(CENTER_GRID)
        * len(WIDTH_GRID)
        * len(BOOST_GRID)
        * len(BOOST_CENTER_GRID)
        * len(BOOST_WIDTH_GRID)
    )
    combo_id = 0
    for mode in MODE_GRID:
        for omega in OMEGA_GRID:
            for m in M_GRID:
                for phase in PHASE_GRID:
                    for center_D in CENTER_GRID:
                        for width_D in WIDTH_GRID:
                            for boost in BOOST_GRID:
                                for boost_center in BOOST_CENTER_GRID:
                                    for boost_width in BOOST_WIDTH_GRID:
                                        combo_id += 1
                                        combo = {
                                            "combo_id": combo_id,
                                            "mode": str(mode),
                                            "omega": float(omega),
                                            "m": float(m),
                                            "phase": float(phase),
                                            "center_D": float(center_D),
                                            "width_D": float(width_D),
                                            "boost": float(boost),
                                            "boost_center": float(boost_center),
                                            "boost_width": float(boost_width),
                                        }
                                        combos.append(combo)
                                        region_summary, slice_summary = _evaluate_combo(combo, kin, region_tables, cache)
                                        all_region_rows.append(region_summary)
                                        all_slice_rows.append(slice_summary)
                                        print(
                                            f"[combo {combo_id}/{total}] mode={mode} omega={omega} m={m} phase={phase:.3f} center={center_D} width={width_D} boost={boost} boost_center={boost_center} boost_width={boost_width}",
                                            flush=True,
                                        )

    region_summary = pd.concat(all_region_rows, ignore_index=True)
    slices = pd.concat(all_slice_rows, ignore_index=True)
    summary_wide = _wide_summary(region_summary)

    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    detail_path = OUTDIR / f"{STEM}_detail.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    summary_wide.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    region_summary.to_csv(detail_path, index=False)
    _plot(summary_wide, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "chain_mode": CHAIN_MODE,
                "observable_mode": OBSERVABLE_MODE,
                "layer": LAYER,
                "beta": BETA,
                "combos": combos,
                "notes": (
                    "Recentered two-lobe runtime-compatible observable D-phase notch shortlist on the current reviewer mainline. A suppressive Lorentz-descended D-phase notch is paired with a compensating positive lobe centered on the Lorentz-contracted D_eff~5.82-5.92 region corresponding to the D21 guard around D~6.4, to test whether the inert first pass was simply mis-centered."
                ),
            },
            indent=2,
        )
    )

    for path in [summary_path, slices_path, detail_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary_wide.head(20).to_string(index=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")
    print(f"Wrote detail:  {detail_path}")


if __name__ == "__main__":
    main()

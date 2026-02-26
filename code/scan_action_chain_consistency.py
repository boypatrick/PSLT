#!/usr/bin/env python3
"""
Full-chain consistency scan on the shared (D, eta, N) map.

Goal:
  Run map-level metrics with a single action-derived operator chain
  (A1/A2(D), eta(D), chi(D), optional t_coh(D)) and compare against
  the legacy scan-side surrogate kinetic closure.

Outputs:
  - output/kinetic_action_chain/surrogate_map.csv
  - output/kinetic_action_chain/action_chain_map.csv
  - output/kinetic_action_chain/action_chain_consistency_summary.csv
  - output/kinetic_action_chain/action_chain_consistency.png
  - paper/action_chain_consistency_summary.csv
  - paper/action_chain_consistency.png
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from hll_observable import HLLChannelPredictor, HLLObservableConfig
from pslt_lib import PSLTKinetics, PSLTParameters
from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile


OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


@dataclass(frozen=True)
class Case:
    name: str
    gamma_mode: str
    gamma_eta_mode: str
    t_coh_mode: str


def make_kinetics(case: Case, chi_profile: Dict[str, object], superrad_profile: Dict[str, object]) -> PSLTKinetics:
    gamma_mode = case.gamma_mode
    if gamma_mode == "action_auto":
        gamma_mode = str(superrad_profile["mode"])
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        g_mode="fp_2d_full",
        g_fp_full_window_blend=0.8,
        g_fp_full_tail_beta=1.1,
        g_fp_full_tail_shell_power=0.0,
        g_fp_full_tail_clip_min=1e-3,
        g_fp_full_tail_clip_max=0.95,
        chi=0.2,
        chi_mode=str(chi_profile["mode"]),
        chi_lr_D=tuple(float(x) for x in np.asarray(chi_profile["d"], dtype=float)),
        chi_lr_vals=tuple(float(x) for x in np.asarray(chi_profile["chi"], dtype=float)),
        A1=1.0,
        A2=1.0,
        gamma_mode=gamma_mode,
        gamma_superrad_csv=str(superrad_profile["path"]),
        gamma_eta_mode=case.gamma_eta_mode,
        t_coh_mode=case.t_coh_mode,
        t_coh_cap=1.0e4,
        b_mode="overlap_2d",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode="eft_wilson_uv_rge",
        hll_observable_nmax=20,
    )
    return PSLTKinetics(params)


def evaluate_case(case: Case, kin: PSLTKinetics, d_vals: np.ndarray, eta_vals: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    t_input = 1.0
    n_max = 20
    mu_obs = 1.4
    sigma_obs = 0.4

    cfg = HLLObservableConfig(
        mode="eft_wilson_uv_rge",
        t_coh=t_input,
        ref_D=10.0,
        ref_eta=1.0,
        n_max=20,
    )
    hll_mumu = HLLChannelPredictor(kin, layer_n=2, cfg=cfg)

    winner_map = np.zeros((len(eta_vals), len(d_vals)), dtype=int)
    r3_map = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    chi2_map = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    mu_map = np.zeros((len(eta_vals), len(d_vals)), dtype=float)

    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            _, _, meta = kin.get_probabilities(float(d), float(eta), t_input, N_max=n_max)
            winner = int(meta["winner"])
            r3 = float(meta["generation_ratio"])
            mu_pred = float(hll_mumu.mu_pred(float(d), float(eta)))
            chi2 = float(((mu_pred - mu_obs) / sigma_obs) ** 2)

            winner_map[i, j] = winner
            r3_map[i, j] = r3
            mu_map[i, j] = mu_pred
            chi2_map[i, j] = chi2

    stats = {
        "case": case.name,
        "gamma_mode_active": kin.active_gamma_mode(),
        "gamma_eta_mode_active": kin.active_gamma_eta_mode(),
        "t_coh_mode_active": kin.active_t_coh_mode(),
        "chi_mode_active": kin.active_chi_mode(),
        "g_mode_active": kin.active_g_mode(),
        "b_mode_active": kin.active_b_mode(),
        "f_R3_gt_0p90": float(np.mean(r3_map >= 0.90)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2_map <= 4.0)),
        "f_winner_gt_3": float(np.mean(winner_map > 3)),
        "best_chi2": float(np.min(chi2_map)),
    }
    maps = {
        "D_vals": d_vals,
        "eta_vals": eta_vals,
        "winner": winner_map,
        "R3": r3_map,
        "chi2": chi2_map,
        "mu_mumu": mu_map,
    }
    return maps, stats


def export_map_csv(path: Path, maps: Dict[str, np.ndarray]) -> None:
    d_vals = maps["D_vals"]
    eta_vals = maps["eta_vals"]
    winner = maps["winner"]
    r3 = maps["R3"]
    chi2 = maps["chi2"]
    mu = maps["mu_mumu"]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["D", "eta", "winner", "R3", "mu_mumu", "chi2_mumu"],
        )
        writer.writeheader()
        for i, eta in enumerate(eta_vals):
            for j, d in enumerate(d_vals):
                writer.writerow(
                    {
                        "D": float(d),
                        "eta": float(eta),
                        "winner": int(winner[i, j]),
                        "R3": float(r3[i, j]),
                        "mu_mumu": float(mu[i, j]),
                        "chi2_mumu": float(chi2[i, j]),
                    }
                )


def make_plot(
    out_png: Path,
    surrogate_stats: Dict[str, float],
    action_stats: Dict[str, float],
    dr3: np.ndarray,
    dmu: np.ndarray,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    labels = ["f(R3>0.90)", "f(chi2<4)", "f(Nwin>3)"]
    s_vals = [
        float(surrogate_stats["f_R3_gt_0p90"]),
        float(surrogate_stats["f_hmumu_chi2_le_4"]),
        float(surrogate_stats["f_winner_gt_3"]),
    ]
    a_vals = [
        float(action_stats["f_R3_gt_0p90"]),
        float(action_stats["f_hmumu_chi2_le_4"]),
        float(action_stats["f_winner_gt_3"]),
    ]
    x = np.arange(len(labels))
    w = 0.36

    axes[0].bar(x - w / 2.0, s_vals, width=w, label="surrogate", color="#5b6770")
    axes[0].bar(x + w / 2.0, a_vals, width=w, label="action_chain", color="#d97706")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Map-Level Fractions")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    im1 = axes[1].imshow(
        dr3,
        origin="lower",
        aspect="auto",
        extent=[float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())],
        cmap="coolwarm",
    )
    axes[1].set_title(r"$\Delta \mathcal{R}_3$ (action - surrogate)")
    axes[1].set_xlabel("D")
    axes[1].set_ylabel(r"$\eta$")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    im2 = axes[2].imshow(
        dmu,
        origin="lower",
        aspect="auto",
        extent=[float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())],
        cmap="coolwarm",
    )
    axes[2].set_title(r"$\Delta \mu_{\mu\mu}$ (action - surrogate)")
    axes[2].set_xlabel("D")
    axes[2].set_ylabel(r"$\eta$")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)

    fig.suptitle("Full-Chain Consistency: surrogate kinetic vs action-derived kinetic", fontsize=12.5)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_scan = scan_d_values(4.0, 20.0, 60)
    eta_scan = np.linspace(0.2, 4.0, 60)
    chi_profile = select_chi_profile(ROOT, d_scan)
    superrad_profile = select_superrad_profile(ROOT, d_scan)
    print(f"[chi profile] {chi_profile['path']} mode={chi_profile['mode']}")
    print(f"[superrad profile] {superrad_profile['path']} mode={superrad_profile['mode']}")

    cases = [
        Case(
            name="surrogate_kinetic",
            gamma_mode="surrogate",
            gamma_eta_mode="scan",
            t_coh_mode="input",
        ),
        Case(
            name="action_chain_full",
            gamma_mode="action_auto",
            gamma_eta_mode="scaled_amp",
            t_coh_mode="dephasing_profile_capped",
        ),
    ]

    maps_by_case: Dict[str, Dict[str, np.ndarray]] = {}
    stats_rows: List[Dict[str, float]] = []
    for case in cases:
        kin = make_kinetics(case, chi_profile, superrad_profile)
        maps, stats = evaluate_case(case, kin, d_scan, eta_scan)
        maps_by_case[case.name] = maps
        stats_rows.append(stats)

    surrogate = maps_by_case["surrogate_kinetic"]
    action = maps_by_case["action_chain_full"]
    dr3 = action["R3"] - surrogate["R3"]
    dmu = action["mu_mumu"] - surrogate["mu_mumu"]

    summary = {
        "delta_f_R3_gt_0p90": float(stats_rows[1]["f_R3_gt_0p90"] - stats_rows[0]["f_R3_gt_0p90"]),
        "delta_f_hmumu_chi2_le_4": float(stats_rows[1]["f_hmumu_chi2_le_4"] - stats_rows[0]["f_hmumu_chi2_le_4"]),
        "delta_f_winner_gt_3": float(stats_rows[1]["f_winner_gt_3"] - stats_rows[0]["f_winner_gt_3"]),
        "max_abs_delta_R3": float(np.max(np.abs(dr3))),
        "max_abs_delta_mu_mumu": float(np.max(np.abs(dmu))),
    }

    out_sur = OUTDIR / "surrogate_map.csv"
    out_act = OUTDIR / "action_chain_map.csv"
    out_summary = OUTDIR / "action_chain_consistency_summary.csv"
    out_png = OUTDIR / "action_chain_consistency.png"

    export_map_csv(out_sur, surrogate)
    export_map_csv(out_act, action)

    summary_rows = [{**stats_rows[0]}, {**stats_rows[1]}, {"case": "delta_action_minus_surrogate", **summary}]
    fieldnames: List[str] = []
    for row in summary_rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    make_plot(
        out_png,
        surrogate_stats=stats_rows[0],
        action_stats=stats_rows[1],
        dr3=dr3,
        dmu=dmu,
        d_vals=surrogate["D_vals"],
        eta_vals=surrogate["eta_vals"],
    )

    paper_summary = PAPER_DIR / out_summary.name
    paper_png = PAPER_DIR / out_png.name
    paper_summary.write_text(out_summary.read_text())
    paper_png.write_bytes(out_png.read_bytes())

    print(f"[saved] {out_sur}")
    print(f"[saved] {out_act}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_png}")
    print(f"[saved] {paper_summary}")
    print(f"[saved] {paper_png}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()

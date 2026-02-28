#!/usr/bin/env python3
"""
Map-level PSLT predictions for H->ll channels (ee, mumu, tautau).

Observable modes:
  - proxy_wratio:
      W_N(D,eta) = B_N * g_N(D) * (1 - exp(-Gamma_N(D,eta) * t_coh))
      mu_ll_pred = W_N / W_N_ref
  - eft_wilson_uv_rge (baseline):
      C_{eH}^{ij}(D,eta) = sum_N Y_{iN}(D,eta) P_N^(kin)(D,eta) Y_{jN}(D,eta)
      mu_ll_pred = |C_ii/C_ii_ref|^2 / (Gamma_tot/Gamma_tot_ref)
  - eft_wilson_uv_tree:
      C_{eH}^{ij}(D,eta) = sum_N g_{iN}(D) [P_N^(kin)(D,eta)/M_N^2(D)] g_{jN}(D)
      mu_ll_pred = |C_ii/C_ii_ref|^2 / (Gamma_tot/Gamma_tot_ref)
  - eft_wilson_uv_rge:
      C_{eH}^{ij}(mu_low) = RGE[ C_{eH}^{ij}(mu_match) ] with
      C_{eH}^{ij}(mu_match) = finite_one_loop_match[ C_{eH}^{tree}(mu_match) ]
      from the UV-tree closure

with layer-channel assignment:
  ee -> N=1, mumu -> N=2, tautau -> N=3.

Outputs:
  - output/hll_signal_strength/hll_signal_strength_map.csv
  - output/hll_signal_strength/hll_signal_strength_summary.csv
  - output/hll_signal_strength/hll_signal_strength_maps.png
  - paper/hll_signal_strength_summary.csv
  - paper/hll_signal_strength_maps.png

Chain profile selection:
  - --chain-mode auto        (default): prefer grid-aligned profiles and allow
                              interpolation fallback when needed.
  - --chain-mode full_direct: require exact localized-direct D-grid profiles
                              and strict grid lookup (no interpolation fallback).
  - --chain-mode full_direct_runtime:
                              build/rebuild active D-grid localized-direct
                              profiles at runtime, then run strict full_direct.
  - --chain-mode cell_direct_runtime:
                              no profile object; evaluate g_N(D), chi_LR(D),
                              and A_l(D) by direct solvers inside scan cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from hll_observable import HLLObservableConfig, HLLChannelPredictor  # noqa: E402
from pslt_lib import PSLTKinetics, PSLTParameters  # noqa: E402
from reference_anchor_utils import select_anchor_candidates_from_fixed_scan  # noqa: E402
from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile  # noqa: E402
from direct_chain_runtime import ensure_runtime_full_direct_profiles  # noqa: E402


OUTDIR = ROOT / "output" / "hll_signal_strength"
PAPER_DIR = ROOT / "paper"
DEFAULT_REF_CHOICE_JSON = ROOT / "output" / "hll_reference_anchor" / "reference_anchor_choice.json"

CHANNEL_TO_LAYER = {
    "ee": 1,
    "mumu": 2,
    "tautau": 3,
}

CHANNEL_TITLE = {
    "ee": r"H$\to ee$ (N=1)",
    "mumu": r"H$\to\mu\mu$ (N=2)",
    "tautau": r"H$\to\tau\tau$ (N=3)",
}

PAPER_BASELINE = {
    "c_eff": 0.5,
    "nu": 5.0,
    "kappa_g": 0.03,
    "g_mode": "fp_2d_full",
    "g_fp_norm_mode": "phase_space",
    "g_fp_full_window_blend": 0.8,
    "g_fp_full_tail_beta": 1.1,
    "g_fp_full_tail_shell_power": 0.0,
    "g_fp_full_tail_clip_min": 1e-3,
    "g_fp_full_tail_clip_max": 0.95,
    "chi_legacy": 0.2,
    "chi_mode": "localized_grid",
    "A1": 1.0,
    "A2": 1.0,
    "gamma_mode": "action_grid",
    "p_B": 0.30,
    "b_mode": "overlap_2d",
    "t_coh": 1.0,
    "ref_D": 10.0,
    "ref_eta": 1.0,
    "hll_observable_mode": "eft_wilson_uv_rge",
    "hll_observable_nmax": 20,
    # UV-tree calibration knobs (used when --observable-mode eft_wilson_uv_tree)
    "hll_uv_blend": 0.00,
    "hll_uv_m2_power": 1.00,
    "hll_uv_match_kappa_diag": 0.0,
    "hll_uv_match_kappa_offdiag": 0.0,
    "hll_uv_rge_mu_low": 1.0,
    "hll_uv_rge_gamma_diag": 2.0,
    "hll_uv_rge_gamma_offdiag": 1.0,
    "hll_uv_rge_log_clip": 6.0,
    "D_min": 4.0,
    "D_max": 20.0,
    "D_num": 60,
    "eta_min": 0.2,
    "eta_max": 4.0,
    "eta_num": 60,
}

B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


@dataclass(frozen=True)
class Observation:
    mu_obs: float
    sigma_obs: float
    source: str


def load_observations() -> Dict[str, Observation]:
    """
    Load optional observed mu values from data/pdg_leptons.json.

    Available by default: H_to_mumu. Others are optional and can be added to JSON.
    """
    obs: Dict[str, Observation] = {}
    path = ROOT / "data" / "pdg_leptons.json"
    key_map = {
        "H_to_ee": "ee",
        "H_to_mumu": "mumu",
        "H_to_tautau": "tautau",
    }

    if path.exists():
        try:
            payload = json.loads(path.read_text())
            section = payload.get("higgs_signal_strength", {})
            for key, channel in key_map.items():
                entry = section.get(key)
                if not isinstance(entry, dict):
                    continue
                mu = entry.get("value")
                sig = entry.get("combined_uncertainty")
                if mu is None or sig is None:
                    continue
                obs[channel] = Observation(
                    mu_obs=float(mu),
                    sigma_obs=float(sig),
                    source=str(entry.get("reference", entry.get("source", "unknown"))),
                )
        except Exception as exc:
            print(f"[warn] failed to load observations from {path}: {exc}")

    if "mumu" not in obs:
        obs["mumu"] = Observation(
            mu_obs=1.4,
            sigma_obs=0.4,
            source="fallback default (ATLAS Run-3 proxy)",
        )
    return obs


def make_baseline_kinetics(
    observable_mode: str,
    chain_mode: str,
    d_min: float,
    d_max: float,
    d_num: int,
    uv_blend: float,
    uv_m2_power: float,
    uv_match_kappa_diag: float,
    uv_match_kappa_offdiag: float,
    uv_rge_mu_low: float,
    uv_rge_gamma_diag: float,
    uv_rge_gamma_offdiag: float,
    uv_rge_log_clip: float,
    runtime_direct_force: bool,
    runtime_direct_no_cache: bool,
    runtime_direct_chi_rho_max: float,
    runtime_direct_chi_z_margin: float,
    runtime_direct_chi_n_mu: int,
    runtime_direct_chi_tol: float,
    runtime_direct_chi_maxiter: int,
    runtime_direct_chi_sigma: float,
    runtime_direct_superrad_zmax: float,
    runtime_direct_superrad_ref_d: float,
    runtime_direct_superrad_n_ref: int,
) -> PSLTKinetics:
    d_scan = scan_d_values(float(d_min), float(d_max), int(d_num))
    chain_mode_eff = str(chain_mode).strip().lower()
    selection_mode = "full_direct" if chain_mode_eff in {"full_direct", "full_direct_runtime"} else "auto"

    g_mode = str(PAPER_BASELINE["g_mode"])
    chi_mode = "localized_grid"
    gamma_mode = "action_grid"
    chi_prof = None
    superrad_prof = None
    g_source = "gn_profile_csv"
    chi_source = "runtime_cell_solver"
    gamma_source = "runtime_cell_solver"

    if chain_mode_eff == "full_direct_runtime":
        ensure_runtime_full_direct_profiles(
            root=ROOT,
            d_scan=d_scan,
            force=bool(runtime_direct_force),
            chi_rho_max=float(runtime_direct_chi_rho_max),
            chi_z_margin=float(runtime_direct_chi_z_margin),
            chi_n_mu=int(runtime_direct_chi_n_mu),
            chi_tol=float(runtime_direct_chi_tol),
            chi_maxiter=int(runtime_direct_chi_maxiter),
            chi_sigma=float(runtime_direct_chi_sigma),
            superrad_zmax=float(runtime_direct_superrad_zmax),
            superrad_ref_d=float(runtime_direct_superrad_ref_d),
            superrad_n_ref=int(runtime_direct_superrad_n_ref),
        )
        chi_prof = select_chi_profile(ROOT, d_scan, selection_mode=selection_mode)
        superrad_prof = select_superrad_profile(ROOT, d_scan, selection_mode=selection_mode)
        chi_mode = str(chi_prof["mode"])
        gamma_mode = str(superrad_prof["mode"])
        chi_source = str(chi_prof["path"])
        gamma_source = str(superrad_prof["path"])
    elif chain_mode_eff == "cell_direct_runtime":
        g_mode = "fp_2d_full_runtime_direct"
        chi_mode = "localized_runtime_direct"
        gamma_mode = "action_runtime_direct"
        g_source = "runtime_cell_solver"
    else:
        chi_prof = select_chi_profile(ROOT, d_scan, selection_mode=selection_mode)
        superrad_prof = select_superrad_profile(ROOT, d_scan, selection_mode=selection_mode)
        chi_mode = str(chi_prof["mode"])
        gamma_mode = str(superrad_prof["mode"])
        chi_source = str(chi_prof["path"])
        gamma_source = str(superrad_prof["path"])

    params = PSLTParameters(
        c_eff=PAPER_BASELINE["c_eff"],
        nu=PAPER_BASELINE["nu"],
        kappa_g=PAPER_BASELINE["kappa_g"],
        g_mode=g_mode,
        g_fp_norm_mode=PAPER_BASELINE["g_fp_norm_mode"],
        g_fp_full_window_blend=PAPER_BASELINE["g_fp_full_window_blend"],
        g_fp_full_tail_beta=PAPER_BASELINE["g_fp_full_tail_beta"],
        g_fp_full_tail_shell_power=PAPER_BASELINE["g_fp_full_tail_shell_power"],
        g_fp_full_tail_clip_min=PAPER_BASELINE["g_fp_full_tail_clip_min"],
        g_fp_full_tail_clip_max=PAPER_BASELINE["g_fp_full_tail_clip_max"],
        runtime_direct_g_rho_max=float(runtime_direct_chi_rho_max),
        runtime_direct_g_z_margin=float(runtime_direct_chi_z_margin),
        runtime_direct_g_n_eigs=40,
        runtime_direct_g_tol=float(runtime_direct_chi_tol),
        runtime_direct_g_maxiter=int(runtime_direct_chi_maxiter),
        runtime_direct_g_sigma=float(runtime_direct_chi_sigma),
        chi=PAPER_BASELINE["chi_legacy"],
        chi_mode=str(chi_mode),
        chi_lr_D=tuple(float(x) for x in (np.asarray(chi_prof["d"], dtype=float) if chi_prof is not None else np.array([6.0, 12.0, 18.0], dtype=float))),
        chi_lr_vals=tuple(float(x) for x in (np.asarray(chi_prof["chi"], dtype=float) if chi_prof is not None else np.array([4.01827e-4, 2.21414e-4, 2.13187e-4], dtype=float))),
        A1=PAPER_BASELINE["A1"],
        A2=PAPER_BASELINE["A2"],
        gamma_mode=str(gamma_mode),
        gamma_superrad_csv=str(superrad_prof["path"]) if superrad_prof is not None else None,
        runtime_direct_use_cache=not bool(runtime_direct_no_cache),
        runtime_direct_chi_rho_max=float(runtime_direct_chi_rho_max),
        runtime_direct_chi_z_margin=float(runtime_direct_chi_z_margin),
        runtime_direct_chi_n_mu=int(runtime_direct_chi_n_mu),
        runtime_direct_chi_tol=float(runtime_direct_chi_tol),
        runtime_direct_chi_maxiter=int(runtime_direct_chi_maxiter),
        runtime_direct_chi_sigma=float(runtime_direct_chi_sigma),
        runtime_direct_superrad_zmax=float(runtime_direct_superrad_zmax),
        runtime_direct_superrad_ref_d=float(runtime_direct_superrad_ref_d),
        runtime_direct_superrad_n_ref=int(runtime_direct_superrad_n_ref),
        b_mode=PAPER_BASELINE["b_mode"],
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=PAPER_BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode=str(observable_mode),
        hll_observable_nmax=PAPER_BASELINE["hll_observable_nmax"],
        hll_uv_blend=float(uv_blend),
        hll_uv_m2_power=float(uv_m2_power),
        hll_uv_match_kappa_diag=float(uv_match_kappa_diag),
        hll_uv_match_kappa_offdiag=float(uv_match_kappa_offdiag),
        hll_uv_rge_mu_low=float(uv_rge_mu_low),
        hll_uv_rge_gamma_diag=float(uv_rge_gamma_diag),
        hll_uv_rge_gamma_offdiag=float(uv_rge_gamma_offdiag),
        hll_uv_rge_log_clip=float(uv_rge_log_clip),
    )
    print(
        "[baseline]",
        f"chain_mode={chain_mode_eff},",
        f"selection_mode={selection_mode},",
        f"g_mode={params.g_mode},",
        f"g_source={g_source},",
        f"chi_mode={params.chi_mode},",
        f"chi_source={chi_source},",
        f"gamma_mode={params.gamma_mode},",
        f"gamma_source={gamma_source}",
        f"runtime_direct_use_cache={params.runtime_direct_use_cache}",
    )
    return PSLTKinetics(params)

def compute_maps(
    kinetics: PSLTKinetics,
    ref_d: float,
    ref_eta: float,
    observable_mode: str,
    d_min: float,
    d_max: float,
    d_num: int,
    eta_min: float,
    eta_max: float,
    eta_num: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    d_vals = np.linspace(float(d_min), float(d_max), int(d_num))
    eta_vals = np.linspace(float(eta_min), float(eta_max), int(eta_num))
    cfg = HLLObservableConfig(
        mode=str(observable_mode),
        t_coh=float(PAPER_BASELINE["t_coh"]),
        ref_D=float(ref_d),
        ref_eta=float(ref_eta),
        n_max=int(PAPER_BASELINE["hll_observable_nmax"]),
    )

    predictors: Dict[str, HLLChannelPredictor] = {
        channel: HLLChannelPredictor(kinetics, layer_n, cfg)
        for channel, layer_n in CHANNEL_TO_LAYER.items()
    }

    ref_amps: Dict[str, float] = {}
    for channel, layer_n in CHANNEL_TO_LAYER.items():
        ref_amps[channel] = predictors[channel].ref_amp

    maps: Dict[str, np.ndarray] = {
        channel: np.zeros((len(eta_vals), len(d_vals)), dtype=float)
        for channel in CHANNEL_TO_LAYER
    }

    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            for channel in CHANNEL_TO_LAYER:
                maps[channel][i, j] = predictors[channel].mu_pred(float(d), float(eta))

    return d_vals, eta_vals, maps, ref_amps


def write_map_csv(
    out_csv: Path,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    maps: Dict[str, np.ndarray],
    observations: Dict[str, Observation],
) -> None:
    fieldnames = ["D", "eta", "mu_ee", "mu_mumu", "mu_tautau", "chi2_mumu"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, eta in enumerate(eta_vals):
            for j, d in enumerate(d_vals):
                row = {
                    "D": float(d),
                    "eta": float(eta),
                    "mu_ee": float(maps["ee"][i, j]),
                    "mu_mumu": float(maps["mumu"][i, j]),
                    "mu_tautau": float(maps["tautau"][i, j]),
                    "chi2_mumu": "",
                }
                if "mumu" in observations:
                    obs = observations["mumu"]
                    mu = float(maps["mumu"][i, j])
                    row["chi2_mumu"] = float(((mu - obs.mu_obs) / obs.sigma_obs) ** 2)
                writer.writerow(row)


def build_summary_rows(
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    maps: Dict[str, np.ndarray],
    observations: Dict[str, Observation],
) -> list[Dict[str, float | str]]:
    rows = []
    for channel, layer_n in CHANNEL_TO_LAYER.items():
        arr2d = maps[channel]
        arr = arr2d.ravel()
        row: Dict[str, float | str] = {
            "channel": channel,
            "layer_N": layer_n,
            "mu_ref": 1.0,
            "mu_min": float(np.min(arr)),
            "mu_p10": float(np.percentile(arr, 10)),
            "mu_p50": float(np.percentile(arr, 50)),
            "mu_p90": float(np.percentile(arr, 90)),
            "mu_max": float(np.max(arr)),
            "mu_mean": float(np.mean(arr)),
            "mu_std": float(np.std(arr)),
            "f_mu_gt_0p8": float(np.mean(arr > 0.8)),
            "f_mu_gt_1p0": float(np.mean(arr > 1.0)),
            "f_mu_gt_1p2": float(np.mean(arr > 1.2)),
            "obs_mu": float("nan"),
            "obs_sigma": float("nan"),
            "f_chi2_le_1": float("nan"),
            "f_chi2_le_4": float("nan"),
            "best_chi2": float("nan"),
            "best_D": float("nan"),
            "best_eta": float("nan"),
            "obs_source": "",
        }

        if channel in observations:
            obs = observations[channel]
            chi2 = ((arr2d - obs.mu_obs) / obs.sigma_obs) ** 2
            best_idx = np.unravel_index(np.argmin(chi2), chi2.shape)
            row.update(
                {
                    "obs_mu": float(obs.mu_obs),
                    "obs_sigma": float(obs.sigma_obs),
                    "f_chi2_le_1": float(np.mean(chi2 <= 1.0)),
                    "f_chi2_le_4": float(np.mean(chi2 <= 4.0)),
                    "best_chi2": float(np.min(chi2)),
                    "best_D": float(d_vals[best_idx[1]]),
                    "best_eta": float(eta_vals[best_idx[0]]),
                    "obs_source": obs.source,
                }
            )

        rows.append(row)
    return rows


def write_summary_csv(path: Path, rows: list[Dict[str, float | str]]) -> None:
    fieldnames = [
        "channel",
        "layer_N",
        "mu_ref",
        "mu_min",
        "mu_p10",
        "mu_p50",
        "mu_p90",
        "mu_max",
        "mu_mean",
        "mu_std",
        "f_mu_gt_0p8",
        "f_mu_gt_1p0",
        "f_mu_gt_1p2",
        "obs_mu",
        "obs_sigma",
        "f_chi2_le_1",
        "f_chi2_le_4",
        "best_chi2",
        "best_D",
        "best_eta",
        "obs_source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_maps(
    out_png: Path,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    maps: Dict[str, np.ndarray],
    observations: Dict[str, Observation],
    observable_mode: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    extent = [float(d_vals.min()), float(d_vals.max()), float(eta_vals.min()), float(eta_vals.max())]

    for ax, channel in zip(axes, ("ee", "mumu", "tautau")):
        arr = maps[channel]
        p05, p95 = np.percentile(arr, [5.0, 95.0])
        if p95 <= p05:
            p05 = float(np.min(arr))
            p95 = float(np.max(arr)) + 1e-12

        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=float(p05),
            vmax=float(p95),
        )
        if float(np.min(arr)) <= 1.0 <= float(np.max(arr)):
            ax.contour(d_vals, eta_vals, arr, levels=[1.0], colors="white", linestyles="--", linewidths=1.5)
        if channel in observations:
            obs_mu = float(observations[channel].mu_obs)
            if float(np.min(arr)) <= obs_mu <= float(np.max(arr)):
                ax.contour(d_vals, eta_vals, arr, levels=[obs_mu], colors="orange", linewidths=1.2)

        ax.set_title(CHANNEL_TITLE[channel])
        ax.set_xlabel("D")
        ax.set_ylabel("eta")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\mu_{\rm pred}$")

    fig.suptitle(f"PSLT signal-strength maps by lepton channel ({observable_mode})", fontsize=13)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Map-level PSLT predictions for H->ll channels.")
    ap.add_argument("--chain-mode", choices=["auto", "full_direct", "full_direct_runtime", "cell_direct_runtime"], default="full_direct")
    ap.add_argument("--ref-mode", choices=["fixed", "chi2_best", "robust_center"], default="fixed")
    ap.add_argument("--ref-d", type=float, default=float(PAPER_BASELINE["ref_D"]))
    ap.add_argument("--ref-eta", type=float, default=float(PAPER_BASELINE["ref_eta"]))
    ap.add_argument("--ref-choice-json", type=str, default=str(DEFAULT_REF_CHOICE_JSON))
    ap.add_argument(
        "--observable-mode",
        choices=["proxy_wratio", "eft_wilson_diag", "eft_wilson_uv_tree", "eft_wilson_uv_rge"],
        default=str(PAPER_BASELINE["hll_observable_mode"]),
    )
    ap.add_argument("--uv-blend", type=float, default=float(PAPER_BASELINE["hll_uv_blend"]))
    ap.add_argument("--uv-m2-power", type=float, default=float(PAPER_BASELINE["hll_uv_m2_power"]))
    ap.add_argument("--uv-match-kappa-diag", type=float, default=float(PAPER_BASELINE["hll_uv_match_kappa_diag"]))
    ap.add_argument("--uv-match-kappa-offdiag", type=float, default=float(PAPER_BASELINE["hll_uv_match_kappa_offdiag"]))
    ap.add_argument("--uv-rge-mu-low", type=float, default=float(PAPER_BASELINE["hll_uv_rge_mu_low"]))
    ap.add_argument("--uv-rge-gamma-diag", type=float, default=float(PAPER_BASELINE["hll_uv_rge_gamma_diag"]))
    ap.add_argument("--uv-rge-gamma-offdiag", type=float, default=float(PAPER_BASELINE["hll_uv_rge_gamma_offdiag"]))
    ap.add_argument("--uv-rge-log-clip", type=float, default=float(PAPER_BASELINE["hll_uv_rge_log_clip"]))
    ap.add_argument("--d-min", type=float, default=float(PAPER_BASELINE["D_min"]))
    ap.add_argument("--d-max", type=float, default=float(PAPER_BASELINE["D_max"]))
    ap.add_argument("--d-num", type=int, default=int(PAPER_BASELINE["D_num"]))
    ap.add_argument("--eta-min", type=float, default=float(PAPER_BASELINE["eta_min"]))
    ap.add_argument("--eta-max", type=float, default=float(PAPER_BASELINE["eta_max"]))
    ap.add_argument("--eta-num", type=int, default=int(PAPER_BASELINE["eta_num"]))
    ap.add_argument("--runtime-direct-force", action="store_true")
    ap.add_argument("--runtime-direct-no-cache", action="store_true")
    ap.add_argument("--runtime-direct-chi-rho-max", type=float, default=3.0)
    ap.add_argument("--runtime-direct-chi-z-margin", type=float, default=6.0)
    ap.add_argument("--runtime-direct-chi-n-mu", type=int, default=120)
    ap.add_argument("--runtime-direct-chi-tol", type=float, default=1e-8)
    ap.add_argument("--runtime-direct-chi-maxiter", type=int, default=30000)
    ap.add_argument("--runtime-direct-chi-sigma", type=float, default=2.5)
    ap.add_argument("--runtime-direct-superrad-zmax", type=float, default=80.0)
    ap.add_argument("--runtime-direct-superrad-ref-d", type=float, default=12.0)
    ap.add_argument("--runtime-direct-superrad-n-ref", type=int, default=2)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--skip-paper-copy", action="store_true")
    return ap.parse_args()


def sanitize_tag(tag: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(tag).strip())


def build_suffix(ref_mode: str, ref_d: float, ref_eta: float, tag: str) -> str:
    clean_tag = sanitize_tag(tag)
    if clean_tag:
        return f"_{clean_tag}"

    baseline_ref = (
        ref_mode == "fixed"
        and np.isclose(float(ref_d), float(PAPER_BASELINE["ref_D"]))
        and np.isclose(float(ref_eta), float(PAPER_BASELINE["ref_eta"]))
    )
    if baseline_ref:
        return ""

    if ref_mode == "fixed":
        d_token = str(float(ref_d)).replace("-", "m").replace(".", "p")
        e_token = str(float(ref_eta)).replace("-", "m").replace(".", "p")
        return f"_refD{d_token}_eta{e_token}"
    return f"_{ref_mode}"


def _load_anchor_from_json(path: Path, mode: str) -> tuple[float, float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        selected = payload.get("selected_by_mode", {}).get(mode, {})
        if not isinstance(selected, dict):
            return None
        return float(selected["D"]), float(selected["eta"])
    except Exception:
        return None


def resolve_reference_anchor(
    args: argparse.Namespace,
    kinetics: PSLTKinetics,
    observations: Dict[str, Observation],
    observable_mode: str,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
) -> tuple[float, float, str]:
    mode = str(args.ref_mode)
    if mode == "fixed":
        return float(args.ref_d), float(args.ref_eta), "fixed_cli"

    json_path = Path(str(args.ref_choice_json))
    loaded = _load_anchor_from_json(json_path, mode)
    if loaded is not None:
        return float(loaded[0]), float(loaded[1]), f"choice_json:{json_path.name}"

    obs = observations.get(
        "mumu",
        Observation(mu_obs=1.4, sigma_obs=0.4, source="fallback default (ATLAS Run-3 proxy)"),
    )

    candidates = select_anchor_candidates_from_fixed_scan(
        kinetics=kinetics,
        d_vals=d_vals,
        eta_vals=eta_vals,
        mode=str(observable_mode),
        t_coh=float(PAPER_BASELINE["t_coh"]),
        n_max=int(PAPER_BASELINE["hll_observable_nmax"]),
        mu_obs=float(obs.mu_obs),
        sigma_obs=float(obs.sigma_obs),
        fixed_ref_d=float(args.ref_d),
        fixed_ref_eta=float(args.ref_eta),
    )
    row = candidates[mode]
    return float(row["ref_D"]), float(row["ref_eta"]), "selector_fallback"


def snap_ref_d_for_full_direct(chain_mode: str, ref_d: float, d_vals: np.ndarray) -> tuple[float, bool]:
    if str(chain_mode) not in {"full_direct", "full_direct_runtime"}:
        return float(ref_d), False
    if len(d_vals) == 0:
        return float(ref_d), False
    arr = np.asarray(d_vals, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(ref_d))))
    snapped = float(arr[idx])
    changed = not np.isclose(snapped, float(ref_d), rtol=0.0, atol=1e-10)
    return snapped, changed


def main() -> None:
    args = parse_args()
    if int(args.d_num) < 2 or int(args.eta_num) < 2:
        raise ValueError("--d-num and --eta-num must be >= 2.")
    if not (0.0 <= float(args.uv_blend) <= 1.0):
        raise ValueError("--uv-blend must be in [0,1].")
    if float(args.uv_m2_power) < 0.0:
        raise ValueError("--uv-m2-power must be >= 0.")
    if float(args.uv_rge_mu_low) <= 0.0:
        raise ValueError("--uv-rge-mu-low must be > 0.")
    if float(args.uv_rge_log_clip) <= 0.0:
        raise ValueError("--uv-rge-log-clip must be > 0.")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    observations = load_observations()
    observable_mode = str(args.observable_mode)
    kinetics = make_baseline_kinetics(
        observable_mode=observable_mode,
        chain_mode=str(args.chain_mode),
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        uv_blend=float(args.uv_blend),
        uv_m2_power=float(args.uv_m2_power),
        uv_match_kappa_diag=float(args.uv_match_kappa_diag),
        uv_match_kappa_offdiag=float(args.uv_match_kappa_offdiag),
        uv_rge_mu_low=float(args.uv_rge_mu_low),
        uv_rge_gamma_diag=float(args.uv_rge_gamma_diag),
        uv_rge_gamma_offdiag=float(args.uv_rge_gamma_offdiag),
        uv_rge_log_clip=float(args.uv_rge_log_clip),
        runtime_direct_force=bool(args.runtime_direct_force),
        runtime_direct_no_cache=bool(args.runtime_direct_no_cache),
        runtime_direct_chi_rho_max=float(args.runtime_direct_chi_rho_max),
        runtime_direct_chi_z_margin=float(args.runtime_direct_chi_z_margin),
        runtime_direct_chi_n_mu=int(args.runtime_direct_chi_n_mu),
        runtime_direct_chi_tol=float(args.runtime_direct_chi_tol),
        runtime_direct_chi_maxiter=int(args.runtime_direct_chi_maxiter),
        runtime_direct_chi_sigma=float(args.runtime_direct_chi_sigma),
        runtime_direct_superrad_zmax=float(args.runtime_direct_superrad_zmax),
        runtime_direct_superrad_ref_d=float(args.runtime_direct_superrad_ref_d),
        runtime_direct_superrad_n_ref=int(args.runtime_direct_superrad_n_ref),
    )
    d_vals_grid = np.linspace(float(args.d_min), float(args.d_max), int(args.d_num))
    eta_vals_grid = np.linspace(float(args.eta_min), float(args.eta_max), int(args.eta_num))
    ref_d, ref_eta, ref_source = resolve_reference_anchor(
        args,
        kinetics,
        observations,
        observable_mode=observable_mode,
        d_vals=d_vals_grid,
        eta_vals=eta_vals_grid,
    )
    ref_d_before_snap = float(ref_d)
    ref_d, snapped = snap_ref_d_for_full_direct(str(args.chain_mode), float(ref_d), d_vals_grid)
    if snapped:
        old_ref_d = float(args.ref_d) if str(args.ref_mode) == "fixed" else float("nan")
        print(
            "[info] chain_mode in {full_direct,full_direct_runtime} snapped ref_D to grid:",
            f"{old_ref_d if old_ref_d == old_ref_d else 'selector'} -> {ref_d:.6g}",
        )
        ref_source = f"{ref_source}+snap_refD_to_grid"
    suffix = build_suffix(ref_mode=str(args.ref_mode), ref_d=ref_d, ref_eta=ref_eta, tag=str(args.tag))

    d_vals, eta_vals, maps, ref_weights = compute_maps(
        kinetics,
        ref_d=ref_d,
        ref_eta=ref_eta,
        observable_mode=observable_mode,
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        eta_min=float(args.eta_min),
        eta_max=float(args.eta_max),
        eta_num=int(args.eta_num),
    )
    print(
        "[info] observable mode:",
        observable_mode,
        f"| uv_blend={float(args.uv_blend):.3f}",
        f"| uv_m2_power={float(args.uv_m2_power):.3f}",
        f"| finite_match(kappa_diag={float(args.uv_match_kappa_diag):.3f},",
        f"kappa_offdiag={float(args.uv_match_kappa_offdiag):.3f})",
        f"| uv_rge(mu_low={float(args.uv_rge_mu_low):.3f},",
        f"gamma_diag={float(args.uv_rge_gamma_diag):.3f},",
        f"gamma_offdiag={float(args.uv_rge_gamma_offdiag):.3f})",
        f"| chain_mode={str(args.chain_mode)}",
        f"| reference (D={ref_d:.6g}, eta={ref_eta:.6g}, source={ref_source}) amplitudes:",
        ref_weights,
    )

    out_map = OUTDIR / f"hll_signal_strength_map{suffix}.csv"
    write_map_csv(out_map, d_vals, eta_vals, maps, observations)

    summary_rows = build_summary_rows(d_vals, eta_vals, maps, observations)
    out_summary = OUTDIR / f"hll_signal_strength_summary{suffix}.csv"
    write_summary_csv(out_summary, summary_rows)

    out_fig = OUTDIR / f"hll_signal_strength_maps{suffix}.png"
    plot_maps(out_fig, d_vals, eta_vals, maps, observations, observable_mode=observable_mode)

    run_meta = {
        "ref_mode": str(args.ref_mode),
        "ref_D": float(ref_d),
        "ref_D_input": float(ref_d_before_snap),
        "ref_D_snapped_to_grid": bool(snapped),
        "ref_eta": float(ref_eta),
        "ref_source": ref_source,
        "suffix": suffix,
        "observable_mode": observable_mode,
        "uv_blend": float(args.uv_blend),
        "uv_m2_power": float(args.uv_m2_power),
        "uv_match_kappa_diag": float(args.uv_match_kappa_diag),
        "uv_match_kappa_offdiag": float(args.uv_match_kappa_offdiag),
        "uv_rge_mu_low": float(args.uv_rge_mu_low),
        "uv_rge_gamma_diag": float(args.uv_rge_gamma_diag),
        "uv_rge_gamma_offdiag": float(args.uv_rge_gamma_offdiag),
        "uv_rge_log_clip": float(args.uv_rge_log_clip),
        "chain_mode": str(args.chain_mode),
        "runtime_direct_force": bool(args.runtime_direct_force),
        "runtime_direct_no_cache": bool(args.runtime_direct_no_cache),
        "runtime_direct_chi_rho_max": float(args.runtime_direct_chi_rho_max),
        "runtime_direct_chi_z_margin": float(args.runtime_direct_chi_z_margin),
        "runtime_direct_chi_n_mu": int(args.runtime_direct_chi_n_mu),
        "runtime_direct_chi_tol": float(args.runtime_direct_chi_tol),
        "runtime_direct_chi_maxiter": int(args.runtime_direct_chi_maxiter),
        "runtime_direct_chi_sigma": float(args.runtime_direct_chi_sigma),
        "runtime_direct_superrad_zmax": float(args.runtime_direct_superrad_zmax),
        "runtime_direct_superrad_ref_d": float(args.runtime_direct_superrad_ref_d),
        "runtime_direct_superrad_n_ref": int(args.runtime_direct_superrad_n_ref),
        "d_min": float(args.d_min),
        "d_max": float(args.d_max),
        "d_num": int(args.d_num),
        "eta_min": float(args.eta_min),
        "eta_max": float(args.eta_max),
        "eta_num": int(args.eta_num),
        "tag": str(args.tag),
    }
    out_meta = OUTDIR / f"hll_signal_strength_run_meta{suffix or '_baseline'}.json"
    out_meta.write_text(json.dumps(run_meta, indent=2))

    paper_summary = PAPER_DIR / out_summary.name
    paper_fig = PAPER_DIR / out_fig.name
    paper_meta = PAPER_DIR / out_meta.name
    if not args.skip_paper_copy:
        paper_summary.write_text(out_summary.read_text())
        paper_fig.write_bytes(out_fig.read_bytes())
        paper_meta.write_text(out_meta.read_text())

    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")
    if not args.skip_paper_copy:
        print(f"[saved] {paper_summary}")
        print(f"[saved] {paper_fig}")
        print(f"[saved] {paper_meta}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()

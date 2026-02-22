#!/usr/bin/env python3
"""
Map-level PSLT proxy predictions for H->ll channels (ee, mumu, tautau).

The proxy follows the paper's H->mumu definition:
  W_N(D,eta) = B_N * g_N(D) * (1 - exp(-Gamma_N(D,eta) * t_coh))
  mu_ll_pred = W_N / W_N_ref

with layer-channel assignment:
  ee -> N=1, mumu -> N=2, tautau -> N=3.

Outputs:
  - output/hll_signal_strength/hll_signal_strength_map.csv
  - output/hll_signal_strength/hll_signal_strength_summary.csv
  - output/hll_signal_strength/hll_signal_strength_maps.png
  - paper/hll_signal_strength_summary.csv
  - paper/hll_signal_strength_maps.png
"""

from __future__ import annotations

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

from pslt_lib import PSLTKinetics, PSLTParameters  # noqa: E402


OUTDIR = ROOT / "output" / "hll_signal_strength"
PAPER_DIR = ROOT / "paper"

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
    "g_fp_full_window_blend": 0.8,
    "g_fp_full_tail_beta": 1.1,
    "g_fp_full_tail_shell_power": 0.0,
    "g_fp_full_tail_clip_min": 1e-3,
    "g_fp_full_tail_clip_max": 0.95,
    "chi_legacy": 0.2,
    "chi_mode": "localized_interp",
    "A1": 1.0,
    "A2": 1.0,
    "p_B": 0.30,
    "b_mode": "overlap_2d",
    "t_coh": 1.0,
    "ref_D": 10.0,
    "ref_eta": 1.0,
    "D_min": 4.0,
    "D_max": 20.0,
    "D_num": 60,
    "eta_min": 0.2,
    "eta_max": 4.0,
    "eta_num": 60,
}

DEFAULT_CHI_D = np.array([6.0, 12.0, 18.0], dtype=float)
DEFAULT_CHI_VALS = np.array([4.01827e-4, 2.21414e-4, 2.13187e-4], dtype=float)
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


@dataclass(frozen=True)
class Observation:
    mu_obs: float
    sigma_obs: float
    source: str


def load_localized_chi_profile() -> Tuple[np.ndarray, np.ndarray]:
    path = ROOT / "output" / "chi_fp_2d" / "localized_chi_D6-12-18.csv"
    if not path.exists():
        print(f"[warn] missing {path}; using built-in chi(D) knots")
        return DEFAULT_CHI_D.copy(), DEFAULT_CHI_VALS.copy()

    rows = []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("level", "").strip().lower() == "fine":
                    rows.append((float(row["D"]), float(row["chi_LR"])))
    except Exception as exc:
        print(f"[warn] failed to parse {path}: {exc}; using built-in chi(D) knots")
        return DEFAULT_CHI_D.copy(), DEFAULT_CHI_VALS.copy()

    if len(rows) < 2:
        print(f"[warn] not enough fine rows in {path}; using built-in chi(D) knots")
        return DEFAULT_CHI_D.copy(), DEFAULT_CHI_VALS.copy()

    rows.sort(key=lambda t: t[0])
    dvals = np.array([r[0] for r in rows], dtype=float)
    chis = np.array([r[1] for r in rows], dtype=float)
    return dvals, chis


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


def make_baseline_kinetics() -> PSLTKinetics:
    chi_d, chi_vals = load_localized_chi_profile()
    params = PSLTParameters(
        c_eff=PAPER_BASELINE["c_eff"],
        nu=PAPER_BASELINE["nu"],
        kappa_g=PAPER_BASELINE["kappa_g"],
        g_mode=PAPER_BASELINE["g_mode"],
        g_fp_full_window_blend=PAPER_BASELINE["g_fp_full_window_blend"],
        g_fp_full_tail_beta=PAPER_BASELINE["g_fp_full_tail_beta"],
        g_fp_full_tail_shell_power=PAPER_BASELINE["g_fp_full_tail_shell_power"],
        g_fp_full_tail_clip_min=PAPER_BASELINE["g_fp_full_tail_clip_min"],
        g_fp_full_tail_clip_max=PAPER_BASELINE["g_fp_full_tail_clip_max"],
        chi=PAPER_BASELINE["chi_legacy"],
        chi_mode=PAPER_BASELINE["chi_mode"],
        chi_lr_D=tuple(float(x) for x in chi_d),
        chi_lr_vals=tuple(float(x) for x in chi_vals),
        A1=PAPER_BASELINE["A1"],
        A2=PAPER_BASELINE["A2"],
        b_mode=PAPER_BASELINE["b_mode"],
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=PAPER_BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def layer_weight(
    kinetics: PSLTKinetics,
    layer_n: int,
    d_val: float,
    eta_val: float,
    t_coh: float,
) -> float:
    gamma_n = kinetics.calculate_gamma_N(layer_n, d_val, eta_val)
    g_n = kinetics.g_N_effective(layer_n, d_val)
    b_n = kinetics.B_N(layer_n, d_val)
    return float(b_n * g_n * (1.0 - np.exp(-gamma_n * t_coh)))


def compute_maps(
    kinetics: PSLTKinetics,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    d_vals = np.linspace(PAPER_BASELINE["D_min"], PAPER_BASELINE["D_max"], PAPER_BASELINE["D_num"])
    eta_vals = np.linspace(PAPER_BASELINE["eta_min"], PAPER_BASELINE["eta_max"], PAPER_BASELINE["eta_num"])
    t_coh = float(PAPER_BASELINE["t_coh"])
    d_ref = float(PAPER_BASELINE["ref_D"])
    eta_ref = float(PAPER_BASELINE["ref_eta"])

    ref_weights: Dict[str, float] = {}
    for channel, layer_n in CHANNEL_TO_LAYER.items():
        w_ref = layer_weight(kinetics, layer_n, d_ref, eta_ref, t_coh)
        if w_ref <= 0.0:
            raise RuntimeError(f"Non-positive reference weight for channel={channel}.")
        ref_weights[channel] = w_ref

    maps: Dict[str, np.ndarray] = {
        channel: np.zeros((len(eta_vals), len(d_vals)), dtype=float)
        for channel in CHANNEL_TO_LAYER
    }

    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            for channel, layer_n in CHANNEL_TO_LAYER.items():
                w_val = layer_weight(kinetics, layer_n, float(d), float(eta), t_coh)
                maps[channel][i, j] = w_val / ref_weights[channel]

    return d_vals, eta_vals, maps, ref_weights


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

    fig.suptitle("PSLT proxy signal-strength maps by lepton channel", fontsize=13)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    observations = load_observations()
    kinetics = make_baseline_kinetics()

    d_vals, eta_vals, maps, ref_weights = compute_maps(kinetics)
    print("[info] reference weights (D=10, eta=1):", ref_weights)

    out_map = OUTDIR / "hll_signal_strength_map.csv"
    write_map_csv(out_map, d_vals, eta_vals, maps, observations)

    summary_rows = build_summary_rows(d_vals, eta_vals, maps, observations)
    out_summary = OUTDIR / "hll_signal_strength_summary.csv"
    write_summary_csv(out_summary, summary_rows)

    out_fig = OUTDIR / "hll_signal_strength_maps.png"
    plot_maps(out_fig, d_vals, eta_vals, maps, observations)

    paper_summary = PAPER_DIR / "hll_signal_strength_summary.csv"
    paper_fig = PAPER_DIR / "hll_signal_strength_maps.png"
    paper_summary.write_text(out_summary.read_text())
    paper_fig.write_bytes(out_fig.read_bytes())

    print(f"[saved] {out_map}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {paper_summary}")
    print(f"[saved] {paper_fig}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()

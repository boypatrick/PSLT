#!/usr/bin/env python3
"""
Systematic sensitivity scan over reference-normalization points (D0, eta0).

Outputs:
  - output/hll_reference_anchor/reference_anchor_sensitivity.csv
  - output/hll_reference_anchor/reference_anchor_sensitivity.png
  - paper/reference_anchor_sensitivity.csv
  - paper/reference_anchor_sensitivity.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hll_observable import HLLObservableConfig, HLLChannelPredictor
from pslt_lib import PSLTKinetics, PSLTParameters
from reference_anchor_utils import compute_r3_and_winner_maps, select_anchor_candidates_from_fixed_scan
from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "hll_reference_anchor"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"
CHOICE_JSON = OUTDIR / "reference_anchor_choice.json"

BASELINE = {
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
    "b_mode": "eft_operator_norm",
    "hll_observable_mode": "eft_wilson_uv_rge",
    "hll_observable_nmax": 20,
    "t_coh": 1.0,
    "mu_obs": 1.4,
    "sigma_obs": 0.4,
    "fixed_ref_D": 10.0,
    "fixed_ref_eta": 1.0,
    "D_min": 4.0,
    "D_max": 20.0,
    "D_num": 60,
    "eta_min": 0.2,
    "eta_max": 4.0,
    "eta_num": 60,
}

def make_kinetics(chi_profile: Dict[str, object], superrad_profile: Dict[str, object]) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=BASELINE["c_eff"],
        nu=BASELINE["nu"],
        kappa_g=BASELINE["kappa_g"],
        g_mode=BASELINE["g_mode"],
        g_fp_norm_mode=BASELINE["g_fp_norm_mode"],
        g_fp_full_window_blend=BASELINE["g_fp_full_window_blend"],
        g_fp_full_tail_beta=BASELINE["g_fp_full_tail_beta"],
        g_fp_full_tail_shell_power=BASELINE["g_fp_full_tail_shell_power"],
        g_fp_full_tail_clip_min=BASELINE["g_fp_full_tail_clip_min"],
        g_fp_full_tail_clip_max=BASELINE["g_fp_full_tail_clip_max"],
        chi=BASELINE["chi_legacy"],
        chi_mode=str(chi_profile["mode"]),
        chi_lr_D=tuple(float(x) for x in chi_profile["d"]),
        chi_lr_vals=tuple(float(y) for y in chi_profile["chi"]),
        A1=BASELINE["A1"],
        A2=BASELINE["A2"],
        gamma_mode=str(superrad_profile["mode"]),
        gamma_superrad_csv=str(superrad_profile["path"]),
        b_mode=BASELINE["b_mode"],
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode=BASELINE["hll_observable_mode"],
        hll_observable_nmax=BASELINE["hll_observable_nmax"],
    )
    return PSLTKinetics(params)


def evaluate_anchor(
    kinetics: PSLTKinetics,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    ref_d: float,
    ref_eta: float,
    r3_map: np.ndarray,
    winner_map: np.ndarray,
) -> Dict[str, float]:
    cfg = HLLObservableConfig(
        mode=BASELINE["hll_observable_mode"],
        t_coh=float(BASELINE["t_coh"]),
        ref_D=float(ref_d),
        ref_eta=float(ref_eta),
        n_max=int(BASELINE["hll_observable_nmax"]),
    )
    predictor = HLLChannelPredictor(kinetics, layer_n=2, cfg=cfg)

    chi2 = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            mu_pred = float(predictor.mu_pred(float(d), float(eta)))
            chi2[i, j] = ((mu_pred - BASELINE["mu_obs"]) / BASELINE["sigma_obs"]) ** 2

    best_idx = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    return {
        "ref_D": float(ref_d),
        "ref_eta": float(ref_eta),
        "f_R3_gt_0p90": float(np.mean(r3_map >= 0.90)),
        "f_winner_gt_3": float(np.mean(winner_map > 3.0)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2 <= 4.0)),
        "best_chi2": float(np.min(chi2)),
        "best_D": float(d_vals[int(best_idx[1])]),
        "best_eta": float(eta_vals[int(best_idx[0])]),
    }


def load_selector_points() -> Dict[str, Tuple[float, float]]:
    if not CHOICE_JSON.exists():
        return {}
    try:
        payload = json.loads(CHOICE_JSON.read_text())
        sel = payload.get("selected_by_mode", {})
        out: Dict[str, Tuple[float, float]] = {}
        for key in ("fixed", "chi2_best", "robust_center"):
            row = sel.get(key)
            if not isinstance(row, dict):
                continue
            out[key] = (float(row["D"]), float(row["eta"]))
        return out
    except Exception:
        return {}


def build_grid_points() -> List[Tuple[float, float, str]]:
    d0_vals = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype=float)
    eta0_vals = np.array([0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4], dtype=float)
    pts = [(float(d0), float(e0), "grid") for e0 in eta0_vals for d0 in d0_vals]
    pts.append((float(BASELINE["fixed_ref_D"]), float(BASELINE["fixed_ref_eta"]), "baseline_fixed"))
    return pts


def dedup_points(points: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    seen = set()
    out: List[Tuple[float, float, str]] = []
    for d, e, label in points:
        key = (round(float(d), 6), round(float(e), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(d), float(e), label))
    return out


def make_plot(rows: List[Dict[str, float | str]], out_png: Path) -> None:
    grid_rows = [r for r in rows if str(r["anchor_source"]).startswith("grid")]
    if not grid_rows:
        return

    d_unique = sorted({float(r["ref_D"]) for r in grid_rows})
    eta_unique = sorted({float(r["ref_eta"]) for r in grid_rows})
    mat_frac = np.full((len(eta_unique), len(d_unique)), np.nan, dtype=float)
    mat_best = np.full_like(mat_frac, np.nan)

    d_index = {d: j for j, d in enumerate(d_unique)}
    e_index = {e: i for i, e in enumerate(eta_unique)}

    for r in grid_rows:
        i = e_index[float(r["ref_eta"])]
        j = d_index[float(r["ref_D"])]
        mat_frac[i, j] = float(r["f_hmumu_chi2_le_4"])
        mat_best[i, j] = float(r["best_chi2"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    extent = [min(d_unique), max(d_unique), min(eta_unique), max(eta_unique)]

    im0 = axes[0].imshow(mat_frac, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].set_title(r"$f(\chi^2_{\mu\mu}<4)$ vs reference-normalization point")
    axes[0].set_xlabel(r"reference $D_0$")
    axes[0].set_ylabel(r"reference $\eta_0$")
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.set_label("accepted fraction")

    im1 = axes[1].imshow(mat_best, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[1].set_title(r"best $\chi^2$ vs reference-normalization point")
    axes[1].set_xlabel(r"reference $D_0$")
    axes[1].set_ylabel(r"reference $\eta_0$")
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.set_label(r"best $\chi^2$")

    # Overlay selector points if present.
    for r in rows:
        src = str(r["anchor_source"])
        if src.startswith("selector_") or src == "baseline_fixed":
            d0 = float(r["ref_D"])
            e0 = float(r["ref_eta"])
            if src == "selector_chi2_best":
                marker, color = "x", "cyan"
            elif src == "selector_robust_center":
                marker, color = "*", "white"
            else:
                marker, color = "o", "red"
            axes[0].scatter([d0], [e0], s=80, marker=marker, c=color, edgecolors="black", linewidths=0.6)
            axes[1].scatter([d0], [e0], s=80, marker=marker, c=color, edgecolors="black", linewidths=0.6)

    fig.suptitle("Reference-normalization sensitivity under EFT/Wilson-matched map", fontsize=12.5)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Systematic sensitivity scan over reference-normalization points.")
    ap.add_argument("--fixed-ref-d", type=float, default=10.0)
    ap.add_argument("--fixed-ref-eta", type=float, default=1.0)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_vals = scan_d_values(BASELINE["D_min"], BASELINE["D_max"], BASELINE["D_num"])
    chi_profile = select_chi_profile(ROOT, d_vals)
    superrad_profile = select_superrad_profile(ROOT, d_vals)
    kinetics = make_kinetics(chi_profile, superrad_profile)
    eta_vals = np.linspace(BASELINE["eta_min"], BASELINE["eta_max"], BASELINE["eta_num"])

    r3_map, winner_map = compute_r3_and_winner_maps(
        kinetics,
        d_vals=d_vals,
        eta_vals=eta_vals,
        t_coh=float(BASELINE["t_coh"]),
        n_max=int(BASELINE["hll_observable_nmax"]),
    )

    # Selector points: prefer existing json; if missing, recompute directly.
    selector = load_selector_points()
    if not selector:
        cand = select_anchor_candidates_from_fixed_scan(
            kinetics=kinetics,
            d_vals=d_vals,
            eta_vals=eta_vals,
            mode=BASELINE["hll_observable_mode"],
            t_coh=float(BASELINE["t_coh"]),
            n_max=int(BASELINE["hll_observable_nmax"]),
            mu_obs=float(BASELINE["mu_obs"]),
            sigma_obs=float(BASELINE["sigma_obs"]),
            fixed_ref_d=float(args.fixed_ref_d),
            fixed_ref_eta=float(args.fixed_ref_eta),
        )
        selector = {
            "fixed": (float(cand["fixed"]["ref_D"]), float(cand["fixed"]["ref_eta"])),
            "chi2_best": (float(cand["chi2_best"]["ref_D"]), float(cand["chi2_best"]["ref_eta"])),
            "robust_center": (float(cand["robust_center"]["ref_D"]), float(cand["robust_center"]["ref_eta"])),
        }

    points = build_grid_points()
    for mode in ("fixed", "chi2_best", "robust_center"):
        if mode in selector:
            points.append((selector[mode][0], selector[mode][1], f"selector_{mode}"))
    points = dedup_points(points)

    rows: List[Dict[str, float | str]] = []
    for ref_d, ref_eta, source in points:
        row = evaluate_anchor(
            kinetics=kinetics,
            d_vals=d_vals,
            eta_vals=eta_vals,
            ref_d=ref_d,
            ref_eta=ref_eta,
            r3_map=r3_map,
            winner_map=winner_map,
        )
        row["anchor_source"] = source
        rows.append(row)

    # Baseline drift columns.
    baseline_row = None
    for r in rows:
        if np.isclose(float(r["ref_D"]), BASELINE["fixed_ref_D"]) and np.isclose(float(r["ref_eta"]), BASELINE["fixed_ref_eta"]):
            baseline_row = r
            break
    if baseline_row is None:
        baseline_row = rows[0]

    base_f = float(baseline_row["f_hmumu_chi2_le_4"])
    base_best = float(baseline_row["best_chi2"])
    for r in rows:
        r["delta_f_hmumu_chi2_le_4_vs_baseline"] = float(r["f_hmumu_chi2_le_4"]) - base_f
        r["delta_best_chi2_vs_baseline"] = float(r["best_chi2"]) - base_best

    # Sort: selector points first, then grid.
    def _sort_key(r: Dict[str, float | str]) -> Tuple[int, float, float]:
        src = str(r["anchor_source"])
        prio = 1
        if src == "baseline_fixed":
            prio = 0
        elif src.startswith("selector_"):
            prio = 0
        return prio, float(r["ref_eta"]), float(r["ref_D"])

    rows = sorted(rows, key=_sort_key)

    out_csv = OUTDIR / "reference_anchor_sensitivity.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_png = OUTDIR / "reference_anchor_sensitivity.png"
    make_plot(rows, out_png)

    paper_csv = PAPER_DIR / out_csv.name
    paper_png = PAPER_DIR / out_png.name
    paper_csv.write_text(out_csv.read_text())
    paper_png.write_bytes(out_png.read_bytes())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_png}")
    print(f"[saved] {paper_csv}")
    print(f"[saved] {paper_png}")
    print(f"[info] baseline f(chi2<4) = {base_f:.6f}")


if __name__ == "__main__":
    main()

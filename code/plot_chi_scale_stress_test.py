#!/usr/bin/env python3
"""
Strong chi-scale stress test:
- quantifies when global-map boundaries start moving under multiplicative chi scaling
- produces one summary figure and one csv table

Outputs:
  - output/chi_fp_2d/chi_scale_stress_test.csv
  - output/chi_fp_2d/chi_scale_stress_test.png
  - paper/chi_scale_stress_test.png
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pslt_lib import PSLTKinetics, PSLTParameters


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_fp_2d"
PAPER_DIR = ROOT / "paper"


def load_fine_knots() -> Tuple[np.ndarray, np.ndarray]:
    path = OUTDIR / "localized_chi_D6-12-18.csv"
    rows: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level", "").strip().lower() == "fine":
                rows.append((float(row["D"]), float(row["chi_LR"])))
    rows.sort(key=lambda t: t[0])
    d = np.array([x for x, _ in rows], dtype=float)
    chi = np.array([y for _, y in rows], dtype=float)
    return d, chi


def make_kinetics(d_knots: np.ndarray, chi_knots: np.ndarray, scale: float) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        chi=0.2,
        chi_mode="localized_interp",
        chi_lr_D=tuple(float(x) for x in d_knots),
        chi_lr_vals=tuple(float(y * scale) for y in chi_knots),
        A1=1.0,
        A2=1.0,
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def scan_masks(scale: float, d_knots: np.ndarray, chi_knots: np.ndarray) -> Dict[str, np.ndarray]:
    kin = make_kinetics(d_knots, chi_knots, scale)
    d_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = 1.0

    r3 = np.zeros((len(eta_vals), len(d_vals)))
    chi2 = np.zeros_like(r3)

    def w2(d: float, eta: float) -> float:
        n = 2
        gamma = kin.calculate_gamma_N(n, d, eta)
        return kin.B_N(n) * kin.g_N_cardy(n) * (1.0 - np.exp(-gamma * t_coh))

    w2_ref = w2(10.0, 1.0)
    mu_obs, sigma = 1.4, 0.4

    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            _, _, meta = kin.get_probabilities(d, eta, t_coh, 20)
            r3[i, j] = meta["generation_ratio"]
            mu_pred = w2(d, eta) / w2_ref
            chi2[i, j] = ((mu_pred - mu_obs) / sigma) ** 2

    return {
        "D": d_vals,
        "eta": eta_vals,
        "r3_mask": r3 >= 0.90,
        "hmumu_mask": chi2 <= 4.0,
        "f_r3_ge_90": np.array([float(np.mean(r3 >= 0.90))]),
        "f_hmumu": np.array([float(np.mean(chi2 <= 4.0))]),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_knots, chi_knots = load_fine_knots()

    # Include user-requested scales (0.5, 2.0) and strong-stress scales.
    scales = [0.5, 1.0, 2.0, 1e2, 1e3, 1e4, 1e5, 1e6]
    scanned: Dict[float, Dict[str, np.ndarray]] = {s: scan_masks(s, d_knots, chi_knots) for s in scales}

    base = scanned[1.0]
    rows: List[Dict[str, float]] = []
    for s in scales:
        cur = scanned[s]
        diff_r3 = float(np.mean(cur["r3_mask"] != base["r3_mask"]))
        diff_hmumu = float(np.mean(cur["hmumu_mask"] != base["hmumu_mask"]))
        rows.append(
            {
                "scale": float(s),
                "changed_fraction_r3mask": diff_r3,
                "changed_fraction_hmumu_mask": diff_hmumu,
                "f_r3_ge_90": float(cur["f_r3_ge_90"][0]),
                "f_hmumu_chi2_le_4": float(cur["f_hmumu"][0]),
            }
        )

    # Determine first non-zero movement scale for hmumu boundary.
    first_shift = None
    for r in rows:
        if r["changed_fraction_hmumu_mask"] > 0:
            first_shift = r["scale"]
            break

    out_csv = OUTDIR / "chi_scale_stress_test.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scale",
                "changed_fraction_r3mask",
                "changed_fraction_hmumu_mask",
                "f_r3_ge_90",
                "f_hmumu_chi2_le_4",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    # Panel A: hmumu boundary overlays (chi2<=4 contour).
    ax = axes[0]
    highlight_scales = [0.5, 1.0, 2.0, 1e5]
    colors = {0.5: "#1f77b4", 1.0: "#000000", 2.0: "#2ca02c", 1e5: "#d62728"}
    styles = {0.5: "--", 1.0: "-", 2.0: "-.", 1e5: ":"}
    for s in highlight_scales:
        cur = scanned[s]
        d_vals = cur["D"]
        eta_vals = cur["eta"]
        # contour at mask edge via float field
        field = cur["hmumu_mask"].astype(float)
        cs = ax.contour(
            d_vals,
            eta_vals,
            field,
            levels=[0.5],
            colors=[colors[s]],
            linestyles=[styles[s]],
            linewidths=2.0,
        )
        # Dummy line for stable legend labeling across matplotlib versions.
        ax.plot([], [], color=colors[s], linestyle=styles[s], linewidth=2.0, label=f"scale ×{s:g}")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"$\chi^2_{\mu\mu}\leq 4$ boundary overlay")
    ax.legend(loc="best", fontsize=9, frameon=True)
    if first_shift is not None:
        ax.text(
            0.03,
            0.04,
            f"first visible shift near scale ×{first_shift:g}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.7"),
        )

    # Panel B: changed-cell fractions vs scale (log-x).
    ax = axes[1]
    xs = np.array([r["scale"] for r in rows], dtype=float)
    yr3 = np.array([r["changed_fraction_r3mask"] for r in rows], dtype=float)
    yh = np.array([r["changed_fraction_hmumu_mask"] for r in rows], dtype=float)
    ax.plot(xs, yr3, marker="o", color="#444444", linewidth=2, label=r"$\Delta$ R3-mask fraction")
    ax.plot(xs, yh, marker="s", color="#d62728", linewidth=2, label=r"$\Delta \chi^2$-mask fraction")
    ax.set_xscale("log")
    ax.set_xlabel("chi amplitude scale")
    ax.set_ylabel("changed grid-cell fraction")
    ax.set_title("Boundary-movement threshold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    out_png = OUTDIR / "chi_scale_stress_test.png"
    paper_png = PAPER_DIR / "chi_scale_stress_test.png"
    fig.savefig(out_png, dpi=260)
    fig.savefig(paper_png, dpi=260)
    plt.close(fig)

    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {paper_png}")
    if first_shift is None:
        print("[summary] no hmumu boundary shift detected in tested scales")
    else:
        print(f"[summary] first hmumu boundary shift detected at scale x{first_shift:g}")


if __name__ == "__main__":
    main()

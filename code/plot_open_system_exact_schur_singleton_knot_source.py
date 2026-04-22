#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_open_system_parity_contrast_self_mirror_source import solve_fields
from extract_chi_localized_2d import Level, PhysicalParams


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "chi_open_system"


def _circle_rho(z_vals: np.ndarray, center: float, radius: float) -> np.ndarray:
    inside = radius * radius - (z_vals - center) ** 2
    out = np.full_like(z_vals, np.nan, dtype=float)
    mask = inside >= 0.0
    out[mask] = np.sqrt(inside[mask])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=float, default=11.0)
    ap.add_argument("--R", type=float, default=1.35)
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    args = ap.parse_args()

    p = PhysicalParams()
    level = Level("fine", dr=float(args.dr), dz=float(args.dz))
    fields = solve_fields(
        D=float(args.D),
        p=p,
        level=level,
        rho_max=float(args.rho_max),
        z_margin=float(args.z_margin),
        sigma=float(args.sigma),
        tol=float(args.tol),
        maxiter=int(args.maxiter),
    )

    rr = fields["rr"]
    zz = fields["zz"]
    exact_density = fields["contrast_density"] * (
        fields["delta_plus"] + fields["delta_minus"] + fields["delta_cross"]
    )

    rp = np.sqrt(rr * rr + (zz - float(args.D) / 2.0) ** 2)
    rm = np.sqrt(rr * rr + (zz + float(args.D) / 2.0) ** 2)
    core_mask = (rp <= float(args.R)) | (rm <= float(args.R))
    core_density = np.where(core_mask, exact_density, np.nan)
    tail_density = np.where(~core_mask, exact_density, np.nan)

    singleton = pd.read_csv(
        OUT_DIR / "chi_open_system_exact_schur_singleton_knot_source_summary.csv"
    ).iloc[0]
    detail = pd.read_csv(
        OUT_DIR / "chi_open_system_exact_total_core_tightening_source_detail.csv"
    )
    windows = {
        "[4,6]": float(singleton["interval_floor_4_6"]),
        "[7,10]": float(singleton["interval_floor_7_10"]),
        "{11}": float(singleton["exact_total_abs_11"]),
    }

    threshold = pd.read_csv(
        OUT_DIR / "chi_open_system_exact_schur_lift_parity_threshold_source_summary.csv"
    ).iloc[0]
    required_global = float(threshold["required_parity_floor_global"])
    required_local = float(threshold["required_parity_floor_local"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)

    ax = axes[0]
    labels = list(windows.keys())
    vals = [windows[k] for k in labels]
    xpos = np.arange(len(labels))
    ax.bar(xpos, vals, color=["#5B8E7D", "#7C6EA6", "#C46A4A"], alpha=0.9)
    ax.axhline(required_global, color="#1f77b4", linestyle="--", linewidth=1.8, label="global threshold")
    ax.axhline(required_local, color="#d62728", linestyle=":", linewidth=2.0, label="local threshold")
    ax.set_xticks(xpos, labels)
    ax.set_ylabel(r"$|V_{11}-V_{22}|$ floor")
    ax.set_title("Window Floors vs Schur Thresholds")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    core = float(singleton["core_abs_11"])
    tail = float(singleton["tail_abs_11"])
    exact = float(singleton["exact_total_abs_11"])
    req_g = float(singleton["required_core_floor_global"])
    req_l = float(singleton["required_core_floor_local"])
    names = ["core", "tail", "net", "req global", "req local"]
    heights = [core, tail, exact, req_g, req_l]
    colors = ["#2A9D8F", "#E76F51", "#264653", "#457B9D", "#E63946"]
    ax.bar(np.arange(len(names)), heights, color=colors, alpha=0.92)
    ax.set_xticks(np.arange(len(names)), names, rotation=15)
    ax.set_ylabel("magnitude")
    ax.set_title(r"Singleton $D=11$ Core/Tail Budget")
    ax.text(0.02, 0.97, rf"$\eta_{{11}}={float(singleton['eta_11']):.6f}$" "\n"
            rf"$\lambda_{{11}}={float(singleton['lambda_11']):.6f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"))

    ax = axes[2]
    vmax = np.nanmax(np.abs(exact_density))
    im = ax.pcolormesh(fields["z"], fields["rho"], exact_density, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    z_line = np.linspace(np.min(fields["z"]), np.max(fields["z"]), 1400)
    ax.plot(z_line, _circle_rho(z_line, float(args.D) / 2.0, float(args.R)), color="black", linewidth=1.2)
    ax.plot(z_line, _circle_rho(z_line, -float(args.D) / 2.0, float(args.R)), color="black", linewidth=1.2)
    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(r"Signed Exact Density at $D=11$ with $R_*=1.35$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = OUT_DIR / "chi_open_system_exact_schur_singleton_knot_source_figure.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"

STEM = "prolate_auxiliary_extraction_audit"


@dataclass(frozen=True)
class PhysicalParams:
    a: float = 0.04
    eps: float = 0.10
    m0: float = 1.0
    xi: float = 0.14


P = PhysicalParams()
Z_MAX = 80.0
N_TOTAL = 1201
D_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(5.4, 6.8, 17),
            np.array(
                [
                    5.627118644067797,
                    6.169492525423729,
                    6.4406779661016955,
                ]
            ),
        ]
    )
)
HOTSPOT_BAND = (5.6, 6.5)
KEY_D = (5.627118644067797, 6.169492525423729, 6.4406779661016955)

PROLATE_MESH_PROFILES = {
    "baseline": {"n_inner": 401, "eta_power": 1.0, "xi_power": 1.0},
    "calibrated": {"n_inner": 401, "eta_power": 0.75, "xi_power": 2.0},
}


def omega_z(z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    r1 = np.sqrt((z - D / 2.0) ** 2 + p.eps**2)
    r2 = np.sqrt((z + D / 2.0) ** 2 + p.eps**2)
    return 1.0 + p.a * (1.0 / r1 + 1.0 / r2)


def laplace_omega_z(z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    r1 = np.sqrt((z - D / 2.0) ** 2 + p.eps**2)
    r2 = np.sqrt((z + D / 2.0) ** 2 + p.eps**2)
    lap1 = -3.0 * p.eps**2 / (r1**5)
    lap2 = -3.0 * p.eps**2 / (r2**5)
    return p.a * (lap1 + lap2)


def u_z(z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    om = omega_z(z, D, p)
    lap_om = laplace_omega_z(z, D, p)
    return p.m0**2 * (om**2 - 1.0) + (1.0 - 6.0 * p.xi) * (lap_om / om)


def build_uniform_axis_mesh(zmax: float, n_total: int) -> np.ndarray:
    return np.linspace(-zmax, zmax, n_total)


def build_parametrized_prolate_axis_mesh(
    D: float,
    zmax: float,
    n_total: int,
    n_inner: int,
    eta_power: float,
    xi_power: float,
) -> np.ndarray:
    c = D / 2.0
    if zmax <= c:
        raise ValueError("zmax must exceed D/2 for the prolate outer branch.")
    if n_inner % 2 == 0:
        raise ValueError("n_inner must be odd.")
    n_outer_each = (n_total - n_inner) // 2
    if n_outer_each <= 0 or (n_total - n_inner) != 2 * n_outer_each:
        raise ValueError("invalid n_total / n_inner combination")
    s_eta = np.linspace(-1.0, 1.0, n_inner)
    eta = np.sign(s_eta) * (np.abs(s_eta) ** eta_power)
    inner = c * eta
    xi_max = zmax / c
    s_xi = np.linspace(0.0, 1.0, n_outer_each + 1)[1:]
    xi = 1.0 + (xi_max - 1.0) * (s_xi**xi_power)
    pos = c * xi
    neg = -pos[::-1]
    z = np.concatenate([neg, inner, pos])
    return np.unique(np.round(z, 14))


def build_prolate_axis_mesh(D: float, zmax: float, n_inner: int, n_outer_each: int) -> np.ndarray:
    return build_parametrized_prolate_axis_mesh(
        D=D,
        zmax=zmax,
        n_total=n_inner + 2 * n_outer_each,
        n_inner=n_inner,
        eta_power=1.0,
        xi_power=1.0,
    )


def solve_nonuniform_bound_states(z: np.ndarray, D: float, p: PhysicalParams, n_states: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if len(z) < 5:
        raise ValueError("Need at least five grid points.")
    x = np.asarray(z, dtype=float)
    x_int = x[1:-1]
    U = u_z(x, D, p)[1:-1]

    hm = x_int - x[:-2]
    hp = x[2:] - x_int
    mdiag = 0.5 * (hm + hp)

    main = 1.0 / hm + 1.0 / hp + mdiag * U
    off = -1.0 / hp[:-1] / np.sqrt(mdiag[:-1] * mdiag[1:])
    main_std = main / mdiag

    evals = la.eigh_tridiagonal(
        main_std,
        off,
        select="i",
        select_range=(0, n_states - 1),
    )[0]
    evals = np.asarray(evals, dtype=float)
    omega2 = p.m0**2 + evals
    omega = np.sqrt(np.maximum(omega2, 0.0))
    return evals, omega


def _nonuniform_second_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    for i in range(1, len(y) - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        a = 2.0 / (hm * (hm + hp))
        b = -2.0 / (hm * hp)
        c = 2.0 / (hp * (hm + hp))
        out[i] = a * y[i - 1] + b * y[i] + c * y[i + 1]
    return out


def _smoothness_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = detail["D"].to_numpy(dtype=float)
    hotspot_mask = (x >= HOTSPOT_BAND[0]) & (x <= HOTSPOT_BAND[1])

    for mesh in ["uniform_axis", "prolate_axis"]:
        for metric in ["E1", "E2", "delta_omega12"]:
            y = detail[f"{mesh}_{metric}"].to_numpy(dtype=float)
            d2 = _nonuniform_second_derivative(x, y)
            finite = np.isfinite(d2)
            hotspot = finite & hotspot_mask
            amp = float(np.nanpercentile(np.abs(y), 90.0))
            norm = max(amp, 1e-12)
            rows.append(
                {
                    "mesh": mesh,
                    "metric": metric,
                    "global_p95_abs_d2": float(np.nanpercentile(np.abs(d2[finite]), 95.0)),
                    "global_max_abs_d2": float(np.nanmax(np.abs(d2[finite]))),
                    "hotspot_p95_abs_d2": float(np.nanpercentile(np.abs(d2[hotspot]), 95.0)),
                    "hotspot_max_abs_d2": float(np.nanmax(np.abs(d2[hotspot]))),
                    "global_p95_rel_d2": float(np.nanpercentile(np.abs(d2[finite]), 95.0) / norm),
                    "hotspot_p95_rel_d2": float(np.nanpercentile(np.abs(d2[hotspot]), 95.0) / norm),
                }
            )
            detail[f"{mesh}_{metric}_d2"] = d2

    out = pd.DataFrame(rows)
    piv = out.pivot(index="metric", columns="mesh", values="hotspot_p95_rel_d2")
    reduction = []
    for metric in piv.index:
        u = float(piv.loc[metric, "uniform_axis"])
        pval = float(piv.loc[metric, "prolate_axis"])
        reduction.append(
            {
                "metric": metric,
                "hotspot_rel_d2_reduction_ratio": float(pval / max(u, 1e-30)),
                "hotspot_rel_d2_improvement": float(1.0 - pval / max(u, 1e-30)),
            }
        )
    red = pd.DataFrame(reduction)
    return out.merge(red, on="metric", how="left")


def _key_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in KEY_D:
        sub = detail.loc[np.isclose(detail["D"], d)].iloc[0]
        rows.append(
            {
                "D": float(d),
                "uniform_E1": float(sub["uniform_axis_E1"]),
                "prolate_E1": float(sub["prolate_axis_E1"]),
                "uniform_E2": float(sub["uniform_axis_E2"]),
                "prolate_E2": float(sub["prolate_axis_E2"]),
                "uniform_delta_omega12": float(sub["uniform_axis_delta_omega12"]),
                "prolate_delta_omega12": float(sub["prolate_axis_delta_omega12"]),
                "uniform_E1_d2": float(sub["uniform_axis_E1_d2"]),
                "prolate_E1_d2": float(sub["prolate_axis_E1_d2"]),
                "uniform_delta_d2": float(sub["uniform_axis_delta_omega12_d2"]),
                "prolate_delta_d2": float(sub["prolate_axis_delta_omega12_d2"]),
            }
        )
    return pd.DataFrame(rows)


def _plot(detail: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), constrained_layout=True)
    x = detail["D"].to_numpy(dtype=float)

    for metric, ax in [("E1", axes[0, 0]), ("delta_omega12", axes[0, 1])]:
        ax.plot(x, detail[f"uniform_axis_{metric}"], label="uniform axis", lw=2)
        ax.plot(x, detail[f"prolate_axis_{metric}"], label="prolate axis", lw=2)
        for d in KEY_D:
            ax.axvline(d, color="0.75", ls="--", lw=1)
        ax.set_title(f"{metric} vs D")
        ax.set_xlabel("D")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    for metric, ax in [("E1", axes[1, 0]), ("delta_omega12", axes[1, 1])]:
        ax.plot(x, np.abs(detail[f"uniform_axis_{metric}_d2"]), label="uniform |d2|", lw=2)
        ax.plot(x, np.abs(detail[f"prolate_axis_{metric}_d2"]), label="prolate |d2|", lw=2)
        for d in KEY_D:
            ax.axvline(d, color="0.75", ls="--", lw=1)
        ax.set_title(f"{metric} local roughness")
        ax.set_xlabel("D")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _stem_for_profile(mesh_profile: str) -> str:
    return STEM if mesh_profile == "baseline" else f"{STEM}_{mesh_profile}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare uniform axis vs prolate auxiliary extraction.")
    ap.add_argument(
        "--mesh-profile",
        choices=sorted(PROLATE_MESH_PROFILES),
        default="baseline",
        help="Named prolate mesh profile.",
    )
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    profile = PROLATE_MESH_PROFILES[args.mesh_profile]
    n_inner = int(profile["n_inner"])
    eta_power = float(profile["eta_power"])
    xi_power = float(profile["xi_power"])
    n_outer_each = (N_TOTAL - n_inner) // 2
    stem = _stem_for_profile(args.mesh_profile)

    rows = []
    for D in D_GRID:
        z_uni = build_uniform_axis_mesh(Z_MAX, N_TOTAL)
        z_pro = build_parametrized_prolate_axis_mesh(
            float(D),
            Z_MAX,
            N_TOTAL,
            n_inner,
            eta_power,
            xi_power,
        )

        evals_u, omega_u = solve_nonuniform_bound_states(z_uni, float(D), P, n_states=3)
        evals_p, omega_p = solve_nonuniform_bound_states(z_pro, float(D), P, n_states=3)

        rows.append(
            {
                "D": float(D),
                "uniform_axis_E1": float(evals_u[0]),
                "uniform_axis_E2": float(evals_u[1]),
                "uniform_axis_E3": float(evals_u[2]),
                "uniform_axis_omega1": float(omega_u[0]),
                "uniform_axis_omega2": float(omega_u[1]),
                "uniform_axis_delta_omega12": float(omega_u[1] - omega_u[0]),
                "prolate_axis_E1": float(evals_p[0]),
                "prolate_axis_E2": float(evals_p[1]),
                "prolate_axis_E3": float(evals_p[2]),
                "prolate_axis_omega1": float(omega_p[0]),
                "prolate_axis_omega2": float(omega_p[1]),
                "prolate_axis_delta_omega12": float(omega_p[1] - omega_p[0]),
                "uniform_axis_n_total": int(len(z_uni)),
                "prolate_axis_n_total": int(len(z_pro)),
                "prolate_axis_n_inner": int(n_inner),
                "prolate_axis_n_outer_each": int(n_outer_each),
                "prolate_axis_eta_power": float(eta_power),
                "prolate_axis_xi_power": float(xi_power),
            }
        )

    detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    summary = _smoothness_rows(detail)
    slices = _key_rows(detail)

    detail_path = OUTDIR / f"{stem}_detail.csv"
    summary_path = OUTDIR / f"{stem}_summary.csv"
    slices_path = OUTDIR / f"{stem}_slices.csv"
    png_path = OUTDIR / f"{stem}.png"
    meta_path = OUTDIR / f"{stem}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(detail, png_path)

    meta_path.write_text(
        json.dumps(
            {
                "z_max": Z_MAX,
                "n_total": N_TOTAL,
                "mesh_profile": args.mesh_profile,
                "n_inner": n_inner,
                "n_outer_each": n_outer_each,
                "eta_power": eta_power,
                "xi_power": xi_power,
                "D_grid": [float(x) for x in D_GRID],
                "hotspot_band": list(HOTSPOT_BAND),
                "key_D": list(KEY_D),
                "physical_params": {
                    "a": P.a,
                    "eps": P.eps,
                    "m0": P.m0,
                    "xi": P.xi,
                },
            },
            indent=2,
        )
    )

    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary.to_string(index=False))
    print("\n[key slices]")
    print(slices.to_string(index=False))


if __name__ == "__main__":
    main()

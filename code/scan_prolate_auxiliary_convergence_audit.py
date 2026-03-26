#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_auxiliary_extraction_audit import (
    PAPER_DIR,
    OUTDIR,
    P,
    PROLATE_MESH_PROFILES,
    Z_MAX,
    build_parametrized_prolate_axis_mesh,
    build_prolate_axis_mesh,
    build_uniform_axis_mesh,
    solve_nonuniform_bound_states,
)


STEM = "prolate_auxiliary_convergence_audit"

D_GRID = np.array(
    [
        5.4,
        5.627118644067797,
        5.8,
        6.0,
        6.169492525423729,
        6.3,
        6.4406779661016955,
        6.6,
        6.8,
    ],
    dtype=float,
)

LEVELS = [
    {"level": "coarse", "uniform_n_total": 401, "prolate_n_total": 401, "prolate_n_inner": 121},
    {"level": "mid", "uniform_n_total": 801, "prolate_n_total": 801, "prolate_n_inner": 241},
    {"level": "fine", "uniform_n_total": 1201, "prolate_n_total": 1201, "prolate_n_inner": 401},
]

def _stem_for_profile(mesh_profile: str) -> str:
    return STEM if mesh_profile == "baseline" else f"{STEM}_{mesh_profile}"


def _solve_level(D: float, level: dict[str, int | str], eta_power: float, xi_power: float) -> dict[str, float | str]:
    uniform_n_total = int(level["uniform_n_total"])
    prolate_n_total = int(level["prolate_n_total"])
    prolate_n_inner = int(level["prolate_n_inner"])
    prolate_n_outer_each = (prolate_n_total - prolate_n_inner) // 2

    z_uni = build_uniform_axis_mesh(Z_MAX, uniform_n_total)
    z_pro = build_parametrized_prolate_axis_mesh(
        float(D),
        Z_MAX,
        prolate_n_total,
        prolate_n_inner,
        eta_power,
        xi_power,
    )

    evals_u, omega_u = solve_nonuniform_bound_states(z_uni, float(D), P, n_states=3)
    evals_p, omega_p = solve_nonuniform_bound_states(z_pro, float(D), P, n_states=3)

    return {
        "D": float(D),
        "level": str(level["level"]),
        "uniform_n_total": uniform_n_total,
        "prolate_n_total": int(len(z_pro)),
        "prolate_n_inner": prolate_n_inner,
        "prolate_n_outer_each": prolate_n_outer_each,
        "prolate_eta_power": float(eta_power),
        "prolate_xi_power": float(xi_power),
        "uniform_E1": float(evals_u[0]),
        "uniform_E2": float(evals_u[1]),
        "uniform_omega1": float(omega_u[0]),
        "uniform_omega2": float(omega_u[1]),
        "uniform_delta_omega12": float(omega_u[1] - omega_u[0]),
        "prolate_E1": float(evals_p[0]),
        "prolate_E2": float(evals_p[1]),
        "prolate_omega1": float(omega_p[0]),
        "prolate_omega2": float(omega_p[1]),
        "prolate_delta_omega12": float(omega_p[1] - omega_p[0]),
    }


def _rel_err(cur: float, ref: float) -> float:
    return float(abs(cur - ref) / max(abs(ref), 1e-30))


def _build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for D, grp in detail.groupby("D", sort=True):
        sub = grp.set_index("level")
        fine = sub.loc["fine"]
        for level in ["coarse", "mid", "fine"]:
            cur = sub.loc[level]
            row = {
                "D": float(D),
                "level": level,
                "uniform_rel_E1_vs_uniform_fine": _rel_err(cur["uniform_E1"], fine["uniform_E1"]),
                "uniform_rel_E2_vs_uniform_fine": _rel_err(cur["uniform_E2"], fine["uniform_E2"]),
                "uniform_rel_delta_omega12_vs_uniform_fine": _rel_err(
                    cur["uniform_delta_omega12"], fine["uniform_delta_omega12"]
                ),
                "prolate_rel_E1_vs_prolate_fine": _rel_err(cur["prolate_E1"], fine["prolate_E1"]),
                "prolate_rel_E2_vs_prolate_fine": _rel_err(cur["prolate_E2"], fine["prolate_E2"]),
                "prolate_rel_delta_omega12_vs_prolate_fine": _rel_err(
                    cur["prolate_delta_omega12"], fine["prolate_delta_omega12"]
                ),
                "crossmesh_rel_E1_vs_uniform_fine": _rel_err(cur["prolate_E1"], fine["uniform_E1"]),
                "crossmesh_rel_E2_vs_uniform_fine": _rel_err(cur["prolate_E2"], fine["uniform_E2"]),
                "crossmesh_rel_delta_omega12_vs_uniform_fine": _rel_err(
                    cur["prolate_delta_omega12"], fine["uniform_delta_omega12"]
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["D", "level"]).reset_index(drop=True)


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in ["coarse", "mid", "fine"]:
        sub = summary.loc[summary["level"] == level].copy()
        rows.append(
            {
                "level": level,
                "uniform_p95_rel_E1_vs_uniform_fine": float(np.percentile(sub["uniform_rel_E1_vs_uniform_fine"], 95.0)),
                "uniform_p95_rel_E2_vs_uniform_fine": float(np.percentile(sub["uniform_rel_E2_vs_uniform_fine"], 95.0)),
                "uniform_p95_rel_delta_omega12_vs_uniform_fine": float(
                    np.percentile(sub["uniform_rel_delta_omega12_vs_uniform_fine"], 95.0)
                ),
                "prolate_p95_rel_E1_vs_prolate_fine": float(np.percentile(sub["prolate_rel_E1_vs_prolate_fine"], 95.0)),
                "prolate_p95_rel_E2_vs_prolate_fine": float(np.percentile(sub["prolate_rel_E2_vs_prolate_fine"], 95.0)),
                "prolate_p95_rel_delta_omega12_vs_prolate_fine": float(
                    np.percentile(sub["prolate_rel_delta_omega12_vs_prolate_fine"], 95.0)
                ),
                "crossmesh_p95_rel_E1_vs_uniform_fine": float(np.percentile(sub["crossmesh_rel_E1_vs_uniform_fine"], 95.0)),
                "crossmesh_p95_rel_E2_vs_uniform_fine": float(np.percentile(sub["crossmesh_rel_E2_vs_uniform_fine"], 95.0)),
                "crossmesh_p95_rel_delta_omega12_vs_uniform_fine": float(
                    np.percentile(sub["crossmesh_rel_delta_omega12_vs_uniform_fine"], 95.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), constrained_layout=True)
    x = D_GRID
    metric_triplets = [
        ("uniform_rel_E1_vs_uniform_fine", "prolate_rel_E1_vs_prolate_fine", "crossmesh_rel_E1_vs_uniform_fine", "E1"),
        ("uniform_rel_E2_vs_uniform_fine", "prolate_rel_E2_vs_prolate_fine", "crossmesh_rel_E2_vs_uniform_fine", "E2"),
        (
            "uniform_rel_delta_omega12_vs_uniform_fine",
            "prolate_rel_delta_omega12_vs_prolate_fine",
            "crossmesh_rel_delta_omega12_vs_uniform_fine",
            "Δω12",
        ),
    ]
    for ax, (u_col, p_col, c_col, title) in zip(axes, metric_triplets):
        mid = summary.loc[summary["level"] == "mid"].sort_values("D")
        coarse = summary.loc[summary["level"] == "coarse"].sort_values("D")
        ax.plot(x, coarse[u_col], label="uniform coarse→fine", lw=1.8)
        ax.plot(x, coarse[p_col], label="prolate coarse→fine", lw=1.8)
        ax.plot(x, mid[u_col], label="uniform mid→fine", lw=1.8, ls="--")
        ax.plot(x, mid[p_col], label="prolate mid→fine", lw=1.8, ls="--")
        ax.plot(x, mid[c_col], label="prolate vs uniform fine", lw=1.4, ls=":")
        ax.set_title(title)
        ax.set_xlabel("D")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Compare uniform/prolate convergence across mesh levels.")
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
    eta_power = float(profile["eta_power"])
    xi_power = float(profile["xi_power"])
    stem = _stem_for_profile(args.mesh_profile)

    rows = []
    total = len(D_GRID) * len(LEVELS)
    idx = 0
    for D in D_GRID:
        for level in LEVELS:
            idx += 1
            print(f"[solve {idx}/{total}] D={D:.6f} level={level['level']}", flush=True)
            rows.append(_solve_level(float(D), level, eta_power, xi_power))

    detail = pd.DataFrame(rows).sort_values(["D", "level"]).reset_index(drop=True)
    summary = _build_summary(detail)
    aggregate = _aggregate(summary)

    detail_path = OUTDIR / f"{stem}_detail.csv"
    summary_path = OUTDIR / f"{stem}_summary.csv"
    slices_path = OUTDIR / f"{stem}_slices.csv"
    png_path = OUTDIR / f"{stem}.png"
    meta_path = OUTDIR / f"{stem}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    aggregate.to_csv(slices_path, index=False)
    _plot(summary, png_path)

    meta_path.write_text(
        json.dumps(
            {
                "D_grid": [float(x) for x in D_GRID],
                "levels": LEVELS,
                "z_max": Z_MAX,
                "mesh_profile": args.mesh_profile,
                "eta_power": eta_power,
                "xi_power": xi_power,
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

    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

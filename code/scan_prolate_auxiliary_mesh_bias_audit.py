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
    OUTDIR,
    PAPER_DIR,
    P,
    Z_MAX,
    build_uniform_axis_mesh,
    solve_nonuniform_bound_states,
)


STEM = "prolate_auxiliary_mesh_bias_audit"

N_TOTAL = 1201
D_CALIBRATION = np.array([6.0, 12.0, 18.0], dtype=float)
D_HOTSPOT = np.array(
    [5.4, 5.627118644067797, 5.8, 6.0, 6.169492525423729, 6.3, 6.4406779661016955, 6.6, 6.8],
    dtype=float,
)

OMEGA_EXACT_PATH = OUTDIR.parent / "omega_fp_1d" / "omega_exact_D6-12-18.csv"

INNER_COUNTS = [241, 321, 401, 481]
ETA_POWERS = [0.75, 1.0, 1.25, 1.5]
XI_POWERS = [0.75, 1.0, 1.5, 2.0]


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


def _nonuniform_second_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    for i in range(1, len(y) - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        a = 2.0 / (hm * (hm + hp))
        b = -2.0 / (hm * hp)
        c = 2.0 / (hp * (hm + hp))
        out[i] = a * y[i - 1] + b * y[i] + c * y[i + 1]
    return out


def _rel_err(cur: float, ref: float) -> float:
    return float(abs(cur - ref) / max(abs(ref), 1e-30))


def _hotspot_roughness(values: np.ndarray) -> float:
    d2 = _nonuniform_second_derivative(D_HOTSPOT, values)
    finite = np.isfinite(d2)
    amp = max(float(np.percentile(np.abs(values), 90.0)), 1e-30)
    return float(np.percentile(np.abs(d2[finite]), 95.0) / amp)


def _solve_uniform_reference() -> dict[float, dict[str, float]]:
    exact = pd.read_csv(OMEGA_EXACT_PATH)
    exact = exact.loc[exact["level"] == "fine"].copy().set_index("D")
    z_uniform = build_uniform_axis_mesh(Z_MAX, N_TOTAL)
    out: dict[float, dict[str, float]] = {}
    for D in D_CALIBRATION:
        evals, omega = solve_nonuniform_bound_states(z_uniform, float(D), P, n_states=3)
        ex = exact.loc[D]
        out[float(D)] = {
            "uniform_E1": float(evals[0]),
            "uniform_E2": float(evals[1]),
            "uniform_omega1": float(omega[0]),
            "uniform_omega2": float(omega[1]),
            "uniform_delta_omega12": float(omega[1] - omega[0]),
            "exact_E1": float(ex["E1"]),
            "exact_E2": float(ex["E2"]),
            "exact_omega1": float(ex["omega1"]),
            "exact_omega2": float(ex["omega2"]),
            "exact_delta_omega12": float(ex["delta_omega12"]),
        }
    return out


def _solve_combo(n_inner: int, eta_power: float, xi_power: float) -> tuple[pd.DataFrame, dict[str, float]]:
    detail_rows = []
    hotspot_omega1 = []
    hotspot_omega2 = []
    hotspot_delta = []

    refs = _solve_uniform_reference()
    for D in D_CALIBRATION:
        z_pro = build_parametrized_prolate_axis_mesh(float(D), Z_MAX, N_TOTAL, n_inner, eta_power, xi_power)
        evals, omega = solve_nonuniform_bound_states(z_pro, float(D), P, n_states=3)
        ref = refs[float(D)]
        detail_rows.append(
            {
                "D": float(D),
                "n_inner": int(n_inner),
                "eta_power": float(eta_power),
                "xi_power": float(xi_power),
                "uniform_omega1": ref["uniform_omega1"],
                "uniform_omega2": ref["uniform_omega2"],
                "uniform_delta_omega12": ref["uniform_delta_omega12"],
                "exact_omega1": ref["exact_omega1"],
                "exact_omega2": ref["exact_omega2"],
                "exact_delta_omega12": ref["exact_delta_omega12"],
                "prolate_E1": float(evals[0]),
                "prolate_E2": float(evals[1]),
                "prolate_omega1": float(omega[0]),
                "prolate_omega2": float(omega[1]),
                "prolate_delta_omega12": float(omega[1] - omega[0]),
                "uniform_rel_omega1_vs_exact": _rel_err(ref["uniform_omega1"], ref["exact_omega1"]),
                "uniform_rel_omega2_vs_exact": _rel_err(ref["uniform_omega2"], ref["exact_omega2"]),
                "uniform_rel_delta_vs_exact": _rel_err(ref["uniform_delta_omega12"], ref["exact_delta_omega12"]),
                "prolate_rel_omega1_vs_exact": _rel_err(float(omega[0]), ref["exact_omega1"]),
                "prolate_rel_omega2_vs_exact": _rel_err(float(omega[1]), ref["exact_omega2"]),
                "prolate_rel_delta_vs_exact": _rel_err(float(omega[1] - omega[0]), ref["exact_delta_omega12"]),
            }
        )

    for D in D_HOTSPOT:
        z_pro = build_parametrized_prolate_axis_mesh(float(D), Z_MAX, N_TOTAL, n_inner, eta_power, xi_power)
        _, omega = solve_nonuniform_bound_states(z_pro, float(D), P, n_states=3)
        hotspot_omega1.append(float(omega[0]))
        hotspot_omega2.append(float(omega[1]))
        hotspot_delta.append(float(omega[1] - omega[0]))

    df = pd.DataFrame(detail_rows)
    calibration_bias = float(
        max(
            np.percentile(df["prolate_rel_omega1_vs_exact"], 95.0),
            np.percentile(df["prolate_rel_omega2_vs_exact"], 95.0),
        )
    )
    split_bias = float(np.percentile(df["prolate_rel_delta_vs_exact"], 95.0))
    uniform_calibration_bias = float(
        max(
            np.percentile(df["uniform_rel_omega1_vs_exact"], 95.0),
            np.percentile(df["uniform_rel_omega2_vs_exact"], 95.0),
        )
    )
    uniform_split_bias = float(np.percentile(df["uniform_rel_delta_vs_exact"], 95.0))

    summary = {
        "n_inner": int(n_inner),
        "eta_power": float(eta_power),
        "xi_power": float(xi_power),
        "uniform_exact_p95_omega_bias": uniform_calibration_bias,
        "prolate_exact_p95_omega_bias": calibration_bias,
        "uniform_exact_p95_split_bias": uniform_split_bias,
        "prolate_exact_p95_split_bias": split_bias,
        "prolate_hotspot_p95_rel_d2_omega1": _hotspot_roughness(np.asarray(hotspot_omega1)),
        "prolate_hotspot_p95_rel_d2_omega2": _hotspot_roughness(np.asarray(hotspot_omega2)),
        "prolate_hotspot_p95_rel_d2_delta": _hotspot_roughness(np.asarray(hotspot_delta)),
        "prolate_omega1_D6": float(df.loc[np.isclose(df["D"], 6.0), "prolate_omega1"].iloc[0]),
        "prolate_omega1_D12": float(df.loc[np.isclose(df["D"], 12.0), "prolate_omega1"].iloc[0]),
        "prolate_omega1_D18": float(df.loc[np.isclose(df["D"], 18.0), "prolate_omega1"].iloc[0]),
        "prolate_delta_D6": float(df.loc[np.isclose(df["D"], 6.0), "prolate_delta_omega12"].iloc[0]),
        "prolate_delta_D12": float(df.loc[np.isclose(df["D"], 12.0), "prolate_delta_omega12"].iloc[0]),
        "prolate_delta_D18": float(df.loc[np.isclose(df["D"], 18.0), "prolate_delta_omega12"].iloc[0]),
    }
    return df, summary


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(
        summary["prolate_exact_p95_omega_bias"],
        summary["prolate_exact_p95_split_bias"],
        c=summary["prolate_hotspot_p95_rel_d2_omega1"],
        cmap="viridis",
        s=55,
    )
    ax.set_xlabel("p95 omega bias vs exact")
    ax.set_ylabel("p95 splitting bias vs exact")
    ax.set_title("Calibration tradeoff")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="hotspot roughness omega1")

    ax = axes[1]
    ax.scatter(summary["eta_power"], summary["tradeoff_score"], c=summary["xi_power"], cmap="plasma", s=55)
    ax.set_xlabel("eta_power")
    ax.set_ylabel("tradeoff score")
    ax.set_title("eta-power scan")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.scatter(summary["n_inner"], summary["tradeoff_score"], c=summary["xi_power"], cmap="cividis", s=55)
    ax.set_xlabel("n_inner")
    ax.set_ylabel("tradeoff score")
    ax.set_title("inner allocation scan")
    ax.grid(True, alpha=0.25)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    all_detail = []
    all_summary = []
    combos = [(n_inner, eta_power, xi_power) for n_inner in INNER_COUNTS for eta_power in ETA_POWERS for xi_power in XI_POWERS]
    for idx, (n_inner, eta_power, xi_power) in enumerate(combos, start=1):
        print(
            f"[combo {idx}/{len(combos)}] n_inner={n_inner} eta_power={eta_power:.2f} xi_power={xi_power:.2f}",
            flush=True,
        )
        detail, summary = _solve_combo(n_inner, eta_power, xi_power)
        all_detail.append(detail)
        all_summary.append(summary)

    detail_df = pd.concat(all_detail, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)

    baseline = summary_df.loc[
        (summary_df["n_inner"] == 401)
        & np.isclose(summary_df["eta_power"], 1.0)
        & np.isclose(summary_df["xi_power"], 1.0)
    ].iloc[0]

    summary_df["omega_bias_ratio_vs_baseline"] = summary_df["prolate_exact_p95_omega_bias"] / max(
        float(baseline["prolate_exact_p95_omega_bias"]), 1e-30
    )
    summary_df["split_bias_ratio_vs_baseline"] = summary_df["prolate_exact_p95_split_bias"] / max(
        float(baseline["prolate_exact_p95_split_bias"]), 1e-30
    )
    summary_df["smooth_omega_ratio_vs_baseline"] = summary_df["prolate_hotspot_p95_rel_d2_omega1"] / max(
        float(baseline["prolate_hotspot_p95_rel_d2_omega1"]), 1e-30
    )
    summary_df["smooth_delta_ratio_vs_baseline"] = summary_df["prolate_hotspot_p95_rel_d2_delta"] / max(
        float(baseline["prolate_hotspot_p95_rel_d2_delta"]), 1e-30
    )
    summary_df["tradeoff_score"] = (
        summary_df["omega_bias_ratio_vs_baseline"]
        * np.sqrt(summary_df["split_bias_ratio_vs_baseline"])
        * np.maximum(1.0, summary_df["smooth_omega_ratio_vs_baseline"])
        * np.maximum(1.0, summary_df["smooth_delta_ratio_vs_baseline"])
    )
    summary_df["beats_baseline_on_bias"] = (
        (summary_df["prolate_exact_p95_omega_bias"] < float(baseline["prolate_exact_p95_omega_bias"]))
        & (summary_df["prolate_exact_p95_split_bias"] <= 1.10 * float(baseline["prolate_exact_p95_split_bias"]))
    )
    summary_df = summary_df.sort_values(["tradeoff_score", "prolate_exact_p95_omega_bias"]).reset_index(drop=True)

    top = summary_df.head(12).copy()

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    top.to_csv(slices_path, index=False)
    _plot(summary_df, png_path)

    meta_path.write_text(
        json.dumps(
            {
                "N_total": N_TOTAL,
                "D_calibration": [float(x) for x in D_CALIBRATION],
                "D_hotspot": [float(x) for x in D_HOTSPOT],
                "inner_counts": INNER_COUNTS,
                "eta_powers": ETA_POWERS,
                "xi_powers": XI_POWERS,
                "baseline": {
                    "n_inner": 401,
                    "eta_power": 1.0,
                    "xi_power": 1.0,
                },
                "target": str(OMEGA_EXACT_PATH),
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

    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

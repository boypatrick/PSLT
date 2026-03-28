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


STEM = "uniform_lorentz_reparam_extraction_audit"

N_TOTAL = 1201
T_COH = 1.0

D_CALIBRATION = np.array([6.0, 12.0, 18.0], dtype=float)
D_HOTSPOT = np.array(
    [5.4, 5.627118644067797, 5.8, 6.0, 6.169492525423729, 6.3, 6.4406779661016955, 6.6, 6.8],
    dtype=float,
)

OMEGA_EXACT_PATH = OUTDIR.parent / "omega_fp_1d" / "omega_exact_D6-12-18.csv"

MODE_GRID = ("length_contraction", "boosted_frame")
BETA_GRID = (0.2, 0.4, 0.6, 0.8)


def _gamma(beta: float) -> float:
    beta = float(beta)
    return float(1.0 / np.sqrt(max(1.0 - beta * beta, 1e-12)))


def _effective_D(D: float, mode: str, beta: float, t_coh: float = T_COH) -> float:
    g = _gamma(beta)
    if mode == "length_contraction":
        d_eff = float(D) / g
    elif mode == "boosted_frame":
        d_eff = g * (float(D) - float(beta) * float(t_coh))
    else:
        raise ValueError(f"unknown mode: {mode}")
    return float(max(d_eff, 1e-6))


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


def _solve_uniform_reference() -> tuple[dict[float, dict[str, float]], dict[str, float]]:
    exact = pd.read_csv(OMEGA_EXACT_PATH)
    exact = exact.loc[exact["level"] == "fine"].copy().set_index("D")
    z_uniform = build_uniform_axis_mesh(Z_MAX, N_TOTAL)

    ref: dict[float, dict[str, float]] = {}
    for D in D_CALIBRATION:
        evals, omega = solve_nonuniform_bound_states(z_uniform, float(D), P, n_states=3)
        ex = exact.loc[D]
        ref[float(D)] = {
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

    hotspot_omega1 = []
    hotspot_omega2 = []
    hotspot_delta = []
    for D in D_HOTSPOT:
        evals, omega = solve_nonuniform_bound_states(z_uniform, float(D), P, n_states=3)
        hotspot_omega1.append(float(omega[0]))
        hotspot_omega2.append(float(omega[1]))
        hotspot_delta.append(float(omega[1] - omega[0]))

    baseline = {
        "uniform_exact_p95_omega_bias": float(
            max(
                np.percentile([_rel_err(v["uniform_omega1"], v["exact_omega1"]) for v in ref.values()], 95.0),
                np.percentile([_rel_err(v["uniform_omega2"], v["exact_omega2"]) for v in ref.values()], 95.0),
            )
        ),
        "uniform_exact_p95_split_bias": float(
            np.percentile([_rel_err(v["uniform_delta_omega12"], v["exact_delta_omega12"]) for v in ref.values()], 95.0)
        ),
        "uniform_hotspot_p95_rel_d2_omega1": _hotspot_roughness(np.asarray(hotspot_omega1)),
        "uniform_hotspot_p95_rel_d2_omega2": _hotspot_roughness(np.asarray(hotspot_omega2)),
        "uniform_hotspot_p95_rel_d2_delta": _hotspot_roughness(np.asarray(hotspot_delta)),
    }
    return ref, baseline


def _solve_combo(mode: str, beta: float, refs: dict[float, dict[str, float]]) -> tuple[pd.DataFrame, dict[str, float]]:
    z_uniform = build_uniform_axis_mesh(Z_MAX, N_TOTAL)
    detail_rows = []
    hotspot_omega1 = []
    hotspot_omega2 = []
    hotspot_delta = []

    for D in D_CALIBRATION:
        d_eff = _effective_D(float(D), mode=mode, beta=float(beta))
        evals, omega = solve_nonuniform_bound_states(z_uniform, d_eff, P, n_states=3)
        ref = refs[float(D)]
        detail_rows.append(
            {
                "D": float(D),
                "mode": mode,
                "beta": float(beta),
                "gamma": _gamma(float(beta)),
                "D_eff": float(d_eff),
                "uniform_omega1": ref["uniform_omega1"],
                "uniform_omega2": ref["uniform_omega2"],
                "uniform_delta_omega12": ref["uniform_delta_omega12"],
                "exact_omega1": ref["exact_omega1"],
                "exact_omega2": ref["exact_omega2"],
                "exact_delta_omega12": ref["exact_delta_omega12"],
                "lorentz_E1": float(evals[0]),
                "lorentz_E2": float(evals[1]),
                "lorentz_omega1": float(omega[0]),
                "lorentz_omega2": float(omega[1]),
                "lorentz_delta_omega12": float(omega[1] - omega[0]),
                "lorentz_rel_omega1_vs_exact": _rel_err(float(omega[0]), ref["exact_omega1"]),
                "lorentz_rel_omega2_vs_exact": _rel_err(float(omega[1]), ref["exact_omega2"]),
                "lorentz_rel_delta_vs_exact": _rel_err(float(omega[1] - omega[0]), ref["exact_delta_omega12"]),
            }
        )

    for D in D_HOTSPOT:
        d_eff = _effective_D(float(D), mode=mode, beta=float(beta))
        _, omega = solve_nonuniform_bound_states(z_uniform, d_eff, P, n_states=3)
        hotspot_omega1.append(float(omega[0]))
        hotspot_omega2.append(float(omega[1]))
        hotspot_delta.append(float(omega[1] - omega[0]))

    df = pd.DataFrame(detail_rows)
    summary = {
        "mode": mode,
        "beta": float(beta),
        "gamma": _gamma(float(beta)),
        "lorentz_exact_p95_omega_bias": float(
            max(
                np.percentile(df["lorentz_rel_omega1_vs_exact"], 95.0),
                np.percentile(df["lorentz_rel_omega2_vs_exact"], 95.0),
            )
        ),
        "lorentz_exact_p95_split_bias": float(np.percentile(df["lorentz_rel_delta_vs_exact"], 95.0)),
        "lorentz_hotspot_p95_rel_d2_omega1": _hotspot_roughness(np.asarray(hotspot_omega1)),
        "lorentz_hotspot_p95_rel_d2_omega2": _hotspot_roughness(np.asarray(hotspot_omega2)),
        "lorentz_hotspot_p95_rel_d2_delta": _hotspot_roughness(np.asarray(hotspot_delta)),
        "lorentz_omega1_D6": float(df.loc[np.isclose(df["D"], 6.0), "lorentz_omega1"].iloc[0]),
        "lorentz_omega1_D12": float(df.loc[np.isclose(df["D"], 12.0), "lorentz_omega1"].iloc[0]),
        "lorentz_omega1_D18": float(df.loc[np.isclose(df["D"], 18.0), "lorentz_omega1"].iloc[0]),
        "lorentz_delta_D6": float(df.loc[np.isclose(df["D"], 6.0), "lorentz_delta_omega12"].iloc[0]),
        "lorentz_delta_D12": float(df.loc[np.isclose(df["D"], 12.0), "lorentz_delta_omega12"].iloc[0]),
        "lorentz_delta_D18": float(df.loc[np.isclose(df["D"], 18.0), "lorentz_delta_omega12"].iloc[0]),
        "D_eff_min": float(min(_effective_D(float(D), mode=mode, beta=float(beta)) for D in D_HOTSPOT)),
        "D_eff_max": float(max(_effective_D(float(D), mode=mode, beta=float(beta)) for D in D_HOTSPOT)),
    }
    return df, summary


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    ax = axes[0]
    for mode, grp in summary.groupby("mode", sort=False):
        ax.plot(grp["beta"], grp["lorentz_exact_p95_omega_bias"], marker="o", label=f"{mode} omega bias")
        ax.plot(grp["beta"], grp["lorentz_exact_p95_split_bias"], marker="s", ls="--", label=f"{mode} split bias")
    ax.set_xlabel("beta")
    ax.set_ylabel("p95 relative error")
    ax.set_title("Calibration vs exact fine")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for mode, grp in summary.groupby("mode", sort=False):
        ax.plot(grp["beta"], grp["lorentz_hotspot_p95_rel_d2_omega1"], marker="o", label=f"{mode} omega1")
        ax.plot(grp["beta"], grp["lorentz_hotspot_p95_rel_d2_delta"], marker="s", ls="--", label=f"{mode} delta")
    ax.set_xlabel("beta")
    ax.set_ylabel("hotspot p95 rel d2")
    ax.set_title("Hotspot roughness")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    for mode, grp in summary.groupby("mode", sort=False):
        ax.plot(grp["beta"], grp["tradeoff_score"], marker="o", label=mode)
    ax.set_xlabel("beta")
    ax.set_ylabel("tradeoff score")
    ax.set_title("Overall tradeoff")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    refs, baseline = _solve_uniform_reference()
    all_detail = []
    all_summary = []

    combos = [(mode, beta) for mode in MODE_GRID for beta in BETA_GRID]
    for idx, (mode, beta) in enumerate(combos, start=1):
        print(f"[combo {idx}/{len(combos)}] mode={mode} beta={beta:.2f}", flush=True)
        detail, summary = _solve_combo(mode, beta, refs)
        all_detail.append(detail)
        all_summary.append(summary)

    detail_df = pd.concat(all_detail, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)

    for key, val in baseline.items():
        summary_df[key] = float(val)

    summary_df["omega_bias_ratio_vs_uniform"] = summary_df["lorentz_exact_p95_omega_bias"] / max(
        float(baseline["uniform_exact_p95_omega_bias"]), 1e-30
    )
    summary_df["split_bias_ratio_vs_uniform"] = summary_df["lorentz_exact_p95_split_bias"] / max(
        float(baseline["uniform_exact_p95_split_bias"]), 1e-30
    )
    summary_df["smooth_omega_ratio_vs_uniform"] = summary_df["lorentz_hotspot_p95_rel_d2_omega1"] / max(
        float(baseline["uniform_hotspot_p95_rel_d2_omega1"]), 1e-30
    )
    summary_df["smooth_delta_ratio_vs_uniform"] = summary_df["lorentz_hotspot_p95_rel_d2_delta"] / max(
        float(baseline["uniform_hotspot_p95_rel_d2_delta"]), 1e-30
    )
    summary_df["tradeoff_score"] = (
        np.maximum(summary_df["omega_bias_ratio_vs_uniform"], 1e-30)
        * np.sqrt(np.maximum(summary_df["split_bias_ratio_vs_uniform"], 1e-30))
        * np.maximum(1.0, summary_df["smooth_omega_ratio_vs_uniform"])
        * np.maximum(1.0, summary_df["smooth_delta_ratio_vs_uniform"])
    )
    summary_df["beats_uniform_on_bias"] = (
        (summary_df["lorentz_exact_p95_omega_bias"] < float(baseline["uniform_exact_p95_omega_bias"]))
        & (summary_df["lorentz_exact_p95_split_bias"] <= 1.10 * float(baseline["uniform_exact_p95_split_bias"]))
    )
    summary_df["beats_uniform_on_smoothness"] = (
        (summary_df["lorentz_hotspot_p95_rel_d2_omega1"] < float(baseline["uniform_hotspot_p95_rel_d2_omega1"]))
        & (summary_df["lorentz_hotspot_p95_rel_d2_delta"] < float(baseline["uniform_hotspot_p95_rel_d2_delta"]))
    )
    summary_df = summary_df.sort_values(["tradeoff_score", "lorentz_exact_p95_omega_bias"]).reset_index(drop=True)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(slices_path, index=False)
    _plot(summary_df, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "D_calibration": [float(x) for x in D_CALIBRATION],
                "D_hotspot": [float(x) for x in D_HOTSPOT],
                "z_max": Z_MAX,
                "n_total": N_TOTAL,
                "mode_grid": list(MODE_GRID),
                "beta_grid": list(BETA_GRID),
                "t_coh": T_COH,
                "physical_params": {
                    "a": P.a,
                    "eps": P.eps,
                    "m0": P.m0,
                    "xi": P.xi,
                },
                "notes": (
                    "Lorentz-like reparameterization audit on the current uniform 1D extraction. "
                    "This is treated as a numerical D->D_eff ansatz, not as a promoted physical Lorentz claim."
                ),
            },
            indent=2,
        )
    )

    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(summary_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()

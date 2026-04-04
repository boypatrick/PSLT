#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv"

REF_D = float(PAPER_BASELINE["ref_D"])
REF_ETA = float(PAPER_BASELINE["ref_eta"])
T_COH = float(PAPER_BASELINE["t_coh"])
WIDTH_FIX = {
    "width_power_base": 0.80,
    "width_power_log_ratio_coeff": 0.15,
    "width_power_min": 0.60,
    "width_power_max": 0.85,
    "width_power_trigger_lo": 2.5,
    "width_power_trigger_hi": 5.0,
    "width_power_turnoff_lo": 10.0,
    "width_power_turnoff_hi": 20.0,
    "width_power_tail_logratio_lo": 1.5,
    "width_power_tail_logratio_hi": 2.5,
    "width_power_tail_reboost_max": 0.15,
}

D40_BETA = float(os.environ.get("D40_BETA", "1.304"))
D48_BETA = float(os.environ.get("D48_BETA", "0.55"))
D72_BETA = float(os.environ.get("D72_BETA", "0.35"))
D80_BETA = float(os.environ.get("D80_BETA", "1.20"))

CURRENT = {
    "g_beta": float(os.environ.get("CURRENT_G_BETA", "0.2468174085")),
    "g_center": float(os.environ.get("CURRENT_G_CENTER", "0.0488372924")),
    "g_left_width": float(os.environ.get("CURRENT_G_LEFT_WIDTH", os.environ.get("CURRENT_G_HALF_WIDTH", "0.0251312702"))),
    "g_right_width": float(os.environ.get("CURRENT_G_RIGHT_WIDTH", os.environ.get("CURRENT_G_HALF_WIDTH", "0.0251312702"))),
}
BOUNDS = {
    "g_beta": (
        float(os.environ.get("G_BETA_MIN", "0.2460")),
        float(os.environ.get("G_BETA_MAX", "0.2476")),
    ),
    "g_center": (
        float(os.environ.get("G_CENTER_MIN", "0.0482")),
        float(os.environ.get("G_CENTER_MAX", "0.0493")),
    ),
    "g_left_width": (
        float(os.environ.get("G_LEFT_WIDTH_MIN", "0.0240")),
        float(os.environ.get("G_LEFT_WIDTH_MAX", "0.0260")),
    ),
    "g_right_width": (
        float(os.environ.get("G_RIGHT_WIDTH_MIN", "0.0240")),
        float(os.environ.get("G_RIGHT_WIDTH_MAX", "0.0260")),
    ),
}

N_SAMPLES = int(os.environ.get("N_SAMPLES", "8"))
OUT_TAG = os.environ.get("OUT_TAG", "d60_d64_asym_sobol_probe_v1")
OUT_SAMPLES = OUTDIR / f"runtime_direct_detlin_{OUT_TAG}_samples.csv"
OUT_EFFECTS = OUTDIR / f"runtime_direct_detlin_{OUT_TAG}_effects.csv"
OUT_SUMMARY = OUTDIR / f"runtime_direct_detlin_{OUT_TAG}_summary.json"


def _build_kinetics(d_min: float, d_max: float, d_num: int):
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode="cell_direct_runtime_release_tailm2gnorm",
        d_min=float(d_min),
        d_max=float(d_max),
        d_num=int(d_num),
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )
    p = kin.params
    for k, v in WIDTH_FIX.items():
        setattr(p, f"hll_match_{k}", float(v))
    p.hll_uv_runtime_direct_m2_cross_blend = 0.75
    return kin


def _patch_width_bands(kin):
    import types

    original = kin._blend_observable_width_ratio
    bands = [
        {"beta": D40_BETA, "center": 4.0, "half_width": 0.40},
        {"beta": D48_BETA, "center": 4.8, "half_width": 0.40},
        {"beta": D72_BETA, "center": 7.2, "half_width": 0.40},
        {"beta": D80_BETA, "center": 8.0, "half_width": 0.40},
    ]

    def patched(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        exponent = np.log(max(base, floor))
        for spec in bands:
            act = float(max(0.0, 1.0 - abs(float(D) - float(spec["center"])) / float(spec["half_width"])))
            exponent += float(spec["beta"]) * act * positive_log_width
        return float(np.exp(exponent))

    kin._blend_observable_width_ratio = types.MethodType(patched, kin)


def _patch_gnorm_asym(kin, g_beta: float, g_center: float, g_left_width: float, g_right_width: float):
    import types

    def patched(self, hll_diag: np.ndarray, direct_diag: np.ndarray, direct_b123: np.ndarray) -> float:
        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        width = g_left_width if metric <= g_center else g_right_width
        activation = float(max(0.0, 1.0 - abs(metric - g_center) / width))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(g_beta * activation * b2_gate)

    kin._runtime_direct_gnorm_blend_weight = types.MethodType(patched, kin)


def _design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    x1 = df["g_beta"].to_numpy()
    x2 = df["g_center"].to_numpy()
    x3 = df["g_left_width"].to_numpy()
    x4 = df["g_right_width"].to_numpy()
    cols = [
        np.ones(len(df)),
        x1, x2, x3, x4,
        x1 * x1, x2 * x2, x3 * x3, x4 * x4,
        x1 * x2, x1 * x3, x1 * x4, x2 * x3, x2 * x4, x3 * x4,
    ]
    names = [
        "const",
        "g_beta", "g_center", "g_left_width", "g_right_width",
        "g_beta2", "g_center2", "g_left_width2", "g_right_width2",
        "g_beta:g_center", "g_beta:g_left_width", "g_beta:g_right_width",
        "g_center:g_left_width", "g_center:g_right_width", "g_left_width:g_right_width",
    ]
    return np.column_stack(cols), names


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    df = pd.read_csv(MAP)
    df["D"] = df["D"].astype(float)
    df["eta"] = df["eta"].astype(float)
    d_grid = np.array(sorted(df["D"].unique()), dtype=float)
    target_D = float(d_grid[int(np.argmin(np.abs(d_grid - 6.4)))])
    sub = df[np.isclose(df["D"], target_D)].copy()

    sampler = qmc.Sobol(d=4, scramble=True, seed=6406)
    m = int(np.log2(N_SAMPLES))
    if 2 ** m != N_SAMPLES:
        raise SystemExit("N_SAMPLES must be a power of two")
    unit = sampler.random_base2(m=m)
    lowers = np.array([BOUNDS["g_beta"][0], BOUNDS["g_center"][0], BOUNDS["g_left_width"][0], BOUNDS["g_right_width"][0]])
    uppers = np.array([BOUNDS["g_beta"][1], BOUNDS["g_center"][1], BOUNDS["g_left_width"][1], BOUNDS["g_right_width"][1]])
    pts = qmc.scale(unit, lowers, uppers)

    rows = []
    for i, (g_beta, g_center, g_left_width, g_right_width) in enumerate(pts, start=1):
        kin = _build_kinetics(float(d_grid.min()), float(d_grid.max()), int(len(d_grid)))
        _patch_width_bands(kin)
        _patch_gnorm_asym(kin, float(g_beta), float(g_center), float(g_left_width), float(g_right_width))
        detail_rows = []
        for row in sub.itertuples(index=False):
            mu = float(
                kin.hll_mu_pred(
                    2,
                    D=float(row.D),
                    eta=float(row.eta),
                    t_coh=T_COH,
                    ref_D=REF_D,
                    ref_eta=REF_ETA,
                    observable_mode="eft_wilson_uv_rge",
                    N_max=kin.params.hll_observable_nmax,
                )
            )
            chi2 = float(((mu - mu_obs) / sigma_obs) ** 2)
            detail_rows.append({
                "abs_delta_mu_mumu": float(abs(mu - float(row.mu_mumu))),
                "acceptance_mismatch": float((float(row.chi2_mumu) <= 4.0) != (chi2 <= 4.0)),
            })
        dd = pd.DataFrame(detail_rows)
        row = {
            "sample_id": i,
            "g_beta": float(g_beta),
            "g_center": float(g_center),
            "g_left_width": float(g_left_width),
            "g_right_width": float(g_right_width),
            "p95_abs_delta_mu_mumu": float(np.percentile(dd["abs_delta_mu_mumu"], 95.0)),
            "max_abs_delta_mu_mumu": float(dd["abs_delta_mu_mumu"].max()),
            "mismatch": float(dd["acceptance_mismatch"].mean()),
        }
        row["score"] = float(row["p95_abs_delta_mu_mumu"] + 25.0 * row["mismatch"])
        rows.append(row)
        pd.DataFrame(rows).sort_values("score").to_csv(OUT_SAMPLES, index=False)
        print(
            f"[progress] {i}/{len(pts)} beta={g_beta:.6f} center={g_center:.6f} left={g_left_width:.6f} right={g_right_width:.6f} "
            f"p95={row['p95_abs_delta_mu_mumu']:.6f} mismatch={row['mismatch']:.6f}",
            flush=True,
        )

    samples = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    X, names = _design_matrix(samples)
    y = samples["p95_abs_delta_mu_mumu"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    effects = pd.DataFrame({"term": names, "coef": coef})
    effects["abs_coef"] = effects["coef"].abs()
    effects = effects.sort_values("abs_coef", ascending=False)
    effects.to_csv(OUT_EFFECTS, index=False)

    rng = np.random.default_rng(6464)
    cloud = np.column_stack([
        rng.uniform(*BOUNDS["g_beta"], size=4000),
        rng.uniform(*BOUNDS["g_center"], size=4000),
        rng.uniform(*BOUNDS["g_left_width"], size=4000),
        rng.uniform(*BOUNDS["g_right_width"], size=4000),
    ])
    cloud_df = pd.DataFrame(cloud, columns=["g_beta", "g_center", "g_left_width", "g_right_width"])
    Xc, _ = _design_matrix(cloud_df)
    cloud_df["pred_p95"] = Xc @ coef
    best_pred = cloud_df.sort_values("pred_p95").iloc[0].to_dict()
    best_obs = samples.iloc[0].to_dict()

    summary = {
        "current_reference": CURRENT,
        "bounds": BOUNDS,
        "n_samples": int(len(samples)),
        "best_observed": best_obs,
        "top_effects": effects.head(8).to_dict(orient="records"),
        "predicted_next_best": best_pred,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    for path in [OUT_SAMPLES, OUT_EFFECTS, OUT_SUMMARY]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {OUT_SAMPLES}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

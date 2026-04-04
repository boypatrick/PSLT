#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv"
V10_DETAIL = ROOT / "output" / "kinetic_action_chain" / "runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64d48_v10_casesplit.csv"

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
GNORM_AMP = float(os.environ.get("GNORM_AMP", "0.2468174085"))
LORENTZ_SHAPE_BETA = float(os.environ.get("LORENTZ_SHAPE_BETA", "0.243"))

OUT_TAG = os.environ.get("OUT_TAG", "d60_d64_lorentz_asym_probe_v1")
OUT_SAMPLES = OUTDIR / f"runtime_direct_detlin_{OUT_TAG}_samples.csv"
OUT_SUMMARY = OUTDIR / f"runtime_direct_detlin_{OUT_TAG}_summary.json"

PROBES = [
    {"probe_id": "P0", "label": "symmetric baseline", "g_center": 0.048897, "g_hw_left": 0.02526, "g_hw_right": 0.02526},
    {"probe_id": "P1", "label": "mild right-heavy", "g_center": 0.048897, "g_hw_left": 0.02400, "g_hw_right": 0.02650},
    {"probe_id": "P2", "label": "mild left-heavy", "g_center": 0.048897, "g_hw_left": 0.02650, "g_hw_right": 0.02400},
    {"probe_id": "P3", "label": "strong right-heavy", "g_center": 0.048897, "g_hw_left": 0.02350, "g_hw_right": 0.02700},
    {"probe_id": "P4", "label": "strong left-heavy", "g_center": 0.048897, "g_hw_left": 0.02700, "g_hw_right": 0.02350},
    {"probe_id": "P5", "label": "balanced wide", "g_center": 0.048907, "g_hw_left": 0.02580, "g_hw_right": 0.02580},
    {"probe_id": "P6", "label": "balanced narrow", "g_center": 0.048887, "g_hw_left": 0.02470, "g_hw_right": 0.02470},
]


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


def _patch_width_bands(kin) -> None:
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


def _patch_lorentz_asym(kin, g_center: float, g_hw_left: float, g_hw_right: float) -> None:
    def patched(self, hll_diag: np.ndarray, direct_diag: np.ndarray, direct_b123: np.ndarray) -> float:
        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        width = float(g_hw_left if metric < g_center else g_hw_right)
        if width <= 0.0:
            return 0.0
        gate = 1.0 / (1.0 + abs((metric - g_center) / width) ** (2.0 * LORENTZ_SHAPE_BETA))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(GNORM_AMP * gate * b2_gate)

    kin._runtime_direct_gnorm_blend_weight = types.MethodType(patched, kin)


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

    v10 = pd.read_csv(V10_DETAIL)
    v10 = v10[(v10["case"] == "D60E21_release") & np.isclose(v10["eval_D"], target_D)]
    current_p95 = float(np.percentile(v10["abs_delta_mu_mumu"], 95.0))
    current_max = float(v10["abs_delta_mu_mumu"].max())

    rows = []
    for spec in PROBES:
        kin = _build_kinetics(float(d_grid.min()), float(d_grid.max()), int(len(d_grid)))
        _patch_width_bands(kin)
        _patch_lorentz_asym(kin, spec["g_center"], spec["g_hw_left"], spec["g_hw_right"])
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
            detail_rows.append(
                {
                    "abs_delta_mu_mumu": float(abs(mu - float(row.mu_mumu))),
                    "acceptance_mismatch": float((float(row.chi2_mumu) <= 4.0) != (chi2 <= 4.0)),
                }
            )
        dd = pd.DataFrame(detail_rows)
        rows.append(
            {
                **spec,
                "g_lorentz_beta": float(LORENTZ_SHAPE_BETA),
                "g_amp": float(GNORM_AMP),
                "p95_abs_delta_mu_mumu": float(np.percentile(dd["abs_delta_mu_mumu"], 95.0)),
                "max_abs_delta_mu_mumu": float(dd["abs_delta_mu_mumu"].max()),
                "mismatch": float(dd["acceptance_mismatch"].mean()),
                "delta_p95_vs_v10": float(np.percentile(dd["abs_delta_mu_mumu"], 95.0) - current_p95),
                "delta_max_vs_v10": float(dd["abs_delta_mu_mumu"].max() - current_max),
            }
        )
        print(
            f"[probe] {spec['probe_id']} center={spec['g_center']:.6f} left={spec['g_hw_left']:.6f} right={spec['g_hw_right']:.6f} "
            f"p95={rows[-1]['p95_abs_delta_mu_mumu']:.6f} delta_vs_v10={rows[-1]['delta_p95_vs_v10']:.6f}",
            flush=True,
        )

    samples = pd.DataFrame(rows).sort_values(["p95_abs_delta_mu_mumu", "max_abs_delta_mu_mumu", "probe_id"]).reset_index(drop=True)
    samples.to_csv(OUT_SAMPLES, index=False)
    summary = {
        "target_D": target_D,
        "baseline_v10": {
            "p95_abs_delta_mu_mumu": current_p95,
            "max_abs_delta_mu_mumu": current_max,
        },
        "best_probe": samples.iloc[0].to_dict(),
        "all_probes": samples.to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    for path in [OUT_SAMPLES, OUT_SUMMARY]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {OUT_SAMPLES}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

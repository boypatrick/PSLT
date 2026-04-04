#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv"
V10_DETAIL = ROOT / "output" / "kinetic_action_chain" / "runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64d48_v10_casesplit.csv"

OUT_SAMPLES = OUTDIR / "runtime_direct_detlin_d60_d64_local_amp_probe_v1_samples.csv"
OUT_SUMMARY = OUTDIR / "runtime_direct_detlin_d60_d64_local_amp_probe_v1_summary.json"

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

D21_G_BETA = 0.30
D21_G_CENTER = 0.06
D21_G_HALF_WIDTH = 0.04

D60_G_BETA = 0.2468174085
D60_G_CENTER = 0.0488372924
D60_G_HALF_WIDTH = 0.0251312702

D40_BETA = 1.304
D48_BETA = 0.55
D72_BETA = 0.35
D80_BETA = 1.20

TARGET_CENTER = 6.4
TARGET_HALF_WIDTH = 0.40
ALPHAS = [-0.30, -0.20, -0.10, -0.05, -0.02, 0.0, 0.02]


def _patch_width_bands(kin, alpha: float) -> None:
    base_bands = [
        {"beta": D40_BETA, "center": 4.0, "half_width": 0.40},
        {"beta": D48_BETA, "center": 4.8, "half_width": 0.40},
        {"beta": D72_BETA, "center": 7.2, "half_width": 0.40},
        {"beta": D80_BETA, "center": 8.0, "half_width": 0.40},
    ]
    original = kin._blend_observable_width_ratio

    def patched_width(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        exponent = np.log(max(base, floor))
        for spec in base_bands:
            act = float(max(0.0, 1.0 - abs(float(D) - float(spec["center"])) / float(spec["half_width"])))
            exponent += float(spec["beta"]) * act * positive_log_width
        extra_act = float(max(0.0, 1.0 - abs(float(D) - TARGET_CENTER) / TARGET_HALF_WIDTH))
        exponent += float(alpha) * extra_act * positive_log_width
        return float(np.exp(exponent))

    kin._blend_observable_width_ratio = types.MethodType(patched_width, kin)


def _patch_gnorm(kin) -> None:
    def patched_gnorm(self, hll_diag, direct_diag, direct_b123):
        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        activation = float(max(0.0, 1.0 - abs(metric - D60_G_CENTER) / D60_G_HALF_WIDTH))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(D60_G_BETA * activation * b2_gate)

    kin._runtime_direct_gnorm_blend_weight = types.MethodType(patched_gnorm, kin)


def build_kinetics(alpha: float, d_min: float, d_max: float, d_num: int):
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
    _patch_gnorm(kin)
    _patch_width_bands(kin, alpha)
    return kin


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MAP)
    target_D = float(sorted(df["D"].unique(), key=lambda x: abs(float(x) - TARGET_CENTER))[0])
    sub = df[np.isclose(df["D"], target_D)].copy()

    v10 = pd.read_csv(V10_DETAIL)
    v10 = v10[(v10["case"] == "D60E21_release") & np.isclose(v10["eval_D"], target_D)]
    baseline_p95 = float(np.percentile(v10["abs_delta_mu_mumu"], 95.0))
    baseline_max = float(v10["abs_delta_mu_mumu"].max())

    rows = []
    for i, alpha in enumerate(ALPHAS, start=1):
        kin = build_kinetics(alpha, float(df["D"].min()), float(df["D"].max()), int(df["D"].nunique()))
        deltas = []
        signed = []
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
            diff = mu - float(row.mu_mumu)
            deltas.append(abs(diff))
            signed.append(diff)
        result = {
            "alpha": float(alpha),
            "p95_abs_delta_mu_mumu": float(np.percentile(deltas, 95.0)),
            "max_abs_delta_mu_mumu": float(np.max(deltas)),
            "mean_signed_delta_mu_mumu": float(np.mean(signed)),
            "min_signed_delta_mu_mumu": float(np.min(signed)),
            "max_signed_delta_mu_mumu": float(np.max(signed)),
            "delta_p95_vs_v10": float(np.percentile(deltas, 95.0) - baseline_p95),
            "delta_max_vs_v10": float(np.max(deltas) - baseline_max),
        }
        rows.append(result)
        print(
            f"[progress] {i}/{len(ALPHAS)} alpha={alpha:+.3f} "
            f"p95={result['p95_abs_delta_mu_mumu']:.6f} "
            f"delta_vs_v10={result['delta_p95_vs_v10']:+.6f} "
            f"mean_signed={result['mean_signed_delta_mu_mumu']:+.6f}"
        )

    with OUT_SAMPLES.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda r: r["p95_abs_delta_mu_mumu"])
    summary = {
        "target_D": float(target_D),
        "baseline_v10": {
            "p95_abs_delta_mu_mumu": baseline_p95,
            "max_abs_delta_mu_mumu": baseline_max,
        },
        "best_probe": best,
        "all_probes": rows,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    for path in [OUT_SAMPLES, OUT_SUMMARY]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

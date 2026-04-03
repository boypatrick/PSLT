#!/usr/bin/env python3
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAPDIR = ROOT / "output" / "hll_signal_strength"

CASES = [
    ("D21E21_holdout", "D21 x E21 holdout", MAPDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E21_20260330_crossgrid_holdout_r1.csv"),
    ("D60E21_release", "D60 x E21 release", MAPDIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv"),
]
TARGETS = [4.0, 4.8, 6.4, 7.2, 8.0]
RETUNE_D48 = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
RETUNE_GNORM_BETA = [0.10, 0.15, 0.20, 0.25, 0.30]
RETUNE_GNORM_CENTER = [0.04, 0.06, 0.08]
RETUNE_GNORM_HALF_WIDTH = [0.03, 0.04, 0.05]

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

WIDTH_BANDS_BASE = [
    {"beta": 1.309, "center": 4.0, "half_width": 0.40},
    {"beta": 0.75, "center": 4.8, "half_width": 0.40},
    {"beta": 0.35, "center": 7.2, "half_width": 0.40},
    {"beta": 1.20, "center": 8.0, "half_width": 0.40},
]
GNORM_BASE = {"beta": 0.25, "center": 0.06, "half_width": 0.04}

OUT_PHASE1 = OUTDIR / "runtime_direct_detlin_generalization_retune48_phase1.csv"
OUT_PHASE2 = OUTDIR / "runtime_direct_detlin_generalization_retune64_phase2.csv"
OUT_FINAL = OUTDIR / "runtime_direct_detlin_generalization_retune48_64_final.csv"
OUT_DETAIL = OUTDIR / "runtime_direct_detlin_generalization_retune48_64_best_detail.csv"
OUT_DECISION = OUTDIR / "runtime_direct_detlin_generalization_retune48_64_decision.json"


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


def _patch_gnorm_bandpass(kin, spec: dict[str, float]) -> None:
    beta = float(spec["beta"])
    center = float(spec["center"])
    half_width = float(spec["half_width"])

    def patched(self, hll_diag: np.ndarray, direct_diag: np.ndarray, direct_b123: np.ndarray) -> float:
        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        activation = float(max(0.0, 1.0 - abs(metric - center) / half_width))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(beta * activation * b2_gate)

    kin._runtime_direct_gnorm_blend_weight = types.MethodType(patched, kin)


def _patch_width_bands(kin, bands: list[dict[str, float]]) -> None:
    original = kin._blend_observable_width_ratio

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


def _evaluate_case(
    case: tuple[str, str, Path],
    width_bands: list[dict[str, float]],
    gnorm_spec: dict[str, float],
    *,
    requested_targets: list[float],
) -> pd.DataFrame:
    case_name, case_label, path = case
    df = pd.read_csv(path)
    df["D"] = df["D"].astype(float)
    df["eta"] = df["eta"].astype(float)
    d_grid = sorted(df["D"].unique())
    slice_ds = sorted({float(d_grid[int(np.argmin(np.abs(np.asarray(d_grid) - t)))]) for t in requested_targets})
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)
    kin = _build_kinetics(min(d_grid), max(d_grid), len(d_grid))
    _patch_gnorm_bandpass(kin, gnorm_spec)
    _patch_width_bands(kin, width_bands)

    rows = []
    for D in slice_ds:
        sub = df[np.isclose(df["D"], D)]
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
            rows.append(
                {
                    "case": case_name,
                    "case_label": case_label,
                    "target_D": float(min(requested_targets, key=lambda t: abs(t - float(row.D)))),
                    "eval_D": float(row.D),
                    "eta": float(row.eta),
                    "abs_delta_mu_mumu": float(abs(mu - float(row.mu_mumu))),
                    "acceptance_mismatch": float((float(row.chi2_mumu) <= 4.0) != (chi2 <= 4.0)),
                }
            )
    return pd.DataFrame(rows)


def _score(sub_df: pd.DataFrame) -> float:
    if sub_df.empty:
        return float("inf")
    mismatch = float(sub_df["acceptance_mismatch"].mean())
    p95 = float(np.percentile(sub_df["abs_delta_mu_mumu"], 95.0))
    return p95 + 25.0 * mismatch


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)

    phase1_rows = []
    for beta48 in RETUNE_D48:
        width_bands = [dict(spec) for spec in WIDTH_BANDS_BASE]
        for spec in width_bands:
            if np.isclose(spec["center"], 4.8):
                spec["beta"] = float(beta48)
        detail = pd.concat(
            [_evaluate_case(case, width_bands, GNORM_BASE, requested_targets=[4.8]) for case in CASES],
            ignore_index=True,
        )
        for case_name, _, _ in CASES:
            sub = detail[(detail["case"] == case_name) & (np.isclose(detail["target_D"], 4.8))]
            phase1_rows.append(
                {
                    "d48_beta": float(beta48),
                    "case": case_name,
                    "mismatch": float(sub["acceptance_mismatch"].mean()),
                    "p95_abs_delta_mu_mumu": float(np.percentile(sub["abs_delta_mu_mumu"], 95.0)),
                    "max_abs_delta_mu_mumu": float(sub["abs_delta_mu_mumu"].max()),
                    "score": _score(sub),
                }
            )
    phase1 = pd.DataFrame(phase1_rows)
    phase1.to_csv(OUT_PHASE1, index=False)
    agg1 = (
        phase1.groupby("d48_beta")
        .agg(
            worst_score=("score", "max"),
            mean_score=("score", "mean"),
            worst_p95=("p95_abs_delta_mu_mumu", "max"),
            worst_mismatch=("mismatch", "max"),
        )
        .reset_index()
        .sort_values(["worst_score", "mean_score", "d48_beta"])
    )
    best_beta48 = float(agg1.iloc[0]["d48_beta"])

    phase2_rows = []
    top1_width_bands = [dict(spec) for spec in WIDTH_BANDS_BASE]
    for spec in top1_width_bands:
        if np.isclose(spec["center"], 4.8):
            spec["beta"] = best_beta48
    for beta in RETUNE_GNORM_BETA:
        for center in RETUNE_GNORM_CENTER:
            for half_width in RETUNE_GNORM_HALF_WIDTH:
                gspec = {"beta": float(beta), "center": float(center), "half_width": float(half_width)}
                detail = pd.concat(
                    [_evaluate_case(case, top1_width_bands, gspec, requested_targets=[6.4]) for case in CASES],
                    ignore_index=True,
                )
                for case_name, _, _ in CASES:
                    sub = detail[(detail["case"] == case_name) & (np.isclose(detail["target_D"], 6.4))]
                    phase2_rows.append(
                        {
                            "d48_beta": best_beta48,
                            "g_beta": float(beta),
                            "g_center": float(center),
                            "g_half_width": float(half_width),
                            "case": case_name,
                            "mismatch": float(sub["acceptance_mismatch"].mean()),
                            "p95_abs_delta_mu_mumu": float(np.percentile(sub["abs_delta_mu_mumu"], 95.0)),
                            "max_abs_delta_mu_mumu": float(sub["abs_delta_mu_mumu"].max()),
                            "score": _score(sub),
                        }
                    )
    phase2 = pd.DataFrame(phase2_rows)
    phase2.to_csv(OUT_PHASE2, index=False)
    agg2 = (
        phase2.groupby(["g_beta", "g_center", "g_half_width"])
        .agg(
            worst_score=("score", "max"),
            mean_score=("score", "mean"),
            worst_p95=("p95_abs_delta_mu_mumu", "max"),
            worst_mismatch=("mismatch", "max"),
        )
        .reset_index()
        .sort_values(["worst_score", "mean_score", "g_beta", "g_center", "g_half_width"])
    )
    best_g = agg2.iloc[0].to_dict()
    best_gspec = {
        "beta": float(best_g["g_beta"]),
        "center": float(best_g["g_center"]),
        "half_width": float(best_g["g_half_width"]),
    }

    final_rows = []
    candidates = []
    for _, row in agg2.head(3).iterrows():
        candidates.append(
            {
                "d48_beta": best_beta48,
                "g_beta": float(row["g_beta"]),
                "g_center": float(row["g_center"]),
                "g_half_width": float(row["g_half_width"]),
            }
        )

    best_detail = None
    for cand in candidates:
        width_bands = [dict(spec) for spec in WIDTH_BANDS_BASE]
        for spec in width_bands:
            if np.isclose(spec["center"], 4.8):
                spec["beta"] = float(cand["d48_beta"])
        gspec = {
            "beta": float(cand["g_beta"]),
            "center": float(cand["g_center"]),
            "half_width": float(cand["g_half_width"]),
        }
        detail = pd.concat(
            [_evaluate_case(case, width_bands, gspec, requested_targets=TARGETS) for case in CASES],
            ignore_index=True,
        )
        if best_detail is None and gspec == best_gspec:
            best_detail = detail.copy()
        for case_name, _, _ in CASES:
            for target in TARGETS:
                sub = detail[(detail["case"] == case_name) & (np.isclose(detail["target_D"], target))]
                final_rows.append(
                    {
                        **cand,
                        "case": case_name,
                        "target_D": float(target),
                        "mismatch": float(sub["acceptance_mismatch"].mean()),
                        "p95_abs_delta_mu_mumu": float(np.percentile(sub["abs_delta_mu_mumu"], 95.0)),
                        "max_abs_delta_mu_mumu": float(sub["abs_delta_mu_mumu"].max()),
                        "score": _score(sub),
                    }
                )
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUT_FINAL, index=False)

    if best_detail is not None:
        best_detail.to_csv(OUT_DETAIL, index=False)

    decision = {
        "best_d48_beta": best_beta48,
        "best_gnorm": best_gspec,
        "phase1_top": agg1.head(5).to_dict(orient="records"),
        "phase2_top": agg2.head(5).to_dict(orient="records"),
    }
    OUT_DECISION.write_text(json.dumps(decision, indent=2))
    for path in [OUT_PHASE1, OUT_PHASE2, OUT_FINAL, OUT_DETAIL, OUT_DECISION]:
        if path.exists():
            (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {OUT_FINAL}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

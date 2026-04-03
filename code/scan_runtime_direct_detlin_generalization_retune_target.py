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
import os

RETUNE_TARGET = float(os.environ.get("RETUNE_TARGET", "4.8"))
BASE_D48_BETA = float(os.environ.get("BASE_D48_BETA", "0.75"))
RAW_CASE_FILTER = os.environ.get("CASE_FILTER", "").strip()
RAW_OUT_TAG = os.environ.get("OUT_TAG", "").strip()
RAW_ETA_VALUES = os.environ.get("ETA_VALUES", "").strip()
ETA_SAMPLE = int(os.environ.get("ETA_SAMPLE", "0"))
if RAW_CASE_FILTER:
    WANTED_CASES = {tok.strip() for tok in RAW_CASE_FILTER.split(",") if tok.strip()}
    CASES = [case for case in CASES if case[0] in WANTED_CASES]

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

WIDTH_BANDS = [
    {"beta": 1.309, "center": 4.0, "half_width": 0.40},
    {"beta": BASE_D48_BETA, "center": 4.8, "half_width": 0.40},
    {"beta": 0.35, "center": 7.2, "half_width": 0.40},
    {"beta": 1.20, "center": 8.0, "half_width": 0.40},
]
GNORM_BASE = {"beta": 0.25, "center": 0.06, "half_width": 0.04}

if np.isclose(RETUNE_TARGET, 4.0):
    raw = os.environ.get("D40_CANDIDATES", "").strip()
    vals = [float(x) for x in raw.split(",") if x.strip()] if raw else [0.9, 1.1, 1.309, 1.5]
    CANDIDATES = [{"d40_beta": x} for x in vals]
    OUT_STEM = "runtime_direct_detlin_generalization_retune_D4p0"
elif np.isclose(RETUNE_TARGET, 4.8):
    raw = os.environ.get("D48_CANDIDATES", "").strip()
    vals = [float(x) for x in raw.split(",") if x.strip()] if raw else [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    CANDIDATES = [{"d48_beta": x} for x in vals]
    OUT_STEM = "runtime_direct_detlin_generalization_retune_D4p8"
elif np.isclose(RETUNE_TARGET, 6.4):
    raw_b = os.environ.get("G_BETA_CANDIDATES", "").strip()
    raw_c = os.environ.get("G_CENTER_CANDIDATES", "").strip()
    raw_w = os.environ.get("G_WIDTH_CANDIDATES", "").strip()
    betas = [float(x) for x in raw_b.split(",") if x.strip()] if raw_b else [0.10, 0.15, 0.20, 0.25, 0.30]
    centers = [float(x) for x in raw_c.split(",") if x.strip()] if raw_c else [0.04, 0.06, 0.08]
    widths = [float(x) for x in raw_w.split(",") if x.strip()] if raw_w else [0.03, 0.04, 0.05]
    CANDIDATES = [{"g_beta": b, "g_center": c, "g_half_width": w} for b in betas for c in centers for w in widths]
    OUT_STEM = "runtime_direct_detlin_generalization_retune_D6p4"
else:
    raise SystemExit(f"Unsupported RETUNE_TARGET={RETUNE_TARGET}")

tag_parts = []
if RAW_CASE_FILTER:
    tag_parts.append(RAW_CASE_FILTER.replace(",", "_"))
if RAW_OUT_TAG:
    tag_parts.append(RAW_OUT_TAG)
tag_suffix = f"_{'_'.join(tag_parts)}" if tag_parts else ""

OUT_SUMMARY = OUTDIR / f"{OUT_STEM}_summary{tag_suffix}.csv"
OUT_DETAIL = OUTDIR / f"{OUT_STEM}_detail{tag_suffix}.csv"
OUT_DECISION = OUTDIR / f"{OUT_STEM}_decision{tag_suffix}.json"


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


def _score(sub_df: pd.DataFrame) -> float:
    mismatch = float(sub_df["acceptance_mismatch"].mean())
    p95 = float(np.percentile(sub_df["abs_delta_mu_mumu"], 95.0))
    return p95 + 25.0 * mismatch


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    rows = []
    for cand in CANDIDATES:
        print(f"[candidate] target_D={RETUNE_TARGET} spec={cand}", flush=True)
        width_bands = [dict(spec) for spec in WIDTH_BANDS]
        gspec = dict(GNORM_BASE)
        if np.isclose(RETUNE_TARGET, 4.0):
            for spec in width_bands:
                if np.isclose(spec["center"], 4.0):
                    spec["beta"] = float(cand["d40_beta"])
        elif np.isclose(RETUNE_TARGET, 4.8):
            for spec in width_bands:
                if np.isclose(spec["center"], 4.8):
                    spec["beta"] = float(cand["d48_beta"])
        else:
            gspec = {"beta": float(cand["g_beta"]), "center": float(cand["g_center"]), "half_width": float(cand["g_half_width"])}

        for case_name, case_label, path in CASES:
            df = pd.read_csv(path)
            df["D"] = df["D"].astype(float)
            df["eta"] = df["eta"].astype(float)
            d_grid = sorted(df["D"].unique())
            target_D = float(d_grid[int(np.argmin(np.abs(np.asarray(d_grid) - RETUNE_TARGET)))])
            sub = df[np.isclose(df["D"], target_D)]
            if RAW_ETA_VALUES:
                wanted_etas = np.array([float(x) for x in RAW_ETA_VALUES.split(",") if x.strip()], dtype=float)
                eta_grid = np.sort(sub["eta"].unique())
                snapped = sorted({float(eta_grid[int(np.argmin(np.abs(eta_grid - eta)))]) for eta in wanted_etas})
                sub = sub[sub["eta"].isin(snapped)]
            elif ETA_SAMPLE > 0 and len(sub) > ETA_SAMPLE:
                eta_grid = np.sort(sub["eta"].unique())
                idx = np.linspace(0, len(eta_grid) - 1, ETA_SAMPLE).round().astype(int)
                snapped = sorted({float(eta_grid[i]) for i in idx})
                sub = sub[sub["eta"].isin(snapped)]
            kin = _build_kinetics(min(d_grid), max(d_grid), len(d_grid))
            _patch_gnorm_bandpass(kin, gspec)
            _patch_width_bands(kin, width_bands)
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
                    "target_D": RETUNE_TARGET,
                    "eval_D": target_D,
                    "case": case_name,
                    "case_label": case_label,
                    **cand,
                    "mismatch": float(dd["acceptance_mismatch"].mean()),
                    "p95_abs_delta_mu_mumu": float(np.percentile(dd["abs_delta_mu_mumu"], 95.0)),
                    "max_abs_delta_mu_mumu": float(dd["abs_delta_mu_mumu"].max()),
                    "score": _score(dd),
                }
            )
            partial = pd.DataFrame(rows)
            partial.to_csv(OUT_SUMMARY, index=False)
            if np.isclose(RETUNE_TARGET, 4.0):
                agg_partial = (
                    partial.groupby("d40_beta")
                    .agg(
                        worst_score=("score", "max"),
                        mean_score=("score", "mean"),
                        worst_p95=("p95_abs_delta_mu_mumu", "max"),
                        worst_mismatch=("mismatch", "max"),
                    )
                    .reset_index()
                    .sort_values(["worst_score", "mean_score", "d40_beta"])
                )
            elif np.isclose(RETUNE_TARGET, 4.8):
                agg_partial = (
                    partial.groupby("d48_beta")
                    .agg(
                        worst_score=("score", "max"),
                        mean_score=("score", "mean"),
                        worst_p95=("p95_abs_delta_mu_mumu", "max"),
                        worst_mismatch=("mismatch", "max"),
                    )
                    .reset_index()
                    .sort_values(["worst_score", "mean_score", "d48_beta"])
                )
            else:
                agg_partial = (
                    partial.groupby(["g_beta", "g_center", "g_half_width"])
                    .agg(
                        worst_score=("score", "max"),
                        mean_score=("score", "mean"),
                        worst_p95=("p95_abs_delta_mu_mumu", "max"),
                        worst_mismatch=("mismatch", "max"),
                    )
                    .reset_index()
                    .sort_values(["worst_score", "mean_score", "g_beta", "g_center", "g_half_width"])
                )
            agg_partial.to_csv(OUT_DETAIL, index=False)
            OUT_DECISION.write_text(
                json.dumps(
                    {
                        "target_D": RETUNE_TARGET,
                        "completed_rows": len(rows),
                        "completed_candidates": int(partial.drop_duplicates(list(cand.keys())).shape[0]),
                        "top": agg_partial.head(10).to_dict(orient="records"),
                    },
                    indent=2,
                )
            )
            for path in [OUT_SUMMARY, OUT_DETAIL, OUT_DECISION]:
                (PAPERDIR / path.name).write_bytes(path.read_bytes())
            print(
                f"[progress] case={case_name} eval_D={target_D:.1f} p95={rows[-1]['p95_abs_delta_mu_mumu']:.6f} mismatch={rows[-1]['mismatch']:.6f}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    if np.isclose(RETUNE_TARGET, 4.0):
        agg = (
            summary.groupby("d40_beta")
            .agg(worst_score=("score", "max"), mean_score=("score", "mean"), worst_p95=("p95_abs_delta_mu_mumu", "max"))
            .reset_index()
            .sort_values(["worst_score", "mean_score", "d40_beta"])
        )
    elif np.isclose(RETUNE_TARGET, 4.8):
        agg = (
            summary.groupby("d48_beta")
            .agg(worst_score=("score", "max"), mean_score=("score", "mean"), worst_p95=("p95_abs_delta_mu_mumu", "max"))
            .reset_index()
            .sort_values(["worst_score", "mean_score", "d48_beta"])
        )
    else:
        agg = (
            summary.groupby(["g_beta", "g_center", "g_half_width"])
            .agg(worst_score=("score", "max"), mean_score=("score", "mean"), worst_p95=("p95_abs_delta_mu_mumu", "max"))
            .reset_index()
            .sort_values(["worst_score", "mean_score", "g_beta", "g_center", "g_half_width"])
        )
    agg.to_csv(OUT_DETAIL, index=False)
    decision = {"target_D": RETUNE_TARGET, "top": agg.head(10).to_dict(orient="records")}
    OUT_DECISION.write_text(json.dumps(decision, indent=2))
    for path in [OUT_SUMMARY, OUT_DETAIL, OUT_DECISION]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())
    print(f"[saved] {OUT_SUMMARY}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

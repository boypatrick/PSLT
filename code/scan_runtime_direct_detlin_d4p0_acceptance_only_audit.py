#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import types
from pathlib import Path

TMP_MPL = Path("/tmp/pslt_mplconfig")
TMP_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_MPL))

import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAPDIR = ROOT / "output" / "hll_signal_strength"

FULL_MAP = MAPDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv"
OUT_PREFIX = "runtime_direct_detlin_d4p0_acceptance_only_audit"

FOCUS_D = 4.0
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
GNORM_BANDPASS = {"beta": 0.25, "center": 0.06, "half_width": 0.04}
D4P8_WIDTH = {"beta": 0.75, "center": 4.8, "half_width": 0.40}


def _parse_beta_grid() -> list[float]:
    raw = os.environ.get("D4P0_BETA_GRID", "").strip()
    if not raw:
        return [1.308, 1.309, 1.310]
    return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]


def _load_map(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    df = pd.read_csv(path)
    rows: dict[tuple[float, float], dict[str, float]] = {}
    for row in df.itertuples(index=False):
        rows[(float(row.D), float(row.eta))] = {
            "mu_mumu": float(row.mu_mumu),
            "chi2_mumu": float(row.chi2_mumu),
        }
    return rows


def _build_kinetics():
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode="cell_direct_runtime_release_tailm2gnorm",
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )
    p = kin.params
    p.hll_match_width_power_base = float(WIDTH_FIX["width_power_base"])
    p.hll_match_width_power_log_ratio_coeff = float(WIDTH_FIX["width_power_log_ratio_coeff"])
    p.hll_match_width_power_min = float(WIDTH_FIX["width_power_min"])
    p.hll_match_width_power_max = float(WIDTH_FIX["width_power_max"])
    p.hll_match_width_power_trigger_lo = float(WIDTH_FIX["width_power_trigger_lo"])
    p.hll_match_width_power_trigger_hi = float(WIDTH_FIX["width_power_trigger_hi"])
    p.hll_match_width_power_turnoff_lo = float(WIDTH_FIX["width_power_turnoff_lo"])
    p.hll_match_width_power_turnoff_hi = float(WIDTH_FIX["width_power_turnoff_hi"])
    p.hll_match_width_power_tail_logratio_lo = float(WIDTH_FIX["width_power_tail_logratio_lo"])
    p.hll_match_width_power_tail_logratio_hi = float(WIDTH_FIX["width_power_tail_logratio_hi"])
    p.hll_match_width_power_tail_reboost_max = float(WIDTH_FIX["width_power_tail_reboost_max"])
    p.hll_uv_runtime_direct_m2_cross_blend = 0.75
    return kin


def _patch_gnorm_bandpass(kin) -> None:
    beta = float(GNORM_BANDPASS["beta"])
    center = float(GNORM_BANDPASS["center"])
    half_width = float(GNORM_BANDPASS["half_width"])

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


def _patch_width_bands(kin, beta_d4p0: float) -> None:
    original = kin._blend_observable_width_ratio

    def patched(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        exponent = np.log(max(base, floor))
        for spec in [D4P8_WIDTH, {"beta": beta_d4p0, "center": 4.0, "half_width": 0.40}]:
            act = float(max(0.0, 1.0 - abs(float(D) - float(spec["center"])) / float(spec["half_width"])))
            exponent += float(spec["beta"]) * act * positive_log_width
        return float(np.exp(exponent))

    kin._blend_observable_width_ratio = types.MethodType(patched, kin)


def _evaluate(beta_d4p0: float, full_map, eta_grid, mu_obs: float, sigma_obs: float) -> pd.DataFrame:
    kin = _build_kinetics()
    _patch_gnorm_bandpass(kin)
    _patch_width_bands(kin, beta_d4p0=beta_d4p0)
    rows = []
    for eta in eta_grid:
        ref = full_map[(FOCUS_D, float(eta))]
        mu = float(
            kin.hll_mu_pred(
                2,
                D=FOCUS_D,
                eta=float(eta),
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
                "focus_D": FOCUS_D,
                "beta_d4p0": float(beta_d4p0),
                "eta": float(eta),
                "mu_candidate": float(mu),
                "mu_full": float(ref["mu_mumu"]),
                "chi2_candidate": float(chi2),
                "chi2_full": float(ref["chi2_mumu"]),
                "abs_delta_mu_mumu": float(abs(mu - ref["mu_mumu"])),
                "acceptance_mismatch": float((ref["chi2_mumu"] <= 4.0) != (chi2 <= 4.0)),
            }
        )
    return pd.DataFrame(rows)


def _boundary_eta(df: pd.DataFrame, column: str) -> float:
    inside = df[df[column] <= 4.0]
    if inside.empty:
        return float("nan")
    return float(inside["eta"].max())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    beta_grid = _parse_beta_grid()
    beta_tag = "single_" + str(beta_grid[0]).replace(".", "p") if len(beta_grid) == 1 else "grid"
    out_summary = OUTDIR / f"{OUT_PREFIX}_summary_{beta_tag}.csv"
    out_detail = OUTDIR / f"{OUT_PREFIX}_detail_{beta_tag}.csv"
    out_meta = OUTDIR / f"{OUT_PREFIX}_run_meta_{beta_tag}.json"

    full_map = _load_map(FULL_MAP)
    eta_grid = sorted({float(key[1]) for key in full_map})
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    detail_frames = []
    summary_rows = []
    for beta in beta_grid:
        df = _evaluate(beta_d4p0=beta, full_map=full_map, eta_grid=eta_grid, mu_obs=mu_obs, sigma_obs=sigma_obs)
        detail_frames.append(df)
        mismatches = df[df["acceptance_mismatch"] > 0.0]
        summary_rows.append(
            {
                "focus_D": FOCUS_D,
                "beta_d4p0": float(beta),
                "acceptance_mismatch": float(df["acceptance_mismatch"].mean()),
                "p95_abs_delta_mu_mumu": float(np.percentile(df["abs_delta_mu_mumu"], 95.0)),
                "max_abs_delta_mu_mumu": float(df["abs_delta_mu_mumu"].max()),
                "candidate_boundary_eta": _boundary_eta(df, "chi2_candidate"),
                "full_boundary_eta": _boundary_eta(df, "chi2_full"),
                "n_mismatches": int(len(mismatches)),
                "first_mismatch_eta": float(mismatches["eta"].min()) if not mismatches.empty else float("nan"),
                "last_mismatch_eta": float(mismatches["eta"].max()) if not mismatches.empty else float("nan"),
            }
        )

    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["acceptance_mismatch", "p95_abs_delta_mu_mumu", "beta_d4p0"])

    detail_df.to_csv(out_detail, index=False)
    summary_df.to_csv(out_summary, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "focus_D": FOCUS_D,
                "beta_grid": beta_grid,
                "gnorm_bandpass": GNORM_BANDPASS,
                "d4p8_width": D4P8_WIDTH,
                "width_fix": WIDTH_FIX,
            },
            indent=2,
        )
    )
    for path in [out_summary, out_detail, out_meta]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    best = summary_df.iloc[0].to_dict()
    print(f"[saved] {out_summary}")
    print(best)


if __name__ == "__main__":
    main()

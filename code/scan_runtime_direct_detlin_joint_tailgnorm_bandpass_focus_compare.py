#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MPLDIR = ROOT / ".mplconfig"
MPLDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics


OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
MAPDIR = ROOT / "output" / "hll_signal_strength"

FULL_MAP = MAPDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv"
OUT_SUMMARY = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_focus_compare_summary.csv"
OUT_DETAIL = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_focus_compare_detail.csv"
OUT_PNG = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_focus_compare.png"
OUT_META = OUTDIR / "runtime_direct_detlin_joint_tailgnorm_bandpass_focus_compare_run_meta.json"

FOCUS_D = [4.0, 4.8, 6.4, 7.2, 8.0]
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
BANDPASS = {"beta": 0.25, "center": 0.06, "half_width": 0.04}


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
    return make_baseline_kinetics(
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


def _set_width_fix(kin) -> None:
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


def _patch_bandpass(kin) -> None:
    beta = float(BANDPASS["beta"])
    center = float(BANDPASS["center"])
    half_width = float(BANDPASS["half_width"])

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


def _evaluate(
    label: str,
    *,
    bandpass: bool,
    full_map: dict[tuple[float, float], dict[str, float]],
    eta_grid: list[float],
    mu_obs: float,
    sigma_obs: float,
) -> pd.DataFrame:
    kin = _build_kinetics()
    _set_width_fix(kin)
    if bandpass:
        _patch_bandpass(kin)

    rows = []
    for D in FOCUS_D:
        for eta in eta_grid:
            ref = full_map[(float(D), float(eta))]
            mu = float(
                kin.hll_mu_pred(
                    2,
                    D=float(D),
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
                    "label": label,
                    "D": float(D),
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


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    full_map = _load_map(FULL_MAP)
    eta_grid = sorted({float(key[1]) for key in full_map})
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    width_df = _evaluate("width_only", bandpass=False, full_map=full_map, eta_grid=eta_grid, mu_obs=mu_obs, sigma_obs=sigma_obs)
    band_df = _evaluate("bandpass", bandpass=True, full_map=full_map, eta_grid=eta_grid, mu_obs=mu_obs, sigma_obs=sigma_obs)
    detail_df = pd.concat([width_df, band_df], ignore_index=True)

    rows = []
    for D in FOCUS_D:
        w = width_df[width_df["D"] == float(D)]
        b = band_df[band_df["D"] == float(D)]
        rows.append(
            {
                "D": float(D),
                "width_only_p95_abs_delta_mu_mumu": float(np.percentile(w["abs_delta_mu_mumu"], 95.0)),
                "width_only_max_abs_delta_mu_mumu": float(w["abs_delta_mu_mumu"].max()),
                "width_only_acceptance_mismatch": float(w["acceptance_mismatch"].mean()),
                "bandpass_p95_abs_delta_mu_mumu": float(np.percentile(b["abs_delta_mu_mumu"], 95.0)),
                "bandpass_max_abs_delta_mu_mumu": float(b["abs_delta_mu_mumu"].max()),
                "bandpass_acceptance_mismatch": float(b["acceptance_mismatch"].mean()),
                "delta_p95_abs_delta_mu_mumu": float(np.percentile(b["abs_delta_mu_mumu"], 95.0) - np.percentile(w["abs_delta_mu_mumu"], 95.0)),
                "delta_max_abs_delta_mu_mumu": float(b["abs_delta_mu_mumu"].max() - w["abs_delta_mu_mumu"].max()),
            }
        )
    summary_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(summary_df["D"], summary_df["width_only_p95_abs_delta_mu_mumu"], marker="o", linewidth=2.0, label="width-only")
    ax.plot(summary_df["D"], summary_df["bandpass_p95_abs_delta_mu_mumu"], marker="s", linewidth=2.0, label="band-pass")
    ax.set_xlabel("D")
    ax.set_ylabel("p95 |Δmu_mumu| across eta")
    ax.set_title("strict all-direct width-only vs band-pass exact compare")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    summary_df.to_csv(OUT_SUMMARY, index=False)
    detail_df.to_csv(OUT_DETAIL, index=False)
    OUT_META.write_text(json.dumps({"focus_D": FOCUS_D, "eta_grid": eta_grid, "bandpass": BANDPASS, "width_fix": WIDTH_FIX}, indent=2))
    for path in [OUT_SUMMARY, OUT_DETAIL, OUT_PNG, OUT_META]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {OUT_SUMMARY}")
    print(f"[saved] {OUT_DETAIL}")
    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_META}")
    print(summary_df.to_dict(orient='records'))


if __name__ == "__main__":
    main()

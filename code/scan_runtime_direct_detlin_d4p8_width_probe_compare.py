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
OUT_SUMMARY = OUTDIR / "runtime_direct_detlin_d4p8_width_probe_compare_summary.csv"
OUT_DETAIL = OUTDIR / "runtime_direct_detlin_d4p8_width_probe_compare_detail.csv"
OUT_PNG = OUTDIR / "runtime_direct_detlin_d4p8_width_probe_compare.png"
OUT_META = OUTDIR / "runtime_direct_detlin_d4p8_width_probe_compare_run_meta.json"

FOCUS_D = [4.0, 4.8, 6.4, 7.2, 8.0]
ETA_TARGET = [0.2, 1.15, 2.10, 3.05, 4.0]
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
WIDTH_PROBES = [
    {"label": "gnorm_bandpass_only", "width_beta": 0.0, "width_center": 4.8, "width_half_width": 0.4},
    {"label": "width_probe_beta0p40_hw0p40", "width_beta": 0.40, "width_center": 4.8, "width_half_width": 0.40},
]


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


def _patch_width_bandpass(kin, *, beta: float, center: float, half_width: float) -> None:
    original = kin._blend_observable_width_ratio

    def patched(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        activation = float(max(0.0, 1.0 - abs(float(D) - center) / max(half_width, 1e-30)))
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        return float(np.exp(np.log(max(base, floor)) + float(beta) * activation * positive_log_width))

    kin._blend_observable_width_ratio = types.MethodType(patched, kin)


def _evaluate_candidate(full_map, eta_probe: list[float], mu_obs: float, sigma_obs: float, probe: dict[str, float]) -> pd.DataFrame:
    kin = _build_kinetics()
    _set_width_fix(kin)
    _patch_gnorm_bandpass(kin)
    if float(probe["width_beta"]) > 0.0:
        _patch_width_bandpass(
            kin,
            beta=float(probe["width_beta"]),
            center=float(probe["width_center"]),
            half_width=float(probe["width_half_width"]),
        )

    rows = []
    for D in FOCUS_D:
        for eta in eta_probe:
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
                    "label": str(probe["label"]),
                    "width_beta": float(probe["width_beta"]),
                    "width_center": float(probe["width_center"]),
                    "width_half_width": float(probe["width_half_width"]),
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
    eta_values = sorted({float(key[1]) for key in full_map})
    eta_probe = [min(eta_values, key=lambda x, target=target: abs(x - target)) for target in ETA_TARGET]
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    detail_frames = [_evaluate_candidate(full_map, eta_probe, mu_obs, sigma_obs, probe) for probe in WIDTH_PROBES]
    detail_df = pd.concat(detail_frames, ignore_index=True)

    summary_rows = []
    for label, sub_df in detail_df.groupby("label", sort=False):
        row = {"label": str(label)}
        meta = sub_df.iloc[0]
        row["width_beta"] = float(meta["width_beta"])
        row["width_center"] = float(meta["width_center"])
        row["width_half_width"] = float(meta["width_half_width"])
        objective = 0.0
        for D in FOCUS_D:
            dsub = sub_df[sub_df["D"] == float(D)]
            p95 = float(np.percentile(dsub["abs_delta_mu_mumu"], 95.0))
            maxv = float(dsub["abs_delta_mu_mumu"].max())
            mismatch = float(dsub["acceptance_mismatch"].mean())
            row[f"d{str(D).replace('.', 'p')}_p95_abs_delta_mu_mumu"] = p95
            row[f"d{str(D).replace('.', 'p')}_max_abs_delta_mu_mumu"] = maxv
            row[f"d{str(D).replace('.', 'p')}_acceptance_mismatch"] = mismatch
        objective = (
            1.20 * row["d4p8_max_abs_delta_mu_mumu"]
            + 0.20 * row["d6p4_max_abs_delta_mu_mumu"]
            + 0.20 * row["d7p2_max_abs_delta_mu_mumu"]
            + 0.20 * row["d8p0_max_abs_delta_mu_mumu"]
            + 8.0 * row["d4p0_acceptance_mismatch"]
            + 0.10 * row["d4p0_max_abs_delta_mu_mumu"]
        )
        row["objective"] = float(objective)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("objective").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, marker in zip(summary_df["label"], ["o", "s", "^", "D"]):
        sub = summary_df[summary_df["label"] == label].iloc[0]
        ax.plot(
            FOCUS_D,
            [
                float(sub["d4p0_p95_abs_delta_mu_mumu"]),
                float(sub["d4p8_p95_abs_delta_mu_mumu"]),
                float(sub["d6p4_p95_abs_delta_mu_mumu"]),
                float(sub["d7p2_p95_abs_delta_mu_mumu"]),
                float(sub["d8p0_p95_abs_delta_mu_mumu"]),
            ],
            marker=marker,
            linewidth=2.0,
            label=str(label),
        )
    ax.set_xlabel("D")
    ax.set_ylabel("p95 |Δmu_mumu| across eta probe")
    ax.set_title("strict all-direct D=4.8 width probe compare")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    summary_df.to_csv(OUT_SUMMARY, index=False)
    detail_df.to_csv(OUT_DETAIL, index=False)
    OUT_META.write_text(json.dumps({"focus_D": FOCUS_D, "eta_target": ETA_TARGET, "eta_probe": eta_probe, "width_probes": WIDTH_PROBES, "gnorm_bandpass": GNORM_BANDPASS, "width_fix": WIDTH_FIX}, indent=2))
    for path in [OUT_SUMMARY, OUT_DETAIL, OUT_PNG, OUT_META]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {OUT_SUMMARY}")
    print(summary_df.to_dict(orient='records'))


if __name__ == "__main__":
    main()

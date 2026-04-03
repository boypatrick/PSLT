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
OUT_PREFIX = "runtime_direct_detlin_tail_single_beta_full_eta"

FOCUS_D = float(os.environ.get("TAIL_D", "7.2"))
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
D4P0_WIDTH = {"beta": 1.30, "center": 4.0, "half_width": 0.40}
TAIL_WIDTH = {
    "beta": float(os.environ.get("TAIL_BETA", "0.20")),
    "center": FOCUS_D,
    "half_width": 0.40,
}


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


def _patch_width_bands(kin, *, include_tail: bool) -> None:
    original = kin._blend_observable_width_ratio

    def patched(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        exponent = np.log(max(base, floor))

        act40 = float(max(0.0, 1.0 - abs(float(D) - float(D4P0_WIDTH["center"])) / float(D4P0_WIDTH["half_width"])))
        exponent += float(D4P0_WIDTH["beta"]) * act40 * positive_log_width
        act48 = float(max(0.0, 1.0 - abs(float(D) - float(D4P8_WIDTH["center"])) / float(D4P8_WIDTH["half_width"])))
        exponent += float(D4P8_WIDTH["beta"]) * act48 * positive_log_width

        if include_tail:
            act_tail = float(max(0.0, 1.0 - abs(float(D) - float(TAIL_WIDTH["center"])) / float(TAIL_WIDTH["half_width"])))
            exponent += float(TAIL_WIDTH["beta"]) * act_tail * positive_log_width
        return float(np.exp(exponent))

    kin._blend_observable_width_ratio = types.MethodType(patched, kin)


def _evaluate(label: str, *, include_tail: bool, full_map, eta_grid, mu_obs: float, sigma_obs: float) -> pd.DataFrame:
    kin = _build_kinetics()
    _set_width_fix(kin)
    _patch_gnorm_bandpass(kin)
    _patch_width_bands(kin, include_tail=include_tail)

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
                "label": label,
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
    dtag = str(FOCUS_D).replace(".", "p")
    btag = str(TAIL_WIDTH["beta"]).replace(".", "p")
    out_summary = OUTDIR / f"{OUT_PREFIX}_D{dtag}_summary_beta{btag}.csv"
    out_detail = OUTDIR / f"{OUT_PREFIX}_D{dtag}_detail_beta{btag}.csv"
    out_png = OUTDIR / f"{OUT_PREFIX}_D{dtag}_beta{btag}.png"
    out_meta = OUTDIR / f"{OUT_PREFIX}_D{dtag}_run_meta_beta{btag}.json"

    full_map = _load_map(FULL_MAP)
    eta_grid = sorted({float(key[1]) for key in full_map})
    obs = load_observations()["mumu"]
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    base_df = _evaluate("baseline_local_closures", include_tail=False, full_map=full_map, eta_grid=eta_grid, mu_obs=mu_obs, sigma_obs=sigma_obs)
    cand_df = _evaluate(f"candidate_tail_beta{btag}", include_tail=True, full_map=full_map, eta_grid=eta_grid, mu_obs=mu_obs, sigma_obs=sigma_obs)
    detail_df = pd.concat([base_df, cand_df], ignore_index=True)

    summary_df = pd.DataFrame(
        [
            {
                "focus_D": FOCUS_D,
                "tail_beta": float(TAIL_WIDTH["beta"]),
                "baseline_p95_abs_delta_mu_mumu": float(np.percentile(base_df["abs_delta_mu_mumu"], 95.0)),
                "baseline_max_abs_delta_mu_mumu": float(base_df["abs_delta_mu_mumu"].max()),
                "baseline_acceptance_mismatch": float(base_df["acceptance_mismatch"].mean()),
                "candidate_p95_abs_delta_mu_mumu": float(np.percentile(cand_df["abs_delta_mu_mumu"], 95.0)),
                "candidate_max_abs_delta_mu_mumu": float(cand_df["abs_delta_mu_mumu"].max()),
                "candidate_acceptance_mismatch": float(cand_df["acceptance_mismatch"].mean()),
                "candidate_mu_min": float(cand_df["mu_candidate"].min()),
                "candidate_mu_max": float(cand_df["mu_candidate"].max()),
                "candidate_chi2_min": float(cand_df["chi2_candidate"].min()),
                "candidate_chi2_max": float(cand_df["chi2_candidate"].max()),
            }
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(base_df["eta"], base_df["mu_candidate"], marker="o", linewidth=1.8, label="baseline")
    axes[0].plot(cand_df["eta"], cand_df["mu_candidate"], marker="s", linewidth=1.8, label="candidate")
    axes[0].plot(base_df["eta"], base_df["mu_full"], linestyle="--", color="black", linewidth=1.2, label="full_direct")
    axes[0].set_xlabel("eta")
    axes[0].set_ylabel("mu_mumu")
    axes[0].set_title(f"D={FOCUS_D:.1f} tail compare (beta={TAIL_WIDTH['beta']:.2f})")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(base_df["eta"], base_df["chi2_candidate"], marker="o", linewidth=1.8, label="baseline")
    axes[1].plot(cand_df["eta"], cand_df["chi2_candidate"], marker="s", linewidth=1.8, label="candidate")
    axes[1].plot(base_df["eta"], base_df["chi2_full"], linestyle="--", color="black", linewidth=1.2, label="full_direct")
    axes[1].set_xlabel("eta")
    axes[1].set_ylabel("chi2_mumu")
    axes[1].set_title("Residual-side compare")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    summary_df.to_csv(out_summary, index=False)
    detail_df.to_csv(out_detail, index=False)
    out_meta.write_text(
        json.dumps(
            {
                "focus_D": FOCUS_D,
                "eta_grid": eta_grid,
                "gnorm_bandpass": GNORM_BANDPASS,
                "d4p0_width": D4P0_WIDTH,
                "d4p8_width": D4P8_WIDTH,
                "tail_width": TAIL_WIDTH,
                "width_fix": WIDTH_FIX,
            },
            indent=2,
        )
    )
    for path in [out_summary, out_detail, out_png, out_meta]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {out_summary}")
    print(summary_df.to_dict(orient="records"))


if __name__ == "__main__":
    main()

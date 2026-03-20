#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eft_wilson_matching import total_width_ratio
from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
SIGDIR = ROOT / "output" / "hll_signal_strength"

FOCUS_D = [4.0, 4.8, 6.4, 7.2]
WIDTH_FOCUS_D = [4.0, 4.8, 7.2]
ETA_GRID = np.linspace(0.2, 4.0, 41)
REF_D = 9.6
REF_ETA = 1.0
OBS_MU = 1.4
OBS_SIGMA = 0.4
T_COH = float(PAPER_BASELINE["t_coh"])

FULL_MAP = SIGDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv"


def _grid_key(d: float, eta: float) -> tuple[float, float]:
    return (round(float(d), 10), round(float(eta), 10))


def _point_diags(kin, D: float, eta: float) -> dict[str, object]:
    c_low, _ = kin.hll_wilson_matrix_uv_rge_with_meta(
        D=float(D),
        eta=float(eta),
        t_coh=T_COH,
        N_max=kin.params.hll_observable_nmax,
    )
    c_diag = np.maximum(np.diag(np.asarray(c_low, dtype=float)), kin.params.b_overlap_floor)
    c22 = float(c_diag[1])
    return {"c_diag": c_diag, "c22": c22}


def _chi2(mu: float) -> float:
    return float(((float(mu) - OBS_MU) / OBS_SIGMA) ** 2)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    if not FULL_MAP.exists():
        raise FileNotFoundError(f"Missing full-direct map: {FULL_MAP}")

    common = dict(
        observable_mode="eft_wilson_uv_rge",
        d_min=4.0,
        d_max=20.0,
        d_num=21,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )
    kin_detlin = make_baseline_kinetics(chain_mode="cell_direct_runtime_release_tailm2", **common)
    base_cfg = kin_detlin._hll_match_config()

    full_map = pd.read_csv(FULL_MAP)
    full_lookup = {
        _grid_key(r.D, r.eta): (float(r.mu_mumu), float(r.chi2_mumu))
        for r in full_map.itertuples(index=False)
        if float(r.D) in FOCUS_D
    }

    ref_diag = _point_diags(kin_detlin, REF_D, REF_ETA)["c_diag"]
    ref_c22 = float(ref_diag[1])

    point_rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            diag = _point_diags(kin_detlin, float(D), float(eta))
            mu_full, chi2_full = full_lookup[_grid_key(D, eta)]
            point_rows.append(
                {
                    "D": float(D),
                    "eta": float(eta),
                    "c11": float(diag["c_diag"][0]),
                    "c22": float(diag["c_diag"][1]),
                    "c33": float(diag["c_diag"][2]),
                    "mu_full": float(mu_full),
                    "chi2_full": float(chi2_full),
                }
            )
    points_df = pd.DataFrame(point_rows).sort_values(["D", "eta"]).reset_index(drop=True)

    base_values = [0.65, 0.70, 0.75, 0.80]
    turnoff_pairs = [(40.0, 80.0), (50.0, 100.0), (60.0, 120.0), (70.0, 140.0)]
    tail_reboost_values = [0.0, 0.05, 0.10, 0.15]

    rows: list[dict[str, float]] = []
    for width_power_base in base_values:
        for turnoff_lo, turnoff_hi in turnoff_pairs:
            for tail_reboost_max in tail_reboost_values:
                cfg = replace(
                    base_cfg,
                    width_power_base=float(width_power_base),
                    width_power_turnoff_lo=float(turnoff_lo),
                    width_power_turnoff_hi=float(turnoff_hi),
                    width_power_tail_reboost_max=float(tail_reboost_max),
                )

                work = points_df.copy()
                c_diag = work[["c11", "c22", "c33"]].to_numpy(dtype=float)
                cand_width = [
                    total_width_ratio(c_diag=row, c_diag_ref=ref_diag, cfg=cfg)
                    for row in c_diag
                ]
                work["width_candidate"] = np.asarray(cand_width, dtype=float)
                work["mu_candidate"] = (work["c22"] / max(ref_c22, 1e-30)) ** 2 / np.maximum(work["width_candidate"], 1e-30)
                work["chi2_candidate"] = [_chi2(v) for v in work["mu_candidate"]]
                work["abs_delta_mu"] = np.abs(work["mu_candidate"] - work["mu_full"])
                work["acceptance_mismatch"] = (
                    (work["chi2_full"] <= 4.0) ^ (work["chi2_candidate"] <= 4.0)
                ).astype(float)

                slice_stats = {}
                for d_val, sub in work.groupby("D", sort=True):
                    slice_stats[float(d_val)] = {
                        "max": float(sub["abs_delta_mu"].max()),
                        "p95": float(np.percentile(sub["abs_delta_mu"], 95.0)),
                        "mismatch": float(sub["acceptance_mismatch"].mean()),
                    }

                width_focus = work[work["D"].isin(WIDTH_FOCUS_D)]
                objective = max(
                    slice_stats[4.8]["max"],
                    slice_stats[7.2]["max"],
                    5.0 * slice_stats[4.0]["mismatch"],
                )

                rows.append(
                    {
                        "width_power_base": float(width_power_base),
                        "turnoff_lo": float(turnoff_lo),
                        "turnoff_hi": float(turnoff_hi),
                        "tail_reboost_max": float(tail_reboost_max),
                        "objective_width_focus": float(objective),
                        "width_focus_p95_abs_delta_mu": float(np.percentile(width_focus["abs_delta_mu"], 95.0)),
                        "width_focus_acceptance_mismatch": float(width_focus["acceptance_mismatch"].mean()),
                        "d4p0_max_abs_delta_mu": float(slice_stats[4.0]["max"]),
                        "d4p0_acceptance_mismatch": float(slice_stats[4.0]["mismatch"]),
                        "d4p8_max_abs_delta_mu": float(slice_stats[4.8]["max"]),
                        "d4p8_p95_abs_delta_mu": float(slice_stats[4.8]["p95"]),
                        "d6p4_max_abs_delta_mu": float(slice_stats[6.4]["max"]),
                        "d7p2_max_abs_delta_mu": float(slice_stats[7.2]["max"]),
                        "d7p2_p95_abs_delta_mu": float(slice_stats[7.2]["p95"]),
                    }
                )

    scan_df = pd.DataFrame(rows).sort_values("objective_width_focus").reset_index(drop=True)
    best = scan_df.iloc[0]
    current = scan_df[
        (scan_df["width_power_base"] == 0.75)
        & (scan_df["turnoff_lo"] == 70.0)
        & (scan_df["turnoff_hi"] == 140.0)
        & (scan_df["tail_reboost_max"] == 0.15)
    ].iloc[0]

    summary_df = pd.DataFrame(
        [
            {
                "best_width_power_base": float(best["width_power_base"]),
                "best_turnoff_lo": float(best["turnoff_lo"]),
                "best_turnoff_hi": float(best["turnoff_hi"]),
                "best_tail_reboost_max": float(best["tail_reboost_max"]),
                "best_objective_width_focus": float(best["objective_width_focus"]),
                "best_d4p0_mismatch": float(best["d4p0_acceptance_mismatch"]),
                "best_d4p8_max_abs_delta_mu": float(best["d4p8_max_abs_delta_mu"]),
                "best_d6p4_max_abs_delta_mu": float(best["d6p4_max_abs_delta_mu"]),
                "best_d7p2_max_abs_delta_mu": float(best["d7p2_max_abs_delta_mu"]),
                "current_objective_width_focus": float(current["objective_width_focus"]),
                "current_d4p0_mismatch": float(current["d4p0_acceptance_mismatch"]),
                "current_d4p8_max_abs_delta_mu": float(current["d4p8_max_abs_delta_mu"]),
                "current_d6p4_max_abs_delta_mu": float(current["d6p4_max_abs_delta_mu"]),
                "current_d7p2_max_abs_delta_mu": float(current["d7p2_max_abs_delta_mu"]),
            }
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for reboost in tail_reboost_values:
        sub = scan_df[scan_df["tail_reboost_max"] == reboost].sort_values("turnoff_lo")
        label = f"reboost={reboost:.2f}"
        axes[0].plot(sub["turnoff_lo"], sub["d4p8_max_abs_delta_mu"], marker="o", label=label)
        axes[1].plot(sub["turnoff_lo"], sub["d7p2_max_abs_delta_mu"], marker="o", label=label)
        axes[2].plot(sub["turnoff_lo"], sub["d4p0_acceptance_mismatch"], marker="o", label=label)
    axes[0].set_title("D=4.8 max |Δμ_μμ|")
    axes[1].set_title("D=7.2 max |Δμ_μμ|")
    axes[2].set_title("D=4.0 mismatch")
    for ax in axes:
        ax.set_xlabel("turnoff_lo")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("detlin width-only retune scan")

    out_scan = OUTDIR / "runtime_direct_detlin_width_retune_scan.csv"
    out_summary = OUTDIR / "runtime_direct_detlin_width_retune_summary.csv"
    out_fig = OUTDIR / "runtime_direct_detlin_width_retune.png"
    out_meta = OUTDIR / "runtime_direct_detlin_width_retune_run_meta.json"

    scan_df.to_csv(out_scan, index=False)
    summary_df.to_csv(out_summary, index=False)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    out_meta.write_text(
        json.dumps(
            {
                "focus_D": FOCUS_D,
                "width_focus_D": WIDTH_FOCUS_D,
                "base_values": base_values,
                "turnoff_pairs": turnoff_pairs,
                "tail_reboost_values": tail_reboost_values,
            },
            indent=2,
        )
    )

    for path in [out_scan, out_summary, out_fig, out_meta]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {out_scan}")
    print(f"[saved] {out_summary}")
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_meta}")
    print(summary_df.to_dict(orient="records")[0])


if __name__ == "__main__":
    main()

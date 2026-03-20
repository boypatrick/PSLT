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

from eft_wilson_matching import EFTWilsonMatchConfig
from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
SIGDIR = ROOT / "output" / "hll_signal_strength"

FOCUS_D = [4.0, 4.8, 6.4, 7.2]
ETA_GRID = np.linspace(0.2, 4.0, 41)
REF_D = 9.6
REF_ETA = 1.0
OBS_MU = 1.4
OBS_SIGMA = 0.4
T_COH = float(PAPER_BASELINE["t_coh"])

FULL_MAP = SIGDIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41.csv"


def _grid_key(d: float, eta: float) -> tuple[float, float]:
    return (round(float(d), 10), round(float(eta), 10))


def _chi2(mu: float) -> float:
    return float(((float(mu) - OBS_MU) / OBS_SIGMA) ** 2)


def _point_diags(kin, D: float, eta: float) -> dict[str, object]:
    c_low, _ = kin.hll_wilson_matrix_uv_rge_with_meta(
        D=float(D),
        eta=float(eta),
        t_coh=T_COH,
        N_max=kin.params.hll_observable_nmax,
    )
    c_diag = np.maximum(np.diag(np.asarray(c_low, dtype=float)), kin.params.b_overlap_floor)
    return {"c_diag": c_diag, "c22": float(c_diag[1])}


def _extended_width_ratio(
    c_diag: np.ndarray,
    c_diag_ref: np.ndarray,
    cfg: EFTWilsonMatchConfig,
    *,
    trigger_gamma: float,
    suppression_gamma: float,
) -> float:
    c = np.maximum(np.asarray(c_diag, dtype=float).reshape(3), cfg.floor)
    c_ref = np.maximum(np.asarray(c_diag_ref, dtype=float).reshape(3), cfg.floor)
    partial = (c / c_ref) ** 2
    br = np.array([cfg.br_ee, cfg.br_mumu, cfg.br_tautau], dtype=float)

    raw_delta = float(np.sum(br * (partial - 1.0)))
    raw_ratio = 1.0 + float(cfg.width_scale) * raw_delta
    log_partial_ratio = float(np.log(max(float(partial[1]), cfg.floor) / max(float(partial[2]), cfg.floor)))

    rho_target = float(cfg.width_power_base) + float(cfg.width_power_log_ratio_coeff) * log_partial_ratio
    rho_target = float(np.clip(rho_target, float(cfg.width_power_min), float(cfg.width_power_max)))

    lo = max(float(cfg.width_power_trigger_lo), 1.0 + cfg.floor)
    hi = max(float(cfg.width_power_trigger_hi), lo + cfg.floor)
    if hi <= lo * (1.0 + 1e-12):
        activation = 1.0 if raw_ratio > lo else 0.0
    else:
        activation = float(
            np.clip(
                np.log(max(raw_ratio, 1.0 + cfg.floor) / lo) / np.log(hi / lo),
                0.0,
                1.0,
            )
        )
    activation = float(activation ** trigger_gamma)
    activation_raw = activation

    turnoff_lo = max(float(cfg.width_power_turnoff_lo), hi)
    turnoff_hi = max(float(cfg.width_power_turnoff_hi), turnoff_lo + cfg.floor)
    if raw_ratio <= turnoff_lo:
        suppression = 1.0
    elif turnoff_hi <= turnoff_lo * (1.0 + 1e-12):
        suppression = 0.0 if raw_ratio > turnoff_lo else 1.0
    else:
        x = float(
            np.clip(
                np.log(max(raw_ratio, turnoff_lo) / turnoff_lo) / np.log(turnoff_hi / turnoff_lo),
                0.0,
                1.0,
            )
        )
        suppression = float((1.0 - x) ** suppression_gamma)
    activation *= suppression

    tail_lo = max(float(cfg.width_power_tail_logratio_lo), 0.0)
    tail_hi = max(float(cfg.width_power_tail_logratio_hi), tail_lo + cfg.floor)
    log_ratio_mag = abs(log_partial_ratio)
    if log_ratio_mag <= tail_lo:
        tail_reboost = 0.0
    elif tail_hi <= tail_lo + cfg.floor:
        tail_reboost = float(cfg.width_power_tail_reboost_max)
    else:
        tail_reboost = float(cfg.width_power_tail_reboost_max) * float(
            np.clip((log_ratio_mag - tail_lo) / (tail_hi - tail_lo), 0.0, 1.0)
        )
    activation += activation_raw * (1.0 - suppression) * tail_reboost

    rho = 1.0 - activation * (1.0 - rho_target)
    partial = np.maximum(partial, cfg.floor) ** rho
    delta = float(np.sum(br * (partial - 1.0)))
    ratio = 1.0 + float(cfg.width_scale) * delta
    return float(max(ratio, cfg.floor))


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

    base_values = [0.80]
    trigger_pairs = [(2.5, 5.0), (3.0, 6.0)]
    turnoff_pairs = [(10.0, 20.0), (15.0, 30.0), (20.0, 40.0), (25.0, 50.0), (30.0, 60.0)]
    tail_reboost_values = [0.15, 0.25, 0.35]
    trigger_gamma_values = [1.0, 1.5]
    suppression_gamma_values = [1.0, 2.0, 3.0, 4.0]

    rows: list[dict[str, float]] = []
    for width_power_base in base_values:
        for trigger_lo, trigger_hi in trigger_pairs:
            for turnoff_lo, turnoff_hi in turnoff_pairs:
                if turnoff_lo < trigger_hi:
                    continue
                for tail_reboost_max in tail_reboost_values:
                    for trigger_gamma in trigger_gamma_values:
                        for suppression_gamma in suppression_gamma_values:
                            cfg = replace(
                                base_cfg,
                                width_power_base=float(width_power_base),
                                width_power_trigger_lo=float(trigger_lo),
                                width_power_trigger_hi=float(trigger_hi),
                                width_power_turnoff_lo=float(turnoff_lo),
                                width_power_turnoff_hi=float(turnoff_hi),
                                width_power_tail_reboost_max=float(tail_reboost_max),
                            )

                            work = points_df.copy()
                            c_diag = work[["c11", "c22", "c33"]].to_numpy(dtype=float)
                            cand_width = [
                                _extended_width_ratio(
                                    c_diag=row,
                                    c_diag_ref=ref_diag,
                                    cfg=cfg,
                                    trigger_gamma=float(trigger_gamma),
                                    suppression_gamma=float(suppression_gamma),
                                )
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
                                    "mismatch": float(sub["acceptance_mismatch"].mean()),
                                }

                            objective = max(
                                slice_stats[4.8]["max"],
                                slice_stats[7.2]["max"],
                                5.0 * slice_stats[4.0]["mismatch"],
                            )

                            rows.append(
                                {
                                    "width_power_base": float(width_power_base),
                                    "trigger_lo": float(trigger_lo),
                                    "trigger_hi": float(trigger_hi),
                                    "turnoff_lo": float(turnoff_lo),
                                    "turnoff_hi": float(turnoff_hi),
                                    "tail_reboost_max": float(tail_reboost_max),
                                    "trigger_gamma": float(trigger_gamma),
                                    "suppression_gamma": float(suppression_gamma),
                                    "objective_width_focus": float(objective),
                                    "d4p0_mismatch": float(slice_stats[4.0]["mismatch"]),
                                    "d4p8_max_abs_delta_mu": float(slice_stats[4.8]["max"]),
                                    "d6p4_max_abs_delta_mu": float(slice_stats[6.4]["max"]),
                                    "d7p2_max_abs_delta_mu": float(slice_stats[7.2]["max"]),
                                }
                            )

    scan_df = pd.DataFrame(rows).sort_values("objective_width_focus").reset_index(drop=True)
    summary_df = pd.DataFrame(
        [
            {
                "best_width_power_base": float(scan_df.iloc[0]["width_power_base"]),
                "best_trigger_lo": float(scan_df.iloc[0]["trigger_lo"]),
                "best_trigger_hi": float(scan_df.iloc[0]["trigger_hi"]),
                "best_turnoff_lo": float(scan_df.iloc[0]["turnoff_lo"]),
                "best_turnoff_hi": float(scan_df.iloc[0]["turnoff_hi"]),
                "best_tail_reboost_max": float(scan_df.iloc[0]["tail_reboost_max"]),
                "best_trigger_gamma": float(scan_df.iloc[0]["trigger_gamma"]),
                "best_suppression_gamma": float(scan_df.iloc[0]["suppression_gamma"]),
                "best_objective_width_focus": float(scan_df.iloc[0]["objective_width_focus"]),
                "best_d4p0_mismatch": float(scan_df.iloc[0]["d4p0_mismatch"]),
                "best_d4p8_max_abs_delta_mu": float(scan_df.iloc[0]["d4p8_max_abs_delta_mu"]),
                "best_d6p4_max_abs_delta_mu": float(scan_df.iloc[0]["d6p4_max_abs_delta_mu"]),
                "best_d7p2_max_abs_delta_mu": float(scan_df.iloc[0]["d7p2_max_abs_delta_mu"]),
            }
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    top = scan_df.head(40)
    axes[0].scatter(top["d4p8_max_abs_delta_mu"], top["d7p2_max_abs_delta_mu"], c=top["suppression_gamma"], cmap="viridis")
    axes[0].set_xlabel("D=4.8 max |Δμ_μμ|")
    axes[0].set_ylabel("D=7.2 max |Δμ_μμ|")
    axes[0].set_title("4.8 vs 7.2 tradeoff")
    axes[1].scatter(top["turnoff_lo"], top["d4p8_max_abs_delta_mu"], c=top["suppression_gamma"], cmap="viridis")
    axes[1].set_title("D=4.8 vs turnoff_lo")
    axes[2].scatter(top["trigger_lo"], top["d7p2_max_abs_delta_mu"], c=top["trigger_gamma"], cmap="plasma")
    axes[2].set_title("D=7.2 vs trigger_lo")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("detlin width family extension scan")

    out_scan = OUTDIR / "runtime_direct_detlin_width_family_extension_scan.csv"
    out_summary = OUTDIR / "runtime_direct_detlin_width_family_extension_summary.csv"
    out_fig = OUTDIR / "runtime_direct_detlin_width_family_extension.png"
    out_meta = OUTDIR / "runtime_direct_detlin_width_family_extension_run_meta.json"

    scan_df.to_csv(out_scan, index=False)
    summary_df.to_csv(out_summary, index=False)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    out_meta.write_text(
        json.dumps(
            {
                "focus_D": FOCUS_D,
                "trigger_pairs": trigger_pairs,
                "turnoff_pairs": turnoff_pairs,
                "tail_reboost_values": tail_reboost_values,
                "trigger_gamma_values": trigger_gamma_values,
                "suppression_gamma_values": suppression_gamma_values,
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

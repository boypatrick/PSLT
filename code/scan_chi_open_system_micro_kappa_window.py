#!/usr/bin/env python3
"""
Calibrate a physical window for the open_system_micro bath normalization.

This script treats kappa_env as a uniform bath-normalization scale that
rescales both gamma_phi and gamma_mix while leaving the localized two-level
system Hamiltonian fixed. It reports:

  1) calibration-consistent windows against the geometry-target ratio profile;
  2) map-level stability windows against the baseline fp_2d_full scan.

Outputs:
  - output/chi_open_system/chi_open_system_micro_kappa_window_scan.csv
  - output/chi_open_system/chi_open_system_micro_kappa_window_summary.csv
  - output/chi_open_system/chi_open_system_micro_kappa_window_ratio_scan.csv
  - output/chi_open_system/chi_open_system_micro_kappa_window.png
  - output/chi_open_system/chi_open_system_micro_kappa_window_run_meta.json
  - mirrored copies under paper/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile
from hll_observable import HLLChannelPredictor, HLLObservableConfig
from pslt_lib import PSLTKinetics, PSLTParameters


OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"
BASELINE_GN_CSV = ROOT / "output" / "gn_fp_impact" / "gn_profile_impact.csv"
CALIB_CSV = OUTDIR / "kappa_env_anchor_calibration.csv"
HOLDOUT_CSV = OUTDIR / "kappa_env_anchor_holdout.csv"


@dataclass(frozen=True)
class DRow:
    D: float
    delta: float
    gamma_phi_src: float
    gamma_mix_src: float
    gamma_ref: float
    chi_lr: float
    ratio_target: float


def _pick_first_present(row: pd.Series, keys: Sequence[str]) -> float:
    for key in keys:
        if key in row.index:
            value = float(row[key])
            if np.isfinite(value):
                return value
    raise KeyError(f"Missing keys: {keys}")


def _build_dataset(micro_csv: Path, geom_csv: Path) -> pd.DataFrame:
    df_micro = pd.read_csv(micro_csv).copy()
    df_geom = pd.read_csv(geom_csv).copy()
    if "ratio_proxy_to_baseline" not in df_geom.columns:
        raise KeyError("Geometry CSV must contain 'ratio_proxy_to_baseline'.")
    dfg = df_geom[["D", "ratio_proxy_to_baseline"]].rename(columns={"ratio_proxy_to_baseline": "ratio_target"})
    rows: List[Dict[str, float]] = []
    for _, row in df_micro.iterrows():
        try:
            rows.append(
                {
                    "D": float(row["D"]),
                    "delta": _pick_first_present(row, ("delta_micro", "delta")),
                    "gamma_phi_src": _pick_first_present(row, ("gamma_phi_micro", "gamma_phi")),
                    "gamma_mix_src": _pick_first_present(row, ("gamma_mix_micro", "gamma_mix")),
                    "gamma_ref": _pick_first_present(row, ("gamma_ref_micro", "gamma_ref", "Gamma_ref")),
                    "chi_lr": _pick_first_present(row, ("chi_LR_baseline",)),
                }
            )
        except KeyError:
            continue
    dfm = pd.DataFrame(rows)
    df = pd.merge(dfm, dfg, on="D", how="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("D").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No overlapping valid D rows after merging micro and geometry CSVs.")
    return df


def _lindblad_cmax(
    delta: float,
    gamma_phi: float,
    gamma_mix: float,
    tmax: float = 200.0,
    nstep: int = 300,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> float:
    h = np.array([[0.0, delta / 2.0], [delta / 2.0, 0.0]], dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    l_ops = [np.sqrt(max(gamma_phi, 0.0)) * sigma_z, np.sqrt(max(gamma_mix, 0.0)) * sigma_x]

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        rho = y.reshape(2, 2)
        drho = -1j * (h @ rho - rho @ h)
        for l_op in l_ops:
            drho += l_op @ rho @ l_op.conj().T - 0.5 * (l_op.conj().T @ l_op @ rho + rho @ l_op.conj().T @ l_op)
        return drho.reshape(-1)

    y0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex).reshape(-1)
    t_eval = np.linspace(0.0, tmax, nstep)
    sol = solve_ivp(rhs, (0.0, tmax), y0, t_eval=t_eval, rtol=rtol, atol=atol)
    cmax = 0.0
    for i in range(sol.y.shape[1]):
        rho = sol.y[:, i].reshape(2, 2)
        cmax = max(cmax, float(abs(rho[0, 1])))
    return cmax


def _ratio_scaled(row: DRow, scale: float) -> float:
    gamma_phi = float(row.gamma_phi_src * scale)
    gamma_mix = float(row.gamma_mix_src * scale)
    cmax = _lindblad_cmax(delta=row.delta, gamma_phi=gamma_phi, gamma_mix=gamma_mix)
    chi_eff = float(max(2.0 * gamma_mix * cmax / max(row.gamma_ref, 1e-30), 0.0))
    return float(chi_eff / max(row.chi_lr, 1e-30))


def _make_kin(
    scale: float,
    d_loc: np.ndarray,
    chi_loc: np.ndarray,
    superrad_profile: Dict[str, object],
    open_csv: Path,
) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        g_mode="fp_2d_full",
        g_fp_full_window_blend=0.8,
        g_fp_full_tail_beta=1.1,
        g_fp_full_tail_shell_power=0.0,
        g_fp_full_tail_clip_min=1e-3,
        g_fp_full_tail_clip_max=0.95,
        chi=0.2,
        A1=1.0,
        A2=1.0,
        gamma_mode=str(superrad_profile["mode"]),
        gamma_superrad_csv=str(superrad_profile["path"]),
        b_mode="eft_operator_norm",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode="eft_wilson_uv_rge",
        hll_observable_nmax=20,
        chi_mode="open_system_micro",
        chi_open_csv=str(open_csv),
        chi_open_phi_scale=scale,
        chi_open_mix_scale=scale,
        chi_open_tmax=200.0,
        chi_open_nstep=300,
        chi_lr_D=tuple(float(x) for x in d_loc),
        chi_lr_vals=tuple(float(x) for x in chi_loc),
    )
    return PSLTKinetics(params)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    calib = pd.read_csv(CALIB_CSV).iloc[0]
    holdout = pd.read_csv(HOLDOUT_CSV).iloc[0]
    micro_csv = Path(str(calib["micro_source_csv"]))
    geom_csv = Path(str(calib["target_source_csv"]))
    df = _build_dataset(micro_csv, geom_csv)
    rows = [DRow(**{k: float(v) for k, v in rec.items()}) for rec in df.to_dict(orient="records")]

    anchor_vals = [float(x) for x in str(calib["D_anchor_list"]).split(",") if str(x).strip()]
    holdout_vals = [float(x) for x in str(calib["D_holdout_list"]).split(",") if str(x).strip()]
    anchor_set = {round(v, 8) for v in anchor_vals}
    holdout_set = {round(v, 8) for v in holdout_vals}

    d_scan = scan_d_values(4.0, 20.0, 60)
    chi_profile = select_chi_profile(ROOT, d_scan)
    superrad_profile = select_superrad_profile(ROOT, d_scan)
    d_loc = np.asarray(chi_profile["d"], dtype=float)
    chi_loc = np.asarray(chi_profile["chi"], dtype=float)

    base_df = pd.read_csv(BASELINE_GN_CSV)
    base = base_df[base_df["case"] == "baseline_fp_2d_full"].iloc[0]
    f_r3_base = float(base["f_R3_gt_0p90"])
    f_acc_base = float(base["f_hmumu_chi2_le_4"])
    f_win_base = float(base["f_winner_gt_3"])

    scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    D_vals = d_scan
    eta_vals = np.linspace(0.2, 4.0, 60)
    hll_cfg = HLLObservableConfig(mode="eft_wilson_uv_rge", t_coh=1.0, ref_D=10.0, ref_eta=1.0, n_max=20)

    scan_rows: List[Dict[str, float | int]] = []
    ratio_rows: List[Dict[str, float]] = []

    for scale in scales:
        kin = _make_kin(scale, d_loc, chi_loc, superrad_profile, micro_csv)
        hll_mumu = HLLChannelPredictor(kin, layer_n=2, cfg=hll_cfg)

        r3_list: List[float] = []
        chi2_list: List[float] = []
        winner_list: List[int] = []
        for eta in eta_vals:
            for D in D_vals:
                _, _, meta = kin.get_probabilities(float(D), float(eta), 1.0, N_max=20)
                r3_list.append(float(meta["generation_ratio"]))
                winner_list.append(int(meta["winner"]))
                mu_pred = hll_mumu.mu_pred(float(D), float(eta))
                chi2_list.append(float(((mu_pred - 1.4) / 0.4) ** 2))

        r3 = np.asarray(r3_list, dtype=float)
        chi2 = np.asarray(chi2_list, dtype=float)
        winners = np.asarray(winner_list, dtype=int)

        pred: List[float] = []
        target: List[float] = []
        abs_err: List[float] = []
        holdout_abs: List[float] = []
        anchor_abs: List[float] = []
        ratio_arr: List[float] = []
        for row in rows:
            ratio_pred = _ratio_scaled(row, scale)
            ratio_arr.append(ratio_pred)
            ratio_rows.append(
                {
                    "kappa_scale": scale,
                    "D": row.D,
                    "ratio_pred": ratio_pred,
                    "ratio_target": row.ratio_target,
                    "abs_err": abs(ratio_pred - row.ratio_target),
                }
            )
            pred.append(ratio_pred)
            target.append(row.ratio_target)
            ae = abs(ratio_pred - row.ratio_target)
            abs_err.append(ae)
            if round(row.D, 8) in holdout_set:
                holdout_abs.append(ae)
            if round(row.D, 8) in anchor_set:
                anchor_abs.append(ae)

        pred_arr = np.asarray(pred, dtype=float)
        target_arr = np.asarray(target, dtype=float)
        holdout_arr = np.asarray(holdout_abs, dtype=float) if holdout_abs else np.asarray([], dtype=float)
        anchor_arr = np.asarray(anchor_abs, dtype=float) if anchor_abs else np.asarray([], dtype=float)
        ratio_np = np.asarray(ratio_arr, dtype=float)

        holdout_rmse = float(np.sqrt(np.mean(holdout_arr**2))) if holdout_arr.size else float("nan")
        holdout_max = float(np.max(holdout_arr)) if holdout_arr.size else float("nan")
        anchor_rmse = float(np.sqrt(np.mean(anchor_arr**2))) if anchor_arr.size else float("nan")
        overall_rmse = float(np.sqrt(np.mean((pred_arr - target_arr) ** 2)))

        delta_r3 = float(np.mean(r3 >= 0.90) - f_r3_base)
        delta_acc = float(np.mean(chi2 <= 4.0) - f_acc_base)
        delta_win = float(np.mean(winners > 3) - f_win_base)
        ratio_dynamic_range = float(np.max(ratio_np) - np.min(ratio_np))

        gate_candidate = int(
            abs(delta_r3) <= 0.01
            and abs(delta_acc) <= 0.01
            and abs(delta_win) <= 5e-4
            and ratio_dynamic_range >= 0.10
            and holdout_rmse <= 0.08
            and holdout_max <= 0.18
        )
        gate_stable = int(
            abs(delta_r3) <= 0.01
            and abs(delta_acc) <= 0.01
            and abs(delta_win) <= 5e-4
            and ratio_dynamic_range >= 0.10
            and holdout_rmse <= 0.10
            and holdout_max <= 0.22
        )

        scan_rows.append(
            {
                "kappa_scale": scale,
                "kappa_env_eff": float(calib["kappa_env_calibrated"]) * scale,
                "f_R3_gt_0p90": float(np.mean(r3 >= 0.90)),
                "f_hmumu_chi2_le_4": float(np.mean(chi2 <= 4.0)),
                "f_winner_gt_3": float(np.mean(winners > 3)),
                "delta_f_R3_gt_0p90": delta_r3,
                "delta_f_hmumu_chi2_le_4": delta_acc,
                "delta_f_winner_gt_3": delta_win,
                "ratio_min": float(np.min(ratio_np)),
                "ratio_max": float(np.max(ratio_np)),
                "ratio_mean": float(np.mean(ratio_np)),
                "ratio_dynamic_range": ratio_dynamic_range,
                "anchor_rmse": anchor_rmse,
                "holdout_rmse": holdout_rmse,
                "holdout_max_abs_err": holdout_max,
                "overall_rmse": overall_rmse,
                "gate_candidate": gate_candidate,
                "gate_stable": gate_stable,
            }
        )

    scan_df = pd.DataFrame(scan_rows)
    cons = scan_df[scan_df["gate_candidate"] == 1]["kappa_scale"].tolist()
    ext = scan_df[scan_df["gate_stable"] == 1]["kappa_scale"].tolist()
    summary_df = pd.DataFrame(
        [
            {
                "kappa_env_calibrated": float(calib["kappa_env_calibrated"]),
                "canonical_kappa_scale": 1.0,
                "candidate_window_min": min(cons) if cons else np.nan,
                "candidate_window_max": max(cons) if cons else np.nan,
                "stable_window_min": min(ext) if ext else np.nan,
                "stable_window_max": max(ext) if ext else np.nan,
                "holdout_threshold_rmse_candidate": 0.08,
                "holdout_threshold_maxabs_candidate": 0.18,
                "holdout_threshold_rmse_stable": 0.10,
                "holdout_threshold_maxabs_stable": 0.22,
                "base_holdout_rmse": float(holdout["holdout_rmse"]),
                "base_holdout_max_abs_err": float(holdout["holdout_max_abs_err"]),
            }
        ]
    )

    out_scan = OUTDIR / "chi_open_system_micro_kappa_window_scan.csv"
    out_summary = OUTDIR / "chi_open_system_micro_kappa_window_summary.csv"
    out_ratio = OUTDIR / "chi_open_system_micro_kappa_window_ratio_scan.csv"
    out_png = OUTDIR / "chi_open_system_micro_kappa_window.png"
    out_meta = OUTDIR / "chi_open_system_micro_kappa_window_run_meta.json"

    scan_df.to_csv(out_scan, index=False)
    summary_df.to_csv(out_summary, index=False)
    pd.DataFrame(ratio_rows).to_csv(out_ratio, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].plot(scan_df["kappa_scale"], scan_df["holdout_rmse"], marker="o", label="holdout RMSE")
    axes[0].axhline(0.08, color="k", ls="--", lw=1, label="candidate gate")
    axes[0].axhline(0.10, color="0.4", ls=":", lw=1, label="stable gate")
    axes[0].set_title("Holdout RMSE")
    axes[0].set_xlabel("kappa scale")
    axes[0].legend(fontsize=8)

    axes[1].plot(scan_df["kappa_scale"], scan_df["ratio_dynamic_range"], marker="o", color="tab:green")
    axes[1].axhline(0.10, color="k", ls="--", lw=1)
    axes[1].set_title("Ratio dynamic range")
    axes[1].set_xlabel("kappa scale")

    axes[2].plot(scan_df["kappa_scale"], scan_df["delta_f_R3_gt_0p90"], marker="o", label="Δf(R3>0.90)")
    axes[2].plot(scan_df["kappa_scale"], scan_df["delta_f_hmumu_chi2_le_4"], marker="s", label="Δf(chi2<=4)")
    axes[2].set_title("Map-level drift")
    axes[2].set_xlabel("kappa scale")
    axes[2].legend(fontsize=8)

    fig.suptitle("Open-system micro bath normalization window")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    meta = {
        "micro_csv": str(micro_csv),
        "geom_csv": str(geom_csv),
        "scales": scales,
        "anchor_Ds": anchor_vals,
        "holdout_Ds": holdout_vals,
        "notes": [
            "kappa_scale uniformly rescales gamma_phi and gamma_mix while leaving delta fixed.",
            "candidate window uses current baseline-candidate holdout thresholds together with map-level stability gates.",
            "stable window relaxes holdout thresholds slightly while retaining the same map-level stability requirements.",
        ],
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    for path in [out_scan, out_summary, out_ratio, out_png, out_meta]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(f"[saved] {out_summary}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Publish-mode runner for full_direct map evidence.

This release mode produces eight reproducible artifacts:
  1) Main-map baseline in full_direct mode (D60 x E60)
  2) Small-surface complete localized-direct audit (D21 x E41)
  3) Large-surface spot-check localized-direct audit (D60 x E21)
  4) Small-surface chain parity audit (full_direct vs cell_direct_runtime, D21 x E41)
  5) Small-surface chain parity audit (full_direct vs cell_direct_runtime_release_tuned, D21 x E41)
  6) Large-surface chain parity audit (full_direct vs cell_direct_runtime, D60 x E21)
  7) Large-surface chain parity audit (full_direct vs cell_direct_runtime_release_tuned, D60 x E21)
  8) Large-surface chain parity audit (full_direct vs cell_direct_runtime_extreme, D60 x E21)

And aggregates them into one reviewer-facing summary table:
  - output/kinetic_action_chain/full_direct_map_release_summary.csv
  - paper/full_direct_map_release_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
OUT_KIN = ROOT / "output" / "kinetic_action_chain"
OUT_HLL = ROOT / "output" / "hll_signal_strength"
PAPER = ROOT / "paper"

SCAN_HLL = ROOT / "code" / "scan_hll_signal_strengths.py"
SCAN_LOC_DIRECT = ROOT / "code" / "scan_localized_direct_surface_bias.py"
SCAN_CHAIN_AUDIT = ROOT / "code" / "scan_chain_mode_cell_direct_audit.py"


def run_cmd(name: str, cmd: List[str], expected: Path | None, force: bool) -> None:
    if expected is not None and expected.exists() and not force:
        print(f"[skip] {name}: {expected} exists")
        return
    print(f"[run] {name}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def read_hll_mumu_summary(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    row = df[df["channel"] == "mumu"].iloc[0]
    return {
        "f_chi2_le_4": float(row["f_chi2_le_4"]),
        "best_chi2": float(row["best_chi2"]),
        "best_D": float(row["best_D"]),
        "best_eta": float(row["best_eta"]),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publish full_direct map release summary.")
    ap.add_argument("--force", action="store_true", help="Recompute all steps even if outputs already exist.")
    ap.add_argument("--main-chain-mode", choices=["full_direct", "full_direct_runtime"], default="full_direct")
    ap.add_argument("--runtime-direct-force", action="store_true", help="When main-chain-mode=full_direct_runtime, force direct profile rebuild.")
    ap.add_argument("--worst-delta-mu-threshold", type=float, default=0.5, help="Point-level |Δmu_mumu| threshold for worst-points table.")
    ap.add_argument("--worst-top-k", type=int, default=50, help="Max number of rows kept in worst-points table.")
    ap.add_argument("--release-gate-mismatch-max", type=float, default=0.01, help="Release gate max for acceptance mismatch fraction.")
    ap.add_argument("--release-gate-delta-mu-max", type=float, default=1.0, help="Release gate max for max |Δmu_mumu|.")
    ap.add_argument("--enforce-release-gate", action="store_true", help="Exit with failure when release gate is not satisfied.")
    return ap.parse_args()


def _map_csv(tag: str) -> Path:
    return OUT_HLL / f"hll_signal_strength_map_{tag}.csv"


def _snap_ref_d_to_grid(ref_d: float, d_values: np.ndarray) -> float:
    if len(d_values) == 0:
        return float(ref_d)
    idx = int(np.argmin(np.abs(np.asarray(d_values, dtype=float) - float(ref_d))))
    return float(d_values[idx])


def _nearest_row_idx(df: pd.DataFrame, d_target: float, eta_target: float) -> int:
    dist = (df["D"].astype(float) - float(d_target)) ** 2 + (df["eta"].astype(float) - float(eta_target)) ** 2
    return int(dist.idxmin())


@dataclass(frozen=True)
class ReleaseGateStatus:
    gate_pass: bool
    worst_frac_acceptance_mismatch: float
    worst_max_abs_delta_mu_mumu: float
    small_frac_acceptance_mismatch: float
    small_max_abs_delta_mu_mumu: float
    large_frac_acceptance_mismatch: float
    large_max_abs_delta_mu_mumu: float
    mismatch_threshold: float
    delta_mu_threshold: float


def build_worst_points_table(
    comparisons: List[Dict[str, str]],
    delta_mu_threshold: float,
    top_k: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for comp in comparisons:
        tag_full = str(comp["full_tag"])
        tag_cmp = str(comp["cmp_tag"])
        grid = str(comp["grid"])
        scenario = str(comp["scenario"])

        full_map = pd.read_csv(_map_csv(tag_full)).sort_values(["eta", "D"]).reset_index(drop=True)
        cmp_map = pd.read_csv(_map_csv(tag_cmp)).sort_values(["eta", "D"]).reset_index(drop=True)
        if len(full_map) != len(cmp_map):
            raise RuntimeError(f"Worst-points map size mismatch for scenario={scenario}.")
        if not np.allclose(full_map["D"].to_numpy(dtype=float), cmp_map["D"].to_numpy(dtype=float), rtol=0, atol=1e-12):
            raise RuntimeError(f"Worst-points D-grid mismatch for scenario={scenario}.")
        if not np.allclose(full_map["eta"].to_numpy(dtype=float), cmp_map["eta"].to_numpy(dtype=float), rtol=0, atol=1e-12):
            raise RuntimeError(f"Worst-points eta-grid mismatch for scenario={scenario}.")

        dmu = np.abs(full_map["mu_mumu"].to_numpy(dtype=float) - cmp_map["mu_mumu"].to_numpy(dtype=float))
        chi2_full = full_map["chi2_mumu"].to_numpy(dtype=float)
        chi2_cmp = cmp_map["chi2_mumu"].to_numpy(dtype=float)
        acc_full = chi2_full <= 4.0
        acc_cmp = chi2_cmp <= 4.0
        mismatch = np.logical_xor(acc_full, acc_cmp)
        sel = np.logical_or(mismatch, dmu > float(delta_mu_threshold))
        idx = np.where(sel)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(-dmu[idx])]
        order = order[: max(1, int(top_k))]
        for k in order:
            rows.append(
                {
                    "scenario": scenario,
                    "grid": grid,
                    "D": float(full_map.iloc[int(k)]["D"]),
                    "eta": float(full_map.iloc[int(k)]["eta"]),
                    "mu_mumu_full": float(full_map.iloc[int(k)]["mu_mumu"]),
                    "mu_mumu_cmp": float(cmp_map.iloc[int(k)]["mu_mumu"]),
                    "abs_delta_mu_mumu": float(dmu[int(k)]),
                    "chi2_mumu_full": float(chi2_full[int(k)]),
                    "chi2_mumu_cmp": float(chi2_cmp[int(k)]),
                    "accept_full": bool(acc_full[int(k)]),
                    "accept_cmp": bool(acc_cmp[int(k)]),
                    "acceptance_mismatch": bool(mismatch[int(k)]),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "scenario",
                "grid",
                "D",
                "eta",
                "mu_mumu_full",
                "mu_mumu_cmp",
                "abs_delta_mu_mumu",
                "chi2_mumu_full",
                "chi2_mumu_cmp",
                "accept_full",
                "accept_cmp",
                "acceptance_mismatch",
            ]
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["acceptance_mismatch", "abs_delta_mu_mumu"], ascending=[False, False]).reset_index(drop=True)


def evaluate_release_gate(
    parity_small_release: Dict[str, float],
    parity_large_release: Dict[str, float],
    mismatch_threshold: float,
    delta_mu_threshold: float,
) -> ReleaseGateStatus:
    small_mis = float(parity_small_release["frac_acceptance_mismatch"])
    small_delta = float(parity_small_release["max_abs_delta_mu_mumu"])
    large_mis = float(parity_large_release["frac_acceptance_mismatch"])
    large_delta = float(parity_large_release["max_abs_delta_mu_mumu"])
    worst_mis = max(small_mis, large_mis)
    worst_delta = max(small_delta, large_delta)
    gate_pass = bool(worst_mis <= float(mismatch_threshold) and worst_delta <= float(delta_mu_threshold))
    return ReleaseGateStatus(
        gate_pass=gate_pass,
        worst_frac_acceptance_mismatch=worst_mis,
        worst_max_abs_delta_mu_mumu=worst_delta,
        small_frac_acceptance_mismatch=small_mis,
        small_max_abs_delta_mu_mumu=small_delta,
        large_frac_acceptance_mismatch=large_mis,
        large_max_abs_delta_mu_mumu=large_delta,
        mismatch_threshold=float(mismatch_threshold),
        delta_mu_threshold=float(delta_mu_threshold),
    )


def _extract_b_metrics(kinetics, d_val: float, eta_val: float, ref_d: float, ref_eta: float) -> Dict[str, float]:
    t_coh = float(PAPER_BASELINE["t_coh"])
    n_max = int(PAPER_BASELINE["hll_observable_nmax"])
    mode = "eft_wilson_uv_rge"
    b123 = np.array([kinetics.B_N(1, d_val), kinetics.B_N(2, d_val), kinetics.B_N(3, d_val)], dtype=float)
    yraw = np.asarray(kinetics._hll_yraw_vector(float(d_val)), dtype=float)
    m2 = np.asarray(kinetics._hll_m2_vector(float(d_val)), dtype=float)
    width_ratio = float(
        kinetics.hll_total_width_ratio_uv_rge(
            D=float(d_val),
            eta=float(eta_val),
            t_coh=t_coh,
            ref_D=float(ref_d),
            ref_eta=float(ref_eta),
            N_max=n_max,
        )
    )
    mu_mumu_model = float(
        kinetics.hll_mu_pred(
            layer_n=2,
            D=float(d_val),
            eta=float(eta_val),
            t_coh=t_coh,
            ref_D=float(ref_d),
            ref_eta=float(ref_eta),
            observable_mode=mode,
            N_max=n_max,
        )
    )
    return {
        "B1": float(b123[0]),
        "B2": float(b123[1]),
        "B3": float(b123[2]),
        "yraw1": float(yraw[0]),
        "yraw2": float(yraw[1]),
        "yraw3": float(yraw[2]),
        "m2_1": float(m2[0]),
        "m2_2": float(m2[1]),
        "m2_3": float(m2[2]),
        "width_ratio": width_ratio,
        "mu_mumu_model": mu_mumu_model,
    }


def build_b_module_diagnostics(
    tag_full_large: str,
    tag_cell_large_runtime: str,
    tag_cell_large_release_tuned: str,
    tag_cell_large_extreme: str,
) -> pd.DataFrame:
    full_map = pd.read_csv(_map_csv(tag_full_large))
    full_map = full_map.sort_values(["D", "eta"]).reset_index(drop=True)
    d_vals = np.sort(full_map["D"].astype(float).unique())
    ref_d_eff = _snap_ref_d_to_grid(float(PAPER_BASELINE["ref_D"]), d_vals)
    ref_eta = float(PAPER_BASELINE["ref_eta"])

    common = {
        "observable_mode": str(PAPER_BASELINE["hll_observable_mode"]),
        "d_min": float(PAPER_BASELINE["D_min"]),
        "d_max": float(PAPER_BASELINE["D_max"]),
        "d_num": int(PAPER_BASELINE["D_num"]),
        "uv_blend": float(PAPER_BASELINE["hll_uv_blend"]),
        "uv_m2_power": float(PAPER_BASELINE["hll_uv_m2_power"]),
        "uv_match_kappa_diag": float(PAPER_BASELINE["hll_uv_match_kappa_diag"]),
        "uv_match_kappa_offdiag": float(PAPER_BASELINE["hll_uv_match_kappa_offdiag"]),
        "uv_rge_mu_low": float(PAPER_BASELINE["hll_uv_rge_mu_low"]),
        "uv_rge_gamma_diag": float(PAPER_BASELINE["hll_uv_rge_gamma_diag"]),
        "uv_rge_gamma_offdiag": float(PAPER_BASELINE["hll_uv_rge_gamma_offdiag"]),
        "uv_rge_log_clip": float(PAPER_BASELINE["hll_uv_rge_log_clip"]),
        "runtime_direct_force": False,
        "runtime_direct_no_cache": False,
        "runtime_direct_chi_rho_max": float(PAPER_BASELINE.get("runtime_direct_chi_rho_max", 3.0)),
        "runtime_direct_chi_z_margin": float(PAPER_BASELINE.get("runtime_direct_chi_z_margin", 6.0)),
        "runtime_direct_chi_n_mu": int(PAPER_BASELINE.get("runtime_direct_chi_n_mu", 120)),
        "runtime_direct_chi_tol": float(PAPER_BASELINE.get("runtime_direct_chi_tol", 1e-8)),
        "runtime_direct_chi_maxiter": int(PAPER_BASELINE.get("runtime_direct_chi_maxiter", 30000)),
        "runtime_direct_chi_sigma": float(PAPER_BASELINE.get("runtime_direct_chi_sigma", 2.5)),
        "runtime_direct_superrad_zmax": float(PAPER_BASELINE.get("runtime_direct_superrad_zmax", 80.0)),
        "runtime_direct_superrad_ref_d": float(PAPER_BASELINE.get("runtime_direct_superrad_ref_d", 12.0)),
        "runtime_direct_superrad_n_ref": int(PAPER_BASELINE.get("runtime_direct_superrad_n_ref", 2)),
    }

    kinetics = {
        "full_direct": make_baseline_kinetics(chain_mode="full_direct", **common),
        "cell_direct_runtime": make_baseline_kinetics(chain_mode="cell_direct_runtime", **common),
        "cell_direct_runtime_release_tuned": make_baseline_kinetics(
            chain_mode="cell_direct_runtime_release_tuned",
            **common,
        ),
        "cell_direct_runtime_extreme": make_baseline_kinetics(chain_mode="cell_direct_runtime_extreme", **common),
    }

    comparisons = [
        {
            "scenario_label": "runtime_profile",
            "chain_mode": "cell_direct_runtime",
            "tag": tag_cell_large_runtime,
        },
        {
            "scenario_label": "runtime_bnorm_release_tuned_candidate",
            "chain_mode": "cell_direct_runtime_release_tuned",
            "tag": tag_cell_large_release_tuned,
        },
        {
            "scenario_label": "runtime_bnorm_extreme",
            "chain_mode": "cell_direct_runtime_extreme",
            "tag": tag_cell_large_extreme,
        },
    ]

    rows: List[Dict[str, object]] = []
    for comp in comparisons:
        cmp_map = pd.read_csv(_map_csv(str(comp["tag"]))).sort_values(["D", "eta"]).reset_index(drop=True)
        if len(cmp_map) != len(full_map):
            raise RuntimeError(
                f"B diagnostics map size mismatch for tag={comp['tag']}: full={len(full_map)} vs cmp={len(cmp_map)}"
            )
        if not np.allclose(full_map["D"].to_numpy(dtype=float), cmp_map["D"].to_numpy(dtype=float), rtol=0, atol=1e-12):
            raise RuntimeError(f"B diagnostics D-grid mismatch for tag={comp['tag']}.")
        if not np.allclose(full_map["eta"].to_numpy(dtype=float), cmp_map["eta"].to_numpy(dtype=float), rtol=0, atol=1e-12):
            raise RuntimeError(f"B diagnostics eta-grid mismatch for tag={comp['tag']}.")

        abs_delta = np.abs(full_map["mu_mumu"].to_numpy(dtype=float) - cmp_map["mu_mumu"].to_numpy(dtype=float))
        idx_max = int(np.argmax(abs_delta))
        idx_ref = _nearest_row_idx(full_map, d_target=ref_d_eff, eta_target=ref_eta)

        for point_kind, idx in (("max_abs_delta_mu_mumu", idx_max), ("reference_anchor_nearest_grid", idx_ref)):
            d_val = float(full_map.iloc[idx]["D"])
            eta_val = float(full_map.iloc[idx]["eta"])
            full_metrics = _extract_b_metrics(kinetics["full_direct"], d_val, eta_val, ref_d=ref_d_eff, ref_eta=ref_eta)
            cmp_metrics = _extract_b_metrics(kinetics[str(comp["chain_mode"])], d_val, eta_val, ref_d=ref_d_eff, ref_eta=ref_eta)
            rows.append(
                {
                    "scenario_label": str(comp["scenario_label"]),
                    "chain_mode": str(comp["chain_mode"]),
                    "tag": str(comp["tag"]),
                    "point_kind": point_kind,
                    "D": d_val,
                    "eta": eta_val,
                    "mu_mumu_full_map": float(full_map.iloc[idx]["mu_mumu"]),
                    "mu_mumu_cmp_map": float(cmp_map.iloc[idx]["mu_mumu"]),
                    "abs_delta_mu_mumu_map": float(abs_delta[idx]),
                    "B1_full": full_metrics["B1"],
                    "B2_full": full_metrics["B2"],
                    "B3_full": full_metrics["B3"],
                    "B1_cmp": cmp_metrics["B1"],
                    "B2_cmp": cmp_metrics["B2"],
                    "B3_cmp": cmp_metrics["B3"],
                    "yraw1_full": full_metrics["yraw1"],
                    "yraw2_full": full_metrics["yraw2"],
                    "yraw3_full": full_metrics["yraw3"],
                    "yraw1_cmp": cmp_metrics["yraw1"],
                    "yraw2_cmp": cmp_metrics["yraw2"],
                    "yraw3_cmp": cmp_metrics["yraw3"],
                    "m2_1_full": full_metrics["m2_1"],
                    "m2_2_full": full_metrics["m2_2"],
                    "m2_3_full": full_metrics["m2_3"],
                    "m2_1_cmp": cmp_metrics["m2_1"],
                    "m2_2_cmp": cmp_metrics["m2_2"],
                    "m2_3_cmp": cmp_metrics["m2_3"],
                    "width_ratio_full": full_metrics["width_ratio"],
                    "width_ratio_cmp": cmp_metrics["width_ratio"],
                    "mu_mumu_full_model": full_metrics["mu_mumu_model"],
                    "mu_mumu_cmp_model": cmp_metrics["mu_mumu_model"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    OUT_KIN.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    tag_main = "full_direct_map_release"
    tag_full_small = "full_direct_map_full_release_D21E41"
    tag_cell_small = "full_direct_map_cell_direct_runtime_release_D21E41"
    tag_cell_small_release_tuned = "full_direct_map_cell_direct_runtime_release_tuned_candidate_D21E41"
    tag_full_large = "full_direct_map_full_release_D60E21"
    tag_cell_large_runtime = "full_direct_map_cell_direct_runtime_release_D60E21"
    tag_cell_large_release_tuned = "full_direct_map_cell_direct_runtime_release_tuned_candidate_D60E21"
    tag_cell_large_extreme = "full_direct_map_cell_direct_runtime_extreme_release_D60E21"

    # 1) Main-map full_direct baseline.
    main_summary = OUT_HLL / f"hll_signal_strength_summary_{tag_main}.csv"
    run_cmd(
        name="scan_hll_signal_strengths_full_direct_main",
        cmd=[
            sys.executable,
            str(SCAN_HLL),
            "--chain-mode",
            str(args.main_chain_mode),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "60",
            "--tag",
            tag_main,
        ]
        + (["--runtime-direct-force"] if (str(args.main_chain_mode) == "full_direct_runtime" and bool(args.runtime_direct_force)) else []),
        expected=main_summary,
        force=bool(args.force),
    )

    # 2) Small-surface complete direct audit.
    small_summary = OUT_KIN / "localized_direct_surface_summary_Dgrid21_Egrid41.csv"
    run_cmd(
        name="localized_direct_surface_bias_small_complete",
        cmd=[
            sys.executable,
            str(SCAN_LOC_DIRECT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "21",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "41",
        ],
        expected=small_summary,
        force=bool(args.force),
    )

    # 3) Large-surface spot-check direct audit.
    large_summary = OUT_KIN / "localized_direct_surface_summary_Dgrid60_Egrid21.csv"
    run_cmd(
        name="localized_direct_surface_bias_large_spotcheck",
        cmd=[
            sys.executable,
            str(SCAN_LOC_DIRECT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "21",
        ],
        expected=large_summary,
        force=bool(args.force),
    )

    # 4) Small-surface chain-mode parity audit (full_direct vs cell_direct_runtime).
    parity_small_summary = OUT_KIN / "chain_mode_cell_direct_audit_Dgrid21_Egrid41.csv"
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_small_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "21",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "41",
            "--full-direct-tag",
            tag_full_small,
            "--cell-direct-tag",
            tag_cell_small,
        ],
        expected=parity_small_summary,
        force=bool(args.force),
    )

    # 5) Small-surface chain-mode parity audit (full_direct vs cell_direct_runtime_release_tuned).
    parity_small_release_summary = (
        OUT_KIN / "chain_mode_cell_direct_audit_Dgrid21_Egrid41_cell_direct_runtime_release_tuned.csv"
    )
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_release_tuned_small_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "21",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "41",
            "--full-direct-tag",
            tag_full_small,
            "--cell-direct-tag",
            tag_cell_small_release_tuned,
            "--cell-chain-mode",
            "cell_direct_runtime_release_tuned",
        ],
        expected=parity_small_release_summary,
        force=bool(args.force),
    )

    # 6) Large-surface chain-mode parity audit (full_direct vs cell_direct_runtime).
    parity_large_runtime_summary = OUT_KIN / "chain_mode_cell_direct_audit_Dgrid60_Egrid21.csv"
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_large_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "21",
            "--full-direct-tag",
            tag_full_large,
            "--cell-direct-tag",
            tag_cell_large_runtime,
            "--cell-chain-mode",
            "cell_direct_runtime",
        ],
        expected=parity_large_runtime_summary,
        force=bool(args.force),
    )

    # 7) Large-surface chain-mode parity audit (full_direct vs cell_direct_runtime_release_tuned).
    parity_large_release_summary = (
        OUT_KIN / "chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_release_tuned.csv"
    )
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_release_tuned_large_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "21",
            "--full-direct-tag",
            tag_full_large,
            "--cell-direct-tag",
            tag_cell_large_release_tuned,
            "--cell-chain-mode",
            "cell_direct_runtime_release_tuned",
        ],
        expected=parity_large_release_summary,
        force=bool(args.force),
    )

    # 8) Large-surface chain-mode parity audit (full_direct vs cell_direct_runtime_extreme).
    parity_large_extreme_summary = OUT_KIN / "chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_extreme.csv"
    run_cmd(
        name="chain_mode_full_vs_cell_direct_runtime_extreme_large_release",
        cmd=[
            sys.executable,
            str(SCAN_CHAIN_AUDIT),
            "--d-min",
            "4",
            "--d-max",
            "20",
            "--d-num",
            "60",
            "--eta-min",
            "0.2",
            "--eta-max",
            "4.0",
            "--eta-num",
            "21",
            "--full-direct-tag",
            tag_full_large,
            "--cell-direct-tag",
            tag_cell_large_extreme,
            "--cell-chain-mode",
            "cell_direct_runtime_extreme",
        ],
        expected=parity_large_extreme_summary,
        force=bool(args.force),
    )

    main_mumu = read_hll_mumu_summary(main_summary)
    small = pd.read_csv(small_summary).iloc[0].to_dict()
    large = pd.read_csv(large_summary).iloc[0].to_dict()
    parity_small = pd.read_csv(parity_small_summary).iloc[0].to_dict()
    parity_small_release = pd.read_csv(parity_small_release_summary).iloc[0].to_dict()
    parity_large_runtime = pd.read_csv(parity_large_runtime_summary).iloc[0].to_dict()
    parity_large_release = pd.read_csv(parity_large_release_summary).iloc[0].to_dict()
    parity_large_extreme = pd.read_csv(parity_large_extreme_summary).iloc[0].to_dict()

    # 9) B-module drift diagnostics (map-level max-drift + anchor points, D60 x E21).
    b_diag_df = build_b_module_diagnostics(
        tag_full_large=tag_full_large,
        tag_cell_large_runtime=tag_cell_large_runtime,
        tag_cell_large_release_tuned=tag_cell_large_release_tuned,
        tag_cell_large_extreme=tag_cell_large_extreme,
    )
    b_diag_csv = OUT_KIN / "full_direct_b_module_diagnostics_D60E21.csv"
    b_diag_paper_csv = PAPER / b_diag_csv.name
    b_diag_df.to_csv(b_diag_csv, index=False)
    b_diag_paper_csv.write_text(b_diag_csv.read_text())

    # 10) Worst-points table for targeted local refinement.
    worst_df = build_worst_points_table(
        comparisons=[
            {
                "scenario": "small_full_vs_cell_direct_runtime",
                "grid": "D21xE41",
                "full_tag": tag_full_small,
                "cmp_tag": tag_cell_small,
            },
            {
                "scenario": "small_full_vs_cell_direct_runtime_release_tuned",
                "grid": "D21xE41",
                "full_tag": tag_full_small,
                "cmp_tag": tag_cell_small_release_tuned,
            },
            {
                "scenario": "large_full_vs_cell_direct_runtime",
                "grid": "D60xE21",
                "full_tag": tag_full_large,
                "cmp_tag": tag_cell_large_runtime,
            },
            {
                "scenario": "large_full_vs_cell_direct_runtime_release_tuned",
                "grid": "D60xE21",
                "full_tag": tag_full_large,
                "cmp_tag": tag_cell_large_release_tuned,
            },
            {
                "scenario": "large_full_vs_cell_direct_runtime_extreme",
                "grid": "D60xE21",
                "full_tag": tag_full_large,
                "cmp_tag": tag_cell_large_extreme,
            },
        ],
        delta_mu_threshold=float(args.worst_delta_mu_threshold),
        top_k=int(args.worst_top_k),
    )
    worst_csv = OUT_KIN / "full_direct_worst_points_table.csv"
    worst_paper_csv = PAPER / worst_csv.name
    worst_df.to_csv(worst_csv, index=False)
    worst_paper_csv.write_text(worst_csv.read_text())

    # 11) Release gate status.
    gate = evaluate_release_gate(
        parity_small_release=parity_small_release,
        parity_large_release=parity_large_release,
        mismatch_threshold=float(args.release_gate_mismatch_max),
        delta_mu_threshold=float(args.release_gate_delta_mu_max),
    )
    gate_csv = OUT_KIN / "full_direct_release_gate_status.csv"
    gate_paper_csv = PAPER / gate_csv.name
    pd.DataFrame(
        [
            {
                "gate_pass": bool(gate.gate_pass),
                "worst_frac_acceptance_mismatch": float(gate.worst_frac_acceptance_mismatch),
                "worst_max_abs_delta_mu_mumu": float(gate.worst_max_abs_delta_mu_mumu),
                "small_frac_acceptance_mismatch": float(gate.small_frac_acceptance_mismatch),
                "small_max_abs_delta_mu_mumu": float(gate.small_max_abs_delta_mu_mumu),
                "large_frac_acceptance_mismatch": float(gate.large_frac_acceptance_mismatch),
                "large_max_abs_delta_mu_mumu": float(gate.large_max_abs_delta_mu_mumu),
                "mismatch_threshold": float(gate.mismatch_threshold),
                "delta_mu_threshold": float(gate.delta_mu_threshold),
                "enforced": bool(args.enforce_release_gate),
            }
        ]
    ).to_csv(gate_csv, index=False)
    gate_paper_csv.write_text(gate_csv.read_text())

    rows: List[Dict[str, object]] = [
        {
            "scenario": "main_map_full_direct_baseline",
            "grid": "D60xE60",
            "n_points": int(60 * 60),
            "f_chi2_mumu_le_4": float(main_mumu["f_chi2_le_4"]),
            "best_chi2_mumu": float(main_mumu["best_chi2"]),
            "best_D": float(main_mumu["best_D"]),
            "best_eta": float(main_mumu["best_eta"]),
            "frac_winner_mismatch": "",
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": "",
            "delta_f_chi2_mumu_le_4": "",
            "source": str(main_summary.relative_to(ROOT)),
        },
        {
            "scenario": "small_surface_complete_direct_bias",
            "grid": "D21xE41",
            "n_points": int(small["n_points"]),
            "f_chi2_mumu_le_4": float(small["f_chi2_mumu_le_4_direct"]),
            "best_chi2_mumu": float(small["best_chi2_mumu_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(small["frac_winner_mismatch"]),
            "max_abs_delta_R3": float(small["max_abs_delta_R3"]),
            "max_abs_delta_mu_mumu": float(small["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(small["delta_f_chi2_mumu_le_4"]),
            "source": str(small_summary.relative_to(ROOT)),
        },
        {
            "scenario": "large_surface_spotcheck_direct_bias",
            "grid": "D60xE21",
            "n_points": int(large["n_points"]),
            "f_chi2_mumu_le_4": float(large["f_chi2_mumu_le_4_direct"]),
            "best_chi2_mumu": float(large["best_chi2_mumu_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(large["frac_winner_mismatch"]),
            "max_abs_delta_R3": float(large["max_abs_delta_R3"]),
            "max_abs_delta_mu_mumu": float(large["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(large["delta_f_chi2_mumu_le_4"]),
            "source": str(large_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_parity_full_direct_vs_cell_direct_runtime_profile",
            "grid": "D21xE41",
            "n_points": int(parity_small["n_points"]),
            "f_chi2_mumu_le_4": float(parity_small["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_small["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_small["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_small["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_small["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_small_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_large_parity_full_direct_vs_cell_direct_runtime_profile",
            "grid": "D60xE21",
            "n_points": int(parity_large_runtime["n_points"]),
            "f_chi2_mumu_le_4": float(parity_large_runtime["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_large_runtime["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_large_runtime["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_large_runtime["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_large_runtime["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_large_runtime_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_parity_full_direct_vs_cell_direct_runtime_bnorm_release_tuned_candidate",
            "grid": "D21xE41",
            "n_points": int(parity_small_release["n_points"]),
            "f_chi2_mumu_le_4": float(parity_small_release["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_small_release["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_small_release["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_small_release["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_small_release["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_small_release_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_large_parity_full_direct_vs_cell_direct_runtime_bnorm_release_tuned_candidate",
            "grid": "D60xE21",
            "n_points": int(parity_large_release["n_points"]),
            "f_chi2_mumu_le_4": float(parity_large_release["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_large_release["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_large_release["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_large_release["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_large_release["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_large_release_summary.relative_to(ROOT)),
        },
        {
            "scenario": "chain_mode_large_parity_full_direct_vs_cell_direct_runtime_bnorm_extreme",
            "grid": "D60xE21",
            "n_points": int(parity_large_extreme["n_points"]),
            "f_chi2_mumu_le_4": float(parity_large_extreme["f_chi2_le_4_mumu_full_direct"]),
            "best_chi2_mumu": float(parity_large_extreme["best_chi2_mumu_full_direct"]),
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(parity_large_extreme["frac_acceptance_mismatch"]),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(parity_large_extreme["max_abs_delta_mu_mumu"]),
            "delta_f_chi2_mumu_le_4": float(parity_large_extreme["delta_f_chi2_le_4_mumu"]),
            "source": str(parity_large_extreme_summary.relative_to(ROOT)),
        },
        {
            "scenario": "b_module_diagnostics_large_surface",
            "grid": "D60xE21",
            "n_points": int(len(b_diag_df)),
            "f_chi2_mumu_le_4": "",
            "best_chi2_mumu": "",
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": "",
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": "",
            "delta_f_chi2_mumu_le_4": "",
            "source": str(b_diag_csv.relative_to(ROOT)),
        },
        {
            "scenario": "release_gate_status",
            "grid": "D21xE41 + D60xE21",
            "n_points": int(parity_small_release["n_points"]) + int(parity_large_release["n_points"]),
            "f_chi2_mumu_le_4": "",
            "best_chi2_mumu": "",
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": float(gate.worst_frac_acceptance_mismatch),
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(gate.worst_max_abs_delta_mu_mumu),
            "delta_f_chi2_mumu_le_4": "",
            "source": str(gate_csv.relative_to(ROOT)),
        },
        {
            "scenario": "worst_points_table",
            "grid": "D21xE41 + D60xE21",
            "n_points": int(len(worst_df)),
            "f_chi2_mumu_le_4": "",
            "best_chi2_mumu": "",
            "best_D": "",
            "best_eta": "",
            "frac_winner_mismatch": "",
            "max_abs_delta_R3": "",
            "max_abs_delta_mu_mumu": float(worst_df["abs_delta_mu_mumu"].max()) if len(worst_df) > 0 else 0.0,
            "delta_f_chi2_mumu_le_4": "",
            "source": str(worst_csv.relative_to(ROOT)),
        },
    ]

    out_csv = OUT_KIN / "full_direct_map_release_summary.csv"
    paper_csv = PAPER / out_csv.name
    out_json = OUT_KIN / "full_direct_map_release_meta.json"

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    paper_csv.write_text(out_csv.read_text())
    out_json.write_text(
        json.dumps(
            {
                "main_tag": tag_main,
                "main_chain_mode": str(args.main_chain_mode),
                "runtime_direct_force": bool(args.runtime_direct_force),
                "small_surface": str(small_summary.relative_to(ROOT)),
                "large_surface": str(large_summary.relative_to(ROOT)),
                "chain_parity_small_runtime": str(parity_small_summary.relative_to(ROOT)),
                "chain_parity_small_release_tuned": str(parity_small_release_summary.relative_to(ROOT)),
                "chain_parity_large_runtime": str(parity_large_runtime_summary.relative_to(ROOT)),
                "chain_parity_large_release_tuned": str(parity_large_release_summary.relative_to(ROOT)),
                "chain_parity_large_extreme": str(parity_large_extreme_summary.relative_to(ROOT)),
                "b_module_diagnostics": str(b_diag_csv.relative_to(ROOT)),
                "worst_points_table": str(worst_csv.relative_to(ROOT)),
                "release_gate_status": str(gate_csv.relative_to(ROOT)),
                "release_gate_pass": bool(gate.gate_pass),
                "release_gate_thresholds": {
                    "mismatch_max": float(gate.mismatch_threshold),
                    "max_abs_delta_mu_mumu_max": float(gate.delta_mu_threshold),
                },
                "summary_csv": str(out_csv.relative_to(ROOT)),
            },
            indent=2,
        )
    )

    print(f"[saved] {out_csv}")
    print(f"[saved] {paper_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {b_diag_csv}")
    print(f"[saved] {b_diag_paper_csv}")
    print(f"[saved] {worst_csv}")
    print(f"[saved] {worst_paper_csv}")
    print(f"[saved] {gate_csv}")
    print(f"[saved] {gate_paper_csv}")
    print(
        "[release gate]",
        f"pass={gate.gate_pass}",
        f"worst_mismatch={gate.worst_frac_acceptance_mismatch:.6f}",
        f"worst_max_abs_delta_mu_mumu={gate.worst_max_abs_delta_mu_mumu:.6f}",
        f"thresholds(mismatch<={gate.mismatch_threshold:.6f}, delta_mu<={gate.delta_mu_threshold:.6f})",
    )
    for row in rows:
        print(row)

    if bool(args.enforce_release_gate) and not bool(gate.gate_pass):
        raise RuntimeError(
            "Release gate failed: "
            f"worst_frac_acceptance_mismatch={gate.worst_frac_acceptance_mismatch:.6f} "
            f"(limit {gate.mismatch_threshold:.6f}), "
            f"worst_max_abs_delta_mu_mumu={gate.worst_max_abs_delta_mu_mumu:.6f} "
            f"(limit {gate.delta_mu_threshold:.6f})."
        )


if __name__ == "__main__":
    main()

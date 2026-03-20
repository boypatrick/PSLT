#!/usr/bin/env python3
"""
Audit map-level bias between chain_mode=full_direct and a selected cell-direct chain mode.

This script runs scan_hll_signal_strengths.py twice on the same grid:
  1) chain_mode=full_direct
  2) chain_mode=<cell_chain_mode>

Then it compares map outputs and writes a compact summary table.

Outputs:
  - output/kinetic_action_chain/chain_mode_cell_direct_audit_Dgrid{N}_Egrid{M}[_{mode}].csv
  - output/kinetic_action_chain/chain_mode_cell_direct_audit_Dgrid{N}_Egrid{M}[_{mode}].json
  - paper/chain_mode_cell_direct_audit_Dgrid{N}_Egrid{M}[_{mode}].csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
SCAN_SCRIPT = ROOT / "code" / "scan_hll_signal_strengths.py"
HLL_OUTDIR = ROOT / "output" / "hll_signal_strength"


def _run_scan(
    chain_mode: str,
    tag: str,
    d_min: float,
    d_max: float,
    d_num: int,
    eta_min: float,
    eta_max: float,
    eta_num: int,
    observable_mode: str,
    ref_mode: str,
    ref_d: float,
    ref_eta: float,
    runtime_direct_b_release_profile_blend_override: float | None = None,
    runtime_direct_b_release_profile_blend_csv_override: str | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(SCAN_SCRIPT),
        "--chain-mode",
        str(chain_mode),
        "--tag",
        str(tag),
        "--d-min",
        str(float(d_min)),
        "--d-max",
        str(float(d_max)),
        "--d-num",
        str(int(d_num)),
        "--eta-min",
        str(float(eta_min)),
        "--eta-max",
        str(float(eta_max)),
        "--eta-num",
        str(int(eta_num)),
        "--observable-mode",
        str(observable_mode),
        "--ref-mode",
        str(ref_mode),
        "--ref-d",
        str(float(ref_d)),
        "--ref-eta",
        str(float(ref_eta)),
        "--skip-paper-copy",
    ]
    if runtime_direct_b_release_profile_blend_override is not None:
        cmd.extend(
            [
                "--runtime-direct-b-release-profile-blend-override",
                str(float(runtime_direct_b_release_profile_blend_override)),
            ]
        )
    if runtime_direct_b_release_profile_blend_csv_override not in {None, ""}:
        cmd.extend(
            [
                "--runtime-direct-b-release-profile-blend-csv-override",
                str(runtime_direct_b_release_profile_blend_csv_override),
            ]
        )
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _read_map(tag: str) -> pd.DataFrame:
    path = HLL_OUTDIR / f"hll_signal_strength_map_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing map CSV: {path}")
    return pd.read_csv(path)


def _read_summary(tag: str) -> pd.DataFrame:
    path = HLL_OUTDIR / f"hll_signal_strength_summary_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    return pd.read_csv(path)


def _p99_abs(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(arr), 99.0))


def _snap_ref_d_to_grid(ref_d: float, d_min: float, d_max: float, d_num: int) -> tuple[float, bool]:
    d_vals = np.linspace(float(d_min), float(d_max), int(d_num))
    idx = int(np.argmin(np.abs(d_vals - float(ref_d))))
    snapped = float(d_vals[idx])
    changed = not np.isclose(snapped, float(ref_d), rtol=0.0, atol=1e-10)
    return snapped, changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare chain_mode=full_direct vs a selected cell-direct chain mode on the same H->ll map grid.")
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=20.0)
    ap.add_argument("--d-num", type=int, default=21)
    ap.add_argument("--eta-min", type=float, default=0.2)
    ap.add_argument("--eta-max", type=float, default=4.0)
    ap.add_argument("--eta-num", type=int, default=41)
    ap.add_argument("--observable-mode", type=str, default="eft_wilson_uv_rge")
    ap.add_argument("--ref-mode", type=str, default="fixed")
    ap.add_argument("--ref-d", type=float, default=10.0)
    ap.add_argument("--ref-eta", type=float, default=1.0)
    ap.add_argument(
        "--runtime-direct-b-release-profile-blend-override",
        type=float,
        default=None,
        help="Optional temporary override passed through to scan_hll_signal_strengths.py.",
    )
    ap.add_argument(
        "--runtime-direct-b-release-profile-blend-csv-override",
        type=str,
        default=None,
        help="Optional D-profile CSV override passed through to scan_hll_signal_strengths.py.",
    )
    ap.add_argument("--full-direct-tag", type=str, default="chain_mode_full_direct")
    ap.add_argument("--cell-direct-tag", type=str, default="chain_mode_cell_direct_runtime")
    ap.add_argument(
        "--cell-chain-mode",
        type=str,
        default="cell_direct_runtime",
        choices=[
            "cell_direct_runtime",
            "cell_direct_runtime_release",
            "cell_direct_runtime_release_widthstable",
            "cell_direct_runtime_release_combo",
            "cell_direct_runtime_release_bandcombo",
            "cell_direct_runtime_release_bandm2",
            "cell_direct_runtime_release_tailm2",
            "cell_direct_runtime_release_tailm2gnorm",
            "cell_direct_runtime_release_tuned",
            "cell_direct_runtime_extreme",
        ],
        help="Target chain mode to compare against full_direct.",
    )
    args = ap.parse_args()

    if int(args.d_num) < 2 or int(args.eta_num) < 2:
        raise ValueError("--d-num and --eta-num must be >= 2.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    ref_d_eff = float(args.ref_d)
    ref_d_snapped = False
    if str(args.ref_mode) == "fixed":
        ref_d_eff, ref_d_snapped = _snap_ref_d_to_grid(
            ref_d=float(args.ref_d),
            d_min=float(args.d_min),
            d_max=float(args.d_max),
            d_num=int(args.d_num),
        )
        if ref_d_snapped:
            print(
                f"[info] fixed reference snapped to active D-grid for fair full_direct/cell_direct comparison: "
                f"{float(args.ref_d):.6g} -> {ref_d_eff:.6g}"
            )

    _run_scan(
        chain_mode="full_direct",
        tag=str(args.full_direct_tag),
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        eta_min=float(args.eta_min),
        eta_max=float(args.eta_max),
        eta_num=int(args.eta_num),
        observable_mode=str(args.observable_mode),
        ref_mode=str(args.ref_mode),
        ref_d=float(ref_d_eff),
        ref_eta=float(args.ref_eta),
        runtime_direct_b_release_profile_blend_override=(
            None
            if args.runtime_direct_b_release_profile_blend_override is None
            else float(args.runtime_direct_b_release_profile_blend_override)
        ),
        runtime_direct_b_release_profile_blend_csv_override=(
            None
            if args.runtime_direct_b_release_profile_blend_csv_override in {None, ""}
            else str(args.runtime_direct_b_release_profile_blend_csv_override)
        ),
    )
    _run_scan(
        chain_mode=str(args.cell_chain_mode),
        tag=str(args.cell_direct_tag),
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        eta_min=float(args.eta_min),
        eta_max=float(args.eta_max),
        eta_num=int(args.eta_num),
        observable_mode=str(args.observable_mode),
        ref_mode=str(args.ref_mode),
        ref_d=float(ref_d_eff),
        ref_eta=float(args.ref_eta),
        runtime_direct_b_release_profile_blend_override=(
            None
            if args.runtime_direct_b_release_profile_blend_override is None
            else float(args.runtime_direct_b_release_profile_blend_override)
        ),
        runtime_direct_b_release_profile_blend_csv_override=(
            None
            if args.runtime_direct_b_release_profile_blend_csv_override in {None, ""}
            else str(args.runtime_direct_b_release_profile_blend_csv_override)
        ),
    )

    full_map = _read_map(str(args.full_direct_tag))
    cell_map = _read_map(str(args.cell_direct_tag))

    keys = ["D", "eta"]
    merged = full_map.merge(cell_map, on=keys, suffixes=("_full", "_cell"))
    if len(merged) != len(full_map) or len(merged) != len(cell_map):
        raise RuntimeError("Map rows do not align between full_direct and selected cell-direct runs.")

    d_mu_ee = (merged["mu_ee_cell"] - merged["mu_ee_full"]).to_numpy(dtype=float)
    d_mu_mumu = (merged["mu_mumu_cell"] - merged["mu_mumu_full"]).to_numpy(dtype=float)
    d_mu_tautau = (merged["mu_tautau_cell"] - merged["mu_tautau_full"]).to_numpy(dtype=float)
    d_chi2 = (merged["chi2_mumu_cell"] - merged["chi2_mumu_full"]).to_numpy(dtype=float)

    accept_full = (merged["chi2_mumu_full"].to_numpy(dtype=float) <= 4.0)
    accept_cell = (merged["chi2_mumu_cell"].to_numpy(dtype=float) <= 4.0)
    mismatch = accept_full ^ accept_cell

    full_sum = _read_summary(str(args.full_direct_tag))
    cell_sum = _read_summary(str(args.cell_direct_tag))
    full_mumu = full_sum[full_sum["channel"] == "mumu"].iloc[0]
    cell_mumu = cell_sum[cell_sum["channel"] == "mumu"].iloc[0]

    row: Dict[str, float | str | int] = {
        "d_num": int(args.d_num),
        "eta_num": int(args.eta_num),
        "n_points": int(len(merged)),
        "observable_mode": str(args.observable_mode),
        "ref_mode": str(args.ref_mode),
        "ref_D_input": float(args.ref_d),
        "ref_D_effective": float(ref_d_eff),
        "ref_D_snapped_to_grid": bool(ref_d_snapped),
        "ref_eta": float(args.ref_eta),
        "runtime_direct_b_release_profile_blend_override": (
            None
            if args.runtime_direct_b_release_profile_blend_override is None
            else float(args.runtime_direct_b_release_profile_blend_override)
        ),
        "runtime_direct_b_release_profile_blend_csv_override": (
            None
            if args.runtime_direct_b_release_profile_blend_csv_override in {None, ""}
            else str(args.runtime_direct_b_release_profile_blend_csv_override)
        ),
        "full_direct_tag": str(args.full_direct_tag),
        "cell_direct_tag": str(args.cell_direct_tag),
        "cell_chain_mode": str(args.cell_chain_mode),
        "frac_acceptance_mismatch": float(np.mean(mismatch)),
        "n_acceptance_mismatch": int(np.sum(mismatch)),
        "max_abs_delta_mu_ee": float(np.max(np.abs(d_mu_ee))),
        "p99_abs_delta_mu_ee": _p99_abs(d_mu_ee),
        "max_abs_delta_mu_mumu": float(np.max(np.abs(d_mu_mumu))),
        "p99_abs_delta_mu_mumu": _p99_abs(d_mu_mumu),
        "max_abs_delta_mu_tautau": float(np.max(np.abs(d_mu_tautau))),
        "p99_abs_delta_mu_tautau": _p99_abs(d_mu_tautau),
        "max_abs_delta_chi2_mumu": float(np.max(np.abs(d_chi2))),
        "p99_abs_delta_chi2_mumu": _p99_abs(d_chi2),
        "f_chi2_le_4_mumu_full_direct": float(full_mumu["f_chi2_le_4"]),
        "f_chi2_le_4_mumu_cell_direct": float(cell_mumu["f_chi2_le_4"]),
        "delta_f_chi2_le_4_mumu": float(cell_mumu["f_chi2_le_4"] - full_mumu["f_chi2_le_4"]),
        "best_chi2_mumu_full_direct": float(full_mumu["best_chi2"]),
        "best_chi2_mumu_cell_direct": float(cell_mumu["best_chi2"]),
        "delta_best_chi2_mumu": float(cell_mumu["best_chi2"] - full_mumu["best_chi2"]),
    }

    tag_base = f"Dgrid{int(args.d_num)}_Egrid{int(args.eta_num)}"
    tag = (
        tag_base if str(args.cell_chain_mode) == "cell_direct_runtime" else f"{tag_base}_{str(args.cell_chain_mode)}"
    )
    out_csv = OUTDIR / f"chain_mode_cell_direct_audit_{tag}.csv"
    out_json = OUTDIR / f"chain_mode_cell_direct_audit_{tag}.json"
    paper_csv = PAPER_DIR / out_csv.name

    pd.DataFrame([row]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(row, indent=2))
    paper_csv.write_text(out_csv.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {paper_csv}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

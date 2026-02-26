#!/usr/bin/env python3
"""
Audit map-level bias between chain_mode=auto and chain_mode=full_direct.

This script runs scan_hll_signal_strengths.py twice on the same grid:
  1) chain_mode=auto
  2) chain_mode=full_direct

Then it compares map outputs and writes a compact summary table.

Outputs:
  - output/kinetic_action_chain/chain_mode_full_direct_audit_Dgrid{N}_Egrid{M}.csv
  - output/kinetic_action_chain/chain_mode_full_direct_audit_Dgrid{N}_Egrid{M}.json
  - paper/chain_mode_full_direct_audit_Dgrid{N}_Egrid{M}.csv
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
    ap = argparse.ArgumentParser(description="Compare chain_mode=auto vs full_direct on the same H->ll map grid.")
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
    ap.add_argument("--auto-tag", type=str, default="chain_mode_auto")
    ap.add_argument("--full-direct-tag", type=str, default="chain_mode_full_direct")
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
            print(f"[info] fixed reference snapped to active D-grid for fair auto/full_direct comparison: {float(args.ref_d):.6g} -> {ref_d_eff:.6g}")

    _run_scan(
        chain_mode="auto",
        tag=str(args.auto_tag),
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
    )

    auto_map = _read_map(str(args.auto_tag))
    full_map = _read_map(str(args.full_direct_tag))

    keys = ["D", "eta"]
    merged = auto_map.merge(full_map, on=keys, suffixes=("_auto", "_full"))
    if len(merged) != len(auto_map) or len(merged) != len(full_map):
        raise RuntimeError("Map rows do not align between auto and full_direct runs.")

    d_mu_ee = (merged["mu_ee_full"] - merged["mu_ee_auto"]).to_numpy(dtype=float)
    d_mu_mumu = (merged["mu_mumu_full"] - merged["mu_mumu_auto"]).to_numpy(dtype=float)
    d_mu_tautau = (merged["mu_tautau_full"] - merged["mu_tautau_auto"]).to_numpy(dtype=float)
    d_chi2 = (merged["chi2_mumu_full"] - merged["chi2_mumu_auto"]).to_numpy(dtype=float)

    accept_auto = (merged["chi2_mumu_auto"].to_numpy(dtype=float) <= 4.0)
    accept_full = (merged["chi2_mumu_full"].to_numpy(dtype=float) <= 4.0)
    mismatch = accept_auto ^ accept_full

    auto_sum = _read_summary(str(args.auto_tag))
    full_sum = _read_summary(str(args.full_direct_tag))
    auto_mumu = auto_sum[auto_sum["channel"] == "mumu"].iloc[0]
    full_mumu = full_sum[full_sum["channel"] == "mumu"].iloc[0]

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
        "auto_tag": str(args.auto_tag),
        "full_direct_tag": str(args.full_direct_tag),
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
        "f_chi2_le_4_mumu_auto": float(auto_mumu["f_chi2_le_4"]),
        "f_chi2_le_4_mumu_full_direct": float(full_mumu["f_chi2_le_4"]),
        "delta_f_chi2_le_4_mumu": float(full_mumu["f_chi2_le_4"] - auto_mumu["f_chi2_le_4"]),
        "best_chi2_mumu_auto": float(auto_mumu["best_chi2"]),
        "best_chi2_mumu_full_direct": float(full_mumu["best_chi2"]),
        "delta_best_chi2_mumu": float(full_mumu["best_chi2"] - auto_mumu["best_chi2"]),
    }

    tag = f"Dgrid{int(args.d_num)}_Egrid{int(args.eta_num)}"
    out_csv = OUTDIR / f"chain_mode_full_direct_audit_{tag}.csv"
    out_json = OUTDIR / f"chain_mode_full_direct_audit_{tag}.json"
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

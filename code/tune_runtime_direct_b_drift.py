#!/usr/bin/env python3
"""
Tune runtime-direct B-module parameters to reduce map-level drift.

Two-stage workflow:
  1) Small-surface search on D21 x E41 against strict full_direct baseline map.
  2) Large-surface gate on top-K candidates on D60 x E21.

Outputs:
  - output/robustness/runtime_direct_b_tuning_small_D21E41.csv
  - output/robustness/runtime_direct_b_tuning_large_D60E21.csv
  - output/robustness/runtime_direct_b_tuning_best.json
  - mirrored copies under paper/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Avoid matplotlib/font cache stalls when scan_hll_signal_strengths imports pyplot.
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from scan_hll_signal_strengths import (
    PAPER_BASELINE,
    compute_maps,
    make_baseline_kinetics,
    snap_ref_d_for_full_direct,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_ROB = ROOT / "output" / "robustness"
OUT_HLL = ROOT / "output" / "hll_signal_strength"
PAPER = ROOT / "paper"


def _compute_map(
    chain_mode: str,
    params_override: Dict[str, float],
    d_num: int,
    eta_num: int,
    ref_d: float,
    ref_eta: float,
) -> pd.DataFrame:
    kin = make_baseline_kinetics(
        observable_mode=str(PAPER_BASELINE["hll_observable_mode"]),
        chain_mode=str(chain_mode),
        d_min=float(PAPER_BASELINE["D_min"]),
        d_max=float(PAPER_BASELINE["D_max"]),
        d_num=int(d_num),
        uv_blend=float(PAPER_BASELINE["hll_uv_blend"]),
        uv_m2_power=float(PAPER_BASELINE["hll_uv_m2_power"]),
        uv_match_kappa_diag=float(PAPER_BASELINE["hll_uv_match_kappa_diag"]),
        uv_match_kappa_offdiag=float(PAPER_BASELINE["hll_uv_match_kappa_offdiag"]),
        uv_rge_mu_low=float(PAPER_BASELINE["hll_uv_rge_mu_low"]),
        uv_rge_gamma_diag=float(PAPER_BASELINE["hll_uv_rge_gamma_diag"]),
        uv_rge_gamma_offdiag=float(PAPER_BASELINE["hll_uv_rge_gamma_offdiag"]),
        uv_rge_log_clip=float(PAPER_BASELINE["hll_uv_rge_log_clip"]),
        runtime_direct_force=False,
        runtime_direct_no_cache=False,
        runtime_direct_chi_rho_max=float(PAPER_BASELINE.get("runtime_direct_chi_rho_max", 3.0)),
        runtime_direct_chi_z_margin=float(PAPER_BASELINE.get("runtime_direct_chi_z_margin", 6.0)),
        runtime_direct_chi_n_mu=int(PAPER_BASELINE.get("runtime_direct_chi_n_mu", 120)),
        runtime_direct_chi_tol=float(PAPER_BASELINE.get("runtime_direct_chi_tol", 1e-8)),
        runtime_direct_chi_maxiter=int(PAPER_BASELINE.get("runtime_direct_chi_maxiter", 30000)),
        runtime_direct_chi_sigma=float(PAPER_BASELINE.get("runtime_direct_chi_sigma", 2.5)),
        runtime_direct_superrad_zmax=float(PAPER_BASELINE.get("runtime_direct_superrad_zmax", 80.0)),
        runtime_direct_superrad_ref_d=float(PAPER_BASELINE.get("runtime_direct_superrad_ref_d", 12.0)),
        runtime_direct_superrad_n_ref=int(PAPER_BASELINE.get("runtime_direct_superrad_n_ref", 2)),
    )

    # Override runtime-direct B tuning knobs for this trial.
    b_override_touched = False
    for key, val in params_override.items():
        setattr(kin.params, key, val)
        if str(key).startswith("runtime_direct_b_"):
            b_override_touched = True
    if b_override_touched:
        # Rebuild B-mode state/caches when runtime-direct B knobs are changed
        # post-construction.
        kin._init_b_profiles()  # type: ignore[attr-defined]

    d_vals, eta_vals, maps, _ = compute_maps(
        kinetics=kin,
        ref_d=float(ref_d),
        ref_eta=float(ref_eta),
        observable_mode=str(PAPER_BASELINE["hll_observable_mode"]),
        d_min=float(PAPER_BASELINE["D_min"]),
        d_max=float(PAPER_BASELINE["D_max"]),
        d_num=int(d_num),
        eta_min=float(PAPER_BASELINE["eta_min"]),
        eta_max=float(PAPER_BASELINE["eta_max"]),
        eta_num=int(eta_num),
    )

    rows: List[Dict[str, float]] = []
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            mu = float(maps["mumu"][i, j])
            chi2 = float(((mu - 1.4) / 0.4) ** 2)
            rows.append({"D": float(d), "eta": float(eta), "mu_mumu": mu, "chi2_mumu": chi2})
    return pd.DataFrame(rows).sort_values(["eta", "D"]).reset_index(drop=True)


def _metrics(full_df: pd.DataFrame, cand_df: pd.DataFrame) -> Dict[str, float]:
    if len(full_df) != len(cand_df):
        raise RuntimeError(f"Row count mismatch: full={len(full_df)} cand={len(cand_df)}")
    if not np.allclose(full_df["D"].to_numpy(), cand_df["D"].to_numpy(), atol=1e-12, rtol=0.0):
        raise RuntimeError("D grid mismatch.")
    if not np.allclose(full_df["eta"].to_numpy(), cand_df["eta"].to_numpy(), atol=1e-12, rtol=0.0):
        raise RuntimeError("eta grid mismatch.")

    dmu = np.abs(full_df["mu_mumu"].to_numpy() - cand_df["mu_mumu"].to_numpy())
    acc_full = (full_df["chi2_mumu"].to_numpy() <= 4.0)
    acc_cand = (cand_df["chi2_mumu"].to_numpy() <= 4.0)
    mismatch = np.logical_xor(acc_full, acc_cand)

    return {
        "max_abs_delta_mu_mumu": float(np.max(dmu)),
        "p99_abs_delta_mu_mumu": float(np.quantile(dmu, 0.99)),
        "mean_abs_delta_mu_mumu": float(np.mean(dmu)),
        "frac_acceptance_mismatch": float(np.mean(mismatch)),
        "n_acceptance_mismatch": int(np.sum(mismatch)),
        "delta_f_chi2_le_4": float(np.mean(acc_cand) - np.mean(acc_full)),
    }


def _sample_candidates(n: int, seed: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    out: List[Dict[str, float]] = []

    # Include baseline default first.
    out.append(
        {
            "runtime_direct_b_window_k": 1,
            "runtime_direct_b_window_gap_scale": 1.0,
            "runtime_direct_b_window_sigma_mult": 2.0,
            "runtime_direct_b_window_floor": 0.05,
            "runtime_direct_b_flavor_sigma_power": 0.08,
            "runtime_direct_b_flavor_sigma_min_scale": 0.70,
            "runtime_direct_b_flavor_sigma_max_scale": 1.50,
            "runtime_direct_b_profile_blend": 0.0,
        }
    )

    for _ in range(max(0, n - 1)):
        min_scale = float(rng.uniform(0.50, 0.90))
        max_scale = float(rng.uniform(max(min_scale + 0.10, 1.10), 2.00))
        out.append(
            {
                "runtime_direct_b_window_k": int(rng.integers(0, 4)),
                "runtime_direct_b_window_gap_scale": float(rng.uniform(0.70, 1.40)),
                "runtime_direct_b_window_sigma_mult": float(rng.uniform(1.00, 3.20)),
                "runtime_direct_b_window_floor": float(rng.uniform(0.015, 0.12)),
                "runtime_direct_b_flavor_sigma_power": float(rng.uniform(0.00, 0.18)),
                "runtime_direct_b_flavor_sigma_min_scale": min_scale,
                "runtime_direct_b_flavor_sigma_max_scale": max_scale,
                "runtime_direct_b_profile_blend": float(rng.uniform(0.0, 1.0)),
            }
        )
    return out


def _objective(row: Dict[str, float]) -> float:
    # prioritize acceptance stability, then suppress max drift.
    return float(row["frac_acceptance_mismatch"] * 100.0 + row["max_abs_delta_mu_mumu"])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tune runtime-direct B-module parameters against full_direct baseline map.")
    ap.add_argument("--n-candidates", type=int, default=16, help="Total candidate count including baseline default.")
    ap.add_argument("--top-k-large", type=int, default=4, help="Number of small-grid winners to gate on D60xE21.")
    ap.add_argument("--seed", type=int, default=20260301)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROB.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    # Use the same fixed-anchor convention as release parity gates.
    ref_eta = float(PAPER_BASELINE["ref_eta"])
    d_vals_small = np.linspace(float(PAPER_BASELINE["D_min"]), float(PAPER_BASELINE["D_max"]), 21)
    d_vals_large = np.linspace(float(PAPER_BASELINE["D_min"]), float(PAPER_BASELINE["D_max"]), 60)
    ref_d_small, _ = snap_ref_d_for_full_direct("full_direct", float(PAPER_BASELINE["ref_D"]), d_vals_small)
    ref_d_large, _ = snap_ref_d_for_full_direct("full_direct", float(PAPER_BASELINE["ref_D"]), d_vals_large)

    full_small = _compute_map(
        chain_mode="full_direct",
        params_override={},
        d_num=21,
        eta_num=41,
        ref_d=float(ref_d_small),
        ref_eta=float(ref_eta),
    )
    full_large = _compute_map(
        chain_mode="full_direct",
        params_override={},
        d_num=60,
        eta_num=21,
        ref_d=float(ref_d_large),
        ref_eta=float(ref_eta),
    )

    candidates = _sample_candidates(int(args.n_candidates), int(args.seed))

    small_rows: List[Dict[str, float]] = []
    for idx, cand in enumerate(candidates):
        cand_df = _compute_map(
            chain_mode="cell_direct_runtime_release",
            params_override=cand,
            d_num=21,
            eta_num=41,
            ref_d=float(ref_d_small),
            ref_eta=float(ref_eta),
        )
        m = _metrics(full_small, cand_df)
        row = {"candidate_id": idx, **cand, **m}
        row["objective"] = _objective(row)
        small_rows.append(row)
        print(
            f"[small] id={idx:02d} obj={row['objective']:.4f} "
            f"mismatch={row['frac_acceptance_mismatch']:.4f} maxΔμ={row['max_abs_delta_mu_mumu']:.4f}"
        )

    small_df = pd.DataFrame(small_rows).sort_values(["objective", "max_abs_delta_mu_mumu"]).reset_index(drop=True)
    top_k = int(max(1, min(args.top_k_large, len(small_df))))

    large_rows: List[Dict[str, float]] = []
    for _, r in small_df.head(top_k).iterrows():
        cand = {
            "runtime_direct_b_window_k": int(r["runtime_direct_b_window_k"]),
            "runtime_direct_b_window_gap_scale": float(r["runtime_direct_b_window_gap_scale"]),
            "runtime_direct_b_window_sigma_mult": float(r["runtime_direct_b_window_sigma_mult"]),
            "runtime_direct_b_window_floor": float(r["runtime_direct_b_window_floor"]),
            "runtime_direct_b_flavor_sigma_power": float(r["runtime_direct_b_flavor_sigma_power"]),
            "runtime_direct_b_flavor_sigma_min_scale": float(r["runtime_direct_b_flavor_sigma_min_scale"]),
            "runtime_direct_b_flavor_sigma_max_scale": float(r["runtime_direct_b_flavor_sigma_max_scale"]),
            "runtime_direct_b_profile_blend": float(r["runtime_direct_b_profile_blend"]),
        }
        cand_df = _compute_map(
            chain_mode="cell_direct_runtime_release",
            params_override=cand,
            d_num=60,
            eta_num=21,
            ref_d=float(ref_d_large),
            ref_eta=float(ref_eta),
        )
        m = _metrics(full_large, cand_df)
        row = {"candidate_id": int(r["candidate_id"]), **cand, **m}
        row["objective"] = _objective(row)
        large_rows.append(row)
        print(
            f"[large] id={int(r['candidate_id']):02d} obj={row['objective']:.4f} "
            f"mismatch={row['frac_acceptance_mismatch']:.4f} maxΔμ={row['max_abs_delta_mu_mumu']:.4f}"
        )

    large_df = pd.DataFrame(large_rows).sort_values(["objective", "max_abs_delta_mu_mumu"]).reset_index(drop=True)

    small_csv = OUT_ROB / "runtime_direct_b_tuning_small_D21E41.csv"
    large_csv = OUT_ROB / "runtime_direct_b_tuning_large_D60E21.csv"
    best_json = OUT_ROB / "runtime_direct_b_tuning_best.json"
    small_df.to_csv(small_csv, index=False)
    large_df.to_csv(large_csv, index=False)

    best_payload = {
        "search_seed": int(args.seed),
        "n_candidates": int(args.n_candidates),
        "top_k_large": int(top_k),
        "small_best": small_df.iloc[0].to_dict() if len(small_df) > 0 else {},
        "large_best": large_df.iloc[0].to_dict() if len(large_df) > 0 else {},
    }
    best_json.write_text(json.dumps(best_payload, indent=2))

    # mirror to paper/
    (PAPER / small_csv.name).write_text(small_csv.read_text())
    (PAPER / large_csv.name).write_text(large_csv.read_text())
    (PAPER / best_json.name).write_text(best_json.read_text())

    print(f"[saved] {small_csv}")
    print(f"[saved] {large_csv}")
    print(f"[saved] {best_json}")
    print(f"[saved] {PAPER / small_csv.name}")
    print(f"[saved] {PAPER / large_csv.name}")
    print(f"[saved] {PAPER / best_json.name}")


if __name__ == "__main__":
    main()

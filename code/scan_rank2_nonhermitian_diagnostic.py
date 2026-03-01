#!/usr/bin/env python3
"""
Rank-2 non-Hermitian diagnostic scan (map-level).

Purpose:
  Quantify map-level drift when replacing the Hermitian 2x2 rate matrix in
  Gamma_N with a minimal non-Hermitian extension:

    M = [[g1 - i*gamma_env, eps_mix],
         [eps_mix,          g2 - i*gamma_env*(1+kappa_split)]]

  and Gamma_N is taken from the largest real-part eigenvalue after the same
  WKB prefactor.

Outputs:
  - output/kinetic_action_chain/nonhermitian_rank2_diagnostic_<tag>.csv
  - output/kinetic_action_chain/nonhermitian_rank2_diagnostic_<tag>.json
  - paper/nonhermitian_rank2_diagnostic_<tag>.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pslt_lib import PSLTKinetics, PSLTParameters
from scan_hll_signal_strengths import (
    PAPER_BASELINE,
    compute_maps,
    make_baseline_kinetics,
    snap_ref_d_for_full_direct,
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER = ROOT / "paper"


class NonHermitianRank2Kinetics(PSLTKinetics):
    """PSLTKinetics with non-Hermitian rank-2 diagnostic Gamma_N closure."""

    def __init__(self, params: PSLTParameters, kappa_diag_scale: float, kappa_split: float):
        super().__init__(params)
        self.kappa_diag_scale = float(kappa_diag_scale)
        self.kappa_split = float(kappa_split)
        self._gamma_prefactor_cache_nh: Dict[Tuple[int, float], float] = {}

    def calculate_gamma_N(self, N: int, D: float, eta: float) -> float:
        if D <= 0:
            return 0.0

        eta_eff = self.eta_effective(D, eta)
        key = (int(N), float(round(D, 8)))
        use_pref_cache = True
        if (
            self._chi_mode_active == "localized_runtime_direct"
            or self._gamma_mode_active == "action_runtime_direct"
        ) and (not self.params.runtime_direct_use_cache):
            use_pref_cache = False

        if use_pref_cache:
            pref = self._gamma_prefactor_cache_nh.get(key, None)
            if pref is not None:
                return max(float(eta_eff * pref), 0.0)

        mu = self.params.M / D
        omega_h = self.params.Omega_H * self.params.M
        alpha = 1.0 / D

        def gamma_sr(l: int, m: int, a_l: float) -> float:
            w = self.omega_N(mu, N, D)
            delta_tilde = (m * omega_h - w) / self.params.M
            if delta_tilde <= 0.0:
                return 0.0
            return self.params.M * a_l * (alpha ** (4 * l + 4)) * delta_tilde

        a1_eff, a2_eff = self._gamma_action_A12(D)
        g1 = gamma_sr(1, 1, a1_eff)
        g2 = gamma_sr(2, 2, a2_eff)

        w_val = self.omega_N(mu, N, D)
        s_wkb = self.action_S(mu, D, w_val)
        chi_eff = self.chi_effective(D)
        eps_mix = chi_eff * math.sqrt(g1 * g2) if (g1 > 0.0 and g2 > 0.0) else 0.0

        gamma_env_1 = self.kappa_diag_scale * g1
        gamma_env_2 = self.kappa_diag_scale * (1.0 + self.kappa_split) * g2
        mat = np.array(
            [
                [complex(g1, -gamma_env_1), complex(eps_mix, 0.0)],
                [complex(eps_mix, 0.0), complex(g2, -gamma_env_2)],
            ],
            dtype=complex,
        )
        eigvals = np.linalg.eigvals(mat)
        lam_plus = eigvals[int(np.argmax(np.real(eigvals)))]
        lam_real = max(float(np.real(lam_plus)), 0.0)
        pref = float(np.exp(-2.0 * s_wkb) * lam_real)

        if use_pref_cache:
            self._gamma_prefactor_cache_nh[key] = pref
        gamma = eta_eff * pref
        return max(float(gamma), 0.0)


def _compute_generation_maps(
    kinetics: PSLTKinetics,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    n_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r3 = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    winner = np.zeros((len(eta_vals), len(d_vals)), dtype=int)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            _, p_n, meta = kinetics.get_probabilities(float(d), float(eta), float(PAPER_BASELINE["t_coh"]), N_max=n_max)
            p_n = np.asarray(p_n, dtype=float)
            r3[i, j] = float(np.sum(p_n[:3]))
            winner[i, j] = int(meta["winner"])
    return r3, winner


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Non-Hermitian rank-2 diagnostic scan.")
    ap.add_argument("--chain-mode", type=str, default="full_direct")
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=20.0)
    ap.add_argument("--d-num", type=int, default=21)
    ap.add_argument("--eta-min", type=float, default=0.2)
    ap.add_argument("--eta-max", type=float, default=4.0)
    ap.add_argument("--eta-num", type=int, default=41)
    ap.add_argument("--kappa-diag-scale", type=float, default=0.10)
    ap.add_argument("--kappa-split", type=float, default=0.0)
    ap.add_argument("--tag", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    if int(args.d_num) < 2 or int(args.eta_num) < 2:
        raise ValueError("--d-num and --eta-num must be >= 2.")

    base_kin = make_baseline_kinetics(
        observable_mode=str(PAPER_BASELINE["hll_observable_mode"]),
        chain_mode=str(args.chain_mode),
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
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
        runtime_direct_chi_rho_max=3.0,
        runtime_direct_chi_z_margin=6.0,
        runtime_direct_chi_n_mu=120,
        runtime_direct_chi_tol=1e-8,
        runtime_direct_chi_maxiter=30000,
        runtime_direct_chi_sigma=2.5,
        runtime_direct_superrad_zmax=80.0,
        runtime_direct_superrad_ref_d=12.0,
        runtime_direct_superrad_n_ref=2,
    )

    nh_params = PSLTParameters(**base_kin.params.__dict__)
    nh_kin = NonHermitianRank2Kinetics(
        nh_params,
        kappa_diag_scale=float(args.kappa_diag_scale),
        kappa_split=float(args.kappa_split),
    )

    d_vals_lin = np.linspace(float(args.d_min), float(args.d_max), int(args.d_num))
    ref_d_eff, ref_d_snapped = snap_ref_d_for_full_direct(str(args.chain_mode), float(PAPER_BASELINE["ref_D"]), d_vals_lin)
    if ref_d_snapped:
        print(f"[info] snapped ref_D for chain_mode={args.chain_mode}: {float(PAPER_BASELINE['ref_D']):.6g} -> {ref_d_eff:.6g}")

    d_vals, eta_vals, base_maps, _ = compute_maps(
        kinetics=base_kin,
        ref_d=float(ref_d_eff),
        ref_eta=float(PAPER_BASELINE["ref_eta"]),
        observable_mode=str(PAPER_BASELINE["hll_observable_mode"]),
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        eta_min=float(args.eta_min),
        eta_max=float(args.eta_max),
        eta_num=int(args.eta_num),
    )
    _, _, nh_maps, _ = compute_maps(
        kinetics=nh_kin,
        ref_d=float(ref_d_eff),
        ref_eta=float(PAPER_BASELINE["ref_eta"]),
        observable_mode=str(PAPER_BASELINE["hll_observable_mode"]),
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        d_num=int(args.d_num),
        eta_min=float(args.eta_min),
        eta_max=float(args.eta_max),
        eta_num=int(args.eta_num),
    )

    base_r3, base_winner = _compute_generation_maps(
        kinetics=base_kin,
        d_vals=d_vals,
        eta_vals=eta_vals,
        n_max=int(PAPER_BASELINE["hll_observable_nmax"]),
    )
    nh_r3, nh_winner = _compute_generation_maps(
        kinetics=nh_kin,
        d_vals=d_vals,
        eta_vals=eta_vals,
        n_max=int(PAPER_BASELINE["hll_observable_nmax"]),
    )

    d_mu = np.abs(nh_maps["mumu"] - base_maps["mumu"])
    d_r3 = np.abs(nh_r3 - base_r3)
    winner_mismatch = (nh_winner != base_winner)

    chi2_base = ((base_maps["mumu"] - 1.4) / 0.4) ** 2
    chi2_nh = ((nh_maps["mumu"] - 1.4) / 0.4) ** 2

    row = {
        "chain_mode": str(args.chain_mode),
        "d_num": int(args.d_num),
        "eta_num": int(args.eta_num),
        "kappa_diag_scale": float(args.kappa_diag_scale),
        "kappa_split": float(args.kappa_split),
        "frac_winner_mismatch": float(np.mean(winner_mismatch)),
        "max_abs_delta_R3": float(np.max(d_r3)),
        "p99_abs_delta_R3": float(np.percentile(d_r3, 99.0)),
        "max_abs_delta_mu_mumu": float(np.max(d_mu)),
        "p99_abs_delta_mu_mumu": float(np.percentile(d_mu, 99.0)),
        "delta_f_R3_gt_0p90": float(np.mean(nh_r3 > 0.90) - np.mean(base_r3 > 0.90)),
        "delta_f_chi2_le_4": float(np.mean(chi2_nh <= 4.0) - np.mean(chi2_base <= 4.0)),
        "delta_f_winner_gt3": float(np.mean(nh_winner > 3) - np.mean(base_winner > 3)),
    }

    tag = str(args.tag).strip()
    if not tag:
        tag = (
            f"D{int(args.d_num)}E{int(args.eta_num)}"
            f"_kdiag{float(args.kappa_diag_scale):.3g}_ksplit{float(args.kappa_split):.3g}"
        )
    tag = tag.replace(".", "p").replace("-", "m")

    out_csv = OUTDIR / f"nonhermitian_rank2_diagnostic_{tag}.csv"
    out_json = OUTDIR / f"nonhermitian_rank2_diagnostic_{tag}.json"
    paper_csv = PAPER / out_csv.name

    pd.DataFrame([row]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(row, indent=2))
    paper_csv.write_text(out_csv.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {paper_csv}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

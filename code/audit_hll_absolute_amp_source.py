#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from audit_hll_absolute_normalization_bridge import BridgeContext, bridge_mu  # noqa: E402
from scan_hll_signal_strengths import make_baseline_kinetics  # noqa: E402


@dataclass(frozen=True)
class AStarCandidate:
    name: str
    amp_abs: float
    amplitude_label: str


def summarize_candidate(
    candidate: AStarCandidate,
    ctx: BridgeContext,
    mu_obs: float,
    sigma_obs: float,
    r_e_star: float,
    r_tau_star: float,
) -> dict[str, float | str]:
    mu = bridge_mu(
        ctx=ctx,
        amp_abs=candidate.amp_abs,
        r_e_star=r_e_star,
        r_tau_star=r_tau_star,
        phi=np.ones(len(ctx.x_mu), dtype=float),
    )
    diff = mu - ctx.mu_target
    abs_diff = np.abs(diff)
    chi2 = ((mu - float(mu_obs)) / max(float(sigma_obs), 1e-30)) ** 2
    chi2_target = ((ctx.mu_target - float(mu_obs)) / max(float(sigma_obs), 1e-30)) ** 2
    accept = chi2 <= 4.0
    accept_target = chi2_target <= 4.0
    return {
        "candidate": candidate.name,
        "amplitude_label": candidate.amplitude_label,
        "amp_abs": float(candidate.amp_abs),
        "mean_abs_delta_mu": float(np.mean(abs_diff)),
        "p95_abs_delta_mu": float(np.quantile(abs_diff, 0.95)),
        "max_abs_delta_mu": float(np.max(abs_diff)),
        "rmse_delta_mu": float(np.sqrt(np.mean(diff * diff))),
        "acceptance_mismatch_fraction": float(np.mean(accept != accept_target)),
        "f_chi2_le_4_candidate": float(np.mean(accept)),
        "f_chi2_le_4_target": float(np.mean(accept_target)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit parent-side source candidates for the H->mumu absolute "
            "normalization constant A_*."
        )
    )
    parser.add_argument(
        "--map-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_summary_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_df = pd.read_csv(args.map_input)
    summary_row = pd.read_csv(args.summary_input).iloc[0]

    d_values = np.sort(map_df["D"].unique())
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode=str(summary_row["chain_mode"]),
        d_min=float(np.min(d_values)),
        d_max=float(np.max(d_values)),
        d_num=int(len(d_values)),
        uv_blend=float(summary_row["uv_blend"]),
        uv_m2_power=float(summary_row["uv_m2_power"]),
        uv_match_kappa_diag=float(summary_row["uv_match_kappa_diag"]),
        uv_match_kappa_offdiag=float(summary_row["uv_match_kappa_offdiag"]),
        uv_match_mode=str(summary_row["uv_match_mode"]),
        uv_match_input_diag_scale=float(summary_row["uv_match_input_diag_scale"]),
        uv_match_input_offdiag_scale=float(summary_row["uv_match_input_offdiag_scale"]),
        uv_rge_mu_low=float(summary_row["uv_rge_mu_low"]),
        uv_rge_gamma_diag=float(summary_row["uv_rge_gamma_diag"]),
        uv_rge_gamma_offdiag=float(summary_row["uv_rge_gamma_offdiag"]),
        uv_rge_log_clip=float(summary_row["uv_rge_log_clip"]),
    )

    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])
    t_coh = 1.0
    n_max = 20

    c_uv_ref = np.diag(kin.hll_wilson_matrix_uv_tree(ref_d, ref_eta, t_coh, n_max))
    c_match_ref, meta_match_ref = kin.hll_wilson_matrix_uv_match_with_meta(ref_d, ref_eta, t_coh, n_max)
    c_match_ref = np.diag(c_match_ref)
    c_ir_ref, meta_ir_ref = kin.hll_wilson_matrix_uv_rge_with_meta(ref_d, ref_eta, t_coh, n_max)
    c_ir_ref = np.diag(c_ir_ref)

    finite_fac_ref = float(meta_ir_ref["finite_fac_diag"])
    log_ratio_ref = float(meta_ir_ref["log_ratio"])
    gamma_diag = float(summary_row["uv_rge_gamma_diag"])
    rge_fac_ref = 1.0 + gamma_diag * log_ratio_ref / (16.0 * np.pi**2)
    chain_fac_ref = finite_fac_ref * rge_fac_ref

    width_cfg = kin._hll_match_config()
    br = np.array([width_cfg.br_ee, width_cfg.br_mumu, width_cfg.br_tautau], dtype=float)

    x_mu = map_df["C_ir_mumu"].to_numpy(dtype=float) ** 2
    r_e_over_mu = map_df["C_ir_ee"].to_numpy(dtype=float) ** 2 / np.maximum(x_mu, 1e-300)
    r_tau_over_mu = map_df["C_ir_tautau"].to_numpy(dtype=float) ** 2 / np.maximum(x_mu, 1e-300)
    ctx = BridgeContext(
        x_mu=x_mu,
        r_e_over_mu=r_e_over_mu,
        r_tau_over_mu=r_tau_over_mu,
        mu_target=map_df["mu_mumu_uv_rge"].to_numpy(dtype=float),
        br=br,
        br_sum=float(np.sum(br)),
        width_scale=float(width_cfg.width_scale),
    )

    r_e_star_ref = float((c_ir_ref[0] / max(c_ir_ref[1], 1e-300)) ** 2)
    r_tau_star_ref = float((c_ir_ref[2] / max(c_ir_ref[1], 1e-300)) ** 2)

    candidates = [
        AStarCandidate(
            name="exact_ir_reference_amplitude",
            amp_abs=float(1.0 / max(c_ir_ref[1] ** 2, 1e-300)),
            amplitude_label="A_* = |C_ir,mumu(ref)|^{-2}",
        ),
        AStarCandidate(
            name="tree_only_reference_amplitude",
            amp_abs=float(1.0 / max(c_uv_ref[1] ** 2, 1e-300)),
            amplitude_label="A_tree,* = |C_uv,mumu(ref)|^{-2}",
        ),
        AStarCandidate(
            name="tree_plus_finite_reference_amplitude",
            amp_abs=float(1.0 / max(c_match_ref[1] ** 2, 1e-300)),
            amplitude_label="A_tree,* / finite_fac_diag(ref)^2",
        ),
        AStarCandidate(
            name="tree_plus_rge_reference_amplitude",
            amp_abs=float(1.0 / max((c_uv_ref[1] * rge_fac_ref) ** 2, 1e-300)),
            amplitude_label="A_tree,* / rge_fac_diag(ref)^2",
        ),
        AStarCandidate(
            name="tree_times_full_universal_dressing",
            amp_abs=float(1.0 / max((c_uv_ref[1] * chain_fac_ref) ** 2, 1e-300)),
            amplitude_label="A_tree,* / (finite_fac_diag * rge_fac_diag)^2",
        ),
    ]

    candidate_rows = [
        summarize_candidate(
            candidate=candidate,
            ctx=ctx,
            mu_obs=float(summary_row["mu_obs"]),
            sigma_obs=float(summary_row["sigma_obs"]),
            r_e_star=r_e_star_ref,
            r_tau_star=r_tau_star_ref,
        )
        for candidate in candidates
    ]

    factorized_pred = (
        map_df["C_uv_mumu"].to_numpy(dtype=float)
        * map_df["finite_fac_diag"].to_numpy(dtype=float)
        * (1.0 + gamma_diag * map_df["log_ratio"].to_numpy(dtype=float) / (16.0 * np.pi**2))
    )
    factorization_residual = np.abs(factorized_pred - map_df["C_ir_mumu"].to_numpy(dtype=float))

    with np.errstate(divide="ignore", invalid="ignore"):
        z_diag_map = map_df["C_ir_mumu"].to_numpy(dtype=float) / np.maximum(
            map_df["C_uv_mumu"].to_numpy(dtype=float),
            1e-300,
        )

    witness = kin.hll_uv_operator_basis_witness(ref_d, ref_eta, t_coh, N_max=n_max)
    coeffs = np.asarray(witness["coefficients"], dtype=float)
    basis = np.asarray(witness["basis_matrices"], dtype=float)
    mu_contrib = coeffs * basis[:, 1, 1]
    mu_total = float(np.sum(mu_contrib))
    mu_shares = mu_contrib / max(mu_total, 1e-300)

    detail = map_df[
        [
            "D",
            "eta",
            "C_uv_mumu",
            "C_match_mumu",
            "C_ir_mumu",
            "finite_fac_diag",
            "log_ratio",
            "kappa_diag_eff",
            "action_abs_diag",
            "eymh_parented_prefactor_diag",
            "coeff_participation_access_parented",
            "tree_diag_compressibility_parented",
        ]
    ].copy()
    detail["rge_fac_diag"] = 1.0 + gamma_diag * detail["log_ratio"] / (16.0 * np.pi**2)
    detail["chain_fac_diag"] = detail["finite_fac_diag"] * detail["rge_fac_diag"]
    detail["C_ir_mumu_factorized"] = factorized_pred
    detail["abs_factorization_residual"] = factorization_residual
    detail["A_tree_local"] = 1.0 / np.maximum(detail["C_uv_mumu"], 1e-300) ** 2
    detail["A_ir_local"] = 1.0 / np.maximum(detail["C_ir_mumu"], 1e-300) ** 2
    detail["A_ir_over_A_tree"] = detail["A_ir_local"] / np.maximum(detail["A_tree_local"], 1e-300)
    detail["Z_diag_local"] = z_diag_map

    summary = pd.DataFrame(
        [
            {
                "section": "exact_factorization",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "finite_fac_diag_ref": finite_fac_ref,
                "rge_fac_diag_ref": rge_fac_ref,
                "chain_fac_diag_ref": chain_fac_ref,
                "A_star_tree": float(1.0 / max(c_uv_ref[1] ** 2, 1e-300)),
                "A_star_match": float(1.0 / max(c_match_ref[1] ** 2, 1e-300)),
                "A_star_ir": float(1.0 / max(c_ir_ref[1] ** 2, 1e-300)),
                "A_star_ir_over_tree": float((1.0 / max(c_ir_ref[1] ** 2, 1e-300)) / max(1.0 / max(c_uv_ref[1] ** 2, 1e-300), 1e-300)),
                "A_star_ir_over_match": float((1.0 / max(c_ir_ref[1] ** 2, 1e-300)) / max(1.0 / max(c_match_ref[1] ** 2, 1e-300), 1e-300)),
                "max_abs_factorization_residual": float(np.max(factorization_residual)),
                "mean_abs_factorization_residual": float(np.mean(factorization_residual)),
                "mu_layer1_share_ref": float(mu_shares[0]),
                "mu_layer2_share_ref": float(mu_shares[1]),
                "mu_layer3_share_ref": float(mu_shares[2]),
            },
            *candidate_rows,
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "source": "reference_chain_constants",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "C_uv_mumu_ref": float(c_uv_ref[1]),
                "C_match_mumu_ref": float(c_match_ref[1]),
                "C_ir_mumu_ref": float(c_ir_ref[1]),
                "finite_fac_diag_ref": finite_fac_ref,
                "rge_fac_diag_ref": rge_fac_ref,
                "chain_fac_diag_ref": chain_fac_ref,
                "A_star_tree": float(1.0 / max(c_uv_ref[1] ** 2, 1e-300)),
                "A_star_match": float(1.0 / max(c_match_ref[1] ** 2, 1e-300)),
                "A_star_ir": float(1.0 / max(c_ir_ref[1] ** 2, 1e-300)),
                "R_e_over_mu_star_ref": r_e_star_ref,
                "R_tau_over_mu_star_ref": r_tau_star_ref,
                "mu_layer1_contrib_ref": float(mu_contrib[0]),
                "mu_layer2_contrib_ref": float(mu_contrib[1]),
                "mu_layer3_contrib_ref": float(mu_contrib[2]),
                "mu_layer1_share_ref": float(mu_shares[0]),
                "mu_layer2_share_ref": float(mu_shares[1]),
                "mu_layer3_share_ref": float(mu_shares[2]),
            }
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_absolute_amp_source_summary.csv"
    detail_out = outdir / "hll_absolute_amp_source_detail.csv"
    constants_out = outdir / "hll_absolute_amp_source_constants.csv"
    meta_out = outdir / "hll_absolute_amp_source_run_meta.json"

    summary.to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "map_input": str(args.map_input),
                "summary_input": str(args.summary_input),
                "chain_mode": str(summary_row["chain_mode"]),
                "uv_match_mode": str(summary_row["uv_match_mode"]),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "n_points": int(len(map_df)),
            },
            indent=2,
        )
    )

    print(summary_out)
    print(detail_out)
    print(constants_out)
    print(meta_out)


if __name__ == "__main__":
    main()

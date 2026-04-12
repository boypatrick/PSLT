#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str((ROOT / "code").resolve()))

from scan_hll_signal_strengths import make_baseline_kinetics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the parent-side source of the tau-to-muon bridge ratio "
            "R_{tau/mu}^* in the canonical UV+LL-RG H->ll map."
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
        "--bridge-summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/"
            "hll_absolute_normalization_bridge_summary.csv"
        ),
    )
    parser.add_argument(
        "--invariant-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "reference_anchor_invariant_ratios.csv"
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
    bridge_summary = pd.read_csv(args.bridge_summary_input).set_index("candidate")
    invariant_df = pd.read_csv(args.invariant_input).set_index("scope")

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
    c_match_ref, _ = kin.hll_wilson_matrix_uv_match_with_meta(ref_d, ref_eta, t_coh, n_max)
    c_match_ref = np.diag(c_match_ref)
    c_ir_ref, _ = kin.hll_wilson_matrix_uv_rge_with_meta(ref_d, ref_eta, t_coh, n_max)
    c_ir_ref = np.diag(c_ir_ref)

    witness = kin.hll_uv_operator_basis_witness(ref_d, ref_eta, t_coh, N_max=n_max)
    coeffs = np.asarray(witness["coefficients"], dtype=float)
    basis = np.asarray(witness["basis_matrices"], dtype=float)

    mu_contrib = coeffs * basis[:, 1, 1]
    tau_contrib = coeffs * basis[:, 2, 2]
    mu_total = float(np.sum(mu_contrib))
    tau_total = float(np.sum(tau_contrib))
    mu_shares = mu_contrib / max(mu_total, 1e-300)
    tau_shares = tau_contrib / max(tau_total, 1e-300)

    c_uv_ee = map_df["C_uv_ee"].to_numpy(dtype=float)
    c_uv_mumu = map_df["C_uv_mumu"].to_numpy(dtype=float)
    c_uv_tautau = map_df["C_uv_tautau"].to_numpy(dtype=float)
    c_match_ee = map_df["C_match_ee"].to_numpy(dtype=float)
    c_match_mumu = map_df["C_match_mumu"].to_numpy(dtype=float)
    c_match_tautau = map_df["C_match_tautau"].to_numpy(dtype=float)
    c_ir_ee = map_df["C_ir_ee"].to_numpy(dtype=float)
    c_ir_mumu = map_df["C_ir_mumu"].to_numpy(dtype=float)
    c_ir_tautau = map_df["C_ir_tautau"].to_numpy(dtype=float)

    ratio_floor = 1e-250
    r_e_uv = np.full(len(map_df), np.nan, dtype=float)
    r_e_match = np.full(len(map_df), np.nan, dtype=float)
    r_e_ir = np.full(len(map_df), np.nan, dtype=float)
    r_tau_uv = np.full(len(map_df), np.nan, dtype=float)
    r_tau_match = np.full(len(map_df), np.nan, dtype=float)
    r_tau_ir = np.full(len(map_df), np.nan, dtype=float)

    mask_e_uv = (np.abs(c_uv_mumu) > ratio_floor) & (np.abs(c_uv_ee) > ratio_floor)
    mask_e_match = (np.abs(c_match_mumu) > ratio_floor) & (np.abs(c_match_ee) > ratio_floor)
    mask_e_ir = (np.abs(c_ir_mumu) > ratio_floor) & (np.abs(c_ir_ee) > ratio_floor)
    mask_tau_uv = (np.abs(c_uv_mumu) > ratio_floor) & (np.abs(c_uv_tautau) > ratio_floor)
    mask_tau_match = (np.abs(c_match_mumu) > ratio_floor) & (np.abs(c_match_tautau) > ratio_floor)
    mask_tau_ir = (np.abs(c_ir_mumu) > ratio_floor) & (np.abs(c_ir_tautau) > ratio_floor)

    r_e_uv[mask_e_uv] = (c_uv_ee[mask_e_uv] / c_uv_mumu[mask_e_uv]) ** 2
    r_e_match[mask_e_match] = (c_match_ee[mask_e_match] / c_match_mumu[mask_e_match]) ** 2
    r_e_ir[mask_e_ir] = (c_ir_ee[mask_e_ir] / c_ir_mumu[mask_e_ir]) ** 2
    r_tau_uv[mask_tau_uv] = (c_uv_tautau[mask_tau_uv] / c_uv_mumu[mask_tau_uv]) ** 2
    r_tau_match[mask_tau_match] = (c_match_tautau[mask_tau_match] / c_match_mumu[mask_tau_match]) ** 2
    r_tau_ir[mask_tau_ir] = (c_ir_tautau[mask_tau_ir] / c_ir_mumu[mask_tau_ir]) ** 2

    active_e_uv_ir = np.isfinite(r_e_uv) & np.isfinite(r_e_ir)
    active_tau_uv_ir = np.isfinite(r_tau_uv) & np.isfinite(r_tau_ir)
    active_tau_match_ir = np.isfinite(r_tau_match) & np.isfinite(r_tau_ir)

    detail = map_df[
        [
            "D",
            "eta",
            "C_uv_ee",
            "C_uv_mumu",
            "C_uv_tautau",
            "C_match_ee",
            "C_match_mumu",
            "C_match_tautau",
            "C_ir_ee",
            "C_ir_mumu",
            "C_ir_tautau",
        ]
    ].copy()
    detail["R_e_over_mu_uv"] = r_e_uv
    detail["R_e_over_mu_match"] = r_e_match
    detail["R_e_over_mu_ir"] = r_e_ir
    detail["R_tau_over_mu_uv"] = r_tau_uv
    detail["R_tau_over_mu_match"] = r_tau_match
    detail["R_tau_over_mu_ir"] = r_tau_ir
    detail["abs_delta_R_e_uv_to_ir"] = np.where(active_e_uv_ir, np.abs(r_e_uv - r_e_ir), np.nan)
    detail["abs_delta_R_tau_uv_to_ir"] = np.where(active_tau_uv_ir, np.abs(r_tau_uv - r_tau_ir), np.nan)
    detail["abs_delta_R_tau_match_to_ir"] = np.where(active_tau_match_ir, np.abs(r_tau_match - r_tau_ir), np.nan)

    r_tau_star_ref = float((c_ir_ref[2] / max(c_ir_ref[1], 1e-300)) ** 2)
    r_tau_tree_ref = float((c_uv_ref[2] / max(c_uv_ref[1], 1e-300)) ** 2)
    r_tau_match_ref = float((c_match_ref[2] / max(c_match_ref[1], 1e-300)) ** 2)

    summary = pd.DataFrame(
        [
            {
                "section": "chain_invariance",
                "active_rows_R_e_uv_to_ir": int(np.sum(active_e_uv_ir)),
                "active_rows_R_tau_uv_to_ir": int(np.sum(active_tau_uv_ir)),
                "active_rows_R_tau_match_to_ir": int(np.sum(active_tau_match_ir)),
                "max_abs_delta_R_e_uv_to_ir": float(np.nanmax(detail["abs_delta_R_e_uv_to_ir"])),
                "mean_abs_delta_R_e_uv_to_ir": float(np.nanmean(detail["abs_delta_R_e_uv_to_ir"])),
                "max_abs_delta_R_tau_uv_to_ir": float(np.nanmax(detail["abs_delta_R_tau_uv_to_ir"])),
                "mean_abs_delta_R_tau_uv_to_ir": float(np.nanmean(detail["abs_delta_R_tau_uv_to_ir"])),
                "max_abs_delta_R_tau_match_to_ir": float(np.nanmax(detail["abs_delta_R_tau_match_to_ir"])),
                "mean_abs_delta_R_tau_match_to_ir": float(np.nanmean(detail["abs_delta_R_tau_match_to_ir"])),
            },
            {
                "section": "reference_ratio_constants",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "R_tau_over_mu_uv_ref": r_tau_tree_ref,
                "R_tau_over_mu_match_ref": r_tau_match_ref,
                "R_tau_over_mu_ir_ref": r_tau_star_ref,
                "R_tau_over_mu_accept_p50": float(invariant_df.loc["chi2_accept", "R_tau_over_mu_p50"]),
                "R_tau_over_mu_fit_global": float(bridge_summary.loc["three_parameter_global_invariant_fit", "r_tau_star"]),
                "R_tau_over_mu_fit_fixed_re": float(bridge_summary.loc["two_parameter_bridge_fixed_re_median", "r_tau_star"]),
                "rel_shift_fit_global_vs_ref": float(
                    bridge_summary.loc["three_parameter_global_invariant_fit", "r_tau_star"] / max(r_tau_star_ref, 1e-300) - 1.0
                ),
                "rel_shift_fit_fixed_re_vs_ref": float(
                    bridge_summary.loc["two_parameter_bridge_fixed_re_median", "r_tau_star"] / max(r_tau_star_ref, 1e-300) - 1.0
                ),
            },
            {
                "section": "reference_layer_support",
                "mu_layer1_share_ref": float(mu_shares[0]),
                "mu_layer2_share_ref": float(mu_shares[1]),
                "mu_layer3_share_ref": float(mu_shares[2]),
                "tau_layer1_share_ref": float(tau_shares[0]),
                "tau_layer2_share_ref": float(tau_shares[1]),
                "tau_layer3_share_ref": float(tau_shares[2]),
                "mu_layer2_contrib_ref": float(mu_contrib[1]),
                "tau_layer3_contrib_ref": float(tau_contrib[2]),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "source": "reference_tau_ratio_chain",
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "C_uv_mumu_ref": float(c_uv_ref[1]),
                "C_uv_tautau_ref": float(c_uv_ref[2]),
                "C_ir_mumu_ref": float(c_ir_ref[1]),
                "C_ir_tautau_ref": float(c_ir_ref[2]),
                "R_tau_over_mu_uv_ref": r_tau_tree_ref,
                "R_tau_over_mu_ir_ref": r_tau_star_ref,
                "mu_layer1_share_ref": float(mu_shares[0]),
                "mu_layer2_share_ref": float(mu_shares[1]),
                "mu_layer3_share_ref": float(mu_shares[2]),
                "tau_layer1_share_ref": float(tau_shares[0]),
                "tau_layer2_share_ref": float(tau_shares[1]),
                "tau_layer3_share_ref": float(tau_shares[2]),
            }
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_tau_ratio_source_summary.csv"
    detail_out = outdir / "hll_tau_ratio_source_detail.csv"
    constants_out = outdir / "hll_tau_ratio_source_constants.csv"
    meta_out = outdir / "hll_tau_ratio_source_run_meta.json"

    summary.to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "map_input": str(args.map_input),
                "summary_input": str(args.summary_input),
                "bridge_summary_input": str(args.bridge_summary_input),
                "invariant_input": str(args.invariant_input),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "chain_mode": str(summary_row["chain_mode"]),
                "uv_match_mode": str(summary_row["uv_match_mode"]),
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

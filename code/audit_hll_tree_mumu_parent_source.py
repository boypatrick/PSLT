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
            "Audit parent-side source candidates for the canonical "
            "C_{mumu,*}^{tree} amplitude."
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

    rows: list[dict[str, float]] = []
    for _, row in map_df.iterrows():
        D = float(row["D"])
        eta = float(row["eta"])
        witness = kin.hll_uv_operator_basis_witness(D, eta, 1.0, N_max=20)
        coeff = np.asarray(witness["coefficients"], dtype=float)
        basis = np.asarray(witness["basis_matrices"], dtype=float)
        y_raw = kin._hll_yraw_vector(D)
        p_kin = kin._hll_pkin_vector(D, eta, 1.0, N_max=20)
        m2 = kin._hll_m2_vector(D)

        mu_contrib = coeff * basis[:, 1, 1]
        c_tree_mu = float(row["C_uv_mumu"])
        exact_single_layer = float(y_raw[1] * p_kin[1] / max(m2[1], 1e-300))

        rows.append(
            {
                "D": D,
                "eta": eta,
                "C_tree_mumu": c_tree_mu,
                "coeff_1": float(coeff[0]),
                "coeff_2": float(coeff[1]),
                "coeff_3": float(coeff[2]),
                "g_mu1_sq": float(basis[0, 1, 1]),
                "g_mu2_sq": float(basis[1, 1, 1]),
                "g_mu3_sq": float(basis[2, 1, 1]),
                "mu_contrib_1": float(mu_contrib[0]),
                "mu_contrib_2": float(mu_contrib[1]),
                "mu_contrib_3": float(mu_contrib[2]),
                "mu_share_1": float(mu_contrib[0] / max(c_tree_mu, 1e-300)),
                "mu_share_2": float(mu_contrib[1] / max(c_tree_mu, 1e-300)),
                "mu_share_3": float(mu_contrib[2] / max(c_tree_mu, 1e-300)),
                "y2_raw": float(y_raw[1]),
                "p2_kin": float(p_kin[1]),
                "m2_2": float(m2[1]),
                "coeff2_kin_over_m2": float(p_kin[1] / max(m2[1], 1e-300)),
                "C_tree_mumu_single_layer_pred": exact_single_layer,
                "abs_single_layer_residual": float(abs(c_tree_mu - exact_single_layer)),
            }
        )

    detail = pd.DataFrame(rows)
    active = detail[detail["C_tree_mumu"] > 1e-250].copy()

    ly = np.log(np.maximum(active["y2_raw"].to_numpy(dtype=float), 1e-300))
    lc = np.log(np.maximum(active["coeff2_kin_over_m2"].to_numpy(dtype=float), 1e-300))
    lt = np.log(np.maximum(active["C_tree_mumu"].to_numpy(dtype=float), 1e-300))

    ref_d = float(summary_row["ref_D"])
    ref_eta = float(summary_row["ref_eta"])
    ref_idx = (np.abs(detail["D"] - ref_d) + np.abs(detail["eta"] - ref_eta)).idxmin()
    ref_row = detail.loc[ref_idx]

    c_ir_ref, meta_ir_ref = kin.hll_wilson_matrix_uv_rge_with_meta(
        float(ref_row["D"]), float(ref_row["eta"]), 1.0, N_max=20
    )
    c_ir_ref = np.diag(c_ir_ref)
    finite_fac_ref = float(meta_ir_ref["finite_fac_diag"])
    rge_fac_ref = 1.0 + float(summary_row["uv_rge_gamma_diag"]) * float(meta_ir_ref["log_ratio"]) / (16.0 * np.pi**2)
    z_diag_ref = finite_fac_ref * rge_fac_ref

    summary = pd.DataFrame(
        [
            {
                "section": "single_layer_exact_reduction",
                "n_active_rows": int(len(active)),
                "max_abs_single_layer_residual": float(active["abs_single_layer_residual"].max()),
                "mean_abs_single_layer_residual": float(active["abs_single_layer_residual"].mean()),
                "p95_abs_single_layer_residual": float(active["abs_single_layer_residual"].quantile(0.95)),
                "max_abs_mu_share1": float(np.max(np.abs(active["mu_share_1"]))),
                "min_mu_share2": float(active["mu_share_2"].min()),
                "max_abs_mu_share3": float(np.max(np.abs(active["mu_share_3"]))),
            },
            {
                "section": "variation_budget",
                "std_log_y2_raw": float(np.std(ly)),
                "std_log_coeff2_kin_over_m2": float(np.std(lc)),
                "std_log_C_tree_mumu": float(np.std(lt)),
                "corr_log_y2_raw_vs_log_C_tree": float(np.corrcoef(ly, lt)[0, 1]),
                "corr_log_coeff2_vs_log_C_tree": float(np.corrcoef(lc, lt)[0, 1]),
            },
            {
                "section": "reference_exact_constants",
                "ref_D": float(ref_row["D"]),
                "ref_eta": float(ref_row["eta"]),
                "uv_blend": float(summary_row["uv_blend"]),
                "y2_raw_ref": float(ref_row["y2_raw"]),
                "p2_kin_ref": float(ref_row["p2_kin"]),
                "m2_2_ref": float(ref_row["m2_2"]),
                "coeff2_kin_over_m2_ref": float(ref_row["coeff2_kin_over_m2"]),
                "C_tree_mumu_ref": float(ref_row["C_tree_mumu"]),
                "A_tree_ref": float(1.0 / max(ref_row["C_tree_mumu"] ** 2, 1e-300)),
                "finite_fac_diag_ref": float(finite_fac_ref),
                "rge_fac_diag_ref": float(rge_fac_ref),
                "z_diag_ref": float(z_diag_ref),
                "A_star_ref": float(1.0 / max(c_ir_ref[1] ** 2, 1e-300)),
            },
        ]
    )

    constants = pd.DataFrame(
        [
            {
                "source": "reference_current_chain_source",
                "ref_D": float(ref_row["D"]),
                "ref_eta": float(ref_row["eta"]),
                "uv_blend": float(summary_row["uv_blend"]),
                "y2_raw_ref": float(ref_row["y2_raw"]),
                "p2_kin_ref": float(ref_row["p2_kin"]),
                "m2_2_ref": float(ref_row["m2_2"]),
                "coeff2_kin_over_m2_ref": float(ref_row["coeff2_kin_over_m2"]),
                "C_tree_mumu_ref": float(ref_row["C_tree_mumu"]),
                "A_tree_ref": float(1.0 / max(ref_row["C_tree_mumu"] ** 2, 1e-300)),
                "finite_fac_diag_ref": float(finite_fac_ref),
                "rge_fac_diag_ref": float(rge_fac_ref),
                "z_diag_ref": float(z_diag_ref),
                "A_star_ref": float(1.0 / max(c_ir_ref[1] ** 2, 1e-300)),
            }
        ]
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_out = outdir / "hll_tree_mumu_parent_source_summary.csv"
    detail_out = outdir / "hll_tree_mumu_parent_source_detail.csv"
    constants_out = outdir / "hll_tree_mumu_parent_source_constants.csv"
    meta_out = outdir / "hll_tree_mumu_parent_source_run_meta.json"

    summary.to_csv(summary_out, index=False)
    detail.to_csv(detail_out, index=False)
    constants.to_csv(constants_out, index=False)
    meta_out.write_text(
        json.dumps(
            {
                "map_input": str(args.map_input),
                "summary_input": str(args.summary_input),
                "ref_D": ref_d,
                "ref_eta": ref_eta,
                "chain_mode": str(summary_row["chain_mode"]),
                "uv_match_mode": str(summary_row["uv_match_mode"]),
                "uv_blend": float(summary_row["uv_blend"]),
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

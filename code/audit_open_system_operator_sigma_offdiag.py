#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def lorentz_factor(r: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + r * r)


def q(v: np.ndarray | pd.Series, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


def evaluate_gap_floor(g_floor: float, r_on_shell: np.ndarray, delta_q_ref: float) -> dict[str, float]:
    sep = lorentz_factor(r_on_shell / max(g_floor, 1e-30)) - lorentz_factor(r_on_shell)
    sep_min = float(np.min(sep))
    eps_factor_max = float(np.max((1.0 + r_on_shell * r_on_shell) / (g_floor * g_floor + r_on_shell * r_on_shell)))
    if delta_q_ref < sep_min:
        tail_upper = float(delta_q_ref / sep_min)
        eps0_upper = float(delta_q_ref / (sep_min - delta_q_ref))
        eps_delta_upper = float(eps0_upper * eps_factor_max)
    else:
        tail_upper = np.nan
        eps0_upper = np.nan
        eps_delta_upper = np.nan
    return {
        "g_floor": float(g_floor),
        "separator_min": sep_min,
        "eps_delta_factor_max": eps_factor_max,
        "tail_mass_upper": tail_upper,
        "eps0_upper": eps0_upper,
        "eps_delta_upper": eps_delta_upper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit conditional sigma/||A|| upper bounds from parent-side off-diagonal witnesses. "
            "The minimal rank-one contraction model identifies a_a=lambda^2 xi_a and sigma_a=-log(1-a_a)."
        )
    )
    parser.add_argument(
        "--parented-map-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--parent-kernel-map-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_action_loop_eymh_parent_kernel_statement_map_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--operator-summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_summary.csv"
        ),
    )
    parser.add_argument(
        "--operator-tolerance-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_tolerance_summary.csv"
        ),
    )
    parser.add_argument(
        "--open-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_micro_bridge_map.csv"
        ),
    )
    parser.add_argument("--delta-q-ref", type=float, default=1e-3)
    parser.add_argument(
        "--proxy-summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_sigma_offdiag_proxy_summary.csv"
        ),
    )
    parser.add_argument(
        "--tolerance-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_sigma_offdiag_tolerance_summary.csv"
        ),
    )
    args = parser.parse_args()

    parented = pd.read_csv(args.parented_map_input)
    kernel = pd.read_csv(args.parent_kernel_map_input)
    op_summary = pd.read_csv(args.operator_summary_input).iloc[0]
    op_tol = pd.read_csv(args.operator_tolerance_input)
    open_df = pd.read_csv(args.open_input)

    df = parented.merge(
        kernel[["D", "eta", "parent_kernel_cross_xi", "parent_kernel_response_action"]],
        on=["D", "eta"],
        how="inner",
    )
    df = df[df["parent_kernel_response_action"] > 1e-12].copy()

    overlap = df.merge(open_df[["D", "omega1", "delta_micro"]], on="D", how="inner")
    r_on_shell = (overlap["delta_micro"] / np.maximum(overlap["omega1"], 1e-30)).to_numpy(dtype=float)

    diag = np.maximum(
        np.asarray(df["coeff_participation_access_parented"], dtype=float)
        * np.asarray(df["tree_diag_compressibility_parented"], dtype=float),
        1e-30,
    )
    hk_abs_over_diag = np.maximum(np.asarray(df["hk_abs_offdiag"], dtype=float), 0.0) / diag
    hk_curv_over_diag = np.maximum(np.asarray(df["hk_curv_screen_offdiag"], dtype=float), 0.0) / diag
    action_abs_over_diag = np.maximum(np.asarray(df["action_abs_offdiag"], dtype=float), 0.0) / diag
    xi_target = np.asarray(df["parent_kernel_cross_xi"], dtype=float)

    proxy_defs = {
        "xi_exact": xi_target,
        "runnerup_blend_abs_action": 0.127472 * hk_abs_over_diag + 0.435064 * action_abs_over_diag,
        "best_blend_curv_action": 0.104934 * hk_curv_over_diag + 0.477834 * action_abs_over_diag,
        "lowd_hk_abs_single": 0.231819 * hk_abs_over_diag,
        "highd_action_abs_single": 0.857050 * action_abs_over_diag,
    }

    overlap_mask = df["D"].isin(open_df["D"].unique()).to_numpy()

    proxy_rows = []
    proxy_env = {}
    for name, proxy in proxy_defs.items():
        resid = np.abs(proxy - xi_target)
        overlap_resid = resid[overlap_mask]
        overlap_proxy = proxy[overlap_mask]
        exact_overlap = xi_target[overlap_mask]
        if name == "xi_exact":
            env_p95 = float(exact_overlap.max())
            env_max = float(exact_overlap.max())
        else:
            env_p95 = float(np.max(overlap_proxy + q(overlap_resid, 0.95)))
            env_max = float(np.max(overlap_proxy + np.max(overlap_resid)))
        proxy_env[name] = {"p95": env_p95, "max": env_max}
        proxy_rows.append(
            {
                "proxy_name": name,
                "overlap_xi_target_max": float(np.max(exact_overlap)),
                "overlap_proxy_max": float(np.max(overlap_proxy)),
                "overlap_p95_abs_residual": q(overlap_resid, 0.95),
                "overlap_max_abs_residual": float(np.max(overlap_resid)),
                "active_p95_abs_residual": q(resid, 0.95),
                "active_max_abs_residual": float(np.max(resid)),
                "xi_upper_p95_envelope": env_p95,
                "xi_upper_max_envelope": env_max,
            }
        )
    proxy_summary = pd.DataFrame(proxy_rows).sort_values(["xi_upper_p95_envelope", "overlap_p95_abs_residual"])

    c_loc = float(op_summary["coercivity_ratio_min"])
    c4 = float(op_summary["lambda_quartic_coeff"])

    tol_rows = []
    for row in op_tol.itertuples(index=False):
        J = float(row.objective_tol)
        lam_upper = float((J / max(c_loc * c4, 1e-30)) ** 0.25)
        family_g_floor = float(row.analytic_g_floor_min)
        for name in ["xi_exact", "runnerup_blend_abs_action", "best_blend_curv_action"]:
            for envelope_name, xi_upper in proxy_env[name].items():
                if name == "xi_exact" and envelope_name == "max":
                    use_label = "exact"
                elif name == "xi_exact" and envelope_name == "p95":
                    continue
                else:
                    use_label = envelope_name
                a_upper = float(min(lam_upper * lam_upper * xi_upper, 1.0 - 1e-12))
                sigma_upper = float(-np.log(max(1.0 - a_upper, 1e-30)))
                norm_upper = a_upper
                g_lifted = float(np.exp(-0.5 * sigma_upper) * family_g_floor)
                evals = evaluate_gap_floor(g_lifted, r_on_shell, args.delta_q_ref)
                tol_rows.append(
                    {
                        "objective_tol": J,
                        "lambda_upper": lam_upper,
                        "family_g_floor": family_g_floor,
                        "proxy_name": name,
                        "envelope": use_label,
                        "xi_upper": float(xi_upper),
                        "a_upper_rank1": norm_upper,
                        "sigma_upper_rank1": sigma_upper,
                        "lifted_g_floor": g_lifted,
                        "separator_min": evals["separator_min"],
                        "eps_delta_factor_max": evals["eps_delta_factor_max"],
                        "tail_mass_upper": evals["tail_mass_upper"],
                        "eps0_upper": evals["eps0_upper"],
                        "eps_delta_upper": evals["eps_delta_upper"],
                        "delta_q_ref": float(args.delta_q_ref),
                    }
                )

    tol_summary = pd.DataFrame(tol_rows).sort_values(["objective_tol", "sigma_upper_rank1", "proxy_name"])

    for out in [args.proxy_summary_output, args.tolerance_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    proxy_summary.to_csv(args.proxy_summary_output, index=False)
    tol_summary.to_csv(args.tolerance_output, index=False)
    print(args.proxy_summary_output)
    print(args.tolerance_output)


if __name__ == "__main__":
    main()

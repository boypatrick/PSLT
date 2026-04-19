#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def closest_row(df: pd.DataFrame, column: str, target: float) -> pd.Series:
    idx = (df[column] - target).abs().idxmin()
    return df.loc[idx]


def lorentz_factor(r: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + r * r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Consolidate the audited open-system parent-bath family rigidity scans into "
            "an exact-lift source statement for the remaining g_a / Schur-lift theorem blocker."
        )
    )
    ap.add_argument(
        "--family-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_family_summary.csv"),
    )
    ap.add_argument(
        "--generator-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_generator_affinity_summary.csv"),
    )
    ap.add_argument(
        "--log-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_log_coordinate_summary.csv"),
    )
    ap.add_argument(
        "--normal-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_normal_coordinate_summary.csv"),
    )
    ap.add_argument(
        "--statement-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_statement_summary.csv"),
    )
    ap.add_argument(
        "--cocycle-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_cocycle_summary.csv"),
    )
    ap.add_argument(
        "--potential-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_potential_summary.csv"),
    )
    ap.add_argument(
        "--neighborhood-summary",
        default=str(OUTDIR / "chi_open_system_operator_detdeficit_neighborhood_bound_summary.csv"),
    )
    ap.add_argument(
        "--deltaq-summary",
        default=str(OUTDIR / "chi_open_system_deltaq_source_summary.csv"),
    )
    ap.add_argument(
        "--open-input",
        default=str(OUTDIR / "chi_open_system_micro_bridge_map.csv"),
    )
    ap.add_argument("--objective-tol", type=float, default=1e-6)
    ap.add_argument(
        "--detail-output",
        default=str(OUTDIR / "chi_open_system_exact_lift_source_detail.csv"),
    )
    ap.add_argument(
        "--summary-output",
        default=str(OUTDIR / "chi_open_system_exact_lift_source_summary.csv"),
    )
    args = ap.parse_args()

    family = pd.read_csv(args.family_summary).iloc[0]
    generator = pd.read_csv(args.generator_summary).iloc[0]
    logc = pd.read_csv(args.log_summary).iloc[0]
    normal = pd.read_csv(args.normal_summary).iloc[0]
    statement = pd.read_csv(args.statement_summary).iloc[0]
    cocycle = pd.read_csv(args.cocycle_summary).iloc[0]
    potential = pd.read_csv(args.potential_summary).iloc[0]
    neigh_df = pd.read_csv(args.neighborhood_summary)
    neigh = closest_row(neigh_df, "objective_tol", args.objective_tol)
    deltaq = pd.read_csv(args.deltaq_summary).iloc[0]
    open_df = pd.read_csv(args.open_input)

    audited = [
        {
            "audit": "family",
            "canonical_objective": float(family["canonical_objective"]),
            "selection_gap": float(family["selection_gap_objective"]),
            "best_parameters": f"m={family['best_m']}, u={family['best_u']}, v={family['best_v']}",
            "runner_up_parameters": f"m={family['runner_up_m']}, u={family['runner_up_u']}, v={family['runner_up_v']}",
        },
        {
            "audit": "generator_affinity",
            "canonical_objective": float(generator["canonical_objective"]),
            "selection_gap": float(generator["selection_gap_objective"]),
            "best_parameters": f"q_ss={generator['best_q_ss']}, q_bb={generator['best_q_bb']}, q_sb={generator['best_q_sb']}",
            "runner_up_parameters": f"q_ss={generator['runner_up_q_ss']}, q_bb={generator['runner_up_q_bb']}, q_sb={generator['runner_up_q_sb']}",
        },
        {
            "audit": "log_coordinate",
            "canonical_objective": float(logc["canonical_objective"]),
            "selection_gap": float(logc["selection_gap_objective"]),
            "best_parameters": f"p_sys={logc['best_p_sys']}, p_spec={logc['best_p_spec']}",
            "runner_up_parameters": f"p_sys={logc['runner_up_p_sys']}, p_spec={logc['runner_up_p_spec']}",
        },
        {
            "audit": "normal_coordinate",
            "canonical_objective": float(normal["canonical_objective"]),
            "selection_gap": float(normal["selection_gap_objective"]),
            "best_parameters": f"zeta_sys={normal['best_zeta_sys']}, zeta_spec={normal['best_zeta_spec']}",
            "runner_up_parameters": f"zeta_sys={normal['runner_up_zeta_sys']}, zeta_spec={normal['runner_up_zeta_spec']}",
        },
    ]
    detail = pd.DataFrame(audited)
    detail["gap_over_objective_tol"] = detail["selection_gap"] / max(float(args.objective_tol), 1e-30)
    detail["gap_over_canonical"] = detail["selection_gap"] / np.maximum(detail["canonical_objective"], 1e-30)

    min_gap = float(detail["selection_gap"].min())
    max_canonical = float(detail["canonical_objective"].max())

    r_on_shell = (open_df["delta_micro"] / np.maximum(open_df["omega1"], 1e-30)).to_numpy(dtype=float)
    g_lift = float(neigh["lifted_g_floor"])
    sep = lorentz_factor(r_on_shell / max(g_lift, 1e-30)) - lorentz_factor(r_on_shell)
    sep_min = float(np.min(sep))
    eps_factor_max = float(np.max((1.0 + r_on_shell * r_on_shell) / (g_lift * g_lift + r_on_shell * r_on_shell)))
    actual_delta_q = float(deltaq["max_abs_delta_q_parent_log_vs_gap"])
    tail_upper = actual_delta_q / max(sep_min, 1e-30)
    eps0_upper = actual_delta_q / max(sep_min - actual_delta_q, 1e-30)
    eps_delta_upper = eps0_upper * eps_factor_max

    summary = pd.DataFrame(
        [
            {
                "objective_tol": float(args.objective_tol),
                "max_canonical_objective_across_families": max_canonical,
                "min_selection_gap_across_families": min_gap,
                "selection_gap_over_objective_tol_min": min_gap / max(float(args.objective_tol), 1e-30),
                "selection_gap_over_max_canonical": min_gap / max(max_canonical, 1e-30),
                "statement_max_parent_log_rate_anisotropy_residual": float(statement["max_abs_parent_log_rate_anisotropy_residual_identifiable"]),
                "cocycle_max_triangle_defect_mix": float(cocycle["max_abs_triangle_defect_mix"]),
                "potential_max_chain_recovery_residual_mix": float(potential["max_abs_chain_recovery_residual_mix"]),
                "u_neighborhood_upper": float(neigh["u_neighborhood_upper"]),
                "trA_upper": float(neigh["trA_upper"]),
                "opA_upper": float(neigh["opA_upper"]),
                "heff_loewner_factor": float(neigh["heff_loewner_factor"]),
                "cor1_logdet_gap_upper": float(neigh["cor1_logdet_gap_upper"]),
                "lifted_g_floor": g_lift,
                "separator_min_from_lifted_g": sep_min,
                "actual_delta_q_max": actual_delta_q,
                "tail_mass_upper_from_actual_delta_q": tail_upper,
                "eps0_upper_from_actual_delta_q": eps0_upper,
                "eps_delta_upper_from_actual_delta_q": eps_delta_upper,
            }
        ]
    )

    for out in [args.detail_output, args.summary_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(args.detail_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()

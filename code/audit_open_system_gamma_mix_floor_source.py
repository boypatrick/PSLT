#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit an independent gamma_mix floor source from the exact "
            "observable-chain identity gamma_mix = gamma_ref * chi_eff / (2 Cmax)."
        )
    )
    ap.add_argument(
        "--bridge-map",
        default=str(OUTDIR / "chi_open_system_micro_bridge_map.csv"),
    )
    ap.add_argument(
        "--mix-summary",
        default=str(OUTDIR / "chi_open_system_mix_factorization_source_summary.csv"),
    )
    ap.add_argument(
        "--witness-transfer-summary",
        default=str(OUTDIR / "chi_open_system_operator_witness_transfer_summary.csv"),
    )
    ap.add_argument(
        "--detail-output",
        default=str(OUTDIR / "chi_open_system_gamma_mix_floor_source_detail.csv"),
    )
    ap.add_argument(
        "--summary-output",
        default=str(OUTDIR / "chi_open_system_gamma_mix_floor_source_summary.csv"),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.bridge_map).copy()
    mix = pd.read_csv(args.mix_summary).iloc[0]
    wit = pd.read_csv(args.witness_transfer_summary).iloc[0]

    gamma_mix = df["gamma_mix_micro"].astype(float)
    gamma_ref = df["gamma_ref"].astype(float)
    chi_eff = df["chi_eff_micro"].astype(float)
    cmax = df["Cmax_micro"].astype(float)
    kappa = df["kappa_env"].astype(float)

    gamma_formula = gamma_ref * chi_eff / (2.0 * cmax.clip(lower=1e-300))
    gamma_formula_resid = (gamma_mix - gamma_formula).abs()

    min_gamma_ref = float(gamma_ref.min())
    min_chi_eff = float(chi_eff.min())
    max_cmax = float(cmax.max())
    sep_gamma_floor = min_gamma_ref * min_chi_eff / (2.0 * max_cmax)
    actual_gamma_floor = float(gamma_mix.min())
    kappa_max = float(kappa.max())
    sep_kernel_floor = sep_gamma_floor / max(kappa_max, 1e-300)
    actual_kernel_floor = actual_gamma_floor / max(kappa_max, 1e-300)

    eps_mix = float(mix["eps_mix_factor_residual"])
    witness_bound_from_sep_gamma = eps_mix / max(sep_kernel_floor, 1e-300)

    p95_half_radius = 0.5 * float(wit["p95_safe_radius"])
    max_half_radius = 0.5 * float(wit["max_safe_radius"])

    detail = pd.DataFrame(
        {
            "D": df["D"].astype(float),
            "gamma_mix_micro": gamma_mix,
            "gamma_mix_formula_recon": gamma_formula,
            "gamma_mix_formula_residual_abs": gamma_formula_resid,
            "gamma_ref": gamma_ref,
            "chi_eff_micro": chi_eff,
            "Cmax_micro": cmax,
            "kappa_env": kappa,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "max_abs_gamma_formula_residual": float(gamma_formula_resid.max()),
                "actual_gamma_mix_floor": actual_gamma_floor,
                "actual_gamma_mix_floor_D": float(
                    detail.loc[gamma_mix.idxmin(), "D"]
                ),
                "min_gamma_ref": min_gamma_ref,
                "min_gamma_ref_D": float(detail.loc[gamma_ref.idxmin(), "D"]),
                "min_chi_eff": min_chi_eff,
                "min_chi_eff_D": float(detail.loc[chi_eff.idxmin(), "D"]),
                "max_Cmax": max_cmax,
                "max_Cmax_D": float(detail.loc[cmax.idxmax(), "D"]),
                "separated_gamma_mix_floor": sep_gamma_floor,
                "actual_over_separated_gamma_floor": actual_gamma_floor
                / max(sep_gamma_floor, 1e-300),
                "kappa_env_max": kappa_max,
                "induced_kernel_floor_from_separated_gamma": sep_kernel_floor,
                "actual_kernel_floor": actual_kernel_floor,
                "actual_over_separated_kernel_floor": actual_kernel_floor
                / max(sep_kernel_floor, 1e-300),
                "eps_mix_factor_residual": eps_mix,
                "mix_witness_bound_from_separated_gamma_floor": witness_bound_from_sep_gamma,
                "p95_half_radius": p95_half_radius,
                "max_half_radius": max_half_radius,
                "p95_half_over_sep_gamma_bound": p95_half_radius
                / max(witness_bound_from_sep_gamma, 1e-300),
                "max_half_over_sep_gamma_bound": max_half_radius
                / max(witness_bound_from_sep_gamma, 1e-300),
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

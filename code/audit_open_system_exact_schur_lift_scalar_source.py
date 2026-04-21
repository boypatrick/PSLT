#!/usr/bin/env python3
"""Synthesize the single-scalar sufficient condition for the exact Schur lift."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "chi_open_system"


def read_summary(name: str) -> dict[str, str]:
    path = OUT_DIR / name
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def to_float(value: str) -> float:
    return float(value)


def main() -> None:
    mix = read_summary("chi_open_system_mix_factorization_source_summary.csv")
    gamma = read_summary("chi_open_system_gamma_mix_floor_source_summary.csv")
    gx = read_summary("chi_open_system_gx_floor_source_summary.csv")
    radius = read_summary("chi_open_system_exact_operator_radius_bound_summary.csv")
    lift = read_summary("chi_open_system_exact_lift_source_summary.csv")

    eps_mix = to_float(mix["eps_mix_factor_residual"])
    p95_half_radius = to_float(mix["p95_half_radius"])
    max_half_radius = to_float(mix["max_half_radius"])
    required_kernel_floor_p95 = eps_mix / p95_half_radius
    required_kernel_floor_max = eps_mix / max_half_radius

    direct_kernel_floor = to_float(mix["min_rate_kernel_mix_direct"])
    gamma_kernel_floor = to_float(mix["min_rate_kernel_mix_from_gamma_floor"])
    separated_gamma_kernel_floor = to_float(gamma["induced_kernel_floor_from_separated_gamma"])
    actual_kernel_floor = to_float(gamma["actual_kernel_floor"])

    kappa_env_max = to_float(gamma["kappa_env_max"])
    required_gamma_floor_p95 = kappa_env_max * required_kernel_floor_p95
    required_gamma_floor_max = kappa_env_max * required_kernel_floor_max

    nonmix_env = max(
        to_float(radius["block11_env"]),
        to_float(radius["trace_env"]),
        to_float(radius["det_env"]),
        to_float(radius["anisotropy_env"]),
        to_float(radius["cocycle_mix_env"]),
        to_float(radius["potential_mix_env"]),
    )

    rows = [
        {
            "source": "required_half_radius_threshold",
            "kernel_floor": required_kernel_floor_p95,
            "gamma_floor": required_gamma_floor_p95,
            "ratio_to_required_p95": 1.0,
            "ratio_to_required_max": required_kernel_floor_p95 / required_kernel_floor_max,
            "argmin_or_reference_D": "",
            "role": "sufficient_condition",
        },
        {
            "source": "direct_exact_mix_kernel_floor",
            "kernel_floor": direct_kernel_floor,
            "gamma_floor": "",
            "ratio_to_required_p95": direct_kernel_floor / required_kernel_floor_p95,
            "ratio_to_required_max": direct_kernel_floor / required_kernel_floor_max,
            "argmin_or_reference_D": gamma["actual_gamma_mix_floor_D"],
            "role": "exact_export",
        },
        {
            "source": "gamma_induced_kernel_floor",
            "kernel_floor": gamma_kernel_floor,
            "gamma_floor": gamma["actual_gamma_mix_floor"],
            "ratio_to_required_p95": gamma_kernel_floor / required_kernel_floor_p95,
            "ratio_to_required_max": gamma_kernel_floor / required_kernel_floor_max,
            "argmin_or_reference_D": gamma["actual_gamma_mix_floor_D"],
            "role": "exact_export",
        },
        {
            "source": "separated_gamma_floor",
            "kernel_floor": separated_gamma_kernel_floor,
            "gamma_floor": gamma["separated_gamma_mix_floor"],
            "ratio_to_required_p95": separated_gamma_kernel_floor / required_kernel_floor_p95,
            "ratio_to_required_max": separated_gamma_kernel_floor / required_kernel_floor_max,
            "argmin_or_reference_D": gamma["actual_gamma_mix_floor_D"],
            "role": "observable_chain_sufficient",
        },
        {
            "source": "nonmix_witness_envelope",
            "kernel_floor": "",
            "gamma_floor": "",
            "ratio_to_required_p95": p95_half_radius / nonmix_env,
            "ratio_to_required_max": max_half_radius / nonmix_env,
            "argmin_or_reference_D": "",
            "role": "already_closed",
        },
    ]

    summary = {
        "eps_mix_factor_residual": eps_mix,
        "p95_half_radius": p95_half_radius,
        "max_half_radius": max_half_radius,
        "required_kernel_floor_p95": required_kernel_floor_p95,
        "required_kernel_floor_max": required_kernel_floor_max,
        "required_gamma_floor_p95": required_gamma_floor_p95,
        "required_gamma_floor_max": required_gamma_floor_max,
        "direct_kernel_floor": direct_kernel_floor,
        "gamma_induced_kernel_floor": gamma_kernel_floor,
        "separated_gamma_kernel_floor": separated_gamma_kernel_floor,
        "actual_kernel_floor": actual_kernel_floor,
        "direct_over_required_p95": direct_kernel_floor / required_kernel_floor_p95,
        "gamma_over_required_p95": gamma_kernel_floor / required_kernel_floor_p95,
        "separated_gamma_over_required_p95": separated_gamma_kernel_floor / required_kernel_floor_p95,
        "direct_over_required_max": direct_kernel_floor / required_kernel_floor_max,
        "gamma_over_required_max": gamma_kernel_floor / required_kernel_floor_max,
        "separated_gamma_over_required_max": separated_gamma_kernel_floor / required_kernel_floor_max,
        "nonmix_env": nonmix_env,
        "p95_half_over_nonmix_env": p95_half_radius / nonmix_env,
        "max_half_over_nonmix_env": max_half_radius / nonmix_env,
        "gx_floor": to_float(gx["gx_floor"]),
        "gx_floor_D": to_float(gx["gx_floor_D"]),
        "Sxx_at_gx_floor": to_float(gx["Sxx_at_gx_floor"]),
        "gx_gamma_floor_alignment": to_float(gx["gx_argmin_aligns_with_gamma_mix"]),
        "actual_gamma_mix_floor": to_float(gamma["actual_gamma_mix_floor"]),
        "actual_gamma_mix_floor_D": to_float(gamma["actual_gamma_mix_floor_D"]),
        "lifted_g_floor": to_float(lift["lifted_g_floor"]),
        "heff_loewner_factor": to_float(lift["heff_loewner_factor"]),
        "remaining_primary_target": "single_scalar_mix_kernel_floor_for_exact_schur_lift",
    }

    summary_path = OUT_DIR / "chi_open_system_exact_schur_lift_scalar_source_summary.csv"
    detail_path = OUT_DIR / "chi_open_system_exact_schur_lift_scalar_source_detail.csv"

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()

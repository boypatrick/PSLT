#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Derive conservative half-radius bounds for the dominant mix-side "
            "factorization witness from the exact exported operator map."
        )
    )
    ap.add_argument(
        "--factorization-map",
        default=str(OUTDIR / "chi_open_system_bath_factorization_map.csv"),
    )
    ap.add_argument(
        "--witness-transfer-summary",
        default=str(OUTDIR / "chi_open_system_operator_witness_transfer_summary.csv"),
    )
    ap.add_argument(
        "--detail-output",
        default=str(OUTDIR / "chi_open_system_mix_factorization_source_detail.csv"),
    )
    ap.add_argument(
        "--summary-output",
        default=str(OUTDIR / "chi_open_system_mix_factorization_source_summary.csv"),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.factorization_map).copy()
    wit = pd.read_csv(args.witness_transfer_summary).iloc[0]

    gamma_mix = df["gamma_mix_micro"].to_numpy(dtype=float)
    kappa = df["kappa_env"].to_numpy(dtype=float)
    gx = df["g_x_micro"].to_numpy(dtype=float)
    sxx = df["Sxx_delta"].to_numpy(dtype=float)
    rate_kernel_mix = df["bath_rate_kernel_mix"].to_numpy(dtype=float)

    num_resid = np.abs(gamma_mix - kappa * rate_kernel_mix)
    witness_mix = np.abs(gamma_mix / np.maximum(rate_kernel_mix, 1e-300) - kappa)

    eps_mix = float(np.max(num_resid))
    min_kernel_direct = float(np.min(rate_kernel_mix))
    min_gamma_mix = float(np.min(gamma_mix))
    kappa_max = float(np.max(kappa))
    min_gx = float(np.min(gx))
    min_sxx = float(np.min(sxx))

    kernel_floor_from_gamma = float(
        max((min_gamma_mix - eps_mix) / max(kappa_max, 1e-300), 0.0)
    )
    kernel_floor_separated = float((min_gx**2) * min_sxx)

    bound_direct = eps_mix / max(min_kernel_direct, 1e-300)
    bound_gamma = eps_mix / max(kernel_floor_from_gamma, 1e-300)
    bound_separated = eps_mix / max(kernel_floor_separated, 1e-300)

    p95_half_radius = 0.5 * float(wit["p95_safe_radius"])
    max_half_radius = 0.5 * float(wit["max_safe_radius"])

    detail = pd.DataFrame(
        {
            "D": df["D"].to_numpy(dtype=float),
            "gamma_mix_micro": gamma_mix,
            "kappa_env": kappa,
            "g_x_micro": gx,
            "Sxx_delta": sxx,
            "rate_kernel_mix": rate_kernel_mix,
            "numerator_residual_abs": num_resid,
            "mix_factorization_witness_abs": witness_mix,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "eps_mix_factor_residual": eps_mix,
                "min_rate_kernel_mix_direct": min_kernel_direct,
                "min_rate_kernel_mix_from_gamma_floor": kernel_floor_from_gamma,
                "min_rate_kernel_mix_from_separated_floor": kernel_floor_separated,
                "max_exact_mix_witness": float(np.max(witness_mix)),
                "argmax_exact_mix_witness_D": float(
                    detail.loc[np.argmax(witness_mix), "D"]
                ),
                "bound_direct_kernel_floor": bound_direct,
                "bound_gamma_floor": bound_gamma,
                "bound_separated_floor": bound_separated,
                "p95_half_radius": p95_half_radius,
                "max_half_radius": max_half_radius,
                "p95_half_over_direct_bound": p95_half_radius / max(bound_direct, 1e-300),
                "max_half_over_direct_bound": max_half_radius / max(bound_direct, 1e-300),
                "p95_half_over_gamma_bound": p95_half_radius / max(bound_gamma, 1e-300),
                "max_half_over_gamma_bound": max_half_radius / max(bound_gamma, 1e-300),
                "p95_half_over_separated_bound": p95_half_radius
                / max(bound_separated, 1e-300),
                "max_half_over_separated_bound": max_half_radius
                / max(bound_separated, 1e-300),
                "exact_over_direct_bound": float(np.max(witness_mix))
                / max(bound_direct, 1e-300),
                "exact_over_gamma_bound": float(np.max(witness_mix))
                / max(bound_gamma, 1e-300),
                "exact_over_separated_bound": float(np.max(witness_mix))
                / max(bound_separated, 1e-300),
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

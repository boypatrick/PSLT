#!/usr/bin/env python3
"""Bridge the exact-Schur scalar threshold to the current static parity floor."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "chi_open_system"


def read_summary(name: str) -> dict[str, str]:
    path = OUT_DIR / name
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def main() -> None:
    scalar = read_summary("chi_open_system_exact_schur_lift_scalar_source_summary.csv")
    gamma = read_summary("chi_open_system_gamma_mix_floor_alignment_summary.csv")

    kappa_env = float(gamma["gamma_mix_floor"]) / float(gamma["kernel_mix_floor"])
    parity_floor_core = 5.70400e-5
    parity_floor_anchor = 5.64043e-5
    gx_floor_core = parity_floor_core / 2.0
    gx_floor_anchor = parity_floor_anchor / 2.0

    sxx_floor = float(gamma["Sxx_floor"])
    sxx_local = float(gamma["Sxx_at_gx_floor"])
    scalar_gamma_threshold = float(scalar["required_gamma_floor_p95"])
    scalar_kernel_threshold = float(scalar["required_kernel_floor_p95"])

    gamma_from_anchor_global = kappa_env * gx_floor_anchor * gx_floor_anchor * sxx_floor
    gamma_from_core_global = kappa_env * gx_floor_core * gx_floor_core * sxx_floor
    gamma_from_anchor_local = kappa_env * gx_floor_anchor * gx_floor_anchor * sxx_local
    gamma_from_core_local = kappa_env * gx_floor_core * gx_floor_core * sxx_local

    rows = [
        {
            "source": "anchor_parity_floor_global_Sxx",
            "parity_floor": parity_floor_anchor,
            "gx_floor": gx_floor_anchor,
            "Sxx_used": sxx_floor,
            "implied_gamma_floor": gamma_from_anchor_global,
            "gamma_over_scalar_threshold": gamma_from_anchor_global / scalar_gamma_threshold,
            "required_gx_factor_to_close": math.sqrt(scalar_gamma_threshold / gamma_from_anchor_global),
            "role": "current_theorem_bridge",
        },
        {
            "source": "core_parity_floor_global_Sxx",
            "parity_floor": parity_floor_core,
            "gx_floor": gx_floor_core,
            "Sxx_used": sxx_floor,
            "implied_gamma_floor": gamma_from_core_global,
            "gamma_over_scalar_threshold": gamma_from_core_global / scalar_gamma_threshold,
            "required_gx_factor_to_close": math.sqrt(scalar_gamma_threshold / gamma_from_core_global),
            "role": "current_theorem_bridge",
        },
        {
            "source": "anchor_parity_floor_local_Sxx",
            "parity_floor": parity_floor_anchor,
            "gx_floor": gx_floor_anchor,
            "Sxx_used": sxx_local,
            "implied_gamma_floor": gamma_from_anchor_local,
            "gamma_over_scalar_threshold": gamma_from_anchor_local / scalar_gamma_threshold,
            "required_gx_factor_to_close": math.sqrt(scalar_gamma_threshold / gamma_from_anchor_local),
            "role": "exact_alignment_reference",
        },
        {
            "source": "core_parity_floor_local_Sxx",
            "parity_floor": parity_floor_core,
            "gx_floor": gx_floor_core,
            "Sxx_used": sxx_local,
            "implied_gamma_floor": gamma_from_core_local,
            "gamma_over_scalar_threshold": gamma_from_core_local / scalar_gamma_threshold,
            "required_gx_factor_to_close": math.sqrt(scalar_gamma_threshold / gamma_from_core_local),
            "role": "exact_alignment_reference",
        },
    ]

    summary = {
        "kappa_env_inferred": kappa_env,
        "scalar_kernel_threshold": scalar_kernel_threshold,
        "scalar_gamma_threshold": scalar_gamma_threshold,
        "parity_floor_anchor": parity_floor_anchor,
        "parity_floor_core": parity_floor_core,
        "gx_floor_anchor": gx_floor_anchor,
        "gx_floor_core": gx_floor_core,
        "Sxx_floor_global": sxx_floor,
        "Sxx_at_gx_floor": sxx_local,
        "gamma_from_anchor_global": gamma_from_anchor_global,
        "gamma_from_core_global": gamma_from_core_global,
        "gamma_from_anchor_local": gamma_from_anchor_local,
        "gamma_from_core_local": gamma_from_core_local,
        "anchor_global_over_scalar_threshold": gamma_from_anchor_global / scalar_gamma_threshold,
        "core_global_over_scalar_threshold": gamma_from_core_global / scalar_gamma_threshold,
        "anchor_local_over_scalar_threshold": gamma_from_anchor_local / scalar_gamma_threshold,
        "core_local_over_scalar_threshold": gamma_from_core_local / scalar_gamma_threshold,
        "required_anchor_gx_factor_global": math.sqrt(scalar_gamma_threshold / gamma_from_anchor_global),
        "required_core_gx_factor_global": math.sqrt(scalar_gamma_threshold / gamma_from_core_global),
        "required_anchor_gx_factor_local": math.sqrt(scalar_gamma_threshold / gamma_from_anchor_local),
        "required_core_gx_factor_local": math.sqrt(scalar_gamma_threshold / gamma_from_core_local),
        "remaining_primary_target": "tighten_static_parity_floor_until_scalar_schur_threshold",
    }

    summary_path = OUT_DIR / "chi_open_system_exact_schur_lift_overlap_bridge_summary.csv"
    detail_path = OUT_DIR / "chi_open_system_exact_schur_lift_overlap_bridge_detail.csv"

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

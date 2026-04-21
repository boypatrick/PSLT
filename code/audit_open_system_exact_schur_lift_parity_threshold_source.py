#!/usr/bin/env python3
"""Reduce the scalar exact-Schur threshold to a single static parity threshold."""

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
    bridge = read_summary("chi_open_system_exact_schur_lift_overlap_bridge_summary.csv")
    gx = read_summary("chi_open_system_gx_floor_source_summary.csv")

    m_mix_star = float(scalar["required_kernel_floor_p95"])
    gamma_mix_star = float(scalar["required_gamma_floor_p95"])

    sxx_global = float(bridge["Sxx_floor_global"])
    sxx_local = float(bridge["Sxx_at_gx_floor"])

    parity_anchor = float(bridge["parity_floor_anchor"])
    parity_core = float(bridge["parity_floor_core"])
    parity_exact = float(gx["parity_diag_contrast_floor"])

    rows = []
    summary: dict[str, float | str] = {
        "required_kernel_floor": m_mix_star,
        "required_gamma_floor": gamma_mix_star,
        "parity_floor_anchor": parity_anchor,
        "parity_floor_core": parity_core,
        "parity_floor_exact": parity_exact,
    }

    for label, sxx in (("global", sxx_global), ("local", sxx_local)):
        required_gx = math.sqrt(m_mix_star / sxx)
        required_parity = 2.0 * required_gx
        anchor_ratio = parity_anchor / required_parity
        core_ratio = parity_core / required_parity
        exact_ratio = parity_exact / required_parity
        anchor_tightening = required_parity / parity_anchor
        core_tightening = required_parity / parity_core

        summary[f"Sxx_{label}"] = sxx
        summary[f"required_gx_floor_{label}"] = required_gx
        summary[f"required_parity_floor_{label}"] = required_parity
        summary[f"anchor_ratio_over_required_{label}"] = anchor_ratio
        summary[f"core_ratio_over_required_{label}"] = core_ratio
        summary[f"exact_ratio_over_required_{label}"] = exact_ratio
        summary[f"anchor_tightening_factor_{label}"] = anchor_tightening
        summary[f"core_tightening_factor_{label}"] = core_tightening

        rows.extend(
            [
                {
                    "response_choice": label,
                    "Sxx_used": sxx,
                    "required_gx_floor": required_gx,
                    "required_parity_floor": required_parity,
                    "parity_source": "anchor_theorem_floor",
                    "parity_floor": parity_anchor,
                    "ratio_over_required": anchor_ratio,
                    "tightening_factor_to_close": anchor_tightening,
                    "role": "current_theorem",
                },
                {
                    "response_choice": label,
                    "Sxx_used": sxx,
                    "required_gx_floor": required_gx,
                    "required_parity_floor": required_parity,
                    "parity_source": "core_theorem_floor",
                    "parity_floor": parity_core,
                    "ratio_over_required": core_ratio,
                    "tightening_factor_to_close": core_tightening,
                    "role": "current_theorem",
                },
                {
                    "response_choice": label,
                    "Sxx_used": sxx,
                    "required_gx_floor": required_gx,
                    "required_parity_floor": required_parity,
                    "parity_source": "exact_bridge_floor",
                    "parity_floor": parity_exact,
                    "ratio_over_required": exact_ratio,
                    "tightening_factor_to_close": required_parity / parity_exact,
                    "role": "exact_reference",
                },
            ]
        )

    summary["remaining_primary_target"] = "single_static_parity_threshold_for_exact_schur_lift"

    summary_path = OUT_DIR / "chi_open_system_exact_schur_lift_parity_threshold_source_summary.csv"
    detail_path = OUT_DIR / "chi_open_system_exact_schur_lift_parity_threshold_source_detail.csv"

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

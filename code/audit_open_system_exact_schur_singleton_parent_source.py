#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from audit_open_system_parity_contrast_self_mirror_source import solve_fields
from extract_chi_localized_2d import Level, PhysicalParams

OUT_DIR = ROOT / "output" / "chi_open_system"


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def main() -> None:
    D = 11.0
    R = 1.35

    p = PhysicalParams()
    level = Level("fine", dr=0.06, dz=0.03)
    f = solve_fields(
        D=D,
        p=p,
        level=level,
        rho_max=3.0,
        z_margin=6.0,
        sigma=2.5,
        tol=1e-8,
        maxiter=30000,
    )

    rr = f["rr"]
    zz = f["zz"]
    contrast = f["contrast_density"]

    self_pair_density = contrast * (f["delta_plus"] + f["delta_minus"])
    plus_density = contrast * f["delta_plus"]
    minus_density = contrast * f["delta_minus"]
    cross_mass_density = contrast * f["delta_cross_m"]
    cross_deriv_density = contrast * f["delta_cross_d"]
    cross_density = contrast * f["delta_cross"]
    total_density = self_pair_density + cross_density

    rp = np.sqrt(rr * rr + (zz - D / 2.0) ** 2)
    rm = np.sqrt(rr * rr + (zz + D / 2.0) ** 2)
    core = (rp <= R) | (rm <= R)

    plus_core = float(np.sum(plus_density[core]))
    minus_core = float(np.sum(minus_density[core]))
    self_pair_core = float(np.sum(self_pair_density[core]))
    cross_mass_core = float(np.sum(cross_mass_density[core]))
    cross_deriv_core = float(np.sum(cross_deriv_density[core]))
    cross_core = float(np.sum(cross_density[core]))
    total_core = float(np.sum(total_density[core]))

    threshold = read_csv_row(OUT_DIR / "chi_open_system_exact_schur_lift_parity_threshold_source_summary.csv")
    required_parity_global = float(threshold["required_parity_floor_global"])
    required_parity_local = float(threshold["required_parity_floor_local"])

    singleton = read_csv_row(OUT_DIR / "chi_open_system_exact_schur_singleton_knot_source_summary.csv")
    lambda_11 = float(singleton["lambda_11"])

    required_core_global = required_parity_global / lambda_11
    required_core_local = required_parity_local / lambda_11

    summary = {
        "D": D,
        "R_star": R,
        "plus_core": plus_core,
        "minus_core": minus_core,
        "self_pair_core": self_pair_core,
        "cross_mass_core": cross_mass_core,
        "cross_deriv_core": cross_deriv_core,
        "cross_core": cross_core,
        "total_core": total_core,
        "cross_same_sign_as_total_core": int(
            cross_core == 0.0 or np.sign(cross_core) == np.sign(total_core)
        ),
        "self_pair_same_sign_as_total_core": int(
            self_pair_core == 0.0 or np.sign(self_pair_core) == np.sign(total_core)
        ),
        "cross_abs_share_of_total_core": abs(cross_core) / abs(total_core),
        "self_pair_abs_share_of_total_core": abs(self_pair_core) / abs(total_core),
        "required_core_floor_global": required_core_global,
        "required_core_floor_local": required_core_local,
        "self_pair_over_required_global": self_pair_core / required_core_global,
        "self_pair_over_required_local": self_pair_core / required_core_local,
        "plus_minus_asymmetry_over_self_pair": abs(plus_core - minus_core) / abs(self_pair_core),
    }

    detail_rows = [
        {"term": "plus_core", "value": plus_core, "role": "self_pair_component"},
        {"term": "minus_core", "value": minus_core, "role": "self_pair_component"},
        {"term": "self_pair_core", "value": self_pair_core, "role": "parent_target_candidate"},
        {"term": "cross_mass_core", "value": cross_mass_core, "role": "same_sign_refinement"},
        {"term": "cross_deriv_core", "value": cross_deriv_core, "role": "same_sign_refinement"},
        {"term": "cross_core", "value": cross_core, "role": "same_sign_refinement"},
        {"term": "total_core", "value": total_core, "role": "exact_core"},
        {"term": "required_core_floor_global", "value": required_core_global, "role": "global_threshold"},
        {"term": "required_core_floor_local", "value": required_core_local, "role": "local_threshold"},
    ]

    summary_path = OUT_DIR / "chi_open_system_exact_schur_singleton_parent_source_summary.csv"
    detail_path = OUT_DIR / "chi_open_system_exact_schur_singleton_parent_source_detail.csv"

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()

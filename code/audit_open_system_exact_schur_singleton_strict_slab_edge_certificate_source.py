#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


CERTIFICATE = [
    ('outer_rho', 'out', 2.834e2, 2.894e-4),
    ('bottom', 'bot', 2.738e2, 1.748e-4),
    ('corner_rb', 'rb', 6.32e1, 6.49e-5),
    ('corner_lb', 'lb', 2.36, 2.41e-4),
]


def main() -> None:
    summary = next(
        csv.DictReader(
            (
                OUT_DIR
                / 'chi_open_system_exact_schur_singleton_strict_slab_edge_product_source_summary.csv'
            ).open()
        )
    )
    detail = list(
        csv.DictReader(
            (
                OUT_DIR
                / 'chi_open_system_exact_schur_singleton_strict_slab_edge_product_source_detail.csv'
            ).open()
        )
    )
    rows = {row['edge']: row for row in detail}

    required_global = float(summary['required_global'])
    required_local = float(summary['required_local'])
    source = float(summary['source_potential_mass'])
    target_global_after_source = float(summary['target_global_after_source'])
    target_local_after_source = float(summary['target_local_after_source'])
    exact_physical = float(summary['physical_trace_mass_product_sum'])

    certificate_rows = []
    cumulative_certified = 0.0
    cumulative_exact = 0.0
    valid = True

    for edge_key, edge_label, defect_floor, weight_floor in CERTIFICATE:
        row = rows[edge_key]
        exact_defect = float(row['boundary_defect_sum'])
        exact_weight = float(row['average_poisson_weight'])
        exact_product = float(row['poisson_weighted_mass'])
        product_floor = defect_floor * weight_floor
        cumulative_certified += product_floor
        cumulative_exact += exact_product
        row_valid = exact_defect >= defect_floor and exact_weight >= weight_floor
        valid = valid and row_valid
        certificate_rows.append(
            {
                'edge': edge_key,
                'edge_label': edge_label,
                'exact_boundary_defect_sum': exact_defect,
                'certified_boundary_defect_floor': defect_floor,
                'boundary_defect_slack': exact_defect - defect_floor,
                'boundary_defect_over_floor': exact_defect / max(defect_floor, 1e-300),
                'exact_average_poisson_weight': exact_weight,
                'certified_average_poisson_weight_floor': weight_floor,
                'average_weight_slack': exact_weight - weight_floor,
                'average_weight_over_floor': exact_weight / max(weight_floor, 1e-300),
                'exact_product': exact_product,
                'certified_product_floor': product_floor,
                'product_slack': exact_product - product_floor,
                'exact_product_over_certified': exact_product / max(product_floor, 1e-300),
                'cumulative_exact_product': cumulative_exact,
                'cumulative_certified_product': cumulative_certified,
                'cumulative_certified_over_global_after_source': cumulative_certified
                / max(target_global_after_source, 1e-300),
                'cumulative_certified_over_local_after_source': cumulative_certified
                / max(target_local_after_source, 1e-300),
                'scalar_certificate_valid': row_valid,
            }
        )

    certified_outer = certificate_rows[0]['certified_product_floor']
    certified_bottom = certificate_rows[1]['certified_product_floor']
    certified_rb = certificate_rows[2]['certified_product_floor']
    certified_lb = certificate_rows[3]['certified_product_floor']
    certified_outer_bottom = certified_outer + certified_bottom
    certified_outer_bottom_rb = certified_outer_bottom + certified_rb
    certified_edge_sum = certified_outer_bottom_rb + certified_lb
    certified_total_with_source = certified_edge_sum + source
    gap_after_outer_bottom_rb = target_global_after_source - certified_outer_bottom_rb

    summary_out = {
        'required_global': required_global,
        'required_local': required_local,
        'source_potential_mass': source,
        'target_global_after_source': target_global_after_source,
        'target_local_after_source': target_local_after_source,
        'exact_physical_trace_mass': exact_physical,
        'certified_outer_wall_mass': certified_outer,
        'certified_bottom_edge_mass': certified_bottom,
        'certified_right_lower_corner_mass': certified_rb,
        'certified_left_lower_corner_mass': certified_lb,
        'certified_outer_bottom_mass': certified_outer_bottom,
        'certified_outer_bottom_right_corner_mass': certified_outer_bottom_rb,
        'certified_edge_sum': certified_edge_sum,
        'certified_total_with_source': certified_total_with_source,
        'certified_edge_sum_over_global_after_source': certified_edge_sum
        / max(target_global_after_source, 1e-300),
        'certified_edge_sum_over_local_after_source': certified_edge_sum
        / max(target_local_after_source, 1e-300),
        'certified_total_over_required_global': certified_total_with_source
        / max(required_global, 1e-300),
        'certified_total_over_required_local': certified_total_with_source
        / max(required_local, 1e-300),
        'certified_gap_over_global_after_source': certified_edge_sum - target_global_after_source,
        'certified_total_gap_over_required_global': certified_total_with_source - required_global,
        'certified_outer_bottom_over_local_after_source': certified_outer_bottom
        / max(target_local_after_source, 1e-300),
        'certified_gap_after_outer_bottom_right_corner': gap_after_outer_bottom_rb,
        'certified_left_corner_over_gap_after_outer_bottom_right_corner': certified_lb
        / max(gap_after_outer_bottom_rb, 1e-300),
        'exact_minus_certified_edge_sum': exact_physical - certified_edge_sum,
        'scalar_certificate_valid': valid,
        'certificate_route': 'rounded_outer_bottom_right_corner_left_corner_edge_products',
    }

    summary_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_edge_certificate_source_summary.csv'
    )
    detail_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_edge_certificate_source_detail.csv'
    )

    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_out.keys()))
        writer.writeheader()
        writer.writerow(summary_out)
    with detail_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(certificate_rows[0].keys()))
        writer.writeheader()
        for row in certificate_rows:
            writer.writerow(row)

    print(summary_path)
    print(detail_path)


if __name__ == '__main__':
    main()

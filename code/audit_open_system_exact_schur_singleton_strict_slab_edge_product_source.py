#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    summary = next(
        csv.DictReader(
            (OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_physical_trace_source_summary.csv').open()
        )
    )
    detail = list(
        csv.DictReader(
            (OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_physical_trace_source_detail.csv').open()
        )
    )
    rows = {row['category']: row for row in detail}

    required_global = float(summary['required_global'])
    required_local = float(summary['required_local'])
    source = float(summary['source_potential_mass'])
    target_global_after_source = float(summary['physical_trace_required_global_after_source'])
    target_local_after_source = float(summary['physical_trace_required_local_after_source'])

    edge_order = ['outer_rho', 'bottom', 'corner_rb', 'corner_lb', 'axis']
    edge_rows = []
    cumulative = 0.0
    for name in edge_order:
        row = rows[name]
        defect_sum = float(row['boundary_defect_sum'])
        avg_weight = float(row['average_poisson_weight'])
        mass = float(row['poisson_weighted_mass'])
        cumulative += mass
        edge_rows.append(
            {
                'edge': name,
                'boundary_defect_sum': defect_sum,
                'average_poisson_weight': avg_weight,
                'poisson_weighted_mass': mass,
                'cumulative_mass': cumulative,
                'cumulative_over_global_after_source': cumulative / max(target_global_after_source, 1e-300),
                'cumulative_over_local_after_source': cumulative / max(target_local_after_source, 1e-300),
            }
        )

    outer = float(rows['outer_rho']['poisson_weighted_mass'])
    bottom = float(rows['bottom']['poisson_weighted_mass'])
    rb = float(rows['corner_rb']['poisson_weighted_mass'])
    lb = float(rows['corner_lb']['poisson_weighted_mass'])
    axis = float(rows['axis']['poisson_weighted_mass'])
    outer_bottom = outer + bottom
    outer_bottom_rb = outer_bottom + rb
    physical = outer_bottom_rb + lb + axis
    gap_after_outer_bottom = target_global_after_source - outer_bottom
    gap_after_outer_bottom_rb = target_global_after_source - outer_bottom_rb

    physical_defect_sum = sum(float(rows[name]['boundary_defect_sum']) for name in edge_order)
    physical_avg_weight = physical / max(physical_defect_sum, 1e-300)
    required_avg_weight = target_global_after_source / max(physical_defect_sum, 1e-300)

    summary_out = {
        'required_global': required_global,
        'required_local': required_local,
        'source_potential_mass': source,
        'target_global_after_source': target_global_after_source,
        'target_local_after_source': target_local_after_source,
        'outer_wall_mass': outer,
        'bottom_edge_mass': bottom,
        'right_lower_corner_mass': rb,
        'left_lower_corner_mass': lb,
        'axis_mass': axis,
        'outer_bottom_mass': outer_bottom,
        'outer_bottom_right_corner_mass': outer_bottom_rb,
        'physical_trace_mass_product_sum': physical,
        'physical_trace_mass_reference': float(summary['physical_trace_mass']),
        'physical_trace_product_resid_abs': abs(physical - float(summary['physical_trace_mass'])),
        'physical_trace_over_global_after_source': physical / max(target_global_after_source, 1e-300),
        'physical_trace_over_local_after_source': physical / max(target_local_after_source, 1e-300),
        'outer_bottom_over_global_after_source': outer_bottom / max(target_global_after_source, 1e-300),
        'outer_bottom_right_corner_over_global_after_source': outer_bottom_rb / max(
            target_global_after_source, 1e-300
        ),
        'gap_after_outer_bottom': gap_after_outer_bottom,
        'gap_after_outer_bottom_right_corner': gap_after_outer_bottom_rb,
        'left_corner_over_gap_after_outer_bottom_right_corner': lb / max(gap_after_outer_bottom_rb, 1e-300),
        'physical_boundary_defect_sum': physical_defect_sum,
        'physical_average_poisson_weight': physical_avg_weight,
        'required_physical_average_poisson_weight': required_avg_weight,
        'physical_average_weight_over_required': physical_avg_weight / max(required_avg_weight, 1e-300),
        'preferred_parent_side_route': 'outer_bottom_right_corner_plus_left_corner_edge_product',
    }

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_edge_product_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_edge_product_source_detail.csv'

    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_out.keys()))
        writer.writeheader()
        writer.writerow(summary_out)
    with detail_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0].keys()))
        writer.writeheader()
        for row in edge_rows:
            writer.writerow(row)

    print(summary_path)
    print(detail_path)


if __name__ == '__main__':
    main()

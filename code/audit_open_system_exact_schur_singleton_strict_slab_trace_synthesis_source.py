#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def read_one(path: Path) -> dict[str, str]:
    return next(csv.DictReader(path.open()))


def main() -> None:
    boundary_flux = read_one(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_flux_certificate_source_summary.csv'
    )
    exterior = read_one(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_summary.csv'
    )
    adjoint = read_one(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source_summary.csv'
    )
    gs_rate = read_one(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_gs_rate_source_summary.csv'
    )
    adjoint_edges = list(
        csv.DictReader(
            (
                OUT_DIR
                / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source_edges.csv'
            ).open()
        )
    )

    required_global = float(exterior['required_global'])
    required_local = float(exterior['required_local'])
    source = float(exterior['source_potential_mass'])
    target_global_after_source = float(exterior['target_global_after_source'])
    target_local_after_source = float(exterior['target_local_after_source'])

    stages = [
        {
            'stage': 'O66_rounded_b_p_floors',
            'description': 'pointwise rounded b_q^- p_q^- certificate',
            'trace_product': float(boundary_flux['certified_edge_sum']),
            'closed_by_parent_side': 'partially_before_O68_O70',
        },
        {
            'stage': 'O68_5sig_flux_with_p_floor',
            'description': 'missing-neighbor flux floors with old p_q^- floors',
            'trace_product': float(boundary_flux['flux_certified_product_with_p_floor_sum']),
            'closed_by_parent_side': 'flux_identity_only',
        },
        {
            'stage': 'O69_4sig_exterior_flux_with_p_floor',
            'description': 'parent exterior phi barrier plus old p_q^- floors',
            'trace_product': float(exterior['direct_flux_product_with_p_floor_sum']),
            'closed_by_parent_side': 'b_side_closed',
        },
        {
            'stage': 'O70_GS548_parent_barrier',
            'description': 'parent exterior phi barrier plus GS548 adjoint barrier',
            'trace_product': float(adjoint['product_sweep_direct_flux_product']),
            'closed_by_parent_side': 'main_closed_certificate',
        },
        {
            'stage': 'O70_GS859_point_floor_completion',
            'description': 'same monotone adjoint sequence also dominates old p_q^- floors',
            'trace_product': float(adjoint['floor_sweep_direct_flux_product']),
            'closed_by_parent_side': 'optional_point_floor_completion',
        },
    ]
    for row in stages:
        trace_product = row['trace_product']
        row['trace_over_global_after_source'] = trace_product / max(target_global_after_source, 1e-300)
        row['trace_gap_over_global_after_source'] = trace_product - target_global_after_source
        row['source_completed_total'] = trace_product + source
        row['source_completed_over_required_global'] = (trace_product + source) / max(required_global, 1e-300)

    gs_outer_bottom = float(adjoint['product_sweep_outer_bottom_product'])
    gs859_outer_bottom = sum(
        float(row['gs_floor_sweep_flux_product']) for row in adjoint_edges if row['edge'] in {'outer_rho', 'bottom'}
    )
    edge_rows = []
    cumulative = 0.0
    for row in adjoint_edges:
        product = float(row['gs_product_sweep_flux_product'])
        cumulative += product
        edge_rows.append(
            {
                'edge': row['edge'],
                'edge_label': row['edge_label'],
                'point_count': row['point_count'],
                'direct_flux_sum': row['direct_flux_sum'],
                'gs548_product': row['gs_product_sweep_flux_product'],
                'gs859_product': row['gs_floor_sweep_flux_product'],
                'old_4sig_product': row['old_4sig_floor_flux_product'],
                'cumulative_gs548_product': cumulative,
                'cumulative_gs548_over_global_after_source': cumulative
                / max(target_global_after_source, 1e-300),
                'cumulative_gs548_over_local_after_source': cumulative
                / max(target_local_after_source, 1e-300),
            }
        )

    summary = {
        'D': exterior['D'],
        'rho_anchor': exterior['rho_anchor'],
        'zeta_anchor': exterior['zeta_anchor'],
        'strict_cap': exterior['strict_cap'],
        'required_global': required_global,
        'required_local': required_local,
        'source_potential_mass': source,
        'target_global_after_source': target_global_after_source,
        'target_local_after_source': target_local_after_source,
        'edge_point_count': exterior['edge_point_count'],
        'exterior_trace_point_count': exterior['exterior_trace_point_count'],
        'missing_flux_component_count': exterior['missing_flux_component_count'],
        'strict_slab_point_count': adjoint['strict_slab_point_count'],
        'closed_boundary_flux_identity': True,
        'closed_exterior_phi_barrier': exterior['trace_floor_certificate_valid'],
        'closed_adjoint_gs_product_barrier': adjoint['product_sweep_direct_route_valid'],
        'closed_adjoint_gs_point_floor_completion': adjoint['floor_sweep_point_floor_certificate_valid'],
        'main_certificate_stage': 'O70_GS548_parent_barrier',
        'main_trace_product': adjoint['product_sweep_direct_flux_product'],
        'main_trace_over_global_after_source': adjoint['product_sweep_direct_flux_product_over_global_after_source'],
        'main_trace_gap_over_global_after_source': adjoint[
            'product_sweep_direct_flux_gap_over_global_after_source'
        ],
        'main_source_completed_total': adjoint['product_sweep_total_with_source'],
        'main_source_completed_over_required_global': adjoint['product_sweep_total_over_required_global'],
        'main_outer_bottom_product': adjoint['product_sweep_outer_bottom_product'],
        'main_outer_bottom_over_local_after_source': adjoint['product_sweep_outer_bottom_over_local_after_source'],
        'main_outer_bottom_gap_over_local_after_source': adjoint[
            'product_sweep_outer_bottom_gap_over_local_after_source'
        ],
        'optional_point_floor_stage': 'O70_GS859_point_floor_completion',
        'optional_point_floor_trace_product': adjoint['floor_sweep_direct_flux_product'],
        'optional_point_floor_trace_over_global_after_source': adjoint[
            'floor_sweep_direct_flux_product_over_global_after_source'
        ],
        'optional_point_floor_total_over_required_global': adjoint['floor_sweep_total_over_required_global'],
        'optional_point_floor_outer_bottom_product': gs859_outer_bottom,
        'optional_point_floor_outer_bottom_over_local_after_source': gs859_outer_bottom
        / max(target_local_after_source, 1e-300),
        'product_sweeps': adjoint['product_sweeps'],
        'floor_sweeps': adjoint['floor_sweeps'],
        'min_exterior_phi_barrier_minus_floor': exterior['min_phi_barrier_minus_floor'],
        'max_exterior_phi_floor_over_exact': exterior['max_phi_floor_over_exact'],
        'gs548_min_point_over_4sig_p_floor': adjoint['product_sweep_min_point_over_4sig_p_floor'],
        'gs859_min_point_over_4sig_p_floor': adjoint['floor_sweep_min_point_over_4sig_p_floor'],
        'first_global_sweep_by_rate_audit': gs_rate['first_global_sweep'],
        'first_local_sweep_by_rate_audit': gs_rate['first_local_sweep'],
        'first_point_floor_sweep_by_rate_audit': gs_rate['first_point_floor_sweep'],
        'gs548_alpha_by_rate_audit': gs_rate['product_sweep_alpha'],
        'gs548_tail_bound_by_rate_audit': gs_rate['product_sweep_flux_tail_bound'],
        'gs548_exact_gap_by_rate_audit': gs_rate['product_sweep_exact_flux_gap'],
        'gs859_alpha_by_rate_audit': gs_rate['floor_sweep_alpha'],
        'gs859_tail_bound_by_rate_audit': gs_rate['floor_sweep_flux_tail_bound'],
        'gs859_exact_gap_by_rate_audit': gs_rate['floor_sweep_exact_flux_gap'],
        'closed_gs_rate_tail_bound': gs_rate['all_checkpoint_tail_bounds_valid'],
        'secondary_tightening_1': 'closed: O68-O70 compressed into one reviewer lemma',
        'secondary_tightening_2': 'closed: analytic GS increment-rate tail bound for 548/859 sweeps',
        'no_new_support_object_carrier': True,
        'roadmap_status': 'parent_side_closed_editorial_compression_only',
    }

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_trace_synthesis_source_summary.csv'
    stages_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_trace_synthesis_source_stages.csv'
    edges_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_trace_synthesis_source_edges.csv'

    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with stages_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stages[0].keys()))
        writer.writeheader()
        for row in stages:
            writer.writerow(row)
    with edges_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0].keys()))
        writer.writeheader()
        for row in edge_rows:
            writer.writerow(row)

    print(summary_path)
    print(stages_path)
    print(edges_path)


if __name__ == '__main__':
    main()

# v3p7 inline t_coh patch

This bundle patches `run_rho_window_grid_refine.py` so that running it will produce:

- `cases_raw_results_repro.csv`
- `final_acceptance_table_repro.csv`
- `final_acceptance_table_with_tcoh_repro.csv`

It also includes:
- `tcoh_geom_curve_axis1d.csv` (omega1(D), C_star anchors)
- `final_acceptance_table_with_tcoh.csv` (precomputed from the included final_acceptance_table.csv)

Coherence time closure:
    t_coh(D) = (2π/C_star) * omega1(D) * exp(Smin(D))
with Smin(D) taken from the 2D refined action (S_fine_rho4).
C_star is calibrated from the paper unified table (D>=12):
    C_star = 1.82925565176562

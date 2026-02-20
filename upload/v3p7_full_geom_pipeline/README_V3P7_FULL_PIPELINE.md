# v3p7 full pipeline (geom-closed t_coh, linear-log extrap)

This folder merges:
1) v3p7 rho-window + true-refinement grid-convergence acceptance (S_min validation)
2) transition_band_scan.py patched to use geometry-closed coherence time:
   t_coh(D) = t_scalar * tcoh_geom(D)

## Quick run (recommended)

### A) Acceptance: window + grid convergence (repro)
python run_rho_window_grid_refine.py

Outputs:
- final_acceptance_table_repro.csv
- final_acceptance_table_with_tcoh_repro.csv
- cases_raw_results_repro.csv
- window_check_fine_*.png
- grid_convergence_true_refine_*.png
- fine_consistency_vs_Sz.png
- todo3B_final_report.txt

### B) Transition band: geom-closed t_coh with linear-log extrap (repro)
python transition_band_scan.py transition-band --preset band --tcoh-mode geom --tcoh-extrap linear-log

Outputs:
- fig_transition_band.png
- fig_transition_band_overlay.png   (includes log10(t_coh) contour overlays)

## Notes

- In --tcoh-mode geom, preset time knobs (t_rep / t_list / t_windows) are interpreted as *multipliers*.
- The code auto-restricts (Dmin,Dmax) to the curve CSV domain unless you pass --tcoh-keep-domain.
- The geometry-closed curve is provided in: tcoh_geom_curve_axis1d.csv

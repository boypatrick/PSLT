# transition_band_scan.py (geometry-closed t_coh patch)

## New CLI flags
- --tcoh-mode {fixed,geom}
- --tcoh-curve PATH (default: tcoh_geom_curve_axis1d.csv)
- --tcoh-extrap {linear-log,clamp,error}  (default: linear-log)
- --tcoh-keep-domain  (keep original Dmin/Dmax instead of restricting to curve domain)

## Semantics
- fixed mode (default): existing behavior (t_rep, t_list, t_windows are absolute t_coh)
- geom mode: t_rep, t_list, t_windows are treated as multipliers:
    t_coh(D) = (multiplier) * tcoh_geom(D)

## Outputs
- transition-band overlay adds log10(t_coh) contour lines
- band-length diagnostic x-axis uses effective t_coh at D_fix

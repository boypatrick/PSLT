scan_v3p1: TODO#1 diagnostic/physical S split (axis-1D)

What changed vs v3:
- Add allow_unstable_S switch (default: True for diagnostic_deepwell, False for physical_gap).
- Compute and output:
    S_z           : diagnostic WKB action (computed for bound states when allow_unstable_S=True OR stable bound)
    S_z_physical  : physical WKB action (ONLY for stable bound: omega2>0)
    is_diagnostic_tunnel = has_bound & has_turning_points & S_z>0
    is_physical_tunnel   = has_stable_bound & has_turning_points & S_z_physical>0
- Turning points (z1,z2) are now recorded regardless of stability.

Speed note:
- eig_1d now uses scipy.linalg.eigh_tridiagonal(select='i') to compute only the lowest nstates eigenpairs.

Reproduce:
    python3 scan_D_axis1d_3_v3p1.py --preset diagnostic_deepwell
    python3 scan_D_axis1d_3_v3p1.py --preset physical_gap

Override allow_unstable_S:
    --allow_unstable_S      force enable
    --no_allow_unstable_S   force disable

Outputs in this zip:
- scan_axis1d_v3p1_{preset}.csv
- rep_axis1d_v3p1_{diag,phys}_{U,psi,meta}_D{6,12,18}.csv

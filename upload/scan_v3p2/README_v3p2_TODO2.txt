scan_v3p2: TODO#2 2D axisymmetric Laplacian validation (rho,z)

Added:
- validate_axis2d_laplacian_v3p2.py
  Computes analytic 3D Laplacian of smoothed dual-center kernel in full (rho,z), derives U(rho,z),
  and validates axis agreement with the 1D scan engine outputs.

Analytic identity used (full 3D scalar Laplacian, valid in cylindrical coords as well):
  ∇^2 (r^2+eps^2)^(-1/2) = -3 eps^2 (r^2+eps^2)^(-5/2)

Outputs (for D=6,12,18):
- U2D_{preset}_D{D}.npy            : U(rho,z) on rho grid and z grid (float32)
- U2D_{preset}_grid_D{D}.npz       : grid + parameter metadata
- rel_error_axis_{preset}_D{D}.csv : full axis comparison table
- rel_error_axis_{preset}.png        : semilogy plot of axis relative error vs z

Preset physical_gap max axis rel error (|U2D(0,z)-U1D(z)| / max|U1D|):
  D=6  : 1.880e-08
  D=12 : 3.015e-08
  D=18 : 3.279e-08

Preset diagnostic_deepwell max axis rel error:
  D=6  : 3.547e-08
  D=12 : 3.868e-08
  D=18 : 3.396e-08

Interpretation:
- The axis agreement is at ~1e-8 level (float32 rounding dominated), i.e. the 1D fast scan
  uses the correct on-axis restriction of the full (rho,z) analytic Laplacian.
- This validates the Laplacian expression; it does NOT yet validate the 1D tunneling action
  as the minimum-action path (that is TODO#3).

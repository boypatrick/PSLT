# Protected Spin^c V2 Nodal/Sturm Ordering Diagnostic

## Scope

This note records V2 in the optional protected-\(Spin^c\) bridge queue.  It
does not modify `paper/main.tex`, the baseline scan, the release figures, or
the A1--A5 submission wording.

V1 already supplies a fixed tapered-polynomial anchor-overlap certificate from
the protected degree flag into the audited \(N=1,2,3\) finite-volume spectral
subspace.  V2 asks whether this bridge also has a physically readable ordering:

\[
1\longleftrightarrow u_1,\qquad
\zeta\longleftrightarrow u_2,\qquad
\zeta^2\longleftrightarrow u_3.
\]

## Test

The protected degree basis has parity/order pattern

\[
1:\ \text{even},\qquad
\zeta:\ \text{odd},\qquad
\zeta^2:\ \text{even}.
\]

On the finite-volume side, for each audited \(D\), the script extracts the
lowest three modes using the same G1 convention: compute a broad shift-invert
Ritz window, sort the returned eigenvalues, then take the lowest three modes.

For each mode \(u_n\), it checks:

1. parity under \(z\mapsto -z\),
2. sign-change count of the radial-weighted axial profile

\[
\phi_n(z)=\int 2\pi\rho\,u_n(\rho,z)\,d\rho,
\]

3. sign-change count along the dominant radial-energy line.

The expected pattern is

\[
\text{parity}(u_1,u_2,u_3)=(\mathrm{even},\mathrm{odd},\mathrm{even}),
\]

\[
\text{nodes}(u_1,u_2,u_3)=(0,1,2).
\]

The parity gate uses a separation ratio rather than a machine-zero absolute
test:

\[
\frac{\text{expected-parity residual}}
{\text{wrong-parity residual}}
\le 10^{-3},
\]

together with the absolute guard that the expected-parity residual is at most
\(2\times10^{-3}\).  This avoids treating harmless Ritz/vector-grid parity
mixing at the \(10^{-3}\) level as a physical ordering failure when the wrong
parity remains separated at order unity.

## Executable check

Run:

```bash
python3 code/check_spinc_nodal_sturm_order.py
```

The script writes:

```text
output/spinc_bridge/spinc_nodal_sturm_detail.csv
output/spinc_bridge/spinc_nodal_sturm_summary.csv
```

This diagnostic is ordering support for the optional protected-index bridge. It
does not change the current submission baseline.

## Current Result

On the audited G1 \(D\)-grid, V2 confirms the protected degree ordering:

```text
NODAL_STURM_ORDER_CONFIRMED
```

The summary certificate reports

\[
\max_D
\frac{\text{expected-parity residual}}
{\text{wrong-parity residual}}
=4.998264650521739\times10^{-4}
<10^{-3},
\]

\[
\max_D(\text{expected-parity residual})
=9.996528052344774\times10^{-4}
<2\times10^{-3},
\]

and both nodal checks are exact:

\[
\max_D |\Delta n_{\rm axial}|=0,\qquad
\max_D |\Delta n_{\rm dominant\ line}|=0.
\]

Thus the first three finite-volume spectral modes have the ordering

\[
u_1:\mathrm{even},0\text{ nodes},\qquad
u_2:\mathrm{odd},1\text{ node},\qquad
u_3:\mathrm{even},2\text{ nodes},
\]

matching the protected degree flag

\[
1,\quad \zeta,\quad \zeta^2.
\]

Together with V1, this gives an optional finite-volume bridge package:

\[
\boxed{
H^0(\mathbb{CP}^1,\mathcal O(2))
\text{ has a fixed anchor overlap and nodal/Sturm ordering compatible with }
N=1,2,3.
}
\]

This remains a conditional protected-index upgrade, not a change to the
submission baseline.

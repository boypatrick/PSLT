# Protected Spin^c V1 Anchor-Overlap Diagnostic

## Scope

This note records the V1 check in the optional protected-\(Spin^c\) bridge
queue.  It does not modify `paper/main.tex`, the baseline scan, the release
figures, or the A1--A5 submission wording.

The question is narrower than a new family-count theorem:

\[
\text{Does a fixed polynomial/moment lift of }
H^0(\mathbb{CP}^1,\mathcal O(2))
\text{ have an invertible overlap with the current }
N=1,2,3
\text{ spectral subspace?}
\]

## Canonical finite-box lift

On the same finite-volume grid used by the G1 low-mode certificate, define

\[
x=\frac{z}{z_{\max}},
\]

and use the Dirichlet-compatible taper

\[
\tau_D(\rho,z)
=
\cos\!\left(\frac{\pi\rho}{2\rho_{\max}}\right)
\cos\!\left(\frac{\pi z}{2z_{\max}}\right).
\]

The degree-ordered polynomial lift is

\[
b_0=\tau_D,\qquad
b_1=\tau_D x,\qquad
b_2=\tau_D x^2.
\]

After weighted Gram-Schmidt under the cylindrical measure

\[
\langle f,g\rangle_D
=
\int 2\pi\rho\, f(\rho,z)g(\rho,z)\,d\rho\,dz,
\]

we obtain an orthonormal flag basis \(q_0,q_1,q_2\) spanning the same degree
flag

\[
\langle b_0\rangle
\subset
\langle b_0,b_1\rangle
\subset
\langle b_0,b_1,b_2\rangle.
\]

Let \(u_1(D),u_2(D),u_3(D)\) be the first three normalized finite-volume
spectral modes extracted from the same localized generalized operator as G1.
The anchor-overlap matrix is

\[
A_{nk}(D)=\langle u_n(D),q_k(D)\rangle_D,
\qquad n,k=1,2,3.
\]

## Promotion criteria

The diagnostic separates two levels:

1. **Invertible anchor bridge.**

\[
\min_D \sigma_{\min}(A(D))>10^{-6}.
\]

This says the fixed polynomial lift is not orthogonal to any protected
direction inside the current three-mode spectral subspace.

2. **Strong canonical promotion.**

\[
\max_{D,k}\sqrt{1-\sum_{n=1}^3 |A_{nk}(D)|^2}\le 0.5.
\]

This stronger condition says the polynomial lift mostly lands inside the first
three spectral modes, rather than merely having an invertible projection onto
them.

The second threshold is intentionally strict and conservative.  Failing it does
not invalidate the dimension/flag bridge; it only prevents claiming a strong
canonical spectral embedding.

## Executable check

Run:

```bash
python3 code/check_spinc_anchor_overlap.py
```

The script writes:

```text
output/spinc_bridge/spinc_anchor_overlap_detail.csv
output/spinc_bridge/spinc_anchor_overlap_summary.csv
```

By default the script records the verdict as a diagnostic artifact and exits
successfully when the computation completes.  Passing `--strict-invertible`
turns the invertibility condition into a hard process gate.

## Current Result

On the audited G1 \(D\)-grid, the fixed tapered-polynomial degree-flag anchor
does promote the protected-index route to a canonical finite-volume spectral
embedding diagnostic.  This remains outside the submission baseline, but it
closes V1 positively.

The output status is

```text
PROMOTABLE_ANCHOR_INTERTWINER
```

with summary values

\[
\min_D\sigma_{\min}(A(D))=0.9759703999227656,
\]

\[
\min_D|\det A(D)|=0.9589442239343868,
\]

\[
\max_D\|P_{>3}T_D\|=0.21780400121763835.
\]

The worst case is \(D=20\), where

\[
\sigma_{\min}(A)=0.9759703999227656,
\qquad
\|P_{>3}T_D\|=0.21780400121763835.
\]

Thus V1 closes positively for this fixed, non-tuned anchor:

\[
\boxed{
\text{the tapered moment-polynomial lift gives an invertible, small-tail
embedding into the first three finite-volume spectral modes.}
}
\]

The next protected-\(Spin^c\) bridge check, if pursued, should therefore be V2:
an invariant nodal/Sturm ordering diagnostic.  V2 is no longer needed to rescue
V1; it is needed to decide whether the positive anchor-overlap certificate has
a physically readable ordering interpretation.

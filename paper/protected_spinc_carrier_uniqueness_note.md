# Protected Spin^c Carrier Uniqueness Note

## Scope

This note is the V5 item in the optional protected \(Spin^c\) bridge queue.  It
does not modify `paper/main.tex`, the baseline scan, figure generation, or the
submission A1--A5 wording.

V3 conditionally selected

\[
C\simeq\mathbb{CP}^1,
\]

and V4 conditionally locked

\[
L_R\simeq\mathcal O(2).
\]

V5 asks why the protected family-count carrier should be the
Kähler/Dolbeault \(Spin^c\) carrier

\[
\Lambda^{0,*}T^*C\otimes L_R,
\]

rather than the ordinary spin carrier

\[
K_C^{1/2}\otimes L_R.
\]

The conservative answer is:

> Ordinary spin is mathematically legal on \(\mathbb{CP}^1\), but with the same
> V4 line bundle \(L_R=\mathcal O(2)\) it gives two positive holomorphic modes,
> not three.  The Dolbeault \(Spin^c\) carrier is selected by the projective
> Kähler quantization requirement and by compatibility with the V1/V2
> three-mode spectral bridge.

Thus V5 is a conditional carrier-selection theorem, not a claim that ordinary
spin is illegal.

## Two Carriers On \(\mathbb{CP}^1\)

The canonical bundle of the Riemann sphere is

\[
K_{\mathbb{CP}^1}\simeq\mathcal O(-2).
\]

An ordinary spin structure has

\[
K_{\mathbb{CP}^1}^{1/2}\simeq\mathcal O(-1).
\]

The Kähler \(Spin^c\) carrier instead uses

\[
S^+_{Spin^c}=\Lambda^{0,0}T^*C,
\qquad
S^-_{Spin^c}=\Lambda^{0,1}T^*C.
\]

After twisting by the V4 line bundle \(L_R=\mathcal O(2)\), the positive
Dolbeault \(Spin^c\) zero modes are

\[
\ker D_{Spin^c,+}
\simeq
H^0(\mathbb{CP}^1,\mathcal O(2)).
\]

Therefore

\[
h^0(\mathbb{CP}^1,\mathcal O(2))=3,
\qquad
h^1(\mathbb{CP}^1,\mathcal O(2))=0.
\]

The ordinary spin carrier with the same parent twist has positive bundle

\[
K^{1/2}\otimes L_R
\simeq
\mathcal O(-1)\otimes\mathcal O(2)
\simeq
\mathcal O(1).
\]

Hence

\[
h^0(\mathbb{CP}^1,\mathcal O(1))=2,
\qquad
h^1(\mathbb{CP}^1,\mathcal O(1))=0.
\]

So ordinary spin with the same \(L_R=\mathcal O(2)\) gives two positive
holomorphic modes, not three.

## Todd Shift Interpretation

The Dolbeault \(Spin^c\) index on a complex curve is

\[
\mathrm{ind}(D_{Spin^c,L})
=
\int_C \mathrm{ch}(L)\,\mathrm{Td}(C)
=
\deg L+1-g.
\]

For \(C=\mathbb{CP}^1\), \(g=0\), and \(L=\mathcal O(2)\), this gives

\[
\mathrm{ind}(D_{Spin^c,L_R})=2+1=3.
\]

The ordinary spin Dirac index twisted by the same \(L_R\) is

\[
\mathrm{ind}(D_{\mathrm{spin},L_R})
=
\int_C c_1(L_R)
=2.
\]

The missing unit is precisely the Kähler/Dolbeault Todd shift.  This is why
the protected family-count carrier must be the projective \(Spin^c\) carrier if
the V4 flux lock is kept fixed.

## Why Ordinary Spin Cannot Be Rescued Without Changing V4

One can force ordinary spin to have three positive holomorphic modes only by
changing the twist:

\[
K^{1/2}\otimes\mathcal O(3)
\simeq
\mathcal O(2).
\]

Then

\[
h^0(\mathbb{CP}^1,\mathcal O(2))=3.
\]

But this uses \(\mathcal O(3)\), not the V4 line bundle
\(\mathcal O(2)\).  It adds one extra flux/ramification unit and therefore
violates the two-center unit-flux lock.

Thus the alternatives are:

1. keep V4 fixed and use the Dolbeault \(Spin^c\) carrier, obtaining three
   protected modes;
2. use ordinary spin with the same V4 flux, obtaining only two modes;
3. use ordinary spin with an added flux unit, obtaining three modes but
   breaking the V4 parent-side lock.

Only the first option is compatible with the protected-index route.

## Compatibility With V1/V2

The current finite-volume bridge package already supplies a three-dimensional
spectral target:

\[
N=1,2,3.
\]

V1 gives a promotable three-column tapered-polynomial anchor overlap, and V2
confirms the physical ordering

\[
\text{even},\text{odd},\text{even},
\qquad
0,1,2
\]

by nodal/Sturm diagnostics.  The Dolbeault \(Spin^c\) basis

\[
\{1,\zeta,\zeta^2\}
\]

has exactly three elements and is compatible with this bridge.  The ordinary
spin same-flux carrier has only

\[
H^0(\mathbb{CP}^1,\mathcal O(1))
=
\mathrm{span}\{1,\zeta\},
\]

so it cannot support the already audited three-layer spectral flag without
changing the parent twist.

## V5 Verdict

The correct V5 statement is

\[
\boxed{
C=\mathbb{CP}^1,\quad L_R=\mathcal O(2),
\quad\text{projective/Kähler family quantization}
\Longrightarrow
\Lambda^{0,*}T^*C\otimes L_R
\text{ is the protected carrier.}
}
\]

It is not

\[
\boxed{
\text{ordinary spin is mathematically illegal.}
}
\]

Ordinary spin is legal; it is simply the wrong carrier for the V3/V4 protected
family-count mechanism.

## Executable Check

The companion gate

```bash
python3 code/check_spinc_carrier_uniqueness.py
```

writes

```text
output/spinc_bridge/spinc_carrier_uniqueness_summary.csv
output/spinc_bridge/spinc_carrier_uniqueness_detail.csv
output/spinc_bridge/spinc_carrier_uniqueness_detail.json
```

and verifies:

1. Dolbeault \(Spin^c\) with \(L_R=\mathcal O(2)\) gives
   \((h^0,h^1,\mathrm{index})=(3,0,3)\),
2. ordinary spin with the same \(L_R=\mathcal O(2)\) gives index \(2\),
3. ordinary spin needs \(\mathcal O(3)\) to get three modes, which adds one
   flux unit beyond V4,
4. the selected carrier is compatible with the V1/V2 three-mode spectral
   bridge.

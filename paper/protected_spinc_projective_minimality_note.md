# Protected Spin^c Projective Minimality Note

## Scope

This note is part of the optional protected \(Spin^c\) bridge queue.  It does
not modify `paper/main.tex`, the baseline scan, figure generation, or the
submission A1--A5 wording.

The question is:

> Can the two-center projected parent force
> \(C\simeq\mathbb{CP}^1\), rather than simply assume it?

The conservative answer is:

> Yes, but only after an explicit projective-minimality principle is stated.
> Without that principle, the index count alone does not force
> \(\mathbb{CP}^1\).

Thus V3 is a conditional parent-side rigidity theorem, not an unconditional
upgrade of the current submission baseline.

## Minimality Principle

Let \((C,p_+,p_-)\) be the compact connected internal projection curve with two
distinguished center marks.  The V3 projective-minimality principle is:

1. the projected parent is connected and single-sheet,
2. the compactification is minimal among smooth projective curves carrying the
   two center marks,
3. no continuous internal curve modulus, Jacobian/Picard modulus, or flat
   Wilson-line modulus is allowed beyond the explicit PSLT scan variables.

The third condition is the essential one.  It says that the two-center parent is
not allowed to hide additional continuously tunable internal data that are not
present in the baseline state vector.

## Theorem

Under the V3 projective-minimality principle, the internal projection curve is

\[
C\simeq\mathbb{CP}^1.
\]

### Proof

Let \(g=g(C)\).  The moduli dimension of smooth genus-\(g\) curves with two
marked points is, for stable cases,

\[
\dim_{\mathbb C}{\mathcal M}_{g,2}=3g-3+2=3g-1
\qquad (g\ge 2),
\]

and for \(g=1\),

\[
\dim_{\mathbb C}{\mathcal M}_{1,2}=2.
\]

For \(g=0\) with two marked points the stable Deligne-Mumford formula is not
directly applicable, but the projective two-center problem is rigid: every
ordered pair of distinct points on \(\mathbb{CP}^1\) is sent to
\((0,\infty)\) by a Möbius transformation.  Hence there is no shape modulus:

\[
\dim_{\mathbb C}[(\mathbb{CP}^1)^2\setminus\Delta]/PGL(2,\mathbb C)=0.
\]

In addition, the degree-\(d\) Picard variety has dimension

\[
\dim_{\mathbb C}\mathrm{Pic}^d(C)=g.
\]

Thus every \(g\ge 1\) candidate carries at least one hidden continuous internal
modulus, either from the complex structure, the marked-point data, or the
Picard/Jacobian family of line bundles.  This violates the V3 minimality
principle.

Therefore \(g=0\).  By the classification of compact genus-zero Riemann
surfaces,

\[
C\simeq\mathbb{CP}^1.
\]

This proves the theorem.

## Why The Minimality Principle Is Necessary

Riemann-Roch on a compact Riemann surface gives

\[
\chi(C,L)=h^0(C,L)-h^1(C,L)=\deg L+1-g.
\]

If one only asks for a three-family index,

\[
\chi(C,L)=3,
\]

then

\[
\deg L=g+2.
\]

Consequently, infinitely many pairs

\[
(g,\deg L)=(0,2),(1,3),(2,4),(3,5),\ldots
\]

have the same formal \(Spin^c\) index.  The index count alone therefore cannot
force \(C=\mathbb{CP}^1\).

The V3 theorem uses a stronger physical selection rule: positive genus
introduces hidden internal moduli, and the current two-center PSLT parent has no
place for such moduli unless a new baseline family is opened.  Under no hidden
moduli, the only remaining compact connected curve is genus zero.

## Consequence For The Spin^c Bridge

V3 can justify the first input of the protected-index route:

\[
C=\mathbb{CP}^1.
\]

It does not yet prove the flux/line-bundle input

\[
R=p_+ + p_-,
\qquad
L_R=\mathcal O(R)\simeq\mathcal O(2).
\]

That is the separate V4 ramification-flux lock.  V3 also does not prove the
\(Spin^c\)-rather-than-spin carrier choice, nor does it exclude exotics; those
are V5 and V6.

The correct V3 verdict is therefore:

\[
\boxed{\text{conditional } C=\mathbb{CP}^1
\text{ from no-hidden-moduli projective minimality}.}
\]

It is not:

\[
\boxed{\text{unconditional EYMH derivation of } C=\mathbb{CP}^1.}
\]

## Executable Check

The companion gate

```bash
python3 code/check_spinc_projective_minimality.py
```

writes

```text
output/spinc_bridge/spinc_projective_minimality_summary.csv
output/spinc_bridge/spinc_projective_minimality_detail.csv
output/spinc_bridge/spinc_projective_minimality_detail.json
```

and verifies:

1. genus zero is the unique candidate with zero curve/Picard hidden moduli,
2. every tested positive genus has positive hidden-moduli dimension,
3. the index condition alone admits \(\deg L=g+2\) for every genus, so it is a
   no-go without minimality,
4. the genus-zero selected curve has the expected
   \(h^0(\mathbb{CP}^1,\mathcal O(2))=3\),
   \(h^1(\mathbb{CP}^1,\mathcal O(2))=0\) check used by V0.

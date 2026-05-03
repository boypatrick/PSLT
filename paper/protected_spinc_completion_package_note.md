# Protected Spin^c Conditional Completion Package

## Scope

This note is a synthesis of the optional protected \(Spin^c\) bridge queue
V0--V6.  It does not modify `paper/main.tex`, the baseline scan, figure
generation, or the submission A1--A5 wording.

The package is best read as:

\[
\boxed{
\text{conditional first-principles completion package}
}
\]

not as:

\[
\boxed{
\text{unconditional EYMH derivation used by the submission baseline}.
}
\]

## Package Statement

Under the following explicit parent-side inputs:

1. no-hidden-moduli projective minimality of the two-center projected internal
   curve,
2. faithful unit ramification/flux at the two projected centers,
3. projective/Kähler \(Spin^c\) family quantization,
4. complete vectorlike inventory and positive gap floor for all non-protected
   SM-charged sectors,

the protected internal family sector is

\[
C\simeq\mathbb{CP}^1,
\qquad
R=p_+ + p_-,
\qquad
L_R=\mathcal O(R)\simeq\mathcal O(2),
\]

with carrier

\[
S_{\rm gen}
=
\Lambda^{0,*}T^*C\otimes L_R.
\]

Then

\[
\ker D_{\rm gen}^+
\simeq
H^0(\mathbb{CP}^1,\mathcal O(2)),
\qquad
\ker D_{\rm gen}^-=0,
\]

and therefore

\[
\dim\ker D_{\rm gen}^+=3,
\qquad
\dim\ker D_{\rm gen}^-=0.
\]

All non-protected SM-charged sectors are gapped if they are exhausted by
vectorlike Serre-dual pairs with positive mass/Bochner floors.

## Gate Ledger

| Gate | Verdict | Meaning |
| --- | --- | --- |
| V0 | `CONDITIONAL_DIMENSION_FLAG_BRIDGE_ONLY` | Exact \(CP^1/O(2)\) index flag and the audited spectral rank flag have compatible dimensions. |
| V1 | `PROMOTABLE_ANCHOR_INTERTWINER` | A fixed tapered-polynomial anchor embeds the protected degree flag into the audited three-mode spectral subspace. |
| V2 | `NODAL_STURM_ORDER_CONFIRMED` | The spectral ordering matches the protected basis ordering by parity and nodal/Sturm count. |
| V3 | `CONDITIONAL_CP1_SELECTED_BY_NO_HIDDEN_MODULI` | No-hidden-moduli projective minimality selects \(C=\mathbb{CP}^1\); index \(3\) alone does not. |
| V4 | `CONDITIONAL_O2_LOCKED_BY_UNIT_TWO_CENTER_FLUX` | Faithful unit two-center flux locks \(R=p_+ + p_-\) and \(L_R=\mathcal O(2)\). |
| V5 | `CONDITIONAL_SPINC_CARRIER_SELECTED_OVER_ORDINARY_SPIN` | Dolbeault \(Spin^c\) gives three modes with \(O(2)\); ordinary spin with the same flux gives two. |
| V6 | `CONDITIONAL_NO_EXOTICS_WITH_COMPLETE_VECTORLIKE_GAP_CERTIFICATE` | Vectorlike Serre-dual pairs plus positive gap floors are sufficient to remove non-protected chiral exotics. |

## Submission Reading

This package strengthens the optional mechanism story, but it does not change
the submission claim.  The manuscript baseline remains a computable EFT-level
spectral-layer occupancy closure with first-three-layer concentration and
audited finite-domain no-fourth-bound-layer evidence.

The protected \(Spin^c\) package can be used in a revision or companion paper
as a conditional first-principles explanation of why a two-center projected
parent naturally supports three protected chiral families.  It should not be
used to claim, before the parent axioms are independently derived, that the
current EYMH baseline unconditionally proves exactly three Standard Model
generations.

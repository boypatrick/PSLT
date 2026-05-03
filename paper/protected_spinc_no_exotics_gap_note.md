# Protected Spin^c No-Exotics Gap Note

## Scope

This note is the V6 item in the optional protected \(Spin^c\) bridge queue.  It
does not modify `paper/main.tex`, the baseline scan, figure generation, or the
submission A1--A5 wording.

V3--V5 conditionally supply

\[
C\simeq\mathbb{CP}^1,
\qquad
L_R\simeq\mathcal O(2),
\qquad
S_{\rm gen}=\Lambda^{0,*}T^*C\otimes L_R.
\]

V6 asks whether all other Standard-Model-charged internal sectors can be
excluded as unpaired chiral exotics.

The conservative answer is:

> Yes, if the parent-side sector inventory is complete and every non-protected
> SM-charged sector is either zero-index and gapped, or appears in a vectorlike
> Serre-dual pair with a positive mass/gap floor.  This is a sufficient
> no-exotics certificate, not an unconditional derivation of the full parent
> inventory.

## Birkhoff-Grothendieck Reduction

On \(\mathbb{CP}^1\), every holomorphic vector bundle splits as a direct sum of
line bundles:

\[
E\simeq\bigoplus_i \mathcal O(n_i).
\]

For the Dolbeault \(Spin^c\) index,

\[
\chi(\mathcal O(n))
=h^0(\mathbb{CP}^1,\mathcal O(n))
-h^1(\mathbb{CP}^1,\mathcal O(n))
=n+1.
\]

Therefore a single unpaired SM-charged summand \(\mathcal O(n)\) would
generically carry chiral index \(n+1\).  Unless it is the protected
\(\mathcal O(2)\) family sector, or unless its index vanishes and it is
separately gapped, it is an exotic.

## Vectorlike Serre-Dual Pairing

The natural vectorlike partner of a line summand \(F=\mathcal O(n)\) is

\[
K_C\otimes F^*
=
\mathcal O(-2)\otimes\mathcal O(-n)
=
\mathcal O(-n-2).
\]

Its index is

\[
\chi(\mathcal O(-n-2))
=(-n-2)+1
=-n-1.
\]

Thus the pair has zero net chiral index:

\[
\chi(\mathcal O(n))
+
\chi(\mathcal O(-n-2))
=(n+1)+(-n-1)=0.
\]

This is the algebraic no-exotics mechanism: non-protected SM-charged sectors
must appear in these vectorlike pairs, not as isolated summands.

## Gap Mechanism

Let \(D_F\) be the internal Dirac/Dolbeault operator on a non-protected pair.
Assume a gauge-invariant mass pairing

\[
M_F:
H^0(C,F)
\longleftrightarrow
H^1(C,K\otimes F^*)^*
\]

with singular-value floor

\[
\sigma_{\min}(M_F)\ge m_*>0.
\]

On the orthogonal complement of the paired zero-mode sector, assume a
Bochner/Lichnerowicz lower bound

\[
D_F^2
=
\nabla_F^*\nabla_F+\mathcal R_F+\Phi_F^\dagger\Phi_F
\ge \lambda_* >0.
\]

Then the full non-protected sector has gap

\[
\Delta_F
\ge
\min(m_*,\sqrt{\lambda_*})>0,
\]

up to the convention for whether the certificate reports the Dirac gap or the
squared-operator gap.  The executable V6 gate uses a positive abstract floor and
therefore checks the certificate structure, not a physical mass calibration.

## Why This Is Still Conditional

V6 is not allowed to assert that the current EYMH parent has no other charged
fields unless one also proves a complete sector inventory theorem:

\[
\mathcal H_{\rm SM\ charged}
=
\mathcal H_{\rm protected}
\oplus
\bigoplus_i
\left(
F_i\oplus K_C\otimes F_i^*
\right).
\]

The V6 certificate proves that this form is sufficient.  It does not, by
itself, prove that the parent contains no additional SM-charged sector outside
the listed pairs.

## V6 Verdict

The correct V6 statement is

\[
\boxed{
\text{complete vectorlike sector inventory}
+
\text{positive mass/Bochner floors}
\Longrightarrow
\text{no unpaired SM-charged exotics}.
}
\]

It is not

\[
\boxed{
\text{the current submission baseline unconditionally excludes all exotics}.
}
\]

This is nevertheless the right theorem shape: it identifies exactly what a
future parent-side completion must certify.

## Executable Check

The companion gate

```bash
python3 code/check_spinc_no_exotics_gap.py
```

writes

```text
output/spinc_bridge/spinc_no_exotics_gap_summary.csv
output/spinc_bridge/spinc_no_exotics_gap_detail.csv
output/spinc_bridge/spinc_no_exotics_gap_detail.json
```

and verifies on a finite degree window:

1. every tested \(\mathcal O(n)\) has the Serre-dual vectorlike partner
   \(\mathcal O(-n-2)\),
2. every pair has zero net chiral index,
3. the assumed mass/Bochner floors are positive,
4. unpaired nonzero-index summands would be chiral exotics,
5. the protected \(\mathcal O(2)\) sector is explicitly excluded from the
   non-protected gap certificate.

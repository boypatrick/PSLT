# Spectral Projector Transport / D_sep Coherence Note

Status: T1 closed as a conditional, non-baseline finite-grid support theorem.

This note records the first pass at a spectral-projector transport/coherence
theorem for the PSLT low modes `N=1,2,3` along the two-center separation
parameter `D_sep`.  It does not modify `paper/main.tex`, does not change the
release scan, and does not promote the optional protected `Spin^c` bridge into
the submission baseline.

The scientific purpose is narrow:

```text
If the first three finite-volume spectral projectors remain separated by open
spectral gaps as D_sep varies, can the labels N=1,2,3 be transported coherently
instead of being treated as unrelated pointwise eigenmodes?
```

The answer is yes under the standard Kato/Riesz hypotheses.  The current G1
artifacts provide finite-grid support for those hypotheses on the audited
`D_sep` grid.

## T0: Claim Firewall

The theorem below is conditional.  It requires an analytic operator-family
input:

1. a self-adjoint finite-volume operator family `H(D)` or a self-adjoint
   representative of a generalized eigenproblem,
2. `C^1` dependence on `D` in norm-resolvent sense on the interval under study,
3. an open spectral gap isolating each of the first three eigenvalues or the
   cumulative first-three cluster.

The executable T1 gate checks only the existing finite-grid evidence:
strict low-mode gaps, exact cumulative ranks, Riesz-contour radius floors, and
adjacent-knot displacement/gap ratios.  It does not by itself prove the
continuum-in-`D` theorem between grid knots.

## Operator Setting

The G1 finite-volume artifact can be read as a generalized self-adjoint
eigenproblem

\[
K(D) u = \lambda M(D) u, \qquad M(D)>0.
\]

Equivalently, after conjugation by `M(D)^{1/2}`, define

\[
H(D) = M(D)^{-1/2}K(D)M(D)^{-1/2}.
\]

Then `H(D)` is self-adjoint on the finite-volume Hilbert space and has real
ordered eigenvalues

\[
\lambda_1(D) < \lambda_2(D) < \lambda_3(D) < \lambda_4(D) \le \cdots
\]

on the audited grid.  For a single mode define the nearest spectral gap

\[
\gamma_n(D)
=
\operatorname{dist}\!\left(\lambda_n(D),
\sigma(H(D))\setminus\{\lambda_n(D)\}\right),
\qquad n=1,2,3.
\]

For the cumulative first-three cluster define

\[
\gamma_{1:3}(D)
=
\lambda_4(D)-\lambda_3(D),
\]

assuming the cluster is separated from the rest of the spectrum.

## Theorem T1: Riesz Projector Transport

Let `I=[D_-,D_+]` be a compact `D_sep` interval.  Assume:

1. `D -> H(D)` is `C^1` in norm-resolvent sense on `I`;
2. for each `n=1,2,3`, `lambda_n(D)` is simple on `I`;
3. there is a uniform gap floor

\[
\gamma_* = \min_{D\in I}\min_{n=1,2,3}\gamma_n(D)>0.
\]

Then the Riesz projectors

\[
P_n(D)
=
\frac{1}{2\pi i}\oint_{\Gamma_n(D)}(z-H(D))^{-1}\,dz,
\qquad n=1,2,3,
\]

where `Gamma_n(D)` is any positively oriented circle around `lambda_n(D)` with
radius

\[
r_n(D)=\alpha\gamma_n(D),\qquad 0<\alpha<\frac12,
\]

are `C^1` in `D`.  They give three continuously transported spectral lines

\[
L_n(D)=\operatorname{Ran}P_n(D),\qquad n=1,2,3,
\]

and therefore coherent labels `N=1,2,3` throughout `I`.  No level crossing or
label swap can occur inside `I` unless the gap floor closes.

### Proof

For `z` on `Gamma_n(D)`, self-adjointness gives the resolvent estimate

\[
\|(z-H(D))^{-1}\|
\le
\frac{1}{\operatorname{dist}(z,\sigma(H(D)))}
=
\frac{1}{r_n(D)}.
\]

Because `r_n(D)<gamma_n(D)/2`, the contour encloses only `lambda_n(D)` and no
other spectral point.  Hence the Riesz formula defines the rank-one spectral
projector onto the `n`th eigenspace.

Let

\[
R(z,D)=(z-H(D))^{-1}.
\]

Differentiating the identity `(z-H(D))R(z,D)=1` gives

\[
-H'(D)R(z,D)+(z-H(D))\partial_D R(z,D)=0,
\]

so

\[
\partial_D R(z,D)=R(z,D)H'(D)R(z,D).
\]

Therefore

\[
P_n'(D)
=
\frac{1}{2\pi i}\oint_{\Gamma_n(D)}
R(z,D)H'(D)R(z,D)\,dz.
\]

Taking norms and using `length(Gamma_n)=2 pi r_n`,

\[
\|P_n'(D)\|
\le
\frac{1}{2\pi}(2\pi r_n(D))
\frac{\|H'(D)\|}{r_n(D)^2}
=
\frac{\|H'(D)\|}{r_n(D)}
=
\frac{\|H'(D)\|}{\alpha\gamma_n(D)}.
\]

A uniform gap floor and bounded `H'(D)` therefore give a uniform projector
variation bound

\[
\|P_n'(D)\|
\le
\frac{\|H'(D)\|}{\alpha\gamma_*}.
\]

Since `I` is contractible, each rank-one eigenbundle `L_n(D)` is trivial.
Choosing a normalized section at one base point and transporting phases by the
parallel gauge

\[
\langle u_n(D),\partial_D u_n(D)\rangle=0
\]

gives a continuous, phase-fixed eigenvector branch.  The projector itself is
phase-independent.  This proves the coherent transport of `N=1,2,3` labels.

## Cumulative First-Three Cluster

The same argument applies to the cumulative projector

\[
P_{1:3}(D)=\sum_{n=1}^3 P_n(D)
=
\frac{1}{2\pi i}\oint_{\Gamma_{1:3}(D)}(z-H(D))^{-1}\,dz,
\]

where `Gamma_{1:3}(D)` encloses `lambda_1,lambda_2,lambda_3` and excludes
`lambda_4`.  If

\[
\gamma_{1:3,*}=\min_{D\in I}\bigl(\lambda_4(D)-\lambda_3(D)\bigr)>0,
\]

then the three-dimensional low-mode subspace

\[
E_3(D)=\operatorname{Ran}P_{1:3}(D)
\]

is also transported coherently.  This is the correct object when only the
three-dimensional low-mode sector matters and internal rotations among
`N=1,2,3` are harmless.  The individual projectors require the stronger
internal gaps `lambda_2-lambda_1` and `lambda_3-lambda_2` to stay open.

## Discrete Audit Gate

The executable check is deliberately narrower than the theorem:

```bash
python3 code/check_spectral_projector_transport.py
```

It consumes the existing G1 low-mode artifacts:

```text
output/gn_fp_2d/gn_phase_space_2d_spectrum_Dgrid21.csv
output/gn_fp_2d/gn_lowN_microstate_count_detail.csv
```

and writes:

```text
output/spectral_transport/spectral_projector_transport_summary.csv
output/spectral_transport/spectral_projector_transport_detail.csv
output/spectral_transport/spectral_projector_transport_steps.csv
output/spectral_transport/spectral_projector_transport_detail.json
```

The current verdict is

```text
CONDITIONAL_PROJECTOR_TRANSPORT_COHERENCE_SUPPORTED_ON_G1_GRID
```

with summary values

\[
\min_D\gamma_1(D)=0.041644263548268556,
\]

\[
\min_D\gamma_2(D)=0.041644263548268556,
\]

\[
\min_D\gamma_3(D)=0.04968688976205282.
\]

Using `alpha=0.45`, the minimum Riesz contour radii are

\[
\min_D r_1(D)=0.01873991859672085,
\]

\[
\min_D r_2(D)=0.01873991859672085,
\]

\[
\min_D r_3(D)=0.02235910039292377.
\]

The worst adjacent-knot displacement/gap ratios are

\[
\max \frac{|\lambda_1(D_{j+1})-\lambda_1(D_j)|}
{\min(\gamma_1(D_j),\gamma_1(D_{j+1}))}
=0.0757806277234115,
\]

\[
\max \frac{|\lambda_2(D_{j+1})-\lambda_2(D_j)|}
{\min(\gamma_2(D_j),\gamma_2(D_{j+1}))}
=0.15091229039053536,
\]

\[
\max \frac{|\lambda_3(D_{j+1})-\lambda_3(D_j)|}
{\min(\gamma_3(D_j),\gamma_3(D_{j+1}))}
=0.21451494180792025.
\]

All are below the conservative diagnostic tolerance `0.25`.  The worst case is
`N=3` on the first interval `D_sep=4.0 -> 4.8`, which is the expected place for
largest motion because the low-mode branch moves fastest near the left edge of
the audited window.

Finite-difference slope diagnostics are

\[
\max|\Delta\lambda_1/\Delta D|=0.00877830064419355,
\]

\[
\max|\Delta\lambda_2/\Delta D|=0.017481426266184045,
\]

\[
\max|\Delta\lambda_3/\Delta D|=0.04489758659062718.
\]

The cumulative total variations over the audited grid are

\[
\operatorname{TV}(\lambda_1)=0.07219682942650563,
\]

\[
\operatorname{TV}(\lambda_2)=0.13018572279987595,
\]

\[
\operatorname{TV}(\lambda_3)=0.26987030413167257.
\]

## Interpretation

T1 supports the following reviewer-facing statement, if this optional line is
needed later:

```text
On the audited G1 finite-volume grid, the N=1,2,3 low-mode projectors form a
coherent separated spectral flag along D_sep.  Under the standard additional
analytic assumption that the finite-volume operator family is C1 in
norm-resolvent sense with no gap closure between audited knots, Kato/Riesz
projector transport gives continuous labels N=1,2,3 over the full interval.
```

It does not support these stronger statements by itself:

- a continuum no-crossing theorem without an analytic inter-knot gap bound;
- a parent EYMH derivation of `H(D)`;
- a proof of exactly three SM generations;
- a replacement of the A1--A5 submission wording;
- any retuning of `g_N`, `Gamma_N`, `B_N`, or `H->mumu`.

## Suggested Next Gates

1. **T2: analytic derivative envelope.**  Bound `||H'(D)||` from the explicit
   `D_sep` derivative of the two-center conformal factor and finite-volume
   operator assembly.  This is now closed below as a verified but loose
   Plummer-core sup-norm envelope, not as an inter-knot no-crossing theorem.

2. **T3a: weighted Hellmann-Feynman transport.**  Use the actual low-mode
   matrix elements `A_mn=<u_m,H'u_n>` to avoid the Plummer-core sup-norm spike.
   This is now closed below as the first useful finite-grid projector
   derivative bound.

3. **T3b: Davis-Kahan / sin-theta step transport.**  Use the resolvent gap and
   a certified `||H(D_{j+1})-H(D_j)||` envelope to bound projector drift between
   adjacent knots.  This is the next useful route because T2's sup-norm bound
   overweights the derivative spike near the regularized cores:

\[
\|P_n(D_{j+1})-P_n(D_j)\|
\lesssim
\frac{\|H(D_{j+1})-H(D_j)\|}{\operatorname{gap}_{j,n}}.
\]

4. **T4: bridge to the optional `Spin^c` anchors.**  Combine T1 with the V1/V2
   anchor-overlap and nodal/Sturm certificates.  The goal would be a single
   conditional statement: the protected degree flag and the finite-volume
   spectral flag remain coherently identified along `D_sep`.

5. **Do not change the submission baseline.**  T1--T3a are a strengthening of the
   optional mechanism narrative, not a reason to reopen family counting,
   visibility, static width, or DSS parent construction.

## T2: Analytic `||H'(D)||` Envelope

T2 asks whether the explicit two-center conformal factor gives a direct bound
on the operator derivative that appears in the Riesz projector estimate.

### Fixed-box operator derivative

On a fixed finite-volume box, or after a pullback to a fixed reference box with
kinetic/boundary terms treated separately, the localized G1 operator has the
form

\[
K(D)=K_0+\operatorname{diag}(\rho\,U(D)),
\qquad
M=\operatorname{diag}(\rho),
\]

where

\[
U(D)=m_0^2(\Omega(D)^2-1)
+(1-6\xi){\Delta\Omega(D)\over \Omega(D)}.
\]

The self-adjoint representative is

\[
H(D)=M^{-1/2}K(D)M^{-1/2}.
\]

Since `M` and `K_0` are independent of `D` on a fixed box,

\[
H'(D)
=M^{-1/2}\operatorname{diag}(\rho\,\partial_D U(D))M^{-1/2}
=\operatorname{diag}(\partial_D U(D)).
\]

Therefore

\[
\boxed{
\|H'(D)\|_{\ell^2}
\le
\|\partial_D U(D)\|_{\ell^\infty}
}
\]

on the finite grid.  In the continuum multiplication-operator reading, the same
bound is

\[
\|H'(D)\|
\le
\sup_{(\rho,z)\in\mathcal B}|\partial_D U(\rho,z;D)|.
\]

The literal G1 artifact uses `z_max=D/2+6`, so its matrix dimension changes
with `D`.  T2 therefore certifies the fixed-box/pullback potential part of the
operator derivative.  A fully literal variable-box theorem would also need the
coordinate-pullback kinetic and boundary terms.

### Explicit derivative of the two-center source

Set

\[
q_+=z-{D\over2},
\qquad
q_-=z+{D\over2},
\]

\[
s_+=\rho^2+q_+^2+\varepsilon^2,
\qquad
s_-=\rho^2+q_-^2+\varepsilon^2.
\]

The two-center conformal factor and its Laplacian are

\[
\Omega
=1+a(s_+^{-1/2}+s_-^{-1/2}),
\]

\[
\Delta\Omega
=-3a\varepsilon^2(s_+^{-5/2}+s_-^{-5/2}).
\]

Differentiating with respect to `D` gives

\[
\partial_D\Omega
={a\over2}\left(q_+s_+^{-3/2}-q_-s_-^{-3/2}\right),
\]

\[
\partial_D\Delta\Omega
=-{15\over2}a\varepsilon^2
\left(q_+s_+^{-7/2}-q_-s_-^{-7/2}\right).
\]

With

\[
c=1-6\xi,
\]

the potential derivative is

\[
\boxed{
\partial_D U
=2m_0^2\Omega\,\partial_D\Omega
+c{(\partial_D\Delta\Omega)\Omega-(\Delta\Omega)(\partial_D\Omega)
\over \Omega^2}
}.
\]

This is the analytic object audited by the T2 script.

### Closed-form global envelope

The elementary one-dimensional maximizers give

\[
\sup_x {|x|\over(x^2+\varepsilon^2)^{3/2}}
={2\over3\sqrt3\,\varepsilon^2},
\]

\[
\sup_x {|x|\over(x^2+\varepsilon^2)^{7/2}}
={6^3\over7^{7/2}\varepsilon^6}.
\]

Hence

\[
1\le \Omega\le 1+{2a\over\varepsilon},
\]

\[
\|\partial_D\Omega\|_\infty
\le
{2a\over3\sqrt3\,\varepsilon^2},
\]

\[
\|\Delta\Omega\|_\infty
\le
{6a\over\varepsilon^3},
\]

\[
\|\partial_D\Delta\Omega\|_\infty
\le
{3240a\over7^{7/2}\varepsilon^4}.
\]

Therefore

\[
\boxed{
\|H'(D)\|
\le
2m_0^2\Omega_{\max}\Omega_{D,\max}
+|c|\left(
(\Delta\Omega)_{D,\max}\Omega_{\max}
+(\Delta\Omega)_{\max}\Omega_{D,\max}
\right)
}
\]

with `Omega_min=1` already used in the denominator.  This bound is independent
of `D`, the finite-volume box, and the grid resolution; it is intentionally a
worst-case Plummer-core bound.

For the current G1 parameters

\[
a=0.04,
\quad
\varepsilon=0.1,
\quad
m_0=1,
\quad
\xi=0.14,
\]

the closed-form envelope is

\[
\|H'(D)\|\le 475.9591007665789.
\]

The finer fixed-box grid evaluation gives a much smaller but still conservative
finite-grid envelope

\[
\max_{D\in[4,20]}\|\partial_D U(D)\|_{\ell^\infty}
=59.958927014358956.
\]

The analytic derivative formula is validated by centered finite differences:

\[
\max |\partial_D U_{\rm analytic}-\partial_D U_{\rm fd}|
=3.0382482897550744\times10^{-5},
\]

\[
\max { |\partial_D U_{\rm analytic}-\partial_D U_{\rm fd}| 
\over \|\partial_D U\|_{\ell^\infty}}
=5.246550306822992\times10^{-7}.
\]

### Consequence for Riesz transport

Combining T1 and T2 gives the formal projector derivative bound

\[
\|P_n'(D)\|
\le
{\|H'(D)\|\over \alpha\gamma_n(D)}.
\]

Using the grid-resolved T2 envelope and the T1 gap floors gives

\[
\|P_1'\|\le3199.529747415806,
\]

\[
\|P_2'\|\le3199.529747415806,
\]

\[
\|P_3'\|\le2681.634142728517.
\]

These numbers are mathematically valid but too loose to prove inter-knot
coherence alone.  The observed eigenvalue slopes are tiny relative to the
sup-norm envelope:

\[
{\max|\Delta\lambda_1/\Delta D|\over\max\|\partial_DU\|_\infty}
=1.4640523240336294\times10^{-4},
\]

\[
{\max|\Delta\lambda_2/\Delta D|\over\max\|\partial_DU\|_\infty}
=2.91556689498422\times10^{-4},
\]

\[
{\max|\Delta\lambda_3/\Delta D|\over\max\|\partial_DU\|_\infty}
=7.488057046096744\times10^{-4}.
\]

Thus T2 closes as a verified analytic `H'(D)` envelope, but not as a standalone
no-crossing theorem.  The core-local singular envelope is simply too coarse for
projector transport.  The next useful step is T3: a Davis-Kahan or weighted
low-mode matrix-element drift bound, which should use the fact that the first
three eigenfunctions have very small weight in the worst Plummer-core derivative
region.

Executable gate:

```bash
python3 code/check_spectral_projector_derivative_envelope.py
```

Current verdict:

```text
T2_ANALYTIC_HPRIME_ENVELOPE_VERIFIED_SUPNORM_LOOSE
```

Artifacts:

```text
output/spectral_transport/spectral_projector_derivative_envelope_summary.csv
output/spectral_transport/spectral_projector_derivative_envelope_detail.csv
output/spectral_transport/spectral_projector_derivative_envelope_detail.json
```

## T3a: Weighted Hellmann-Feynman Matrix-Element Bound

T2 gives a rigorous sup-norm envelope for `||H'(D)||`, but it is too loose
because the Plummer-core derivative spike dominates `||partial_D U||_infty`.
T3a therefore uses the low-mode weights directly.

Let the finite-volume eigenvectors be `M`-orthonormal:

\[
K(D)u_n(D)=\lambda_n(D)M(D)u_n(D),
\qquad
u_m^T M u_n=\delta_{mn}.
\]

In the fixed-box/pullback reading, the derivative operator is multiplication by
`partial_D U`.  Hence its matrix elements in the generalized eigenbasis are

\[
A_{mn}(D)
=\langle u_m,H'(D)u_n\rangle
=u_m^T\operatorname{diag}(\rho\,\partial_DU)u_n.
\]

The diagonal Hellmann-Feynman identity is

\[
\lambda_n'(D)=A_{nn}(D).
\]

For the eigenvector derivative, choose the parallel gauge

\[
\langle u_n,\partial_Du_n\rangle=0.
\]

Differentiating the eigenvalue equation and projecting onto `u_m`, `m\ne n`,
gives

\[
\langle u_m,\partial_Du_n\rangle
={A_{mn}\over \lambda_n-\lambda_m}.
\]

Thus

\[
\partial_Du_n
=\sum_{m\ne n}{A_{mn}\over\lambda_n-\lambda_m}u_m,
\]

and the rank-one projector derivative satisfies

\[
P_n'=|\partial_Du_n\rangle\langle u_n|
+|u_n\rangle\langle\partial_Du_n|.
\]

Since the two off-diagonal blocks have singular value
`||partial_D u_n||`,

\[
\boxed{
\|P_n'(D)\|
=
\left(\sum_{m\ne n}{|A_{mn}(D)|^2\over
|\lambda_m(D)-\lambda_n(D)|^2}\right)^{1/2}
}
\]

when the full spectral basis is available.

### Computed-mode plus tail certificate

The script computes the first `K=40` Ritz modes, uses modes `1..39` explicitly,
and reserves mode `40` as the lower spectral floor for the unexpanded tail.  The
unresolved coupling is controlled by the variance identity

\[
\sum_{m>K_*}|A_{mn}|^2
\le
\langle u_n,(H')^2u_n\rangle
-
\sum_{m\le K_*}|A_{mn}|^2,
\]

where `K_*=39` in the current audit.  Therefore

\[
\|P_n'\|
\le
\left(\sum_{m\le K_*,m\ne n}{|A_{mn}|^2\over
|\lambda_m-\lambda_n|^2}\right)^{1/2}
+
{\left(\langle u_n,(H')^2u_n\rangle
-
\sum_{m\le K_*}|A_{mn}|^2\right)_+^{1/2}
\over |\lambda_{40}-\lambda_n|}.
\]

This is conservative: the 40th mode is not silently discarded; it is included in
the residual tail and the same `lambda_40-lambda_n` denominator is used for all
unexpanded modes.

### Executable result

Run:

```bash
python3 code/check_spectral_projector_hf_weighted_bound.py
```

The current verdict is

```text
T3A_WEIGHTED_HF_PROJECTOR_BOUND_PASS
```

with summary values

\[
\max_{D,n}|A_{nn}|=0.003779455755604381,
\]

\[
\max_{D,n}\langle u_n,(H')^2u_n\rangle^{1/2}
=0.3890159737257284,
\]

\[
\max_{D,n}\|P_n'(D)\|_{\rm included}
=0.02785252488812737,
\]

\[
\max_{D,n}\|P_n'(D)\|_{\rm tail}
=0.057494130960578885,
\]

and therefore

\[
\boxed{
\max_{D,n}\|P_n'(D)\|
\le
0.08409504588944938.
}
\]

On the audited grid spacing `Delta D=0.8`, this gives the local integrated
projector-drift estimate

\[
\Delta D\,\max\|P_n'\|
\le
0.06727603671155945.
\]

This is below the T3a diagnostic tolerance `0.25`.  It also shows why T2 was too
coarse: the fixed-grid sup-norm envelope was

\[
\|\partial_DU\|_\infty=59.958927014358956,
\]

whereas the largest low-mode weighted RMS is only

\[
0.3890159737257284,
\]

a reduction factor of about `154.13`.  Relative to the T2 projector derivative
bound for `N=3`, the weighted bound improves the estimate by about
`3.19e4`.

The recomputed eigenvalues match the canonical G1 spectrum with

\[
\max|\lambda_{\rm recomputed}-\lambda_{\rm G1}|
=2.0892176877396196\times10^{-12}.
\]

### Interpretation

T3a is the first genuinely useful inter-knot control certificate for this line.
It still remains a finite-grid, fixed-box/pullback-style certificate: a full
continuum statement would need either a fixed-box rerun over a denser `D` mesh,
or a coordinate-pullback treatment of the G1 `z_max=D/2+6` variable box.  But it
strongly supports the physical point that the `N=1,2,3` low modes do not see the
Plummer-core derivative spike that dominated the T2 sup-norm estimate.

Artifacts:

```text
output/spectral_transport/spectral_projector_hf_weighted_bound_summary.csv
output/spectral_transport/spectral_projector_hf_weighted_bound_detail.csv
output/spectral_transport/spectral_projector_hf_weighted_bound_matrix.csv
output/spectral_transport/spectral_projector_hf_weighted_bound_detail.json
```

## T3b: Fixed-Box / Davis-Kahan Step Certificate

T3b addresses the remaining Hilbert-space issue in T3a.  The literal G1
artifact uses

\[
z_{\max}(D)=D/2+6,
\]

so adjacent `D` values do not have the same number of axial grid points.  T3b
therefore performs a fixed-box pullback surrogate: every audited `D` is solved
on the common box

\[
z_{\max}^{\rm fixed}=20/2+6=16.
\]

On this fixed box, `M` and the kinetic finite-difference matrix are
`D`-independent; only the multiplication potential changes.  Projectors at
adjacent `D` values can therefore be compared directly in one finite-dimensional
Hilbert space.

### Exact fixed-box projector distance

For two `M`-normalized rank-one eigenvectors `u,v`, the operator-norm distance
between projectors is exact:

\[
\|uu^*-vv^*\|
=
\sqrt{1-|\langle u,v\rangle_M|^2}.
\]

For the three-dimensional low-mode cluster, let `U_0,U_1` be `M`-orthonormal
bases for the first-three subspaces at adjacent `D` values.  The principal-angle
formula gives

\[
\|P_{1:3}(D_1)-P_{1:3}(D_0)\|
=
\sqrt{1-\sigma_{\min}(U_0^*MU_1)^2}.
\]

These are exact finite-dimensional identities, not perturbative estimates.

### Davis-Kahan residual comparator

T3b also records a raw a posteriori Davis-Kahan comparator.  If `v` is a unit
trial vector for `H(D_1)`, `mu` is its Rayleigh quotient, and

\[
r=(H(D_1)-\mu)v,
\]

then for the target isolated eigenline

\[
\sin\theta
\le
{\|r\|\over
\operatorname{dist}(\mu,\sigma(H(D_1))\setminus\{\lambda_n(D_1)\})}.
\]

Using the untransported `u_n(D_0)` as the trial vector gives a legitimate but
very conservative bound.  It still sees core/tail residual components and is
not expected to match the actual low-mode projector angle.  The useful strict
step evidence is therefore the exact projector overlap plus the T3a integrated
path bound, not the raw residual bound alone.

### Executable result

Run:

```bash
python3 code/check_spectral_projector_davis_kahan_step.py
```

The current verdict is

```text
T3B_FIXED_BOX_PROJECTOR_STEP_PASS_RAW_DK_LOOSE
```

with fixed-box summary values

\[
\max_{j,n}\|P_n(D_{j+1})-P_n(D_j)\|
=0.041167738779789895,
\]

\[
\max_j\|P_{1:3}(D_{j+1})-P_{1:3}(D_j)\|
=0.028715419399639706,
\]

and the T3a path bound gives

\[
\max_{j,n}\Delta D_j\max\{\|P_n'(D_j)\|,\|P_n'(D_{j+1})\|\}
=0.0672760367115592.
\]

All three are below the adopted diagnostic tolerances

\[
0.05,
\qquad
0.075,
\qquad
0.25.
\]

The raw residual Davis-Kahan comparator is conservative:

\[
\max {\|r\|\over \operatorname{gap}_{\rm raw}}
=3.680697023022397,
\]

so T3b is not recorded as a strict raw-residual Davis-Kahan pass.  The correct
reading is:

```text
actual fixed-box projector drift passes; T3a path bound passes; raw residual
Davis-Kahan is too loose as a standalone proof.
```

The fixed-box surrogate shifts eigenvalues relative to the original variable-box
G1 artifact, with

\[
\max_{D,n\le3}|\lambda_n^{\rm fixed\ box}(D)-\lambda_n^{\rm G1}(D)|
=0.26734335884564975.
\]

This is why T3b is not promoted as a literal replacement for the G1 spectrum.
It is a coordinate/pullback-style step-coherence certificate on a common Hilbert
space.  A literal G1 theorem would need a full coordinate-pullback treatment of
the `z_max=D/2+6` box, including axial kinetic and boundary terms.

Artifacts:

```text
output/spectral_transport/spectral_projector_davis_kahan_step_summary.csv
output/spectral_transport/spectral_projector_davis_kahan_step_detail.csv
output/spectral_transport/spectral_projector_davis_kahan_step_cluster.csv
output/spectral_transport/spectral_projector_davis_kahan_step_detail.json
```

## T3c: Literal Variable-Box Coordinate Pullback

T3c removes the fixed-physical-box caveat from T3b.  The literal G1 boxes are

\[
z\in[-L(D),L(D)],
\qquad
L(D)=D/2+6.
\]

Set

\[
z=L(D)y,
\qquad
-1\le y\le1.
\]

For a pulled-back function

\[
\psi(\rho,z)=\phi(\rho,y),
\]

the derivatives transform as

\[
\partial_z\psi={1\over L(D)}\partial_y\phi.
\]

The quadratic form becomes

\[
\int \rho\,d\rho\,dz
\left(|\partial_\rho\psi|^2+|\partial_z\psi|^2+U(\rho,z;D)|\psi|^2\right)
\]

\[
=
L(D)\int \rho\,d\rho\,dy
\left(|\partial_\rho\phi|^2
+{1\over L(D)^2}|\partial_y\phi|^2
+U(\rho,L(D)y;D)|\phi|^2\right).
\]

The mass form is

\[
\int \rho\,d\rho\,dz\,|\psi|^2
=
L(D)\int \rho\,d\rho\,dy\,|\phi|^2.
\]

After cancelling the common factor `L(D)` in the generalized eigenproblem, the
pullback operator on the fixed reference Hilbert space is

\[
\boxed{
K_{\rm pull}(D)
=K_\rho+L(D)^{-2}K_y
+\operatorname{diag}\!\left(\rho\,U(\rho,L(D)y;D)\right),
\qquad
M_{\rm pull}=\operatorname{diag}(\rho).
}
\]

This formula explicitly includes the axial kinetic scaling, the measure, the
potential, and the moving Dirichlet boundary as a fixed `y=\pm1` boundary.

### Executable result

Run:

```bash
python3 code/check_spectral_projector_pullback_variable_box.py
```

The current verdict is

```text
T3C_VARIABLE_BOX_PULLBACK_PROJECTOR_STEP_PASS_RAW_DK_LOOSE
```

with reference-grid parameters

\[
N_y=1067,
\qquad
\Delta y=0.0018744142455482662,
\]

so the physical axial spacing ranges from

\[
\Delta z_{\min}=0.01499531396438613
\]

at `D=4` to

\[
\Delta z_{\max}=0.02999062792877226
\]

at `D=20`, matching the G1 target spacing at the largest box and refining the
smaller boxes.

The exact pullback projector drifts are

\[
\max_{j,n}\|P_n(D_{j+1})-P_n(D_j)\|
=0.013928181522331775,
\]

and

\[
\max_j\|P_{1:3}(D_{j+1})-P_{1:3}(D_j)\|
=0.011846382262997952.
\]

Both are comfortably below the adopted tolerances `0.05` and `0.075`.  The
pullback eigenvalues also remain close to the canonical G1 variable-box low
modes:

\[
\max_{D,n\le3}|\lambda_n^{\rm pullback}(D)-\lambda_n^{\rm G1}(D)|
=0.0008025737978598624.
\]

The raw residual Davis-Kahan comparator remains conservative,

\[
\max {\|r\|\over\operatorname{gap}_{\rm raw}}
=1.2324618065225783,
\]

so T3c is not recorded as a strict residual-Davis-Kahan pass.  The meaningful
certificate is stronger and more direct: after pulling the literal variable box
to a fixed coordinate, the actual adjacent spectral projectors and the actual
first-three spectral cluster move only mildly.

Artifacts:

```text
output/spectral_transport/spectral_projector_pullback_variable_box_summary.csv
output/spectral_transport/spectral_projector_pullback_variable_box_detail.csv
output/spectral_transport/spectral_projector_pullback_variable_box_cluster.csv
output/spectral_transport/spectral_projector_pullback_variable_box_detail.json
```

### Interpretation

T3c is the strongest current version of the projector-coherence package.  It
keeps the literal G1 box law `L(D)=D/2+6`, includes the induced axial kinetic
factor `L(D)^{-2}`, keeps the pulled-back measure fixed, and compares all
projectors in one reference Hilbert space.  It therefore supersedes T3b as the
preferred step-coherence certificate, while preserving the same claim firewall:
no baseline scan retuning, no new family, and no replacement of the submission
A1--A5 wording.

# Gap 3 Derivation Note: Linear Excess Variables as Zero-Second-Jet Normal Coordinates

## Aim

Convert the exact coordinate-selection audits into a parent-action-side statement:

> once the projected EYMH parent kernel has been reduced to its positive
> background-normalized \(2\times2\) form and its mixed entry has been fixed to
> the geometric mean of the two diagonal excess sectors, the natural local
> coordinates are exactly the anchored linear excess variables with zero second
> jet.

This note closes the conceptual gap labeled `derive_linear_excess_coordinates`
in
`/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/eymh_parent_kernel_bridge_status_gaps.csv`.

The point is not to invent a new coordinate family. The point is to show that
the current coordinates already behave like normal-form coordinates for the
localized projected parent block.

## Starting Point: What Lemmas 1 and 2 Already Fix

After Gap 1 and Gap 2, the projected parent block is already organized as

\[
\mathcal K_{\rm parent}
=
\begin{pmatrix}
K_{11} & \sqrt{(K_{11}-1)(K_{22}-K_{\rm bg})}\\
\sqrt{(K_{11}-1)(K_{22}-K_{\rm bg})} & K_{22}
\end{pmatrix},
\qquad
K_{\rm bg}=1+\mathrm{gap}_{\rm cv}.
\]

The only remaining ambiguity is therefore the **choice of local coordinates**
used to describe the deviations away from the fixed points

\[
K_{11}=1,
\qquad
K_{22}=K_{\rm bg}.
\]

Define the canonical excess variables

\[
E_{\rm part}=K_{11}-1,
\qquad
E_{\rm tree}=K_{22}-K_{\rm bg}.
\]

Gap 3 asks why these are not merely convenient variables, but the natural local
coordinates singled out by the projected EYMH block.

## Fixed Points and Tangent Normalization

Any admissible coordinate system for the projected parent block must satisfy
three minimal requirements:

1. it must vanish exactly at the fixed points,
2. it must preserve the same unit tangent normalization there,
3. it should not turn on nonlinear jets unless those jets are forced by the
   projected parent geometry itself.

The first requirement determines the anchors. The second determines the local
linear normalization. The third is the genuine normal-coordinate condition.

In this language, the canonical claim is:

> the projected parent block is naturally centered on the fixed points
> \((K_{11},K_{22})=(1,K_{\rm bg})\), linearized by the additive excess
> variables, and flattened to zero second jet at leading order.

## Step 1: Anchor Choice Is Structural

The excess-coordinate audit tests the anchored family

\[
E_{\rm part}^{(r)} = K_{11}-r_{\rm part},
\qquad
E_{\rm tree}^{(r)} = K_{22}-r_{\rm tree}K_{\rm bg}.
\]

Its exact closure is unique at

\[
(r_{\rm part},r_{\rm tree})=(1,1),
\]

with machine-precision determinant, Schur, weight, and cross-ratio closure.

The first nontrivial runner-up already opens visible onset residuals:

- participation onset residual:
  `4.5273863863531494e-02`
- tree onset residual:
  `9.776545883388052e-02`
- \(\max|\Delta(\det/K_{\rm bg})|\):
  `1.0817624243038004e-03`
- \(\max|\Delta A|\):
  `2.4817216177697166e-04`
- \(\max|\Delta \xi|\):
  `1.6050610166524815e-02`

So the fixed points are not arbitrary bookkeeping choices. They are the unique
anchors around which the projected parent kernel closes exactly.

## Step 2: Linear Additive Coordinates Are the Unique Tangent-Normalized Choice

Once the anchors are fixed, the next question is whether one may still use a
warped local variable while keeping the same tangent normalization at the
origin.

The fixed-point Box--Cox audit tests this by promoting the excess sectors to

\[
\widetilde E_{\rm part}^{(p)}
=
\frac{(1+E_{\rm part})^{p_{\rm part}}-1}{p_{\rm part}},
\qquad
\widetilde E_{\rm tree}^{(p)}
=
\frac{(1+E_{\rm tree}/K_{\rm bg})^{p_{\rm tree}}-1}{p_{\rm tree}}\,K_{\rm bg},
\]

with the linear limit recovered at

\[
(p_{\rm part},p_{\rm tree})=(1,1).
\]

The exact audit closes uniquely at that point. The first nearby
tangent-normalized runner-up already opens visible distortions:

- participation onset residual:
  `2.021327297322517e-03`
- tree onset residual:
  `8.233330302492792e-03`
- \(\max|\Delta(\det/K_{\rm bg})|\):
  `8.248720976493651e-04`
- \(\max|\Delta A|\):
  `9.79556874433074e-05`
- \(\max|\Delta \xi|\):
  `7.851734132779453e-03`

This is the precise sense in which the additive excess variables are the unique
unit-tangent coordinates selected by the projected parent kernel.

## Step 3: Normal Coordinates Require Zero Second Jet

The remaining freedom is to keep the same anchors and unit tangent, but turn on
a nonlinear second jet. The normal-coordinate audit tests exactly this family:

\[
E_{\rm part}^{(\zeta_p)}=E_{\rm part}+\zeta_p E_{\rm part}^2,
\qquad
E_{\rm tree}^{(\zeta_t)}=E_{\rm tree}+\zeta_t E_{\rm tree}^2/K_{\rm bg}.
\]

The exact closure is unique at

\[
(\zeta_p,\zeta_t)=(0,0),
\]

with canonical second jets

\[
\partial_{E_{\rm part}}^2 E_{\rm part}^{(\zeta_p)}\big|_{0}=0,
\qquad
\partial_{E_{\rm tree}}^2 E_{\rm tree}^{(\zeta_t)}\big|_{0}=0.
\]

The first nontrivial runner-up already opens visible residuals:

- participation onset residual:
  `2.0994781304852217e-03`
- tree onset residual:
  `9.634856104015432e-03`
- \(\max|\Delta(\det/K_{\rm bg})|\):
  `9.90200241782535e-04`
- \(\max|\Delta A|\):
  `1.1822974383268736e-04`
- \(\max|\Delta \xi|\):
  `9.47186852082002e-03`

So the projected parent block is not merely anchored and tangent-normalized by
the linear excess variables; it is also flattened by them to zero second jet.

## Lemma 3 Statement

**Lemma 3.**
Let \(\mathcal K_{\rm parent}\) be the positive background-normalized projected
EYMH parent kernel obtained after the projected Hessian reduction of Gap 1 and
the geometric-mean mixed-entry reduction of Gap 2. Then the natural local
coordinates of \(\mathcal K_{\rm parent}\) near the fixed point
\((K_{11},K_{22})=(1,K_{\rm bg})\) are exactly

\[
E_{\rm part}=K_{11}-1,
\qquad
E_{\rm tree}=K_{22}-K_{\rm bg},
\]

in the sense that they are simultaneously:

1. the unique anchor-preserving coordinates,
2. the unique unit-tangent-normalized local coordinates, and
3. the unique zero-second-jet local coordinates

compatible with exact determinant / Schur / weight closure.

## Proof Skeleton

1. **Anchor uniqueness.**
   Use the excess-coordinate audit to show exact closure only at
   \((r_{\rm part},r_{\rm tree})=(1,1)\).

2. **Linear tangent uniqueness.**
   Use the fixed-point Box--Cox audit to show exact closure only at
   \((p_{\rm part},p_{\rm tree})=(1,1)\).

3. **Zero-second-jet uniqueness.**
   Use the normal-coordinate audit to show exact closure only at
   \((\zeta_p,\zeta_t)=(0,0)\).

4. **Conclusion.**
   The coordinate system that is simultaneously anchor-preserving,
   tangent-normalized, and zero-second-jet is exactly the linear excess system.

## What This Closes

Gap 3 is therefore no longer a search over candidate coordinates. It reduces to
writing a concise parent-action-side normal-form argument:

- the projected parent kernel is centered at the identity/background fixed
  points,
- its first derivatives define the linear excess coordinates,
- and its second jets vanish in those same coordinates at the canonical point.

Together with Gap 1 and Gap 2, this makes the bridge statement read like a
localized Gaussian fluctuation theorem for the EYMH parent block rather than a
collection of exact audits.

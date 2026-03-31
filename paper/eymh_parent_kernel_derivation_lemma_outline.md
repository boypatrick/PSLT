# EYMH Projected Parent-Kernel Derivation Lemma Outline

## Goal

Turn the current exact audit closure into a derivation-ready chain that starts from the projected EYMH parent action and ends with the canonical projected parent-kernel

\[
\mathcal K_{\rm parent}
=
\begin{pmatrix}
K_{11} & C_{\rm parent} \\
C_{\rm parent} & K_{22}
\end{pmatrix},
\qquad
C_{\rm parent}=\sqrt{(K_{11}-1)(K_{22}-K_{\rm bg})},
\]

with response weight

\[
A_{\rm resp}\propto
\left(\frac{\det \mathcal K_{\rm parent}}{K_{\rm bg}}\right)^{-1/2}
=
\left(K_{11}\,\widehat G_{\rm Schur}\right)^{-1/2}.
\]

The bridge audit under
`/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/eymh_parent_kernel_bridge_status_summary.csv`
already closes the structural ambiguity (`10/10` gates). The remaining work is therefore a derivation/write-up task, not another family search.

## What Is Already Closed

The following points should be treated as established numerical closure, not as open scan problems:

1. The projected response action is exactly a `logdet + Schur` object.
2. The parent-kernel excess functional matches the direct mismatch functional exactly.
3. The participation/tree split closes exactly.
4. The background-normalized parent block-determinant / Schur complement closes exactly.
5. The mixed block is uniquely selected as the geometric mean within the minimal symmetric-excess family.
6. The first ratio-warp, symmetry/normalization, and local generator-curvature deformations are all rejected.
7. The identity/background anchors are uniquely selected.
8. The linear excess variables are uniquely selected among tangent-normalized coordinate warps.
9. The zero-second-jet normal coordinates coincide with those same linear excess variables.

This means the paper no longer needs to motivate another scan. It needs to explain why these exact identities are the natural projected form of the EYMH parent action.

## Notation To Keep Fixed

Reuse the notation already introduced in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex`:

\[
K_{\rm part}=\frac{\|c\|_1}{\|c\|_2},
\qquad
K_{\rm bg}=1+\mathrm{gap}_{\rm cv},
\qquad
G_{\rm Schur}=\frac{1+\mathrm{gap}_{\rm cv}+c_{{\rm tree},{\rm diag},{\rm cv}}}{1+\mathrm{gap}_{\rm cv}}.
\]

\[
K_{11}=e^{S_{\rm part}},
\qquad
K_{22}=K_{\rm bg}e^{S_{\rm schur}},
\qquad
E_{\rm part}=K_{11}-1,
\qquad
E_{\rm tree}=K_{22}-K_{\rm bg}.
\]

The derivation should preserve this language so the new section plugs directly into the existing EYMH narrative around
`/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:917`
through
`/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1120`.

## Proposed Derivation Flow

### Lemma 1. Projected Hessian Reduction

**Statement**

Let \(S_{\rm EYMH}[\Psi;g,\Phi,A]\) be the parent EYMH action expanded to quadratic order about the localized background used in the canonical D21xE21 fix grid. After projection onto the two dominant low-mode fluctuation directions,

\[
\delta\Psi
=
a_{\rm part}\,u_{\rm part}
+
a_{\rm tree}\,u_{\rm tree},
\]

the quadratic fluctuation form reduces to a positive \(2\times 2\) kernel

\[
\delta^2 S_{\rm EYMH}\big|_{\rm proj}
=
\begin{pmatrix}
a_{\rm part} & a_{\rm tree}
\end{pmatrix}
\mathcal K_{\rm parent}
\begin{pmatrix}
a_{\rm part} \\
a_{\rm tree}
\end{pmatrix},
\]

whose diagonal entries are the canonical projected participation and shell-background-normalized tree/background blocks:

\[
K_{11}=e^{S_{\rm part}},
\qquad
K_{22}=K_{\rm bg}e^{S_{\rm schur}}.
\]

Moreover, the projected response weight is the inverse square root of the normalized determinant / Schur factorization:

\[
A_{\rm resp}
\propto
\left(\frac{\det\mathcal K_{\rm parent}}{K_{\rm bg}}\right)^{-1/2}
=
\left(K_{11}\widehat G_{\rm Schur}\right)^{-1/2}.
\]

**Why this is the next lemma**

This is the main gap identified in
`/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/eymh_parent_kernel_bridge_status_gaps.csv`
as `derive_projected_hessian`.

**Inputs already available**

1. Exact response-action closure.
2. Exact `logdet + Schur` rewrite.
3. Exact parent-kernel excess functional.
4. Exact block split.
5. Exact background-normalized block-determinant / Schur identity.

**Proof skeleton**

1. Start from the parent EYMH second variation around the localized background.
2. Define the projected two-mode subspace and its Gram-normalized basis.
3. Show that the diagonal projected coefficients reproduce the already exported participation and tree/background witnesses.
4. Normalize the tree block by \(K_{\rm bg}\) and identify the Schur complement.
5. Read the response weight as the projected Gaussian fluctuation determinant.
6. Match the resulting projected action to the already exact audit identities.

**Minimal deliverable**

Not a full continuum derivation of every exported witness. Only show that once the two-mode localized projection is fixed, the canonical projected kernel must take the same determinant / Schur form already observed numerically.

### Lemma 2. Geometric-Mean Mixed Entry

**Statement**

Within the projected parent block, assume the mixed entry depends only on the excess magnitudes away from the two fixed-point anchors,

\[
E_{\rm part}=K_{11}-1,
\qquad
E_{\rm tree}=K_{22}-K_{\rm bg},
\]

and require:

1. positivity of the projected block,
2. vanishing mixed term at either fixed point,
3. symmetry under interchange of the two excess sectors after background normalization,
4. local affine behavior in multiplicative excess variables,
5. no extra ratio warp at the first nontrivial order.

Then the mixed block is forced to be

\[
C_{\rm parent}=\sqrt{E_{\rm part}E_{\rm tree}}
\]

up to the sign fixed by continuity and positivity.

**Why this is the next lemma**

This is the clean conceptual upgrade of the geomean / ratio-warp / symmetry-normalization / generator-affinity audits.

**Inputs already available**

1. Geometric-mean audit closes exactly at \((u,v,\kappa)=(1/2,1/2,1)\).
2. Ratio warp is uniquely rejected at \((\delta,\nu)=(0,0)\).
3. Symmetry/normalization coordinates fix \((m,s,a)=(0,1,0)\).
4. Generator-affinity fixes the local log generator at zero curvature.

**Proof skeleton**

1. Write the most general minimal symmetric-excess coupling \(C(E_{\rm part},E_{\rm tree})\).
2. Impose vanishing at both fixed-point anchors.
3. Impose equal homogeneity in the two excess sectors after background normalization.
4. Expand the local log generator about the canonical point.
5. Use the ratio-warp and generator-affinity rejections to remove the first nonlinear alternatives.
6. Conclude that the only surviving mixed bilinear is the geometric mean.

**Minimal deliverable**

Explain why the geometric mean is not just the best scan point, but the unavoidable mixed bilinear once the projected kernel is assumed to be positive, anchor-respecting, symmetric, and locally affine in excess variables.

### Lemma 3. Linear Excess Variables As Normal Coordinates

**Statement**

Let the projected kernel be organized around the fixed points

\[
K_{11}=1,
\qquad
K_{22}=K_{\rm bg}.
\]

Among local coordinates that preserve both fixed points and the same unit tangent normalization, the canonical projected parent action naturally selects the zero-second-jet coordinates

\[
E_{\rm part}=K_{11}-1,
\qquad
E_{\rm tree}=K_{22}-K_{\rm bg},
\]

and these coincide with the unique exact linear excess variables already isolated by the excess-coordinate, Box-Cox, and normal-coordinate audits.

**Why this is the next lemma**

This is the coordinate-level closure still missing from the parent-action narrative.

**Inputs already available**

1. Excess-coordinate audit fixes the anchors at \((r_{\rm part},r_{\rm tree})=(1,1)\).
2. Fixed-point Box-Cox audit fixes the tangent-normalized coordinate family at \((p_{\rm part},p_{\rm tree})=(1,1)\).
3. Normal-coordinate audit fixes the second jet at \((\zeta_p,\zeta_t)=(0,0)\).

**Proof skeleton**

1. Define the fixed-point manifold in the projected kernel variables.
2. State the coordinate conditions:
   - fixed-point preservation,
   - unit tangent normalization,
   - vanishing second jet.
3. Show that the linear excess variables satisfy all three.
4. Show that the first nonlinear alternatives immediately reintroduce onset distortions.
5. Conclude that the natural projected parent coordinates are exactly the linear excess variables.

**Minimal deliverable**

Promote the existing coordinate-selection audits into a concise normal-coordinate lemma that explains why the parent block is naturally expanded in linear excess variables rather than in arbitrary warped coordinates.

## Final Corollary To Target

After Lemmas 1--3, the paper should be able to state a compact corollary:

> The localized low-mode EYMH fluctuation operator projects to a background-normalized \(2\times 2\) parent kernel whose diagonal blocks are the participation and shell-background tree/background sectors, whose mixed block is the geometric mean of their linear excess coordinates, and whose response weight is the inverse square root of the associated normalized determinant / Schur factorization.

This corollary is the derivation-side replacement for the current phrase “the remaining gap is to explain why the parent action selects the canonical projected form.”

## Suggested Paper Integration

### Option A: Add a short new subsection after the current EYMH audit chain

Insert after the current normal-coordinate paragraph around
`/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1120`

Suggested subsection title:

- `\subsection{Projected Parent-Kernel Derivation Outline}`

Suggested structure:

1. one paragraph stating that structural ambiguity is now closed by exact audits,
2. one paragraph introducing the three derivation lemmas,
3. one displayed theorem/corollary statement,
4. a short roadmap sentence saying the fully explicit Hessian reduction is deferred but now sharply localized.

### Option B: Replace repeated “remaining gap” sentences with one lemma roadmap paragraph

Keep the current audit chain intact, but add a closing synthesis paragraph after the normal-coordinate audit:

- Lemma 1: projected Hessian reduction
- Lemma 2: geometric-mean mixed bilinear
- Lemma 3: zero-second-jet excess coordinates

This is lower risk if we do not want to touch the detailed audit narrative yet.

## Recommended Writing Order

1. Write Lemma 1 first.
   - It is the real bottleneck.
   - It also makes Lemmas 2 and 3 feel motivated instead of appended.

2. Write Lemma 3 second.
   - It is the cleanest audit-to-derivation bridge.
   - It pins down the coordinate language used in Lemma 2.

3. Write Lemma 2 third.
   - Once coordinates are fixed, the geometric-mean statement becomes easier to phrase as the unique symmetric locally affine mixed bilinear.

## What Not To Re-Open

Do not reopen:

1. new `alpha/beta/lambda` scans,
2. new ratio-warp families,
3. new Box-Cox families,
4. new normal-coordinate families.

Those ambiguities are already numerically closed. The next step is to convert closure into derivation.

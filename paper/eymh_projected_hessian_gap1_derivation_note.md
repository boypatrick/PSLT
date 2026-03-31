# Gap 1 Derivation Note: Projected 2x2 Hessian Reduction

## Aim

Convert the current EYMH audit closure into a parent-action-side statement:

> after expanding the EYMH action to quadratic order around the localized background and projecting onto the two dominant low-mode directions, the projected fluctuation operator reduces to a positive background-normalized \(2\times 2\) kernel whose determinant / Schur factorization reproduces the canonical response weight.

This note does **not** attempt a full continuum derivation of every exported witness. The immediate goal is narrower:

1. define the projected Hessian precisely,
2. show how the canonical \(2\times 2\) kernel arises from it,
3. show why the response weight is the inverse square root of the normalized determinant / Schur factorization.

That is the exact gap labeled `derive_projected_hessian` in
`/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/eymh_parent_kernel_bridge_status_gaps.csv`.

## Starting Point: Parent Action and Quadratic Expansion

The parent action is the minimal EYMH functional already stated in
`/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:112`:

\[
S[\Psi]
=
\int d^4x\sqrt{-g}\,
\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac14 F_{\mu\nu}^aF^{a\mu\nu}
-(D_\mu\Phi)^\dagger(D^\mu\Phi)
-\lambda(|\Phi|^2-v^2)^2
-\xi R|\Phi|^2
\right],
\]

where \(\Psi\) denotes the collective fluctuation variables around the localized background:

\[
\Psi=\Psi_0+\delta\Psi.
\]

Expanding to second order gives

\[
S[\Psi_0+\delta\Psi]
=
S[\Psi_0]
+\delta S[\Psi_0;\delta\Psi]
+\frac12 \langle \delta\Psi,\mathcal H_{\rm EYMH}\delta\Psi\rangle
+\mathcal O(\delta\Psi^3).
\]

On the background solving the projected equations of motion, the linear term vanishes, so the relevant object is the quadratic form

\[
\delta^2 S_{\rm EYMH}[\delta\Psi]
=
\frac12 \langle \delta\Psi,\mathcal H_{\rm EYMH}\delta\Psi\rangle.
\]

The whole gap is therefore a statement about the projected Hessian operator \(\mathcal H_{\rm EYMH}\).

## Two-Mode Projection

The exact audits already show that the surviving normalization block is organized by two dominant low-mode sectors:

1. a participation sector,
2. a shell-background/tree sector.

We therefore restrict to the two-dimensional projected fluctuation subspace

\[
\delta\Psi
=
a_{\rm part}\,u_{\rm part}
+
a_{\rm tree}\,u_{\rm tree},
\]

where \(u_{\rm part}\) and \(u_{\rm tree}\) are the two localized projected directions extracted from the canonical EYMH chain.

Define the projected Hessian entries

\[
H_{ij}
=
\langle u_i,\mathcal H_{\rm EYMH}u_j\rangle,
\qquad
i,j\in\{{\rm part},{\rm tree}\}.
\]

After Gram normalization of the projected basis, the quadratic form becomes

\[
\delta^2 S_{\rm EYMH}\big|_{\rm proj}
=
\frac12
\begin{pmatrix}
a_{\rm part} & a_{\rm tree}
\end{pmatrix}
H_{\rm proj}
\begin{pmatrix}
a_{\rm part}\\
a_{\rm tree}
\end{pmatrix},
\qquad
H_{\rm proj}
=
\begin{pmatrix}
H_{pp} & H_{pt}\\
H_{pt} & H_{tt}
\end{pmatrix}.
\]

The projected kernel is positive because the canonical point is selected by a stable quadratic form in the \((\alpha,\beta)\) sector; numerically this is exactly the content of the stationarity and kernel-selection audits:

- `paper/hll_uv_action_loop_eymh_kernel_selection_audit_summary_D21E21_fix.csv`
- `paper/hll_uv_action_loop_eymh_stationarity_audit_summary_D21E21_fix.csv`

with positive eigenvalues in the quadratic sector.

## Identification of the Diagonal Blocks

The existing exact audit chain already tells us how the diagonal sectors should be read.

### Participation block

The participation side is exported as the canonical projected block

\[
K_{\rm part}=\frac{\|c\|_1}{\|c\|_2}.
\]

At the parent-kernel level this is encoded as

\[
K_{11}=e^{S_{\rm part}}.
\]

Interpretationally, \(S_{\rm part}\) is the projected contribution of the participation mode to the quadratic EYMH fluctuation action, normalized so that the identity fixed point sits at

\[
K_{11}=1
\quad\Longleftrightarrow\quad
S_{\rm part}=0.
\]

### Tree/background block

The tree-side block is not used in bare form. The exact audits show that the correct object is the shell-background-normalized block

\[
G_{\rm Schur}
=
\frac{1+\mathrm{gap}_{\rm cv}+c_{{\rm tree},{\rm diag},{\rm cv}}}{1+\mathrm{gap}_{\rm cv}}
=
\frac{K_{22}}{K_{\rm bg}},
\qquad
K_{\rm bg}=1+\mathrm{gap}_{\rm cv}.
\]

Equivalently,

\[
K_{22}=K_{\rm bg}e^{S_{\rm schur}}.
\]

So the projected Hessian is not compared directly to an absolute tree/background block. It is compared to a background-normalized block whose fixed point is

\[
K_{22}=K_{\rm bg}
\quad\Longleftrightarrow\quad
S_{\rm schur}=0.
\]

This is exactly what the parent block-determinant / Schur audit closes:

- `paper/hll_uv_action_loop_eymh_parent_blockdet_audit_summary_D21E21_fix.csv`

## From Gaussian Fluctuations to the Response Weight

For a quadratic projected action, the two-mode Gaussian fluctuation integral is

\[
Z_{\rm proj}
\propto
\left(\det H_{\rm proj}\right)^{-1/2}.
\]

If we divide out the shell-background reference block, the physically relevant normalized weight is

\[
A_{\rm resp}
\propto
\left(\frac{\det \mathcal K_{\rm parent}}{K_{\rm bg}}\right)^{-1/2},
\]

where \(\mathcal K_{\rm parent}\) is the dimensionless projected parent kernel obtained from \(H_{\rm proj}\) after the canonical EYMH normalization.

Writing

\[
\mathcal K_{\rm parent}
=
\begin{pmatrix}
K_{11} & K_{12}\\
K_{12} & K_{22}
\end{pmatrix},
\]

we have the exact algebraic identity

\[
\frac{\det \mathcal K_{\rm parent}}{K_{\rm bg}}
=
K_{11}
\frac{K_{22}-K_{12}^2/K_{11}}{K_{\rm bg}}
=
K_{11}\,\widehat G_{\rm Schur}.
\]

Therefore

\[
A_{\rm resp}
\propto
\left(K_{11}\,\widehat G_{\rm Schur}\right)^{-1/2},
\]

and hence the projected response action is

\[
S_{\rm parent}
=
\log K_{11}+\log \widehat G_{\rm Schur}.
\]

This is precisely the `logdet + Schur` structure already established numerically in
`/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:917`
and sharpened at
`/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1000`.

## Core Lemma Statement

The derivation target can now be stated cleanly.

### Lemma 1 (projected 2x2 Hessian reduction)

Let \(\Psi_0\) be the localized background used in the canonical D21xE21 fix grid, and let \(u_{\rm part},u_{\rm tree}\) be the two dominant projected low-mode fluctuation directions. After Gram normalization of the projected basis, the EYMH quadratic fluctuation operator reduces to a positive \(2\times 2\) parent kernel

\[
\delta^2 S_{\rm EYMH}\big|_{\rm proj}
=
\frac12
\begin{pmatrix}
a_{\rm part} & a_{\rm tree}
\end{pmatrix}
\mathcal K_{\rm parent}
\begin{pmatrix}
a_{\rm part}\\
a_{\rm tree}
\end{pmatrix},
\]

with

\[
\mathcal K_{\rm parent}
=
\begin{pmatrix}
K_{11} & K_{12}\\
K_{12} & K_{22}
\end{pmatrix},
\qquad
K_{11}=e^{S_{\rm part}},
\qquad
K_{22}=K_{\rm bg}e^{S_{\rm schur}},
\]

such that the corresponding projected response weight is

\[
A_{\rm resp}
\propto
\left(\frac{\det \mathcal K_{\rm parent}}{K_{\rm bg}}\right)^{-1/2}
=
\left(K_{11}\widehat G_{\rm Schur}\right)^{-1/2}.
\]

Equivalently, the canonical projected response action is the normalized block-determinant / Schur-complement factorization

\[
S_{\rm parent}
=
\log K_{11}+\log \widehat G_{\rm Schur}.
\]

## What This Lemma Uses and What It Does Not Yet Prove

### Already used

1. The projected low-mode subspace exists and is numerically stable.
2. The canonical response action is exactly reproduced by the `logdet + Schur` witness.
3. The parent block-determinant / Schur factorization closes to machine precision.
4. The \((\alpha,\beta)\) sector is quadratically stable.

### Still deferred to Gap 2 and Gap 3

1. Why the mixed block must be the geometric mean.
2. Why the natural coordinates are exactly the linear excess variables.

So Gap 1 stops at the generic \(2\times 2\) projected kernel with determinant / Schur response. It does not yet fix the detailed form of \(K_{12}\).

## Minimal Proof Strategy

To turn this note into paper-ready proof prose, the shortest path is:

1. Write the second variation of the EYMH action in block form around the localized background.
2. State the two-mode projection explicitly.
3. Define the projected diagonal matrix elements as the participation and tree/background blocks.
4. Normalize the tree block by \(K_{\rm bg}\).
5. Read the Gaussian fluctuation determinant as the response weight.
6. Use the Schur identity to rewrite the result in canonical `logdet + Schur` form.

This is enough to upgrade the existing audit chain from a purely exact-selection statement to a genuine projected-Hessian statement.

## Immediate Corollary for the Paper

Once Lemma 1 is accepted, the remaining EYMH derivation gap is sharply localized:

- Gap 2 explains why \(K_{12}\) is the geometric mean of the two excess blocks.
- Gap 3 explains why the excess blocks themselves are naturally written in anchored linear coordinates.

That is why the bridge checklist now reports

\[
\texttt{STRUCTURAL\_BRIDGE\_CLOSED\_DERIVATION\_LEMMA\_PENDING}
\]

rather than another open family ambiguity.

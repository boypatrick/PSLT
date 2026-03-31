# EYMH Generic Off-Diagonal Audit Design

## Goal

Design a non-circular audit for the off-diagonal parent-kernel entry in
Lemma 2.

The current `parent_blockdet` and `parent_kernel_statement` exact identities are
not independent of the geometric-mean ansatz, because both already insert

\[
C_{\rm parent}=\sqrt{(K_{11}-1)(K_{22}-K_{\rm bg})}.
\]

The new audit should therefore avoid scanning `C` inside a family that already
hard-codes the geometric mean.

## Core idea

Work with the squared mixed entry

\[
Q(E_{\rm part},E_{\rm tree}) := K_{12}^2,
\]

rather than with `C` itself.

Once Lemma 1 has fixed the generic projected determinant relation

\[
A_{\rm resp}\propto
\left(\frac{\det\mathcal K_{\rm parent}}{K_{\rm bg}}\right)^{-1/2},
\]

the direct response map gives

\[
\frac{\det\mathcal K_{\rm parent}}{K_{\rm bg}}
=
\exp(S_{\rm resp}),
\qquad
S_{\rm resp}=-2\log A_{\rm resp}.
\]

For a generic symmetric parent kernel

\[
\mathcal K_{\rm parent}
=
\begin{pmatrix}
K_{11} & K_{12}\\
K_{12} & K_{22}
\end{pmatrix},
\]

this implies

\[
Q_{\rm inferred}
=
K_{11}K_{22}-K_{\rm bg}\exp(S_{\rm resp}).
\]

With the canonical excess variables

\[
E_{\rm part}=K_{11}-1,
\qquad
E_{\rm tree}=K_{22}-K_{\rm bg},
\]

the non-circular determinant-matching test is therefore:

\[
Q_{\rm inferred}\stackrel{?}{=}E_{\rm part}E_{\rm tree}.
\]

If true on the local positive cone, then

\[
K_{12}^2 = E_{\rm part}E_{\rm tree},
\]

and the positive branch gives

\[
K_{12}=\sqrt{E_{\rm part}E_{\rm tree}}.
\]

## Audit outputs

The new script

- `/Users/boypatrick/codex/PSLT_quantam/code/scan_hll_uv_action_loop_eymh_parent_generic_offdiag_audit.py`

exports:

- inferred determinant-ratio map
- inferred squared coupling `Q_inferred`
- geometric-mean square `E_part E_tree`
- normalized ratio
  \[
  \rho_{\rm inferred}=Q_{\rm inferred}/(E_{\rm part}E_{\rm tree})
  \]
- D-slice summaries for `rho_inferred`

## Why this matters

This audit isolates the only ingredient needed to upgrade Lemma 2 into a true
determinant-matching proof:

- it uses the diagonal sectors from Lemma 1,
- it uses the direct response action,
- and it does **not** assume the geometric-mean mixed entry as an input.

## Next step after running the audit

If the map closes at machine precision with

\[
\rho_{\rm inferred}\equiv 1,
\]

then the main proof strategy for Lemma 2 can be upgraded from

- generator-family uniqueness,

to

- determinant-matching uniqueness of the squared off-diagonal entry,

with the current generator-family scans retained only as supporting structural
evidence.

## First-pass result on the canonical D21xE21 fix map

The initial prototype run of

- `/Users/boypatrick/codex/PSLT_quantam/code/scan_hll_uv_action_loop_eymh_parent_generic_offdiag_audit.py`

shows that this naive determinant-only inversion does **not** yet recover the
geometric-mean square. Instead it gives

\[
Q_{\rm inferred}\approx 0
\]

across the canonical map, so that

\[
\rho_{\rm inferred}\approx 0.
\]

This is not a numerical failure; it reflects the current exact audit chain.
The canonical direct response corresponds to the block-diagonal point
`\lambda = 0` in the parent-block family, so the determinant at the physical
point does not by itself determine the mixed entry.

## Consequence

The generic off-diagonal audit cannot stop at canonical determinant inversion.
To identify the mixed entry independently, the next audit must probe a quantity
that is sensitive to the off-diagonal sector away from the `\lambda=0` point.

Two promising directions are:

1. **Curvature audit around the canonical point**
   - infer the quadratic `\lambda^2` coefficient of the response action and
     match it against a generic squared mixed entry
     \[
     Q(E_{\rm part},E_{\rm tree}).
     \]

2. **Off-diagonal observable audit**
   - use the independently exported off-diagonal parented channel data, such as
     `/Users/boypatrick/codex/PSLT_quantam/output/hll_uv_matching/hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv`
     columns
     - `eymh_parented_prefactor_offdiag`
     - `hk_loop_local_prefactor_offdiag`
     - `offdiag_mix`
   - to build a non-circular target for the mixed block.

At the moment, this note closes one design question cleanly:

- **determinant matching at the canonical physical point is not enough by itself**
  to upgrade Lemma 2.

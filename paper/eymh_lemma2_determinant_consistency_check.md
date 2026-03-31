# EYMH Lemma 2 Determinant-Matching Consistency Check

## Question

Can Lemma 2 be strengthened by deriving the geometric-mean mixed entry directly from
the determinant identity

\[
\frac{\det \mathcal K_{\rm parent}}{K_{\rm bg}}
=
1+E_{\rm part}+\frac{E_{\rm tree}}{K_{\rm bg}},
\]

instead of from the audited mixed-coupling generator families?

## Short answer

Not with the current audit chain alone.

At present, the exact determinant law used in the EYMH parent-block section is not
independent of the geometric-mean ansatz. The current `parent_blockdet` and
`parent_kernel_statement` constructions already define the mixed entry by

\[
C_{\rm parent}=\sqrt{(K_{11}-1)(K_{22}-K_{\rm bg})}.
\]

Therefore a proof of Lemma 2 that takes the existing determinant identity as an
external theorem input would be circular.

## Evidence

### 1. The parent-block determinant audit already hard-codes the geometric mean

In `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex`, the section beginning near
line 1000 introduces the parent-block determinant family by defining

\[
K_{11}=e^{\alpha S_{\rm part}},
\qquad
K_{22}=K_{\rm bg}e^{\beta S_{\rm schur}},
\qquad
C_{\rm parent}=\sqrt{(K_{11}-1)(K_{22}-K_{\rm bg})}.
\]

The corresponding scan file confirms that the audit parameters are only
`alpha`, `beta`, and `lambda`:

- `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_blockdet_scan_D21E21_fix.csv`

No independent mixed-entry degree of freedom is scanned in that audit.

### 2. The parent-kernel statement audit also assumes the same mixed entry

The exact parent-kernel statement section uses the same canonical parent block with
geometric-mean mixing already inserted. Its scan file likewise varies only
`alpha`, `beta`, and `lambda`:

- `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_kernel_statement_scan_D21E21_fix.csv`

Thus the exact identity

\[
\frac{\det\mathcal K_{\rm parent}}{K_{\rm bg}}
=
K_{11}\widehat G_{\rm Schur}
\]

is currently established only inside a family where the off-diagonal entry has
already been set to the geometric mean.

### 3. The actual independence tests for the mixed entry live elsewhere

The non-circular evidence for Lemma 2 currently comes from the mixed-coupling audits:

- `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_mix_geomean_audit_summary_D21E21_fix.csv`
- `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_ratio_warp_audit_summary_D21E21_fix.csv`
- `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_symnorm_audit_summary_D21E21_fix.csv`
- `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_generator_affinity_audit_summary_D21E21_fix.csv`

These files do provide genuine uniqueness information for the mixed entry, but only
within the audited multiplicative/generator families. They do **not** yet establish
the determinant law for a completely generic off-diagonal function
`C(E_part, E_tree)`.

## Consequence for Lemma 2

The stronger proof strategy

1. assume generic symmetric kernel with unknown `C(E_part,E_tree)`,
2. use an independently established determinant law,
3. conclude `C^2 = E_part E_tree`,

would be mathematically clean, but it is not supported by the current audit chain
without an additional independence step.

Right now, the safe logical order is:

1. Lemma 1 fixes the projected `2x2` Hessian structure.
2. Lemma 2 fixes the mixed entry inside the audited positive multiplicative class.
3. The corollary then writes the explicit parent kernel and determinant law.

## Safe upgrade paths

### Option A: add an independent determinant law audit

Introduce a new audit that scans a generic local off-diagonal family

\[
C(E_{\rm part},E_{\rm tree})
\]

without hard-coding the geometric mean, and test whether exact determinant closure
forces

\[
C^2=E_{\rm part}E_{\rm tree}.
\]

If that audit closes uniquely, then Lemma 2 can be rewritten as a genuine
determinant-matching theorem.

### Option B: keep the current proof order, but narrow the claim

Retain the current generator-family proof as the primary argument, and present the
determinant formula only as the canonical corollary after Lemmas 1--3. This avoids
circularity and stays faithful to the present evidence.

## Recommended next step

Do **not** yet rewrite Lemma 2 as if the determinant law were an independent input.

Instead, either:

1. keep the current generator-based proof and maybe tighten its assumptions further,
   or
2. add a new generic-off-diagonal audit specifically designed to establish the
   determinant law without assuming the geometric mean.

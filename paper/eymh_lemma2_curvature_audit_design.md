# EYMH Lemma 2 Curvature-Based Generic Off-Diagonal Audit Design

## Goal

Design a non-circular follow-up to the failed determinant-only inversion in

- `/Users/boypatrick/codex/PSLT_quantam/paper/eymh_lemma2_determinant_consistency_check.md`

for the mixed parent-kernel entry in Lemma 2.

The determinant at the canonical physical point only sees the block-diagonal
`\lambda = 0` configuration, so it cannot by itself recover the mixed entry.
The next informative quantity is therefore not the canonical determinant, but
the even `\lambda`-direction curvature around the canonical point.

## Core structural observation

The current EYMH stationarity / variational-selection chain treats the explicit
mixing parameter as an even direction:

\[
J(\lambda)=C_4 \lambda^4 + \mathcal O(\lambda^6).
\]

In the existing canonical construction this quartic coefficient is written as

\[
C_4
=
\left\langle
\frac14 A_{\rm ref}^2 \, \xi_{\rm cross}^2
\right\rangle,
\qquad
\xi_{\rm cross}
=
\frac{(K_{\rm part}-1)(G_{\rm Schur}-1)}{K_{\rm part}G_{\rm Schur}},
\]

which already encodes the geometric-mean square. That makes the current
`stationarity` and `variational_selection` audits unsuitable as a direct
non-circular proof of Lemma 2.

## Audit strategy

Instead of taking `\xi_{\rm cross}` as given, the new audit treats the
quartic direction as a target and asks whether any independently exported
off-diagonal witness can reproduce it stably.

The audit therefore introduces a generic proxy family

\[
\xi_{\rm proxy}(D,\eta)
=
c \, P(D,\eta),
\]

where `P` is built only from columns already exported by the parented map, such
as

- `action_abs_offdiag`
- `action_norm_offdiag`
- `hk_mass_access_offdiag`
- `hk_loop_prefactor_offdiag`
- `hk_curv_screen_offdiag`
- `hk_abs_offdiag`
- `hk_abs_offdiag_geom`

possibly after simple diagonal/background normalizations.

The proxy-induced quartic density is then

\[
q_4^{\rm proxy}
=
\frac14 A_{\rm ref}^2 \, \xi_{\rm proxy}^2.
\]

## What this audit is and is not

This is a **design / proxy audit**, not yet a theorem.

It is intended to answer:

1. Which independently exported off-diagonal witness carries the strongest
   signal for the canonical even `\lambda` direction?
2. Is there a proxy whose quartic coefficient and D-profile are stable enough
   to motivate a new non-circular mixed-entry audit?

It is **not** intended to claim that the mixed entry is already proved from the
parent action, because the target quartic profile is still read from the
current canonical EYMH chain.

## Ranking metrics

For each proxy family the audit records:

1. weighted least-squares scale `c` onto the canonical quartic target,
2. `corr(\xi_{\rm proxy}, \xi_{\rm target})`,
3. `corr(q_4^{\rm proxy}, q_4^{\rm target})`,
4. `|C_4^{\rm proxy} - C_4^{\rm target}|`,
5. p95 absolute residuals in `\xi` and quartic density,
6. D-slice stability of those residuals.

The goal is to identify a small shortlist of proxy families worth carrying into
the next proof-strengthening step.

## Expected next step after the proxy audit

If one proxy family is clearly dominant, the next audit should move from
canonical matching to a genuinely non-circular curvature statement, for example:

1. a `\lambda`-curvature audit that treats the mixed square
   `Q(E_{\rm part},E_{\rm tree})` as unknown and uses the chosen proxy to infer
   it, or
2. an off-diagonal observable audit built directly from a parented-map channel
   that varies independently of the current geometric-mean chain.

At this stage the practical objective is narrower:

- identify the least arbitrary off-diagonal witness that can stand in for the
  even `\lambda` curvature without already assuming the result of Lemma 2.

## First-pass proxy result on the canonical D21xE21 fix map

The first prototype run of

- `/Users/boypatrick/codex/PSLT_quantam/code/scan_hll_uv_action_loop_eymh_parent_curvature_proxy_audit.py`

produces the summary

- `/Users/boypatrick/codex/PSLT_quantam/output/hll_uv_matching/hll_uv_action_loop_eymh_parent_curvature_proxy_audit_summary_D21E21_fix.csv`

and leads to three practical conclusions.

### 1. `action_abs_offdiag` is not the best curvature proxy once quartic matching matters

`action_abs_offdiag` looked attractive from simple correlation with the
geometric-mean target, but after weighted quartic matching it is no longer the
best-behaved family. Its normalized variants remain competitive, but they are
not the most stable against the quartic target.

### 2. The current shortlist is narrower than expected

The best first-pass candidates are:

1. `hk_curv_screen_offdiag:over_diag`
2. `hk_abs_offdiag:over_diag`
3. `action_abs_offdiag:over_diag` (or `:over_tree`)

Among these, the first family gives the best overall residual score, while the
second gives the strongest direct correlation with the canonical `\xi` target.

### 3. The next audit should start from a shortlist, not a broad family search

The prototype already suggests that the useful mixed-curvature signal is being
carried mainly by:

- curvature-screened off-diagonal access, and
- normalized absolute off-diagonal access.

So the next stage should not reopen a large proxy zoo. It should focus on a
small shortlist and ask whether one of these witnesses can support a more
theorem-like curvature statement, for example via:

- a two-proxy blended curvature model, or
- a D-slice stability audit restricted to the shortlisted families.

## Second-pass shortlist result

The follow-up shortlist run

- `/Users/boypatrick/codex/PSLT_quantam/code/scan_hll_uv_action_loop_eymh_parent_curvature_shortlist_audit.py`

with outputs

- `/Users/boypatrick/codex/PSLT_quantam/output/hll_uv_matching/hll_uv_action_loop_eymh_parent_curvature_shortlist_audit_summary_D21E21_fix.csv`
- `/Users/boypatrick/codex/PSLT_quantam/output/hll_uv_matching/hll_uv_action_loop_eymh_parent_curvature_shortlist_audit_dsummary_D21E21_fix.csv`

shows that a small two-proxy blend improves materially over every single-family
proxy.

### Best current family

The strongest current candidate is

\[
\xi_{\rm proxy}
=
c_1\,P_{\rm curv}(D,\eta)+c_2\,P_{\rm abs}(D,\eta),
\]

with

- `P_curv = hk_curv_screen_offdiag:over_diag`,
- `P_abs = action_abs_offdiag:over_diag`,
- `(c_1,c_2) \approx (0.1049, 0.4778)`.

This blend gives

- `rank_score = 0.01194`,
- `p95_abs_quartic_resid = 0.00107`,
- `p95_abs_xi_resid = 0.04300`,

which is clearly better than the best single proxy

- `hk_curv_screen_offdiag:over_diag`
  with
  - `rank_score = 0.01874`,
  - `p95_abs_quartic_resid = 0.00121`,
  - `p95_abs_xi_resid = 0.06948`.

### Important interpretation

This does **not** yet prove a mixed-entry theorem.  What it shows is more
targeted:

1. a pure curvature-screen witness is not enough on its own;
2. a pure absolute off-diagonal witness is also not enough on its own;
3. the most stable quartic `\lambda` proxy currently appears to be a blended
   family combining curvature-screened access with absolute off-diagonal
   strength.

### Consequence for the next proof-strengthening step

The next non-circular audit should therefore begin from a **shortlisted blended
proxy family**, not from a single witness.  The most natural next tests are:

1. D-slice / holdout stability of the best blend coefficients, and
2. whether the blended quartic witness can be rewritten as a function of
   canonical excess variables alone.

# PSLT Family Decision Table

Last updated: 2026-03-28

This file is the one-page decision map for family scans around the reviewer mainline. It is intentionally operational: the goal is to answer "should we keep investing in this family?" without reopening the whole audit history.

## Canonical Reviewer Mainline

| Family | Status | Why it stays canonical | Main evidence |
| --- | --- | --- | --- |
| `cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost` | `KEEP / MAINLINE` | Best current reviewer-side compromise after refreshed baselines. Remaining gap is known and structured rather than random. | `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv`, `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D21E41.csv`, `output/kinetic_action_chain/model_chain_fullwidthrefamp_pointamp2_widthboost_d60_reviewerref_hotspot_audit_summary.csv` |

## Closed Negative Families

These families were tested and should be treated as closed for reviewer-mainline promotion unless a genuinely different mechanism is introduced.

| Family | Layer / mechanism | Decision | Why it is closed | Main evidence |
| --- | --- | --- | --- | --- |
| `pointamp3` | reviewer observable `point-amp` extension | `CLOSED - NEGATIVE` | Helps `D60 D≈6.169/6.441`, but reopens `D21 D=6.4`. | `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp3_widthboost_D60E21_refresh_20260324.csv`, `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp3_widthboost_D21E41_refresh_20260324.csv` |
| `widthband` | reviewer observable width family | `CLOSED - NEGATIVE` | Lowers D60 width hotspots, but reopens `D21 D=6.4` badly. | `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthband_D60E21_refresh_20260325.csv`, `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthband_D21E41_refresh_20260325.csv` |
| `partialguard` | reviewer observable partial guard | `CLOSED - NEGATIVE` | Shortlist looked promising, but true `D60×E21` reviewer map was worse than mainline. | `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_partialguard_pointamp2_widthboost_D60E21_refresh_20260326.csv` |
| reviewer-scalar | comparison-layer scalar patch | `CLOSED - NEGATIVE` | Helpful in map-layer decomposition, but true runtime reviewer map degrades. | `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_reviewerscalar_D60E21.csv`, `output/kinetic_action_chain/model_chain_d60_reviewer_scalar_partial_audit_summary.csv` |
| reviewer ref-handoff | local `ref_D(D)` handoff | `CLOSED - NEGATIVE` | Does not cleanly remove `D≈5.627` and worsens nearby reviewer bands. | `output/kinetic_action_chain/model_chain_d60_reviewer_ref_handoff_audit_summary.csv` |
| reviewer-aware `refamp2` | reviewer ref-aware ref-amplitude family | `CLOSED - NEGATIVE` | No combination simultaneously improves `D≈5.627/5.898/6.169` without reopening `6.441/6.712`. | `output/kinetic_action_chain/model_chain_d60_reviewer_anchor_aware_refamp_audit_summary.csv` |
| `partial2` reviewer promotion | same-ref intrinsic partial extension promoted to reviewer | `CLOSED - NEGATIVE` | Same-ref leverage is real, but projection to reviewer reopens the wrong band. | `output/kinetic_action_chain/model_chain_fullwidthrefamp_partial2_pointamp2_widthboost_reviewer_hotspot_smoke_summary.csv` |
| `chi/g raw` cloud | high-D parabolic cloud on extraction-side `chi_LR/g_raw` | `CLOSED - NEGATIVE` | Consistently `D21`-friendly / `D60`-hostile. | `output/kinetic_action_chain/prolate_parabolic_cloud_extraction_audit_summary.csv` |
| observable high-D cloud | high-D cloud on observable `partial / point-amp` path | `CLOSED - NEGATIVE` | No shortlisted combo improves `D60`; best rows are materially worse. | `output/kinetic_action_chain/observable_highd_cloud_counterfactual_audit_summary.csv` |
| monotone `Gamma_ref(D)` gate | reviewer-level D-only `Gamma_ref` counterfactual | `CLOSED - NEGATIVE` | Has strong leverage, but never improves `D60` and `D21` at the same time. | `output/kinetic_action_chain/gamma_ref_highd_counterfactual_audit_summary.csv` |
| monotone Schur/logdet notch | D-only Schur/logdet scalar guard | `CLOSED - NEGATIVE` | No usable monotone separability; feature alone is not a viable guard. | `output/kinetic_action_chain/model_chain_schur_guard_correlation_audit_summary.csv` |

## Research-Value Families

These lines are not mainline candidates today, but they produced real signal and should be remembered as ingredient families rather than discarded outright.

| Family | Current role | Why it still matters | Reopen only if... | Main evidence |
| --- | --- | --- | --- | --- |
| calibrated `prolate` auxiliary mesh | extraction-side numerical method | Smoothness, mesh convergence, and splitting accuracy improved dramatically after calibration. | A downstream path can use the improved extraction without worsening reviewer D60. | `output/kinetic_action_chain/prolate_auxiliary_mesh_bias_audit_summary.csv`, `output/kinetic_action_chain/prolate_auxiliary_extraction_audit_calibrated_summary.csv`, `output/kinetic_action_chain/prolate_auxiliary_convergence_audit_calibrated_summary.csv` |
| `prolate` hybrid `chi/g` extraction | extraction-side profile substitution | Improves compatibility for `chi_LR`, `g2_raw`, `g3_raw`, and hotspot anchor metrics. | A downstream observable use can preserve D21 gains without harming D60. | `output/kinetic_action_chain/prolate_hybrid_gn_chi_propagation_audit_summary.csv`, `output/kinetic_action_chain/prolate_gn_chi_compatibility_audit_summary.csv` |
| `prolate` hybrid observable effect | reviewer-side counterfactual witness | Gives a clean directional signal: helps D21, harms D60. Useful as a sign diagnostic. | It is paired with an explicit D60-compensating mechanism. | `output/kinetic_action_chain/prolate_hybrid_reviewer_counterfactual_audit_summary.csv` |
| Schur/logdet conditional features | feature-level discriminator | Not strong enough alone, but does improve separability when conditioning an existing residual family. | Used only as a conditional feature, not as a standalone scalar patch. | `output/kinetic_action_chain/model_chain_schur_conditional_feature_audit_summary.csv` |
| same-ref intrinsic partial family | intrinsic same-ref diagnostic | Confirms there is still true leverage in localized partial-side structure before reviewer projection. | A future family can preserve that leverage under reviewer projection. | `output/kinetic_action_chain/model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_amp_extension_audit_summary.csv` |
| `Gamma_ref` high-D family | reviewer-level sign probe | Shows a real D60 lever exists in principle, but D-only monotone topology is wrong. | The next version is non-monotone or explicitly `(D, eta)`-conditional. | `output/kinetic_action_chain/gamma_ref_highd_counterfactual_audit_summary.csv` |

## Practical Reopen Rules

Use these rules before spending more time on a family:

1. Do not reopen any family in **Closed Negative Families** unless the new version changes the mechanism class, not just the parameters.
2. Treat `D21 D=6.4` as a hard guard on every reviewer-side candidate.
3. Treat `D60 D≈6.169/6.441` as the dominant reviewer-side hotspot pair.
4. If a family only helps `D21` or only helps `D60`, record it as a directional witness, not a promotion candidate.
5. For `prolate` and Schur-derived ideas, prefer extraction-side or conditional-feature audits before touching reviewer mainline.

## Current Bottom Line

- **Reviewer mainline stays frozen** on `cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost`.
- **Do not** spend more time on `pointamp3`, `widthband`, `partialguard`, reviewer-scalar, reviewer ref-handoff, observable cloud, or monotone `Gamma_ref(D)` gates unless a new mechanism class is introduced.
- **Keep alive as research ingredients**:
  - calibrated `prolate` extraction,
  - `prolate` hybrid `chi/g`,
  - Schur/logdet conditional features,
  - non-monotone or `(D, eta)`-conditional `Gamma_ref`.

# PSLT Roadmap Status

Last updated: 2026-04-29

This file is the submission-facing roadmap freeze.  It replaces the earlier running audit log with a compact decision map: what is active for submission, what is closed and should not be reopened before submission, and what remains only as optional future strengthening.  Detailed provenance is preserved in `paper/main.tex`, `FAMILY_DECISION_TABLE.md`, `paper/artifact_status.csv`, `output/repro/artifact_status.csv`, and the git history.

## Submission Freeze Reading

The current paper is in submission-closure mode.  The manuscript should not open new support/object/family searches before submission.  The only active work is packaging, consistency, reproducibility hygiene, and final PDF/readability checks.

## Submission Blockers

These items are not new theorem/object/family searches.  They are submission-facing consistency blockers exposed by the reviewer-style audit.  Any unresolved row below must be resolved before a final submission tag.

| Item | Status | Required resolution |
| --- | --- | --- |
| Table III WKB consistency | `RESOLVED / GATED` | The canonical artifact is `output/true_single_track/true_results.json`.  `paper/true_unified_table.tex` is regenerated from that artifact by `python3 code/check_table_iii_wkb_consistency.py --rewrite-table`, and the standing gate `python3 code/check_table_iii_wkb_consistency.py` enforces the displayed relation `r_1 = exp(-2 S_1)`.  Current pass: `max_D |log r_1(D) + 2 S_1(D)| = 2.182020e-04` for displayed rounded values. |
| Flavor-mixing "projector" terminology | `RESOLVED / WORDING GATED` | The matched `H->mumu` diagnostic map now calls the non-idempotent matrix a bounded nearest-neighbor row-stochastic flavor-mixing kernel, not a projector.  The text records the non-idempotence check, the eigenvalues `1`, `1-epsilon_mix`, `1-2 epsilon_mix`, and the positivity guard `epsilon_max < 1/2` with executable release setting `epsilon_max=0.25`. |
| Three-generation claim calibration | `RESOLVED / WORDING GATED` | Abstract, introduction, conclusion, limitations, README, and the submission status wording now use the defensible claim: baseline spectral-layer occupancy is concentrated in `N=1,2,3`, and the audited kinetic proxy has no fourth bound layer on the certified domain.  The current submission is not phrased as a full action-level proof of exactly three SM fermion generations. |
| Fig. 05 / `R3` plot-generator parity | `RESOLVED / GATED` | The canonical map table is now `output/gn_fp_impact/gn_baseline_replacement.csv`, with `baseline_fp_2d_full` reporting `f_R3_gt_0p90 = 0.9272222222`, `f_R3_gt_0p95 = 0.9263888889`, and `f_winner_gt_3 = 0.0002777778`.  The executable gate `python3 code/check_fig05_r3_parity.py` compares those fractions against the live `code/generate_plots.py` Fig. 05 plotting path before regenerated PNGs are accepted for the manuscript. |

| Area | Submission status | Operational reading |
| --- | --- | --- |
| Main paper | `READY FOR FINAL PACKAGING` | Abstract, front status table, limitations, and reviewer appendix ledger now expose the same closed/monitor/diagnostic statuses. |
| Artifact registry | `PASSING HYGIENE GATE` | `python3 code/build_artifact_status_registry.py` and `python3 code/check_artifact_status_hygiene.py` pass; old comparator/diagnostic/candidate outputs are not canonical baselines. |
| Reviewer mainline branch | `KEEP / MAINLINE` | Use `cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compd60snapmaplate`; keep the older widthboost branch only as historical comparator. |
| Release parity | `PASS` | `cell_direct_runtime` and `cell_direct_runtime_release_tuned` pass both active release gates; visibility parity remains qualified rather than strict all-direct. |
| Strict static width | `CLOSED - LOCAL THEOREM` | S25 is adopted only on the certified parent tube `D=6, R_C=400, R=60->80, c=640`; outside it, keep `Gamma_{N,l}^{(geo)}=omega_N exp(-2 S_{N,l})`. |
| Open-system exact bridge | `REVIEWER CLOSED / DIAGNOSTIC IN SCAN` | O39/O56/O71/O72 close the exact-bridge bottleneck; no new support/object/carrier/trace theorem before submission. |
| EYMH normalization | `STRUCTURALLY CLOSED / U14 MARGIN-CERTIFIED TOPOLOGY BRANCH` | U9 closes the structural normalization; U10 direct `y_2^{raw}(D_*)` is not strict no-topology adoption-safe; U12 proves the fixed-bridge crossing obstruction; U13 certifies explicit monotone topology-changing adoption as a no-loss one-component slab expansion; U14 certifies the direct branch margin; U11 closes the complement/Feshbach budget. |
| Fermionic lift and low-N counting | `FOUNDATIONAL CLOSED` | F1 closes the projected conformal Dirac lift; G1 closes low-mode ranks `1,2,3`; N1 closes the audited no-fourth-layer threshold. |
| Localized `A_l(D,eta,N)` tensor | `MONITOR / BOUND-ONLY SAFE` | Tensor extraction is useful, but all-valid lookup is not baseline-safe; only `bound_only_fallback` is production-safe. |
| `t_coh` and `eta_fp` gates | `DIAGNOSTIC / SCALED-ONLY SAFE` | `t_coh^(deph)` is diagnostic-only; `eta_fp(D)` is adoption-safe only as profile-scaled `eta_eff=eta*eta_amp(D)`. |

## Active For Submission

1. Keep the Table III WKB consistency gate passing with `python3 code/check_table_iii_wkb_consistency.py`.
2. Keep the non-idempotent matched-map flavor kernel wording and positivity guard intact.
3. Keep the three-generation claim calibration wording intact: report baseline first-three-layer occupancy plus audited no-fourth kinetic layer, not a full exactly-three SM-generation theorem.
4. Keep the Fig. 05 / `R3` plot-generator parity gate passing with `python3 code/check_fig05_r3_parity.py`.
5. Final PDF build and visual spot-check.
6. Artifact hygiene and registry consistency check.
7. Final commit/push after any wording-only readiness edits.

No active item above requires new support/object/family searches.  The Table III item is now a gated numerical-consistency repair rather than a new physics line.

## Frozen / Do Not Reopen Before Submission

- Do not reopen CAP, ECS, local Robin/Siegert, Whittaker backend replacement, tail-backend replacement, or unconstrained complex-pole branch picking.
- Do not reopen open-system support/object/carrier/trace-set searches; the exact-bridge bottleneck is reviewer-facing closed.
- Do not reopen EYMH normalization as a complement/Feshbach/kernel-family problem; U11 already closes the explicit `1e-12` complement guard.
- Do not silently baseline-adopt the U10 direct inserted-history `y_2^{raw}(D_*)` scalar as a no-topology replacement. U12 shows the first no-topology-change crossing occurs at continuation fraction `s=0.457084`, while the direct endpoint is `s=1`. If the direct scalar is used, U13/U14 require explicit topology-changing wording: the accepted set expands monotonically from `{D=8.8,9.6} x E21` to `{D=8.0,8.8,9.6} x E21`, with no losses, no new disconnected component, direct uniform `mu` margin `0.145404`, release margin `0.104937`, and gain-slab `2sigma` slack `0.363509 sigma`.
- Do not promote all-valid `A_l(D,eta,N)` tensor lookup; use bound-sector fallback if production promotion is ever desired.
- Do not tune `t_coh` caps or promote `t_coh^(deph)` as baseline.
- Do not adopt fully closed eta-axis replacements; keep only profile-scaled `eta_fp(D)` as adoption-safe.
- Do not reopen closed-negative reviewer families such as `pointamp3`, `widthband`, `partialguard`, reviewer-scalar, reviewer ref-handoff, observable cloud, monotone `Gamma_ref(D)`, `twolobe_latedstrip`, `twolobe_maplate`, or `twolobe_d8compd724snapmaplate` unless the mechanism class changes.

## Optional Future Strengthening

These are post-submission or revision-triggered items only:

| Item | Why optional | Reopen condition |
| --- | --- | --- |
| U10/U12 compensated bridge theorem | U13/U14 already supply the explicit topology-changing adoption branch plus margin certificate; a compensated theorem is only needed if a reviewer rejects topology-changing adoption but still asks for direct-scalar promotion. | A reviewer demands no acceptance-topology change while also rejecting the conservative release scalar. |
| High-N continuum-tail / D-continuous interpolation | Would strengthen N1 beyond the audited finite-volume/Sturm domain. | Needed only for a stronger global no-fourth-layer claim. |
| Static-width domain broadening | Would extend S25 beyond the local parent tube. | Only after submission; not a baseline blocker. |
| Open-system continuum/global lift | Would globalize the reviewer-closed exact bridge. | Only if the reservoir diagnostic becomes a baseline claim. |
| Microscopic EYMH bath derivation | Would turn the diagnostic Lindblad reservoir into a parent-side physical bath. | Only for a future mechanism paper or a direct reviewer request. |
| Research ingredients from `FAMILY_DECISION_TABLE.md` | Prolate, Lorentz, Schur/logdet conditional features, and non-monotone D/eta ideas remain useful signals. | Only with a genuinely new mechanism class and all existing guards preserved. |

## Current Release Gate Snapshot

| Branch | Gate | Acceptance mismatch | `max|Delta mu_mumu|` | Status |
| --- | --- | ---: | ---: | --- |
| `cell_direct_runtime` | `D21xE41` | `0.000000` | `0.000000` | PASS |
| `cell_direct_runtime` | `D60xE21` | `0.000000` | `0.000000` | PASS |
| `cell_direct_runtime_release_tuned` | `D21xE41` | `0.000000` | `0.092481` | PASS |
| `cell_direct_runtime_release_tuned` | `D60xE21` | `0.000000` | `0.837269` | PASS |

Interpretation: `full_direct` remains the release baseline.  `cell_direct_runtime` is the release-production parity branch.  `cell_direct_runtime_release_tuned` is a risk-weighted profile-anchored visibility parity branch, not a strict all-direct observable-sector proof.

## Submission Commands

Run these before a final submission tag:

```bash
python3 code/build_artifact_status_registry.py
python3 code/check_artifact_status_hygiene.py
python3 code/check_table_iii_wkb_consistency.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex  # from paper/
```

For final visual verification, render the abstract/status-map pages, the limitations pages, and the reviewer appendix ledger pages with `pdftoppm` and inspect the PNGs.

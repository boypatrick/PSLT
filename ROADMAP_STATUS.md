# PSLT Roadmap Status

Last updated: 2026-03-11

This file tracks roadmap items that are still open, partially promoted, or explicitly held behind release gates. It is intended as an operational companion to `/Users/boypatrick/codex/PSLT_quantam/README.md` and `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex`.

## Current Release Gate Snapshot

Sources:
- `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/chain_mode_cell_direct_audit_Dgrid21_Egrid41.csv`
- `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/chain_mode_cell_direct_audit_Dgrid60_Egrid21.csv`
- `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/chain_mode_cell_direct_audit_Dgrid21_Egrid41_cell_direct_runtime_release_tuned.csv`
- `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_release_tuned.csv`

| Branch | Gate | Acceptance mismatch | `max|Δμ_μμ|` | Release Threshold | Status |
| --- | --- | ---: | ---: | --- | --- |
| `cell_direct_runtime` | `D21xE41` | `0.000000` | `0.000000` | mismatch `<=0.01`, `max|Δμ|<=1.0` | PASS |
| `cell_direct_runtime` | `D60xE21` | `0.000000` | `0.000000` | mismatch `<=0.01`, `max|Δμ|<=1.0` | PASS |
| `cell_direct_runtime_release_tuned` | `D21xE41` | `0.000000` | `0.092481` | mismatch `<=0.01`, `max|Δμ|<=1.0` | PASS |
| `cell_direct_runtime_release_tuned` | `D60xE21` | `0.000000` | `0.837269` | mismatch `<=0.01`, `max|Δμ|<=1.0` | PASS |

Interpretation:
- `full_direct` remains the release baseline.
- `cell_direct_runtime` has now passed both release gates and is promoted as the release-production parity branch.
- `cell_direct_runtime_release_tuned` now also passes both gates with a D-only risk-weighted profile anchor `alpha(D)` rather than a fixed `0.99` blend.
- In physical terms, the promoted release-production parity branch closes the direct spectral-selection side of the chain (`g_N+\chi+A`) against `full_direct`, while the promoted `cell_direct_runtime_release_tuned` branch closes a **risk-weighted profile-anchored runtime-direct visibility parity path** in the observable sector.
- The current canonical release profile is `output/kinetic_action_chain/runtime_direct_visibility_alphaD_profile_release.csv`, selected by minimum mean anchor among gate-passing candidates (`alpha_min=0.96`, `alpha_max=0.99`, `power=1.0`, `mean(alpha)=0.9637`, `p90(alpha)=0.9734`).
- A post-promotion de-anchoring probe is now recorded in `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_visibility_alphaD_deanchoring_probe_summary.csv`. In the tested neighborhood, every lower-anchoring candidate re-opened the `D60xE21` gate through either acceptance mismatch (`alpha_min < 0.96`) or `max|Δμ_{\mu\mu}| > 1` (`alpha_max < 0.99`). The current canonical `alpha(D)` should therefore be treated as **close to the minimal necessary anchoring** for the present runtime-direct visibility implementation.

## Must-Have Before Stronger PRD Claim

| Item | Current State | Main Evidence | Next Gate / Requirement |
| --- | --- | --- | --- |
| Push risk-weighted `alpha(D)` anchoring closer to strict all-direct closure | Promoted parity branch now uses canonical D-only risk-weighted profile anchoring (`mean(alpha)=0.9637`). De-anchoring probes in the local neighborhood indicate this profile is already close to the minimum anchoring that keeps both `D21xE41` and `D60xE21` gates closed | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_visibility_alphaD_summary.json`, `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_visibility_alphaD_scan.csv`, `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_visibility_alphaD_deanchoring_probe_summary.csv` | Any further reduction should be treated as a runtime-direct visibility extraction problem rather than a simple `alpha(D)` retune; the next gate is to lower anchoring without reopening the `D60xE21` acceptance or `max|Δμ_{\mu\mu}|` failures |
| Full UV-to-EFT matching from EYMH action | Partial++++: UV-tree + finite one-loop + LL-RG audit, explicit layer-resolved operator-basis witness, refreshed input-tied finite-match comparator, Phase-2 diagonal-threshold window audit, action-normalized parent-side comparator, action-absolute comparator, a contrast-based non-local heat-kernel comparator, an absolute loop-prefactor comparator, and now an EYMH-side absolute loop-prefactor comparator are all implemented. The canonical diagonal-only input-tied witness `(diag_scale,offdiag_scale)=(1,0)` remains small-deformation (`max |Δμ_{\mu\mu}| = 7.87e-03`, zero acceptance mismatch). The action-normalized witness keeps zero acceptance mismatch with parent-side normalization factors `action_norm_diag`/`action_norm_offdiag` built from shell-gap and UV-coefficient invariants; the action-absolute witness replaces the external diagonal scale by local coefficient-alignment prefactors `action_abs_diag`/`action_abs_offdiag`; the `action_loop_contrast` witness shows that a pure local heat-kernel `a_2` route is not sufficient, while a non-local well/barrier curvature-contrast witness remains numerically exact at the basis level and small-deformation at map level (`mean |Δμ_{\mu\mu}| = 4.88e-04`, `p95 = 8.76e-04`, `max = 7.67e-03`, zero acceptance mismatch); `action_loop_absolute` adds a stable, nonzero absolute loop-prefactor witness (`hk_loop_prefactor_diag` median `= 0.351`, `p90 = 0.404`) with zero acceptance mismatch and small deformation relative to the refreshed constant baseline (`mean |Δμ_{\mu\mu}| = 3.37e-04`, `p95 = 6.92e-04`, `max = 5.33e-03`); and the new `action_loop_eymh_absolute` mode further screens this prefactor through EYMH-side mass-access and curvature-screen witnesses, remaining exact at the basis level and even closer to the refreshed constant baseline (`mean |Δμ_{\mu\mu}| = 4.17e-05`, `p95 = 7.63e-05`, `max = 6.46e-04`, zero acceptance mismatch) with a nonzero but smaller prefactor (`eymh_loop_prefactor_diag` median `= 0.0378`, `p90 = 0.0438`) | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:151`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:764`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_operator_basis_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_input_tied_vs_constant_refresh_mapdiff_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_input_tied_diag_window_summary_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_to_eft_summary_uv_action_normalized_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_operator_basis_summary_uv_action_normalized_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_normalized_pairwise_summary_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_to_eft_summary_uv_action_absolute_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_operator_basis_summary_uv_action_absolute_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_absolute_pairwise_summary_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_to_eft_summary_uv_action_loop_contrast_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_operator_basis_summary_uv_action_loop_contrast_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_contrast_pairwise_summary_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_to_eft_summary_uv_action_loop_absolute_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_operator_basis_summary_uv_action_loop_absolute_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_absolute_pairwise_summary_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_to_eft_summary_uv_action_loop_eymh_absolute_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_operator_basis_summary_uv_action_loop_eymh_absolute_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_absolute_pairwise_summary_D21E21.csv` | Replace the comparator-level EYMH-side absolute loop prefactor by a bona fide loop-matching normalization from the parent EYMH action. The remaining gap is no longer basis reconstruction, map stability, or the existence of a stable absolute prefactor witness, but deriving the correct absolute loop normalization rather than witness-level local, contrast, screened, or prefactor comparators |
| Open-system promotion from diagnostic to baseline candidate | Partial++: geometry + micro profiles implemented, multi-anchor + holdout gate closed, bridge audit verifies the microscopic system/bath/coupling chain (`delta_micro -> g_z/g_x -> tau_env/S(omega) -> gamma_phi/gamma_mix -> chi_eff`) together with exact `pslt_lib` loader reconstruction, and a Phase-2 `kappa_env` window audit now bounds the bath normalization as a uniform susceptibility on the localized two-level witness. Current windows: candidate `kappa_scale in [0.5,1.0]`, stable `kappa_scale in [0.25,1.5]` | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:149`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:857`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1526`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_micro_bridge_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_micro_kappa_window_summary.csv` | Derive microscopic EYMH bath `(L_k, gamma_k)` and replace the phenomenological normalization of `kappa_env`; the bridge bookkeeping and bath-stability window are now established |
| Full `(D, eta, N)` localized projection / model-chain unification | Partial: direct branches exist; release-production parity is closed for `g_N+\chi+A`, but release baseline still profile-based in the visibility sector | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:858` | Promote runtime-direct chain without failing parity gate, especially on the observable-side `B_N` closure |

## Good-to-Have

| Item | Current State | Main Evidence | Next Gate / Requirement |
| --- | --- | --- | --- |
| Propagate `t_coh` first-principles candidate into baseline | Benchmark only | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:865`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1638` | Show map-level stability and replace control-parameter baseline if acceptable |
| Propagate `eta_fp(D)` candidate into baseline | Benchmark only | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:866`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1702` | Show stable impact and production-quality closure |
| Fully localized channel-resolved barrier-leakage normalization | Partial: action-derived `A_l(D)` profile baseline exists | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:148`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:867`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1777` | Replace profile-level insertion with full scan-grid localized extraction |

## Post-Submission / Longer Horizon

| Item | Current State | Main Evidence | Next Gate / Requirement |
| --- | --- | --- | --- |
| Higher-dimensional parent-source derivation for projected Poisson source | Hook implemented, not integrated into baseline | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:145`, `/Users/boypatrick/codex/PSLT_quantam/README.md:172` | Provide a specific parent solution and connect it to baseline geometry |

## Completed Recently

These should not be re-listed as "not done":

- `full_direct_map` release mode exists and is published through `/Users/boypatrick/codex/PSLT_quantam/code/publish_full_direct_map.py`.
- Release-gate protocol exists and exports:
  - `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/full_direct_release_gate_status.csv`
  - `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/full_direct_worst_points_table.csv`
- Worst-point local refinement helper exists:
  - `/Users/boypatrick/codex/PSLT_quantam/code/run_localized_direct_refine_from_worst.py`
- `open_system_micro` multi-anchor + holdout calibration is implemented.
- `cell_direct_runtime` now passes both release gates exactly (`D21xE41` and `D60xE21`) and is promoted as the release-production parity branch.
- Minimal Dirac-conformal Yukawa-overlap roadmap text is implemented in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex`.

## Recommended Execution Order

1. Extend UV-to-EFT closure beyond the current UV-tree + finite one-loop + LL-RG audit by replacing witness-level absolute loop prefactors with bona fide EYMH loop normalization.
2. Decide whether `open_system_micro` can become a baseline candidate after microscopic bath derivation of the bath origin and absolute normalization.
3. Revisit strict all-direct visibility closure only if the runtime-direct visibility extraction itself is improved; the current `alpha(D)` profile is already close to the minimal necessary anchoring under the present implementation.
4. Revisit `t_coh`, `eta_fp(D)`, and fully localized barrier-leakage normalization after the main release path is stronger.
5. Treat higher-dimensional source integration as the longer-horizon follow-up rather than the immediate PRD blocker.

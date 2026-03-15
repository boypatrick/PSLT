# PSLT Roadmap Status

Last updated: 2026-03-16

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
| Close EYMH projected parent-kernel derivation from parent action | Advanced comparator closure: canonical EYMH normalization is now controlled by an exact projected `logdet + Schur` kernel, exact parent-kernel excess functional, exact block split, a background-normalized parent block-determinant / Schur-complement identity, a uniqueness audit showing that the mixed block is the geometric mean selected by the minimal symmetric-excess family, a ratio-warp audit showing that the first non-minimal extension is rejected on the D21xE21 fix grid, a symmetry/normalization audit showing that the same family is uniquely selected by zero normalization shift, unit symmetric degree, and zero antisymmetric tilt, a generator-affinity audit showing that the first local non-affine log-curvature corrections are also rejected, an excess-coordinate audit showing that the unique exact reference offsets are the identity participation block and the shell/background tree block themselves, a fixed-point Box-Cox audit showing that even among smooth tangent-normalized coordinate warps the unique exact point remains the linear additive excess choice, and a normal-coordinate audit showing that the first nonlinear local jets are uniquely rejected at zero second jet | `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_kernel_statement_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_block_split_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_blockdet_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_mix_geomean_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_ratio_warp_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_symnorm_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_parent_generator_affinity_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_excess_coordinate_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_fixedpoint_boxcox_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_action_loop_eymh_normal_coordinate_audit_summary_D21E21_fix.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` | Final gate is to derive, from the projected parent action itself rather than only from exact audit identities, why the low-mode fluctuation kernel is organized around the identity/background fixed points in zero-second-jet tangent-normalized linear excess coordinates, so that the locally affine multiplicative class emerges as the natural projected parent family before the canonical coefficient choices are enforced |
| Push risk-weighted `alpha(D)` anchoring closer to strict all-direct closure | Promoted parity branch now uses canonical D-only risk-weighted profile anchoring (`mean(alpha)=0.9637`). De-anchoring probes in the local neighborhood indicate this profile is already close to the minimum anchoring that keeps both `D21xE41` and `D60xE21` gates closed | `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_visibility_alphaD_summary.json`, `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_visibility_alphaD_scan.csv`, `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/runtime_direct_visibility_alphaD_deanchoring_probe_summary.csv` | Any further reduction should be treated as a runtime-direct visibility extraction problem rather than a simple `alpha(D)` retune; the next gate is to lower anchoring without reopening the `D60xE21` acceptance or `max|Δμ_{\mu\mu}|` failures |
| Open-system promotion from diagnostic to baseline candidate | Advanced closure: geometry + micro profiles implemented, multi-anchor + holdout gate closed, bridge audit verifies the microscopic system/bath/coupling chain (`delta_micro -> g_z/g_x -> tau_env/S(omega) -> gamma_phi/gamma_mix -> chi_eff`) together with exact `pslt_lib` loader reconstruction, the bath-factorization audit isolates `kappa_env` as the amplitude block on identifiable channels, the `kappa_env` absolute-normalization audit shows canonical constant normalization is the unique exact amplitude choice while nearby warps only improve holdout by leaking into bath shape, the parent-bath statement rewrites the witness as `K_bath = kappa_env sqrt(K_sys) K_spec sqrt(K_sys)`, the family/log-coordinate/normal-coordinate/generator-affinity audits uniquely return the canonical projected bath class (`m=u=v=0`, `p_sys=p_spec=0`, `zeta_sys=zeta_spec=0`, `q_ss=q_bb=q_sb=0`), the cocycle audit shows that after dividing out `kappa_env` the projected bath block obeys an exact additive pairwise cocycle with flat triangle defects in the canonical log variables, and the potential audit upgrades this to an exact single-valued anchored potential on the projected bath block | `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_bath_factorization_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_kappa_absolute_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_parent_bath_statement_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_parent_bath_family_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_parent_bath_log_coordinate_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_parent_bath_normal_coordinate_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_parent_bath_generator_affinity_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_parent_bath_cocycle_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_parent_bath_potential_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` | Final gate is to derive, from the projected bath action itself rather than only from exact family rejections, cocycle identities, or anchored-potential closure, why the localized open-system witness naturally organizes into this affine log generator and its associated anchored potential built from the fixed-point normalized system and spectral blocks |
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

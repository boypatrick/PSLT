# PSLT Roadmap Status

Last updated: 2026-03-08

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
| `cell_direct_runtime_release_tuned` | `D21xE41` | `0.000000` | `0.083678` | mismatch `<=0.01`, `max|Δμ|<=1.0` | PASS |
| `cell_direct_runtime_release_tuned` | `D60xE21` | `0.000000` | `0.688093` | mismatch `<=0.01`, `max|Δμ|<=1.0` | PASS |

Interpretation:
- `full_direct` remains the release baseline.
- `cell_direct_runtime` has now passed both release gates and is promoted as the release-production parity branch.
- `cell_direct_runtime_release_tuned` now also passes both gates after fixing the release blend at `runtime_direct_b_release_profile_blend=0.99`.
- In physical terms, the promoted release-production parity branch closes the direct spectral-selection side of the chain (`g_N+\chi+A`) against `full_direct`, while the promoted `cell_direct_runtime_release_tuned` branch closes a **profile-anchored runtime-direct visibility parity path** in the observable sector.
- Because the observable-side branch is promoted with a `0.99` profile anchor, it should not be presented as a strict all-direct closure; the remaining upgrade task is to reduce profile anchoring while preserving gate pass.

## Open Roadmap Items

| Priority | Item | Current State | Main Evidence | Next Gate / Requirement |
| --- | --- | --- | --- | --- |
| P1 | Reduce profile anchoring in promoted runtime-direct visibility branch (`cell_direct_runtime_release_tuned`) | Promoted parity branch with `runtime_direct_b_release_profile_blend=0.99` | `/Users/boypatrick/codex/PSLT_quantam/README.md:143`, `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/full_direct_release_gate_status.csv` | Lower profile anchoring while preserving gate pass on `D21xE41` and `D60xE21` |
| P1 | Full UV-to-EFT matching from EYMH action | Partial++: UV-tree + finite one-loop + LL-RG audit, explicit layer-resolved operator-basis witness, refreshed input-tied finite-match comparator, and Phase-2 diagonal-threshold window audit implemented. The canonical diagonal-only witness `(diag_scale,offdiag_scale)=(1,0)` remains small-deformation (`max |Δμ_{\mu\mu}| = 7.87e-03`, zero acceptance mismatch), with a conservative comparator window `diag_scale in [0.25,1.0]` and an extended stable window `diag_scale in [0.25,1.5]`; off-diagonal tying remains numerically inactive | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:151`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:764`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_operator_basis_summary.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_input_tied_vs_constant_refresh_mapdiff_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_input_tied_vs_constant_refresh_scale_summary_D21E21.csv`, `/Users/boypatrick/codex/PSLT_quantam/paper/hll_uv_input_tied_diag_window_summary_D21E21.csv` | Derive or calibrate the diagonal threshold susceptibility from parent-action / loop-level matching, then decide whether the input-tied witness should become a baseline candidate; off-diagonal tying no longer appears to be the limiting gap in the current basis |
| P1 | Open-system promotion from diagnostic to baseline candidate | Partial+: geometry + micro profiles implemented, multi-anchor + holdout gate closed, and a new bridge audit now verifies the microscopic system/bath/coupling chain (`delta_micro -> g_z/g_x -> tau_env/S(omega) -> gamma_phi/gamma_mix -> chi_eff`) together with exact `pslt_lib` loader reconstruction at the exported D-knot points | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:149`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:857`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1526`, `/Users/boypatrick/codex/PSLT_quantam/paper/chi_open_system_micro_bridge_summary.csv` | Derive microscopic EYMH bath `(L_k, gamma_k)` and rerun baseline-candidate gate; the remaining gap is bath-side normalization/derivation rather than the bridge bookkeeping itself |
| P1 | Full `(D, eta, N)` localized projection / model-chain unification | Partial: direct branches exist; release-production parity is closed for `g_N+\chi+A`, but release baseline still profile-based in the visibility sector | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:858` | Promote runtime-direct chain without failing parity gate, especially on the observable-side `B_N` closure |
| P2 | Propagate `t_coh` first-principles candidate into baseline | Benchmark only | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:865`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1638` | Show map-level stability and replace control-parameter baseline if acceptable |
| P2 | Propagate `eta_fp(D)` candidate into baseline | Benchmark only | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:866`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1702` | Show stable impact and production-quality closure |
| P2 | Fully localized channel-resolved barrier-leakage normalization | Partial: action-derived `A_l(D)` profile baseline exists | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:148`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:867`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1777` | Replace profile-level insertion with full scan-grid localized extraction |
| P3 | Higher-dimensional parent-source derivation for projected Poisson source | Hook implemented, not integrated into baseline | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:145`, `/Users/boypatrick/codex/PSLT_quantam/README.md:172` | Provide a specific parent solution and connect it to baseline geometry |

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

1. Reduce profile anchoring in `cell_direct_runtime_release_tuned` while preserving release-gate pass.
2. Extend UV-to-EFT closure beyond current UV-tree + finite one-loop + LL-RG audit.
3. Decide whether `open_system_micro` can become a baseline candidate after microscopic bath derivation.
4. Revisit `t_coh`, `eta_fp(D)`, and higher-dimensional source integration only after the main release path is stable.

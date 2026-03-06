# PSLT Roadmap Status

Last updated: 2026-03-06

This file tracks roadmap items that are still open, partially promoted, or explicitly held behind release gates. It is intended as an operational companion to `/Users/boypatrick/codex/PSLT_quantam/README.md` and `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex`.

## Current Release Gate Snapshot

Source: `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/full_direct_release_gate_status.csv`

| Metric | Value | Release Threshold | Status |
| --- | ---: | ---: | --- |
| `worst_frac_acceptance_mismatch` | `0.029036` | `<= 0.010000` | HOLD |
| `worst_max_abs_delta_mu_mumu` | `5.387299` | `<= 1.000000` | HOLD |
| `small_frac_acceptance_mismatch` | `0.029036` | reference | above target |
| `small_max_abs_delta_mu_mumu` | `5.387299` | reference | above target |
| `large_frac_acceptance_mismatch` | `0.007143` | reference | within mismatch target |
| `large_max_abs_delta_mu_mumu` | `4.344758` | reference | above target |

Interpretation:
- `full_direct` is the current release baseline.
- `cell_direct_runtime_release_tuned` is still a candidate branch and has not passed promotion.
- The dominant remaining drift is in the observable sector (`B_N` / EFT-operator runtime-direct branch), not in the localized-direct small-surface bias audit itself.

## Open Roadmap Items

| Priority | Item | Current State | Main Evidence | Next Gate / Requirement |
| --- | --- | --- | --- | --- |
| P0 | Promote per-cell direct production branch | Partial, not promoted | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:146`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:147`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:661` | Reduce parity drift so release gate passes on `D21xE41 + D60xE21` |
| P0 | Promote all-direct visibility branch (`cell_direct_runtime_release_tuned`) | HOLD | `/Users/boypatrick/codex/PSLT_quantam/README.md:142`, `/Users/boypatrick/codex/PSLT_quantam/README.md:143`, `/Users/boypatrick/codex/PSLT_quantam/output/kinetic_action_chain/full_direct_release_gate_status.csv` | Bring `max|Δμ_μμ|` below `1.0` and mismatch below `0.01` |
| P1 | Full UV-to-EFT matching from EYMH action | Partial: UV-tree + finite one-loop + LL-RG audit implemented | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:151`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:764`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:765` | Extend to explicit operator-basis and loop-level matching beyond scan-level closure |
| P1 | Open-system promotion from diagnostic to baseline candidate | Partial: geometry + micro profiles implemented, not promoted | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:149`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:857`, `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:1526` | Derive microscopic EYMH bath `(L_k, gamma_k)` and rerun baseline-candidate gate |
| P1 | Full `(D, eta, N)` localized projection / model-chain unification | Partial: direct branches exist, release baseline still profile-based for visibility | `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex:858` | Promote runtime-direct chain without failing parity gate |
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
- Minimal Dirac-conformal Yukawa-overlap roadmap text is implemented in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex`.

## Recommended Execution Order

1. Reduce release-gate drift for `cell_direct_runtime_release_tuned`.
2. Promote the runtime-direct visibility branch only after gate pass.
3. Extend UV-to-EFT closure beyond current UV-tree + finite one-loop + LL-RG audit.
4. Decide whether `open_system_micro` can become a baseline candidate after microscopic bath derivation.
5. Revisit `t_coh`, `eta_fp(D)`, and higher-dimensional source integration only after the main release path is stable.

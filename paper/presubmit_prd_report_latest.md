# PRD Pre-Submission Freeze Report

- Generated (UTC): 2026-03-08T14:24:38Z
- Run ID: presubmit
- Git branch: main
- Git commit: 5eed2289ebcb6f650c1cdf08fc8d2444e82a05e3
- Git dirty at report time: True

## Key Metrics

- Baseline `f(R3>0.90)`: 0.926944
- `f(chi2_mumu<4)`: 0.150000
- Best `chi2_mumu`: 2.207981e-05
- UV audit `max |delta mu_mumu|`: 9.738091e-03
- UV audit `max |delta C_match_mumu|`: 1.000000e-30
- Full-direct release gate: **GO**
- Runtime-tuned promotion gate: **GO**

## Release Gates

| Gate | Decision | Pass | Key metrics vs thresholds |
|---|---|---|---|
| `full_direct_release` | GO | PASS | small_frac_winner_mismatch=0.000000e+00<= 0.010000; small_max_abs_delta_R3=1.788196e-05<= 1.000000e-03; small_max_abs_delta_mu_mumu=3.369519e-03<= 0.100000; large_frac_winner_mismatch=0.000000e+00<= 0.010000; large_max_abs_delta_R3=0.000000e+00<= 1.000000e-03; large_max_abs_delta_mu_mumu=0.000000e+00<= 0.100000 |
| `runtime_release_tuned_promotion` | GO | PASS | small_frac_acceptance_mismatch=0.000000e+00<= 0.010000; small_max_abs_delta_mu_mumu=0.092481<= 1.000000; large_frac_acceptance_mismatch=0.000000e+00<= 0.010000; large_max_abs_delta_mu_mumu=0.837269<= 1.000000 |

## Checks

| Check | Status | Detail |
|---|---|---|
| `file:paper/main.tex` | PASS | exists and non-empty |
| `file:paper/main.pdf` | PASS | exists and non-empty |
| `file:paper/hll_signal_strength_summary.csv` | PASS | exists and non-empty |
| `file:paper/hll_uv_to_eft_summary.csv` | PASS | exists and non-empty |
| `file:paper/hll_rge_sensitivity.csv` | PASS | exists and non-empty |
| `file:paper/full_direct_map_release_summary.csv` | PASS | exists and non-empty |
| `file:output/first_principles_migration/first_principles_migration_summary.csv` | PASS | exists and non-empty |
| `file:output/hll_signal_strength/hll_signal_strength_run_meta_baseline.json` | PASS | exists and non-empty |
| `paper_sync` | PASS | main.pdf mtime >= main.tex mtime |
| `hll_summary_row_mumu` | PASS | mumu row found |
| `uv_summary_present` | PASS | uv summary row found |
| `chi2_fraction_consistency` | PASS | f_chi2_le_4_uv_rge=0.150000, hll_mumu_f_chi2_le_4=0.150000 |
| `rge_sensitivity_parameters` | PASS | found=['gamma_diag', 'gamma_offdiag', 'kappa_diag', 'kappa_offdiag', 'mu_low'] |
| `baseline_migration_row` | PASS | baseline_fp2d_full_action_grid row found |
| `run_meta_has_finite_match_knobs` | PASS | kappa_diag=0.0, kappa_offdiag=0.0 |
| `release_gate_rows_present` | PASS | small/large direct-bias rows and profile-anchored tuned parity rows found |
| `gate_full_direct_release` | PASS | decision=GO; small_frac_winner_mismatch=0.000000e+00<= 0.010000, small_max_abs_delta_R3=1.788196e-05<= 1.000000e-03, small_max_abs_delta_mu_mumu=3.369519e-03<= 0.100000, large_frac_winner_mismatch=0.000000e+00<= 0.010000, large_max_abs_delta_R3=0.000000e+00<= 1.000000e-03, large_max_abs_delta_mu_mumu=0.000000e+00<= 0.100000 |
| `gate_runtime_release_tuned_promotion_rows` | PASS | decision=GO; small_frac_acceptance_mismatch=0.000000e+00<= 0.010000, small_max_abs_delta_mu_mumu=0.092481<= 1.000000, large_frac_acceptance_mismatch=0.000000e+00<= 0.010000, large_max_abs_delta_mu_mumu=0.837269<= 1.000000 |

**Overall:** PASS

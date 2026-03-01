# PRD Pre-Submission Freeze Report

- Generated (UTC): 2026-03-01T14:10:28Z
- Run ID: 20260301T140833Z_3e61a1a_prd_freeze
- Git branch: main
- Git commit: 3e61a1a36ce55a0b811716757e95fdd9e3213e1d
- Git dirty at report time: True

## Key Metrics

- Baseline `f(R3>0.90)`: 0.926944
- `f(chi2_mumu<4)`: 0.150000
- Best `chi2_mumu`: 2.207981e-05
- UV audit `max |delta mu_mumu|`: 9.738091e-03
- UV audit `max |delta C_match_mumu|`: 1.000000e-30

## Checks

| Check | Status | Detail |
|---|---|---|
| `file:paper/main.tex` | PASS | exists and non-empty |
| `file:paper/main.pdf` | PASS | exists and non-empty |
| `file:paper/hll_signal_strength_summary.csv` | PASS | exists and non-empty |
| `file:paper/hll_uv_to_eft_summary.csv` | PASS | exists and non-empty |
| `file:paper/hll_rge_sensitivity.csv` | PASS | exists and non-empty |
| `file:output/first_principles_migration/first_principles_migration_summary.csv` | PASS | exists and non-empty |
| `file:output/hll_signal_strength/hll_signal_strength_run_meta_baseline.json` | PASS | exists and non-empty |
| `paper_sync` | PASS | main.pdf mtime >= main.tex mtime |
| `hll_summary_row_mumu` | PASS | mumu row found |
| `uv_summary_present` | PASS | uv summary row found |
| `chi2_fraction_consistency` | PASS | f_chi2_le_4_uv_rge=0.150000, hll_mumu_f_chi2_le_4=0.150000 |
| `rge_sensitivity_parameters` | PASS | found=['gamma_diag', 'gamma_offdiag', 'kappa_diag', 'kappa_offdiag', 'mu_low'] |
| `baseline_migration_row` | PASS | baseline_fp2d_full_action_grid row found |
| `run_meta_has_finite_match_knobs` | PASS | kappa_diag=0.0, kappa_offdiag=0.0 |

**Overall:** PASS

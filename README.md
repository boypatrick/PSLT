# PSLT Research Bundle (2026-02-01)

**Title:** The Projection Spectral Layer Theory (PSLT): A Rank-2 Computable Closure for the Three-Generation Structure and Higgs Signal Strength
**Author:** Bo-Yu Chen (Independent Researcher)
**Date:** February 1, 2026

This bundle contains the complete source code, data, and manuscript for the PSLT verification project.

**Reproducibility badge:** clone -> one command -> packaged Fig/Table artifacts with manifest/checksums.

## Directory Structure
- `paper/`: Contains the main Latex manuscript (`main.tex`).
- `code/`: Contains the Python verification scripts.
  - `pslt_lib.py`: The core unified library (Kinetics, Visibility, Parameters).
  - `generate_plots.py`: Main plotting script for phase maps and H->mumu proxy plots.
  - `scan_hll_signal_strengths.py`: Proxy maps and summaries for H->ee, H->mumu, H->tautau.
  - `scan_hll_uv_to_eft_matching.py`: UV-tree -> finite one-loop match -> LL-RG Wilson-map audit (`C_{eH}^{tree}`, `C_{eH}^{match}`, `C_{eH}^{IR}`) and map-level drift summaries.
  - `publish_full_direct_map.py`: release-mode full_direct runner (main map + small complete direct audit + large spot-check direct audit + merged summary table).
  - `scan_hll_uv_envelope.py`: nonzero finite-match UV control envelope (`mu_mumu` pointwise min/max band + case summary).
  - `build_artifact_status_registry.py`: writes canonical vs diagnostic vs legacy artifact status tables for reviewer hygiene.
  - `scripts/repro/presubmit_prd_freeze.sh`: pre-submission freeze check runner (recompute, compile, package, and report).
- `data/`: Contains PDG data files (`pdg_leptons.json`, `pdg_quarks.json`).
- `output/`: Contains the generated plots and figures used in the paper.
- `scripts/repro/`: One-click reproducibility pipeline and packaging tools.
- `repro/`: Reproducibility docs, artifact map, and run outputs.

## Reproduction Instructions (One Click)

### 1. Generate + Package Fig/Table Artifacts
From repository root:

```bash
bash scripts/repro/reproduce_paper.sh
```

This runs the reproducible pipeline, writes step logs, and packages normalized outputs under:
- `repro/runs/<RUN_ID>/figures/Fig_XX_*.png`
- `repro/runs/<RUN_ID>/tables/Tab_XX_*.csv`
- `repro/runs/<RUN_ID>/manifest.json`
- `repro/runs/<RUN_ID>/checksums.sha256`

`repro/latest` points to the latest packaged run.
Detailed conventions and layout are documented in `repro/README.md`.

### 2. Optional Paper Compile in Same Run
If you also want to compile `paper/main.tex` in the same command:

```bash
bash scripts/repro/reproduce_paper.sh --with-paper
```

### 3. Package Existing Outputs Only
If you already ran scripts manually and only want standardized packaging:

```bash
bash scripts/repro/reproduce_paper.sh --package-only
```

### 4. Legacy Manual Sequence (Advanced)
The old manual script-by-script sequence is still valid, but the reproducible entrypoint above is now the recommended path.

### 5. PRD/JHEP Pre-Submission Freeze Check
Run the targeted pre-submission check pipeline:

```bash
bash scripts/repro/presubmit_prd_freeze.sh
```

This executes core UV/EFT scans, compiles `paper/main.tex`, packages reproducibility artifacts, and writes:
- `repro/runs/<RUN_ID>/presubmit_prd_report.json`
- `repro/runs/<RUN_ID>/presubmit_prd_report.md`
- `paper/presubmit_prd_report_latest.json`
- `paper/presubmit_prd_report_latest.md`

### 6. Artifact Status Registry (Reviewer Hygiene)
To avoid misreading old comparator outputs as baseline claims:

```bash
python3 code/build_artifact_status_registry.py
```

This writes:
- `output/repro/artifact_status.csv`
- `paper/artifact_status.csv`

Each row marks an artifact as `canonical_baseline`, `diagnostic_variant`, or `legacy_comparator`.


## Requirements
- Python 3.8+
- `numpy`, `scipy`, `matplotlib`
- `latexmk` + `pdflatex` (only if `--with-paper`)

## Key Results
- **Three-Generation Stability:** In the current full-grid action-derived profile baseline, the theory gives Generation Ratio $\mathcal{R}_3>90\%$ over **92.7%** of the sampled $(D,\eta)$ grid, with $\mathcal{R}_3>95\%$ over **21.7%** of the same grid.
- **H→μμ EFT/Wilson-Matched Compatibility (UV+LL-RG baseline):** The illustrative acceptance region ($\chi^2<4$) occupies about **15.0%** of the sampled grid, with best grid point near $(D,\eta)\approx(4.00,0.264)$ and $\chi^2\approx 2.21\times10^{-5}$.
- **Overlap Visibility + High-$N$ Stability:** Baseline visibility uses overlap-defined `b_mode=overlap_2d` profiles from `y_N^{eff}(D)` extraction (with tracked microcanonical windows), while baseline `g_mode=fp_2d_full` uses phase-space normalization (no Cardy anchor) plus bounded low-$N$ microcanonical windowing and controlled high-$N$ tail prescription.

## First-Principles Migration Status
- **g_N migration comparator:** `code/scan_gn_profile_impact.py` compares `g_mode=fp_2d_full/cardy/fp_1d/fp_2d`; relative to baseline `fp_2d_full`, the largest drift is the `fp_2d` case with $\Delta f(\mathcal{R}_3>0.90)=-0.0669$, while high-$N$ runaway remains controlled at `f(N_win>3)\approx2.78\times10^{-4}`.
- **g_N low-N alignment + N_max convergence:** `code/plot_gn_cardy_vs_phase_space.py` and `code/scan_gn_nmax_convergence.py` export low-$N$ Cardy-vs-phase-space overlays and $N_{\max}=20/30/40$ stability tables.
- **g_N baseline cross-check:** `code/scan_gn_baseline_replacement.py` now uses baseline `fp_2d_full` and compares against legacy `cardy` reference, exporting:
  - `output/gn_fp_impact/gn_baseline_replacement.csv`
  - `output/gn_fp_impact/gn_baseline_replacement_nmax.csv`
- **Open-system chi module (geometry + micro gate):** `code/scan_chi_open_system_sensitivity.py` keeps the geometry-only diagnostic band, while `extract_chi_open_system_micro.py` + `calibrate_kappa_env_micro_anchor.py` now run a **multi-anchor** calibration (default anchors `D={6,9,12,15,18}`) with explicit **holdout** validation (`kappa_env_anchor_holdout.csv`) before `assess_open_system_micro_baseline_candidate.py` evaluates baseline-candidate gates.
- **Surrogate-vs-direct chi transfer audit (B3):** `code/scan_surrogate_vs_action_points.py` reports point-level drift between interpolated `chi_LR(D)` and direct fine localized extraction injection, exporting:
  - `output/chi_fp_2d/surrogate_vs_action_points.csv`
  - `output/chi_fp_2d/surrogate_vs_action_points_summary.csv`
- **Localized-direct full D-grid mode (D60):** `code/scan_localized_direct_surface_bias.py` now promotes $\chi_{LR}(D)$ and $\tilde A_\ell(D)$ to localized-direct `Dgrid60` profiles, and scan scripts prioritize these grid-aligned profiles (no D-interpolation on scan points; off-grid references use standard profile evaluation). Representative exports:
  - `output/kinetic_action_chain/localized_direct_profiles_chi_Dgrid60.csv`
  - `output/kinetic_action_chain/localized_direct_profiles_superrad_Dgrid60.csv`
  - `output/kinetic_action_chain/localized_direct_surface_summary_Dgrid60_Egrid21.csv`
- **Primary full_direct release mode:** `code/publish_full_direct_map.py` publishes one integrated release table combining:
  - main-map strict full_direct baseline (`D60 x E60`)
  - small-surface complete localized-direct audit (`D21 x E41`)
  - large-surface spot-check localized-direct audit (`D60 x E21`)
  - chain-mode parity audit (`auto` vs `full_direct`, `D21 x E41`)
  and exports `output/kinetic_action_chain/full_direct_map_release_summary.csv` (mirrored to `paper/full_direct_map_release_summary.csv`).
  Baseline scan scripts now default to `chain_mode=full_direct` (`scan_hll_signal_strengths.py`, `scan_hll_uv_to_eft_matching.py`), while `auto` is retained for comparator audits.
- **Core-parameter robustness table (B2):** `code/scan_core_param_robustness.py` exports local one-at-a-time windows for `c_eff`, `nu`, `p_B` and their map fractions:
  - `output/robustness/core_param_robustness_table.csv`
  - `output/robustness/core_param_robustness_cases.csv`
- **Cross-module summary:** `code/scan_first_principles_migration_summary.py` aggregates baseline `fp_2d_full+localized` vs legacy `cardy+localized` vs `fp_2d_full+open_system` into:
  - `output/first_principles_migration/first_principles_migration_summary.csv`
  - `output/first_principles_migration/first_principles_migration_summary.png`
- **Full-chain consistency comparator (kinetic unification audit):** `code/scan_action_chain_consistency.py` keeps a legacy surrogate-vs-action comparator run for historical context; the current strict chain-mode parity evidence is provided by `code/scan_chain_mode_full_direct_audit.py`. Comparator exports:
  - `output/kinetic_action_chain/action_chain_consistency_summary.csv`
  - `output/kinetic_action_chain/action_chain_consistency.png`
- **UV-to-EFT matching audit:** `code/scan_hll_uv_to_eft_matching.py` exports per-point UV-tree / finite-match / IR Wilson diagonals and running metadata on the same scan grid:
  - `output/hll_uv_matching/hll_uv_to_eft_map.csv`
  - `output/hll_uv_matching/hll_uv_to_eft_summary.csv`
  - `output/hll_uv_matching/hll_uv_to_eft_maps.png`
- **RG-window robustness (UV+LL-RG controls):** `code/scan_hll_rge_sensitivity.py` scans one-at-a-time windows for `mu_low`, `gamma_diag`, `gamma_offdiag`, `kappa_diag`, and `kappa_offdiag`, exporting:
  - `output/robustness/hll_rge_sensitivity_cases.csv`
  - `output/robustness/hll_rge_sensitivity_table.csv`
- **UV control envelope (nonzero finite-match center):** `code/scan_hll_uv_envelope.py` scans one-at-a-time UV windows around nonzero `(kappa_diag,kappa_offdiag)` and exports:
  - `output/robustness/hll_uv_envelope_cases.csv`
  - `output/robustness/hll_uv_envelope_map.csv`
  - `output/robustness/hll_uv_envelope_summary.csv`
  - `output/robustness/hll_uv_envelope.png`
  - mirrored copies under `paper/`.

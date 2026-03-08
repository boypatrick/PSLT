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
  - `scan_hll_uv_operator_basis_audit.py`: explicit layer-resolved operator-basis audit for the UV-tree/finite-match/LL-RG closure, including basis reconstruction residuals and `mu_mumu` reconstruction checks.
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

The presubmit report now includes explicit auto-gated release decisions:
- `full_direct_release` (`GO/HOLD`) from direct-bias thresholds on `D21xE41` + `D60xE21`
- `runtime_release_tuned_promotion` (`GO/HOLD`) from tuned-candidate parity thresholds on the same gates

To enforce gate decisions as CI-style pass/fail:

```bash
bash scripts/repro/presubmit_prd_freeze.sh --skip-recompute --require-gate-go full_direct
```

Supported modes for `--require-gate-go` are:
- `full_direct` (require `full_direct_release=GO`)
- `runtime_tuned` (require `runtime_release_tuned_promotion=GO`)
- `both` (require both gates are `GO`)

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
- **Fully Normalized EFT Visibility + High-$N$ Stability:** Baseline visibility now uses `b_mode=eft_operator_norm`, i.e. operator-normalized layer weights built from overlap-extracted flavor-layer couplings and mediator scales with finite one-loop + LL-RG normalization; legacy `overlap_2d`/Yukawa visibility remains available as comparator. Baseline `g_mode=fp_2d_full` uses phase-space normalization (no Cardy anchor) plus bounded low-$N$ microcanonical windowing and controlled high-$N$ tail prescription.

## First-Principles Migration Status
- A concise open-items tracker is maintained in `/Users/boypatrick/codex/PSLT_quantam/ROADMAP_STATUS.md`, including current release-gate numbers and remaining promotion blockers.
- **g_N migration comparator:** `code/scan_gn_profile_impact.py` compares `g_mode=fp_2d_full/cardy/fp_1d/fp_2d`; relative to baseline `fp_2d_full`, the largest drift is the `fp_2d` case with $\Delta f(\mathcal{R}_3>0.90)=-0.0669$, while high-$N$ runaway remains controlled at `f(N_win>3)\approx2.78\times10^{-4}`.
- **g_N low-N alignment + N_max convergence:** `code/plot_gn_cardy_vs_phase_space.py` and `code/scan_gn_nmax_convergence.py` export low-$N$ Cardy-vs-phase-space overlays and $N_{\max}=20/30/40$ stability tables.
- **g_N baseline cross-check:** `code/scan_gn_baseline_replacement.py` now uses baseline `fp_2d_full` and compares against legacy `cardy` reference, exporting:
  - `output/gn_fp_impact/gn_baseline_replacement.csv`
  - `output/gn_fp_impact/gn_baseline_replacement_nmax.csv`
- **Open-system chi module (geometry + micro gate):** `code/scan_chi_open_system_sensitivity.py` keeps the geometry-only diagnostic band, while `extract_chi_open_system_micro.py` + `calibrate_kappa_env_micro_anchor.py` now run a **multi-anchor** calibration (default anchors `D={6,9,12,15,18}`) with explicit **holdout** validation (`kappa_env_anchor_holdout.csv`) before `assess_open_system_micro_baseline_candidate.py` evaluates baseline-candidate gates. The new bridge audit `code/scan_chi_open_system_micro_bridge.py` aligns the microscopic chain as
  - system: `delta_micro`
  - localized couplings: `g_z_micro`, `g_x_micro`
  - bath: `tau_env`, `Szz_0`, `Sxx_delta`
  - rates: `gamma_phi_micro`, `gamma_mix_micro`
  - observable: `chi_eff_micro`
  and verifies exact formula/loader reconstruction through:
  - `output/chi_open_system/chi_open_system_micro_bridge_map.csv`
  - `output/chi_open_system/chi_open_system_micro_bridge_summary.csv`
  - `output/chi_open_system/chi_open_system_micro_bridge.png`
  On the current calibrated `D=4..20` knot set, the bridge residuals are numerically negligible (`max |Δgamma_phi| = 2.03e-20`, `max |Δgamma_mix| = 4.40e-14`, `max |Δchi_eff(loader)| = 5.58e-17`), so the remaining missing piece is the bath-side EYMH derivation, not the system-to-Lindblad bookkeeping.
  `code/scan_chi_open_system_micro_kappa_window.py` now adds a Phase-2 bath-normalization audit: treating `kappa_env` as a uniform susceptibility that rescales both `gamma_phi` and `gamma_mix` while leaving the localized two-level Hamiltonian fixed. The resulting canonical summaries are
  - `output/chi_open_system/chi_open_system_micro_kappa_window_scan.csv`
  - `output/chi_open_system/chi_open_system_micro_kappa_window_summary.csv`
  - `output/chi_open_system/chi_open_system_micro_kappa_window_ratio_scan.csv`
  - `output/chi_open_system/chi_open_system_micro_kappa_window.png`
  For the current micro witness, the calibration-consistent candidate window is `kappa_scale in [0.5, 1.0]`, while a broader stable scan window `kappa_scale in [0.25, 1.5]` keeps the map-level fractions unchanged and only relaxes the holdout tolerance slightly. This promotes `kappa_env` from a single fitted constant to a bounded bath-normalization parameter with an explicit physical/stability interpretation.
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
  - small-surface chain-mode parity audit (`full_direct` vs `cell_direct_runtime`, labeled `runtime_profile`, `D21 x E41`)
  - small-surface chain-mode parity audit (`full_direct` vs `cell_direct_runtime_release_tuned`, labeled `runtime_bnorm_release_tuned_profile_anchored`, `D21 x E41`)
  - large-surface chain-mode parity audit (`full_direct` vs `cell_direct_runtime`, labeled `runtime_profile`, `D60 x E21`)
  - large-surface chain-mode parity audit (`full_direct` vs `cell_direct_runtime_release_tuned`, labeled `runtime_bnorm_release_tuned_profile_anchored`, `D60 x E21`)
  - large-surface chain-mode parity audit (`full_direct` vs `cell_direct_runtime_extreme`, labeled `runtime_bnorm_extreme`, `D60 x E21`)
  - large-surface B-module diagnostics (`B_N, y_raw, m2, width_ratio`, max-drift + anchor points, `D60 x E21`)
  and exports `output/kinetic_action_chain/full_direct_map_release_summary.csv` (mirrored to `paper/full_direct_map_release_summary.csv`).
  Baseline scan scripts now default to `chain_mode=full_direct` (`scan_hll_signal_strengths.py`, `scan_hll_uv_to_eft_matching.py`), while `auto` is retained for comparator audits.
  A runtime-direct extension is also available via `chain_mode=full_direct_runtime`, which builds/rebuilds active D-grid localized-direct profiles (`chi_LR(D)`, `A_l(D)`) inside the scan entrypoint before strict full-direct evaluation.
  Both scan scripts expose a release-production runtime-direct `g_N+\chi+A` mode via `chain_mode=cell_direct_runtime`: `g_N(D)` is taken from active-grid direct `fp_2d_full` phase-space tables generated for the scan D-grid, while `chi_LR(D)` and `A_l(D)` are evaluated by direct solvers inside scan evaluation; `B_N` remains on release EFT-operator profile closure (`b_mode=eft_operator_norm`). After promoting active-grid direct `g_N` into the strict branch, this mode now matches `full_direct` exactly on both release gates: D21×E41 (`Δf(χ²<4)=0`, acceptance mismatch `0`, `max|Δμ_μμ|=0`; source: `paper/chain_mode_cell_direct_audit_Dgrid21_Egrid41.csv`) and D60×E21 (`Δf(χ²<4)=0`, acceptance mismatch `0`, `max|Δμ_μμ|=0`; source: `paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21.csv`).
  The tuned runtime-direct visibility branch is `chain_mode=cell_direct_runtime_release_tuned` (`b_mode=eft_operator_norm_runtime_direct` with tuned production defaults selected by `code/tune_runtime_direct_b_drift.py`, see `paper/runtime_direct_b_tuning_best.json`). With the release blend fixed at `runtime_direct_b_release_profile_blend=0.99`, it now passes both production gates: D21×E41 (`Δf(χ²<4)=0`, mismatch `0`, `max|Δμ_μμ|=0.0837`; source: `paper/chain_mode_cell_direct_audit_Dgrid21_Egrid41_cell_direct_runtime_release_tuned.csv`) and D60×E21 (`Δf(χ²<4)=0`, mismatch `0`, `max|Δμ_μμ|=0.688`; source: `paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_release_tuned.csv`).
  **Gate decision (current release):** `cell_direct_runtime` remains the release-production parity branch for the direct spectral-selection chain, and `cell_direct_runtime_release_tuned` is now promoted as a **profile-anchored runtime-direct visibility parity branch**. Because this promoted branch uses a `0.99` profile anchor in the observable sector, it should not be described as a strict all-direct closure.
  The stress-only no-profile-object branch `chain_mode=cell_direct_runtime_extreme` is retained for strict per-call recomputation with `--runtime-direct-no-cache` (very expensive; diagnostic only) and remains clearly outside release parity on D60×E21 (`Δf(χ²<4)=-0.0444`, mismatch `0.0444`, `max|Δμ_μμ|=55.03`).
  To make residual B-module drift attributable, release publishing exports `output/kinetic_action_chain/full_direct_b_module_diagnostics_D60E21.csv` (mirrored to `paper/full_direct_b_module_diagnostics_D60E21.csv`) with per-point `B_N`, `y_raw`, `m2`, and width-ratio decomposition.
  Runtime-direct B tuning artifacts are exported to `output/robustness/runtime_direct_b_tuning_small_D21E41.csv`, `output/robustness/runtime_direct_b_tuning_large_D60E21.csv`, `output/robustness/runtime_direct_b_tuning_best.json` (mirrored under `paper/`).
  A non-Hermitian rank-2 diagnostic comparator is provided by `code/scan_rank2_nonhermitian_diagnostic.py`, with map-level drift summaries such as `paper/nonhermitian_rank2_diagnostic_D21E41_kdiag0p10.csv`.
- **Core-parameter robustness table (B2):** `code/scan_core_param_robustness.py` exports local one-at-a-time windows for `c_eff`, `nu`, `p_B` and their map fractions:
  - `output/robustness/core_param_robustness_table.csv`
  - `output/robustness/core_param_robustness_cases.csv`
- **Cross-module summary:** `code/scan_first_principles_migration_summary.py` aggregates baseline `fp_2d_full+localized` vs legacy `cardy+localized` vs `fp_2d_full+open_system` into:
  - `output/first_principles_migration/first_principles_migration_summary.csv`
  - `output/first_principles_migration/first_principles_migration_summary.png`
- **Full-chain consistency comparator (kinetic unification audit):** `code/scan_action_chain_consistency.py` keeps a legacy surrogate-vs-action comparator run for historical context; the current strict chain-mode parity evidence is provided by `code/scan_chain_mode_cell_direct_audit.py`. Comparator exports:
  - `output/kinetic_action_chain/action_chain_consistency_summary.csv`
  - `output/kinetic_action_chain/action_chain_consistency.png`
- **UV-to-EFT matching audit:** `code/scan_hll_uv_to_eft_matching.py` exports per-point UV-tree / finite-match / IR Wilson diagonals and running metadata on the same scan grid:
  - `output/hll_uv_matching/hll_uv_to_eft_map.csv`
  - `output/hll_uv_matching/hll_uv_to_eft_summary.csv`
  - `output/hll_uv_matching/hll_uv_to_eft_maps.png`
- **Explicit operator-basis witness audit:** `code/scan_hll_uv_operator_basis_audit.py` now exports a layer-resolved basis decomposition
  \[
  C_{eH}^{\rm tree}=\sum_{N=1}^3 \left(P_N^{\rm kin}/M_N^2\right)\,(g_N g_N^T),
  \]
  together with blockwise finite-match / LL-RG reconstruction checks:
  - `output/hll_uv_matching/hll_uv_operator_basis_map.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_maps.png`
  In the current baseline (`full_direct`, `D60 x E60`), the tree/match/IR matrix reconstruction residuals are exactly zero at map level, while the operator-basis `\mu_{\mu\mu}` reconstruction agrees with the native `eft_wilson_uv_rge` observable to `max |Δμ_{\mu\mu}| = 7.31e-08`.
- **Input-tied finite-match comparator:** the UV matching layer now also supports `--uv-match-mode input_tied`, which replaces fixed finite-match shifts by effective
  \[
  \kappa_{\rm diag}^{\rm eff},\ \kappa_{\rm offdiag}^{\rm eff}
  \]
  tied to local UV-basis invariants (`shell_spread`, `coeff_cv`, `offdiag_mix`). This is implemented in `code/eft_rge.py` and exposed through both `code/scan_hll_uv_to_eft_matching.py` and `code/scan_hll_uv_operator_basis_audit.py`. It is currently a structured comparator only; the release baseline remains `uv_match_mode=constant`.
  A reproducible `full_direct` D21×E21 comparator is exported as
  - `output/hll_uv_matching/hll_uv_to_eft_summary_uv_input_tied_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary_uv_input_tied_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_input_tied_vs_constant_refresh_mapdiff_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_input_tied_vs_constant_refresh_scale_summary_D21E21.csv`
  and mirrored under `paper/`. At this stage the input-tied mode is witness-level consistent (basis reconstruction residuals remain zero and `max |Δμ_{\mu\mu}^{\rm recon}| = 5.55e-08`). Using the refreshed constant D21×E21 comparator and the canonical diagonal-only setting `(diag_scale, offdiag_scale)=(1,0)`, the map drift remains small (`mean |Δμ_{\mu\mu}| = 5.06e-4`, `p95 = 1.03e-3`, `max = 7.87e-3`) with zero acceptance mismatch. The scale scan further shows that `offdiag_scale` is numerically inactive in the current UV basis (`offdiag_mix_max ≈ 1.69e-26`), so the input-tied mode should be interpreted as a structured diagonal threshold witness rather than a new inter-mode mixing mechanism. It remains a comparator because the diagonal threshold scale is still externally chosen rather than derived from parent-action loop matching.
- **Diagonal-threshold window audit (Phase 2):** `code/scan_hll_uv_input_tied_diag_window.py` turns the refreshed `diag_scale` sweep into a bounded physical window report using two constraints: a conservative small-deformation cap `max |Δμ_{\mu\mu}| <= 0.01` plus `p90(|\Delta C_{\mu\mu}^{\rm match}|/|C_{\mu\mu}^{\rm tree}|) <= 0.002`, and an extended stable cap `max |Δμ_{\mu\mu}| <= 0.015` plus `p90(...) <= 0.0025`, both with zero acceptance mismatch and `offdiag_scale=0`. The canonical outputs are:
  - `output/hll_uv_matching/hll_uv_input_tied_diag_window_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_input_tied_diag_window_scan_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_input_tied_diag_window_D21E21.png`
  The resulting interpretation is: `diag_scale in [0.25, 1.0]` forms a conservative comparator window, `diag_scale in [0.25, 1.5]` is an extended stable window, and the canonical witness choice `diag_scale=1.0` sits at the top of the conservative range. Numerically, `offdiag_scale` remains inactive in the current basis, so the remaining missing ingredient is the parent-action normalization of the diagonal threshold susceptibility rather than any off-diagonal mixing closure.
- **RG-window robustness (UV+LL-RG controls):** `code/scan_hll_rge_sensitivity.py` scans one-at-a-time windows for `mu_low`, `gamma_diag`, `gamma_offdiag`, `kappa_diag`, and `kappa_offdiag`, exporting:
  - `output/robustness/hll_rge_sensitivity_cases.csv`
  - `output/robustness/hll_rge_sensitivity_table.csv`
- **UV control envelope (nonzero finite-match center):** `code/scan_hll_uv_envelope.py` scans one-at-a-time UV windows around nonzero `(kappa_diag,kappa_offdiag)` and exports:
  - `output/robustness/hll_uv_envelope_cases.csv`
  - `output/robustness/hll_uv_envelope_map.csv`
  - `output/robustness/hll_uv_envelope_summary.csv`
  - `output/robustness/hll_uv_envelope.png`
  - mirrored copies under `paper/`.
- **High-dimensional source bridge (upgrade hook):** `code/derive_projected_source_from_parent_profile.py` projects a parent extra-dimensional profile `w(z)` to an effective 3D source `sigma_eff(r)` and fits the Plummer regulator family used by PSLT. Example artifacts:
  - `output/source_projection/source_projection_profile_gaussian_w0p2.csv`
  - `output/source_projection/source_projection_fit_gaussian_w0p2.json`
  - mirrored CSV under `paper/`.

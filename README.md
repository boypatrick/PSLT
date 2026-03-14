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
  The tuned runtime-direct visibility branch is `chain_mode=cell_direct_runtime_release_tuned` (`b_mode=eft_operator_norm_runtime_direct` with tuned production defaults selected by `code/tune_runtime_direct_b_drift.py`, see `paper/runtime_direct_b_tuning_best.json`). The current release no longer uses a fixed `0.99` anchor; it uses a D-only risk-weighted profile anchor `alpha(D)` with canonical release profile `output/kinetic_action_chain/runtime_direct_visibility_alphaD_profile_release.csv` (selected by minimum mean anchor among gate-passing candidates, with `alpha_min=0.96`, `alpha_max=0.99`, `power=1.0`, `mean(alpha)=0.9637`, `p90(alpha)=0.9734`). With this release profile, the branch passes both production gates: D21×E41 (`Δf(χ²<4)=0`, mismatch `0`, `max|Δμ_μμ|=0.0925`; source: `paper/chain_mode_cell_direct_audit_Dgrid21_Egrid41_cell_direct_runtime_release_tuned.csv`) and D60×E21 (`Δf(χ²<4)=0`, mismatch `0`, `max|Δμ_μμ|=0.8373`; source: `paper/chain_mode_cell_direct_audit_Dgrid60_Egrid21_cell_direct_runtime_release_tuned.csv`).
  **Gate decision (current release):** `cell_direct_runtime` remains the release-production parity branch for the direct spectral-selection chain, and `cell_direct_runtime_release_tuned` is promoted as a **risk-weighted profile-anchored runtime-direct visibility parity branch**. Because this promoted branch still uses localized profile anchoring in the observable sector, it should not be described as a strict all-direct closure.
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
- **Action-normalized UV comparator:** the finite-match layer now also supports `--uv-match-mode action_normalized`, which multiplies the diagonal/off-diagonal input-tied witness by parent-action-side normalization factors built from localized shell-gap and UV-coefficient invariants (`gap_cv`, `gap_asym`, `g_col_norm_cv`, `c_tree_diag_cv`, `pkin_entropy`). Canonical `full_direct` D21×E21 outputs are:
  - `output/hll_uv_matching/hll_uv_to_eft_summary_uv_action_normalized_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary_uv_action_normalized_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_normalized_mode_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_normalized_pairwise_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_normalized_comparator_D21E21.png`
  The basis witness remains exact (`max |Δμ_{\mu\mu}^{\rm recon}| = 5.54e-08`, reconstruction residuals zero). Relative to the refreshed constant-threshold baseline, the action-normalized map still shows zero acceptance mismatch with small deformation (`mean |Δμ_{\mu\mu}| = 1.00e-3`, `p95 = 1.86e-3`, `max = 1.59e-2`), while its internal UV-tree→IR drift is smaller than the canonical input-tied comparator (`mean |Δμ_{\mu\mu}| = 3.65e-4`, `max = 5.93e-3`). We therefore interpret `action_normalized` as a stronger parent-action-side comparator, but not yet a baseline replacement: the normalization is now tied to action-derived invariants, yet still not fixed by a full EYMH loop calculation.
- **Action-absolute UV comparator:** the finite-match layer now also supports `--uv-match-mode action_absolute`, which removes the external `diag_scale` choice and instead derives the diagonal threshold normalization from a parent-action-side absolute witness built out of the same local invariants plus coefficient-alignment factors (`coeff_l1`, `coeff_l2`, `coeff_align`, `action_abs_diag`). Canonical `full_direct` D21×E21 outputs are:
  - `output/hll_uv_matching/hll_uv_to_eft_summary_uv_action_absolute_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary_uv_action_absolute_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_absolute_mode_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_absolute_pairwise_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_absolute_comparator_D21E21.png`
  The basis witness remains exact (`max |Δμ_{\mu\mu}^{\rm recon}| = 5.55e-08`, reconstruction residuals zero) and the map remains stable with zero acceptance mismatch. Relative to the refreshed constant-threshold baseline, `action_absolute` stays in the same small-deformation regime (`mean |Δμ_{\mu\mu}| = 7.17e-4`, `p95 = 1.17e-3`, `max = 1.14e-2`) while sitting between `input_tied` and `action_normalized`; relative to `action_normalized` the drift is smaller still (`mean |Δμ_{\mu\mu}| = 2.87e-4`, `max = 4.43e-3`). We therefore interpret `action_absolute` as the current strongest parent-action-side absolute-normalization witness, but still not a baseline replacement: what remains missing is a bona fide loop-level derivation of this absolute prefactor from the parent EYMH action.
- **Contrast-based heat-kernel UV comparator:** we then tested whether a curved-background one-loop witness can push the absolute prefactor closer to a parent-action loop origin. A purely local heat-kernel route (`scan_hll_uv_heat_kernel_witness.py`) was not enough: even after flat-space subtraction, the local `a_2`-type diagonal witness remains nearly saturated in the well region and does not align with the current `action_absolute` normalization. The current useful signal instead comes from a non-local contrast family built from well/barrier curvature separation. The canonical comparator is `--uv-match-mode action_loop_contrast`, with D21×E21 outputs:
  - `output/hll_uv_matching/hll_uv_to_eft_summary_uv_action_loop_contrast_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary_uv_action_loop_contrast_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_contrast_mode_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_contrast_pairwise_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_contrast_comparator_D21E21.png`
  This mode is again numerically exact at the basis level (`max |Δμ_{\mu\mu}^{\rm recon}| = 5.55e-08`, zero reconstruction residuals) and remains a small-deformation comparator with zero acceptance mismatch. Relative to the refreshed constant baseline the map drift is `mean |Δμ_{\mu\mu}| = 4.88e-4`, `p95 = 8.76e-4`, `max = 7.67e-3`; relative to `action_absolute` it is smaller still (`mean = 2.28e-4`, `max = 3.75e-3`). We interpret `action_loop_contrast` as the current best non-local loop-normalization comparator. It strengthens the case that the remaining UV gap is in the absolute EYMH loop normalization itself, but it is still a comparator rather than a baseline replacement.
- **Absolute loop-prefactor UV comparator:** we then add one more comparator layer by modulating the parent-action absolute witness with the same contrast-based loop family. The resulting mode is `--uv-match-mode action_loop_absolute`, with canonical D21×E21 outputs:
  - `output/hll_uv_matching/hll_uv_to_eft_summary_uv_action_loop_absolute_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary_uv_action_loop_absolute_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_absolute_mode_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_absolute_pairwise_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_absolute_comparator_D21E21.png`
  This mode remains exact at the basis level (`max |Δμ_{\mu\mu}^{\rm recon}| = 5.56e-08`, zero reconstruction residuals) and keeps zero acceptance mismatch. Relative to the refreshed constant baseline it remains a small deformation (`mean |Δ\mu_{\mu\mu}| = 3.37e-4`, `p95 = 6.92e-4`, `max = 5.33e-3`), while relative to `action_loop_contrast` it is smaller still (`mean = 1.53e-4`, `max = 2.35e-3`). The exported absolute loop-prefactor witness is now nonzero and stable (`hk_loop_prefactor_diag` median `= 0.351`, `p90 = 0.404`), so `action_loop_absolute` becomes the strongest comparator-only bridge between the current parent-action witnesses and a future bona fide EYMH loop normalization. It is still not a baseline replacement: the missing step is the actual loop-derived absolute normalization, not the existence of a stable absolute-prefactor comparator.
- **EYMH-side absolute loop-prefactor comparator:** we then fold one more layer of EYMH-side structure into the loop prefactor, using a local mass-access factor, curvature screening, shell spread, and coefficient-alignment screens to build `--uv-match-mode action_loop_eymh_absolute`. Canonical D21×E21 outputs are:
  - `output/hll_uv_matching/hll_uv_to_eft_summary_uv_action_loop_eymh_absolute_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary_uv_action_loop_eymh_absolute_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_absolute_mode_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_absolute_pairwise_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_absolute_comparator_D21E21.png`
  This mode also remains exact at the basis level (`max |Δμ_{\mu\mu}^{\rm recon}| = 5.56e-08`, zero reconstruction residuals) and keeps zero acceptance mismatch. Relative to the refreshed constant baseline it is an even smaller deformation (`mean |Δ\mu_{\mu\mu}| = 4.17e-5`, `p95 = 7.63e-5`, `max = 6.46e-4`), while relative to `action_loop_absolute` it still stays in the small-deformation regime (`mean = 2.95e-4`, `max = 4.68e-3`). The EYMH-side absolute prefactor is nonzero but more strongly screened (`eymh_loop_prefactor_diag` median `= 0.0378`, `p90 = 0.0438`), so this mode should be read as the current best comparator-level proxy for an eventual EYMH absolute loop normalization, not yet as a baseline replacement.
- **Source-informed EYMH comparator:** we then promote the two dominant source factors already isolated by the EYMH audits into a direct comparator mode, `--uv-match-mode action_loop_eymh_source_informed`, retaining only the source-informed diagonal prefactor
  - `eymh_source_prefactor_diag`
  - `coeff_participation_access`
  - `tree_diag_compressibility`
  Canonical refreshed D21×E21 outputs are:
  - `output/hll_uv_matching/hll_uv_to_eft_summary_uv_action_loop_eymh_source_informed_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary_uv_action_loop_eymh_source_informed_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_source_informed_mode_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_source_informed_pairwise_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_source_informed_comparator_D21E21_fix.png`
  This refreshed source-informed mode is again exact at the basis level (`max |Δμ_{\mu\mu}^{\rm recon}| = 5.56e-08`) and remains a small deformation relative both to the refreshed constant baseline (`mean |Δ\mu_{\mu\mu}| = 5.2e-5`, `p95 = 1.09e-4`, `max = 7.81e-4`, zero acceptance mismatch) and to `action_loop_eymh_absolute` (`mean = 1.0e-5`, `p95 = 3.8e-5`, `max = 1.34e-4`, zero acceptance mismatch). The resulting source-informed prefactor stays nonzero across the full D21×E21 map (`eymh_source_prefactor_diag` median `= 0.0435`, `p90 = 0.0580`) with a participation/coherence witness and tree-diagonal compressibility witness that also remain nonzero over the whole grid (`coeff_participation_access` median `= 0.816`, `tree_diag_compressibility` median `= 0.781`). This makes `action_loop_eymh_source_informed` the current best source-informed comparator below a full parent-action normalization.
- **Parented EYMH comparator:** we then rewrite the same source-informed block in explicit parent-action language via `--uv-match-mode action_loop_eymh_parented`, exposing
  - `eymh_parented_prefactor_diag`
  - `coeff_participation_access_parented`
  - `tree_diag_compressibility_parented`
  Canonical D21×E21 fix outputs are:
  - `output/hll_uv_matching/hll_uv_to_eft_summary_uv_action_loop_eymh_parented_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_operator_basis_summary_uv_action_loop_eymh_parented_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parented_mode_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parented_pairwise_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parented_comparator_D21E21_fix.png`
  This parented rewrite remains exact at the basis level (`max |Δμ_{\mu\mu}^{\rm recon}| = 5.56e-08`) and, on the canonical D21×E21 fix grid, is map-identical to `action_loop_eymh_source_informed` (`mean/p95/max pairwise |Δ\mu_{\mu\mu}| = 0`, zero acceptance mismatch). Relative to the refreshed constant baseline it therefore inherits the same small-deformation scale (`mean |Δ\mu_{\mu\mu}| = 5.2e-5`, `p95 = 1.09e-4`, `max = 7.81e-4`). We interpret `action_loop_eymh_parented` as the current best parent-action participation/compressibility rewrite of the EYMH source-informed comparator: it does not introduce a new numerical baseline, but it sharpens the physical reading of the same comparator in loop-trace participation and shell-background compressibility language.
- **EYMH-prefactor decomposition audit:** to sharpen the remaining UV gap, we now explicitly decompose the canonical `action_loop_eymh_absolute` prefactor into
  - `hk_loop_local_prefactor_diag`
  - `shell_access`
  - `align_access`
  - `diag_dispersion_screen`
  and export:
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_prefactor_decomposition_map_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_prefactor_decomposition_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_prefactor_decomposition_D21E21.png`
  The reconstruction residual is numerically zero (`max residual = 9.71e-17`, `max log-residual = 1.48e-15`). The dominant contributors are now explicit: `align_access` tracks the EYMH prefactor most strongly (`corr = 0.9655`), while `diag_dispersion_screen` supplies the leading suppression (`corr = -0.8774`). In other words, the remaining missing physics is no longer “whether a stable absolute prefactor exists”, but how the parent EYMH loop fixes coefficient alignment and dispersion screening in an absolute way.
- **EYMH-prefactor source audit:** we then resolve those two dominant pieces one level deeper into source factors tied to the canonical map itself:
  - `align_access = coeff_participation_access = N_eff^{-1/4}` with `N_eff = (coeff_l1 / coeff_l2)^2`
  - `diag_dispersion_screen = shell_gap_screen × tree_diag_screen`
  and export
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_source_audit_map_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_source_audit_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_source_audit_D21E21.png`
  This source-level reconstruction remains exact (`max residual = 9.71e-17`), and it sharpens the physical interpretation: `coeff_participation_access` now matches `align_access` identically (`corr = 1.0`) and inherits the same strong correlation with the EYMH prefactor (`0.9655`), while the suppression factor splits into a weak shell-gap screen (`corr = 0.0907`) and a much stronger tree-diagonal screen (`corr = -0.8226`). The remaining parent-action gap is therefore even narrower: the unresolved normalization is no longer a generic “alignment + dispersion” block, but specifically the action-side origin of coefficient participation coherence and tree-diagonal dispersion screening.
- **Tree-diagonal susceptibility/compressibility audit:** we now rewrite the dominant screening term as
  - `tree_diag_susceptibility = c_tree_diag_cv / (1 + gap_cv)`
  - `tree_diag_compressibility = (1 + tree_diag_susceptibility)^(-1/2) = tree_diag_screen`
  and export
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_tree_diag_audit_map_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_tree_diag_audit_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_tree_diag_audit_D21E21.png`
  The rewrite is exact (`max residual = 0`). Numerically, the susceptibility itself tracks the EYMH prefactor positively (`corr = 0.7409`), while the compressibility witness carries the same leading suppression as `tree_diag_screen` (`corr = -0.8226`). A complementary pressure-style ratio built from the tree-diagonal contribution to the total shell+tree denominator is also strongly correlated (`corr = 0.8449`). This sharpens the remaining UV gap one step further: the unresolved parent-action physics is no longer a generic dispersion block, but specifically the origin of tree-level diagonal susceptibility/compressibility in the loop-prefactor screening.
- **EYMH parent-source model audit:** we now collect the two dominant source-informed blocks into a parent-source rewrite that stays exact cell by cell:
  - `coeff_participation_access` is reconstructed from a two-mode loop-trace participation model with
    `loop_trace_neff = 1 / (p1^2 + p2^2)` and `loop_trace_entropy_norm`
  - `tree_diag_compressibility` is reconstructed from the shell-background-normalized diagonal susceptibility and its pressure fraction
  and export
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_source_model_map_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_source_model_summary_D21E21.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_source_model_D21E21.png`
  This parent-source rewrite remains exact to machine precision (`max coeff-participation residual = 2.22e-16`, `max tree-diagonal residual = 1.11e-16`). The participation side is now readable as a loop-trace concentration witness (`corr(loop_trace_neff, coeff_participation_access) = 0.9608`, `corr(loop_trace_entropy_norm, coeff_participation_access) = -0.9560`), while the tree-diagonal side is readable as a susceptibility/compressibility response with a correlated pressure fraction (`corr(tree_diag_pressure_fraction, eymh_source_prefactor_diag) = -0.7213`, `corr(tree_diag_compressibility, eymh_source_prefactor_diag) = 0.7066`). This narrows the remaining EYMH normalization gap again: the unresolved physics is now why the parent action fixes loop-trace participation coherence and shell-background-normalized tree-diagonal pressure/compressibility in precisely this combination.
- **EYMH exact participation audit:** the projected two-mode participation block can now be closed one step further. Using the canonical parented map, we reconstruct `coeff_participation_access_parented` exactly from the projected loop-trace participation number,
  \[
  N_{\rm eff}=\frac{1}{p_1^2+p_2^2},\qquad
  d=\sqrt{\frac{2}{N_{\rm eff}}-1},\qquad
  A_{\rm part}^{\rm exact}=\sqrt{\frac{1-d}{1+d}},
  \]
  and export
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_exact_audit_map_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_exact_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_exact_audit_D21E21_fix.png`
  The residual is now at machine precision (`max exact participation residual = 9.99e-16`). This improves the previous Hessian-response reading: the participation side is no longer merely strongly correlated with a projected response, but exactly determined by the two-mode participation imbalance implied by the loop trace. The remaining parent-action gap is therefore pushed onto an even narrower question: why the parent EYMH fluctuation operator dynamically selects this two-mode participation structure in the first place.
- **EYMH coefficient-norm participation audit:** we can push that closure one step closer to the parent-action coefficient block itself. Writing the projected coefficient vector as `c_N = P_N^kin / M_N^2`, the canonical parented participation factor is reconstructed exactly from its norm ratio,
  \[
  Q_2=\frac{\|c\|_2^2}{\|c\|_1^2},\qquad
  N_{\rm eff}^{\rm norm}=\frac{\|c\|_1^2}{\|c\|_2^2},\qquad
  A_{\rm part}^{\rm norm}=Q_2^{1/4}=\sqrt{\frac{\|c\|_2}{\|c\|_1}},
  \]
  and export
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_norm_audit_map_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_norm_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_norm_audit_D21E21_fix.png`
  The reconstruction is again exact to machine precision (`max norm-participation residual = 2.22e-16`). This is a cleaner parent-action reading than the loop-trace rewrite alone: the participation side can now be read directly as a projected coefficient-vector coherence, with `corr(A_part^norm, eymh_parented_prefactor_diag) = 0.7129`. The remaining gap is narrower still: explain why the EYMH fluctuation operator fixes this exact projected norm-ratio coherence together with the shell-background-normalized tree-diagonal compressibility response.
- **EYMH coefficient-norm tilt audit:** the same projected coefficient block also admits an exact free-energy-tilt reading. Defining
  \[
  \Delta F_{\rm norm}=\log\frac{\|c\|_1}{\|c\|_2},\qquad
  A_{\rm part}^{\rm tilt}=e^{-\Delta F_{\rm norm}/2}=\sqrt{\frac{\|c\|_2}{\|c\|_1}},
  \]
  we reconstruct both the participation block and the full canonical parented prefactor,
  \[
  \Pi_{\rm parent}
  = \Pi_{\rm hk,local}\;
    A_{\rm shell}\;
    A_{\rm part}^{\rm tilt}\;
    S_{\rm tree},
  \]
  by exporting
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_tilt_audit_map_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_tilt_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_participation_tilt_audit_D21E21_fix.png`
  The residuals remain at machine precision (`max tilt-participation residual = 2.22e-16`, `max parented-prefactor residual = 9.02e-17`). This is the strongest parent-action reading so far: the canonical participation/compressibility block can now be read as a projected coefficient-norm free-energy tilt multiplied by shell access and tree-diagonal compressibility. The remaining gap is now very narrow: explain why the EYMH fluctuation operator dynamically fixes this exact norm-tilt coherence and the accompanying shell-background-normalized tree-diagonal pressure/compressibility response.
- **EYMH projected response-action audit:** the same parented block now admits a single projected response-action rewrite. Defining
  \[
  S_{\rm resp}=\Delta F_{\rm norm}+\log(1+\chi_{\rm tree}),
  \qquad
  A_{\rm resp}=e^{-S_{\rm resp}/2},
  \]
  we reconstruct the canonical parented response factor and full prefactor exactly as
  \[
  \Pi_{\rm parent}
  = \Pi_{\rm hk,local}\;
    A_{\rm shell}\;
    A_{\rm resp},
  \]
  and export
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_response_action_audit_map_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_response_action_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_response_action_audit_D21E21_fix.png`
  The closure is again exact to machine precision (`max response-weight residual = 2.22e-16`, `max parented-prefactor residual = 9.71e-17`). This is the cleanest mother-action reading so far: coefficient-norm tilt and tree-diagonal compressibility do not merely co-occur, they enter additively in a single projected response action whose exponential fixes the canonical parented prefactor.
- **EYMH projected log-det / Schur audit:** this same response action now admits a more explicit kernel reading. Writing
  \[
  K_{\rm part}=\frac{\|c\|_1}{\|c\|_2},
  \qquad
  G_{\rm Schur}=\frac{1+\mathrm{gap}_{\rm cv}+c_{{\rm tree},{\rm diag},{\rm cv}}}{1+\mathrm{gap}_{\rm cv}}=1+\chi_{\rm tree},
  \]
  gives
  \[
  S_{\rm resp}=\log\det K_{\rm part}+\log G_{\rm Schur},
  \qquad
  A_{\rm resp}=e^{-S_{\rm resp}/2}.
  \]
  The audit exports
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_logdet_schur_audit_map_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_logdet_schur_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_logdet_schur_audit_D21E21_fix.png`
  and again closes both the response weight and the canonical parented prefactor to machine precision (`max logdet/Schur weight residual = 2.22e-16`, `max parented-prefactor residual = 9.71e-17`). This is the strongest parent-action reading so far: the surviving EYMH normalization block can now be read as an exact projected log-det participation kernel plus a shell-normalized Schur-kernel response.
- **EYMH projected kernel-selection audit:** the final open question is then whether this exact `logdet + Schur` structure is dynamically selected, or merely one algebraically convenient rewrite. The new audit under
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_kernel_selection_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_kernel_selection_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_kernel_selection_audit_D21E21_fix.png`
  probes the minimal deformed family
  \[
  K_{\rm sel}=
  \begin{pmatrix}
  e^{\alpha S_{\rm part}} &
  \lambda\sqrt{(e^{\alpha S_{\rm part}}-1)(e^{\beta S_{\rm schur}}-1)} \\
  \lambda\sqrt{(e^{\alpha S_{\rm part}}-1)(e^{\beta S_{\rm schur}}-1)} &
  e^{\beta S_{\rm schur}}
  \end{pmatrix}.
  \]
  On the canonical D21xE21 fix grid the unique best-fit point is exactly the parented kernel, `(\alpha,\beta,\lambda)=(1,1,0)`. The first nontrivial runner-up is `(\alpha,\beta,\lambda)=(1,1,-0.1)`, but it already opens a prefactor RMSE gap of `2.67e-05`. The finite-difference stationarity test at the canonical point gives vanishing gradients (`∂J/∂\alpha=-3.00e-09`, `∂J/∂\beta=-6.34e-09`, `∂J/∂\lambda=0`) and non-negative Hessian eigenvalues `(2.99e-09, 1.33e-03, 7.59e-02)`. This is the strongest dynamic-selection statement so far: the projected fluctuation operator selects unit log-det / Schur weights and suppresses projected participation-tree cross-coupling.
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

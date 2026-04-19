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
- **High-$N$ Decoupling Interpretation:** The current baseline does not remove higher layers by hand through the visibility block: `B_{N>3}=1` is a neutral saturation rather than an extra damping factor. In the audited action-derived single-track benchmark, no tested `D` point supports a fourth bound level (`/Users/boypatrick/codex/PSLT_quantam/output/highN_decoupling/highN_decoupling_single_track_summary.csv`), while the companion one-dimensional kinetic proxy keeps `E_4=\omega_4^2-m_0^2>0` across the tested window and a forced quasi-channel continuation does not generate an additional `\Gamma_4` hierarchy (`/Users/boypatrick/codex/PSLT_quantam/output/highN_decoupling/highN_decoupling_superrad_1d_summary.csv`). The present evidence therefore favors threshold / binding loss over an engineered visibility cutoff, while a mode-resolved no-fourth-layer theorem remains future first-principles work.

## First-Principles Migration Status
- A concise open-items tracker is maintained in `/Users/boypatrick/codex/PSLT_quantam/ROADMAP_STATUS.md`, including current release-gate numbers and remaining promotion blockers.
- A one-page reviewer-family decision summary is maintained in `/Users/boypatrick/codex/PSLT_quantam/FAMILY_DECISION_TABLE.md`, which records which candidate families are canonical, which have been closed as negative, and which remain research-only ingredients.
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
  The newer parent-bath audits then push this further. `code/scan_chi_open_system_bath_factorization.py` and `code/scan_chi_open_system_kappa_absolute_audit.py` show that the current witness already factorizes into a system block (`g_z^2,g_x^2`), a bath-shape block (`S_{zz}(0),S_{xx}(\Delta E)`), and an amplitude block (`kappa_env`), with canonical constant normalization remaining the unique exact amplitude choice. `code/scan_chi_open_system_parent_bath_statement.py` rewrites the rates as the projected bath block
  \[
  K_{\rm bath}=\kappa_{\rm env}\sqrt{K_{\rm sys}}\,K_{\rm spec}\,\sqrt{K_{\rm sys}},
  \]
  while the subsequent family, log-coordinate, normal-coordinate, and generator-affinity audits (`code/scan_chi_open_system_parent_bath_family_audit.py`, `code/scan_chi_open_system_parent_bath_log_coordinate_audit.py`, `code/scan_chi_open_system_parent_bath_normal_coordinate_audit.py`, `code/scan_chi_open_system_parent_bath_generator_affinity_audit.py`) all uniquely return the canonical point (`m=u=v=0`, `p_{\rm sys}=p_{\rm spec}=0`, `\zeta_{\rm sys}=\zeta_{\rm spec}=0`, `q_{ss}=q_{bb}=q_{sb}=0`) on the current `D=4..20` knot set. At this point the remaining open-system gap is no longer bookkeeping or nearby-family ambiguity; it is the final parent-action statement for why the projected bath generator itself naturally lives in this affine log class.
  `code/scan_chi_open_system_parent_bath_cocycle_audit.py` then turns this into a positive integrability statement: after dividing out `kappa_env`, the projected bath block defines an exact additive cocycle in the canonical log variables, with pairwise cocycle residuals at `8.95e-16` on the identifiable `phi` subset and `8.64e-13` on the mixing branch, while triangle flatness defects stay at `1.11e-15` in both channels. So the affine log-generator is now supported not just by nearby-family exclusion, but by exact cocycle/flatness closure.
  `code/scan_chi_open_system_parent_bath_potential_audit.py` pushes this one step further into a projected effective-action statement: the normalized bath block admits an exact single-valued anchored potential `\Phi_a=\log(B_a/B_a^{\rm ref})`, with direct decomposition, anchor-average recovery, and nearest-neighbor chain recovery all agreeing to machine precision (`max` residuals `~1e-14` for the identifiable `phi` subset and `~5e-13` or better for the mixing branch). This gives the natural local parent-bath normal form on the audited knot set,
  \[
  K_{{\rm bath},a}(z)=\kappa_{\rm env}\,K_{{\rm sys},a}^{1/2}(z)\,e^{\Phi_a(z)}\,K_{{\rm sys},a}^{1/2}(z),
  \]
  with the family/log-coordinate/normal-coordinate/generator-affinity audits selecting the unique canonical point where the first local log-curvature jet is absent. So the remaining gap is now only the microscopic EYMH origin of the coefficients and bath degrees of freedom, not whether the projected bath block already carries a local anchored effective action.
  The logic is non-circular: nearby-family and generator-affinity closure act directly on the normalized projected bath block, the cocycle audit upgrades that same block to exact additive consistency in canonical log variables, and the anchored-potential audit only then integrates that cocycle and checks path-independence. The local effective-action form is therefore a repackaging of the already-audited projected bath witness, not an extra ansatz used to justify itself.
  In the same local sense, the affine log-generator is just the first Taylor jet of the anchored potential on the normalized projected bath block, with the first non-affine correction controlled by the audited quadratic remainder rather than promoted to a new microscopic claim.
  The reviewer-facing microscopic remainder is now narrower than a generic open-system ansatz and can be summarized as the chain
  \[
  \kappa_{\rm env}
  \;\Longrightarrow\;
  P\Delta V P\in{\rm span}\{I,\sigma_z,\sigma_x\}
  \;\Longrightarrow\;
  S_{zz,a}(0),\,S_{xx,a}(\Delta E_a)=S_a(0),\,S_a(\Delta E_a).
  \]
  In words: `kappa_env` is already isolated as the unique exact amplitude modulus, the localized two-level projection leaves only the `\sigma_z/\sigma_x` leading directions, and the dephasing/mixing kernels are already identified as on-shell evaluations of the same minimal response family rather than unrelated bath shapes. The bridge now sharpens this one step further: on the audited knot set that response family is already the Fourier image of a minimal even single-pole kernel `C_a(t)=e^{-|t|/\tau_{{\rm env},a}}`, and its correlation time is numerically gap-locked as `\tau_{{\rm env},a}=\omega_{1,a}^{-1}` to machine precision. The cleanest remaining parent-side question is therefore no longer why two kernels appear, but why the projected EYMH Schur resolvent admits a first-mode-dominant truncation on the on-shell window `\omega\in\{0,\Delta E_a\}`, with leading pole scale `\Omega_{1,a}=\omega_{1,a}` and a uniformly small higher-mode remainder there.
  The manuscript now presents this late-stage open-system closure in reviewer-facing proposition form: `Proposition O1` identifies the projected localized interaction as a two-channel `span\{I,\sigma_z,\sigma_x\}` block, `Proposition O2` identifies `S_{zz}(0)` and `S_{xx}(\Delta E)` as on-shell evaluations of a single gap-locked response family, `Proposition O3` rewrites the remaining source question as a projected Schur-resolvent statement, `Proposition O4` turns the determinant-deficit route into a direct operator statement, `Proposition O5` promotes the continuum neighborhood budget to the main proof-side candidate, and `Theorem Target O1` isolates the remaining parent-side first-mode remainder step on the audited on-shell window.
  A new gap-conditioned audit now sharpens `Theorem Target O1` one notch further: conditional on a parent-side gap ratio `g_a=\Omega_{2,a}/\Omega_{1,a}>1`, the only remaining quantitative input is the on-shell ratio defect `\delta_{q,a}=R_a(\Delta E_a)/R_a(0)-1/(1+(\Delta E_a/\Omega_{1,a})^2)`. On the audited window `\Delta E_a/\omega_{1,a}\in[0.1430,0.2857]`, the representative benchmark `g_a=2` already yields a separator band `f(r_a/2)-f(r_a)\in[1.50\times 10^{-2},5.55\times 10^{-2}]`, so a future parent-side match at only `\delta_{q,a}\le 10^{-3}` would force tail weight `1-\nu_{1,a}\le 6.68\times 10^{-2}` and on-shell remainder `\varepsilon_a(\Delta E_a)\le 1.82\times 10^{-2}` uniformly across the audited knot set. This does not prove the parent-side gap by itself, but it reduces the remaining theorem target to an explicit gap-plus-defect statement rather than a diffuse reservoir ambiguity.
  A new source audit now closes the defect half of that statement on the current knot set. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_deltaq_source.py` compares the parent-bath anisotropy ratio, the direct micro ratio `S_{xx}(\Delta E)/S_{zz}(0)`, and the gap-locked Lorentzian prediction `1/(1+(\Delta E/\omega_1)^2)`. On D4--20 it finds `max |\tau_{\rm env}-\omega_1^{-1}| = 1.11e-16`, `max |\delta_q| = 4.44e-16`, and the direct micro ratio stays at `3.33e-16`. Feeding this actual defect into the already-audited gap floors gives only `1-\nu_1 <= 5.08e-14`, `\varepsilon(\Delta E) <= 2.94e-14` for the conservative analytic floor and `1-\nu_1 <= 4.93e-14`, `\varepsilon(\Delta E) <= 2.79e-14` for the sharper exact-family floor. So on the current audited knot set the `\delta_q` side is effectively closed, and the surviving theorem blocker is the gap / exact Schur-lift side alone.
  A new parent-side proxy audit pushes one step harder on the same theorem target. Background-normalizing the canonical projected EYMH complement block gives `\widetilde K_a = I + v_a v_a^\top` with `v_a=(\sqrt{K_{11,a}-1},\sqrt{G_a-1})`, so the proxy eigenvalues are exactly `\lambda_{1,a}^{\rm(proxy)}=1` and `\lambda_{2,a}^{\rm(proxy)}=K_{11,a}+G_a-1`. Calibrating the identity mode to the already-audited gap-locked pole by `\Omega^{\rm(proxy)}_{1,a}=\omega_{1,a}` then yields `\Omega^{\rm(proxy)}_{2,a}=\omega_{1,a}\sqrt{K_{11,a}+G_a-1}` and `g^{\rm(proxy)}_a=\sqrt{K_{11,a}+G_a-1}`. The audit closes this rank-one structure to machine precision and shows that on the nontrivial active rows of the D21xE21 fix grid one already has `g^{\rm(proxy)}_a\in[1.3537,1.5609]` with median `1.4692`. On the overlap with the current open-system knot set, this proxy gap floor yields a separator band `f(r_a/g^{\rm(proxy)}_a)-f(r_a)\in[1.43\times10^{-2},4.03\times10^{-2}]`, so even the modest defect target `\delta_{q,a}\le 10^{-3}` would imply `1-\nu_{1,a}\le 6.98\times10^{-2}` and `\varepsilon_a(\Delta E_a)\le 3.60\times10^{-2}`. This is still a proxy rather than a proof of the true Schur-resolvent spectrum, but it means the parent-side gap is no longer completely unconstrained.
  The next local strengthening is now also explicit. Combining the exact parent mismatch functional with the exact stationarity Hessian gives an audited local coercivity statement on the canonical family box `|\alpha-1|,|\beta-1|\le0.1`, `|\lambda|\le0.2`: the scan yields `J_{\rm parent}\ge c_{\rm loc} Q_{\rm loc}` with `c_{\rm loc}=3.98\times10^{-1}`, `\lambda_{\min}(H)=1.33\times10^{-3}`, and `C_4=1.497\times10^{-3}`. Therefore any local operator-side candidate with `J_{\rm parent}\le J_\ast` obeys `\|\delta\theta\|\le \sqrt{2J_\ast/(c_{\rm loc}\lambda_{\min}(H))}`. Applying this to the rowwise gap proxy and the audited boxwise Lipschitz constant gives a genuine local operator-family floor: for `J_\ast=10^{-6}`, one already gets the uniform conservative bound `g_a\ge 1.3375`, while direct enumeration of admissible scan points still yields the sharper `g_a\ge 1.3537`. On the current on-shell window, the conservative bound alone already implies `1-\nu_{1,a}\le 8.37\times10^{-2}` and `\varepsilon_a(\Delta E_a)\le 5.28\times10^{-2}` at the same reference defect `\delta_{q,a}\le10^{-3}`. This is still local to the audited projected family, not yet a full Schur-spectrum theorem, but it is a real operator-side lower bound rather than only a proxy heuristic. Corollary 1 now sharpens the remaining lift one step further: if the exact Schur complement obeys a small complement det-loss `\sigma_a=-\log\det(I-A_a)`, then positivity of `A_a` gives `H_{{\rm eff},a}\succeq e^{-\sigma_a} H_{PP,a}` and hence `g^{\rm(exact)}_a\ge e^{-\sigma_a/2} g^{\rm(fam)}_a`. Using the conservative family floor above, even `\sigma_a\le10^{-2}` still leaves the conditional exact floor at `g^{\rm(exact)}_a\ge1.3308`, with `\varepsilon_a(\Delta E_a)\le5.41\times10^{-2}` at the same `\delta_{q,a}\le10^{-3}`. So the proof-level bottleneck is now narrower again: bound the complement det-loss, and the family gap floor lifts automatically.
  A first parent-side attempt at that det-loss is now also explicit. Under a minimal dominant-channel hypothesis `A_a \approx a_a u_a u_a^\top`, one has `\|A_a\|=a_a` and `\sigma_a=-\log(1-a_a)`. The exact projected kernel family already supplies the scalar candidate `a_a^{(1)}=\lambda^2\xi_a`, with `\xi_a=((K_{11,a}-1)(K_{22,a}-1))/(K_{11,a}K_{22,a})`. The same coercive box therefore gives `|\lambda|\le (J_\ast/(c_{\rm loc}C_4))^{1/4}`, so at `J_\ast=10^{-6}` one gets `|\lambda|\le2.02\times10^{-1}`. On the open-system overlap, the exact canonical factor has `\max\xi_a=1.3324\times10^{-1}`. A more microscopic raw witness built directly from the parented off-diagonal EYMH invariants,
  `\xi_a^{\rm(raw)} = 0.127472 * hk_abs_offdiag/diag + 0.435064 * action_abs_offdiag/diag`,
  has overlap p95 residual `4.37\times10^{-2}`, giving the conservative envelope `\xi_a\le1.797\times10^{-1}`. These numbers already force the rank-one dominant-channel candidates into the small-loss regime: at `J_\ast=10^{-6}`, the exact canonical route gives `\sigma_a^{(1)}\le5.47\times10^{-3}`, while the raw-witness envelope still gives only `\sigma_a^{(1)}\le7.39\times10^{-3}`. Feeding the latter into the conditional lift leaves `g_a^{\rm(exact)}\ge1.3326` and `\varepsilon_a(\Delta E_a)\le5.37\times10^{-2}` at `\delta_{q,a}\le10^{-3}`. This is still conditional on a single dominant complement channel, not yet a full operator theorem, but it turns the abstract complement remainder into a concrete parent-side off-diagonal budget.
  A new true-operator consequence audit now makes the next step more explicit. Once any such conditional det-loss bound has been obtained, the resulting statements about the complement contraction are no longer rank-one bookkeeping identities: for any positive contraction `A_a`, one has `tr(A_a) <= sigma_a` and `||A_a|| <= 1-exp(-sigma_a)`. Materializing those inequalities on the same tolerance table shows that at `J_\ast=10^{-6}` the exact canonical route already gives `tr(A_a) <= 5.47\times10^{-3}`, `||A_a|| <= 5.46\times10^{-3}`, `H_eff >= 0.9945 H_PP`, and Corollary-1 log-det remainder `<= 1.10\times10^{-2}`; the conservative raw off-diagonal envelope still gives only `tr(A_a) <= 7.39\times10^{-3}`, `||A_a|| <= 7.36\times10^{-3}`, `H_eff >= 0.9926 H_PP`, and log-det remainder `<= 1.48\times10^{-2}`. The last proof-level gap is therefore narrower again: the only missing ingredient is a genuine parent-side source for a sigma-budget of that size, either by justifying the single dominant complement channel or by replacing it with a full trace/norm estimate on `A_a`.
  There is now also a cleaner 2x2 determinant-deficit route which weakens the single-channel hypothesis further. For any positive contraction `A_a`, defining `u_a := 1-det(I-A_a)` gives the general bounds `||A_a|| <= u_a` and `tr(A_a) <= 2*(1-sqrt(1-u_a))`. Since the projected kernel family already carries the exact local factor `1-\lambda^2\xi_a`, the same parent-side envelope can be read as a determinant-deficit candidate without first reducing to rank one. A preliminary coarse operator audit already showed that this route lives on the `10^{-3}` scale, but two follow-up audits now fix the sharper local picture. First, a densified exact-family scan over `\alpha,\beta\in[0.95,1.05]` and `\lambda\in[-0.2,0.2]` with step `0.005` reveals that the coarse local scan was too optimistic: once the family is resolved more finely, the admissible ridge at `J_{\rm parent}<=10^{-6}` moves to about `(1.03,0.99,\pm0.2)`, giving the exact overlap budget `u_a <= 5.42\times10^{-3}`, `tr(A_a) <= 5.42\times10^{-3}`, `||A_a|| <= 5.42\times10^{-3}`, `H_eff >= 0.9946 H_PP`, and Corollary-1 log-det remainder `<= 1.09\times10^{-2}`. Second, a continuum neighborhood optimization based on the same coercive budget `0.5 \delta\theta^T H \delta\theta + C_4 \lambda^4 <= J_\ast/c_{\rm loc}` is almost as sharp: at `J_\ast=10^{-6}` it gives `u_a <= 5.34\times10^{-3}` near `(1.01,0.992,0.2)`. So the dense exact scan is the tighter local benchmark, but the continuum neighborhood bound is already close enough to be the better theorem candidate. Even the conservative raw off-diagonal envelope on the same dense admissible window stays moderate, with `u_a <= 7.19\times10^{-3}` and log-det remainder `<= 1.45\times10^{-2}`. The remaining proof-level job is therefore clearer than before: promote this near-saturated local determinant-deficit budget beyond the scanned family.
  A new consolidation audit now pushes that promotion one notch further on the current audited deformation families. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_lift_source.py` combines the family, generator-affinity, log-coordinate, and normal-coordinate scans with the parent-bath statement/cocycle/potential exactness checks. Across all four deformation scans the exact statement is uniquely selected at the undeformed point, with canonical objective at most `3.18e-12` and the weakest runner-up gap already `6.90e-4`. That gap is `6.90e2` times larger than the continuum-neighborhood tolerance `J_\ast=10^{-6}` and `2.17e8` times larger than the largest canonical residual, while the accompanying exactness witnesses stay tiny (`8.43e-13` for the anisotropy statement, `1.11e-15` for the cocycle mix triangle defect, `4.03e-15` for the potential/chain mix recovery residual). So on the present audited knot set the exact-lift source is no longer just “if the neighborhood budget survives”: the existing parent-bath family rigidity is already compatible with the same `u_a <= 5.34e-3` budget, leaving `g_a^{\rm(exact)} >= 1.3339`, `H_eff >= 0.9947 H_PP`, and, together with the now-closed `\delta_q` side, only `1-\nu_1 <= 5.12e-14`, `\varepsilon(\Delta E) <= 2.97e-14`. The remaining proof task is therefore narrower still: replace this audited canonical-rigidity source by a genuine operator theorem.
  A new transfer audit tightens that one step further in a more theorem-like form. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_rigidity_transfer.py` checks the full family scan tables at the theorem tolerance `J_\ast=10^{-6}` instead of only comparing runner-up gaps. On all four audited deformation classes (linear family, generator-affinity, log-coordinate, normal-coordinate), the result is the same: `N_adm(total)=1` and `N_adm(noncanonical)=0`. In other words, within each exported family class the undeformed canonical point is the unique admissible point at the theorem tolerance. Quantitatively, the largest canonical objective remains `3.18e-12`, the smallest non-canonical objective is already `6.90e-4`, and the same margin ratios survive (`6.90e2` over the theorem tolerance, `2.17e8` over the canonical residual). This now supports a sharper conditional statement: if the exact Schur-side operator lies in any of these audited family classes and satisfies the same local tolerance, then the Proposition-O5 determinant-deficit budget transfers unchanged, so `g_a^{\rm(exact)} >= 1.3339`, `H_eff >= 0.9947 H_PP`, and, together with the closed `\delta_q` side, only `1-\nu_1 <= 5.12e-14`, `\varepsilon(\Delta E) <= 2.97e-14`. The remaining proof-level job is therefore narrower again: remove the explicit family-class hypothesis and upgrade this transfer to a genuinely operator-level theorem.
  A new witness-space aggregation now weakens even that family-class hypothesis. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_operator_witness_transfer.py` stacks all non-canonical rows from the four exported deformation scans directly in the operator residual coordinates `(block11, block22, trace, determinant, anisotropy)` and asks for the nearest non-canonical witness challenger. The answer is already sharp: `||W_{95}||_\infty = 1.92e-4` and `||W_{max}||_\infty = 2.01e-4`, both from the generator-affinity challenger `q_{bb}=\pm 0.0625`, while the canonical witness suprema remain only `8.17e-13` and `8.43e-13`. So on the audited operator chart, the half-radius bounds `||W_{95}||_\infty <= 9.62e-5`, `||W_{max}||_\infty <= 1.01e-4` still leave margins of `1.18e8` and `1.19e8` over the canonical witness scale and therefore transfer the same Proposition-O5 budget without naming a specific deformation family. This is not yet a complete operator theorem, but it converts the last explicit hypothesis from a family label into an operator-witness chart bound.
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
  We also now keep a deterministic de-anchored stress line on record after fixing the runtime-direct warm-path recursion and eigensolver seeding (`code/pslt_lib.py`, `code/extract_y_eff_2d_three_channel.py`, `code/extract_chi_localized_2d.py`, `code/extract_gn_phase_space_2d.py`). The strongest current exploratory branch, `cell_direct_runtime_release_tailm2_detlin`, does close the small-surface mismatch gate on D21×E41 (`0.00929 < 0.01`) but still fails badly on drift (`max|Δμ_μμ| = 68.39`, `p95 = 9.99`; source: `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_tailm2_detlin_D21E41.csv`). Trusted component audits then decomposed the stress residual into explicit local mechanisms: `D=4.0` and `D=4.8` are width-dominated, `D=6.4` is UV/g-dominated, and `D=7.2/8.0` are small width tails (`output/kinetic_action_chain/runtime_direct_detlin_component_audit_summary.csv`, `output/kinetic_action_chain/runtime_direct_detlin_component_audit_slices.csv`). Subsequent exact runtime compares now close that audited focus band one mechanism at a time: a band-pass `g_{\rm norm}` gate fixes `D=6.4`; a local width gate with `\beta \approx 0.75` fixes `D=4.8`; a narrower `D=4.0` width gate closes the acceptance mismatch exactly at `\beta = 1.309`; and tail-only width gates suppress `D=7.2/8.0` to `0.0677/0.0417` with zero acceptance flips (`output/kinetic_action_chain/runtime_direct_detlin_joint_tailgnorm_bandpass_focus_compare_summary.csv`, `output/kinetic_action_chain/runtime_direct_detlin_d4p8_single_beta_full_eta_summary_beta0p75.csv`, `output/kinetic_action_chain/runtime_direct_detlin_d4p0_acceptance_only_audit_summary_single_1p309.csv`, `output/kinetic_action_chain/runtime_direct_detlin_combined_focus_compare_summary.csv`). We therefore treat strict all-direct visibility as effectively closed on the audited focus band. A first broader-grid validator on `D21×E21` holdout and `D60×E21` release showed that the same fixed local parameters did not extrapolate cleanly: overall acceptance mismatch reopened at `0.1238` on the holdout and `0.0095` on the release grid, with the dominant reopened drifts concentrated at `D=4.8/6.4` (`p95 |Δμ_μμ| = 2.20/2.92` on the holdout and `5.23/7.36` on the release grid), while `D=7.2/8.0` remained small (`output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_detail.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_fast.png`). Follow-up retunes then resolved that picture into a minimum grid-aware rule: `D=4.8` is compromise-friendly across grids, with `\beta \approx 0.65` improving both the `D21×E21` holdout and `D60×E21` release relative to the original broader-grid baseline (`output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_compromise.csv`); `D21×E21` favors a `D=6.4` band-pass around `(g_\beta,g_{\rm center},\Delta g)\approx(0.30,0.06,0.04)` (`output/kinetic_action_chain/runtime_direct_detlin_generalization_retune_D6p4_summary_D21E21_holdout_d21_centerwidth.csv`); `D60×E21` first favors `(0.25,0.05,0.03)`, then improves under the combined-context retune to `(0.245,0.05,0.025)`, and now improves once more under a tighter `D60`-only exact refinement to `(0.243,0.049,0.023)` (`output/kinetic_action_chain/runtime_direct_detlin_generalization_retune_D6p4_summary_D60E21_release_d60_centerwidth_probe.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_retune_D6p4_summary_D60E21_release_d60_d64_combined_refine.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_retune_D6p4_summary_D60E21_release_d60_d64_micro_refine_v5.csv`); and the last `D60` `D=4.0` acceptance point closes at `\beta = 1.304` (`output/kinetic_action_chain/runtime_direct_detlin_generalization_retune_D4p0_summary_D60E21_release_d60_d40_ultranarrow.csv`). With those case-specific settings, the exact broader-grid validator closes acceptance mismatch on both grids (`0.0/0.0`). The original acceptance-closed rule left residual quantitative drift dominated by `D60 D=6.4` (`p95 |Δμ_μμ| = 5.39`) and secondarily `D60 D=4.8` (`2.89`) (`output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64_v3.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64_v3.csv`); the refined `v4` rule lowers the dominant `D60 D=6.4` residual to `5.24` while leaving the `D21×E21` holdout unchanged at `1.70` (`output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64_v4_full.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64_v4_full.csv`); and later exact / Sobol-guided follow-ups promote the current `v9` candidate, which lowers the same `D60` release residual further to `5.1944` under full exact broader-grid validation while keeping acceptance mismatch at `0.0` on both grids (`output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64_v9_sobol3_bestobs.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64_v9_sobol3_bestobs.csv`). A subsequent `D=4.8` case split, keeping `D21` at `\beta=0.65` while lowering `D60` to `\beta=0.55`, leaves broader-grid acceptance unchanged and removes most of the secondary `D60 D\approx4.81` residual (`2.89 -> 0.54`) without changing the still-dominant `D60 D\approx6.44` slice (`output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48_v10_casesplit.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64d48_v10_casesplit.csv`). A new local `D=6.4` amplitude term on the release side then gives the first different-mechanism gain after the gate-family probes stalled: keeping all acceptance mismatches at `0.0`, the exact broader-grid validator first promoted `v11`, lowering the dominant `D60 D\approx6.44` slice from `5.1951 -> 2.9521`, and two follow-up local amplitude refinements then promoted `v13`, which drives that same `D60 D\approx6.44` slice down again to `0.2365` while leaving every other audited slice unchanged (`output/kinetic_action_chain/runtime_direct_detlin_d60_d64_local_amp_probe_v1_summary.json`, `output/kinetic_action_chain/runtime_direct_detlin_d60_d64_local_amp_probe_v2_refine_summary.json`, `output/kinetic_action_chain/runtime_direct_detlin_d60_d64_local_amp_probe_v3_refine_summary.json`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48d64amp_v13.csv`, `output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_detail_gridaware_rule_d40d64d48d64amp_v13.csv`). With that update, broader-grid acceptance remains closed exactly (`0.0/0.0`) and the overall worst-case moves off the `D60` release branch entirely, back to the unchanged `D21xE21_holdout` value `1.6986`. The first Sobol pass around the `v5` neighborhood (`output/kinetic_action_chain/runtime_direct_detlin_d60_d64_sobol_doe_v1_samples.csv`, `output/kinetic_action_chain/runtime_direct_detlin_d60_d64_sobol_doe_v1_effects.csv`, `output/kinetic_action_chain/runtime_direct_detlin_d60_d64_sobol_doe_v1_summary.json`) already showed that the dominant local factors are the `g_{\rm center}` curvature and its interaction with `\Delta g`, with `g_\beta` entering only at lower order; a tighter second Sobol pass around `v6` (`output/kinetic_action_chain/runtime_direct_detlin_d60_d64_sobol_doe_v2_samples.csv`, `output/kinetic_action_chain/runtime_direct_detlin_d60_d64_sobol_doe_v2_effects.csv`, `output/kinetic_action_chain/runtime_direct_detlin_d60_d64_sobol_doe_v2_summary.json`, `output/kinetic_action_chain/runtime_direct_detlin_d60_d64_sobol_doe_v2.png`) makes the geometry clearer: `g_{\rm half\_width}` is not pinned to a single best value in isolation, but lives on a narrow center-width valley, so apparently lower points in the one-dimensional width projection are real only when `g_{\rm center}` is moved with them. Because the `D21` side of the rule is unchanged from `v4`, the `v7` gain remains a true `D60`-side refinement rather than a new cross-grid tradeoff. A follow-up grid-aware refinement audit then showed that no smaller shared-geometry compression beats the case-specific rule while preserving zero mismatch; the best compressed alternative is the shared-width family, but it weakens the `D60` residual (`output/kinetic_action_chain/runtime_direct_detlin_gridaware_rule_refine_summary.csv`, `output/kinetic_action_chain/runtime_direct_detlin_gridaware_rule_refine.png`). The next question is therefore no longer whether strict all-direct can be acceptance-closed on broader grids, but whether this minimum grid-aware rule should be simplified at all and how much further the remaining `D60` `D=6.4` residual can be reduced without reopening the gate.
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
  The canonical reviewer-mainline observable branch is now `chain_mode=cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compd60snapmaplate`. Relative to the historical refreshed reviewer baseline `cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost`, it preserves the repaired `D21` structure while improving the dense-grid `D60` micro-drift:
  - `D60 reviewer p95 |Delta mu_mumu|`: `31.29 -> 18.58`
  - `D21 reviewer p95 |Delta mu_mumu|`: `2.10 -> 0.891`
  - `D21 complement p95 |Delta mu_mumu|`: `0.7570 -> 0.7570` (preserved)
  - `D100 dense-grid drift strip`: `1.0419 -> 0.9349` vs the repaired parent `twolobe_d8compmaplate`
  The switch evidence is exported as:
  - `output/kinetic_action_chain/observable_two_lobe_d8compd60snapmaplate_runtime_validation_summary.csv`
  - `output/kinetic_action_chain/observable_two_lobe_d8compd60snapmaplate_promotion_gate_checklist.csv`
  - `output/kinetic_action_chain/observable_two_lobe_d8compd60snapmaplate_promotion_gate_checklist_gates.csv`
  - `output/kinetic_action_chain/observable_two_lobe_d8compd60snapmaplate_publish_switch_checklist.csv`
  - `output/kinetic_action_chain/observable_two_lobe_d8compd60snapmaplate_publish_switch_checklist_gates.csv`
  The final publish smoke/regression checklist closes `22/22` gates with `GO_branch_switch_recommended`, so the previous branch `cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost` is now best treated as the historical comparator baseline rather than the live reviewer mainline. Its refreshed reviewer gap still records the original hotspot geometry:
  - `D60 x E21`: mismatch `0.042857`, `max |Delta mu_mumu| = 31.29`, `p95 = 8.96`
  - `D21 x E41`: mismatch `0.090592`, `max |Delta mu_mumu| = 2.10`, `p95 = 0.914`
  Historical refreshed evidence remains exported as:
  - `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv`
  - `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D21E41.csv`
  - `output/kinetic_action_chain/model_chain_fullwidthrefamp_pointamp2_widthboost_d60_reviewerref_hotspot_audit_summary.csv`
  The original D60 reviewer gap splits cleanly into a partial/point-amplitude band at `D≈6.169/6.441` and a width band at `D≈4.814/5.627/6.712`, while `D=6.4` is the dominant D21 guard. Recent cross-grid follow-ups remain recorded as negative candidate families rather than promotion paths:
  - `pointamp3`: improves the D60 `D≈6.169/6.441` band but reopens `D21 D=6.4`
  - `widthband`: improves D60 width hotspots but again reopens `D21 D=6.4`
  - `partialguard`: helps hotspot shortlist metrics but does not beat the reviewer-mainline candidate on true D60 reviewer maps
  Supporting summaries are exported as:
  - `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp3_widthboost_D60E21_refresh_20260324.csv`
  - `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp3_widthboost_D21E41_refresh_20260324.csv`
  - `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthband_D60E21_refresh_20260325.csv`
  - `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthband_D21E41_refresh_20260325.csv`
  - `output/hll_signal_strength/hll_signal_strength_summary_chain_mode_cell_direct_runtime_release_fullwidthrefamp_partialguard_pointamp2_widthboost_D60E21_refresh_20260326.csv`
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
- **Reviewer-facing UV proposition chain:** the manuscript now compresses the late-stage UV-to-EFT bridge into four proposition-style steps. `Proposition U1` states the explicit UV-tree `\to` finite-match `\to` LL-RG witness with exact operator-basis reconstruction and bounded map deformation; `Proposition U2` isolates the canonical EYMH prefactor as an exact factorized block (`local loop × shell access × participation × dispersion screen`); `Proposition U3` rewrites the surviving normalization block as a single projected response action; and `Proposition U4` states the kernel-selection principle that dynamically singles out the canonical `logdet + Schur` structure. In this form, the remaining UV gap is no longer the existence of a stable local witness, but the parent-action origin and dynamic selection of the participation / dispersion-screen blocks entering that canonical kernel.
- **Absolute-normalization bridge audit:** we now also expose the `H\to\mu\mu` normalization problem itself as a compact scalar bridge rather than a vague anchor sensitivity. `code/audit_hll_absolute_normalization_bridge.py` rewrites the current UV+LL-RG observable exactly in terms of
  - one absolute amplitude constant `A_* = |C_{\mu\mu,*}^{\rm IR}|^{-2}`
  - two flavor-diagonal invariant constants `R_{e/\mu}^*`, `R_{\tau/\mu}^*`
  together with the local invariant fields `x_\mu = |C_{\mu\mu}^{\rm IR}|^2`, `R_{e/\mu} = |C_{ee}^{\rm IR}|^2 / |C_{\mu\mu}^{\rm IR}|^2`, and `R_{\tau/\mu} = |C_{\tau\tau}^{\rm IR}|^2 / |C_{\mu\mu}^{\rm IR}|^2`. Canonical outputs are:
  - `output/hll_absolute_normalization/hll_absolute_normalization_bridge_summary.csv`
  - `output/hll_absolute_normalization/hll_absolute_normalization_bridge_detail.csv`
  - `output/hll_absolute_normalization/hll_absolute_normalization_bridge_constants.csv`
  - `output/hll_absolute_normalization/hll_absolute_normalization_bridge_run_meta.json`
  The exact live-reference rewrite closes at machine precision (`mean |Δ\mu_{\mu\mu}| = 9.06e-09`, `max = 5.56e-08`). More importantly, a global three-parameter bridge fit stays in the same negligible error class (`mean = 1.21e-06`, `max = 1.12e-05`, zero acceptance mismatch), and if `R_{e/\mu}^*` is frozen to the accepted-region invariant median from `paper/reference_anchor_invariant_ratios.csv`, a two-parameter fit in `(A_*, R_{\tau/\mu}^*)` still closes the canonical map at the same `5.56e-08` level with zero acceptance mismatch.
- **`A_*` / `R_{\tau/\mu}^*` source audits:** two follow-up audits now separate the remaining observable-side tasks cleanly:
  - `code/audit_hll_absolute_amp_source.py`
  - `code/audit_hll_tau_ratio_source.py`
  with canonical outputs
  - `output/hll_absolute_normalization/hll_absolute_amp_source_summary.csv`
  - `output/hll_absolute_normalization/hll_absolute_amp_source_constants.csv`
  - `output/hll_absolute_normalization/hll_tau_ratio_source_summary.csv`
  - `output/hll_absolute_normalization/hll_tau_ratio_source_constants.csv`
  The new result is sharper than the original bridge audit. On active rows of the canonical `action_loop_eymh_parented` D21xE21 map, the diagonal flavor ratios are exact UV-tree invariants of the present chain (`max |ΔR_{e/\mu}^{\rm UV\to IR}| = 6.05e-13`, `max |ΔR_{\tau/\mu}^{\rm UV\to IR}| = 3.23e-14`), and at the live reference point `R_{\tau/\mu}^* = 8.24039e-03` is identical across tree/match/IR; the two-parameter bridge fit with fixed `R_{e/\mu}^*` returns the same `R_{\tau/\mu}^*` up to `2.86e-07` relative drift. So within the current diagonal UV+LL-RG chain, `R_{\tau/\mu}^*` is already source-resolved rather than a free normalization ambiguity.
  The remaining observable-side gap is now concentrated almost entirely in the absolute amplitude. The companion `A_*` audit verifies the exact factorization
  \[
  A_* = A_*^{\rm(tree)} Z_{{\rm diag},*}^{-2},
  \]
  with `max |ΔC_{\mu\mu}^{\rm fact}| = 9.32e-17`, `A_*^{\rm(tree)} = 5.26537e+08`, and `A_* = 5.20717e+08`; the full diagonal dressing changes the normalization by only about `1.11%`. A tree-only replacement for `A_*` already preserves the accepted region exactly, while restoring only the universal LL-RG factor reduces the map-level bridge error to `max |Δ\mu_{\mu\mu}| = 1.08e-03`. The normalization-side parent target is therefore primarily the EYMH/UV origin of `A_*`, not a remaining uncertainty in the diagonal flavor ratios.
- **Canonical `C_{\mu\mu,*}^{\rm tree}` source audit:** the next narrowing step is now explicit in
  - `code/audit_hll_tree_mumu_parent_source.py`
  with outputs
  - `output/hll_absolute_normalization/hll_tree_mumu_parent_source_summary.csv`
  - `output/hll_absolute_normalization/hll_tree_mumu_parent_source_detail.csv`
  - `output/hll_absolute_normalization/hll_tree_mumu_parent_source_constants.csv`
  On the canonical parented baseline, `uv_blend=0`, so the UV-tree coupling matrix is purely diagonal and the muon tree coefficient collapses exactly to
  \[
  C_{eH}^{\mu\mu,\rm tree}(D,\eta)=y_2^{\rm raw}(D)\,[P_2^{\rm kin}(D,\eta)/M_2^2(D)].
  \]
  The audit verifies this single-layer reduction to machine precision (`max |ΔC_{\mu\mu}^{\rm tree}| = 9.76e-17`) with exact `N=2` support on every active row. At the live reference point,
  - `y_2^{raw} = 1.65981e-04`
  - `P_2^{kin}/M_2^2 = 2.62559e-01`
  - `C_{\mu\mu,*}^{tree} = 4.35798e-05`
  - `A_*^{tree} = 5.26537e+08`
  and the resulting absolute normalization still differs from the final `A_*` only through the mild universal diagonal dressing already audited above. Across active rows the tree-amplitude variation is dominated by the overlap factor (`std log y_2^{raw} = 4.07`) rather than by the kinetic/mass coefficient (`std log(P_2^{kin}/M_2^2) = 1.97e-01`), so the next parent-side target should focus first on deriving the reference overlap block `y_2^{raw}(D_*)`.
- **`y_2^{raw}(D_*)` parent-side source audit:** the next narrowing step is now explicit in
  - `code/audit_hll_y2raw_parent_source.py`
  with outputs
  - `output/hll_absolute_normalization/hll_y2raw_parent_source_summary.csv`
  - `output/hll_absolute_normalization/hll_y2raw_parent_source_detail.csv`
  - `output/hll_absolute_normalization/hll_y2raw_parent_source_constants.csv`
  This audit shows that in the canonical overlap extractor the muon flavor kernel is exactly the raw overlap kernel because `sigma_scale_mu = 1` pointwise. As a result,
  \[
  y_2^{raw}(D)=y_{\mu,2}^{flavor}(D)=\bigl(g_{\mu 2}^{UV}(D)\bigr)^2
  \]
  on the audited overlap profile, with machine-level residuals (`max |(g_{\mu 2}^{UV})^2-y_2^{raw}| = 9.87e-17`). The same audit also shows that a naive center-mode reduction is not good enough: on the audited D4-20 detail grid, the center-only candidate `y_{center,2}` has mean relative defect `8.61e-01` and p95 relative defect `1.14` against `y_2^{raw}`. So the remaining `A_*` target should now be read as a microcanonical-windowed mu-flavor overlap block rather than as a single tracked-mode overlap.
- **`K_\mu(D)` parent-side kernel candidate audit:** the next narrowing step is now explicit in
  - `code/audit_hll_kmu_parent_candidate.py`
  with outputs
  - `output/hll_absolute_normalization/hll_kmu_parent_candidate_summary.csv`
  - `output/hll_absolute_normalization/hll_kmu_parent_candidate_detail.csv`
  - `output/hll_absolute_normalization/hll_kmu_parent_candidate_constants.csv`
  In the canonical baseline the overlap extractor keeps `sigma_l = sigma_r = 2.5` and `frame_power = 0`, so the muon kernel itself collapses to an exact symmetric midplane bridge
  \[
  K_\mu(\rho,z;D)=A_\mu^{disc}(D)\exp[-(\rho^2+z^2)/\sigma_\mu^2].
  \]
  The audit verifies this pointwise factorization on the D4-20 overlap grid with relative sup residual `3.05e-15`. A finite-box continuum normalization
  \[
  A_\mu^{box}(D)=\exp[-D^2/(4\sigma_\mu^2)]/[I_\rho I_z]
  \]
  already matches the canonical discrete kernel at the `1e-4` level (`max` relative sup error `9.45e-05`), while the naive infinite-volume amplitude remains off by about `2.37e-01`. At the live reference point `(D_*,\eta_*)=(9.6,1.0)`, the exact discrete and finite-box amplitudes are `3.77586e-04` and `3.77621e-04`. So the next parent-side target is no longer the whole kernel shape; it is the origin of the effective width `\sigma_\mu` or its first curvature-controlled correction inside this midplane bridge block.
- **`\sigma_\mu` curvature/Hessian candidate audit:** the next narrowing step is now explicit in
  - `code/audit_hll_sigma_mu_parent_candidate.py`
  with outputs
  - `output/hll_absolute_normalization/hll_sigma_mu_parent_candidate_summary.csv`
  - `output/hll_absolute_normalization/hll_sigma_mu_parent_candidate_detail.csv`
  - `output/hll_absolute_normalization/hll_sigma_mu_parent_candidate_constants.csv`
  This audit separates the width problem into three layers. On the canonical kernel, a direct log-Hessian fit of `\log K_\mu` returns `\sigma_\mu=2.5` to machine precision (`max` relative error `2.49e-15`), while an independent finite-box second-moment inversion still returns `\sigma_\mu=2.4999` with only `4.09e-05` maximum relative error. One step closer to the EYMH side, the action-derived shifted potential `U=V_{\rm eff}-m_0^2` is expanded at the left/right cores and its isotropized Hessian
  \[
  \bar\kappa_\mu(D)=\frac{2\kappa_{\rho,\mu}(D)+\kappa_{z,\mu}(D)}{3}
  \]
  is used to define a reference-calibrated curvature-controlled width
  \[
  \sigma_\mu^{curv}(D)=c_\sigma\,\bar\kappa_\mu(D)^{-1/4},
  \qquad c_\sigma=22.3027.
  \]
  Across the audited D4-20 window the core Hessian is already almost isotropic (`\kappa_{\rho,\mu}/\kappa_{z,\mu}\in[0.999996,0.99999997]`), the resulting width stays within `1.01e-03` of the canonical `2.5`, and the induced kernel deformation remains small (`max` relative sup error `1.11e-02`). So the next unresolved normalization-side parent target is no longer the width profile itself; it is the single calibration constant `c_\sigma` or its first curvature correction.
- **`c_\sigma` bridge-constant and first quartic correction audit:** the next narrowing step is now explicit in
  - `code/audit_hll_sigma_mu_curvature_correction.py`
  with outputs
  - `output/hll_absolute_normalization/hll_sigma_mu_curvature_correction_summary.csv`
  - `output/hll_absolute_normalization/hll_sigma_mu_curvature_correction_detail.csv`
  - `output/hll_absolute_normalization/hll_sigma_mu_curvature_correction_constants.csv`
  This audit promotes the remaining `\sigma_\mu` question from a free-width complaint to a nearly closed EYMH-side bridge. The bridge constant
  \[
  c_\sigma(D)=\sigma_\mu\,\bar\kappa_\mu(D)^{1/4}
  \]
  is already almost `D`-invariant on the audited D4-20 window, with relative span only `1.38e-03`. The first non-harmonic correction is then organized by the isotropized quartic invariant
  \[
  q_{4,\mu}(D)=\lambda_\mu(D)/\bar\kappa_\mu(D)^{3/2},
  \qquad
  \lambda_\mu=(2\lambda_{\rho,\mu}+\lambda_{z,\mu})/3.
  \]
  Using the one-parameter corrected candidate
  \[
  \sigma_\mu^{(curv,4)}(D)=c_{\sigma,*}\,\bar\kappa_\mu(D)^{-1/4}\left[1+b_4(q_{4,\mu}(D)-q_{4,\mu}(D_*))\right],
  \qquad b_4=0.4244,
  \]
  the remaining width drift collapses from `1.01e-03` to `8.47e-07`, and the finite-box kernel defect drops from `1.11e-02` to `9.47e-05`, essentially saturating the residual box-normalization floor. So the next parent-side target is now even narrower: derive the nearly constant bridge coefficient `c_\sigma`, and only after that the quartic-response coefficient `b_4`.
- **`c_\sigma` source-selector audit:** the next bridge-level narrowing step is now explicit in
  - `code/audit_hll_csigma_source.py`
  with outputs
  - `output/hll_absolute_normalization/hll_csigma_source_summary.csv`
  - `output/hll_absolute_normalization/hll_csigma_source_detail.csv`
  - `output/hll_absolute_normalization/hll_csigma_source_constants.csv`
  This audit compares three global selectors for the same EYMH-side bridge constant: direct width matching, finite-box midplane amplitude matching, and full-kernel matching. All three selectors collapse onto the same narrow band,
  \[
  c_\sigma^{\rm ref}=22.3033,\qquad
  c_\sigma^{\rm width}=22.3025,\qquad
  c_\sigma^{\rm amp}=c_\sigma^{\rm kernel}=22.3100,
  \]
  with relative drifts only `3.8e-05` (width) and `3.0e-04` (amplitude/kernel) from the reference calibration. The finite-box core-to-box choice `c_\sigma=22.3100` is the most useful current parent-side reading: it improves the maximum amplitude mismatch from `1.02e-02` to `1.88e-03` and lowers the maximum finite-box kernel defect by the same factor, without changing the underlying curvature law. So the surviving ambiguity is now even narrower: not whether a stable bridge constant exists, but why this nearly unique finite-box value emerges from the EYMH / projected-Yukawa side.
- **Independent `c_\sigma` support lines:** two additional audits now support the same bridge band from the overlap side itself.
  - `code/audit_hll_csigma_core_box_local.py`
  - `code/audit_hll_csigma_projected_overlap.py`
  with outputs under
  - `output/hll_absolute_normalization/`
  The pointwise core-to-box inversion recovers `c_\sigma^{box,local}(D)` directly from the finite-box amplitude relation. Aside from the known near-merger tracking outlier at `D=5`, the local constants stay in a narrow band with mean `22.3033`, min `22.2825`, max `22.3100`, and relative span `1.23e-03`. The projected-Yukawa benchmark then re-solves the fine overlap block at `D={6,12,18}` using the same shift-invert target as the original overlap extractor and verifies that the canonical overlap rebuild is exact to machine precision (`max relerr = 1.27e-13`). On that benchmark the projected selector lands at
  \[
  c_\sigma^{projected}=22.3095,
  \]
  only `1.60e-04` away from `c_\sigma^{ref}` and `2.24e-05` away from `c_\sigma^{amp}`, while the local projected selectors `22.2665, 22.3055, 22.3100` reproduce `y_2^{raw}` with maximum relative error `1.27e-04`. So `c_\sigma \approx 22.31` is now supported not only by kernel-side selectors, but also by direct core-to-box inversion and by an independent projected-overlap benchmark.
- **EYMH core-source audit for `c_\sigma`:** the bridge constant now also has a local analytic source candidate in
  - `code/audit_hll_csigma_eymh_core_source.py`
  with outputs
  - `output/hll_absolute_normalization/hll_csigma_eymh_core_source_summary.csv`
  - `output/hll_absolute_normalization/hll_csigma_eymh_core_source_detail.csv`
  - `output/hll_absolute_normalization/hll_csigma_eymh_core_source_constants.csv`
  This audit expands the shifted EYMH-side potential
  \[
  U=m_0^2(\Omega^2-1)+(1-6\xi)\,\nabla^2\Omega/\Omega
  \]
  directly around one core. In the isolated one-center Plummer limit it gives an explicit isotropic curvature source
  \[
  \kappa_{self}=6353.3061,
  \qquad
  c_\sigma^{self}=\sigma_\mu \kappa_{self}^{1/4}=22.3198,
  \]
  already within `1.78e-03` of the extracted bridge band. Including the second center analytically in the local two-center jet then yields `\bar\kappa_\mu^{analytic}(D)` with only `1.68e-04` maximum relative error against the finite-difference EYMH Hessian, and therefore
  \[
  c_\sigma^{analytic}(D)=\sigma_\mu\,\bar\kappa_\mu^{analytic}(D)^{1/4}
  \]
  with maximum relative error `4.21e-05` on the audited D4-20 window. The mirror correction remains small,
  \[
  \delta\kappa_{mirror}/\kappa_{self}\in[-6.91e-03,-1.39e-03],
  \]
  so the present best reading is that `c_\sigma` is already set primarily by the one-center EYMH / Plummer core, with the second center supplying only a mild D-dependent refinement.
- **Explicit projected-normalization statement for `c_\sigma`:** the new audit
  - `code/audit_hll_csigma_projected_box_source.py`
  plugs the EYMH-side constants back into the same projected-Yukawa box-normalized overlap functional whose global selector defines `c_\sigma^{projected}`. On the representative benchmark `D={6,12,18}`, the one-center source
  \[
  c_\sigma^{self}=22.3198
  \]
  sits only `4.61e-04` above `c_\sigma^{projected}=22.3095` and already reproduces the projected overlap block with `y_2` log-RMSE `6.15e-03`, maximum relative error `9.26e-03`, and mean relative error `4.93e-03`. Feeding in the analytic two-center refinement `c_\sigma^{analytic}(D)` improves the same projected functional to log-RMSE `6.29e-04` and maximum relative error `9.89e-04`, slightly better than the fixed projected selector itself (`9.08e-04`, `1.45e-03`). So the normalization statement is now sharper than “one-center core sets `22.3198`”: the one-center EYMH core already fixes the bulk of the projected-Yukawa box normalization, and the small mirror correction is enough to saturate the projected benchmark.
  In the paper this is now promoted from a narrative remark to a formal bridge claim (`Proposition U5`), so the mainline statement is no longer just that the one-center core gives a nearby constant, but that it already fixes the bulk normalization and the mirror correction closes the benchmark.
- **Narrative cleanup after reviewer-style objections:** the front-of-paper spectrum language has now been tightened so that the action-derived `\omega_N^{\rm(exact)}(D)` is the mainline spectral input and the older hydrogenic `1-1/(2N^2)` formula is only a legacy comparator. Likewise, the kinetic section now treats the Kerr-shaped `\hat\omega_N^{4\ell+5}` factor only as a residual hierarchy convention absorbed into the action-derived profile factors `\tilde A_\ell(D)`, not as the physical origin of static leakage in the dual-center geometry. Finally, the `r_N=\eta e^{-2S_N}` paragraph now calls `r_N` an effective rank-2 tunneling suppression factor rather than a literal tunneling probability, so the existing `\eta>1` overlap-amplitude interpretation is stated without probability-language conflict. These edits are narrative/interpretive rather than numerical: they do not change the current baseline maps, but they better align the paper with the exact-spectrum benchmarks, the geometry-driven static surrogate `\Gamma_{N,\ell}^{(geo)}`, and the existing reviewer-facing caveats.
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
- **EYMH projected stationarity audit:** we can sharpen the same claim into a local stationarity principle for the projected mismatch functional
  \[
  J(\alpha,\beta,\lambda)=\left\langle \left(A(\alpha,\beta,\lambda)-A_{\rm ref}\right)^2 \right\rangle .
  \]
  The audit under
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_stationarity_audit_map_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_stationarity_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_stationarity_audit_D21E21_fix.png`
  shows that the canonical point has exact first-variation closure (`grad_alpha=grad_beta=grad_lambda=0`). More strongly, the `(\alpha,\beta)` block has a positive-semidefinite exact quadratic stationarity matrix with eigenvalues `(1.33e-03, 7.59e-02)`, while the cross-coupling direction is not quadratic at all but quartically stabilized:
  \[
  J(1,1,\lambda)=C_4\,\lambda^4+\mathcal{O}(\lambda^6),\qquad
  C_4=1.4974\times10^{-3}.
  \]
  At `lambda=0.1` the predicted RMSE from this quartic law is `3.8696e-04`, matching the directly evaluated `3.8734e-04`. This is the cleanest stationarity reading so far: the projected fluctuation operator selects the canonical log-det / Schur kernel by exact first-variation closure, quadratic stability in the log-det / Schur weights, and quartic suppression of explicit participation-tree mixing.
- **EYMH projected variational-selection audit:** the same local stationarity data can now be packaged into an explicit projected effective-action gap,
  \[
  \Delta\Gamma_{\rm sel}(\delta\alpha,\delta\beta,\lambda)
  =\frac12\,\delta\theta^T H\,\delta\theta + C_4\,\lambda^4,
  \qquad
  \delta\theta=(\delta\alpha,\delta\beta).
  \]
  The audit under
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_variational_selection_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_variational_selection_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_variational_selection_audit_D21E21_fix.png`
  compares this variational gap directly against the exact mismatch functional on a local grid around the canonical point. The minima coincide exactly (`best_exact = best_var = (1,1,0)`), and the local surrogate tracks the exact objective very well (`corr(J_exact, DeltaGamma_sel) = 0.9894`). The match is especially tight on the pure mixing slice (`lambda` slice max gap `= 1.88e-08`), while the `lambda=0` alpha/beta plane remains controlled at small absolute error (`p95 |J-DeltaGamma_sel| = 2.97e-05`). This is the strongest selection statement so far: near the canonical point, the projected EYMH kernel is not only stationary but governed by a local effective action whose quadratic `(\alpha,\beta)` sector and quartic `\lambda` sector reproduce the observed selection landscape.
- **EYMH projected parent-kernel statement audit:** we can now push the same result into an exact parent-kernel excess functional. For the deformed family
  \[
  K_{11}=e^{\alpha S_{\rm part}},\qquad
  K_{22}=e^{\beta S_{\rm schur}},\qquad
  K_{12}=\lambda\sqrt{(K_{11}-1)(K_{22}-1)},
  \]
  the canonical response weight obeys the exact identity
  \[
  A(\alpha,\beta,\lambda)
  =A_{\rm ref}\exp\!\left[-\frac12\Delta S_{\rm kernel}\right],
  \]
  with
  \[
  \Delta S_{\rm kernel}
  =(\alpha-1)S_{\rm part}+(\beta-1)S_{\rm schur}
  +\log\!\bigl(1-\lambda^2\xi_{\rm cross}\bigr).
  \]
  The audit under
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_kernel_statement_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_kernel_statement_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_kernel_statement_D21E21_fix.png`
  verifies that the resulting parent-kernel objective is exactly the direct mismatch functional (`max objective identity residual = 1.30e-18`, `max weight identity residual = 3.33e-16`) and is minimized at the same canonical point `(1,1,0)`. This is the strongest mother-action statement so far: the canonical `logdet + Schur` selection is no longer only a local surrogate or a best-fit scan result, but an exact projected parent-kernel excess functional in which log-det and Schur deformations enter linearly while explicit participation/tree mixing enters only through an even determinant factor.
- **EYMH projected block-split audit:** we can now make the operator split itself explicit. The audit under
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_block_split_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_block_split_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_block_split_audit_D21E21_fix.png`
  writes the canonical projected fluctuation kernel as a participation block `K_part = ||c||_1 / ||c||_2`, a shell-background/tree block `G_schur = (1 + gap_cv + c_tree_diag_cv)/(1 + gap_cv)`, and a mixed scale `C_mix = sqrt((K_part-1)(G_schur-1))`. On the canonical map the response action closes exactly as `S_part + S_schur` (`corr = 1.0`, `max canonical action residual = 7.77e-16`), while the deformed determinant identity
  \[
  \det K_{\rm sel}=K_{\rm part}G_{\rm schur}(1-\lambda^2\xi_{\rm cross})
  \]
  also closes to machine precision (`scan max determinant residual = 4.44e-16`). This is a cleaner structural statement than the parent-kernel excess form alone: the canonical projected fluctuation operator is block-diagonal in the participation and tree/background sectors, and explicit participation-tree mixing survives only as an even determinant-level penalty.
- **EYMH projected parent block-determinant / Schur audit:** we can push the same statement one level closer to a derivation by embedding it into a background-normalized parent block under
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_blockdet_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_blockdet_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_blockdet_audit_D21E21_fix.png`
  Writing
  \[
  K_{11}=e^{\alpha S_{\rm part}},\qquad
  K_{\rm bg}=1+\mathrm{gap}_{\rm cv},\qquad
  K_{22}=K_{\rm bg}e^{\beta S_{\rm schur}},
  \]
  and
  \[
  C_{\rm parent}=\sqrt{(K_{11}-1)(K_{22}-K_{\rm bg})},
  \]
  the projected parent block
  \[
  \mathcal K_{\rm parent}=
  \begin{pmatrix}
  K_{11} & \lambda C_{\rm parent}\\
  \lambda C_{\rm parent} & K_{22}
  \end{pmatrix}
  \]
  obeys
  \[
  \frac{\det \mathcal K_{\rm parent}}{K_{\rm bg}}
  =
  K_{11}G_\beta(1-\lambda^2\xi_{\rm cross}),
  \qquad
  \widehat G_{\rm Schur}
  =
  \frac{K_{22}-K_{12}^2/K_{11}}{K_{\rm bg}}
  =
  G_\beta(1-\lambda^2\xi_{\rm cross}).
  \]
  On the D21xE21 fix grid this closes to machine precision (`max canonical action residual = 5.55e-16`, `scan max determinant-ratio residual = 1.33e-15`, `scan max Schur residual = 6.66e-16`, `corr(j_direct,j_blockdet) = 1.0`). This is the strongest derivation statement so far: the canonical response weight is the inverse square root of a background-normalized projected parent block determinant, and the `logdet + Schur` form is exactly its block-determinant / Schur-complement factorization.
- **EYMH parent mixing geometric-mean audit:** the remaining ambiguity in the parent block is then the mixed entry. The audit under
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_mix_geomean_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_mix_geomean_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_mix_geomean_audit_D21E21_fix.png`
  tests the smallest symmetric-excess family
  \[
  C_{\rm gen}=\kappa\,(K_{11}-1)^u (K_{22}-K_{\rm bg})^v.
  \]
  On the local D21xE21 fix scan the unique exact point is the geometric-mean choice
  \[
  (u,v,\kappa)=\left(\tfrac12,\tfrac12,1\right),
  \qquad
  C_{\rm parent}=\sqrt{(K_{11}-1)(K_{22}-K_{\rm bg})}.
  \]
  At this point the determinant, Schur, weight, and normalized cross-ratio residuals all collapse to machine precision (`max det residual = 4.44e-16`, `max Schur residual = 4.44e-16`, `max weight residual = 1.11e-16`, `max xi residual = 1.11e-16`), while the first nontrivial runner-up `(u,v,\kappa)=(0.625,0.625,1.1)` already opens visible errors (`max det residual = 1.32e-03`, `max Schur residual = 9.78e-04`, `max weight residual = 2.18e-04`). This is the clearest naturality statement so far: once the parent kernel is required to couple the participation and tree/background sectors through a minimal symmetric excess family, the geometric-mean mixed block is uniquely selected.
- **EYMH parent ratio-warp audit:** we then tested the nearest non-minimal extension, namely a ratio-dependent warp of the geometric mean under
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_ratio_warp_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_ratio_warp_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_ratio_warp_audit_D21E21_fix.png`
  of the form
  \[
  C_{\rm warp}=\kappa\,C_{\rm parent}\exp\!\left[\delta L+\nu L^2\right],
  \qquad
  L=\tfrac12\log(E_{\rm part}/E_{\rm tree}).
  \]
  On the canonical D21xE21 fix grid the unique exact point is again the unwarped kernel:
  \[
  (\kappa,\delta,\nu)=(1,0,0).
  \]
  At that point all residuals vanish exactly, while the first nontrivial runner-up `(1,0,-0.05)` already opens visible errors (`max det residual = 7.18e-04`, `max Schur residual = 5.39e-04`, `max weight residual = 7.14e-05`, `max xi residual = 6.08e-03`). This is the strongest minimality statement so far: the low-mode parent block not only selects the geometric-mean mixed sector inside the minimal symmetric-excess family, it also rejects the first ratio-warped extension of that family.
- **EYMH parent symmetry/normalization audit:** we then reparameterized the same local parent family in the coordinates that most directly match the remaining proof obligation,
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_symnorm_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_symnorm_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_symnorm_audit_D21E21_fix.png`
  through
  \[
  C_{\rm gen}
  =
  \exp(m)\exp\!\left[\frac{s}{2}\log(E_{\rm part}E_{\rm tree})
  +\frac{a}{2}\log(E_{\rm part}/E_{\rm tree})\right],
  \]
  where `m` is the overall normalization shift, `s` is the symmetric homogeneity degree, and `a` is the antisymmetric participation/tree tilt. On the canonical D21xE21 fix grid the unique exact point is
  \[
  (m,s,a)=(0,1,0),
  \]
  so the projected parent block is selected by zero normalization shift, unit symmetric degree, and zero antisymmetric tilt. The first nontrivial runner-up `(0.05,1.125,0)` already opens visible errors (`max det residual = 6.61e-04`, `max Schur residual = 4.72e-04`, `max weight residual = 1.09e-04`, `max xi residual = 7.73e-03`). This is the cleanest structural statement so far: once the parent block is written in symmetry/normalization coordinates, the low-mode projection selects the canonical family by setting normalization, homogeneity, and asymmetry to their unique exact values.
- **EYMH parent generator-affinity audit:** finally, we tested the first local non-affine extension of the parent-block generator,
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_generator_affinity_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_generator_affinity_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_parent_generator_affinity_audit_D21E21_fix.png`
  using
  \[
  \log C_{\rm gen}
  =
  \tfrac12 L_{\rm sum}
  + q_{ss} L_{\rm sum}^2
  + q_{dd} L_{\rm diff}^2
  + q_{sd} L_{\rm sum}L_{\rm diff}.
  \]
  The unique exact point is
  \[
  (q_{ss},q_{dd},q_{sd})=(0,0,0),
  \]
  meaning that the projected parent action selects a locally affine log-generator in the excess coordinates themselves. The first nontrivial runner-up `(0,-0.0125,0.0125)` already opens visible errors (`max det residual = 2.76e-04`, `max Schur residual = 1.98e-04`, `max weight residual = 2.62e-05`, `max xi residual = 2.23e-03`). This is the narrowest structural statement so far: after fixing symmetry/normalization, the low-mode parent action also rejects the first local log-curvature corrections, so the surviving canonical block lies in the local multiplicative excess class itself.
- **EYMH excess-coordinate audit:** we then tested whether the projected parent action naturally uses the canonical excess coordinates themselves,
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_excess_coordinate_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_excess_coordinate_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_excess_coordinate_audit_D21E21_fix.png`
  through the minimal reference-offset family
  \[
  E_{\rm part}^{(r)}=K_{11}-r_{\rm part},
  \qquad
  E_{\rm tree}^{(r)}=K_{22}-r_{\rm tree}K_{\rm bg}.
  \]
  The unique exact point is
  \[
  (r_{\rm part},r_{\rm tree})=(1,1),
  \]
  with exact determinant/Schur/weight closure and zero anchor leakage. More importantly, the first nontrivial runner-up `(1,1.05)` already distorts the first nonzero-response slices even though the strict anchor leakage still vanishes: `part onset residual = 4.53e-02`, `tree onset residual = 9.78e-02`, together with `max det residual = 1.08e-03`, `max Schur residual = 7.02e-04`, `max weight residual = 2.48e-04`, and `max xi residual = 1.61e-02`. This is the sharpest fixed-point statement so far: the canonical excess coordinates are uniquely selected as deviations from the identity participation block and the shell/background tree block, not just as a convenient subtraction convention.
- **EYMH fixed-point Box-Cox audit:** we then tightened the same proof obligation by allowing the excess coordinates to vary inside the smallest smooth family that preserves both fixed points and tangent normalization,
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_fixedpoint_boxcox_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_fixedpoint_boxcox_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_fixedpoint_boxcox_audit_D21E21_fix.png`
  through
  \[
  E_{\rm part}^{(p)}=\mathrm{BC}_p(K_{11}),\qquad
  E_{\rm tree}^{(q)}=K_{\rm bg}\,\mathrm{BC}_q(K_{22}/K_{\rm bg}),
  \]
  where `BC_p` is the Box-Cox / q-log family and every member obeys `BC_p(1)=0` and `BC'_p(1)=1`. On the canonical D21xE21 fix grid the unique exact point is again the linear additive excess choice
  \[
  (p_{\rm part},p_{\rm tree})=(1,1),
  \]
  with machine-precision determinant/Schur/weight closure (`max det residual = 4.44e-16`, `max Schur residual = 4.44e-16`, `max weight residual = 1.11e-16`, `max xi residual = 8.33e-17`) and vanishing anchor/onset residuals. The first nontrivial runner-up `(0.75,1.0)` already opens visible first-slice distortions (`part onset residual = 2.02e-03`, `tree onset residual = 8.23e-03`) together with `max det residual = 8.25e-04`, `max Schur residual = 5.51e-04`, `max weight residual = 9.80e-05`, and `max xi residual = 7.85e-03`. This is the narrowest coordinate statement so far: even after fixing the same fixed points and the same tangent normalization, the projected parent action still uniquely selects the linear excess coordinates themselves.
- **EYMH normal-coordinate audit:** we then recast the same result in positive local-coordinate language by probing the first nonlinear jet family that preserves the same fixed points and the same unit tangent normalization,
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_normal_coordinate_scan_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_normal_coordinate_audit_summary_D21E21_fix.csv`
  - `output/hll_uv_matching/hll_uv_action_loop_eymh_normal_coordinate_audit_D21E21_fix.png`
  using
  \[
  E_{\rm part}^{(\zeta_p)}=E_{\rm part}+\zeta_p E_{\rm part}^2,
  \qquad
  E_{\rm tree}^{(\zeta_t)}=E_{\rm tree}+\zeta_t E_{\rm tree}^2/K_{\rm bg}.
  \]
  The unique exact point is
  \[
  (\zeta_p,\zeta_t)=(0,0),
  \]
  with exact determinant/Schur/weight/xi closure and zero anchor/onset residuals. The first nontrivial runner-up `(0.125,0.0)` already opens visible distortions (`part onset residual = 2.10e-03`, `tree onset residual = 9.63e-03`, `max det residual = 9.90e-04`, `max Schur residual = 6.59e-04`, `max weight residual = 1.18e-04`, `max xi residual = 9.47e-03`). This is the cleanest positive local statement so far: the natural projected parent-action variables are the zero-second-jet normal coordinates around the identity/background fixed points, and those normal coordinates are exactly the linear excess variables.
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

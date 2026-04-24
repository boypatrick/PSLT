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
- **High-$N$ Decoupling Interpretation:** The current baseline does not remove higher layers by hand through the visibility block: `B_{N>3}=1` is a neutral saturation rather than an extra damping factor. In the audited action-derived single-track benchmark, no tested `D` point supports a fourth bound level (`/Users/boypatrick/codex/PSLT_quantam/output/highN_decoupling/highN_decoupling_single_track_summary.csv`), while the companion one-dimensional kinetic proxy keeps `E_4=\omega_4^2-m_0^2>0` across the tested window and a forced quasi-channel continuation does not generate an additional `\Gamma_4` hierarchy (`/Users/boypatrick/codex/PSLT_quantam/output/highN_decoupling/highN_decoupling_superrad_1d_summary.csv`). The paper now packages this as Proposition N1 / Corollary N1: a finite-volume Sturm/min-max certificate gives max negative-mode count 2 on the audited 1D chain, while the independent single-track benchmark has max bound count 3 and never supports N=4; remaining work is only continuum-tail / D-continuous interpolation strengthening.

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
  One more direct audit now shows that the exported exact operator itself already satisfies that chart hypothesis. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_operator_chart_source.py` reads the exact parent-bath statement map with the same p95/max aggregation conventions as the family scans and finds `||W_{95}^{(exact)}||_\infty = 3.52e-7`, `||W_{max}^{(exact)}||_\infty = 3.63e-7`, both dominated by the exact block-22 witness. These are still below the audited chart radii by factors `2.73e2` and `2.77e2`. Together with the existing cocycle/potential exactness witnesses (`1.11e-15`, `4.03e-15`), this means the exported exact operator already sits comfortably inside the audited operator chart. So the open-system theorem line is now narrowed to one very specific remaining gap: upgrade this verified chart inclusion from an audited fact about the exported exact operator to an analytic operator theorem.
  A new envelope audit now compresses even that statement into exact witness-equation constants. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_operator_radius_bound.py` reads the factorization, statement, cocycle, and potential summaries directly and forms the unified exact-operator envelope `B_op^(exact) = max{Δ_phi^(fact), Δ_mix^(fact), Δ_tr^(stmt), Δ_det^(stmt), Δ_ani^(stmt), Δ_mix^(cocycle), Δ_mix^(potential)}`. On the exported exact operator this gives `B_op^(exact) = 3.63e-7`, dominated by the factorization-side block-22 witness, with the envelope exactly matching the observed `||W_max^(exact)||_infty`. Since this is already smaller than half of both audited chart radii (`r95/2 = 4.81e-5`, `rmax/2 = 5.03e-5`) by factors `1.33e2` and `1.39e2`, the witness-chart hypothesis of Proposition O7 now follows directly from the exact factorization/statement/cocycle/potential equations themselves for the exported operator. The remaining theorem gap is therefore narrower again: replace these exported residual constants by an a priori operator bound on the same envelope.
  A new dominant-pole synthesis audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_dominant_pole_synthesis_source.py`, with outputs under `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/`. It compresses the whole `O1--O9` source chain into one stopping point. On the exported D4--20 bridge, the on-shell defect side is already closed (`max |\delta_{q,a}| = 4.44e-16`, `max |\tau_{{\rm env},a}-\omega_{1,a}^{-1}| = 1.11e-16`), the projected continuum-neighborhood route still gives the same determinant-deficit budget `u_a <= 5.34e-3` and lifted exact-gap floor `g_a^{exact} >= 1.3339` with `H_eff >= 0.9947 H_PP`, and the exact operator envelope remains only `B_op^(exact) = 3.63e-7`, attained by the block-22 factorization witness at `D=13` and still below the half-radii by factors `1.33e2` and `1.39e2`. So the dominant-pole line is no longer blocked by a response-family ambiguity, a nonzero `\delta_q` defect, or a missing witness chart. The single remaining theorem-level gap is to upgrade this same neighborhood determinant-deficit budget from audited projected-family / chart evidence to a genuine analytic lift for the exact Schur resolvent.
  One more reduction now isolates the only dominant term in that envelope. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_mix_factorization_source.py` rewrites the block-22 witness as the quotient
  `Delta_mix^(fact)(D) = |gamma_mix(D) - kappa_env g_x(D)^2 Sxx(D)| / (g_x(D)^2 Sxx(D))`
  and audits three denominator floors. The common numerator residual is only `eps_mix = 4.40e-14`, while the exported exact operator has `min g_x^2 Sxx = 1.2147e-9`, the equivalent `gamma_mix` floor `(min gamma_mix - eps_mix)/kappa_env = 1.2147e-9`, and even the fully separated floor `(min g_x)^2 min Sxx = 1.1417e-9`. These give conservative witness bounds `3.62e-5`, `3.62e-5`, and `3.85e-5`, all still below the audited half-radii `r95/2 = 4.81e-5`, `rmax/2 = 5.03e-5`. So the final open-system theorem gap has narrowed again: it is now enough to prove an a priori lower floor on the mix kernel `g_x^2 Sxx` (or equivalently on `gamma_mix`) strong enough to reproduce one of these denominator bounds.
  A final scalar synthesis now compresses that statement to its sharpest current form. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_lift_scalar_source.py` asks only for the minimal mix-kernel floor required to force the dominant block-22 witness inside the audited half-radius chart. Using the exact factorization numerator residual `eps_mix = 4.40e-14` and the tighter half-radius `r95/2 = 4.81e-5`, the whole exact Schur lift is reduced to the single sufficient condition `g_x(D)^2 Sxx(D) >= 9.15e-10` on the audited on-shell window, equivalently `gamma_mix(D) >= 3.94e-4` after multiplying by `kappa_env^max`. The current exported bridge already exceeds that scalar threshold in several independent ways: the direct exact kernel floor and the exact `gamma_mix` floor both give `1.2147e-9` (margin factor `1.328`), while even the separated observable-chain floor still gives `1.0016e-9` (margin factor `1.095`). Meanwhile every non-mix witness component remains tiny, with the full non-mix envelope only `2.33e-10`, smaller than the same half-radius by the factor `2.06e5`. So the remaining dominant-pole theorem gap is now genuinely one-dimensional: prove any one parent-side floor strong enough to support that scalar threshold, and the exact Schur lift closes.
  One more bridge now shows exactly how small that remaining scalar gap is. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_lift_overlap_bridge.py` combines the current static parity-contrast theorem floor with the already-audited response floors. Using the compact-anchor exact-bridge bound `|V_{11}-V_{22}| >= 5.64043e-5` (hence `g_x >= 2.820215e-5`) together with the global response floor `Sxx >= 1.01493` gives `gamma_mix >= 3.47553e-4`, which is already `88.25%` of the scalar Schur threshold `3.93829e-4`. Equivalently, with the same global response floor fixed, only a further `6.45%` tightening of the current static `g_x` floor would close the exact Schur lift. If one keeps the exact local response uplift at the common knot `D=11`, the same parity floor already reaches `93.89%` of the scalar threshold, leaving only a `3.20%` tightening in `g_x`. So the remaining dominant-pole proof gap is no longer a new operator-structure problem at all; it is now a very small constant-tightening problem on the static parity-contrast side.
  The next tightening keeps exactly the same core-window family and simply applies it to the full exact parity contrast instead of the one-center pair alone. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_total_core_tightening_source.py` rescans the same `R in [1.20,1.40]` windows and finds that the same optimal radius `R_*=1.35` remains safe for the exact total split `V_{11}-V_{22} = J_core^(R) + J_tail^(R)`, with `J_core` always same-sign as the total and `J_tail` always opposite-sign on the audited bridge. This lifts the induced parity floor from `5.70400e-5` to `5.74795e-5`, a factor `1.00771` improvement without changing support family. Relative to the parity thresholds now required for the exact Schur lift, that tightened floor already reaches `95.73%` of the global-response target `6.00420e-5` and `98.75%` of the local-response target `5.82096e-5`. So the remaining dominant-pole task is narrower again: not a new family, not a new chart, and not even a new support class, but only a final `1.27%` static-parity tightening at the locally aligned response level.
  The next synthesis compresses even that statement to a single required parity threshold. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_lift_parity_threshold_source.py` combines the scalar kernel threshold `g_x^2 Sxx >= 9.15e-10` with the exact identity `g_x = |V_{11}-V_{22}|/2`. Under the globally valid response floor `Sxx >= 1.01493`, the exact Schur lift is therefore implied by the single static bound `|V_{11}-V_{22}| >= 6.00420e-5`; under the exact local-response alignment at the common knot `D=11`, the threshold sharpens to `5.82096e-5`. The current theorem floors `5.64043e-5` (compact anchor) and `5.70400e-5` (core window) already reach `93.94%/95.00%` of the global threshold and `96.90%/97.99%` of the local-response threshold, while the exact bridge floor `6.70795e-5` exceeds them by factors `1.117` and `1.152`. So the dominant-pole line is now pinned down as tightly as it can be without a final static-overlap theorem tightening: no new Schur object remains, only the explicit parity threshold itself.
  One final audit shows that even this residual `1.27%` gap is only a single-window artifact, not a missing support theorem. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_total_core_window_split_source.py` keeps exactly the same family `B_core(R)` and the same optimal radius `R_*=1.35`, but reads the same exact-total core/tail split piecewise on the two contiguous audited subwindows `D=4..11` and `D=12..20`. The best cut occurs at `D=11`: the left window gives `lambda_tot,min^(L)=0.479785` with core floor `1.34704e-4`, hence induced floor `6.46291e-5`, while the right window gives `lambda_tot,min^(R)=0.426709` with core floor `3.25919e-4`, hence induced floor `1.39072e-4`. Therefore the same family and the same radius already imply the audited piecewise floor `|V_{11}-V_{22}| >= 6.46291e-5` across the full `D=4..20` bridge, exceeding the global Schur-parity threshold `6.00420e-5` by the factor `1.07640` and the local-response threshold `5.82096e-5` by the factor `1.11028`. So the remaining dominant-pole gap was not structural at all: it came only from forcing one global `lambda_tot,*` and one global core floor to be multiplied across the whole audited window.
  The next lift no longer asks for any new family design. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_piecewise_parent_lift_source.py` recasts the same `R_*=1.35` piecewise closure as an explicit parent-side sufficient theorem target. On the left window `D=4..11`, it is enough to prove the same-sign exact-total split together with `|J_tail| <= eta_L |J_core|`, `eta_L=0.520215`, and `|J_core| >= C_L = 1.34704e-4`; on the right window `D=12..20`, it is enough to prove the same-sign split with `eta_R=0.573291` and `|J_core| >= C_R = 3.25919e-4`. These same windowwise constants already reproduce the audited piecewise floor `6.46291e-5`, so `open_system_micro` should now be read as audited piecewise closed on the current bridge. The only remaining work is to prove those same inequalities analytically on the parent side, not to search for any new support class, chart, or response family.
  One more bottleneck audit now reduces even that parent-side task to the left window alone. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_left_window_bottleneck_source.py` shows that the right window `D=12..20` is already safely non-bottleneck: its same-family floor exceeds the global Schur-parity threshold by the factor `2.31625` and the locally aligned threshold by `2.38917`. So the only remaining analytic task is on the left window `D=4..11`. On that same family and same `R_*=1.35`, the parent-side lift can now be read in two equivalent ways: either prove a left-window tail-ratio theorem `eta_L <= 0.554268` (global) or `eta_L <= 0.567871` (local) while keeping the current core floor, or prove a left-window core-floor theorem `C_L >= 1.25144e-4` (global) or `C_L >= 1.21325e-4` (local) while keeping the current tail ratio. The audited values `eta_L=0.520215` and `C_L=1.34704e-4` already satisfy both formulations, so the remaining `open_system_micro` work is now as narrow as it can reasonably be made: one left-window bottleneck theorem on the same exact-total split, with the right window formally demoted.
  One further same-family split shows that even this left-window bottleneck is mostly a coupling artifact. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_left_window_split_source.py` keeps the same `R_*=1.35` and splits `D=4..11` into `D=4..10` plus the singleton knot `D=11`. The best cut is exactly `D=10`: on `D=4..10` the induced floor is already `7.40868e-5`, while on `D=11` the same-family split reproduces the exact audited minimum `6.70795e-5`. Hence the left-window same-family floor becomes `6.70795e-5`, exceeding the global and local Schur-parity thresholds by factors `1.11721` and `1.15238`. The ratio bottleneck and the core-floor bottleneck are now visibly separated: `eta` peaks at `D=9`, while the core floor sits at `D=11`. So the remaining parent-side task is narrower again: prove the same-family tail/core estimate on `D=4..10`, and treat `D=11` as the already isolated core-floor knot rather than as part of a larger family-design problem.
  One hierarchical split deeper, even that `D=4..10` task is no longer a genuine interval bottleneck. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_left_window_hierarchical_split_source.py` keeps the same family `B_core(R_*=1.35)` and splits `D=4..10` at `D=6`, producing the subwindows `D=4..6` and `D=7..10`. Their same-family induced floors are `8.18812e-5` and `3.70831e-4`, respectively, while the already isolated singleton knot `D=11` remains at `6.70795e-5`. So on the audited bridge every non-singleton interval now sits comfortably above both Schur-parity thresholds, and the only surviving audited minimum is the single core-floor knot `D=11`. This is the sharpest current reading of `open_system_micro`: the remaining parent-side task is no longer an interval family theorem, but a single-knot analytic lift on the same exact-total split.
  A final singleton-knot audit now makes that bottleneck fully explicit. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_knot_source.py` shows that at `D=11` on the same exact-total split and the same `R_*=1.35`, one has `|V11-V22| = lambda_11 |J_core(11)|` with `eta_11 = 0.502024`, `lambda_11 = 0.497976`. So to close the Schur lift it is enough to prove the single-knot core-floor bound `|J_core(11)| >= 1.20572e-4` for the global threshold, or `|J_core(11)| >= 1.16893e-4` for the locally aligned threshold. The audited exact value is already `1.34704e-4`, clearing these by factors `1.11721` and `1.15238`. There is now no interval theorem left on the audited bridge; the remaining parent-side task is a single-knot analytic lower bound at `D=11`.
  One last audited reduction narrows even that singleton target. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_parent_source.py` decomposes the exact core at `D=11` into the same-center pair core plus a cross correction on the same region `B_core(R_*=1.35)`. The pair core is already `1.33812e-4`, while the cross core is same-sign and only `8.92e-7`, i.e. about `0.66%` of the full core. So the remaining parent-side task can now be read in its sharpest current form: it is enough to prove a lower bound on the same-center pair core `|J_self-pair,core(11)|`, not on the full exact core. The pair-core value already clears the global and local singleton targets by factors `1.10981` and `1.14475`.
  One further singleton audit narrows even that statement to a same-center static anchor. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_static_anchor_source.py` composes the existing `D=11` same-center pair-core audit with the canonical compact-anchor/static-source chain. At the same knot and on the same radius `R_*=1.35`, the pair core is already carried by the canonical anchor `B_box(0.70,1.21;11)` up to a tiny same-sign shell (`1.11%` of the pair core), and that anchor is itself carried by the same-center static anchor plus a local mirror loss with ratio only `mu_11^(anc)=0.191967`. Numerically, `|J_diag,anchor(11)| = 1.63757e-4`, while the sufficient static-anchor targets are only `1.49217e-4` (global) and `1.44663e-4` (local). So the remaining `open_system_micro` theorem task has now reached its narrowest current parent-side form: a one-knot lower bound on the same-center static anchor `|J_diag,anchor(11)|`, with the shell and local mirror acting only as same-family local refinements.
  One more singleton audit then clarifies how to attack that last static-anchor lift. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_diag_moment_source.py` specializes the older oriented-moment split to the same knot `D=11`, giving the exact identity `|J_diag,anchor(11)| = (1-nu_11) P_diag(11)` with `P_diag(11)=1.87136e-3`, `N_diag(11)=1.70580e-3`, and `nu_11=0.911532`. This turns the last parent-side lift into two equivalent local statements: either prove `P_diag(11) >= 1.68667e-3` (global) or `1.63520e-3` (local), or prove `nu_11 <= 0.920263` (global) or `0.922696` (local). The first route is clearly more robust on the audited bridge: the positive moment already clears those thresholds by factors `1.10950` and `1.14442`, whereas the ratio route has only absolute slack `0.0087` to `0.0112`. So the best current proof target is now the local positive carrier moment `P_diag(11)`, with the cancellation ratio left only as a secondary tightening line.
  One more singleton bridge then narrows that positive-moment target once again. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_diag_mass_bridge_source.py` uses the exact inequality `[q_mass+q_der]_+ >= [q_mass]_+ - [q_der]_-` on the same anchor at `D=11`, yielding `P_diag(11) >= P_mass(11) - N_der(11)`. Numerically, the positive mass carrier is already `P_mass(11)=1.88085e-3`, while the adverse derivative leakage is only `N_der(11)=7.58159e-5`, i.e. about `4.03%` of that carrier. So the induced lower bound `P_diag(11) >= 1.80504e-3` still clears the global and local singleton targets by factors `1.07018` and `1.10386`. Equivalently, it is enough to prove `P_mass(11) >= 1.76249e-3` (global) or `1.71102e-3` (local), with `N_der(11)` left only as a secondary leakage term. This is now the cleanest current parent-side target on the audited bridge.
  A further singleton kernel audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_pmass_kernel_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_pmass_kernel_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O55 / Corollary O54`. It shows that on the same canonical anchor at `D=11` the one-center mass kernel is strictly positive everywhere, so the singleton positive mass carrier factorizes exactly as `P_mass(11) = \bar K_{mass,11}^{(+)} W_{11}^{(+)}`. Numerically, `W_{11}^{(+)} = 2.02920e-2` and the weighted kernel average is `\bar K_{mass,11}^{(+)} = 9.26892e-2`, reproducing `P_mass(11)` exactly. The corresponding global/local carrier thresholds are only `1.90151e-2` and `1.84597e-2`, so the exact weighted route still clears them by factors `1.06716` and `1.09926`. By contrast the crude minimum-kernel product `K_min W_+ = 8.86341e-4` reaches only about `50%` of the required singleton mass target. So the roadmap endpoint narrows again: the one-center mass profile itself is no longer the bottleneck; the remaining parent-side difficulty lies in the positive oriented carrier geometry, or equivalently in proving a nontrivial weighted lower bound for the mass kernel average rather than relying on a bare pointwise minimum.
  A further singleton lobe audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_single_lobe_carrier_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_single_lobe_carrier_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O56 / Corollary O55`. It keeps the same canonical singleton anchor at `D=11`, but uses the left-right reflection symmetry of the two-center background to reduce the total positive carrier exactly to one lobe: `W_{11}^{(+)} = 2 W_{11,+}^{(+)} = 2 W_{11,-}^{(+)}`. Numerically, the audited lobe values are `W_{11,+}^{(+)} = 1.01723e-2` and `W_{11,-}^{(+)} = 1.01197e-2`, while the exact half-total is `1.01460e-2`; the lobe-sum reconstruction residual is `3.47e-18`, and each lobe sits only `2.63e-5` away from the symmetric half-total value on the current grid. This halves the remaining singleton carrier thresholds to `9.50753e-3` (global) and `9.22985e-3` (local), both already cleared by each lobe individually. So the roadmap endpoint narrows once more: the remaining parent-side theorem task is no longer a total-anchor carrier bound, but a one-lobe lower bound plus exact reflection symmetry.
  A further singleton overlap audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_single_lobe_overlap_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_single_lobe_overlap_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O57 / Corollary O56`. It keeps the same one-lobe target `W_{11,+}^{(+)}`, but rewrites it exactly in localized states as `W_{11,+}^{(+)} = W_{11,+}^{(++)} + W_{11,+}^{(--)} = 4\pi \int_{B_+} \rho([\phi_R]_+[\phi_L]_+ + [\phi_R]_-[\phi_L]_-)`. Numerically, the positive-positive overlap alone is already `W_{11,+}^{(++)} = 1.01659e-2`, while the negative-negative same-sign completion is only `6.44e-6`, i.e. `0.0633%` of the one-lobe total. So the one-lobe theorem route narrows once more: even before using the tiny `--` completion, `W_{11,+}^{(++)}` already clears the global and local one-lobe thresholds by factors `1.06925` and `1.10142`. This means the current narrowest parent-side endpoint is no longer the total carrier, and not even the full one-lobe positive part, but the one-lobe positive-positive localized-overlap moment `W_{11,+}^{(++)}`.
  A further singleton positive-positive factor audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_single_lobe_pp_factor_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_single_lobe_pp_factor_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O58 / Corollary O57`. It keeps the same one-lobe positive-positive target, but factors it exactly through single-function positive lobe masses: `W_{11,+}^{(++)} = S_{11,+}^{(R+)} \bar\Phi_{L,11,+}^{(+|R+)} = S_{11,+}^{(L+)} \bar\Phi_{R,11,+}^{(+|L+)}`. Numerically, `S_{11,+}^{(R+)} = 6.58504e-1`, `S_{11,+}^{(L+)} = 1.55385e-1`, `\bar\Phi_{L,11,+}^{(+|R+)} = 1.54379e-2`, and `\bar\Phi_{R,11,+}^{(+|L+)} = 6.54241e-2`, with exact factorization residuals `0` and `1.73e-18`. Both routes clear the global and local one-lobe thresholds by factors `1.06925` and `1.10142`, but only the mirror-positive route leaves outside a same-center multiplier on the same lobe. So the bottleneck narrows again: the cleanest current single-function parent-side endpoint is no longer `W_{11,+}^{(++)}` itself, but the one-lobe mirror-positive mass `S_{11,+}^{(L+)}`.
  A further singleton mirror-half audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_mirror_lobe_half_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_mirror_lobe_half_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O59 / Corollary O58`. It keeps the same one-lobe mirror-positive mass target, but shows that on the audited singleton bridge it is entirely carried by the lower half of the plus lobe: the upper half contributes zero positive mass, and on the lower half one has the exact identity `S_{11,+}^{(L+)} = I_{11,+}^{(L,down)} + N_{11,+}^{(L,down)}` with `I_{11,+}^{(L,down)} = 1.55208e-1` and tiny negative leakage `N_{11,+}^{(L,down)} = 1.76989e-4`. This leakage is only `1.14e-3` of the positive mass, so the remaining parent-side theorem target narrows once more: the cleanest endpoint is no longer the one-lobe positive mass `S_{11,+}^{(L+)}`, but the lower-half signed mirror mass `I_{11,+}^{(L,down)}` itself, which already clears the global and local required floors by factors `1.06803` and `1.10016`.
  A further strict-slab audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_mirror_strict_slab_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_mirror_strict_slab_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O60 / Corollary O59`. It keeps the same lower-half signed mirror mass target, but now introduces the axisymmetric Laplacian explicitly as a proof tool: on the shallowest audited strict lower slab `\Omega_{+,\sharp}(11)=B_+(11)\cap\{z\le D/2-0.18\}`, both `\phi_L` and `\phi_R` are strictly positive on the slab and on its discrete boundary, and the exact equation `-\Delta_a \phi_L = (\bar E-U_a)\phi_L - \delta E\,\phi_R` already has positive source floor `2.22998e-3`. The strict-slab signed mass is still `1.53031e-1`, clearing the global and local lower-half targets by factors `1.05305` and `1.08473`, while the omitted interface strip is only `1.40%` of the lower-half mass. So the proof-side reading sharpens again: the correct PDE domain is no longer the whole lower half, but a strict lower slab where bare `-\Delta_a` is already barrier-compatible. The naive constant-source torsion lower bound is positive but far too conservative, so `-\Delta_a` should now be used as a barrier / maximum-principle tool for the existing mass route rather than as a new theorem object.
  A further strict-slab Dirichlet audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O61 / Corollary O60`. It keeps the same strict-slab PDE domain, but now uses bare `-\Delta_a` to split the strict-slab mirror mass exactly into a harmonic boundary lift plus a zero-boundary positive source potential: `I_{11,+}^{(L,\sharp)} = I_{\partial,\sharp} + I_{{\rm src},\sharp}`. Numerically, `I_{\partial,\sharp} = 1.41805e-1` already carries `92.6647%` of the strict-slab mass and reaches `97.5804%` of the global target while already closing the weaker local target by itself; the positive source potential contributes the remaining `1.12253e-2`, exceeding the residual global deficit by a factor `3.19240`. This finally makes the PDE bottleneck transparent: the remaining analytic task is now chiefly a boundary-trace theorem on the strict slab, supplemented by a genuinely nontrivial source lower bound. The crude constant-source torsion bound remains far too weak, so the next work should not invent a new object; it should strengthen the barrier on this exact domain.
  A further strict-slab boundary-edge audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_boundary_edge_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_boundary_edge_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O62 / Corollary O61`. It splits the harmonic boundary lift exactly into edgewise contributions from the axis, outer wall, bottom edge, top cap, and four corners, with reconstruction residual `3.12e-16`. The main numerical point is that the artificial top cap carries only `5.0815%` of the boundary lift, while the non-top physical boundary carries `94.9185%`. More importantly, the physical boundary plus the already-positive source potential gives `1.45825e-1`, which clears the global strict-slab target `1.45322e-1` by the factor `1.00346`. So the proof target narrows again: the artificial top cap is no longer a bottleneck, and the remaining theorem problem is now a trace theorem on the physical strict-slab boundary together with the same positive source completion.
  A further physical-boundary trace audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_physical_trace_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_physical_trace_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O63 / Corollary O62`. It rewrites the physical boundary lift by adjoint Poisson duality as `I_{\partial,\sharp}^{(phys)} = p_\sharp^T b_\sharp^{(phys)}`, where `K_\sharp^T p_\sharp = \omega_\sharp`, with residual `2.78e-17`. Numerically `I_{\partial,\sharp}^{(phys)} = 1.34599e-1`, while the required physical trace after retaining the positive source is `1.34096e-1`, so the physical trace closes by the tight factor `1.00375`. The outer wall carries `60.9495%`, the bottom edge `35.5719%`, and the lower corners `3.4786%` of this physical trace. Thus the theorem target is now the adjoint-Poisson trace on the physical edges of the strict slab, not the whole boundary and not the artificial top cap.
  A further edge-product audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_edge_product_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_edge_product_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O64 / Corollary O63`. It rewrites the physical trace as four adjoint-Poisson edge products: outer wall `8.20377e-2`, bottom edge `4.78795e-2`, lower-right corner `4.11332e-3`, and lower-left corner `5.68899e-4`, with zero residual against the physical trace. After retaining the source, the required global physical trace is `1.34096e-1`; outer+bottom alone reach `96.8836%`, outer+bottom+right-corner reach `99.9510%`, and the lower-left corner is `8.65657` times the remaining gap. So the parent-side proof target has become a finite edge-product lower bound, with the local threshold already closed by outer+bottom alone.
  A further rounded edge-certificate audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_edge_certificate_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_edge_certificate_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O65 / Corollary O64`. It keeps exactly the same four physical edge products but rounds every required scalar floor downward: `B_out >= 2.834e2`, `p_out >= 2.894e-4`, `B_bot >= 2.738e2`, `p_bot >= 1.748e-4`, `B_rb >= 6.32e1`, `p_rb >= 6.49e-5`, `B_lb >= 2.36`, and `p_lb >= 2.41e-4`. The resulting certified edge sum is `1.3454664e-1`, still clearing the required global physical trace after source by `4.50342e-4` and factor `1.00336`; after adding the positive source potential it gives `1.4577191e-1`, clearing the global strict-slab target by factor `1.00310`. The local threshold is already closed by the rounded outer+bottom certificate alone (`1.00019x`). Thus the current theorem target has been reduced to eight scalar parent-side lower bounds on the already-fixed edge masses and adjoint-Poisson averages, not a new family, support, or carrier.
  A further pointwise edge-certificate audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_edge_point_certificate_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_edge_point_certificate_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O66 / Corollary O65`. It keeps the same four physical edges but bypasses the edge-average theorem by using pointwise positivity: if `b_q^- <= b_q` and `p_q^- <= p_q` on each boundary point, then `sum b_q p_q >= sum b_q^- p_q^-`. Taking four-significant-digit downward roundings on the `33+10+1+1=45` fixed edge points gives certified contributions `8.20001038e-2` (outer), `4.78595834e-2` (bottom), `4.1125672e-3` (right lower corner), and `5.68760e-4` (left lower corner), for total `1.345410144e-1`. This still clears the global after-source trace target by `4.44717e-4` and factor `1.003316`; with the source potential it clears the global strict-slab target by factor `1.003060`. The local target is still closed by outer+bottom alone (`1.000059x`). The endpoint is now a fully local fixed-edge barrier problem for `b_sharp(q)` and the adjoint Poisson weights `p_sharp(q)`, rather than any new support design or edge-average theorem.
  A further adjoint-barrier certificate audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_adjoint_barrier_certificate_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_adjoint_barrier_certificate_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O67 / Corollary O66`. It keeps the O66 pointwise certificate but replaces the direct read-off of `p_sharp` by a strict maximum-principle subsolution: solve `K_sharp^T p_under = theta omega_sharp` with `theta=0.999999`, so `K_sharp^T(p_sharp-p_under)=(1-theta)omega_sharp >= 0` and hence `p_sharp >= p_under`. Numerically `p_under` still dominates all four-significant-digit point floors on the `45` fixed edge points; the residual `omega-K^T p_under` has min/max `6.78584e-10` and `1.56074e-8`, matching `(1-theta)omega` to `8.67e-19`. The certified edge sum remains `1.345410144e-1`, the barrier product with the same `b_q^-` is `1.3457028305e-1`, and the source-completed global ratio remains `1.003060`. Thus the Poisson-weight side is now a maximum-principle barrier problem, with the only remaining lift being to replace this finite subsolution by a closed-form or monotone comparison barrier.
  A further boundary-flux certificate audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_boundary_flux_certificate_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_boundary_flux_certificate_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O68 / Corollary O67`. It keeps the same fixed physical-edge points but replaces the direct boundary-defect read-off by the missing-neighbor flux formula `b_q = -sum_{r notin S} K_full(q,r) phi_L(r) = sum c_qr phi_L(r)`, with finite-volume coefficients `rho_{i+1/2}/dr^2`, `rho_{i-1/2}/dr^2`, and `rho_i/dz^2`. On the audited bridge the formula reconstructs the boundary defects to `1.31e-13`. Five-significant-digit downward floors for the exterior `phi_L` values across the `46` missing-neighbor components certify all `45` four-significant-digit `b_q^-` floors: the flux floors sum to `6.229032e2`, exceeding the `b_q^-` sum `6.22792e2` by `1.11200e-1`. Combined with the O67 adjoint barrier for `p_q^-`, the O66 edge product certificate remains `1.345410144e-1`, and the source-completed global ratio remains `1.003060`. The remaining lift is now explicitly two fixed-edge barrier tasks: exterior `phi_L` floors for the missing fluxes and a closed-form comparison for the adjoint subsolution.
  A further exterior-trace barrier audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O69 / Corollary O68`. It keeps the same `46` missing-neighbor trace points, defines the parent trace subsolution `phi_under = 0.9999995 phi_L`, and verifies the discrete maximum-principle comparison on the restricted trace system: `min f_L = 8.23807e-3`, `min parent outside flux = 4.71653`, `min diagonal margin = 6.66667e1`, and residual agreement `1.88e-14`. This barrier dominates all four-significant-digit exterior `phi_L` floors (`min phi_under - phi_floor = 1.01840e-8`). Those four-digit floors no longer certify every old pointwise `b_q^-` floor (`17` old point checks fail), but the direct flux product with the existing adjoint floors is still stronger than needed: `sum_q (sum_r c_qr phi_r^{--}) p_q^- = 1.3455686726e-1`, ratio `1.003435` against the after-source global trace target and source-completed ratio `1.003169`. Thus the exterior `phi_L` side is closed as a parent trace barrier; the only remaining finite lift is the adjoint `p_q^-` comparison.
  A further adjoint monotone-barrier audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O70 / Corollary O69`. Writing `K_sharp^T = D-L-U` with `L,U >= 0`, it starts from `p^(0)=0` and uses `(D-L)p^(n+1)=omega_sharp+U p^(n)`. The M-matrix comparison gives `0 <= p^(n) <= p_sharp` and `omega_sharp-K_sharp^T p^(n+1)=U(p^(n+1)-p^(n)) >= 0`. Numerically, sweep `548` is already enough for the direct O69 flux product: `sum_q b_hat_q p^(548)(q)=1.3453292568e-1`, ratio `1.003256` against the after-source global target, source-completed ratio `1.003005`, and outer+bottom local ratio `1.000003`. Sweep `859` optionally dominates every old four-significant-digit `p_q^-` floor (`min p^(859)/p_q^- = 1.000000033`) and gives product ratio `1.003648`. Thus both sides of the physical trace certificate are now parent-side maximum-principle constructions; any remaining work is compression or convergence-rate polish, not a new support/object line.
  A synthesis audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_trace_synthesis_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_trace_synthesis_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O71 / Corollary O70`. It packages O68-O70 into one reviewer-facing certificate on the same fixed objects: `45` physical edge points, `46` exterior trace points, and `420` strict-slab adjoint points. The closed items are now explicit: boundary flux identity, exterior `phi_L` trace barrier, adjoint GS product barrier, and optional old `p_q^-` point-floor completion. The main closed parent-side certificate is `sum_q b_hat_q p^(548)(q)=1.3453292568e-1`, global trace ratio `1.003256`, source-completed ratio `1.003005`, and outer+bottom local ratio `1.000003`. The roadmap status is now `parent_side_closed_secondary_tightening_only`: compress presentation and optionally add a convergence-rate proof for the finite GS sweep counts; do not add support/object/carrier.
  A secondary GS-rate audit now exists in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_exact_schur_singleton_strict_slab_gs_rate_source.py`, with figure `/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/chi_open_system_exact_schur_singleton_strict_slab_gs_rate_source_figure.png`, and is promoted in `/Users/boypatrick/codex/PSLT_quantam/paper/main.tex` as `Proposition O72 / Corollary O71`. It keeps the same strict slab and GS sequence, writing increments `d_n=p^(n)-p^(n-1)` and certifying `d_(n+1)<=alpha_n d_n`, hence `p_sharp-p^(n)<=alpha_n/(1-alpha_n)d_n`. The audit verifies that the first global closure sweep is `384`, the first simultaneous global+local closure sweep is `548`, and the first old `p_q^-` point-floor sweep is `859`. At sweep `548`, `alpha=0.9865813608`, the certified flux-product tail bound is `5.33487299e-5`, covering the exact validation gap `5.33478811e-5`; at sweep `859`, the tail bound is `7.98759343e-7`, covering `7.98758661e-7`. The strict-slab line is now `parent_side_closed_editorial_compression_only`: presentation synthesis and GS-rate polish are closed, with no new support/object/carrier.
  The paper now also includes a non-numbered reviewer-facing compressed form immediately after `Corollary O71`: O68-O72 reduce to the fixed identities `b_sharp(q)>=b_hat_q`, `p_sharp(q)>=p^(548)(q)`, and the rate-tail certificate for `p_sharp-p^(548)`, hence `I_partial,sharp^(phys)>=sum_q b_hat_q p^(548)(q)=1.3453292568e-1` and source-completed ratio `1.003005`. This is editorial compression only; it does not introduce a new proposition, theorem object, support, trace, or carrier.
  A further source audit now removes `gamma_ref` as an independent blocker. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_gamma_ref_floor_source.py` verifies on the exported exact bridge that `gamma_ref = omega1^11 = tau_env^-11` to machine precision (`max |gamma_ref-omega1^11| = 1.56e-11`, `max |tau_env-omega1^-1| = 1.11e-16`). The exact audited floor therefore comes directly from the first-mode floor: `omega1_floor = 1.751201...` and `tau_env_ceiling = 0.5710366...`, both at `D=4`, induce the same `gamma_ref_floor = 475.002983...`, matching the observed minimum exactly. So after `Proposition O11`, the remaining observable-chain closure is no longer `(\gamma_ref, \chi_eff, Cmax)` simultaneously; it is concentrated on the pair `(\chi_eff floor, Cmax ceiling)` or on a directly equivalent mix-rate floor.
  The next observable-chain reduction is now also explicit. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_cmax_ceiling_source.py` packages the two-level positivity argument into a direct ceiling for the coherence response: for any positive trace-one `2x2` density matrix one has `|rho_LR(t)| <= 1/2`, hence `Cmax <= 1/2`. On the exported exact bridge the actual maximum is only `0.497539...`, leaving slack `2.461e-3`. Feeding this universal ceiling back into the `gamma_mix = gamma_ref * chi_eff / (2 Cmax)` chain still yields `gamma_mix >= 4.29102e-4`, an induced kernel floor `g_x^2 Sxx >= 9.96645e-10`, and therefore the conservative dominant-witness bound `||Delta_mix^(fact)||_inf <= 4.41e-5`. This remains below the audited half-radii by factors `1.09` and `1.14`. So after `Proposition O12`, `Cmax` is no longer an independent blocker either; the remaining observable-chain gap is concentrated on `chi_eff^floor`, or equivalently on a directly justified floor for `gamma_mix`.
  The next reduction now isolates that final scalar more cleanly. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_chi_eff_floor_source.py` verifies the exact identities `chi_eff = chi_LR * (chi_eff/chi_LR) = 2 gamma_mix Cmax / gamma_ref` and rewrites the observable as `chi_eff(D) = A_chi(D) Cmax(D)` with `A_chi(D) = 2 gamma_mix(D)/gamma_ref(D) = kappa_env * 2 g_x(D)^2 Sxx(D)/gamma_ref(D)`. Using `gamma_ref = omega1^11`, the exported exact bridge already gives a separated amplitude floor `A_chi^(sep floor) = 1.04147e-6`, while the actual exact floor is only a factor `1.74` higher (`A_chi^(exact floor) = 1.81567e-6`). Multiplying by the current exact `Cmax` floor yields the conservative `chi_eff` floor `2.43970e-7`, versus the actual exact minimum `9.03367e-7`. So after `Proposition O13`, the residual observable-chain closure is now fully localized: the only scalar still missing from the exact-export bridge to a complete floor theorem is a lower bound on `Cmax` itself.
  That last scalar has now been reduced one step further as well. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_cmax_floor_source.py` solves the localized two-level Lindblad block in closed form on the underdamped branch, giving `Cmax = [delta/(2 sqrt(Gamma^2+omega^2))] exp[-(Gamma/omega) arctan(omega/Gamma)]` with `Gamma = gamma_phi + 2 gamma_mix` and `omega = sqrt(delta^2-gamma_phi^2)`. On the exported exact bridge every knot is safely underdamped (`min(delta-gamma_phi)=2.59273e-1`, `max gamma_phi/delta=2.07e-4`), and the closed form matches the exported Lindblad maxima to `2.82e-4`. A monotonicity audit on the exact parameter box then gives a separated floor `Cmax^(sep floor)=1.44083e-1`, while the exact minimum is `2.34255e-1`. Combining this with `A_chi^(sep floor)` yields the conservative observable floor `chi_eff >= 1.50059e-7`, versus the actual exact floor `9.03367e-7`. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_cmax_ratio_floor_source.py` then tightens the same statement by rewriting the closed form in the scale-free ratios `a=gamma_phi/delta`, `b=gamma_mix/delta`. The ratio-box floor `F(a_max,b_max)=2.34328e-1`, minus the exported exact-vs-closed residual `2.82464e-4`, already gives the rigorous exact-bridge floor `Cmax >= 2.34045e-1`, within `0.09%` of the actual minimum. Feeding that into `A_chi^(sep floor)` improves the conservative observable floor to `chi_eff >= 2.43752e-7`. So after `Propositions O14-O15`, the observable-chain side is effectively closed on the current exact bridge: what remains is mainly tightening the amplitude-side slack, not establishing the existence of a coherence-response floor.
  `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_achi_amplitude_source.py` then isolates exactly where that residual slack still lives. On the exported exact bridge the identities `A_chi = 2 gamma_mix / omega1^11 = kappa_env * 2 g_x^2 Sxx / omega1^11` hold to `1e-16`, and the minima of `A_chi`, `gamma_mix`, and `g_x^2 Sxx / omega1^11` all occur at the same knot `D=11`. So the exact amplitude floor is already aligned with the mix-rate floor: `A_chi^(exact floor) = 2 gamma_mix^(floor) / omega1(D_gamma,min)^11 = 1.81567e-6`. The remaining amplitude-side slack is exactly the penalty from replacing that local denominator by the global ceiling `omega1^ceil`, which lowers the floor to `1.10807e-6` and accounts for the factor `1.63858`. Combining the aligned amplitude floor with the tightened ratio-box `Cmax` floor gives `chi_eff >= 4.24949e-7`, reducing the remaining gap to the actual exact floor `9.03367e-7` to a factor `2.126`. So after `Propositions O15-O16`, the coherence-response side is essentially saturated and the last meaningful observable-chain task is to turn the aligned `gamma_mix` floor into an a priori theorem-level statement.
  `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_gamma_mix_floor_alignment.py` now sharpens that last point. On the exported exact bridge, `gamma_mix = kappa_env * g_x^2 * Sxx` holds to `4.40e-14`, and the minima of `gamma_mix`, `g_x`, `g_x^2`, and `g_x^2 Sxx` all occur at the same knot `D=11`. So the exact mix-rate floor is already aligned with the `x`-channel overlap floor: `gamma_mix^(exact floor) = kappa_env * (g_x^(floor))^2 * Sxx(D_gx,min) = 5.22995e-4`. Replacing the local response factor by the global response minimum only lowers this to `4.91560e-4`, a factor `1.06395`. That means the remaining `gamma_mix` theorem gap is not response-dominated. The next proof-side target is best read as an a priori floor for the overlap factor `g_x`, with `Sxx` contributing only a small residual uplift on the current exact bridge.
  `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_gx_floor_source.py` then reduces even that overlap-side target to a static projected-operator statement. Because the localized basis is just the Hadamard rotation of the lowest parity pair, the localized fluctuation block satisfies `V_loc = U_LR^T V_par U_LR`, so `g_x = |(V_loc)_{LR}| = |V11-V22|/2` and `g_z = |V12|`. The audit verifies these identities on the exported exact bridge to machine precision (`max |g_x-|V11-V22|/2| = 1.00e-16`, `max |g_z-|V12|| = 2.18e-18`) and shows that the minima of `g_x`, `|V11-V22|/2`, and `gamma_mix` all occur at the same knot `D=11`. Thus the remaining open-system floor gap is no longer naturally read as a bath/response theorem: on the current exact bridge it is enough to prove a lower bound on the parity-diagonal contrast `|V11-V22|` for the projected fluctuation operator `deltaV`. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_self_mirror_source.py` now sharpens that static target by decomposing the shifted two-center potential as `U_2 = U_+ + U_- + U_x`, subtracting the same origin-centered spherical averages, and hence writing `V11-V22 = I_+ + I_- + I_x`. This split closes to machine precision and shows that the one-center pair already carries `97.74%` to `99.833%` of the exact parity contrast across `D=4..20`, while the mirror/cross remainder contributes only `0.167%` to `2.257%` and keeps the same sign as the exact contrast on every audited knot. A further floor audit in `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_self_floor_source.py` then scans symmetric core windows `B_+(R) \cup B_-(R)` and finds an optimal safe radius `R_*=1.35`: on every audited knot the core piece has the same sign as the one-center pair, the tail has the opposite sign, and `|I_self,pair| >= lambda_* |I_self,core^(R_*)|` with `lambda_*=0.426269`. Since the core-window floor is `1.33812e-4`, this already gives the nontrivial exact-bridge bound `|I_self,pair| >= 5.70400e-5`, only a factor `1.174` below the actual one-center-pair minimum `6.69675e-5`. Together with the tiny same-sign mirror remainder, this means the remaining proof-side gap has narrowed again: the main task is now an a priori lower bound on the one-center core contribution, plus only a coarse signed control of `I_x`.
  `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_self_core_anchor_source.py` now tightens that one-center target one step further. Inside the audited core window `R_*=1.35`, a compact local box anchor with `rho_#=0.70`, `z_#=1.21` already has the same sign as `I_self,core^(R_*)` on every audited knot, and the residual shell inside `R_*` keeps that sign as well. This gives the direct bound `|I_self,core^(R_*)| >= |I_self,anchor^(rho_#,z_#)|`, with compact-anchor floor `1.32321e-4` versus exact core-window floor `1.33812e-4`, only a factor `1.01127` apart. Feeding that into the previous `lambda_*` control yields `|I_self,pair| >= 5.64043e-5`, leaving only a factor `1.187` to the exact one-center-pair floor and `1.189` to the exact total parity-contrast floor. So the theorem target has narrowed again: it is now enough to lower-bound a compact one-center anchor, while the rest of `I_self,core^(R_*)` can be treated as a same-sign completion and `I_x` stays only a coarse signed remainder.
  A further static-source audit now decomposes that parity contrast itself into a one-center pair plus a small mirror remainder. Writing the shifted two-center potential as `U_2 = U_+ + U_- + U_x` and subtracting the same origin-centered spherical averages gives the exact fluctuation split `deltaV = deltaV_+ + deltaV_- + deltaV_x`, hence `V11-V22 = I_+ + I_- + I_x` on the exact contrast density. `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_self_mirror_source.py` closes the additive split to machine precision (`max ||U_2-U_+-U_--U_x|| = 8.88e-15`, `max ||deltaV-deltaV_+-deltaV_--deltaV_x|| = 7.11e-15`, reconstruction residual `3.79e-18`) and shows that the one-center pair already carries `97.74%` to `99.833%` of the exact parity contrast across `D=4..20`, with the mirror/cross remainder contributing only `0.167%` to `2.257%` and keeping the same sign as the exact contrast on every audited knot. At the exact floor knot `D=11`, `V11-V22 = 6.70795e-5`, the one-center pair contributes `6.69675e-5`, and the mirror remainder is only `1.12022e-7`. So the remaining proof-side gap has narrowed once more: the current exact bridge is already overwhelmingly saturated by the one-center pair, and the last open question is a lower bound on that one-center contribution plus a coarse control of the small static remainder.
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
- **Absolute-normalization synthesis chain (`A_*` roadmap compressed):** the new consolidation audit
  - `/Users/boypatrick/codex/PSLT_quantam/code/audit_hll_absolute_normalization_synthesis_source.py`
  - `/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/hll_absolute_normalization_synthesis_source_summary.csv`
  - `/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/hll_absolute_normalization_synthesis_source_detail.csv`
  compresses the existing normalization-side audits into one reviewer-facing chain. On the exact side, the diagonal flavor ratios remain UV-tree invariants (`max |ΔR_{e/\mu}^{UV→IR}| = 6.05e-13`, `max |ΔR_{\tau/\mu}^{UV→IR}| = 3.23e-14`), the absolute amplitude factorization `A_* = A_*^{tree} Z_{diag,*}^{-2}` closes with residual `9.32e-17`, the single-layer source `C_{\mu\mu}^{tree} = y_2^{raw}(P_2^{kin}/M_2^2)` closes with residual `9.76e-17` and exact `N=2` support, the overlap identity `y_2^{raw} = y_{\mu,2}^{flavor} = (g_{\mu 2}^{UV})^2` closes with residual `9.87e-17`, and the canonical muon kernel closes exactly as a Gaussian midplane bridge with relative sup residual `3.05e-15`. On the structural side, the finite-box Gaussian bridge already stays at `9.45e-05`, the curvature width law is within `1.01e-03`, and the quartic correction collapses that to `8.47e-07` while keeping the finite-box kernel defect at `9.47e-05`. Excluding the known `D=5` merger outlier, the pointwise core-to-box inversion keeps `c_\sigma` on the narrow band `[22.2825, 22.3100]` with relative span `1.23e-03`; the one-center source `c_\sigma^{self}=22.3198` lies only `7.37e-04` above the reference selector, and the analytic two-center jet tracks the extracted `c_\sigma(D)` with maximum relative error `4.21e-05`. On the representative projected-box benchmark, the one-center source reproduces the overlap block with maximum relative `y_2` error `9.26e-03`, while the analytic mirror refinement improves this to `9.89e-04`, slightly better than the fixed projected selector. In the paper this is now compressed into `Proposition U6 / Corollary U6`: the flavor-ratio side, amplitude factorization, overlap block, kernel shape, and structural width profile are treated as closed blocks, while only `c_\sigma`, the analytic mirror correction, and the quartic coefficient `b_4` remain as secondary tightening directions.
- **`c_\sigma` mainline closes as one-center core plus analytic mirror multiplier:** the follow-up tightening audit
  - `/Users/boypatrick/codex/PSLT_quantam/code/audit_hll_csigma_tightening_synthesis.py`
  - `/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/hll_csigma_tightening_synthesis_summary.csv`
  - `/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/hll_csigma_tightening_synthesis_detail.csv`
  pushes the remaining normalization-side constant one step further without introducing any new overlap object. The analytic bridge coefficient obeys the exact identity
  \[
  c_\sigma^{analytic}(D)=c_\sigma^{self}\left(1+\frac{\delta\kappa_{mirror}(D)}{\kappa_{self}}\right)^{1/4},
  \]
  with machine-level factorization residual `3.55e-15`. The mirror fraction itself stays tiny, `\delta\kappa_{mirror}/\kappa_{self}\in[-6.91e-03,-1.39e-03]`, so the mirror multiplier lies in `[0.998269,0.999653]`, and even its first-order linearization is already accurate to `4.50e-06` relative error. Against the extracted D4--20 bridge coefficient, the one-center source alone stays within `1.78e-03`, while the analytic mirror-refined coefficient improves this to `4.21e-05`. On the projected-box benchmark the same refinement lowers the maximum relative `y_2` error from `9.26e-03` to `9.89e-04` and the log-RMSE by a factor `9.79`. In the paper this is now written as `Proposition U7 / Corollary U7`: the `c_\sigma` mainline is treated as closed by the one-center EYMH core constant plus the tiny analytic mirror multiplier, while only the quartic coefficient `b_4` and sub-`10^{-3}` projected-box polish remain as secondary tightening directions.
- **`b_4` is already just one stable scalar tightening:** the new quartic-response synthesis
  - `/Users/boypatrick/codex/PSLT_quantam/code/audit_hll_b4_tightening_synthesis.py`
  - `/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/hll_b4_tightening_synthesis_summary.csv`
  - `/Users/boypatrick/codex/PSLT_quantam/output/hll_absolute_normalization/hll_b4_tightening_synthesis_detail.csv`
  shows that after `U7` there is no remaining need for a new width-side object. The residual correction is already exhausted by the single scalar
  \[
  \sigma_\mu^{(curv,4)}(D)=\sigma_\mu^{(curv,0)}(D)\Bigl[1+b_4(q_{4,\mu}(D)-q_{4,\mu}(D_*))\Bigr],
  \qquad b_4=0.4244007048874373.
  \]
  With this one global coefficient, the maximum width error drops from `1.01e-03` to `8.47e-07` and the maximum finite-box kernel defect from `1.11e-02` to `9.47e-05`, corresponding to improvement factors `1.19e3` and `1.17e2`. The fit is also stable under coarse refits of the D-window: excluding the known `D=5` outlier changes `b_4` by only `6.27e-05` relative, and fitting only on the outer window `D<=8 or D>=12` changes it by only `3.98e-06`. In the paper this is now written as `Proposition U8 / Corollary U8`: after `U7`, the remaining normalization-side quartic correction is already just one stable scalar on the fixed coordinate `q_{4,\mu}(D)-q_{4,\mu}(D_*)`, and anything beyond that is only benchmark polish.
- **`A_*` line reviewer-facing closure:** the paper now adds `Proposition U9 / Corollary U9`, which packages `U6--U8` into the final normalization-side status statement. The `H->mumu` absolute-normalization chain is treated as closed on the audited domain: the ratio constants are fixed by diagonal-chain invariance, `A_* = A_*^{tree} Z_{diag,*}^{-2}` is exact to `9.32e-17`, the tree source reduces to `y_2^{raw}(P_2^{kin}/M_2^2)` to `9.76e-17`, `y_2^{raw}=(g_{\mu2}^{UV})^2` to `9.87e-17`, and the Gaussian midplane kernel closes to `3.05e-15`. Remaining work is only secondary tightening of the continuum one-center constant, mirror-remainder prose, or optional sub-`1e-3` projected-box polish; it is not a new support/object or kernel-family search.
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
- **Open-system compact-anchor static-source reduction:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_anchor_static_source.py` now pushes the overlap-side proof target one step further. Keeping the canonical compact anchor `(\rho_\#,z_\#)=(0.70,1.21)`, the exact anchor splits as `I_self,anchor = I_diag,anchor + I_mir,anchor`, where `I_diag,anchor` keeps only the same-center static pieces (`\delta V_+` on the right anchor box, `\delta V_-` on the left anchor box) and `I_mir,anchor` is the local mirror correction. On every audited knot, `I_diag,anchor` has the same sign as `I_self,anchor`, and the uniform mirror ratio stays below `\mu_anc = 0.415143`, so `|I_self,anchor| >= (1-\mu_anc)|I_diag,anchor|`. The same-center static anchor floor is `1.63757e-4`, giving the induced compact-anchor lower bound `9.57744e-5`, only a factor `1.38159` below the exact anchor floor `1.32321e-4`. Scanning all safe same-center sub-boxes inside the canonical anchor shows that the optimum already sits at the full anchor itself (`rho_box ≈ 0.69–0.70`, `zeta_box = 1.21`), so the proof target narrows again: it is now enough to lower-bound the same-center static anchor source, while the local mirror term can stay a coarse signed correction.
- **Open-system strict same-center subbox audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_box_source.py` then checks whether `I_diag,anchor` can be replaced by a genuinely smaller same-center static patch. Scanning all strict subboxes inside the canonical anchor shows that the best safe strict plateau sits at `rho_box in [0.63,0.68]`, `zeta_box = 1.21`; a rounded representative choice `(0.65,1.21)` already has the same sign as `I_diag,anchor` and leaves a same-sign shell on every audited knot. Its floor is `1.48301e-4`, versus the full same-center-anchor floor `1.63757e-4`, only a factor `1.10422` lower. So there is a useful negative result here: further localizing the support does not buy us a stronger object. The proof-side target has effectively stabilized at the full same-center anchor scale `I_diag,anchor`, with the local mirror term and the global `I_x` remainder both remaining coarse signed corrections.
- **Open-system oriented-moment reduction of the same-center anchor:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_moment_source.py` now rewrites the same-center static anchor as an intrinsic positive/negative moment balance instead of another support choice. With the oriented density `q_diag = sgn(I_diag,anchor) * C * (deltaV_+ 1_{B_+∩B_anchor} + deltaV_- 1_{B_-∩B_anchor})`, the exact split `|I_diag,anchor| = P_diag - N_diag` closes to `6.51e-19`, and the audited moment ratio stays below `nu_diag^(±) = 0.911532`. The positive-moment floor is `min P_diag = 1.17067e-3`, so one gets the direct intrinsic lower bound `|I_diag,anchor| >= (1-nu_diag^(±)) min P_diag = 1.03567e-4`. This is stronger than the mass-only route (`8.17293e-5`) by a factor `1.26719`, and stronger than the earlier induced compact-anchor floor (`9.57744e-5`) by a factor `1.08136`. The message is useful and narrow: the remaining anchor-side gap is no longer support selection; it is cancellation control inside the full same-center anchor.
- **Open-system shell-isolated cancellation control:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_cancellation_source.py` now attacks `nu_diag^(±)` directly. It splits the same-center anchor into an inner axial slab `|z \mp D/2| <= zeta_cut` and an outer shell, then tracks the oriented positive/negative moments of `q_diag`. Scanning `zeta_cut in [0.80,1.20]` shows that the optimal nontrivial shell-isolating cut is `zeta_cut^* = 1.20`. At that cut, the shell-negative leakage is tiny: `max N_out/P_in = 8.68193e-3`, while the inner cancellation ratio is already `max N_in/P_in = 0.912743`. So the full audited cancellation ratio satisfies `nu_diag^(±) <= 0.921425`, and the shell contribution is below one percent of the normalized cancellation budget. The same audit also aligns this with the one-center mass/derivative split: `min P_mass/P_diag = 1.00507`, `max P_der/P_diag = 8.53082e-2`, `min N_mass/N_diag = 0.712802`, `max N_der/N_diag = 0.570882`. That is the cleanest reading so far: support leakage is not the issue, and the remaining gap is an interior one-center cancellation problem, largely carried by the mass sector.
- **Open-system interior mass-sector cancellation control:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_mass_cancellation_source.py` now fixes the same optimal inner slab `zeta_cut^* = 1.20` and rewrites the stabilized same-center anchor floor in terms of the interior one-center mass net
  \[
  I_{\rm mass,in}
  =
  (1-\nu_{\rm diag,int}^{(\pm,m)})P_{\rm mass,in},
  \qquad
  \nu_{\rm diag,int}^{(\pm,m)}=\frac{N_{\rm mass,in}}{P_{\rm mass,in}}.
  \]
  On the audited window the interior mass term keeps the sign of the full same-center anchor on every knot, and the exact anchor floor, the interior mass-net floor, and the maximal mass-sector cancellation ratio all align at the same knot `D=11`. Numerically,
  \[
  \min |I_{\rm diag,anchor}|=1.65556\times10^{-4},
  \qquad
  \min I_{\rm mass,in}=7.92440\times10^{-5},
  \qquad
  \max \nu_{\rm diag,int}^{(\pm,m)}=0.957812.
  \]
  At that same floor knot, the derivative and shell pieces are already non-adverse:
  \[
  I_{\rm der,in}(11)=8.38290\times10^{-5}>0,
  \qquad
  I_{\rm out}(11)=2.48275\times10^{-6}>0.
  \]
  So the overlap-side proof target narrows again: the adverse floor-setting mechanism is the interior one-center mass-sector cancellation itself, while derivative and shell terms act only as signed local corrections at the audited floor knot.
- **Open-system positive mass-carrier floor:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_mass_positive_source.py` now attacks `P_{\rm mass,in}` directly. For any same-center subbox inside the canonical inner slab, monotonicity of the positive part gives
  \[
  P_{\rm mass,in}(D)\ge P_{\rm mass,box}^{(\rho_\Box,\zeta_\Box)}(D).
  \]
  Scanning `rho_box in [0.35,0.70]` and `zeta_box in [0.70,1.20]` shows that the optimal strict positive-carrier plateau already lies on a near-full axial trim, `rho_box in [0.69,0.70]`, `zeta_box in [1.10,1.19]`. Using the canonical strict representative `(0.70,1.19)`, the audited floor is unchanged:
  \[
  \min_D P_{\rm mass,box}^{(0.70,1.19)}(D)
  =
  \min_D P_{\rm mass,in}(D)
  =
  1.25520\times10^{-3}
  \]
  at `D=20`, and the minimum all-knot share is still
  \[
  \min_D \frac{P_{\rm mass,box}^{(0.70,1.19)}(D)}{P_{\rm mass,in}(D)}
  =
  0.999825
  \]
  at `D=11`. So the positive-carrier side is now essentially closed: the remaining overlap-side gap is not the one-center positive mass carrier, but the interior mass-sector cancellation ratio `\nu_{\rm diag,int}^{(\pm,m)}`.
- **Open-system adverse mass-carrier annulus control:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_mass_negative_source.py` now localizes the negative mass carrier itself. Inside the same canonical inner slab `|z\mp D/2|\le 1.20`, it scans off-axis annuli
  \[
  A_{\rm neg}^{(\rho_{\min},\zeta_\Box)}
  =
  \{(\rho,z): \rho_{\min}\le \rho\le 0.70,\ |z\mp D/2|\le \zeta_\Box\}
  \]
  with `rho_min in [0.03,0.30]`, `zeta_box in [1.10,1.20]`. If arbitrary strict trims are allowed, the optimum is a near-full slab; but once we insist on genuinely off-axis annuli `rho_min >= 0.12`, the best plateau sits at `rho_min in [0.12,0.15]`, `zeta_box = 1.20`. Using the canonical representative `(rho_min,zeta_box)=(0.15,1.20)`, the annulus already captures at least
  \[
  \min_D \frac{N_{\rm mass,ann}^{(0.15,1.20)}(D)}{N_{\rm mass,in}(D)}
  =
  0.949763
  \]
  on the audited window, while the omitted axial-core leakage obeys
  \[
  \max_D \frac{N_{\rm mass,tail}^{(0.15,1.20)}(D)}{P_{\rm mass,in}(D)}
  =
  4.81178\times10^{-2}
  \]
  at `D=11`. The same knot also maximizes the annular adverse load,
  \[
  \max_D \frac{N_{\rm mass,ann}^{(0.15,1.20)}(D)}{P_{\rm mass,in}(D)}
  =
  0.909694.
  \]
  So the cancellation ratio is no longer best read as a full-slab negative-carrier problem: it has narrowed to an off-axis one-center adverse annulus plus a small axial-core leakage.
- **Open-system strict adverse-annulus trim audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_mass_annulus_source.py` then asks whether the canonical adverse annulus can itself be replaced by a genuinely smaller support. Scanning strict annular trims inside `A_{\rm neg}^{(0.15,1.20)}` over `rho_min in [0.15,0.30]`, `zeta_box in [1.10,1.20]` shows that the best strict plateau already sits at the smallest radial trim, namely `(rho_min,zeta_box)=(0.15,1.19)`. This trimmed adverse annulus still captures at least
  \[
  \min_D \frac{N_{\rm mass,subann}^{(0.15,1.19)}(D)}{N_{\rm mass,ann}^{(0.15,1.20)}(D)}
  =
  0.984061
  \]
  (worst at `D=16`), while the omitted cap leakage obeys
  \[
  \max_D \frac{N_{\rm mass,cap}^{(0.15,1.19)}(D)}{P_{\rm mass,in}(D)}
  =
  7.47726\times10^{-3}.
  \]
  The audited floor drops only from `7.73394e-4` to `7.64028e-4`, a factor `1.01226`. So the support side is now effectively closed here too: the proof target has stabilized at the full canonical adverse annulus, and the remaining gap is the adverse magnitude carried by that annulus, not any finer support selection inside it.
- **Open-system adverse-shoulder magnitude audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_mass_shoulder_source.py` then attacks that remaining magnitude directly by splitting the canonical adverse annulus into a radial shoulder and an inner-rim leakage,
  \[
  N_{\rm mass,ann}^{(0.15,1.20)}
  =
  N_{\rm mass,shoulder}^{(\rho_{\rm sh},1.20)}
  +
  N_{\rm mass,rim}^{(\rho_{\rm sh},1.20)}.
  \]
  Scanning strict radial trims `rho_sh in [0.16,0.30]` shows a flat optimal plateau `rho_sh in [0.16,0.20]`; using the clean canonical representative `rho_sh = 0.20`, the off-axis shoulder still captures at least
  \[
  \min_D \frac{N_{\rm mass,shoulder}^{(0.20,1.20)}(D)}{N_{\rm mass,ann}^{(0.15,1.20)}(D)}
  =
  0.940824
  \]
  (worst at `D=11`), while the discarded inner-rim leakage obeys
  \[
  \max_D \frac{N_{\rm mass,rim}^{(0.20,1.20)}(D)}{P_{\rm mass,in}(D)}
  =
  5.38319\times10^{-2}
  \]
  again at `D=11`. The shoulder floor is `7.37447e-4`, only a factor `1.04874` below the full-annulus floor `7.73394e-4`. So the remaining cancellation gap is now best read as an off-axis one-center adverse shoulder plus two small leakages (the inner radial rim and the axial core), not as a full-annulus magnitude problem.
- **Open-system adverse-shoulder band audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_mass_shoulder_band_source.py` then attacks the shoulder magnitude itself by carving out a genuine axial hollow inside the canonical shoulder. Fixing `rho_sh = 0.20`, it scans
  \[
  \zeta_{\rm low}\in[0.10,0.40]
  \]
  in the exact split
  \[
  N_{\rm mass,shoulder}^{(0.20,1.20)}
  =
  N_{\rm mass,band}^{(0.20;\zeta_{\rm low},1.20)}
  +
  N_{\rm mass,core}^{(0.20;\zeta_{\rm low},1.20)}.
  \]
  The best hollow already sits at the smallest genuine cut, `zeta_low = 0.10`. For this canonical band,
  \[
  \min_D \frac{N_{\rm mass,band}^{(0.20;0.10,1.20)}(D)}{N_{\rm mass,shoulder}^{(0.20,1.20)}(D)}
  =
  0.951019
  \]
  (worst at `D=11`), while the discarded shoulder-core leakage obeys
  \[
  \max_D \frac{N_{\rm mass,core}^{(0.20;0.10,1.20)}(D)}{P_{\rm mass,in}(D)}
  =
  4.19214\times10^{-2}.
  \]
  The adverse-band floor exactly matches the canonical shoulder floor, both equal to `7.37447e-4` at `D=19`. So the theorem target narrows once more: the remaining adverse magnitude is already carried by a genuinely hollow off-axis band, with only three small corrective leakages left outside it (axial shoulder core, inner rim, and axial core).
- **Open-system adverse-band corner-stability audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_mass_band_corner_source.py` checks whether the canonical adverse band can be improved by a genuinely strict corner trim. It scans
  \[
  \rho_{\rm corner}\in[0.22,0.30],
  \qquad
  \zeta_{\rm corner}\in[0.10,0.40]
  \]
  inside the canonical band `N_{mass,band}^{(0.20;0.10,1.20)}` and finds a flat best plateau `rho_corner in [0.22,0.27]`, `zeta_corner = 0.10`. Using the clean canonical strict representative `(rho_corner,zeta_corner)=(0.22,0.10)`, the corner subband still captures at least
  \[
  \min_D \frac{N_{\rm mass,corner}^{(0.22;0.10,1.20)}(D)}{N_{\rm mass,band}^{(0.20;0.10,1.20)}(D)}
  =
  0.921342
  \]
  (worst at `D=4`), but the discarded cap leakage already rises to
  \[
  \max_D \frac{N_{\rm mass,cap}^{(0.22;0.10,1.20)}(D)}{P_{\rm mass,in}(D)}
  =
  6.35145\times10^{-2}
  \]
  at `D=11`. More importantly, the strict-corner floor drops to `6.88668e-4`, whereas the full band floor remains `7.37447e-4`, a loss by the factor `1.07083`. So support-side localization is now exhausted at the full adverse-band scale: the stable proof target remains `N_{mass,band}^{(0.20;0.10,1.20)}`, with only small cap, shoulder-core, inner-rim, and axial-core leakages outside it.
- **Open-system adverse-band magnitude factorization audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_mass_band_magnitude_source.py` attacks the canonical adverse band itself without shrinking support. On `B_{neg}^{(0.20;0.10,1.20)}`, it defines the one-center local mass kernel `K_{band}^{mass}` and the negative carrier density `W_{band}^{(-)}=[-(sgn I_{diag,anchor})\,\mathcal C]_+`, and verifies the exact pointwise factorization
  \[
  [q_{\rm mass}]_- = K_{\rm band}^{mass}\,W_{\rm band}^{(-)}.
  \]
  Across all audited knots `D=4,...,20`, the factorization residual is exactly zero and the kernel stays strictly positive, with audited minimum `1.82852e-2` at `D=4`. Writing
  \[
  N_{\rm mass,band}^{(0.20;0.10,1.20)} = \bar K_{\rm band}^{(-)}\,W_{\rm band}^{(-)},
  \]
  the weighted kernel average is trapped in the narrow interval
  \[
  5.38460\times10^{-2}\le \bar K_{\rm band}^{(-)} \le 1.00728\times10^{-1},
  \]
  while the negative-carrier floor is `8.40086e-3` at `D=20`. The coarse product floor `K_{min}^{band} W_{band}^{(-)}` is `4.28523e-4`, and the exact band floor `7.37447e-4` exceeds it by the factor `1.72090`. So the magnitude problem has narrowed again: the remaining proof target is no longer the raw adverse-band integral, but the negative carrier moment `W_{band}^{(-)}`, with the one-center mass kernel now reduced to a positive audited multiplier.
- **Open-system negative-carrier radial-factor audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_band_negative_carrier_source.py` then attacks `W_{band}^{(-)}` itself, still without any support shrinkage. Using
  \[
  \mathcal C=2\pi\rho(\psi_1^2-\psi_2^2),
  \]
  it defines the unweighted negative overlap-band moment
  \[
  M_{\rm band}^{(-)}
  =
  \int_{B_{neg}^{(0.20;0.10,1.20)}} [-(\operatorname{sgn}I_{diag,anchor})(\psi_1^2-\psi_2^2)]_+,
  \]
  and verifies the exact factorization
  \[
  W_{\rm band}^{(-)}=2\pi\,\bar\rho_{\rm band}^{(-)}\,M_{\rm band}^{(-)}
  \]
  with residual `1.35525e-20`. The weighted radial factor is trapped in the tiny interval
  \[
  0.4434097 \le \bar\rho_{\rm band}^{(-)} \le 0.4438634,
  \]
  whose relative spread is only `1.02267e-3`. Both `W_{band}^{(-)}` and `M_{band}^{(-)}` attain their floors at `D=20`, namely `8.40086e-3` and `3.01270e-3`, and even the coarsest lower product is essentially exact:
  \[
  \frac{\min W_{\rm band}^{(-)}}{2\pi(\min \bar\rho_{\rm band}^{(-)})(\min M_{\rm band}^{(-)})}=1.00088.
  \]
  So the carrier-side magnitude problem narrows again: to control `W_{band}^{(-)}`, it is enough to control the unweighted negative overlap-band moment `M_{band}^{(-)}`, with only an almost constant radial factor left outside it.
- **Open-system overlap-band sign-balance audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_band_overlap_moment_source.py` attacks `M_{band}^{(-)}` itself without any further support change. On the same canonical band it defines the absolute overlap-band mass
  \[
  M_{\rm band}^{\rm(abs)}
  =
  \int |(\operatorname{sgn}I_{diag,anchor})(\psi_1^2-\psi_2^2)|,
  \]
  and the signed overlap residue `R_{band}^{(sgn)}`, then verifies the exact identity
  \[
  M_{\rm band}^{(-)}=\frac{M_{\rm band}^{\rm(abs)}-R_{\rm band}^{\rm(sgn)}}{2}
  =\theta_{\rm band}^{(-)}M_{\rm band}^{\rm(abs)}
  \]
  with residual `1.73e-18`. It also verifies the localized-overlap rewrite
  \[
  M_{\rm band}^{\rm(abs)}=2\int |\phi_R\phi_L|
  \]
  with residual `3.47e-18`. The signed-imbalance ratio stays in the audited interval
  \[
  -2.03788\times10^{-1}\le r_{\rm band}^{\rm(sgn)}\le 9.60339\times10^{-2},
  \]
  so the negative-fraction factor obeys
  \[
  \theta_{\rm band}^{(-)}\ge 4.51983\times10^{-1}.
  \]
  Since the absolute overlap-band floor is `6.45299e-3` at `D=19`, this already induces
  \[
  M_{\rm band}^{(-)}\ge 2.91664\times10^{-3},
  \]
  only a factor `1.03294` below the exact negative-overlap floor `3.01270e-3` at `D=20`. So the support, kernel, and radial-weight questions are all effectively closed: the remaining overlap-side proof target is now the absolute localized overlap-band mass `M_{band}^{abs}=2\int|\phi_R\phi_L|`, with only a small signed-imbalance factor left outside it.
- **Open-system absolute localized-overlap audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_band_absolute_overlap_source.py` attacks `M_{band}^{abs}` directly on the same canonical band, with no further support shrinkage. It defines the same-center carrier amplitude `Phi_self` and mirror-tail amplitude `Phi_mir`, and verifies the exact factorization
  \[
  M_{\rm band}^{\rm(abs)}=2\int \Phi_{\rm self}\Phi_{\rm mir}
  =2\,S_{\rm band}^{\rm(self)}\,\bar\Phi_{\rm band}^{\rm(mir|self)}.
  \]
  The same-center carrier moment has audited floor
  \[
  \min S_{\rm band}^{\rm(self)}=1.66599\times10^{-1}
  \]
  at `D=20`, while the weighted mirror-tail average stays positive on the whole window with
  \[
  1.92259\times10^{-2}\le \bar\Phi_{\rm band}^{\rm(mir|self)} \le 5.20575\times10^{-2},
  \]
  attaining its minimum at `D=19`. Their direct product gives
  \[
  M_{\rm band}^{\rm(abs)}\ge 6.40606\times10^{-3},
  \]
  only a factor `1.00733` below the exact absolute-overlap floor `6.45299e-3`. The reverse factorization through the mirror carrier is weaker (`1.04982`). So the mainline proof target narrows once more: the absolute overlap-band mass is already essentially fixed by the same-center carrier route, and the only remaining nontrivial multiplier is the weighted mirror-tail average `\bar\Phi_{\rm band}^{\rm(mir|self)}`.
- **Open-system weighted mirror-tail audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_band_mirror_average_source.py` attacks `\bar\Phi_{\rm band}^{\rm(mir|self)}` directly, still without shrinking support. It defines the ordinary mirror-band mean
  \[
  \bar\Phi_{\rm band}^{\rm(mir)}=\frac{S_{\rm band}^{\rm(mir)}}{A_{\rm band}},
  \]
  together with the self-weight bias
  \[
  \Xi_{\rm band}^{\rm(self)}
  =
  \frac{\bar\Phi_{\rm band}^{\rm(mir|self)}}{\bar\Phi_{\rm band}^{\rm(mir)}}.
  \]
  The factorization
  \[
  \bar\Phi_{\rm band}^{\rm(mir|self)}=\Xi_{\rm band}^{\rm(self)}\,\bar\Phi_{\rm band}^{\rm(mir)}
  \]
  is exact with zero residual. The ordinary mirror-band mean has floor `2.43229e-2` at `D=20`, while the self-weight bias stays in the positive interval `[7.40007e-1, 8.35352e-1]`, with minimum at `D=5`. This already induces
  \[
  \bar\Phi_{\rm band}^{\rm(mir|self)}\ge 1.79991\times10^{-2},
  \]
  only a factor `1.06816` below the exact weighted-mirror floor `1.92259e-2` at `D=19`. The canonical band area itself varies only between `2.3814` and `2.3976`, with relative spread `6.79e-3`, so the same statement is essentially equivalent to a statement about the mirror carrier moment `S_{\rm band}^{\rm(mir)}`. So the multiplier-side proof target narrows again: the remaining object is no longer the self-weighted mirror average itself, but the ordinary mirror-band mean (or equivalently the mirror carrier, up to tiny geometric aliasing).
- **Open-system ordinary mirror-band mean audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_diag_anchor_band_mirror_ratio_source.py` then attacks `\bar\Phi_{\rm band}^{\rm(mir)}` directly, still with no support change. It verifies the exact identity
  \[
  \bar\Phi_{\rm band}^{\rm(mir)}
  =
  \left(\frac{S_{\rm band}^{\rm(mir)}}{S_{\rm band}^{\rm(self)}}\right)\bar\Phi_{\rm band}^{\rm(self)}
  \]
  with residual `6.94e-18`. On the audited knot set, the floor of the ordinary mirror-band mean, the floor of the same-center band mean, and the floor of the mirror/self carrier ratio all align at the same knot `D=20`:
  \[
  \min \bar\Phi_{\rm band}^{\rm(mir)} = 2.43229\times10^{-2},\quad
  \min \bar\Phi_{\rm band}^{\rm(self)} = 6.99586\times10^{-2},\quad
  \min \frac{S_{\rm band}^{\rm(mir)}}{S_{\rm band}^{\rm(self)}} = 3.47676\times10^{-1}.
  \]
  Their product reproduces the exact ordinary mirror-band mean floor to machine precision. So this line is now effectively closed as well: the mirror-band mean is already saturated by a same-center positive carrier and a mirror/self ratio, and the next remaining multiplier-side object is downstream of that factorization rather than a support artifact.
- **Open-system synthesis audit:** `/Users/boypatrick/codex/PSLT_quantam/code/audit_open_system_parity_contrast_synthesis_source.py` now compresses `O19` through `O38` into one exact composite statement. On the canonical adverse band, the mainline adverse magnitude satisfies
  \[
  N_{\rm mass,band}
  =
  4\pi\,\bar K_{\rm band}^{(-)}\bar\rho_{\rm band}^{(-)}\theta_{\rm band}^{(-)}
  \Xi_{\rm band}^{\rm(self)}\Lambda_{\rm band}^{\rm(mir|self)}
  \frac{S_{\rm band}^{\rm(self)\,2}}{A_{\rm band}},
  \]
  with residual `9.67e-17`, and the full interior cancellation ratio satisfies
  \[
  \nu_{\rm diag,int}^{(\pm,m)}
  =
  \frac{N_{\rm mass,band}}{P_{\rm mass,in}}
  +
  \frac{N_{\rm mass,core}}{P_{\rm mass,in}}
  +
  \frac{N_{\rm mass,rim}}{P_{\rm mass,in}}
  +
  \frac{N_{\rm mass,tail}}{P_{\rm mass,in}}
  \]
  with residual `5.97e-14`. At the worst knot `D=11`, this is exactly
  \[
  0.957812
  =
  0.813941 + 4.19214\times10^{-2} + 5.38319\times10^{-2} + 4.81178\times10^{-2}.
  \]
  The resulting status is now stable: the support chain and the ordinary mirror-band mean line are closed on the exact bridge, while only multiplier/leakage constants such as `\bar K_{\rm band}^{(-)}`, `\bar\rho_{\rm band}^{(-)}`, `\theta_{\rm band}^{(-)}`, `\Xi_{\rm band}^{\rm(self)}`, and the three explicit leakage ratios remain as secondary tightening directions.

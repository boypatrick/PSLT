# Consistency Fixlist v1

## Scope
This v1 patch fixes internal definition mismatches and paper/code parameter drift without changing the core theory narrative.

## Completed
- [x] Fixed undefined equation references in manuscript summary section.
  - `paper/main.tex`: `eq:Veff_unified -> eq:Veff_true`, `eq:mu_D_fit -> eq:omega_N`
- [x] Corrected Cardy entropy numeric coefficients and finite-N error estimates.
  - `paper/main.tex`: `0.91 sqrt(N) -> 1.81 sqrt(N)` and derived percentages updated.
- [x] Unified WKB convention statement.
  - `paper/main.tex`: rates/probabilities use `exp(-2S)`, splitting remains amplitude-level `exp(-S)`.
- [x] Synced Yukawa-visibility representative values with lepton-cumulative formula at `p_B=0.30`.
  - `paper/main.tex`: `B1~0.085`, `B2~0.42`, `B3=1`.
- [x] Synced code defaults to manuscript baseline for kinetics amplitudes.
  - `code/pslt_lib.py`: `A1=A2=1.0` (was `1e6,1e8`).
- [x] Synced visibility implementation to paper baseline.
  - `code/pslt_lib.py`: lepton-only cumulative Yukawa mapping; `N>3` default saturation (`B_N=1`).
- [x] Synced plotting script to baseline parameter source and H->mumu reference point.
  - `code/generate_plots.py`: baseline dict, `t_coh=1.0`, reference `(D0,eta0)=(10,1)`, proxy threshold `chi2<4`.
- [x] Removed hard-coded machine paths from reproducibility scripts.
  - `code/convergence_fixed_dz.py`, `code/validate_2d_laplacian.py`, `code/splitting_action_analysis.py`.
- [x] Fixed invalid reference-grid logic in fixed-dz convergence script.
  - `code/convergence_fixed_dz.py`: reference now uses finest available grid in the current run.

## Pending (v2 candidates)
- [ ] Recompute and regenerate all manuscript figures/tables under the new synchronized baseline, then refresh text claims that depend on those plots.
- [ ] Add a small CI-style consistency checker (paper constants vs code constants vs generated metadata).
- [ ] Add BibTeX build step (`main.bbl`) to fully eliminate citation warnings in standalone compile.

## Suggested validation commands
- `python3 /Users/boypatrick/codex/PSLT_20260202/code/convergence_fixed_dz.py`
- `python3 /Users/boypatrick/codex/PSLT_20260202/code/validate_2d_laplacian.py`
- `python3 /Users/boypatrick/codex/PSLT_20260202/code/splitting_action_analysis.py`
- `python3 /Users/boypatrick/codex/PSLT_20260202/code/generate_plots.py`
- `pdflatex -interaction=nonstopmode -halt-on-error /Users/boypatrick/codex/PSLT_20260202/paper/main.tex`

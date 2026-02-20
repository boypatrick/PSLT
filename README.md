# PSLT Research Bundle (2026-02-01)

**Title:** The Projection Spectral Layer Theory (PSLT): A Rank-2 Computable Closure for the Three-Generation Structure and Higgs Signal Strength
**Author:** Bo-Yu Chen (Independent Researcher)
**Date:** February 1, 2026

This bundle contains the complete source code, data, and manuscript for the PSLT verification project.

**Reproducibility badge:** Core figures/tables are reproducible via `python3 code/generate_plots.py` and `python3 code/extract_chi_localized_2d.py`.

## Directory Structure
- `paper/`: Contains the main Latex manuscript (`main.tex`).
- `code/`: Contains the Python verification scripts.
  - `pslt_lib.py`: The core unified library (Kinetics, Visibility, Parameters).
  - `generate_plots.py`: One-click script to reproduce all figures.
- `data/`: Contains PDG data files (`pdg_leptons.json`, `pdg_quarks.json`).
- `output/`: Contains the generated plots and figures used in the paper.

## Reproduction Instructions

### 1. Verification Plots
To reproduce all phase diagrams and verification plots, run:

```bash
cd code
python3 generate_plots.py
```

This will populate the `output/` directory with:
- `three_generation_phase_diagram.png`
- `three_generation_bars.png`
- `hmumu_exclusion.png`
- `hmumu_signal_strength.png`

## Full Reproducibility
All manuscript-level outputs can be regenerated with:

```bash
cd code
python3 generate_plots.py
python3 extract_chi_localized_2d.py --Ds 6,12,18
python3 extract_chi_localized_2d.py --full-scan
python3 extract_omega_exact_convergence.py --Ds 6,12,18
python3 extract_tcoh_dephasing_1d.py --Ds 6,12,18
python3 extract_tcoh_dephasing_1d.py --full-scan
python3 scan_tcoh_profile_impact.py
python3 extract_gn_phase_space_candidate.py --D 12
python3 lindblad_chi_minimal.py
python3 extract_gn_phase_space_2d.py --Ds 6,12,18
python3 extract_chi_open_system_geometry.py --Ds 6,12,18
```

This also writes the full localized-channel profile tables to:
- `output/chi_fp_2d/localized_chi_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv`
- `output/chi_fp_2d/localized_chi_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_relerr.csv`
- `output/omega_fp_1d/omega_exact_D6-12-18.csv`
- `output/omega_fp_1d/omega_exact_D6-12-18_relerr.csv`
- `output/tcoh_fp_1d/tcoh_dephasing_D6-12-18.csv`
- `output/tcoh_fp_1d/tcoh_dephasing_D6-12-18_relerr.csv`
- `output/tcoh_fp_1d/tcoh_dephasing_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv`
- `output/tcoh_fp_1d/tcoh_profile_impact.csv`
- `output/gn_fp_1d/gn_phase_space_candidate_D12.csv`
- `output/chi_open_system/lindblad_minimal_timeseries.csv`
- `output/gn_fp_2d/gn_phase_space_2d_D6-12-18.csv`
- `output/gn_fp_2d/gn_phase_space_2d_D6-12-18_relerr.csv`
- `output/chi_open_system/chi_open_system_geometry_D6-12-18.csv`

### 2. Paper Compilation
To compile the manuscript (requires a standard TexLive distribution with RevTeX 4.2):

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

## Requirements
- Python 3.8+
- `numpy`, `scipy`, `matplotlib`
- `pdflatex` (for manuscript)

## Key Results
- **Three-Generation Stability:** In the current D-interpolated localized-channel scan, the theory gives Generation Ratio $\mathcal{R}_3>90\%$ over **80.0%** of the sampled $(D,\eta)$ grid, while $\mathcal{R}_3>95\%$ is not reached in this setup.
- **H→μμ Proxy Compatibility:** The proxy-accepted region ($\chi^2<4$) occupies about **9.4%** of the sampled grid, with best grid point near $(D,\eta)\approx(9.97,1.36)$ and $\chi^2\sim 3.0\times10^{-7}$.
- **Yukawa-Anchored Visibility + High-$N$ Stability:** Visibility is anchored to SM Yukawa couplings via a Yukawa-proportional law with sublinear compression ($B_{1,2,3}\propto Y_{1,2,3}^{p_B}$), while a $q^{(N-1)^2}$-type regulator in $g_N$ prevents high-$N$ runaway.

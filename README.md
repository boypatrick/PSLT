# PSLT Research Bundle (2026-02-01)

**Title:** The Projection Spectral Layer Theory (PSLT): A Rank-2 Computable Closure for the Three-Generation Structure and Higgs Signal Strength
**Author:** Bo-Yu Chen (Independent Researcher)
**Date:** February 1, 2026

This bundle contains the complete source code, data, and manuscript for the PSLT verification project.

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
- **Three-Generation Stability:** The theory generates a stable 3-generation structure (Generation Ratio > 95%) in over 56% of the parameter space.
- **H→μμ Compatibility:** The predicted signal strength is consistent with ATLAS/CMS measurements ($\chi^2 \approx 0$ at best fit).
- **Yukawa-Anchored Visibility + High-$N$ Stability:** Visibility is anchored to SM Yukawa couplings via a Yukawa-proportional law with sublinear compression ($B_{1,2,3}\propto Y_{1,2,3}^{p_B}$), while a $q^{(N-1)^2}$-type regulator in $g_N$ prevents high-$N$ runaway.

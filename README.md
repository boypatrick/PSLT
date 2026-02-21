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


## Requirements
- Python 3.8+
- `numpy`, `scipy`, `matplotlib`
- `latexmk` + `pdflatex` (only if `--with-paper`)

## Key Results
- **Three-Generation Stability:** In the current D-interpolated localized-channel scan, the theory gives Generation Ratio $\mathcal{R}_3>90\%$ over **80.0%** of the sampled $(D,\eta)$ grid, while $\mathcal{R}_3>95\%$ is not reached in this setup.
- **H→μμ Proxy Compatibility:** The proxy-accepted region ($\chi^2<4$) occupies about **9.4%** of the sampled grid, with best grid point near $(D,\eta)\approx(9.97,1.36)$ and $\chi^2\sim 3.0\times10^{-7}$.
- **Yukawa-Anchored Visibility + High-$N$ Stability:** Visibility is anchored to SM Yukawa couplings via a Yukawa-proportional law with sublinear compression ($B_{1,2,3}\propto Y_{1,2,3}^{p_B}$), while a $q^{(N-1)^2}$-type regulator in $g_N$ prevents high-$N$ runaway.

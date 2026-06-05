# EPJC Official Springer Template Track

This directory is the official-template preparation track for EPJC.  It is derived from `paper_epjc/`, but uses the Springer Nature December 2024 `sn-jnl` template with the EPJC-recommended `[iicol]` option.

## Official-format choices

- Class: `\documentclass[pdflatex,sn-mathphys-num,iicol]{sn-jnl}`.
- Bibliography style: `sn-mathphys-num`, supplied by the official template package.
- Main manuscript: `main.tex`, with neutral `Release...` macros and the release-result table inlined rather than loaded through `\input`.
- Release numbers: `release_numbers.json` mirrors the canonical release values used by the inlined table.
- Figures: copied into this folder with `Fig*.png` names to reduce submission-upload path risk; Fig. 7 uses the refreshed readable diagnostic callout.
- Statements and Declarations: funding, author contribution, competing interests, data/code availability, and supplementary-information statements are included.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The downloaded official package is kept as `springer-nature-latex-template.zip`, and the unpacked source is kept under `sn-article-template/` for provenance.

## Final pre-submission checklist

- Corresponding-author email is set in `main.tex`; the affiliation location is set to Hsinchu City, Taiwan.
- Current temporary repository-version statement: `https://github.com/boypatrick/PSLT`, tag `epjc-submission-v2`.
- Replace the temporary repository-version statement with a Zenodo/OSF/GitHub archive DOI if a persistent archive is minted before or during final submission.
- If this becomes the final source archive, make the shared release-figure generator output-directory aware.
- Re-run visual spot-check after any figure or table edits.

# EPJC Preparation Track

This directory is the EPJC-facing short-article preparation track.  It is derived from `paper_prd/`, but it is intended to be journal-neutral/EPJC-facing rather than PRD-branded.  The long manuscript in `paper/` remains the technical companion / supplement.

## Target Positioning

- Target journal: European Physical Journal C (EPJC), as a computational/theoretical high-energy physics article.
- Length target: <= 40 pages after figures and references.
- Scientific claim: conditional EFT-level spectral-layer occupancy closure on a fixed projected two-center background.
- Do not claim: full EYMH derivation of exactly three Standard Model generations, global pole family, formal Higgs likelihood, or absolute Higgs-sector prediction.

## Editorial Priorities Before Submission

1. Keep the article independent of the long supplement: definitions, A1--A5, main formulas, headline numbers, and limitations must be understandable in this folder.
2. Keep all PRD-specific wording out of reader-facing text.
3. Add EPJC-style data/code/supplement availability language before final submission.
4. Keep `H->mumu` as a reference-normalized diagnostic, not an experimental validation claim.
5. Keep optional rare-B/WET, Spin^c, DSS/Floquet, open-system, and static-width material outside the baseline unless explicitly promoted after a separate gate.

## Page Budget

| Section | Target pages | Purpose |
| --- | ---: | --- |
| Introduction and claim calibration | 3 | State the narrow conditional claim and comparison class. |
| Assumption map | 3 | A1--A5 firewall before theorems/results. |
| Projected two-center operator | 5 | Geometry, conformal reduction, V_eff, first physical figure. |
| Rank-2 occupancy closure | 5 | Master equation, WKB, rank-2 two-lobe kinetics, second physical figure. |
| Main release results | 8 | R3 map, no-fourth certificate, minimal tables. |
| Projector coherence | 4 | T1--T3c distilled theorem and margins. |
| Higgs diagnostic | 4 | Reference-normalized H->mumu target-region map only. |
| Limitations/reproducibility/conclusion | 6 | Honest boundary, artifact map, supplement relation. |

## Figure and Generated-Asset Policy

The current EPJC preparation draft reuses most short-article generated assets copied from `paper_prd/`, while regenerating the journal-neutral master closure diagram locally:

- `figures/epjc_master_closure_diagram.png`
- `figures/prd_r3_occupancy_map.png`
- `figures/prd_no_fourth_bound_certificate.png`
- `figures/prd_hmumu_diagnostic_map.png`
- `figures/prd_projector_transport_margins.png`
- `figures/prd_short_release_numbers.json`
- `generated/prd_short_release_macros.tex`
- `generated/prd_short_release_results_table.tex`

Most filenames remain historical for now; reader-facing captions in `main.tex` should not refer to PRD.  If this track becomes the final target, either rename the remaining generated assets/macros or update `code/generate_prd_short_figures.py` to accept an output directory.

## Supplement Policy

Keep these in the technical companion unless a reviewer asks for promotion:

- O/S/U/A proposition chains.
- CAP/ECS/Whittaker/static-width audits.
- Open-system exact-bridge details.
- Protected Spin^c V0--V6 conditional completion package.
- DSS/Floquet D0--D3 diagnostic/negative package.
- Rare-B / b -> s mu+ mu- WET diagnostic bridge.
- Long artifact paths and old comparator branches.

Status tags used when reading the companion:

- `Baseline input`: used in release numbers.
- `Certificate`: finite-domain proof or numerical guard for the baseline.
- `Diagnostic / monitor`: checked but not propagated into release numbers.
- `Negative result`: failed route retained for transparency and to avoid repeated searches.
- `Future route`: possible strengthening not assumed by the short article.

## Build

From this directory:

```bash
python3 generate_epjc_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The bibliography is resolved through `../paper/mainRefs.bib`, and figures are loaded from this folder plus the long-paper/output figure paths.  If release figures are regenerated from the repository root, copy the refreshed generated files from `paper_prd/` into this directory until the generator is made output-directory aware, then rerun `python3 generate_epjc_figures.py` to keep the master closure diagram journal-neutral.

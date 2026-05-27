# PRD Short Manuscript Track

This directory is the submission-facing short-paper track.  It treats the existing long manuscript in `paper/` as a technical companion / supplement rather than trying to compress every audit chain into the PRD article.

## Target

- Length target: <= 40 pages after figures and references.
- Positioning: conditional EFT-level spectral-layer occupancy closure on a fixed projected two-center background.
- Do not claim: full EYMH derivation of exactly three Standard Model generations, global pole family, formal Higgs likelihood, or absolute Higgs-sector prediction.

## Page Budget

| Section | Target pages | Purpose |
| --- | ---: | --- |
| Introduction and claim calibration | 3 | State the narrow PRD claim and comparison class. |
| Assumption map | 3 | A1--A5 firewall before theorems/results. |
| Projected two-center operator | 5 | Geometry, conformal reduction, V_eff, first physical figure. |
| Rank-2 occupancy closure | 5 | Master equation, WKB, rank-2 two-lobe kinetics, second physical figure. |
| Main release results | 8 | R3 map, no-fourth certificate, minimal tables. |
| Projector coherence | 4 | T1--T3c distilled theorem and margins. |
| Higgs diagnostic | 4 | Reference-normalized H->mumu target-region map only. |
| Limitations/reproducibility/conclusion | 6 | Honest boundary, artifact map, supplement relation. |

## Figure Plan

1. `paper/figures/two_center_conformal_dumbbell.png`
2. `paper/figures/spectral_layers_double_well.png`
3. Compact master-equation module diagram.
4. First-three-layer occupancy / R3 map.
5. No-fourth finite-domain kinetic certificate or bound-layer threshold.
6. Reference-normalized H->mumu diagnostic map.
7. Optional projector-transport coherence margin plot.

## Supplement Policy

Keep these in the technical companion unless a reviewer asks for promotion:

- O/S/U/A proposition chains.
- CAP/ECS/Whittaker/static-width audits.
- Open-system exact-bridge details.
- Protected Spin^c V0--V6 conditional completion package.
- DSS/Floquet D0--D3 diagnostic/negative package.
- Long artifact paths and old comparator branches.

## Build

From this directory:

```bash
python3 ../code/generate_prd_short_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The bibliography is resolved through `../paper/mainRefs.bib`, and figures are loaded from `../paper/`, `../paper/figures/`, and `../output/`.

Initial sanity check: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passes and produces a short PDF before generated build artifacts are cleaned.

## Generated Short-Paper Figures

Run `python3 code/generate_prd_short_figures.py` from the repository root, or `python3 ../code/generate_prd_short_figures.py` from this directory.  The script reads existing gated artifacts and writes:

- `paper_prd/figures/prd_master_closure_diagram.png`
- `paper_prd/figures/prd_r3_occupancy_map.png`
- `paper_prd/figures/prd_no_fourth_bound_certificate.png`
- `paper_prd/figures/prd_hmumu_diagnostic_map.png`
- `paper_prd/figures/prd_projector_transport_margins.png`
- `paper_prd/figures/prd_short_release_numbers.json`
- `paper_prd/generated/prd_short_release_macros.tex`
- `paper_prd/generated/prd_short_release_results_table.tex`

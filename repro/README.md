# Reproducible Delivery (Fig/Table)

This folder defines a clone-and-run reproducibility contract for the paper outputs.

## Entry Point

Run from repository root:

```bash
bash scripts/repro/reproduce_paper.sh
```

Optional variants:

```bash
# Also compile paper/main.tex (requires latexmk + RevTeX toolchain)
bash scripts/repro/reproduce_paper.sh --with-paper

# Package existing outputs only (skip generation)
bash scripts/repro/reproduce_paper.sh --package-only
```

## Output Layout

Each run is isolated under:

`repro/runs/<RUN_ID>/`

Generated contents:

- `logs/`: per-step execution logs.
- `steps.tsv`: step status and timing.
- `figures/`: normalized names `Fig_XX_*.png`.
- `tables/`: normalized names `Tab_XX_*.csv`.
- `paper/`: optional packaged `Paper_main.pdf`.
- `artifact_index.csv`: source -> artifact mapping with checksums.
- `checksums.sha256`: sha256 file for packaged artifacts.
- `manifest.json`: run metadata (timestamp, commit, missing/packaged counts).

`repro/latest` is a symlink to the most recent run directory.

## Naming Convention

- Figures: `Fig_<2-digit index>_<slug>.png`
- Tables: `Tab_<2-digit index>_<slug>.csv`
- Paper: `Paper_main.pdf` (optional)

The source-to-artifact mapping is versioned in `repro/artifact_map.csv`.

## Scope

This reproducibility layer targets paper-level deliverables (figures, table CSVs, optional PDF) and does not rename canonical raw outputs under `output/`.

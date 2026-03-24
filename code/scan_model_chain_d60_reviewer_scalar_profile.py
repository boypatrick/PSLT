#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"

REF_MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D60E21_tuned.csv"
CUR_MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21.csv"

HOTSPOT_CENTERS = (
    5.627118644067797,
    5.898305084745763,
    6.4406779661016955,
)


def _mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REF_MAP)
    cur = pd.read_csv(CUR_MAP)
    merged = ref.merge(cur, on=["D", "eta"], suffixes=("_ref", "_cur"))

    rows: list[dict[str, float]] = []
    for d_val in HOTSPOT_CENTERS:
        grp = merged[np.isclose(merged["D"], float(d_val))].copy()
        ratio = (grp["mu_mumu_ref"] / grp["mu_mumu_cur"]).astype(float)
        mu_scalar = float((ratio.iloc[0]))
        rows.append({"D": float(d_val), "mu_scalar": mu_scalar})

    out = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    csv_path = OUTDIR / "model_chain_d60_reviewer_scalar_profile.csv"
    meta_path = OUTDIR / "model_chain_d60_reviewer_scalar_profile_run_meta.json"
    out.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "ref_map": str(REF_MAP),
                "cur_map": str(CUR_MAP),
                "hotspot_centers": [float(x) for x in HOTSPOT_CENTERS],
            },
            indent=2,
        )
    )
    _mirror([csv_path, meta_path])
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

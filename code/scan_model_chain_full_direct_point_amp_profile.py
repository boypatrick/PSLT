#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"

D_GRID60 = np.linspace(4.0, 20.0, 60)
FOCUS_TARGETS = np.array([5.9, 6.17, 6.44], dtype=float)
FOCUS_D = np.array(
    [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - target)))]) for target in FOCUS_TARGETS],
    dtype=float,
)
ETA_GRID = np.linspace(0.2, 4.0, 21)
LAYER = 2
OBSERVABLE_MODE = "eft_wilson_uv_rge"
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
        d_min=4.0,
        d_max=20.0,
        d_num=60,
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
        runtime_direct_force=False,
        runtime_direct_no_cache=False,
        runtime_direct_chi_rho_max=3.0,
        runtime_direct_chi_z_margin=6.0,
        runtime_direct_chi_n_mu=120,
        runtime_direct_chi_tol=1e-8,
        runtime_direct_chi_maxiter=30000,
        runtime_direct_chi_sigma=2.5,
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    kin = make_baseline_kinetics(chain_mode="full_direct", **_common_kwargs())
    rows: list[dict[str, float | int | str]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            amp = float(
                kin.hll_channel_amplitude(
                    LAYER,
                    float(D),
                    float(eta),
                    T_COH,
                    observable_mode=OBSERVABLE_MODE,
                    N_max=N_MAX,
                )
            )
            rows.append(
                {
                    "observable_mode": OBSERVABLE_MODE,
                    "layer": int(LAYER),
                    "D": float(D),
                    "eta": float(eta),
                    "amp": amp,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = OUTDIR / "model_chain_full_direct_point_amp_profile_D60_hotspot_layer2.csv"
    meta_path = OUTDIR / "model_chain_full_direct_point_amp_profile_D60_hotspot_layer2_run_meta.json"
    df.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "chain_mode": "full_direct",
                "observable_mode": OBSERVABLE_MODE,
                "layer": int(LAYER),
                "focus_D": [float(x) for x in FOCUS_D],
                "eta_grid": [float(x) for x in ETA_GRID],
            },
            indent=2,
        )
    )
    for src in (csv_path, meta_path):
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()

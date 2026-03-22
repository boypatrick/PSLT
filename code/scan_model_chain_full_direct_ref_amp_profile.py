#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
REF_D = 9.6
REF_ETA = 1.0
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])
OBSERVABLE_MODE = "eft_wilson_uv_rge"
LAYERS = (1, 2, 3)


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
        d_min=4.0,
        d_max=20.0,
        d_num=21,
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
    for layer in LAYERS:
        amp_ref = float(
            kin.hll_channel_amplitude(
                layer,
                REF_D,
                REF_ETA,
                T_COH,
                observable_mode=OBSERVABLE_MODE,
                N_max=N_MAX,
            )
        )
        rows.append(
            {
                "observable_mode": OBSERVABLE_MODE,
                "layer": int(layer),
                "ref_D": float(REF_D),
                "ref_eta": float(REF_ETA),
                "amp_ref": amp_ref,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUTDIR / "model_chain_full_direct_ref_amp_profile.csv"
    meta_path = OUTDIR / "model_chain_full_direct_ref_amp_profile_run_meta.json"
    df.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "chain_mode": "full_direct",
                "observable_mode": OBSERVABLE_MODE,
                "ref_D": REF_D,
                "ref_eta": REF_ETA,
                "layers": list(LAYERS),
            },
            indent=2,
        )
    )
    for src in (csv_path, meta_path):
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"

D_GRID = np.linspace(4.0, 20.0, 21)
ETA_GRID = np.array([0.2, 0.6, 1.0, 2.0, 4.0], dtype=float)
REF_D = 9.6
REF_ETA = 1.0
T_COH = float(PAPER_BASELINE["t_coh"])
N_MAX = int(PAPER_BASELINE["hll_observable_nmax"])


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode="eft_wilson_uv_rge",
        d_min=float(D_GRID.min()),
        d_max=float(D_GRID.max()),
        d_num=int(len(D_GRID)),
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
    rows: list[dict[str, float]] = []
    for D in D_GRID:
        for eta in ETA_GRID:
            width_ratio = float(
                kin.hll_total_width_ratio_uv_rge(
                    D=float(D),
                    eta=float(eta),
                    t_coh=T_COH,
                    ref_D=REF_D,
                    ref_eta=REF_ETA,
                    N_max=N_MAX,
                )
            )
            rows.append({"D": float(D), "eta": float(eta), "width_ratio": width_ratio})

    df = pd.DataFrame(rows)
    csv_path = OUTDIR / "model_chain_full_direct_width_profile_Dgrid21_Egrid5.csv"
    png_path = OUTDIR / "model_chain_full_direct_width_profile_Dgrid21_Egrid5.png"
    meta_path = OUTDIR / "model_chain_full_direct_width_profile_Dgrid21_Egrid5_run_meta.json"

    df.to_csv(csv_path, index=False)

    pivot = df.pivot(index="eta", columns="D", values="width_ratio").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.1f}" for y in pivot.index], fontsize=8)
    ax.set_xlabel("D")
    ax.set_ylabel("eta")
    ax.set_title("Full-Direct Width-Ratio Profile")
    fig.colorbar(im, ax=ax, label="width_ratio")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    meta_path.write_text(
        json.dumps(
            {
                "d_grid": [float(x) for x in D_GRID],
                "eta_grid": [float(x) for x in ETA_GRID],
                "chain_mode": "full_direct",
                "observable_mode": "eft_wilson_uv_rge",
                "ref_D": REF_D,
                "ref_eta": REF_ETA,
            },
            indent=2,
        )
    )

    for src in (csv_path, png_path, meta_path):
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

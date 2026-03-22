#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"

D_GRID = np.linspace(4.0, 20.0, 21)


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
        b = np.asarray([float(kin.B_N(n, float(D))) for n in (1, 2, 3)], dtype=float)
        rows.append(
            {
                "D": float(D),
                "B1": float(b[0]),
                "B2": float(b[1]),
                "B3": float(b[2]),
                "Bsum": float(np.sum(b)),
            }
        )
    df = pd.DataFrame(rows)

    csv_path = OUTDIR / "model_chain_full_direct_bsum_profile_Dgrid21.csv"
    png_path = OUTDIR / "model_chain_full_direct_bsum_profile_Dgrid21.png"
    meta_path = OUTDIR / "model_chain_full_direct_bsum_profile_Dgrid21_run_meta.json"

    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["D"], df["Bsum"], marker="o", label="Bsum")
    ax.plot(df["D"], df["B2"], marker="s", label="B2")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("D")
    ax.set_ylabel("value")
    ax.set_title("Full-Direct Observable B Profile")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    meta_path.write_text(
        json.dumps(
            {
                "d_grid": [float(x) for x in D_GRID],
                "chain_mode": "full_direct",
                "observable_mode": "eft_wilson_uv_rge",
            },
            indent=2,
        )
    )

    for src in (csv_path, png_path, meta_path):
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

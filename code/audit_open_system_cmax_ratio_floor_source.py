#!/usr/bin/env python3
"""
Audit a tighter Cmax floor on the exact open-system bridge by using the
ratio-variable closed form.

For
    a = gamma_phi / delta,
    b = gamma_mix / delta,

one has the exact underdamped closed form

    Cmax = F(a,b)

with F depending only on (a,b), not on delta separately.  Since the exported
exact bridge lives in a small ratio box and F is monotone decreasing in both
variables there, a tighter floor is obtained from F(a_max,b_max).  Subtracting
the exported exact-vs-closed residual bound then yields a rigorous floor for the
exported exact Cmax itself.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


def read_csv_any(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(paths[0])


def cmax_ratio_closed_form(a: float, b: float) -> float:
    if a >= 1.0:
        return float("nan")
    s = math.sqrt(max(1.0 - a * a, 0.0))
    if s <= 0.0:
        return 0.0
    q = a + 2.0 * b
    return float(
        1.0 / (2.0 * math.sqrt(1.0 + 4.0 * b * b + 4.0 * a * b))
        * math.exp(-(q / s) * math.atan2(s, q))
    )


def main() -> None:
    bridge = read_csv_any(
        OUTDIR / "chi_open_system_micro_bridge_map.csv",
        PAPER_DIR / "chi_open_system_micro_bridge_map.csv",
    ).copy()
    cmax_summary = read_csv_any(
        OUTDIR / "chi_open_system_cmax_floor_source_summary.csv",
        PAPER_DIR / "chi_open_system_cmax_floor_source_summary.csv",
    ).iloc[0]
    chi_summary = read_csv_any(
        OUTDIR / "chi_open_system_chi_eff_floor_source_summary.csv",
        PAPER_DIR / "chi_open_system_chi_eff_floor_source_summary.csv",
    ).iloc[0]

    delta = bridge["delta_micro"].astype(float).to_numpy()
    gamma_phi = bridge["gamma_phi_micro"].astype(float).to_numpy()
    gamma_mix = bridge["gamma_mix_micro"].astype(float).to_numpy()
    c_exact = bridge["Cmax_micro"].astype(float).to_numpy()

    a = gamma_phi / np.maximum(delta, 1e-300)
    b = gamma_mix / np.maximum(delta, 1e-300)
    c_ratio = np.array([cmax_ratio_closed_form(float(ai), float(bi)) for ai, bi in zip(a, b)], dtype=float)

    bridge["a_ratio"] = a
    bridge["b_ratio"] = b
    bridge["Cmax_ratio_closed_form"] = c_ratio
    bridge["Cmax_ratio_closed_form_residual_abs"] = np.abs(c_exact - c_ratio)

    amax = float(a.max())
    bmax = float(b.max())
    floor_ratio_closed = cmax_ratio_closed_form(amax, bmax)
    eps_closed = float(cmax_summary["max_abs_Cmax_closed_form_residual"])
    floor_ratio_rig = max(floor_ratio_closed - eps_closed, 0.0)

    # finite-difference monotonicity audit on the ratio box
    h = 1.0e-6
    min_dF_da = float("inf")
    max_dF_da = -float("inf")
    min_dF_db = float("inf")
    max_dF_db = -float("inf")
    for aa in np.linspace(0.0, amax, 61):
        for bb in np.linspace(0.0, bmax, 61):
            aa_lo = max(0.0, aa - h)
            aa_hi = min(amax, aa + h, 1.0 - 1.0e-9)
            bb_lo = max(0.0, bb - h)
            bb_hi = min(bmax, bb + h)
            dF_da = (cmax_ratio_closed_form(aa_hi, bb) - cmax_ratio_closed_form(aa_lo, bb)) / max(aa_hi - aa_lo, 1.0e-12)
            dF_db = (cmax_ratio_closed_form(aa, bb_hi) - cmax_ratio_closed_form(aa, bb_lo)) / max(bb_hi - bb_lo, 1.0e-12)
            min_dF_da = min(min_dF_da, dF_da)
            max_dF_da = max(max_dF_da, dF_da)
            min_dF_db = min(min_dF_db, dF_db)
            max_dF_db = max(max_dF_db, dF_db)

    a_chi_sep_floor = float(chi_summary["A_chi_separated_floor"])
    chi_floor_from_ratio_rig = a_chi_sep_floor * floor_ratio_rig

    out_detail = OUTDIR / "chi_open_system_cmax_ratio_floor_source_detail.csv"
    out_summary = OUTDIR / "chi_open_system_cmax_ratio_floor_source_summary.csv"
    bridge.to_csv(out_detail, index=False)

    summary = pd.DataFrame([
        {
            "n_rows": int(len(bridge)),
            "a_ratio_max": amax,
            "a_ratio_max_D": float(bridge.loc[np.argmax(a), "D"]),
            "b_ratio_max": bmax,
            "b_ratio_max_D": float(bridge.loc[np.argmax(b), "D"]),
            "max_abs_ratio_closed_form_residual": float(np.max(np.abs(c_exact - c_ratio))),
            "Cmax_ratio_closed_form_floor": floor_ratio_closed,
            "Cmax_ratio_rigorous_floor": floor_ratio_rig,
            "Cmax_exact_floor": float(np.min(c_exact)),
            "actual_over_ratio_closed_form_floor": float(np.min(c_exact) / max(floor_ratio_closed, 1e-300)),
            "actual_over_ratio_rigorous_floor": float(np.min(c_exact) / max(floor_ratio_rig, 1e-300)),
            "min_dF_da_on_ratio_box": min_dF_da,
            "max_dF_da_on_ratio_box": max_dF_da,
            "min_dF_db_on_ratio_box": min_dF_db,
            "max_dF_db_on_ratio_box": max_dF_db,
            "A_chi_separated_floor": a_chi_sep_floor,
            "chi_eff_floor_from_ratio_rigorous_Cmax_floor": chi_floor_from_ratio_rig,
            "actual_chi_eff_floor": float(chi_summary["actual_chi_eff_floor"]),
            "actual_over_ratio_rigorous_chi_eff_floor": float(float(chi_summary["actual_chi_eff_floor"]) / max(chi_floor_from_ratio_rig, 1e-300)),
        }
    ])
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f"[saved] {out_detail}")
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()

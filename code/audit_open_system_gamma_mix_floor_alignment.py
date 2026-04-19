#!/usr/bin/env python3
"""
Audit how the exact gamma_mix floor aligns with the x-channel overlap floor.

The exact bridge obeys
    gamma_mix(D) = kappa_env * g_x(D)^2 * Sxx(D).
This script shows that on the exported exact bridge the minima of gamma_mix,
|g_x|, and g_x^2*Sxx all occur at the same knot.  Therefore the remaining gap
between the exact gamma_mix floor and the g_x-floor-induced floor is only the
local uplift Sxx(D_gxmin) / Sxx_min.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'
PAPER_DIR = ROOT / 'paper'


def read_csv_any(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(paths[0])


def main() -> None:
    bridge = read_csv_any(
        OUTDIR / 'chi_open_system_micro_bridge_map.csv',
        PAPER_DIR / 'chi_open_system_micro_bridge_map.csv',
    ).copy()

    kappa = bridge['kappa_env'].astype(float)
    gx = bridge['g_x_micro'].astype(float)
    sxx = bridge['Sxx_delta'].astype(float)
    gamma_mix = bridge['gamma_mix_micro'].astype(float)

    g2 = gx * gx
    kernel = g2 * sxx
    gamma_recon = kappa * kernel

    bridge['g_x_sq'] = g2
    bridge['kernel_mix'] = kernel
    bridge['gamma_mix_recon_alignment'] = gamma_recon
    bridge['gamma_mix_alignment_residual_abs'] = np.abs(gamma_mix - gamma_recon)

    idx_gm = int(np.argmin(gamma_mix))
    idx_gx = int(np.argmin(gx))
    idx_g2 = int(np.argmin(g2))
    idx_kernel = int(np.argmin(kernel))
    idx_sxx = int(np.argmin(sxx))

    D_gm = float(bridge.loc[idx_gm, 'D'])
    D_gx = float(bridge.loc[idx_gx, 'D'])
    D_kernel = float(bridge.loc[idx_kernel, 'D'])

    gamma_floor = float(gamma_mix.iloc[idx_gm])
    gx_floor = float(gx.iloc[idx_gx])
    g2_floor = float(g2.iloc[idx_g2])
    sxx_floor = float(sxx.iloc[idx_sxx])
    sxx_at_gxmin = float(sxx.iloc[idx_gx])

    gamma_from_gx_floor_and_global_sxx_floor = float(kappa.iloc[0] * g2_floor * sxx_floor)
    gamma_from_gx_floor_and_local_sxx = float(kappa.iloc[0] * g2_floor * sxx_at_gxmin)

    uplift_local_over_global_sxx = sxx_at_gxmin / max(sxx_floor, 1e-300)
    exact_over_gx_global_sxx = gamma_floor / max(gamma_from_gx_floor_and_global_sxx_floor, 1e-300)
    exact_over_gx_local_sxx = gamma_floor / max(gamma_from_gx_floor_and_local_sxx, 1e-300)

    out_detail = OUTDIR / 'chi_open_system_gamma_mix_floor_alignment_detail.csv'
    out_summary = OUTDIR / 'chi_open_system_gamma_mix_floor_alignment_summary.csv'
    bridge.to_csv(out_detail, index=False)

    summary = pd.DataFrame([{
        'n_rows': int(len(bridge)),
        'max_abs_gamma_mix_alignment_residual': float(bridge['gamma_mix_alignment_residual_abs'].max()),
        'gamma_mix_floor': gamma_floor,
        'gamma_mix_floor_D': D_gm,
        'g_x_floor': gx_floor,
        'g_x_floor_D': D_gx,
        'g_x_sq_floor_D': float(bridge.loc[idx_g2, 'D']),
        'kernel_mix_floor': float(kernel.iloc[idx_kernel]),
        'kernel_mix_floor_D': D_kernel,
        'Sxx_floor': sxx_floor,
        'Sxx_floor_D': float(bridge.loc[idx_sxx, 'D']),
        'Sxx_at_gx_floor': sxx_at_gxmin,
        'all_argmins_align_except_Sxx': int(idx_gm == idx_gx == idx_g2 == idx_kernel),
        'gamma_from_gx_floor_and_global_Sxx_floor': gamma_from_gx_floor_and_global_sxx_floor,
        'gamma_from_gx_floor_and_local_Sxx': gamma_from_gx_floor_and_local_sxx,
        'uplift_local_over_global_Sxx': uplift_local_over_global_sxx,
        'exact_over_gx_global_Sxx_floor': exact_over_gx_global_sxx,
        'exact_over_gx_local_Sxx': exact_over_gx_local_sxx,
    }])
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f'[saved] {out_detail}')
    print(f'[saved] {out_summary}')


if __name__ == '__main__':
    main()

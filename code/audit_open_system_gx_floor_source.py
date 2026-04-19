#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPATH = ROOT / 'output' / 'chi_open_system' / 'chi_open_system_micro_bridge_map.csv'
OUTDIR = ROOT / 'output' / 'chi_open_system'
DETAIL = OUTDIR / 'chi_open_system_gx_floor_source_detail.csv'
SUMMARY = OUTDIR / 'chi_open_system_gx_floor_source_summary.csv'


def main() -> None:
    df = pd.read_csv(INPATH).sort_values('D').reset_index(drop=True)

    df['gx_from_localized_offdiag'] = np.abs(df['VLR_deltaV'].astype(float))
    df['gx_from_parity_diag_contrast'] = 0.5 * np.abs(df['V11_deltaV'].astype(float) - df['V22_deltaV'].astype(float))
    df['gz_from_parity_offdiag'] = np.abs(df['V12_deltaV'].astype(float))
    df['parity_diag_contrast_abs'] = np.abs(df['V11_deltaV'].astype(float) - df['V22_deltaV'].astype(float))
    df['gx_localized_offdiag_residual_abs'] = np.abs(df['g_x_micro'].astype(float) - df['gx_from_localized_offdiag'])
    df['gx_parity_diag_residual_abs'] = np.abs(df['g_x_micro'].astype(float) - df['gx_from_parity_diag_contrast'])
    df['gz_parity_offdiag_residual_abs'] = np.abs(df['g_z_micro'].astype(float) - df['gz_from_parity_offdiag'])

    gx = df['g_x_micro'].astype(float)
    contrast = df['parity_diag_contrast_abs'].astype(float)
    gx_idx = int(gx.idxmin())
    contrast_idx = int(contrast.idxmin())
    gamma_idx = int(df['gamma_mix_micro'].astype(float).idxmin())

    summary = pd.DataFrame([
        {
            'max_abs_gx_localized_offdiag_residual': float(df['gx_localized_offdiag_residual_abs'].max()),
            'max_abs_gx_parity_diag_residual': float(df['gx_parity_diag_residual_abs'].max()),
            'max_abs_gz_parity_offdiag_residual': float(df['gz_parity_offdiag_residual_abs'].max()),
            'gx_floor': float(gx.iloc[gx_idx]),
            'gx_floor_D': float(df.loc[gx_idx, 'D']),
            'parity_diag_contrast_floor': float(contrast.iloc[contrast_idx]),
            'parity_diag_contrast_floor_D': float(df.loc[contrast_idx, 'D']),
            'gamma_mix_floor_D': float(df.loc[gamma_idx, 'D']),
            'gx_argmin_aligns_with_contrast': int(gx_idx == contrast_idx),
            'gx_argmin_aligns_with_gamma_mix': int(gx_idx == gamma_idx),
            'parity_diag_contrast_over_2_equals_gx_floor': float(contrast.iloc[contrast_idx] / max(2.0 * gx.iloc[gx_idx], 1e-300)),
            'parity_diag_contrast_floor_over_gamma_mix_floor': float(contrast.iloc[contrast_idx] / max(float(df.loc[gamma_idx, 'gamma_mix_micro']), 1e-300)),
            'Sxx_at_gx_floor': float(df.loc[gx_idx, 'Sxx_delta']),
            'gamma_mix_from_parity_contrast_floor': float(df.loc[gx_idx, 'kappa_env'] * (0.5 * contrast.iloc[contrast_idx]) ** 2 * df.loc[gx_idx, 'Sxx_delta']),
            'gamma_mix_exact_floor': float(df.loc[gamma_idx, 'gamma_mix_micro']),
            'gamma_mix_reconstruction_from_parity_contrast_residual_abs': float(abs(df.loc[gamma_idx, 'gamma_mix_micro'] - (df.loc[gx_idx, 'kappa_env'] * (0.5 * contrast.iloc[contrast_idx]) ** 2 * df.loc[gx_idx, 'Sxx_delta']))),
        }
    ])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DETAIL, index=False)
    summary.to_csv(SUMMARY, index=False)
    print(DETAIL)
    print(SUMMARY)


if __name__ == '__main__':
    main()

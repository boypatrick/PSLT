#!/usr/bin/env python3
"""
convergence_fixed_dz.py

Fixed-dz convergence test for PSLT 1D axis scan.

Key difference from v3 convergence:
  - Fix dz (grid spacing), vary zmax
  - Nz = int(2 * zmax / dz) + 1
  - This isolates discretization error from domain truncation

Verification criteria:
  - |ΔE₁|/|E₁| < 10⁻³
  - |ΔS₁|/S₁ < 10⁻³
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# Import from uploaded scan engine
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'upload'))
from scan_D_axis1d_3 import omega_z, laplace_omega_z, U_z, eig_1d, wkb_action_central

def convergence_fixed_dz(
    D_points: List[float],
    dz_list: List[float],
    zmax_list: List[float],
    a: float, eps: float, m0: float, xi: float,
    n_states: int = 3
) -> pd.DataFrame:
    """
    For each D, for each dz, for each zmax:
      - Compute Nz = int(2*zmax/dz) + 1
      - Solve eigenproblem
      - Record E1, S1
    """
    rows = []
    
    for D in D_points:
        for dz in dz_list:
            for zmax in zmax_list:
                Nz = int(2 * zmax / dz) + 1
                z = np.linspace(-zmax, zmax, Nz)
                actual_dz = (2 * zmax) / (Nz - 1)
                
                U = U_z(z, D, a, eps, m0, xi)
                Evals, _ = eig_1d(U, z, n_states=n_states)
                
                E1 = float(Evals[0])
                E2 = float(Evals[1]) if len(Evals) > 1 else np.nan
                omega2_1 = m0**2 + E1
                stable1 = (E1 < 0.0) and (omega2_1 > 0.0)
                
                if stable1:
                    S1, tp = wkb_action_central(U, z, E1)
                else:
                    S1, tp = (np.nan, None)
                
                rows.append({
                    'D': D,
                    'dz_target': dz,
                    'dz_actual': actual_dz,
                    'zmax': zmax,
                    'Nz': Nz,
                    'E1': E1,
                    'E2': E2,
                    'delta_E': E2 - E1 if np.isfinite(E2) else np.nan,
                    'omega2_1': omega2_1,
                    'stable1': stable1,
                    'S1': S1 if np.isfinite(S1) else np.nan,
                    'has_tp': tp is not None
                })
    
    return pd.DataFrame(rows)


def analyze_convergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each D and dz, take the largest zmax as reference and compute relative errors.
    """
    results = []
    
    for D in df['D'].unique():
        for dz in df['dz_target'].unique():
            subset = df[(df['D'] == D) & (df['dz_target'] == dz)].sort_values('zmax')
            if len(subset) < 2:
                continue
            
            # Reference: largest zmax
            ref = subset.iloc[-1]
            E1_ref = ref['E1']
            S1_ref = ref['S1']
            
            for _, row in subset.iterrows():
                rel_E1 = abs(row['E1'] - E1_ref) / abs(E1_ref) if E1_ref != 0 else np.nan
                rel_S1 = abs(row['S1'] - S1_ref) / abs(S1_ref) if np.isfinite(S1_ref) and S1_ref != 0 else np.nan
                
                results.append({
                    'D': D,
                    'dz': dz,
                    'zmax': row['zmax'],
                    'Nz': row['Nz'],
                    'E1': row['E1'],
                    'S1': row['S1'],
                    'rel_error_E1': rel_E1,
                    'rel_error_S1': rel_S1,
                    'E1_converged': rel_E1 < 1e-3 if np.isfinite(rel_E1) else False,
                    'S1_converged': rel_S1 < 1e-3 if np.isfinite(rel_S1) else False
                })
    
    return pd.DataFrame(results)


def plot_convergence(analysis_df: pd.DataFrame, output_path: str):
    """Plot convergence analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    D_values = sorted(analysis_df['D'].unique())
    dz_values = sorted(analysis_df['dz'].unique())
    
    # Top-left: E1 vs zmax for different dz
    ax = axes[0, 0]
    for D in D_values[:1]:  # Just D=6 for clarity
        for dz in dz_values:
            subset = analysis_df[(analysis_df['D'] == D) & (analysis_df['dz'] == dz)]
            ax.plot(subset['zmax'], subset['E1'], 'o-', label=f'dz={dz}')
    ax.set_xlabel('zmax')
    ax.set_ylabel('E₁')
    ax.set_title(f'E₁ convergence (D={D_values[0]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: S1 vs zmax
    ax = axes[0, 1]
    for D in D_values[:1]:
        for dz in dz_values:
            subset = analysis_df[(analysis_df['D'] == D) & (analysis_df['dz'] == dz)]
            ax.plot(subset['zmax'], subset['S1'], 'o-', label=f'dz={dz}')
    ax.set_xlabel('zmax')
    ax.set_ylabel('S₁')
    ax.set_title(f'S₁ convergence (D={D_values[0]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Relative error E1 vs dz (for largest zmax)
    ax = axes[1, 0]
    max_zmax = analysis_df['zmax'].max()
    for D in D_values:
        subset = analysis_df[(analysis_df['D'] == D) & (analysis_df['zmax'] == max_zmax)]
        ax.semilogy(subset['dz'], subset['rel_error_E1'], 'o-', label=f'D={D}')
    ax.axhline(1e-3, color='r', linestyle='--', label='10⁻³ threshold')
    ax.set_xlabel('dz')
    ax.set_ylabel('|ΔE₁|/|E₁|')
    ax.set_title('Convergence vs dz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Relative error S1 vs dz
    ax = axes[1, 1]
    for D in D_values:
        subset = analysis_df[(analysis_df['D'] == D) & (analysis_df['zmax'] == max_zmax)]
        ax.semilogy(subset['dz'], subset['rel_error_S1'], 'o-', label=f'D={D}')
    ax.axhline(1e-3, color='r', linestyle='--', label='10⁻³ threshold')
    ax.set_xlabel('dz')
    ax.set_ylabel('|ΔS₁|/|S₁|')
    ax.set_title('S₁ convergence vs dz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    outdir = Path(__file__).parent.parent / 'output'
    outdir.mkdir(parents=True, exist_ok=True)

    # Parameters from physical_gap preset
    a = 0.040
    eps = 0.10
    m0 = 1.0
    xi = 0.14
    
    # Test points - reduced to avoid timeout
    D_points = [6.0, 12.0, 18.0]
    dz_list = [0.04, 0.02, 0.01]  # Removed 0.005 to avoid huge matrices
    zmax_list = [60.0, 80.0]  # Reduced range
    
    print("Running fixed-dz convergence test...")
    print(f"D points: {D_points}")
    print(f"dz values: {dz_list}")
    print(f"zmax values: {zmax_list}")
    
    # Run convergence sweep
    df = convergence_fixed_dz(D_points, dz_list, zmax_list, a, eps, m0, xi)
    df.to_csv(outdir / 'convergence_fixed_dz.csv', index=False)
    print(f"\nRaw results saved.")
    
    # Analyze
    analysis = analyze_convergence(df)
    analysis.to_csv(outdir / 'convergence_fixed_dz_analysis.csv', index=False)
    print(f"\nAnalysis saved.")
    
    # Plot
    plot_convergence(analysis, str(outdir / 'convergence_fixed_dz.png'))
    
    # Summary table
    print("\n" + "="*70)
    print("CONVERGENCE SUMMARY")
    print("="*70)
    
    for D in D_points:
        print(f"\nD = {D}:")
        for zmax in zmax_list:
            subset = df[(df['D'] == D) & (df['zmax'] == zmax)]
            print(f"  zmax={zmax}:")
            for _, row in subset.iterrows():
                print(f"    dz={row['dz_target']:.4f}, Nz={row['Nz']:>5}, E1={row['E1']:.6f}, S1={row['S1']:.4f}")
    
    # Convergence check
    print("\n" + "="*70)
    print("CONVERGENCE CHECK (vs finest grid)")
    print("="*70)
    
    # Use the finest available grid in this run as reference
    ref_dz = min(dz_list)
    ref_zmax = max(zmax_list)
    ref_df = df[(df['dz_target'] == ref_dz) & (df['zmax'] == ref_zmax)]
    if ref_df.empty:
        raise RuntimeError("Reference grid is empty; check dz_list/zmax_list setup.")
    
    for D in D_points:
        ref = ref_df[ref_df['D'] == D].iloc[0]
        E1_ref, S1_ref = ref['E1'], ref['S1']
        
        print(f"\nD = {D} (ref dz={ref_dz}, zmax={ref_zmax}: E1={E1_ref:.6f}, S1={S1_ref:.4f}):")
        subset = df[(df['D'] == D) & (df['zmax'] == 80.0)]  # Compare at zmax=80
        for _, row in subset.iterrows():
            rel_E = abs(row['E1'] - E1_ref) / abs(E1_ref)
            rel_S = abs(row['S1'] - S1_ref) / S1_ref if S1_ref > 0 else np.nan
            E_ok = "✓" if rel_E < 1e-3 else "✗"
            S_ok = "✓" if rel_S < 1e-3 else "✗"
            print(f"  dz={row['dz_target']:.4f}: |ΔE₁|/|E₁|={rel_E:.2e} {E_ok}, |ΔS₁|/S₁={rel_S:.2e} {S_ok}")


if __name__ == '__main__':
    main()

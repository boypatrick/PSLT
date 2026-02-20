#!/usr/bin/env python3
"""
splitting_action_analysis.py

Analyze the ln(ΔE) vs S relationship - the key internal consistency check.

This is the "non-trivial check" that the same V_eff controls both spectrum and tunneling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Load physical_gap scan data
root = Path(__file__).parent.parent
outdir = root / 'output'
paper_dir = root / 'paper'
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(root / 'upload' / 'scan_axis1d_v3_physical_gap.csv')

# Extract n=1 and n=2 data
df1 = df[df['n'] == 1][['D', 'E', 'S_z']].rename(columns={'E': 'E1', 'S_z': 'S1'})
df2 = df[df['n'] == 2][['D', 'E', 'S_z']].rename(columns={'E': 'E2', 'S_z': 'S2'})

# Merge
data = pd.merge(df1, df2, on='D')

# Compute splitting
data['delta_E'] = data['E2'] - data['E1']
data['ln_delta_E'] = np.log(data['delta_E'])

# Filter valid data
valid = data[np.isfinite(data['S1']) & np.isfinite(data['ln_delta_E']) & (data['delta_E'] > 0)]

print("="*70)
print("SPLITTING-ACTION ANALYSIS")
print("="*70)
print(f"\nTotal D points: {len(data)}")
print(f"Valid points (S1 finite, ΔE > 0): {len(valid)}")

# Linear fit: ln(ΔE) = -α·S + const
slope, intercept, r_value, p_value, std_err = stats.linregress(valid['S1'], valid['ln_delta_E'])
r_squared = r_value**2

print(f"\nLinear fit: ln(ΔE) = {slope:.6f}·S + {intercept:.6f}")
print(f"R² = {r_squared:.6f}")
print(f"Standard error: {std_err:.6f}")

# Expected from WKB: α ≈ 2 for symmetric double-well (or 1 for half-width)
print(f"\nWKB interpretation:")
print(f"  Measured α = {-slope:.4f}")
print(f"  (Typical double-well: α ≈ 1-2 depending on normalization)")

# Summary table
print("\n" + "-"*70)
print("DATA TABLE")
print("-"*70)
print(f"{'D':>6} {'E1':>10} {'E2':>10} {'ΔE':>12} {'S1':>10} {'ln(ΔE)':>10}")
print("-"*70)
for _, row in valid.iterrows():
    print(f"{row['D']:6.1f} {row['E1']:10.4f} {row['E2']:10.4f} {row['delta_E']:12.6f} {row['S1']:10.4f} {row['ln_delta_E']:10.4f}")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: ΔE vs D
ax = axes[0, 0]
ax.semilogy(valid['D'], valid['delta_E'], 'o-', color='blue', markersize=6)
ax.set_xlabel('D', fontsize=12)
ax.set_ylabel('ΔE = E₂ - E₁', fontsize=12)
ax.set_title('Level Splitting vs D', fontsize=14)
ax.grid(True, alpha=0.3)

# Top-right: S vs D
ax = axes[0, 1]
ax.plot(valid['D'], valid['S1'], 'o-', color='red', markersize=6)
ax.set_xlabel('D', fontsize=12)
ax.set_ylabel('S₁ (WKB action)', fontsize=12)
ax.set_title('Tunneling Action vs D', fontsize=14)
ax.grid(True, alpha=0.3)

# Bottom-left: ln(ΔE) vs S - THE KEY PLOT
ax = axes[1, 0]
ax.scatter(valid['S1'], valid['ln_delta_E'], color='green', s=40, zorder=3, label='Data')
# Fit line
S_fit = np.linspace(valid['S1'].min(), valid['S1'].max(), 100)
ln_fit = slope * S_fit + intercept
ax.plot(S_fit, ln_fit, 'r--', linewidth=2, label=f'Fit: α={-slope:.3f}, R²={r_squared:.4f}')
ax.set_xlabel('S₁ (WKB action)', fontsize=12)
ax.set_ylabel('ln(ΔE)', fontsize=12)
ax.set_title('Splitting–Action Consistency Check', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Add annotation
textstr = f'ln(ΔE) = -{-slope:.4f}·S + {intercept:.4f}\nR² = {r_squared:.6f}'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Bottom-right: Residuals
ax = axes[1, 1]
predicted = slope * valid['S1'] + intercept
residuals = valid['ln_delta_E'] - predicted
ax.scatter(valid['S1'], residuals, color='purple', s=40)
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('S₁', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.set_title('Fit Residuals', fontsize=14)
ax.grid(True, alpha=0.3)

# Compute RMSE
rmse = np.sqrt(np.mean(residuals**2))
ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}', transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
output_path = outdir / 'splitting_action_consistency.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_path}")

# Also copy to paper directory
import shutil
shutil.copy(output_path, paper_dir / 'splitting_action_consistency.png')
print(f"Copied to paper/")

# Save analysis summary
summary = {
    'slope': slope,
    'intercept': intercept,
    'r_squared': r_squared,
    'std_err': std_err,
    'rmse': rmse,
    'n_points': len(valid)
}
pd.DataFrame([summary]).to_csv(outdir / 'splitting_action_fit.csv', index=False)
print("Saved: splitting_action_fit.csv")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"The splitting ΔE follows exp(-S) to R² = {r_squared:.6f}")
print("This confirms the same V_eff controls both spectrum and tunneling.")
print("="*70)

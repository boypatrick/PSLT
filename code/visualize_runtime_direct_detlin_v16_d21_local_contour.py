from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path('/Users/boypatrick/codex/PSLT_quantam')
CENTER_CSV = ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_generalization_retune_D6p4_summary_D21E21_holdout_d21_centerwidth.csv'
BETA_JSON = ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d21_d64_gnorm_beta_probe_v2_upward_summary.json'
OUT = ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_v16_d21_local_contour.png'

rows = []
with CENTER_CSV.open() as fh:
    for row in csv.DictReader(fh):
        rows.append(row)

x = np.array([float(r['g_center']) for r in rows])
y = np.array([float(r['g_half_width']) for r in rows])
z = np.array([float(r['p95_abs_delta_mu_mumu']) for r in rows])
tri = mtri.Triangulation(x, y)

beta_data = json.loads(BETA_JSON.read_text())
beta_rows = beta_data['completed_probes']
bx = np.array([float(r['gnorm_beta_d21']) for r in beta_rows])
by = np.array([float(r['p95_abs_delta_mu_mumu']) for r in beta_rows])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)

levels = np.linspace(z.min(), z.max(), 14)
cntr = ax0.tricontourf(tri, z, levels=levels, cmap='YlGnBu_r')
ax0.tricontour(tri, z, levels=levels[::2], colors='k', linewidths=0.45, alpha=0.35)
ax0.scatter(x, y, color='white', edgecolors='black', s=42, linewidths=0.5)

center_best = (0.06, 0.04)
ax0.scatter([center_best[0]], [center_best[1]], marker='*', s=220, color='gold', edgecolors='black', linewidths=0.9, zorder=5)
ax0.annotate(
    'v15 geometry anchor\ncenter=0.06, width=0.04\n(fixed g_beta=0.30 surface)',
    xy=center_best,
    xytext=(0.0608, 0.0475),
    arrowprops=dict(arrowstyle='->', lw=1.1, color='black'),
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', alpha=0.95),
)
ax0.set_title('D21 @ D=6.4 center-width contour\n(fixed g_beta = 0.30 retune surface)')
ax0.set_xlabel('g_center')
ax0.set_ylabel('g_half_width')
ax0.grid(alpha=0.2)
cb = fig.colorbar(cntr, ax=ax0, shrink=0.94, pad=0.02)
cb.set_label('slice p95 |Δμ_μμ|')

ax1.plot(bx, by, marker='o', linewidth=2.4, color='#7f3c8d')
best_idx = int(np.argmin(by))
ax1.scatter([bx[best_idx]], [by[best_idx]], color='crimson', s=80, zorder=5)
ax1.annotate(
    'v16 chosen beta\n0.31 -> 0.0485',
    xy=(bx[best_idx], by[best_idx]),
    xytext=(0.311, by[best_idx] + 0.08),
    arrowprops=dict(arrowstyle='->', lw=1.1, color='black'),
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', alpha=0.95),
)
ax1.axvline(0.30, color='0.65', linestyle='--', linewidth=1)
ax1.text(0.3002, max(by) * 0.92, 'v15 beta', fontsize=8, color='0.35')
ax1.set_title('Beta micro-refine after fixing center/width')
ax1.set_xlabel('g_norm beta (D21 @ D=6.4)')
ax1.set_ylabel('slice p95 |Δμ_μμ|')
ax1.grid(alpha=0.25)

fig.suptitle('D21 @ D=6.4 local convergence: geometry first, beta second', fontsize=15, weight='bold')
fig.savefig(OUT, dpi=180, bbox_inches='tight')
print(OUT)

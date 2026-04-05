from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

ROOT = Path('/Users/boypatrick/codex/PSLT_quantam')
CSV_PATH = ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d60_d64_sobol_doe_v3_samples.csv'
OUT = ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_v16_d60_local_contour.png'

rows = []
with CSV_PATH.open() as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        rows.append(row)

x = np.array([float(r['g_center']) for r in rows])
y = np.array([float(r['g_half_width']) for r in rows])
z = np.array([float(r['p95_abs_delta_mu_mumu']) for r in rows])
beta = np.array([float(r['g_beta']) for r in rows])

tri = mtri.Triangulation(x, y)
fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
levels = np.linspace(z.min(), z.max(), 14)
cntr = ax.tricontourf(tri, z, levels=levels, cmap='YlOrRd_r')
ax.tricontour(tri, z, levels=levels[::2], colors='k', linewidths=0.5, alpha=0.4)

sc = ax.scatter(x, y, c=beta, cmap='viridis', s=48, edgecolors='white', linewidths=0.6, zorder=3)

best_idx = int(np.argmin(z))
ax.scatter([x[best_idx]], [y[best_idx]], marker='*', s=220, color='deepskyblue', edgecolors='black', linewidths=0.8, zorder=5)
ax.annotate(
    'chosen D60 g_norm point\ncenter=0.04884, width=0.02513\nbeta=0.24682',
    xy=(x[best_idx], y[best_idx]),
    xytext=(x[best_idx] + 0.00035, y[best_idx] + 0.00032),
    arrowprops=dict(arrowstyle='->', lw=1.2, color='black'),
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', alpha=0.95),
)

ax.set_title('Local valley for D60 @ D=6.4 g_norm geometry\n(contour = slice p95, marker color = g_beta)')
ax.set_xlabel('g_center')
ax.set_ylabel('g_half_width')
ax.grid(alpha=0.2)

cb1 = fig.colorbar(cntr, ax=ax, shrink=0.92, pad=0.02)
cb1.set_label('slice p95 |Δμ_μμ|')
cb2 = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.12)
cb2.set_label('g_beta')

fig.savefig(OUT, dpi=180, bbox_inches='tight')
print(OUT)

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/Users/boypatrick/codex/PSLT_quantam')
OUT = ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_v16_convergence_dashboard.png'

VERSIONS = [
    ('v10', ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48_v10_casesplit.csv'),
    ('v11', ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48d64amp_v11.csv'),
    ('v12', ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48d64amp_v12.csv'),
    ('v13', ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48d64amp_v13.csv'),
    ('v14', ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48d64amp_v14.csv'),
    ('v15', ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48d64amp_v15.csv'),
    ('v16', ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_generalization_fast_summary_gridaware_rule_d40d64d48d64amp_v16.csv'),
]

PARAM_SNAPSHOTS = {
    'v10': {'D21 D4.8 beta': 0.65, 'D21 D4.0 beta': 1.10, 'D21 D6.4 gbeta': 0.30, 'D60 D6.4 amp': 0.00},
    'v11': {'D21 D4.8 beta': 0.65, 'D21 D4.0 beta': 1.10, 'D21 D6.4 gbeta': 0.30, 'D60 D6.4 amp': 0.02},
    'v12': {'D21 D4.8 beta': 0.65, 'D21 D4.0 beta': 1.10, 'D21 D6.4 gbeta': 0.30, 'D60 D6.4 amp': 0.03},
    'v13': {'D21 D4.8 beta': 0.65, 'D21 D4.0 beta': 1.10, 'D21 D6.4 gbeta': 0.30, 'D60 D6.4 amp': 0.05},
    'v14': {'D21 D4.8 beta': 0.70, 'D21 D4.0 beta': 1.10, 'D21 D6.4 gbeta': 0.30, 'D60 D6.4 amp': 0.05},
    'v15': {'D21 D4.8 beta': 0.70, 'D21 D4.0 beta': 1.20, 'D21 D6.4 gbeta': 0.30, 'D60 D6.4 amp': 0.05},
    'v16': {'D21 D4.8 beta': 0.70, 'D21 D4.0 beta': 1.20, 'D21 D6.4 gbeta': 0.31, 'D60 D6.4 amp': 0.05},
}

FINAL_RULE_TEXT = [
    ('D21 @ D=4.0', 'beta = 1.20'),
    ('D21 @ D=4.8', 'beta = 0.70'),
    ('D21 @ D=6.4', 'g = (0.31, 0.06, 0.04)'),
    ('D60 @ D=4.0', 'beta = 1.304'),
    ('D60 @ D=4.8', 'beta = 0.55'),
    ('D60 @ D=6.4', 'g = (0.2468, 0.0488, 0.0251) + amp 0.05'),
]


def read_summary(csv_path: Path):
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    vals = {r['case']: float(r['p95_abs_delta_mu_mumu']) for r in rows}
    return vals['D21E21_holdout'], vals['D60E21_release']


def load_probe_points(path: Path, key_name: str):
    data = json.loads(path.read_text())
    arr = data.get('all_probes') or data.get('completed_probes') or []
    pts = []
    for row in arr:
        pts.append((float(row[key_name]), float(row['p95_abs_delta_mu_mumu'])))
    return pts


def merge_unique(points):
    out = {}
    for x, y in points:
        out[round(x, 10)] = y
    xs = sorted(out)
    return np.array(xs), np.array([out[x] for x in xs])


version_names = []
d21_vals = []
d60_vals = []
for name, path in VERSIONS:
    d21, d60 = read_summary(path)
    version_names.append(name)
    d21_vals.append(d21)
    d60_vals.append(d60)

amp_points = []
for p in [
    ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d60_d64_local_amp_probe_v1_summary.json',
    ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d60_d64_local_amp_probe_v2_refine_summary.json',
    ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d60_d64_local_amp_probe_v3_refine_summary.json',
]:
    amp_points.extend(load_probe_points(p, 'alpha'))

beta48_points = []
for p in [
    ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d21_d48_beta_probe_v1_summary.json',
    ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d21_d48_beta_probe_v2_upward_summary.json',
]:
    beta48_points.extend(load_probe_points(p, 'beta_d21_d48'))

beta40_points = load_probe_points(
    ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d21_d40_beta_probe_v1_summary.json',
    'beta_d21_d40',
)

beta64_points = []
for p in [
    ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d21_d64_gnorm_beta_probe_v1_summary.json',
    ROOT / 'output/kinetic_action_chain/runtime_direct_detlin_d21_d64_gnorm_beta_probe_v2_upward_summary.json',
]:
    beta64_points.extend(load_probe_points(p, 'gnorm_beta_d21'))

amp_x, amp_y = merge_unique(amp_points)
b48_x, b48_y = merge_unique(beta48_points)
b40_x, b40_y = merge_unique(beta40_points)
b64_x, b64_y = merge_unique(beta64_points)

fig = plt.figure(figsize=(15, 11), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(version_names, d21_vals, marker='o', linewidth=2.5, color='#1f4e79', label='D21 x E21 holdout')
ax0.plot(version_names, d60_vals, marker='s', linewidth=2.5, color='#c75b12', label='D60 x E21 release')
ax0.axhline(0.5, color='0.75', linestyle='--', linewidth=1)
ax0.set_title('Strict all-direct convergence by promoted version')
ax0.set_ylabel('p95 |Δμ_μμ|')
ax0.grid(alpha=0.25)
ax0.legend(frameon=False, loc='upper right')
for idx, (x, y) in enumerate(zip(version_names, d21_vals)):
    ax0.text(idx, y + 0.08, f'{y:.3f}', ha='center', va='bottom', fontsize=8, color='#1f4e79')
for idx, (x, y) in enumerate(zip(version_names, d60_vals)):
    ax0.text(idx, y + 0.08, f'{y:.3f}', ha='center', va='bottom', fontsize=8, color='#c75b12')

ax1 = fig.add_subplot(gs[0, 1])
ax1.axis('off')
ax1.set_title('Why 2D dashboard beats a single 3D plot', pad=12)
text = (
    'Best overview:\n'
    '1. version-to-version p95 drop\n'
    '2. one local sweep per active knob\n'
    '3. final case-aware rule table\n\n'
    'Use contour only for a single local valley\n'
    '(for example D=6.4 center vs width).\n'
    'For the whole convergence path, 3D hides\n'
    'which D-specific edit actually helped.'
)
ax1.text(0.02, 0.98, text, va='top', fontsize=11)
for i, (label, val) in enumerate(FINAL_RULE_TEXT):
    y = 0.72 - i * 0.10
    ax1.text(0.02, y, label, fontsize=10, weight='bold')
    ax1.text(0.44, y, val, fontsize=10, family='monospace')

panels = [
    (fig.add_subplot(gs[1, 0]), amp_x, amp_y, 'D60 @ D=6.4 local amplitude', 'alpha', '#c75b12'),
    (fig.add_subplot(gs[1, 1]), b48_x, b48_y, 'D21 @ D=4.8 beta probe', 'beta', '#1f4e79'),
    (fig.add_subplot(gs[2, 0]), b40_x, b40_y, 'D21 @ D=4.0 beta probe', 'beta', '#2a7f62'),
    (fig.add_subplot(gs[2, 1]), b64_x, b64_y, 'D21 @ D=6.4 g_norm beta probe', 'g_beta', '#7f3c8d'),
]

for ax, xs, ys, title, xlabel, color in panels:
    ax.plot(xs, ys, marker='o', linewidth=2.2, color=color)
    best_idx = int(np.argmin(ys))
    ax.scatter([xs[best_idx]], [ys[best_idx]], color='crimson', zorder=5)
    ax.text(xs[best_idx], ys[best_idx], f'  best {xs[best_idx]:.3f}\n  {ys[best_idx]:.3f}', fontsize=8, va='bottom')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('slice p95 |Δμ_μμ|')
    ax.grid(alpha=0.25)

fig.suptitle('Strict all-direct v10→v16 convergence dashboard', fontsize=16, weight='bold')
fig.savefig(OUT, dpi=180, bbox_inches='tight')
print(OUT)

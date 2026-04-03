#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/Users/boypatrick/codex/PSLT_quantam')
sys.path.insert(0, str(ROOT / 'code'))
import scan_runtime_direct_detlin_generalization_validation as mod

OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPERDIR = ROOT / 'paper'
OUT_SUMMARY = OUTDIR / 'runtime_direct_detlin_generalization_core_summary.csv'
OUT_DETAIL = OUTDIR / 'runtime_direct_detlin_generalization_core_detail.csv'
OUT_PNG = OUTDIR / 'runtime_direct_detlin_generalization_core.png'
OUT_META = OUTDIR / 'runtime_direct_detlin_generalization_core_run_meta.json'
OUT_DECISION = OUTDIR / 'runtime_direct_detlin_generalization_core_decision.json'

SELECT = ['D21E21_holdout', 'D60E21_release']
selected_cases = [case for case in mod.ALL_CASES if case['case'] in SELECT]

obs = mod.load_observations()['mumu']
mu_obs = float(obs.mu_obs)
sigma_obs = max(float(obs.sigma_obs), 1e-12)

detail_frames = []
summary_rows = []
for case in selected_cases:
    detail = mod._evaluate_case(case, mu_obs=mu_obs, sigma_obs=sigma_obs)
    detail_frames.append(detail)
    for subset_name, subset_df in [
        ('all_points', detail),
        ('focus_band', detail[detail['subset'] == 'focus_band']),
        ('complement', detail[detail['subset'] == 'complement']),
    ]:
        summary_rows.append({'case': case['case'], 'case_label': case['label'], **mod._subset_metrics(subset_df, subset_name)})

detail_df = pd.concat(detail_frames, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)
case_order = [case['case'] for case in selected_cases]
label_map = {case['case']: case['label'] for case in selected_cases}
summary_df['case'] = pd.Categorical(summary_df['case'], categories=case_order, ordered=True)
summary_df = summary_df.sort_values(['case', 'subset']).reset_index(drop=True)
summary_df.to_csv(OUT_SUMMARY, index=False)
detail_df.to_csv(OUT_DETAIL, index=False)

all_df = summary_df[summary_df['subset'] == 'all_points'].copy()
focus_df = summary_df[summary_df['subset'] == 'focus_band'].copy()
comp_df = summary_df[summary_df['subset'] == 'complement'].copy()

x = np.arange(len(case_order))
labels = [label_map[c] for c in case_order]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
axes[0].bar(x, all_df['p95_abs_delta_mu_mumu'], color='#4e79a7')
axes[0].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
axes[0].set_title('All-points p95 |Δmu|')
axes[0].set_xticks(x, labels, rotation=20, ha='right')
axes[0].grid(alpha=0.25, axis='y')

axes[1].bar(x - 0.18, focus_df['p95_abs_delta_mu_mumu'], width=0.36, label='focus', color='#59a14f')
axes[1].bar(x + 0.18, comp_df['p95_abs_delta_mu_mumu'], width=0.36, label='complement', color='#f28e2b')
axes[1].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
axes[1].set_title('Subset p95 |Δmu|')
axes[1].set_xticks(x, labels, rotation=20, ha='right')
axes[1].grid(alpha=0.25, axis='y')
axes[1].legend(fontsize=8)

axes[2].bar(x - 0.18, all_df['acceptance_mismatch'], width=0.36, label='all', color='#4e79a7')
axes[2].bar(x + 0.18, focus_df['acceptance_mismatch'], width=0.36, label='focus', color='#59a14f')
axes[2].set_title('Acceptance mismatch')
axes[2].set_xticks(x, labels, rotation=20, ha='right')
axes[2].grid(alpha=0.25, axis='y')
axes[2].legend(fontsize=8)
fig.suptitle('Strict all-direct broader-grid core validation')
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200)
plt.close(fig)

decision = {
    'all_cases_allpoints_zero_mismatch': bool((all_df['acceptance_mismatch'] == 0.0).all()),
    'all_cases_focus_band_zero_mismatch': bool((focus_df['acceptance_mismatch'] == 0.0).all()),
    'worst_allpoints_p95_abs_delta_mu_mumu': float(all_df['p95_abs_delta_mu_mumu'].max()),
    'worst_case': str(all_df.loc[all_df['p95_abs_delta_mu_mumu'].idxmax(), 'case']),
}
OUT_META.write_text(json.dumps({'selected_cases': selected_cases}, indent=2))
OUT_DECISION.write_text(json.dumps(decision, indent=2))
for path in [OUT_SUMMARY, OUT_DETAIL, OUT_PNG, OUT_META, OUT_DECISION]:
    (PAPERDIR / path.name).write_bytes(path.read_bytes())

print(f'[saved] {OUT_SUMMARY}')
print(json.dumps(decision, indent=2))
